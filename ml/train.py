import json
import os
import pickle
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.preprocessing import OrdinalEncoder
import optuna
from xgboost import XGBClassifier

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRAIN_CSV = DATA_DIR / "fraudTrain.csv"
TEST_CSV = DATA_DIR / "fraudTest.csv"
TARGET = "is_fraud"

CATEGORICAL_COLS = ["category", "gender"]

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT = "fraud-detection"

OPTUNA_N_TRIALS = 30
BEST_PARAMS_FILE = Path(__file__).resolve().parent / "best_params.json"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["dob"] = pd.to_datetime(df["dob"])
    df["hour"] = df["trans_date_trans_time"].dt.hour
    df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek
    df["month"] = df["trans_date_trans_time"].dt.month
    df["is_night"] = df["hour"].between(0, 5).astype(int)
    df["age"] = (df["trans_date_trans_time"] - df["dob"]).dt.days // 365
    df["distance"] = np.sqrt(
        (df["lat"] - df["merch_lat"]) ** 2 + (df["long"] - df["merch_long"]) ** 2
    )
    # Velocity features : comportement par carte
    # On trie par date pour que les calculs expanding() soient causaux
    df = df.sort_values("trans_date_trans_time").reset_index(drop=True)

    # Temps depuis la dernière transaction (diff() est déjà causal)
    df["time_since_last_tx"] = (
        df.groupby("cc_num")["trans_date_trans_time"]
        .diff()
        .dt.total_seconds()
        .fillna(-1)
    )

    # Nombre de transactions du jour : on compte les transactions PRÉCÉDENTES
    # du même jour (shift(1) exclut la courante, fillna(1) = première du jour)
    df["tx_date"] = df["trans_date_trans_time"].dt.date
    df["tx_count_day"] = (
        df.groupby(["cc_num", "tx_date"]).cumcount() + 1
    )
    df.drop(columns=["tx_date"], inplace=True)

    # Stats de montant par carte : expanding() sur l'historique passé (shift(1))
    grp_card = df.groupby("cc_num")["amt"]
    amt_mean_card = (
        grp_card.expanding().mean().shift(1).reset_index(level=0, drop=True)
    ).fillna(df["amt"])
    amt_std_card = (
        grp_card.expanding().std().shift(1).reset_index(level=0, drop=True)
    ).fillna(0.0)

    df["amt_std_card"] = amt_std_card
    df["amt_z_score"] = (df["amt"] - amt_mean_card) / (amt_std_card + 1e-9)

    # Montant moyen par marchand : idem, historique passé uniquement
    df["amt_mean_merchant"] = (
        df.groupby("merchant")["amt"]
        .expanding()
        .mean()
        .shift(1)
        .reset_index(level=0, drop=True)
    ).fillna(df["amt"])

    return df


FEATURE_COLS = [
    "amt",
    "city_pop",
    "lat",
    "long",
    "merch_lat",
    "merch_long",
    "hour",
    "day_of_week",
    "month",
    "is_night",
    "age",
    "distance",
    "category",
    "gender",
    "time_since_last_tx",
    "tx_count_day",
    "amt_z_score",
    "amt_std_card",
    "amt_mean_merchant",
]


def load_and_prepare(csv_path: Path):
    df = pd.read_csv(
        csv_path, index_col=0, parse_dates=["trans_date_trans_time", "dob"]
    )
    df = engineer_features(df)
    return df[FEATURE_COLS].copy(), df[TARGET]


print("Chargement des données...")
X_train, y_train = load_and_prepare(TRAIN_CSV)
X_test, y_test = load_and_prepare(TEST_CSV)

n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
ratio = n_neg / n_pos

print(
    f"   Train : {len(X_train):>10,} lignes  |  fraudes : {n_pos:,} ({n_pos / len(y_train) * 100:.2f}%)"
)
print(f"   Test  : {len(X_test):>10,} lignes")
print(f"   Ratio : {ratio:.0f}x")

encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

X_train_enc = X_train.copy()
X_test_enc = X_test.copy()
X_train_enc[CATEGORICAL_COLS] = encoder.fit_transform(X_train[CATEGORICAL_COLS])
X_test_enc[CATEGORICAL_COLS] = encoder.transform(X_test[CATEGORICAL_COLS])

xgb_params = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": ratio,
    "tree_method": "hist",
    "eval_metric": "aucpr",
    "random_state": 42,
    "n_jobs": -1,
}

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT)

if BEST_PARAMS_FILE.exists():
    print(
        f"\nChargement des paramètres depuis {BEST_PARAMS_FILE.name} (supprimez ce fichier pour relancer la recherche)..."
    )
    with open(BEST_PARAMS_FILE) as f:
        best_params = json.load(f)
    for k, v in best_params.items():
        print(f"   {k}: {v}")
else:
    print(f"\nRecherche d'hyperparamètres Optuna ({OPTUNA_N_TRIALS} trials)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "scale_pos_weight": ratio,
            "tree_method": "hist",
            "eval_metric": "aucpr",
            "random_state": 42,
            "n_jobs": -1,
        }
        with mlflow.start_run(run_name=f"optuna_trial_{trial.number}", nested=True):
            mlflow.log_params(params)
            clf_trial = XGBClassifier(**params, early_stopping_rounds=20)
            clf_trial.fit(
                X_train_enc,
                y_train,
                eval_set=[(X_test_enc, y_test)],
                verbose=False,
            )
            proba = clf_trial.predict_proba(X_test_enc)[:, 1]
            score = average_precision_score(y_test, proba)
            mlflow.log_metric("avg_precision", score)
        return score

    with mlflow.start_run(run_name="optuna_search"):
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=OPTUNA_N_TRIALS, show_progress_bar=True)
        best_params = study.best_params
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_avg_precision", study.best_value)
        print("\nMeilleurs paramètres trouvés :")
        for k, v in best_params.items():
            print(f"   {k}: {v}")
        print(f"Meilleur AUCPR : {study.best_value:.4f}")

    with open(BEST_PARAMS_FILE, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"Paramètres sauvegardés → {BEST_PARAMS_FILE.name}")

xgb_params.update(best_params)

run_name = "xgb_v1_full_optuna"
print(f"\nEntraînement XGBoost (run : {run_name})...")

with mlflow.start_run(run_name=run_name):
    mlflow.log_params(
        {
            **xgb_params,
            **{
                "train_size": len(X_train),
                "n_fraud_train": int(n_pos),
                "imbalance_ratio": round(ratio, 1),
            },
        }
    )

    clf = XGBClassifier(**xgb_params, early_stopping_rounds=30)
    clf.fit(X_train_enc, y_train, eval_set=[(X_test_enc, y_test)], verbose=50)

    proba_test = clf.predict_proba(X_test_enc)[:, 1]

    precisions, recalls, thresholds = precision_recall_curve(y_test, proba_test)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = f1_scores.argmax()
    best_threshold = float(thresholds[best_idx])

    y_pred_default = (proba_test >= 0.5).astype(int)
    y_pred_tuned = (proba_test >= best_threshold).astype(int)

    roc_auc = roc_auc_score(y_test, proba_test)
    f1_tuned = f1_score(y_test, y_pred_tuned)

    mlflow.log_metrics(
        {
            "roc_auc": roc_auc,
            "f1_tuned": f1_tuned,
            "precision_tuned": float(precisions[best_idx]),
            "recall_tuned": float(recalls[best_idx]),
            "threshold": best_threshold,
            "f1_default": f1_score(y_test, y_pred_default),
        }
    )

    print(
        f"\nSeuil optimal : {best_threshold:.4f}  —  P={precisions[best_idx]:.4f}  R={recalls[best_idx]:.4f}  F1={f1_tuned:.4f}"
    )

    print("\n" + "=" * 60)
    print("Seuil par défaut (0.50)")
    print("=" * 60)
    print(
        classification_report(
            y_test, y_pred_default, target_names=["Légitimes", "Fraudes"], digits=4
        )
    )
    print(confusion_matrix(y_test, y_pred_default))

    print("\n" + "=" * 60)
    print(f"Seuil optimisé ({best_threshold:.4f})")
    print("=" * 60)
    print(
        classification_report(
            y_test, y_pred_tuned, target_names=["Légitimes", "Fraudes"], digits=4
        )
    )
    print(confusion_matrix(y_test, y_pred_tuned))

    print(f"\nROC-AUC : {roc_auc:.4f}  |  F1 fraude : {f1_tuned:.4f}")

    feat_imp = pd.Series(clf.feature_importances_, index=FEATURE_COLS).sort_values(
        ascending=False
    )
    print("\nImportance des features (top 10) :")
    print(feat_imp.head(10).to_string())

    signature_input = X_train_enc.astype(float)
    signature = infer_signature(signature_input, clf.predict_proba(X_train_enc)[:, 1])
    mlflow.xgboost.log_model(
        xgb_model=clf,
        name="model",
        signature=signature,
        registered_model_name="fraud_detection_xgb",
    )

    encoder_path = Path(__file__).resolve().parent / "encoder.pkl"
    with open(encoder_path, "wb") as f:
        pickle.dump(encoder, f)
    mlflow.log_artifact(str(encoder_path), artifact_path="encoder")
    print(f"Encoder sauvegardé → {encoder_path.name}")

    mlflow.set_tags(
        {
            "model_type": "XGBoost",
            "dataset": "credit_card_fraud",
            "imbalanced": "scale_pos_weight",
        }
    )

    print(f"\nRun ID : {mlflow.active_run().info.run_id}")
