import os
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRAIN_CSV = DATA_DIR / "fraudTrain.csv"
TEST_CSV = DATA_DIR / "fraudTest.csv"
TARGET = "is_fraud"

CATEGORICAL_COLS = ["category", "gender"]

FAST_MODE = False
FAST_N_LEGIT = 200_000

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT = "fraud-detection"


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
]


def load_and_prepare(csv_path: Path):
    df = pd.read_csv(
        csv_path, index_col=0, parse_dates=["trans_date_trans_time", "dob"]
    )
    df = engineer_features(df)
    return df[FEATURE_COLS].copy(), df[TARGET]


def stratified_sample(X: pd.DataFrame, y: pd.Series, n_legit: int, seed: int = 42):
    fraud_idx = y[y == 1].index
    legit_idx = (
        y[y == 0].sample(n=min(n_legit, (y == 0).sum()), random_state=seed).index
    )
    idx = fraud_idx.union(legit_idx)
    return X.loc[idx], y.loc[idx]


print("Chargement des données...")
X_train, y_train = load_and_prepare(TRAIN_CSV)
X_test, y_test = load_and_prepare(TEST_CSV)

if FAST_MODE:
    print(f"FAST_MODE — {FAST_N_LEGIT:,} légitimes + toutes les fraudes")
    X_train, y_train = stratified_sample(X_train, y_train, FAST_N_LEGIT)

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

run_name = f"xgb_v1{'_fast' if FAST_MODE else '_full'}"
print(f"\nEntraînement XGBoost (run : {run_name})...")

with mlflow.start_run(run_name=run_name):
    mlflow.log_params(
        {
            **xgb_params,
            **{
                "fast_mode": FAST_MODE,
                "train_size": len(X_train),
                "n_fraud_train": int(n_pos),
                "imbalance_ratio": round(ratio, 1),
            },
        }
    )

    clf = XGBClassifier(**xgb_params)
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

    signature = infer_signature(X_train_enc, clf.predict_proba(X_train_enc)[:, 1])
    mlflow.xgboost.log_model(
        xgb_model=clf,
        artifact_path="model",
        signature=signature,
        registered_model_name="fraud_detection_xgb",
    )

    mlflow.set_tags(
        {
            "model_type": "XGBoost",
            "dataset": "credit_card_fraud",
            "imbalanced": "scale_pos_weight",
        }
    )

    print(f"\nRun ID : {mlflow.active_run().info.run_id}")
