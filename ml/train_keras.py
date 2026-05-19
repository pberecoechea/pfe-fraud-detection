"""
Comparaison Deep Learning : MLP avec Keras
Même pipeline de features que XGBoost (train.py)
Objectif : comparer AUCPR et point opérationnel COST_FN=5
"""
import os
import warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

tf.get_logger().setLevel("ERROR")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRAIN_CSV = DATA_DIR / "fraudTrain.csv"
TEST_CSV = DATA_DIR / "fraudTest.csv"
TARGET = "is_fraud"

CATEGORICAL_COLS = ["category", "gender"]

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT = "fraud-detection"

COST_FN = 5  # une fraude non détectée vaut 5× une fausse alarme


# ---------------------------------------------------------------------------
# Feature engineering — identique à train.py pour comparaison équitable
# ---------------------------------------------------------------------------
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
    df = df.sort_values("trans_date_trans_time").reset_index(drop=True)

    df["time_since_last_tx"] = (
        df.groupby("cc_num")["trans_date_trans_time"]
        .diff()
        .dt.total_seconds()
        .fillna(-1)
    )

    df["tx_date"] = df["trans_date_trans_time"].dt.date
    df["tx_count_day"] = df.groupby(["cc_num", "tx_date"]).cumcount() + 1
    df.drop(columns=["tx_date"], inplace=True)

    grp_card = df.groupby("cc_num")["amt"]
    amt_mean_card = (
        grp_card.expanding().mean().shift(1).reset_index(level=0, drop=True)
    ).fillna(df["amt"])
    amt_std_card = (
        grp_card.expanding().std().shift(1).reset_index(level=0, drop=True)
    ).fillna(0.0)
    df["amt_std_card"] = amt_std_card
    df["amt_z_score"] = (df["amt"] - amt_mean_card) / (amt_std_card + 1e-9)

    df["amt_mean_merchant"] = (
        df.groupby("merchant")["amt"]
        .expanding()
        .mean()
        .shift(1)
        .reset_index(level=0, drop=True)
    ).fillna(df["amt"])

    grp_cat = df.groupby("category")["amt"]
    amt_mean_cat = (
        grp_cat.expanding().mean().shift(1).reset_index(level=0, drop=True)
    ).fillna(df["amt"])
    amt_std_cat = (
        grp_cat.expanding().std().shift(1).reset_index(level=0, drop=True)
    ).fillna(0.0)
    df["amt_z_score_category"] = (df["amt"] - amt_mean_cat) / (amt_std_cat + 1e-9)

    df["tx_hour_key"] = df["trans_date_trans_time"].dt.floor("h")
    df["tx_count_hour"] = df.groupby(["cc_num", "tx_hour_key"]).cumcount()
    df.drop(columns=["tx_hour_key"], inplace=True)

    df["merchant_novelty"] = (
        df.groupby(["cc_num", "merchant"]).cumcount() == 0
    ).astype(int)

    return df


FEATURE_COLS = [
    "amt", "city_pop", "lat", "long", "merch_lat", "merch_long",
    "hour", "day_of_week", "month", "is_night", "age", "distance",
    "category", "gender",
    "time_since_last_tx", "tx_count_day",
    "amt_z_score", "amt_std_card", "amt_mean_merchant", "amt_z_score_category",
    "tx_count_hour", "merchant_novelty",
]


def load_and_prepare(csv_path: Path):
    df = pd.read_csv(
        csv_path, index_col=0, parse_dates=["trans_date_trans_time", "dob"]
    )
    df = engineer_features(df)
    return df[FEATURE_COLS].copy(), df[TARGET]


# ---------------------------------------------------------------------------
# Architecture MLP
# ---------------------------------------------------------------------------
def build_model(n_features: int) -> keras.Model:
    inputs = keras.Input(shape=(n_features,))
    x = layers.Dense(256, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print("Chargement des données...")
X_train, y_train = load_and_prepare(TRAIN_CSV)
X_test, y_test = load_and_prepare(TEST_CSV)

n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
ratio = n_neg / n_pos

print(f"   Train : {len(X_train):>10,} lignes  |  fraudes : {n_pos:,} ({n_pos / len(y_train) * 100:.2f}%)")
print(f"   Test  : {len(X_test):>10,} lignes")
print(f"   Ratio : {ratio:.0f}x")

# Encodage catégoriel (même encoder que train.py)
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_train_enc = X_train.copy()
X_test_enc = X_test.copy()
X_train_enc[CATEGORICAL_COLS] = encoder.fit_transform(X_train[CATEGORICAL_COLS])
X_test_enc[CATEGORICAL_COLS] = encoder.transform(X_test[CATEGORICAL_COLS])

# Normalisation — indispensable pour les réseaux de neurones
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_enc).astype(np.float32)
X_test_scaled = scaler.transform(X_test_enc).astype(np.float32)

# Poids de classe pour compenser le déséquilibre (équivalent à scale_pos_weight)
class_weight = {0: 1.0, 1: float(ratio)}

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT)

print("\nEntraînement MLP Keras...")
with mlflow.start_run(run_name="keras_mlp"):
    model = build_model(X_train_scaled.shape[1])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.AUC(curve="PR", name="auc_pr"),
            keras.metrics.AUC(curve="ROC", name="auc_roc"),
        ],
    )

    model.summary()

    early_stop = callbacks.EarlyStopping(
        monitor="val_auc_pr", patience=5, restore_best_weights=True, mode="max"
    )

    history = model.fit(
        X_train_scaled, y_train.values,
        validation_data=(X_test_scaled, y_test.values),
        epochs=50,
        batch_size=4096,
        class_weight=class_weight,
        callbacks=[early_stop],
        verbose=1,
    )

    proba_test = model.predict(X_test_scaled, batch_size=4096, verbose=0).ravel()

    aucpr = float(average_precision_score(y_test, proba_test))
    roc_auc = float(roc_auc_score(y_test, proba_test))

    precisions, recalls, thresholds = precision_recall_curve(y_test, proba_test)
    n_pos_test = int(y_test.sum())
    tp_counts = n_pos_test * recalls[:-1]
    fp_counts = tp_counts * (1.0 / (precisions[:-1] + 1e-9) - 1.0)
    fn_counts = n_pos_test * (1.0 - recalls[:-1])

    costs = COST_FN * fn_counts + fp_counts
    idx = int(np.argmin(costs))
    threshold = float(thresholds[idx])

    y_pred_default = (proba_test >= 0.5).astype(int)
    y_pred_tuned = (proba_test >= threshold).astype(int)

    f1_default = f1_score(y_test, y_pred_default)
    f1_tuned = f1_score(y_test, y_pred_tuned)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_tuned).ravel()

    mlflow.log_params({
        "architecture": "256-128-64",
        "dropout": "0.3-0.3-0.2",
        "batch_size": 4096,
        "optimizer": "Adam lr=1e-3",
        "class_weight_fraud": round(ratio, 1),
        "cost_fn": COST_FN,
        "n_features": len(FEATURE_COLS),
    })
    mlflow.log_metrics({
        "aucpr": aucpr,
        "roc_auc": roc_auc,
        "f1_tuned": f1_tuned,
        "f1_default": f1_default,
        "precision_tuned": float(precisions[idx]),
        "recall_tuned": float(recalls[idx]),
        "threshold_cost5": threshold,
        "fn_count": float(fn),
        "fp_count": float(fp),
    })
    mlflow.set_tags({
        "model_type": "Keras MLP",
        "dataset": "credit_card_fraud",
        "imbalanced": "class_weight",
    })

    print(f"\nAUCPR  : {aucpr:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    print("\n" + "=" * 60)
    print("Seuil par défaut (0.50)")
    print("=" * 60)
    print(classification_report(y_test, y_pred_default, target_names=["Légitimes", "Fraudes"], digits=4))
    print(confusion_matrix(y_test, y_pred_default))

    print("\n" + "=" * 60)
    print(f"Seuil coût optimisé COST_FN=5 ({threshold:.4f})  ← point opérationnel")
    print("=" * 60)
    print(classification_report(y_test, y_pred_tuned, target_names=["Légitimes", "Fraudes"], digits=4))
    print(confusion_matrix(y_test, y_pred_tuned))

    print(f"\nF1 fraude (COST_FN=5) : {f1_tuned:.4f}")
    print(f"\nRun ID : {mlflow.active_run().info.run_id}")
