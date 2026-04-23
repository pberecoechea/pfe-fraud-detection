from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier

DATA_DIR  = Path(__file__).resolve().parent.parent / "data"
OUT_DIR   = Path(__file__).resolve().parent / "models"
OUT_DIR.mkdir(exist_ok=True)
TRAIN_CSV = DATA_DIR / "fraudTrain.csv"
TEST_CSV  = DATA_DIR / "fraudTest.csv"
TARGET = "is_fraud"
CATEGORICAL_COLS = ["category", "gender"]

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["dob"] = pd.to_datetime(df["dob"])
    df["hour"]        = df["trans_date_trans_time"].dt.hour
    df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek  # 0=lundi
    df["month"]       = df["trans_date_trans_time"].dt.month
    df["is_night"]    = df["hour"].between(0, 5).astype(int)
    df["age"] = (df["trans_date_trans_time"] - df["dob"]).dt.days // 365
    df["distance"] = np.sqrt(    # Distance euclidienne client ↔ marchand (en degrés)
        (df["lat"] - df["merch_lat"]) ** 2 +
        (df["long"] - df["merch_long"]) ** 2
    )

    return df


FEATURE_COLS = [
    "amt", "city_pop",
    "lat", "long", "merch_lat", "merch_long",
    "hour", "day_of_week", "month", "is_night",
    "age", "distance",
    "category", "gender",
]


def load_and_prepare(csv_path: Path):
    df = pd.read_csv(csv_path, index_col=0,
                     parse_dates=["trans_date_trans_time", "dob"])
    df = engineer_features(df)
    X = df[FEATURE_COLS].copy()
    y = df[TARGET]
    return X, y

print("Chargement des données...")
X_train, y_train = load_and_prepare(TRAIN_CSV)
X_test,  y_test  = load_and_prepare(TEST_CSV)

n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
ratio = n_neg / n_pos

print(f"   Train : {len(X_train):>10,} transactions  |  fraudes : {n_pos:,} ({n_pos/len(y_train)*100:.2f}%)")
print(f"   Test  : {len(X_test):>10,} transactions")
print(f"   Ratio déséquilibre : {ratio:.0f}x")

# OrdinalEncoder pour category et gender, les colonnes numériques passent telles quelles.
# XGBoost gère nativement les valeurs manquantes (nan), pas besoin d'imputer.
cat_idx = [FEATURE_COLS.index(c) for c in CATEGORICAL_COLS]

encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1,
)

xgb_params = {
    "n_estimators":     500,
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": ratio,   # paramètre utile pour les classes déséquilibrées
    "tree_method":      "hist",
    "eval_metric":      "aucpr", # méthode d'évaluation adaptée pour les classes déséquilibrées
    "random_state":     42,
    "n_jobs":           -1,
}

# On prétraite séparément pour garder les noms de colonnes
print("\nEncodage des variables catégorielles...")
X_train_enc = X_train.copy()
X_test_enc  = X_test.copy()
X_train_enc[CATEGORICAL_COLS] = encoder.fit_transform(X_train[CATEGORICAL_COLS])
X_test_enc[CATEGORICAL_COLS]  = encoder.transform(X_test[CATEGORICAL_COLS])

print("\nEntraînement XGBoost...")
clf = XGBClassifier(**xgb_params)
clf.fit(
    X_train_enc, y_train,
    eval_set=[(X_test_enc, y_test)],
    verbose=50,
)

print("\nOptimisation du seuil de décision...")
proba_test = clf.predict_proba(X_test_enc)[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_test, proba_test)
f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-9)
best_idx       = f1_scores.argmax()
best_threshold = float(thresholds[best_idx])

print(f"   Seuil optimal  : {best_threshold:.4f}")
print(f"   Précision      : {precisions[best_idx]:.4f}")
print(f"   Rappel         : {recalls[best_idx]:.4f}")
print(f"   F1             : {f1_scores[best_idx]:.4f}")

y_pred_default = (proba_test >= 0.5).astype(int)
y_pred_tuned   = (proba_test >= best_threshold).astype(int)

roc_auc = roc_auc_score(y_test, proba_test)

print("\n" + "="*60)
print("RÉSULTATS — Seuil par défaut (0.50)")
print("="*60)
print(classification_report(y_test, y_pred_default, target_names=["Légitimes", "Fraudes"], digits=4))
print("Matrice de confusion :")
print(confusion_matrix(y_test, y_pred_default))

print("\n" + "="*60)
print(f"RÉSULTATS — Seuil optimisé ({best_threshold:.4f})")
print("="*60)
print(classification_report(y_test, y_pred_tuned, target_names=["Légitimes", "Fraudes"], digits=4))
print("Matrice de confusion :")
print(confusion_matrix(y_test, y_pred_tuned))

print(f"\nROC-AUC  : {roc_auc:.4f}")
print(f"F1 fraude (seuil optimisé) : {f1_score(y_test, y_pred_tuned):.4f}")

feat_imp = (
    pd.Series(clf.feature_importances_, index=FEATURE_COLS)
    .sort_values(ascending=False)
)
print("\nImportance des features (top 10) :")
print(feat_imp.head(10).to_string())

artifact = {
    "model":       clf,
    "encoder":     encoder,
    "feature_cols": FEATURE_COLS,
    "categorical_cols": CATEGORICAL_COLS,
    "threshold":   best_threshold,
    "roc_auc":     roc_auc,
}
out_path = OUT_DIR / "xgb_fraud_v1.pkl"
joblib.dump(artifact, out_path)
print(f"\nModèle sauvegardé → {out_path}")
