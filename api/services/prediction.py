"""
Service de prédiction de fraude.

Charge le modèle XGBoost et l'OrdinalEncoder depuis MLflow au démarrage,
puis expose une méthode predict() utilisée par les endpoints FastAPI.

Les features de vélocité (time_since_last_tx, tx_count_day, amt_z_score,
amt_std_card, amt_mean_merchant) sont récupérées depuis Redis si le
processor les a pré-calculées, sinon des valeurs par défaut sont utilisées.
"""

import os
import pickle
import tempfile

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "fraud_detection_xgb")

CATEGORICAL_COLS = ["category", "gender"]

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


class PredictionService:
    def __init__(self) -> None:
        self.model = None
        self.encoder = None
        self.threshold: float = 0.5
        self.ready: bool = False

    def load(self) -> None:
        """Charge le modèle et l'encoder depuis MLflow."""
        mlflow.set_tracking_uri(MLFLOW_URI)
        client = MlflowClient()

        versions = client.get_latest_versions(MODEL_NAME)
        if not versions:
            raise RuntimeError(
                f"Aucun modèle '{MODEL_NAME}' trouvé dans le registre MLflow."
            )

        latest = versions[0]
        run_id = latest.run_id

        model_uri = f"models:/{MODEL_NAME}/latest"
        self.model = mlflow.xgboost.load_model(model_uri)
        print(f"Modèle '{MODEL_NAME}' chargé (run_id={run_id})")

        tmp_dir = tempfile.mkdtemp()
        encoder_local = client.download_artifacts(run_id, "encoder/encoder.pkl", tmp_dir)
        with open(encoder_local, "rb") as f:
            self.encoder = pickle.load(f)
        print("Encoder chargé.")

        run = client.get_run(run_id)
        self.threshold = float(run.data.metrics.get("threshold", 0.5))
        print(f"Seuil de décision : {self.threshold:.4f}")

        self.ready = True

    def _compute_features(self, row: dict, redis_client=None) -> dict:
        """Calcule les features d'une transaction brute."""
        trans_time = pd.to_datetime(row["trans_date_trans_time"])
        dob = pd.to_datetime(row["dob"])
        amt = float(row["amt"])
        cc_num = str(row["cc_num"])
        merchant = str(row["merchant"])

        features: dict = {
            "amt": amt,
            "city_pop": int(row["city_pop"]),
            "lat": float(row["lat"]),
            "long": float(row["long"]),
            "merch_lat": float(row["merch_lat"]),
            "merch_long": float(row["merch_long"]),
            "hour": trans_time.hour,
            "day_of_week": trans_time.dayofweek,
            "month": trans_time.month,
            "is_night": 1 if 0 <= trans_time.hour <= 5 else 0,
            "age": (trans_time - dob).days // 365,
            "distance": float(
                np.sqrt(
                    (float(row["lat"]) - float(row["merch_lat"])) ** 2
                    + (float(row["long"]) - float(row["merch_long"])) ** 2
                )
            ),
            "category": row["category"],
            "gender": row["gender"],
        }

        # Features de vélocité : depuis Redis si disponibles
        time_since_last_tx: float = -1.0
        tx_count_day: int = 1
        amt_std_card: float = 0.0
        amt_z_score: float = 0.0
        amt_mean_merchant: float = amt

        if redis_client is not None:
            try:
                # Temps depuis la dernière transaction par carte
                last_tx_str = redis_client.get(f"last_tx_time:{cc_num}")
                if last_tx_str:
                    last_time = pd.to_datetime(last_tx_str)
                    time_since_last_tx = (trans_time - last_time).total_seconds()

                # Nombre de transactions du jour par carte
                day_str = str(trans_time.date())
                count_raw = redis_client.get(f"tx_count_day:{cc_num}:{day_str}")
                if count_raw:
                    tx_count_day = int(count_raw)

                # Stats de montant par carte
                card_stats = redis_client.hgetall(f"card_stats:{cc_num}")
                if card_stats:
                    amt_mean_card = float(card_stats.get("amt_mean", amt))
                    amt_std_card = float(card_stats.get("amt_std", 0.0))
                    amt_z_score = (amt - amt_mean_card) / (amt_std_card + 1e-9)

                # Montant moyen par marchand
                merchant_stats = redis_client.hgetall(f"merchant_stats:{merchant}")
                if merchant_stats:
                    amt_mean_merchant = float(merchant_stats.get("amt_mean", amt))
            except Exception:
                # Redis inaccessible : on conserve les defaults
                pass

        features["time_since_last_tx"] = time_since_last_tx
        features["tx_count_day"] = tx_count_day
        features["amt_std_card"] = amt_std_card
        features["amt_z_score"] = amt_z_score
        features["amt_mean_merchant"] = amt_mean_merchant

        return features


    def predict(self, transaction: dict, redis_client=None) -> dict:
        """
        Prédit si une transaction est frauduleuse.

        Returns:
            dict avec fraud_probability, is_fraud, threshold
        """
        if not self.ready:
            raise RuntimeError("Le modèle n'est pas chargé.")

        features = self._compute_features(transaction, redis_client)
        df = pd.DataFrame([features])[FEATURE_COLS]
        df[CATEGORICAL_COLS] = self.encoder.transform(df[CATEGORICAL_COLS])
        df = df.astype(float)

        proba = float(self.model.predict_proba(df)[0][1])
        is_fraud = proba >= self.threshold

        return {
            "fraud_probability": round(proba, 6),
            "is_fraud": bool(is_fraud),
            "threshold": self.threshold,
        }


prediction_service = PredictionService()
