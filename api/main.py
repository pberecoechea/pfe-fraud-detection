from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
import redis
import mlflow
from mlflow.tracking import MlflowClient
import os

from api.models.transaction import TransactionInput, PredictionResponse
from api.services.prediction import prediction_service


# ---------------------------------------------------------------------------
# Redis
# ---------------------------------------------------------------------------

def _make_redis() -> redis.Redis | None:
    try:
        return redis.Redis(host="redis_cache", port=6379, db=0, decode_responses=True)
    except Exception as e:
        print(f"Erreur de connexion Redis : {e}")
        return None


r = _make_redis()


# ---------------------------------------------------------------------------
# Lifespan : chargement du modèle au démarrage
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        prediction_service.load()
    except Exception as e:
        print(f"[WARN] Modèle non chargé au démarrage : {e}")
    yield


app = FastAPI(title="API Détection Fraude (PFE)", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def read_root():
    return {
        "message": "Bienvenue sur l'API de Détection de Fraude",
        "status": "Online",
        "model_ready": prediction_service.ready,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionInput):
    """
    Prédit si une transaction est frauduleuse.

    Accepte les champs bruts du CSV (mêmes colonnes que fraudTrain/fraudTest).
    Les features de vélocité sont enrichies depuis Redis si disponibles.
    """
    if not prediction_service.ready:
        raise HTTPException(
            status_code=503,
            detail="Le modèle n'est pas encore disponible. Réessayez dans quelques instants.",
        )
    try:
        result = prediction_service.predict(transaction.model_dump(), r)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return PredictionResponse(trans_num=transaction.trans_num, **result)


@app.get("/transaction/{trans_num}")
def get_transaction(trans_num: str):
    """
    Récupère une transaction stockée dans Redis par le Spark Processor.
    Renvoie également la prédiction de fraude si le modèle est chargé.
    """
    if r is None:
        raise HTTPException(status_code=503, detail="Redis non disponible")

    tx_key = f"transaction:{trans_num}"
    data = r.hgetall(tx_key)
    if not data:
        raise HTTPException(
            status_code=404,
            detail=f"Transaction '{trans_num}' non trouvée dans Redis.",
        )

    response: dict = {"trans_num": trans_num, "data": data}

    # Ajout de la prédiction si le modèle est chargé et les champs suffisants
    if prediction_service.ready:
        try:
            result = prediction_service.predict(data, r)
            response["prediction"] = result
        except Exception as e:
            response["prediction_error"] = str(e)

    return response


@app.get("/health")
def health_check():
    redis_ok = False
    try:
        if r:
            redis_ok = r.ping()
    except Exception:
        pass
    return {
        "redis_connected": redis_ok,
        "model_ready": prediction_service.ready,
    }


@app.get("/transactions")
def list_transactions(limit: int = Query(default=50, le=500)):
    """
    Retourne les N dernières transactions traitées par le Spark Processor,
    avec leur prédiction si le modèle est disponible.
    """
    if r is None:
        raise HTTPException(status_code=503, detail="Redis non disponible")

    trans_nums = r.zrevrange("transactions:recent", 0, limit - 1)
    if not trans_nums:
        return {"transactions": [], "total": 0}

    results = []
    for trans_num in trans_nums:
        data = r.hgetall(f"transaction:{trans_num}")
        if not data:
            continue
        entry = {"trans_num": trans_num, **data}
        if prediction_service.ready:
            try:
                pred = prediction_service.predict(data, r)
                entry["fraud_probability"] = pred["fraud_probability"]
                entry["is_fraud_predicted"] = pred["is_fraud"]
            except Exception:
                entry["fraud_probability"] = None
                entry["is_fraud_predicted"] = None
        results.append(entry)

    return {"transactions": results, "total": len(results)}


@app.get("/stats")
def get_stats():
    """
    Statistiques agrégées sur les transactions récentes stockées dans Redis.
    """
    if r is None:
        raise HTTPException(status_code=503, detail="Redis non disponible")

    trans_nums = r.zrevrange("transactions:recent", 0, -1)
    if not trans_nums:
        return {"total": 0}

    total = len(trans_nums)
    total_fraud_label = 0   # is_fraud du dataset (ground truth)
    total_fraud_pred = 0    # prédiction du modèle
    total_amt = 0.0
    merchants: dict = {}
    categories: dict = {}

    for trans_num in trans_nums:
        data = r.hgetall(f"transaction:{trans_num}")
        if not data:
            continue
        try:
            amt = float(data.get("amt", 0))
            total_amt += amt
        except (ValueError, TypeError):
            pass

        if str(data.get("is_fraud", "0")) == "1":
            total_fraud_label += 1

        merchant = data.get("merchant", "inconnu")
        merchants[merchant] = merchants.get(merchant, 0) + 1

        category = data.get("category", "inconnu")
        categories[category] = categories.get(category, 0) + 1

        if prediction_service.ready:
            try:
                pred = prediction_service.predict(data, r)
                if pred["is_fraud"]:
                    total_fraud_pred += 1
            except Exception:
                pass

    top_merchants = sorted(merchants.items(), key=lambda x: x[1], reverse=True)[:10]
    top_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)

    return {
        "total": total,
        "fraud_rate_label": round(total_fraud_label / total * 100, 2) if total else 0,
        "fraud_rate_predicted": round(total_fraud_pred / total * 100, 2) if total else 0,
        "avg_amount": round(total_amt / total, 2) if total else 0,
        "total_amount": round(total_amt, 2),
        "top_merchants": [{"merchant": m, "count": c} for m, c in top_merchants],
        "top_categories": [{"category": c, "count": n} for c, n in top_categories],
    }


@app.get("/model/info")
def model_info():
    """
    Retourne les métriques du modèle enregistré dans MLflow.
    """
    try:
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        mlflow.set_tracking_uri(mlflow_uri)
        client = MlflowClient()
        versions = client.get_latest_versions(
            os.getenv("MLFLOW_MODEL_NAME", "fraud_detection_xgb")
        )
        if not versions:
            raise HTTPException(status_code=404, detail="Modèle non trouvé dans MLflow")

        latest = versions[0]
        run = client.get_run(latest.run_id)
        return {
            "model_name": latest.name,
            "version": latest.version,
            "run_id": latest.run_id,
            "metrics": run.data.metrics,
            "params": run.data.params,
            "tags": run.data.tags,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

