from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
import redis
import json

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

