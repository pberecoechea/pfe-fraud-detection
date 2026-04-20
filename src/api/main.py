from fastapi import FastAPI, HTTPException
import redis
import json

app = FastAPI(title="API Détection Fraude (PFE)")

# Connexion à Redis (assure-toi que le conteneur tourne !)
try:
    r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
except Exception as e:
    print(f":x: Erreur de connexion Redis : {e}")

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API de Détection de Fraude", "status": "Online"}

@app.get("/client/{client_id}")
def get_client_features(client_id: str):
    """
    Récupère les caractéristiques (Features) d'un client depuis le Feature Store (Redis)
    """
    # On cherche la clé dans Redis (ex: client:C0001)
    key = f"client:{client_id}"
    data = r.get(key)
    
    if not data:
        raise HTTPException(status_code=404, detail="Client non trouvé dans le Feature Store")
    
    # On décode le JSON stocké par Spark
    features = json.loads(data)
    
    return {
        "client_id": client_id,
        "features": features,
        "recommendation": "Calcul de probabilité en Semaine 3..."
    }

@app.get("/health")
def health_check():
    return {"redis_connected": r.ping()}