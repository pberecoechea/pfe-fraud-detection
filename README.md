# PFE : Système de détection de fraude bancaire en temps réel 💳

*Par Pablo BERECOECHEA et Rubens TUEUX*

![Fraud detection](https://media.istockphoto.com/id/2214860342/fr/vectoriel/pirate-informatique-volant-carte-de-paiement-illustration-vectorielle.jpg?s=2048x2048&w=is&k=20&c=jyrA1mrry68HSD8A72W6My0Wp9vYW4kRn535kReUgKU=)

> [!NOTE]
> Sujet de notre projet de fin d'études pour notre dernière année du cycle ingénieur à Cy Tech PAU.

## SOMMAIRE

1. [**Objectif du projet**](#objectif-du-projet)
2. [**Architecture**](#architecture)
3. [**Organisation du dossier**](#organisation-du-dossier)
4. [**Prérequis**](#prérequis)
5. [**Premier lancement (depuis zéro)**](#premier-lancement-depuis-zéro)
    1. [Cloner le dépôt](#1-cloner-le-dépôt)
    2. [Démarrer le stockage et MLflow](#2-démarrer-le-stockage-et-mlflow)
    3. [Configurer Garage (S3)](#3-configurer-garage-s3)
    4. [Entraîner le modèle](#4-entraîner-le-modèle)
    5. [Lancer tous les services](#5-lancer-tous-les-services)
6. [**Lancements suivants**](#lancements-suivants)
7. [**Interfaces disponibles**](#interfaces-disponibles)
8. [**Jeu de données**](#jeu-de-données)


## Objectif du projet

Simuler un système de transactions bancaires en temps réel et détecter automatiquement les fraudes grâce à un modèle de machine learning (XGBoost). Les transactions sont diffusées via Kafka, traitées par Spark, classifiées par une API FastAPI, et visualisées dans un dashboard Streamlit.

## Architecture

```
CSV data/ ──► Producer ──► Kafka (Redpanda) ──► Spark Processor ──► Redis
                                                                       │
                                                                  fraud_api ◄── MLflow (modèle XGBoost)
                                                                       │
                                                               Dashboard Streamlit
```

| Composant | Rôle |
|---|---|
| **Producer** | Lit le CSV et envoie les transactions vers Kafka (~10 tx/s) |
| **Redpanda** | Broker Kafka qui transporte les transactions |
| **Spark Processor** | Consomme Kafka, calcule les features de vélocité, stocke dans Redis |
| **Redis** | Feature Store (transactions, stats par carte/marchand) |
| **MLflow + Garage** | Registre de modèle + stockage S3 des artefacts |
| **fraud_api** | API FastAPI : charge le modèle MLflow, expose `/predict`, `/transactions`, `/stats` |
| **Dashboard** | Interface Streamlit multipage (flux temps réel, stats, carte, détail, modèle) |

## Organisation du dossier

```
pfe-fraud-detection/
├── api/                    # API FastAPI
│   ├── models/             # Schémas Pydantic
│   ├── services/           # Chargement modèle + prédiction
│   └── main.py
├── data/                   # fraudTrain.csv / fraudTest.csv (ignorés par git)
├── ml/                     # Entraînement du modèle
│   ├── train.py            # XGBoost + Optuna + MLflow
│   └── import_dataset.py   # Téléchargement Kaggle
├── processing/             # Spark Processor (Kafka → Redis)
├── producer/               # Envoi des transactions vers Kafka
├── web/                    # Dashboard Streamlit
│   └── pages/              # 5 pages (flux, stats, carte, détail, modèle)
├── docker-compose.yml
└── garage.toml
```

## Prérequis

- Python 3.11+
- Docker + Docker Compose
- Git

## Premier lancement (depuis zéro)

> [!IMPORTANT]
> Les étapes 1 à 4 ne sont à effectuer **qu'une seule fois**. Le modèle et les artefacts sont ensuite persistés dans les volumes Docker. Les lancements suivants se résument à `docker compose up -d`.

### 1. Cloner le dépôt

```bash
git clone https://github.com/pberecoechea/pfe-fraud-detection.git
cd pfe-fraud-detection
```

### 2. Démarrer le stockage et MLflow

Démarrer uniquement les services nécessaires à l'entraînement (Garage, Postgres, MLflow) :

```bash
docker compose up -d garage postgres mlflow redis_cache redpanda
```

Attendre ~30 secondes que MLflow soit prêt, puis vérifier sur http://localhost:5000.

### 3. Configurer Garage (S3)

> [!NOTE]
> Cette étape configure le stockage objet auto-hébergé utilisé par MLflow pour persister les artefacts du modèle. À faire **une seule fois**.

**a. Récupérer l'ID du nœud Garage :**

```bash
docker exec -it garage /garage status
```

Copier la valeur `ID` affichée (ex: `33ac546766e28a6d`).

**b. Assigner le layout :**

```bash
docker exec -it garage /garage layout assign -z dc1 -c 1G <node_id>
docker exec -it garage /garage layout apply --version 1
```

**c. Créer le bucket :**

```bash
docker exec -it garage /garage bucket create mlflow-bucket
```

**d. Créer une clé d'accès et l'autoriser sur le bucket :**

```bash
docker exec -it garage /garage key create mlflow-key
```

La commande affiche un **Key ID** et une **Secret key** — les noter.

```bash
docker exec -it garage /garage bucket allow --read --write --owner mlflow-bucket --key mlflow-key
```

**e. Renseigner les clés dans `docker-compose.yml` :**

Dans le service `mlflow`, remplacer les valeurs des variables d'environnement :

```yaml
environment:
  - AWS_ACCESS_KEY_ID=<Key ID obtenu à l'étape d>
  - AWS_SECRET_ACCESS_KEY=<Secret key obtenu à l'étape d>
```

Puis redémarrer MLflow pour qu'il prenne en compte les nouvelles clés :

```bash
docker compose up -d mlflow
```

### 4. Entraîner le modèle

Installer les dépendances ML localement puis lancer l'entraînement (~15-30 min) :

```bash
pip install -r ml/requirements.txt
```

```bash
MLFLOW_TRACKING_URI=http://localhost:5000 \
MLFLOW_S3_ENDPOINT_URL=http://localhost:3900 \
AWS_ACCESS_KEY_ID=<Key ID> \
AWS_SECRET_ACCESS_KEY=<Secret key> \
AWS_DEFAULT_REGION=garage \
python3 ml/train.py
```

Suivre la progression sur http://localhost:5000. L'entraînement est terminé quand `Run ID : ...` s'affiche dans le terminal. Le modèle `fraud_detection_xgb` est alors enregistré dans le registre MLflow.

### 5. Lancer tous les services

```bash
docker compose up -d
```

Vérifier que l'API a bien chargé le modèle :

```bash
python3 -c "import urllib.request; print(urllib.request.urlopen('http://localhost:8000/health').read().decode())"
```

Le champ `"model_ready"` doit valoir `true`. Si ce n'est pas le cas (l'API a démarré avant MLflow), forcer un redémarrage :

```bash
docker compose restart fraud_api
```

Ouvrir le dashboard : **http://localhost:8501**

Les transactions s'affichent automatiquement dans la page **Flux temps réel** au fur et à mesure que le producer les envoie.

---

## Lancements suivants

Le modèle et les données sont persistés dans les volumes Docker. Un simple `docker compose up -d` suffit pour relancer l'intégralité du projet.

```bash
docker compose up -d
```

---

## Interfaces disponibles

| Interface | URL | Description |
|---|---|---|
| **Dashboard** | http://localhost:8501 | Streamlit — visualisation temps réel |
| **API** | http://localhost:8000 | FastAPI — `/docs` pour la doc interactive |
| **MLflow** | http://localhost:5000 | Suivi des expériences et registre de modèles |
| **Redpanda Console** | http://localhost:8080 | Monitoring Kafka |
| **RedisInsight** | http://localhost:8002 | Inspection du Feature Store Redis |

---

## Jeu de données

Le jeu de données utilisé provient de [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection). Il contient des transactions bancaires simulées avec un label `is_fraud` (0 ou 1), présentant un fort déséquilibre de classes (~0.5% de fraudes).
