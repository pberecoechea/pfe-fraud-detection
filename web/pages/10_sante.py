import time

import plotly.graph_objects as go
import requests
import streamlit as st

API = "http://fraud_api:8000"


def check_health() -> dict:
    try:
        resp = requests.get(f"{API}/health", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


def check_api() -> tuple[bool, float]:
    try:
        start = time.time()
        resp = requests.get(f"{API}/", timeout=5)
        latency = (time.time() - start) * 1000
        return resp.status_code == 200, latency
    except Exception:
        return False, -1


def fetch_model_info() -> dict:
    try:
        resp = requests.get(f"{API}/model/info", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {}


def fetch_recent_transactions(n: int = 5) -> list:
    try:
        resp = requests.get(f"{API}/transactions", params={"limit": n}, timeout=5)
        resp.raise_for_status()
        return resp.json().get("transactions", [])
    except Exception:
        return []


def fetch_features() -> list:
    try:
        resp = requests.get(f"{API}/model/features", timeout=5)
        resp.raise_for_status()
        return resp.json().get("features", [])
    except Exception:
        return []


st.title("Santé Système")

if st.button("Rafraîchir"):
    st.rerun()

st.divider()

# Statut des services
st.subheader("État des services")

health = check_health()
api_ok, latency = check_api()
model_info = fetch_model_info()

c1, c2, c3, c4 = st.columns(4)

with c1:
    if api_ok:
        st.success("API")
        st.metric("Latence", f"{latency:.0f} ms")
    else:
        st.error("API")
        st.metric("Latence", "—")

with c2:
    redis_ok = health.get("redis_connected", False)
    if redis_ok:
        st.success("Redis")
        st.caption("Connecté")
    else:
        st.error("Redis")
        st.caption("Déconnecté")

with c3:
    model_ok = health.get("model_ready", False)
    if model_ok:
        st.success("Modèle ML")
        st.caption("Chargé")
    else:
        st.warning("Modèle ML")
        st.caption("Non chargé")

with c4:
    recent_tx = fetch_recent_transactions(1)
    if recent_tx:
        last_ts = recent_tx[0].get("trans_date_trans_time", "—")
        st.success("Producer")
        st.caption(f"Dernier: {last_ts}")
    else:
        st.warning("Producer")
        st.caption("Aucune transaction")

if "error" in health:
    st.error(f"Erreur health check : {health['error']}")

st.divider()

# Jauge latence API
if latency > 0:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=latency,
        number={"suffix": " ms"},
        title={"text": "Latence API"},
        gauge={
            "axis": {"range": [0, 500]},
            "bar": {"color": "#2ecc71" if latency < 100 else "#f39c12" if latency < 300 else "#e74c3c"},
            "steps": [
                {"range": [0, 100], "color": "#d5f5e3"},
                {"range": [100, 300], "color": "#fef9e7"},
                {"range": [300, 500], "color": "#fadbd8"},
            ],
        },
    ))
    fig.update_layout(height=250)
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# Infos modèle
st.subheader("Modèle MLflow")

if model_info:
    m1, m2, m3 = st.columns(3)
    m1.metric("Nom", model_info.get("model_name", "—"))
    m2.metric("Version", model_info.get("version", "—"))
    m3.metric("Run ID", str(model_info.get("run_id", "—"))[:8] + "...")

    metrics = model_info.get("metrics", {})
    if metrics:
        mm1, mm2, mm3, mm4 = st.columns(4)
        mm1.metric("ROC-AUC", f"{float(metrics['roc_auc']):.4f}" if "roc_auc" in metrics else "—")
        mm2.metric("F1 (tuned)", f"{float(metrics['f1_tuned']):.4f}" if "f1_tuned" in metrics else "—")
        mm3.metric("Précision", f"{float(metrics['precision_tuned']):.4f}" if "precision_tuned" in metrics else "—")
        mm4.metric("Seuil", f"{float(metrics['threshold']):.4f}" if "threshold" in metrics else "—")
else:
    st.warning("Informations MLflow non disponibles.")

st.divider()

# Feature importances
st.subheader("Importance des features")

features = fetch_features()
if features:
    import pandas as pd
    import plotly.express as px

    df_feat = pd.DataFrame(features)
    fig = px.bar(
        df_feat.sort_values("importance"),
        x="importance", y="feature", orientation="h",
        title="Feature importances (modèle chargé)",
        color="importance", color_continuous_scale="Blues",
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Endpoint /model/features non disponible — modèle non chargé ou endpoint absent.")

st.divider()

# Dernières transactions reçues
st.subheader("Dernières transactions reçues")

recent = fetch_recent_transactions(10)
if recent:
    import pandas as pd
    df_recent = pd.DataFrame(recent)
    cols = [c for c in ["trans_num", "trans_date_trans_time", "amt", "merchant", "category", "is_fraud_predicted", "fraud_probability"] if c in df_recent.columns]
    st.dataframe(df_recent[cols], use_container_width=True, hide_index=True)
else:
    st.info("Aucune transaction en mémoire.")

st.caption(f"Vérifié à {__import__('pandas').Timestamp.now().strftime('%H:%M:%S')}")
