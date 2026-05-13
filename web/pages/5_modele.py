import plotly.graph_objects as go
import requests
import streamlit as st

API = "http://fraud_api:8000"


@st.cache_data(ttl=60)
def fetch_model_info() -> dict:
    try:
        resp = requests.get(f"{API}/model/info", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"Impossible de récupérer les infos modèle : {e}")
        return {}


st.title("🤖 Performances du modèle")

if st.button("🔄 Rafraîchir"):
    st.cache_data.clear()

info = fetch_model_info()

if not info:
    st.warning("Modèle non disponible dans MLflow.")
    st.stop()

# --- Infos générales ---
col1, col2, col3 = st.columns(3)
col1.metric("Modèle", info.get("model_name", "—"))
col2.metric("Version", info.get("version", "—"))
col3.metric("Run ID", info.get("run_id", "—")[:8] + "...")

st.divider()

metrics = info.get("metrics", {})

# --- Métriques clés ---
st.subheader("Métriques de classification")

key_metrics = {
    "ROC-AUC": metrics.get("roc_auc"),
    "F1 (seuil optimisé)": metrics.get("f1_tuned"),
    "F1 (seuil 0.5)": metrics.get("f1_default"),
    "Précision": metrics.get("precision_tuned"),
    "Rappel": metrics.get("recall_tuned"),
    "Seuil optimal": metrics.get("threshold"),
}

cols = st.columns(3)
for i, (label, value) in enumerate(key_metrics.items()):
    cols[i % 3].metric(label, f"{value:.4f}" if value is not None else "—")

st.divider()

# --- Jauge ROC-AUC ---
roc = metrics.get("roc_auc")
if roc is not None:
    st.subheader("ROC-AUC")
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=roc,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "ROC-AUC"},
        gauge={
            "axis": {"range": [0.5, 1.0]},
            "bar": {"color": "#2ecc71" if roc >= 0.9 else "#f39c12" if roc >= 0.8 else "#e74c3c"},
            "steps": [
                {"range": [0.5, 0.7], "color": "#fadbd8"},
                {"range": [0.7, 0.9], "color": "#fdebd0"},
                {"range": [0.9, 1.0], "color": "#d5f5e3"},
            ],
        },
    ))
    fig_gauge.update_layout(height=300)
    st.plotly_chart(fig_gauge, use_container_width=True)

st.divider()

# --- Hyperparamètres ---
params = info.get("params", {})
if params:
    st.subheader("Hyperparamètres")
    key_params = {
        k: v for k, v in params.items()
        if k in ("n_estimators", "max_depth", "learning_rate",
                  "subsample", "colsample_bytree", "scale_pos_weight")
    }
    if key_params:
        cols = st.columns(3)
        for i, (k, v) in enumerate(key_params.items()):
            cols[i % 3].metric(k, v)

with st.expander("Toutes les métriques MLflow"):
    st.json(metrics)

with st.expander("Tous les paramètres MLflow"):
    st.json(params)
