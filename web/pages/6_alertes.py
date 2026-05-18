import time

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API = "http://fraud_api:8000"


def fetch_transactions(limit: int = 500) -> list[dict]:
    try:
        resp = requests.get(f"{API}/transactions", params={"limit": limit}, timeout=10)
        resp.raise_for_status()
        return resp.json().get("transactions", [])
    except Exception as e:
        st.error(f"Impossible de joindre l'API : {e}")
        return []


st.title("Alertes & Fraudes")

col_l, col_r = st.columns([2, 1])
with col_l:
    limit = st.slider("Transactions analysées", 50, 500, 200, step=50)
with col_r:
    auto = st.checkbox("Auto-refresh (10s)", value=False)

transactions = fetch_transactions(limit)

if not transactions:
    st.info("Aucune transaction disponible.")
    st.stop()

df = pd.DataFrame(transactions)

for col in ("fraud_probability", "amt"):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

if "is_fraud_predicted" in df.columns:
    df["_fraud"] = df["is_fraud_predicted"].astype(str).isin(["True", "1", "true"])
else:
    df["_fraud"] = False

df_fraud = df[df["_fraud"]].sort_values("fraud_probability", ascending=False)

# KPIs
st.divider()
k1, k2, k3, k4 = st.columns(4)
k1.metric("Fraudes détectées", len(df_fraud))
k2.metric("Sur transactions analysées", len(df))
k3.metric(
    "Taux de fraude",
    f"{len(df_fraud) / len(df) * 100:.1f}%" if len(df) else "—",
)
k4.metric(
    "Probabilité max",
    f"{df_fraud['fraud_probability'].max():.2%}" if not df_fraud.empty and "fraud_probability" in df_fraud else "—",
)
st.divider()

if df_fraud.empty:
    st.success("Aucune fraude détectée dans les transactions en mémoire.")
    st.stop()

# Distribution des probabilités
col1, col2 = st.columns(2)

with col1:
    if "fraud_probability" in df.columns:
        fig = px.histogram(
            df, x="fraud_probability", nbins=40,
            title="Distribution des probabilités de fraude",
            color="_fraud",
            color_discrete_map={True: "#e74c3c", False: "#2ecc71"},
            labels={"fraud_probability": "Probabilité", "_fraud": "Fraude"},
        )
        fig.add_vline(
            x=df_fraud["fraud_probability"].min() if not df_fraud.empty else 0.5,
            line_dash="dash", line_color="orange",
            annotation_text="Seuil",
        )
        st.plotly_chart(fig, use_container_width=True)

with col2:
    if "amt" in df_fraud.columns:
        fig = px.box(
            df, x="_fraud", y="amt",
            title="Montant : fraudes vs légitimes",
            color="_fraud",
            color_discrete_map={True: "#e74c3c", False: "#2ecc71"},
            labels={"_fraud": "Fraude", "amt": "Montant (€)"},
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# Fraudes par catégorie
if "category" in df_fraud.columns:
    cat_counts = df_fraud["category"].value_counts().reset_index()
    cat_counts.columns = ["category", "count"]
    fig = px.bar(
        cat_counts, x="category", y="count",
        title="Fraudes par catégorie",
        color="count", color_continuous_scale="Reds",
        text="count",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# Table des alertes
st.subheader(f"Transactions frauduleuses ({len(df_fraud)})")

display_cols = [c for c in [
    "trans_num", "trans_date_trans_time", "cc_num",
    "merchant", "category", "amt",
    "fraud_probability", "is_fraud", "gender",
] if c in df_fraud.columns]

df_show = df_fraud[display_cols].copy()
if "fraud_probability" in df_show.columns:
    df_show["fraud_probability"] = df_show["fraud_probability"].round(4)
if "amt" in df_show.columns:
    df_show["amt"] = df_show["amt"].round(2)

st.dataframe(
    df_show.style.background_gradient(subset=["fraud_probability"] if "fraud_probability" in df_show.columns else [], cmap="Reds"),
    use_container_width=True,
    hide_index=True,
)

st.caption(f"Dernière mise à jour : {pd.Timestamp.now().strftime('%H:%M:%S')}")

if auto:
    time.sleep(10)
    st.rerun()
