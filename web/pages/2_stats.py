import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

API = "http://fraud_api:8000"


@st.cache_data(ttl=10)
def fetch_stats() -> dict:
    try:
        resp = requests.get(f"{API}/stats", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"Impossible de joindre l'API : {e}")
        return {}


@st.cache_data(ttl=10)
def fetch_transactions(limit: int = 500) -> list[dict]:
    try:
        resp = requests.get(f"{API}/transactions", params={"limit": limit}, timeout=10)
        resp.raise_for_status()
        return resp.json().get("transactions", [])
    except Exception:
        return []


st.title("📊 Statistiques globales")

if st.button("🔄 Rafraîchir"):
    st.cache_data.clear()

stats = fetch_stats()

if not stats or stats.get("total", 0) == 0:
    st.info("Aucune donnée disponible. Démarrez le producer et le Spark processor.")
    st.stop()

# --- KPIs ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total transactions", stats.get("total", 0))
c2.metric("Taux fraude (label)", f"{stats.get('fraud_rate_label', 0):.2f}%")
c3.metric("Taux fraude (modèle)", f"{stats.get('fraud_rate_predicted', 0):.2f}%")
c4.metric("Montant moyen", f"{stats.get('avg_amount', 0):.2f} €")

st.divider()

col1, col2 = st.columns(2)

# --- Top marchands ---
with col1:
    merchants = stats.get("top_merchants", [])
    if merchants:
        df_m = pd.DataFrame(merchants)
        fig = px.bar(
            df_m, x="count", y="merchant", orientation="h",
            title="Top 10 marchands", color="count",
            color_continuous_scale="Blues",
        )
        fig.update_layout(showlegend=False, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

# --- Top catégories ---
with col2:
    categories = stats.get("top_categories", [])
    if categories:
        df_c = pd.DataFrame(categories)
        fig = px.pie(
            df_c, values="count", names="category",
            title="Répartition par catégorie",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# --- Montant des transactions dans le temps ---
transactions = fetch_transactions(500)
if transactions:
    df = pd.DataFrame(transactions)
    if "trans_date_trans_time" in df.columns and "amt" in df.columns:
        df["trans_date_trans_time"] = pd.to_datetime(
            df["trans_date_trans_time"], errors="coerce"
        )
        df["amt"] = pd.to_numeric(df["amt"], errors="coerce")
        df = df.dropna(subset=["trans_date_trans_time", "amt"])
        df = df.sort_values("trans_date_trans_time")

        color_col = None
        if "is_fraud_predicted" in df.columns:
            df["Fraude"] = df["is_fraud_predicted"].astype(str).isin(
                ["True", "1", "true"]
            ).map({True: "Fraude", False: "Légitime"})
            color_col = "Fraude"

        fig = px.scatter(
            df,
            x="trans_date_trans_time",
            y="amt",
            color=color_col,
            color_discrete_map={"Fraude": "#e74c3c", "Légitime": "#2ecc71"},
            title="Montant des transactions dans le temps",
            labels={"trans_date_trans_time": "Date", "amt": "Montant (€)"},
            opacity=0.7,
        )
        st.plotly_chart(fig, use_container_width=True)
