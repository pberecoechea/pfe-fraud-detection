import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API = "http://fraud_api:8000"


@st.cache_data(ttl=30)
def fetch_transactions(limit: int = 500) -> list[dict]:
    try:
        resp = requests.get(f"{API}/transactions", params={"limit": limit}, timeout=10)
        resp.raise_for_status()
        return resp.json().get("transactions", [])
    except Exception as e:
        st.error(f"Impossible de joindre l'API : {e}")
        return []


st.title("Carte géographique")

if st.button("Rafraîchir"):
    st.cache_data.clear()

transactions = fetch_transactions(500)

if not transactions:
    st.info("Aucune donnée disponible.")
    st.stop()

df = pd.DataFrame(transactions)

for col in ["lat", "long", "merch_lat", "merch_long", "amt"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["lat", "long"])

if "is_fraud_predicted" in df.columns:
    df["Fraude"] = df["is_fraud_predicted"].astype(str).isin(
        ["True", "1", "true"]
    ).map({True: "Fraude", False: "Légitime"})
else:
    df["Fraude"] = "Inconnu"

col1, col2 = st.columns(2)
with col1:
    show_clients = st.checkbox("Positions clients", value=True)
with col2:
    show_merchants = st.checkbox("Positions marchands", value=True)

filter_fraud = st.radio(
    "Filtrer",
    ["Toutes", "Fraudes uniquement", "Légitimes uniquement"],
    horizontal=True,
)
if filter_fraud == "Fraudes uniquement":
    df = df[df["Fraude"] == "Fraude"]
elif filter_fraud == "Légitimes uniquement":
    df = df[df["Fraude"] == "Légitime"]

if df.empty:
    st.warning("Aucune transaction à afficher avec ce filtre.")
    st.stop()

st.subheader(f"{len(df)} transactions affichées")

# Carte clients
if show_clients:
    fig_clients = px.scatter_mapbox(
        df,
        lat="lat",
        lon="long",
        color="Fraude",
        color_discrete_map={"Fraude": "#e74c3c", "Légitime": "#2ecc71", "Inconnu": "#95a5a6"},
        hover_data=["trans_num", "amt", "merchant", "category"],
        size_max=12,
        zoom=4,
        title="Localisation des clients",
        mapbox_style="carto-positron",
    )
    st.plotly_chart(fig_clients, use_container_width=True)

# Carte marchands
if show_merchants and "merch_lat" in df.columns:
    df_m = df.dropna(subset=["merch_lat", "merch_long"])
    if not df_m.empty:
        fig_merchants = px.scatter_mapbox(
            df_m,
            lat="merch_lat",
            lon="merch_long",
            color="Fraude",
            color_discrete_map={"Fraude": "#e74c3c", "Légitime": "#2ecc71", "Inconnu": "#95a5a6"},
            hover_data=["trans_num", "merchant", "amt"],
            size_max=12,
            zoom=4,
            title="Localisation des marchands",
            mapbox_style="carto-positron",
        )
        st.plotly_chart(fig_merchants, use_container_width=True)
