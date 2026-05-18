import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API = "http://fraud_api:8000"


@st.cache_data(ttl=30)
def fetch_transactions(limit: int = 500) -> pd.DataFrame:
    try:
        resp = requests.get(f"{API}/transactions", params={"limit": limit}, timeout=10)
        resp.raise_for_status()
        data = resp.json().get("transactions", [])
        df = pd.DataFrame(data)
        for col in ("amt", "fraud_probability", "lat", "long", "merch_lat", "merch_long"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "is_fraud_predicted" in df.columns:
            df["fraude"] = df["is_fraud_predicted"].astype(str).isin(["True", "1", "true"])
        else:
            df["fraude"] = False
        return df
    except Exception as e:
        st.error(f"Impossible de joindre l'API : {e}")
        return pd.DataFrame()


st.title("Analyse Marchands")

col_l, col_r = st.columns([2, 1])
with col_l:
    limit = st.slider("Transactions chargées", 50, 500, 300, step=50)
with col_r:
    if st.button("Rafraîchir"):
        st.cache_data.clear()

df = fetch_transactions(limit)

if df.empty:
    st.info("Aucune donnée disponible.")
    st.stop()

# KPIs
st.divider()
k1, k2, k3, k4 = st.columns(4)
k1.metric("Marchands distincts", df["merchant"].nunique() if "merchant" in df.columns else "—")
k2.metric("Catégories", df["category"].nunique() if "category" in df.columns else "—")
k3.metric("Montant total", f"{df['amt'].sum():,.0f} €" if "amt" in df.columns else "—")
k4.metric("Taux de fraude global", f"{df['fraude'].mean() * 100:.1f}%")
st.divider()

# Stats par catégorie
st.subheader("Par catégorie")

if "category" in df.columns:
    cat = (
        df.groupby("category")
        .agg(
            transactions=("amt", "count"),
            montant_moyen=("amt", "mean"),
            montant_total=("amt", "sum"),
            fraudes=("fraude", "sum"),
        )
        .reset_index()
    )
    cat["taux_fraude"] = cat["fraudes"] / cat["transactions"] * 100

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            cat.sort_values("transactions", ascending=True),
            x="transactions", y="category", orientation="h",
            title="Transactions par catégorie",
            color="transactions", color_continuous_scale="Blues",
            text="transactions",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            cat.sort_values("taux_fraude", ascending=True),
            x="taux_fraude", y="category", orientation="h",
            title="Taux de fraude (%) par catégorie",
            color="taux_fraude", color_continuous_scale="Reds",
            text=cat.sort_values("taux_fraude")["taux_fraude"].round(1).astype(str) + "%",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(
        cat,
        x="montant_moyen", y="taux_fraude",
        size="transactions", color="category",
        title="Montant moyen vs Taux de fraude par catégorie",
        labels={"montant_moyen": "Montant moyen (€)", "taux_fraude": "Taux de fraude (%)"},
        size_max=60,
    )
    st.plotly_chart(fig, use_container_width=True)

    fig = px.box(
        df, x="category", y="amt",
        title="Distribution des montants par catégorie",
        color="category",
        labels={"amt": "Montant (€)", "category": "Catégorie"},
    )
    fig.update_layout(showlegend=False, xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# Stats par marchand (top 20)
st.subheader("Par marchand (top 20)")

if "merchant" in df.columns:
    merch = (
        df.groupby("merchant")
        .agg(
            transactions=("amt", "count"),
            montant_moyen=("amt", "mean"),
            montant_total=("amt", "sum"),
            fraudes=("fraude", "sum"),
        )
        .reset_index()
    )
    merch["taux_fraude"] = merch["fraudes"] / merch["transactions"] * 100

    col3, col4 = st.columns(2)

    with col3:
        top_vol = merch.nlargest(20, "transactions")
        fig = px.bar(
            top_vol.sort_values("transactions"),
            x="transactions", y="merchant", orientation="h",
            title="Top 20 marchands par volume",
            color="transactions", color_continuous_scale="Blues",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        top_fraud = merch[merch["transactions"] >= 2].nlargest(20, "taux_fraude")
        fig = px.bar(
            top_fraud.sort_values("taux_fraude"),
            x="taux_fraude", y="merchant", orientation="h",
            title="Top 20 marchands par taux de fraude",
            color="taux_fraude", color_continuous_scale="Reds",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Tableau détaillé marchands")
    merch_display = merch.sort_values("fraudes", ascending=False).head(50).copy()
    merch_display["montant_moyen"] = merch_display["montant_moyen"].round(2)
    merch_display["montant_total"] = merch_display["montant_total"].round(2)
    merch_display["taux_fraude"] = merch_display["taux_fraude"].round(2)
    st.dataframe(merch_display, use_container_width=True, hide_index=True)
