import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

API = "http://fraud_api:8000"

JOURS = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]


@st.cache_data(ttl=30)
def fetch_transactions(limit: int = 500) -> pd.DataFrame:
    try:
        resp = requests.get(f"{API}/transactions", params={"limit": limit}, timeout=10)
        resp.raise_for_status()
        data = resp.json().get("transactions", [])
        df = pd.DataFrame(data)

        if "trans_date_trans_time" in df.columns:
            df["ts"] = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")
            df["hour"] = df["ts"].dt.hour
            df["day_of_week"] = df["ts"].dt.dayofweek
            df["day_name"] = df["day_of_week"].map(dict(enumerate(JOURS)))
            df["month"] = df["ts"].dt.month
            df["date"] = df["ts"].dt.date

        if "amt" in df.columns:
            df["amt"] = pd.to_numeric(df["amt"], errors="coerce")

        if "is_fraud_predicted" in df.columns:
            df["fraude"] = df["is_fraud_predicted"].astype(str).isin(["True", "1", "true"])
        else:
            df["fraude"] = False

        return df.dropna(subset=["ts"])
    except Exception as e:
        st.error(f"Impossible de joindre l'API : {e}")
        return pd.DataFrame()


st.title("Analyse Temporelle")

col_l, col_r = st.columns([2, 1])
with col_l:
    limit = st.slider("Transactions chargées", 50, 500, 500, step=50)
with col_r:
    if st.button("Rafraîchir"):
        st.cache_data.clear()

df = fetch_transactions(limit)

if df.empty or "ts" not in df.columns:
    st.info("Aucune donnée temporelle disponible.")
    st.stop()

st.divider()
k1, k2, k3, k4 = st.columns(4)
k1.metric("Transactions", len(df))
k2.metric("Fraudes", int(df["fraude"].sum()))
k3.metric("Taux de fraude", f"{df['fraude'].mean() * 100:.1f}%")
k4.metric(
    "Période couverte",
    f"{df['ts'].min().strftime('%d/%m')} → {df['ts'].max().strftime('%d/%m/%Y')}",
)
st.divider()

# Heatmap heure × jour de semaine
st.subheader("Heatmap : transactions par heure et jour")

heatmap_mode = st.radio("Afficher", ["Total transactions", "Fraudes uniquement", "Taux de fraude (%)"], horizontal=True)

pivot_total = df.pivot_table(index="day_name", columns="hour", values="ts", aggfunc="count").fillna(0)
pivot_fraud = df[df["fraude"]].pivot_table(index="day_name", columns="hour", values="ts", aggfunc="count").reindex_like(pivot_total).fillna(0)

ordered_days = [d for d in JOURS if d in pivot_total.index]
pivot_total = pivot_total.reindex(ordered_days)
pivot_fraud = pivot_fraud.reindex(ordered_days)

if heatmap_mode == "Total transactions":
    z = pivot_total
    colorscale = "Blues"
    title_suffix = "transactions"
elif heatmap_mode == "Fraudes uniquement":
    z = pivot_fraud
    colorscale = "Reds"
    title_suffix = "fraudes"
else:
    z = (pivot_fraud / pivot_total.replace(0, 1) * 100).round(1)
    colorscale = "OrRd"
    title_suffix = "taux fraude (%)"

fig = go.Figure(go.Heatmap(
    z=z.values,
    x=[f"{h}h" for h in z.columns],
    y=z.index.tolist(),
    colorscale=colorscale,
    hoverongaps=False,
    text=z.values.round(1),
    texttemplate="%{text}",
))
fig.update_layout(title=f"Heatmap — {title_suffix}", height=350)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# Transactions par heure
col1, col2 = st.columns(2)

with col1:
    hourly = df.groupby("hour").agg(total=("fraude", "count"), fraudes=("fraude", "sum")).reset_index()
    hourly["taux"] = hourly["fraudes"] / hourly["total"] * 100

    fig = px.bar(
        hourly, x="hour", y="total",
        title="Volume de transactions par heure",
        labels={"hour": "Heure", "total": "Transactions"},
        color="taux",
        color_continuous_scale="RdYlGn_r",
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    daily = df.groupby("day_name").agg(total=("fraude", "count"), fraudes=("fraude", "sum")).reset_index()
    daily["taux"] = daily["fraudes"] / daily["total"] * 100
    daily["day_name"] = pd.Categorical(daily["day_name"], categories=JOURS, ordered=True)
    daily = daily.sort_values("day_name")

    fig = px.bar(
        daily, x="day_name", y="total",
        title="Volume de transactions par jour",
        labels={"day_name": "Jour", "total": "Transactions"},
        color="taux",
        color_continuous_scale="RdYlGn_r",
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# Évolution dans le temps
st.subheader("Évolution dans le temps")

if "date" in df.columns:
    timeline = (
        df.groupby("date")
        .agg(total=("fraude", "count"), fraudes=("fraude", "sum"))
        .reset_index()
    )
    timeline["taux"] = timeline["fraudes"] / timeline["total"] * 100
    timeline["date"] = pd.to_datetime(timeline["date"])

    fig = go.Figure()
    fig.add_trace(go.Bar(x=timeline["date"], y=timeline["total"], name="Transactions", marker_color="#4C78A8", opacity=0.6))
    fig.add_trace(go.Bar(x=timeline["date"], y=timeline["fraudes"], name="Fraudes", marker_color="#e74c3c"))
    fig.update_layout(
        title="Transactions et fraudes par jour",
        barmode="overlay",
        xaxis_title="Date",
        yaxis_title="Nombre",
    )
    st.plotly_chart(fig, use_container_width=True)

    fig = px.line(
        timeline, x="date", y="taux",
        title="Taux de fraude (%) par jour",
        labels={"date": "Date", "taux": "Taux fraude (%)"},
        markers=True,
        color_discrete_sequence=["#e74c3c"],
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# Montant moyen par heure
if "amt" in df.columns:
    st.subheader("Montant moyen par heure")
    amt_hour = df.groupby(["hour", "fraude"])["amt"].mean().reset_index()
    amt_hour["type"] = amt_hour["fraude"].map({True: "Fraude", False: "Légitime"})

    fig = px.line(
        amt_hour, x="hour", y="amt", color="type",
        title="Montant moyen par heure selon le type",
        labels={"hour": "Heure", "amt": "Montant moyen (€)", "type": "Type"},
        color_discrete_map={"Fraude": "#e74c3c", "Légitime": "#2ecc71"},
        markers=True,
    )
    st.plotly_chart(fig, use_container_width=True)
