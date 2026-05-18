import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from datetime import datetime

API = "http://fraud_api:8000"


@st.cache_data(ttl=60)
def load_clients(limit: int = 500) -> pd.DataFrame:
    try:
        resp = requests.get(f"{API}/clients", params={"limit": limit}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        df = pd.DataFrame(data["clients"])
    except Exception as e:
        st.error(f"Impossible de joindre l'API : {e}")
        return pd.DataFrame()

    if df.empty:
        return df

    if "dob" in df.columns:
        df["dob"] = pd.to_datetime(df["dob"], errors="coerce")
        df["age"] = (datetime.now() - df["dob"]).dt.days // 365

    for col in ("lat", "long", "city_pop"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ---------------------------------------------------------------------------
st.title("Statistiques Clients")

col_refresh, col_limit = st.columns([1, 3])
with col_refresh:
    if st.button("Rafraîchir"):
        st.cache_data.clear()
with col_limit:
    limit = st.slider("Nombre de clients", 50, 500, 200, step=50)

df = load_clients(limit)

if df.empty:
    st.warning("Aucun client reçu depuis l'API.")
    st.stop()

# KPIs
st.divider()
k1, k2, k3, k4, k5 = st.columns(5)

k1.metric("Clients", len(df))

if "age" in df.columns:
    k2.metric("Âge moyen", f"{df['age'].mean():.1f} ans")

if "gender" in df.columns:
    pct_f = df["gender"].value_counts(normalize=True).get("F", 0) * 100
    k3.metric("Femmes", f"{pct_f:.1f}%")

if "state" in df.columns:
    k4.metric("États distincts", df["state"].nunique())

if "job" in df.columns:
    top_job = df["job"].value_counts().idxmax()
    k5.metric("Métier le + fréquent", top_job)

st.divider()

# Demographics
st.subheader("Démographie")
demo_col1, demo_col2 = st.columns(2)

with demo_col1:
    if "age" in df.columns:
        fig = px.histogram(
            df, x="age", nbins=25,
            title="Distribution des âges",
            labels={"age": "Âge"},
            color_discrete_sequence=["#4C78A8"],
        )
        fig.update_layout(bargap=0.05)
        st.plotly_chart(fig, use_container_width=True)

with demo_col2:
    if "gender" in df.columns:
        counts = df["gender"].value_counts().reset_index()
        counts.columns = ["genre", "count"]
        fig = px.pie(
            counts, values="count", names="genre",
            title="Répartition par genre",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        st.plotly_chart(fig, use_container_width=True)

# Age by gender box plot
if "age" in df.columns and "gender" in df.columns:
    fig = px.box(
        df, x="gender", y="age", color="gender",
        title="Distribution des âges par genre",
        labels={"gender": "Genre", "age": "Âge"},
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# Employment
if "job" in df.columns:
    st.subheader("Emploi")

    job_counts = df["job"].value_counts().head(15).reset_index()
    job_counts.columns = ["job", "count"]

    fig = px.bar(
        job_counts, x="count", y="job", orientation="h",
        title="Top 15 métiers",
        labels={"count": "Clients", "job": "Métier"},
        color="count",
        color_continuous_scale="Blues",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

# Geography
st.subheader("Géographie")

if "state" in df.columns:
    geo_col1, geo_col2 = st.columns(2)

    with geo_col1:
        state_counts = df["state"].value_counts().reset_index()
        state_counts.columns = ["state", "count"]
        fig = px.bar(
            state_counts.head(20), x="state", y="count",
            title="Top 20 états par nombre de clients",
            labels={"state": "État", "count": "Clients"},
            color="count",
            color_continuous_scale="Teal",
            text="count",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with geo_col2:
        fig = px.choropleth(
            state_counts,
            locations="state",
            locationmode="USA-states",
            color="count",
            scope="usa",
            title="Densité de clients par état",
            color_continuous_scale="Blues",
            labels={"count": "Clients"},
        )
        st.plotly_chart(fig, use_container_width=True)

if "lat" in df.columns and "long" in df.columns:
    hover = [c for c in ["city", "state", "job", "age"] if c in df.columns]
    fig = px.scatter_mapbox(
        df.dropna(subset=["lat", "long"]),
        lat="lat", lon="long",
        hover_name="first" if "first" in df.columns else None,
        hover_data=hover,
        title="Localisation des clients",
        zoom=3, height=500,
        color_discrete_sequence=["#4C78A8"],
    )
    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# City population
if "city_pop" in df.columns:
    st.subheader("Population des villes")

    pop_col1, pop_col2 = st.columns(2)

    with pop_col1:
        fig = px.histogram(
            df.dropna(subset=["city_pop"]), x="city_pop",
            nbins=30,
            title="Distribution de la population des villes",
            labels={"city_pop": "Population"},
            color_discrete_sequence=["#54A24B"],
            log_y=True,
        )
        st.plotly_chart(fig, use_container_width=True)

    with pop_col2:
        if "state" in df.columns:
            avg_pop = (
                df.groupby("state")["city_pop"]
                .mean()
                .sort_values(ascending=False)
                .head(15)
                .reset_index()
            )
            avg_pop.columns = ["state", "avg_city_pop"]
            fig = px.bar(
                avg_pop, x="state", y="avg_city_pop",
                title="Population moyenne des villes par état (top 15)",
                labels={"state": "État", "avg_city_pop": "Pop. moyenne"},
                color="avg_city_pop",
                color_continuous_scale="Greens",
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

st.subheader("Liste des clients")

search = st.text_input("Rechercher (nom, ville, métier, état…)")
display_df = df.copy()

if search:
    mask = display_df.apply(
        lambda col: col.astype(str).str.contains(search, case=False, na=False)
    ).any(axis=1)
    display_df = display_df[mask]

display_cols = [c for c in ["cc_num", "first", "last", "gender", "age", "job", "city", "state", "dob"] if c in display_df.columns]
st.dataframe(display_df[display_cols], use_container_width=True)
