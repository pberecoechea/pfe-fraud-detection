import streamlit as st

st.set_page_config(
    page_title="Détection de Fraude",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.logo("imgs/caissedepargne.jpeg", size="large")

pages = {
    "Accueil": [
        st.Page("pages/accueil.py", title="Accueil"),
    ],
    "Monitoring": [
        st.Page("pages/1_flux.py", title="Flux temps réel"),
        st.Page("pages/2_stats.py", title="Statistiques globales"),
        st.Page("pages/3_carte.py", title="Carte géographique"),
        st.Page("pages/6_alertes.py", title="Alertes & Fraudes"),
    ],
    "Analyse": [
        st.Page("pages/7_marchands.py", title="Analyse Marchands"),
        st.Page("pages/8_temporel.py", title="Analyse Temporelle"),
        st.Page("pages/4_detail.py", title="Détail transaction"),
    ],
    "Clients": [
        st.Page("pages/stats_clients.py", title="Statistiques Clients"),
    ],
    "Modèle & Système": [
        st.Page("pages/5_modele.py", title="Performances modèle"),
        st.Page("pages/9_simulateur.py", title="Simulateur"),
        st.Page("pages/10_sante.py", title="Santé Système"),
    ],
}

pg = st.navigation(pages)
pg.run()
