import streamlit as st

st.set_page_config(
    page_title="Détection de Fraude",
    layout="wide",
    initial_sidebar_state="expanded",
)

pages = {
    "Monitoring": [
        st.Page("pages/1_flux.py", title="Flux temps réel"),
        st.Page("pages/2_stats.py", title="Statistiques"),
        st.Page("pages/3_carte.py", title="Carte géographique"),
    ],
    "Analyse": [
        st.Page("pages/4_detail.py", title="Détail transaction"),
        st.Page("pages/5_modele.py", title="Performances modèle"),
    ],
}

pg = st.navigation(pages)
pg.run()
