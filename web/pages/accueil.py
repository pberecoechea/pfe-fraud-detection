import streamlit as st
from streamlit_extras.colored_header import colored_header

# ---------- CONFIG ----------
st.set_page_config(
    page_title="Détection de Fraude Bancaire",
    page_icon="🔐",
    layout="wide"
)

# ---------- STYLE CSS CUSTOM ----------
st.markdown("""
<style>
.big-title {
    font-size: 3rem;
    font-weight: 800;
    color: #1E88E5;
}
.subtitle {
    font-size: 1.3rem;
    color: #ddd;
}
.feature-card {
    padding: 20px;
    border-radius: 15px;
    background-color: #0F1116;
    border: 1px solid #222;
}
.metric-box {
    padding: 18px;
    border-radius: 12px;
    background-color: #0E1A2B;
    text-align: center;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------- TITRE ----------
st.markdown("<p class='big-title'>🔐 Détection de Fraude Bancaire</p>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>L’intelligence artificielle au service de la sécurité financière.</p>", unsafe_allow_html=True)

st.write("---")

# ---------- METRICS ----------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
    st.metric("Fraudes détectées", "312", "+18.2%")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
    st.metric("Transactions analysées", "1 248 392", "+24.5%")
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
    st.metric("Précision du modèle", "97.6%")
    st.markdown("</div>", unsafe_allow_html=True)

st.write("---")

# ---------- DESCRIPTION ----------
st.write("""
Cette application utilise des modèles de Machine Learning avancés pour analyser 
les transactions bancaires en temps réel et détecter les comportements suspects.  
Notre objectif : **protéger les clients**, **réduire les pertes financières** et **renforcer la confiance**.
""")

st.write("")

st.button("🚀 Explorer le tableau de bord", type="primary")

st.write("---")

# ---------- FEATURES ----------
st.subheader("🔎 Fonctionnalités principales")

colA, colB, colC, colD = st.columns(4)

with colA:
    st.markdown("<div class='feature-card'>", unsafe_allow_html=True)
    st.markdown("### 🧠 Analyse Intelligente")
    st.markdown("Modèles prédictifs identifiant les transactions potentiellement frauduleuses.")
    st.markdown("</div>", unsafe_allow_html=True)

with colB:
    st.markdown("<div class='feature-card'>", unsafe_allow_html=True)
    st.markdown("### ⚡ Détection en Temps Réel")
    st.markdown("Analyse continue des transactions pour une détection immédiate.")
    st.markdown("</div>", unsafe_allow_html=True)

with colC:
    st.markdown("<div class='feature-card'>", unsafe_allow_html=True)
    st.markdown("### 🛡️ Protection Client op")
    st.markdown("Réduction des risques financiers et protection des utilisateurs.")
    st.markdown("</div>", unsafe_allow_html=True)

with colD:
    st.markdown("<div class='feature-card'>", unsafe_allow_html=True)
    st.markdown("### 📊 Insights & Reporting")
    st.markdown("Tableaux de bord interactifs et rapports détaillés.")
    st.markdown("</div>", unsafe_allow_html=True)

st.write("---")

# ---------- FOOTER ----------
st.markdown("""
<center>
<p style='color:#888'>Conçu pour les banques. Propulsé par la donnée. Sécurisé par l’IA.</p>
</center>
""", unsafe_allow_html=True)