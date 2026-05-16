import requests
import streamlit as st

API = "http://fraud_api:8000"


def fetch_transaction(trans_num: str) -> dict | None:
    try:
        resp = requests.get(f"{API}/transaction/{trans_num}", timeout=5)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"Erreur API : {e}")
        return None


st.title("Détail d'une transaction")

trans_num = st.text_input(
    "Numéro de transaction (trans_num)",
    placeholder="ex: 2ef68ffe9b4b4de1...",
)

if not trans_num:
    st.info("Entrez un numéro de transaction pour afficher les détails.")
    st.stop()

result = fetch_transaction(trans_num.strip())

if result is None:
    st.error(f"Transaction **{trans_num}** introuvable dans Redis.")
    st.stop()

data = result.get("data", {})
prediction = result.get("prediction")

# --- Résultat prédiction ---
if prediction:
    is_fraud = prediction.get("is_fraud", False)
    proba = prediction.get("fraud_probability", 0)
    threshold = prediction.get("threshold", 0.5)

    if is_fraud:
        st.error(f"FRAUDE DÉTECTÉE — Probabilité : **{proba:.2%}** (seuil : {threshold:.2%})")
    else:
        st.success(f"Transaction légitime — Probabilité de fraude : **{proba:.2%}** (seuil : {threshold:.2%})")

    st.progress(proba, text=f"Score de fraude : {proba:.4f}")
else:
    st.warning("Modèle non disponible — prédiction impossible.")

st.divider()

# --- Détails ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Transaction")
    st.write(f"**Numéro :** {trans_num}")
    st.write(f"**Date :** {data.get('trans_date_trans_time', '—')}")
    st.write(f"**Montant :** {data.get('amt', '—')} €")
    st.write(f"**Marchand :** {data.get('merchant', '—')}")
    st.write(f"**Catégorie :** {data.get('category', '—')}")
    st.write(f"**Label réel :** {'Fraude' if str(data.get('is_fraud', '0')) == '1' else 'Légitime'}")

with col2:
    st.subheader("Titulaire")
    st.write(f"**Numéro carte :** {data.get('cc_num', '—')}")
    st.write(f"**Genre :** {data.get('gender', '—')}")
    st.write(f"**Date naissance :** {data.get('dob', '—')}")
    st.write(f"**Population ville :** {data.get('city_pop', '—')}")

st.subheader("Localisation")
col3, col4 = st.columns(2)
with col3:
    st.write(f"**Client lat/lon :** {data.get('lat', '—')} / {data.get('long', '—')}")
with col4:
    st.write(f"**Marchand lat/lon :** {data.get('merch_lat', '—')} / {data.get('merch_long', '—')}")

with st.expander("Données brutes Redis"):
    st.json(data)
