import time
import uuid
from datetime import datetime

import plotly.graph_objects as go
import requests
import streamlit as st

API = "http://fraud_api:8000"

CATEGORIES = [
    "misc_net", "grocery_pos", "entertainment", "gas_transport",
    "misc_pos", "grocery_net", "shopping_net", "shopping_pos",
    "food_dining", "personal_care", "health_fitness",
    "travel", "kids_pets", "home",
]

US_STATES = [
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN",
    "IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV",
    "NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN",
    "TX","UT","VT","VA","WA","WV","WI","WY",
]


st.title("Simulateur de Transaction")
st.caption("Remplis les champs et soumets la transaction pour obtenir une prédiction en temps réel.")

st.divider()

with st.form("transaction_form"):
    st.subheader("Informations de la transaction")
    col1, col2, col3 = st.columns(3)

    with col1:
        amt = st.number_input("Montant (€)", min_value=0.01, value=50.0, step=0.01)
        category = st.selectbox("Catégorie", CATEGORIES)
        merchant = st.text_input("Marchand", value="fraud_Test_Merchant")

    with col2:
        trans_date = st.date_input("Date de transaction", value=datetime.now().date())
        trans_time = st.time_input("Heure de transaction", value=datetime.now().time())
        cc_num = st.text_input("Numéro de carte (cc_num)", value="1234567890123456")

    with col3:
        merch_lat = st.number_input("Latitude marchand", value=48.8566, format="%.4f")
        merch_long = st.number_input("Longitude marchand", value=2.3522, format="%.4f")

    st.subheader("Informations du titulaire")
    col4, col5, col6 = st.columns(3)

    with col4:
        first = st.text_input("Prénom", value="Jean")
        last = st.text_input("Nom", value="Dupont")
        gender = st.selectbox("Genre", ["M", "F"])

    with col5:
        dob = st.date_input("Date de naissance", value=datetime(1985, 6, 15).date())
        job = st.text_input("Métier", value="Ingénieur")
        city_pop = st.number_input("Population ville", min_value=100, value=50000, step=1000)

    with col6:
        lat = st.number_input("Latitude client", value=48.8566, format="%.4f")
        long_ = st.number_input("Longitude client", value=2.3522, format="%.4f")
        state = st.selectbox("État", US_STATES, index=US_STATES.index("NY") if "NY" in US_STATES else 0)

    submitted = st.form_submit_button("Analyser la transaction", type="primary", use_container_width=True)

# Résultat
if submitted:
    trans_datetime = datetime.combine(trans_date, trans_time)
    payload = {
        "trans_date_trans_time": trans_datetime.strftime("%Y-%m-%d %H:%M:%S"),
        "cc_num": cc_num,
        "merchant": merchant,
        "category": category,
        "amt": amt,
        "first": first,
        "last": last,
        "gender": gender,
        "street": "1 Rue de la Paix",
        "city": "Paris",
        "state": state,
        "zip": "75001",
        "lat": lat,
        "long": long_,
        "city_pop": city_pop,
        "job": job,
        "dob": dob.strftime("%Y-%m-%d"),
        "trans_num": str(uuid.uuid4()).replace("-", ""),
        "unix_time": int(trans_datetime.timestamp()),
        "merch_lat": merch_lat,
        "merch_long": merch_long,
    }

    with st.spinner("Analyse en cours..."):
        try:
            resp = requests.post(f"{API}/predict", json=payload, timeout=10)
            resp.raise_for_status()
            result = resp.json()
        except Exception as e:
            st.error(f"Erreur API : {e}")
            st.stop()

    proba = result.get("fraud_probability", 0)
    is_fraud = result.get("is_fraud", False)
    threshold = result.get("threshold", 0.5)

    st.divider()

    if is_fraud:
        st.error(f"FRAUDE DÉTECTÉE — Probabilité : **{proba:.2%}** (seuil : {threshold:.2%})")
    else:
        st.success(f"Transaction légitime — Probabilité de fraude : **{proba:.2%}** (seuil : {threshold:.2%})")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba * 100,
        number={"suffix": "%", "font": {"size": 40}},
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Score de fraude"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#e74c3c" if is_fraud else "#2ecc71"},
            "threshold": {
                "line": {"color": "orange", "width": 4},
                "thickness": 0.75,
                "value": threshold * 100,
            },
            "steps": [
                {"range": [0, threshold * 100], "color": "#d5f5e3"},
                {"range": [threshold * 100, 100], "color": "#fadbd8"},
            ],
        },
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Détails de la transaction soumise")
    col_a, col_b = st.columns(2)
    with col_a:
        st.write(f"**Montant :** {amt} €")
        st.write(f"**Catégorie :** {category}")
        st.write(f"**Marchand :** {merchant}")
        st.write(f"**Date :** {trans_datetime.strftime('%d/%m/%Y %H:%M')}")
    with col_b:
        st.write(f"**Titulaire :** {first} {last} ({gender})")
        st.write(f"**Métier :** {job}")
        st.write(f"**État :** {state}")
        st.write(f"**Seuil décision :** {threshold:.4f}")

    with st.expander("Payload brut envoyé à l'API"):
        st.json(payload)
    with st.expander("Réponse brute de l'API"):
        st.json(result)
