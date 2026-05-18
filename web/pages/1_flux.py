import time

import pandas as pd
import requests
import streamlit as st

API = "http://fraud_api:8000"
REFRESH_INTERVAL = 5  # secondes


def fetch_transactions(limit: int = 20) -> list[dict]:
    try:
        resp = requests.get(f"{API}/transactions", params={"limit": limit}, timeout=15)
        resp.raise_for_status()
        return resp.json().get("transactions", [])
    except Exception as e:
        st.error(f"Impossible de joindre l'API : {e}")
        return []


st.title("Flux de transactions en temps réel")

col_left, col_right = st.columns([2, 1])
with col_left:
    limit = st.slider("Nombre de transactions à afficher", 10, 100, 20, step=10)
with col_right:
    refresh = st.slider("Rafraîchissement (secondes)", 2, 30, REFRESH_INTERVAL)

placeholder = st.empty()

while True:
    transactions = fetch_transactions(limit)

    with placeholder.container():
        if not transactions:
            st.info("Aucune transaction dans Redis. Le producer est-il démarré ?")
        else:
            df = pd.DataFrame(transactions)

            # Colonnes à afficher
            display_cols = [
                "trans_num",
                "trans_date_trans_time",
                "cc_num",
                "merchant",
                "category",
                "amt",
                "fraud_probability",
                "is_fraud_predicted",
                "is_fraud",
            ]
            display_cols = [c for c in display_cols if c in df.columns]
            df_display = df[display_cols].copy()

            if "amt" in df_display.columns:
                df_display["amt"] = pd.to_numeric(
                    df_display["amt"], errors="coerce"
                ).round(2)
            if "fraud_probability" in df_display.columns:
                df_display["fraud_probability"] = pd.to_numeric(
                    df_display["fraud_probability"], errors="coerce"
                ).round(4)

            def row_color(row):
                if str(row.get("is_fraud_predicted", "False")) in ("True", "1", "true"):
                    return ["background-color: #ffcccc"] * len(row)
                return [""] * len(row)

            n_fraud = (
                df_display["is_fraud_predicted"]
                .astype(str)
                .isin(["True", "1", "true"])
                .sum()
                if "is_fraud_predicted" in df_display.columns
                else 0
            )

            m1, m2, m3 = st.columns(3)
            m1.metric("Transactions affichées", len(df_display))
            m2.metric("Fraudes détectées", n_fraud)
            m3.metric(
                "Taux de fraude",
                f"{n_fraud / len(df_display) * 100:.1f}%" if len(df_display) else "—",
            )

            st.dataframe(
                df_display.style.apply(row_color, axis=1),
                use_container_width=True,
                hide_index=True,
            )

        st.caption(f"Dernière mise à jour : {pd.Timestamp.now().strftime('%H:%M:%S')}")

    time.sleep(refresh)
    st.rerun()
