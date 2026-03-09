# tabs/realestate.py

import streamlit as st
import plotly.express as px

from services.pxweb import (
    fetch_realestate_counts,
    fetch_realestate_prices,
    clean_realestate_df
)


def render():
    st.subheader("🏠 Asuntokaupat Suomessa")
    st.caption("Lähde: Tilastokeskus / PXWeb")

    try:
        df_counts = clean_realestate_df(fetch_realestate_counts())
        df_prices = clean_realestate_df(fetch_realestate_prices())
    except Exception as e:
        st.error(f"Kiinteistödata ei latautunut: {e}")
        return

    # --------------------------------------------------
    # RAAKADATA
    # --------------------------------------------------
    with st.expander("🔍 Raakadata: kauppojen lukumäärä"):
        st.dataframe(df_counts, use_container_width=True)

    with st.expander("🔍 Raakadata: neliöhinta"):
        st.dataframe(df_prices, use_container_width=True)

    # --------------------------------------------------
    # KAUPPOJEN LUKUMÄÄRÄ
    # --------------------------------------------------
    st.subheader("📊 Asuntokauppojen lukumäärä")

    fig_counts = px.bar(
        df_counts,
        x="Kvartaali",
        y="Arvo",
        title="Asuntokauppojen lukumäärä kvartaaleittain",
        labels={"Arvo": "Kauppojen lukumäärä"}
    )

    st.plotly_chart(fig_counts, use_container_width=True)

    # --------------------------------------------------
    # NELIÖHINNAT
    # --------------------------------------------------
    st.subheader("💶 Uusien asuntojen neliöhinta")

    fig_prices = px.line(
        df_prices,
        x="Kvartaali",
        y="Arvo",
        markers=True,
        title="Uusien asuntojen keskimääräinen neliöhinta (€/m²)",
        labels={"Arvo": "€/m²"}
    )

    st.plotly_chart(fig_prices, use_container_width=True)




