# tabs/oil.py
import streamlit as st
import plotly.express as px

from services.market_data import fetch_price_history


def render():
    st.subheader("🛢 Brent-raakaöljy")

    df = fetch_price_history("BZ=F", period="5y")

    # ✅ TÄRKEIN KORJAUS
    if df.empty:
        st.warning("Öljyn markkinadataa ei saatu ladattua.")
        return

    fig = px.line(
        df,
        x="Date",
        y="Close",
        title="Brent-raakaöljyn hinta (USD / barreli)",
        labels={"Close": "USD"}
    )

    st.plotly_chart(fig, use_container_width=True)

    latest = df.iloc[-1]["Close"]
    st.metric(
        "Viimeisin hinta",
        f"{latest:,.2f} USD"
    )
