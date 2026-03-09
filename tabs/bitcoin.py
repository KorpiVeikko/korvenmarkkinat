# tabs/bitcoin.py
import streamlit as st
import plotly.express as px

from services.market_data import fetch_price_history
from services.assets import AssetSpec, render_asset_panel


def render():
    st.subheader("₿ Bitcoin")

    btc_df, usd_per_eur = render_asset_panel(
        fetch_price_history,
        AssetSpec(name="Bitcoin", ticker="BTC-USD", unit="USD", decimals=0, show_eur=True, icon="₿"),
    )

    if btc_df is None or btc_df.empty:
        return

    # Hintakehitys
    
    st.plotly_chart(
        px.line(btc_df, x="Date", y="Close", title="Bitcoinin hinta (USD)"),
        use_container_width=True,
    )

    if usd_per_eur:
        st.caption("EUR-muunnos: USD → EUR laskettu reaaliaikaisella EURUSD=X-kurssilla (Yahoo Finance).")


