# tabs/gold.py
import streamlit as st
import pandas as pd
import plotly.express as px

from services.market_data import fetch_price_history
from services.assets import AssetSpec, render_asset_panel


def render():
    st.subheader("Kulta ja hopea")

    # --- Kulta ---
    gold_df, usd_per_eur = render_asset_panel(
        fetch_price_history,
        AssetSpec(name="Kullan hinta", ticker="GC=F", unit="USD", decimals=0, show_eur=True, icon="🥇"),
    )

    # --- Hopea ---
    silver_df, _ = render_asset_panel(
        fetch_price_history,
        AssetSpec(name="Hopean hinta", ticker="SI=F", unit="USD", decimals=2, show_eur=True, icon="🥈"),
    )

    # Jos dataa ei tullut, lopetetaan
    if gold_df is None or gold_df.empty or silver_df is None or silver_df.empty:
        return

    # --- Hintakäyrät (selkeästi) ---
    
    c1, c2 = st.columns(2)

    with c1:
        st.plotly_chart(
            px.line(gold_df, x="Date", y="Close", title="Kullan hinta (USD / unssi)"),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(
            px.line(silver_df, x="Date", y="Close", title="Hopean hinta (USD / unssi)"),
            use_container_width=True,
        )

    st.divider()

    # -----------------------------
    # KULTA/HOPEA -SUHDE OMA OSIO
    # -----------------------------
    st.markdown("## ⚖️ Kulta / Hopea -suhde")

    ratio_df = pd.merge(
        gold_df[["Date", "Close"]],
        silver_df[["Date", "Close"]],
        on="Date",
        suffixes=("_gold", "_silver"),
    ).dropna()

    ratio_df["Gold_Silver_Ratio"] = ratio_df["Close_gold"] / ratio_df["Close_silver"]

    ratio_now = float(ratio_df.iloc[-1]["Gold_Silver_Ratio"])
    ratio_1m = float(ratio_df.iloc[-21]["Gold_Silver_Ratio"]) if len(ratio_df) > 21 else None
    ratio_1y = float(ratio_df.iloc[-252]["Gold_Silver_Ratio"]) if len(ratio_df) > 252 else None

    def pct(now, then):
        if now is None or then is None or then == 0:
            return None
        return (now / then - 1) * 100

    cA, cB, cC = st.columns(3)
    with cA:
        st.metric("Suhde nyt", f"{ratio_now:.1f}")

    with cB:
        if ratio_1m is not None:
            st.metric("1 kk", f"{ratio_1m:.1f}")
            p = pct(ratio_now, ratio_1m)
            st.metric("", "", f"{p:+.1f} %")

    with cC:
        if ratio_1y is not None:
            st.metric("1 v", f"{ratio_1y:.1f}")
            p = pct(ratio_now, ratio_1y)
            st.metric("", "", f"{p:+.1f} %")

    st.plotly_chart(
        px.line(ratio_df, x="Date", y="Gold_Silver_Ratio", title="Kulta / Hopea -suhteen kehitys"),
        use_container_width=True,
    )

    if usd_per_eur:
        st.caption("EUR-muunnos: USD → EUR laskettu reaaliaikaisella EURUSD=X-kurssilla (Yahoo Finance).")









