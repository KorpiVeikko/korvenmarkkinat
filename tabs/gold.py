# tabs/gold.py
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from services.market_data import fetch_price_history_eur
from services.asset_ui import (
    latest_period_values,
    pct_change,
    safe_eur_card,
    safe_number_card,
    filter_by_period,
    period_selector,
    render_price_chart,
)


@st.cache_data
def load_gold():
    return fetch_price_history_eur("GC=F", period="5y")


@st.cache_data
def load_silver():
    return fetch_price_history_eur("SI=F", period="5y")


def render():
   
    st.subheader("🟨 Kulta & ⬜ Hopea")
    
    st.caption(
        "Lähteet: Yahoo Finance (kulta ja hopea hinnat), "
        "World Gold Council / Our World in Data (tuotanto ja kysyntä)."
    )

    gold_df = load_gold()
    silver_df = load_silver()

    if gold_df is None or gold_df.empty or silver_df is None or silver_df.empty:
        st.error("Dataa ei saatu ladattua.")
        return

    gold_df = gold_df.copy()
    silver_df = silver_df.copy()

    gold_df["Date"] = pd.to_datetime(gold_df["Date"], errors="coerce")
    silver_df["Date"] = pd.to_datetime(silver_df["Date"], errors="coerce")

    gold_df = gold_df.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)
    silver_df = silver_df.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)

    latest_date = max(gold_df["Date"].max(), silver_df["Date"].max())
    st.caption(f"Viimeisin markkinadata: {latest_date.date()}")

    gold_vals = latest_period_values(gold_df, "Close")
    silver_vals = latest_period_values(silver_df, "Close")

    ratio_df = pd.merge(
        gold_df[["Date", "Close"]].rename(columns={"Close": "Gold"}),
        silver_df[["Date", "Close"]].rename(columns={"Close": "Silver"}),
        on="Date",
        how="inner",
    ).dropna()

    ratio_df = ratio_df[ratio_df["Silver"] != 0].copy()
    ratio_df["Ratio"] = ratio_df["Gold"] / ratio_df["Silver"]

    ratio_vals = latest_period_values(ratio_df.rename(columns={"Ratio": "Close"}), "Close")
    ratio_mean = float(ratio_df["Ratio"].mean()) if not ratio_df.empty else None
    ratio_vs_mean = pct_change(ratio_vals["now"], ratio_mean) if ratio_mean is not None else None
    ratio_df["LongRunMean"] = ratio_mean

    t1, t2, t3 = st.tabs(["🥇 Kulta", "🥈 Hopea", "⚖️ Kulta / hopea"])

    with t1:
        st.markdown("### 🥇 Kullan hinta")
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            safe_eur_card("Nyt", gold_vals["now"], None, 0)
        with c2:
            safe_eur_card("1 kk", gold_vals["1m"], gold_vals["pct_1m"], 0)
        with c3:
            safe_eur_card("1 v", gold_vals["1y"], gold_vals["pct_1y"], 0)
        with c4:
            safe_eur_card("5 v", gold_vals["5y"], gold_vals["pct_5y"], 0)

        st.divider()
        render_price_chart(
            gold_df,
            "Kulta (€)",
            key="gold_chart",
            y_col="Close",
            y_title="€",
            options=["1 kk", "1 v", "5 v"],
            default="1 v",
        )

    with t2:
        st.markdown("### 🥈 Hopean hinta")
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            safe_eur_card("Nyt", silver_vals["now"], None, 2)
        with c2:
            safe_eur_card("1 kk", silver_vals["1m"], silver_vals["pct_1m"], 2)
        with c3:
            safe_eur_card("1 v", silver_vals["1y"], silver_vals["pct_1y"], 2)
        with c4:
            safe_eur_card("5 v", silver_vals["5y"], silver_vals["pct_5y"], 2)

        st.divider()
        render_price_chart(
            silver_df,
            "Hopea (€)",
            key="silver_chart",
            y_col="Close",
            y_title="€",
            options=["1 kk", "1 v", "5 v"],
            default="1 v",
        )

    with t3:
        st.markdown("### ⚖️ Kulta / hopea -suhde")
        c1, c2, c3, c4, c5 = st.columns(5)

        with c1:
            safe_number_card("Nyt", ratio_vals["now"], None, 2)
        with c2:
            safe_number_card("1 kk", ratio_vals["1m"], ratio_vals["pct_1m"], 2)
        with c3:
            safe_number_card("1 v", ratio_vals["1y"], ratio_vals["pct_1y"], 2)
        with c4:
            safe_number_card("5 v", ratio_vals["5y"], ratio_vals["pct_5y"], 2)
        with c5:
            safe_number_card(
                "Pitkän ajan keskiarvo",
                ratio_mean,
                ratio_vs_mean,
                2,
                caption="Nykyinen suhde vs keskiarvo",
            )

        st.divider()

        period = period_selector(
            "Kuvaajan tarkasteluväli",
            key="gold_silver_ratio_period",
            options=["1 kk", "1 v", "5 v"],
            default="1 v",
        )
        ratio_plot_df = filter_by_period(ratio_df, period, date_col="Date")

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=ratio_plot_df["Date"],
                y=ratio_plot_df["Ratio"],
                mode="lines",
                name="Kulta / hopea",
            )
        )

        if ratio_plot_df["LongRunMean"].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=ratio_plot_df["Date"],
                    y=ratio_plot_df["LongRunMean"],
                    mode="lines",
                    name="Pitkän ajan keskiarvo",
                    line=dict(dash="dash"),
                )
            )

        fig.update_layout(
            title=f"Kulta / hopea -suhde ({period})",
            xaxis_title="Päivä",
            yaxis_title="Suhdeluku",
            legend_title_text="Sarja",
        )
        st.plotly_chart(fig, use_container_width=True)






