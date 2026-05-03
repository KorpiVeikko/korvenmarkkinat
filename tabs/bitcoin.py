# tabs/bitcoin.py
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from services.market_data import fetch_price_history, fetch_price_history_eur
from services.asset_ui import (
    latest_period_values,
    pct_change,
    safe_eur_card,
    filter_by_period,
    period_selector,
    render_price_chart_with_extra_lines,
)


HALVING_DATES = [
    ("2012-11-28", "Halving 2012"),
    ("2016-07-09", "Halving 2016"),
    ("2020-05-11", "Halving 2020"),
    ("2024-04-20", "Halving 2024"),
]


def _fmt_num(x: float | None, decimals: int = 0, suffix: str = "") -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x:,.{decimals}f}".replace(",", " ") + suffix


def _fmt_money(x: float | None, currency: str = "€", decimals: int = 0) -> str:
    if x is None or pd.isna(x):
        return "—"

    x = float(x)
    ax = abs(x)

    if ax >= 1_000_000_000_000:
        return f"{x / 1_000_000_000_000:,.2f}".replace(",", " ") + f" bilj. {currency}"
    if ax >= 1_000_000_000:
        return f"{x / 1_000_000_000:,.2f}".replace(",", " ") + f" mrd {currency}"
    if ax >= 1_000_000:
        return f"{x / 1_000_000:,.2f}".replace(",", " ") + f" milj. {currency}"

    return f"{x:,.{decimals}f}".replace(",", " ") + f" {currency}"


def _latest_valid(series: pd.Series) -> float | None:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.iloc[-1])


def _add_halving_lines(fig, plot_df: pd.DataFrame, _period: str):
    if plot_df is None or plot_df.empty:
        return fig

    min_date = pd.to_datetime(plot_df["Date"], errors="coerce").min()
    max_date = pd.to_datetime(plot_df["Date"], errors="coerce").max()

    for date_str, label in HALVING_DATES:
        d = pd.to_datetime(date_str)
        if min_date <= d <= max_date:
            fig.add_vline(x=d, line_dash="dash", opacity=0.6)
            fig.add_annotation(
                x=d,
                y=1,
                yref="paper",
                text=label,
                showarrow=False,
                yanchor="bottom",
                textangle=-90,
            )

    return fig


@st.cache_data
def load_btc_eur():
    return fetch_price_history_eur("BTC-USD", period="10y")


@st.cache_data
def load_btc_usd():
    return fetch_price_history("BTC-USD", period="10y")


def render():
    st.subheader("₿ Bitcoin")

    st.caption(
        "Lähteet: Yahoo Finance / CoinGecko (hinta), "
        "Blockchain.com / Glassnode (on-chain data ja verkon tunnusluvut)."
    )


    btc_eur_df = load_btc_eur()
    btc_usd_df = load_btc_usd()

    if (
        btc_eur_df is None or btc_eur_df.empty or
        btc_usd_df is None or btc_usd_df.empty
    ):
        st.error("Bitcoin-dataa ei saatu.")
        return

    btc_eur_df = btc_eur_df.copy()
    btc_usd_df = btc_usd_df.copy()

    btc_eur_df["Date"] = pd.to_datetime(btc_eur_df["Date"], errors="coerce")
    btc_usd_df["Date"] = pd.to_datetime(btc_usd_df["Date"], errors="coerce")

    btc_eur_df = btc_eur_df.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)
    btc_usd_df = btc_usd_df.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)

    latest_date = max(btc_eur_df["Date"].max(), btc_usd_df["Date"].max())
    st.caption(f"Viimeisin markkinadata: {latest_date.date()}")

    btc_vals = latest_period_values(btc_eur_df, "Close")

    ath_usd = float(pd.to_numeric(btc_usd_df["Close"], errors="coerce").max())
    atl_usd = float(pd.to_numeric(btc_usd_df["Close"], errors="coerce").min())
    ath_eur = float(pd.to_numeric(btc_eur_df["Close"], errors="coerce").max())
    atl_eur = float(pd.to_numeric(btc_eur_df["Close"], errors="coerce").min())

    drawdown_from_ath = pct_change(btc_vals["now"], ath_eur)

    btc_eur_df["ret"] = pd.to_numeric(btc_eur_df["Close"], errors="coerce").pct_change()
    vol30 = btc_eur_df["ret"].tail(30).std() * np.sqrt(365) * 100 if len(btc_eur_df) >= 30 else None

    vol_now = None
    vol_30avg = None
    if "Volume" in btc_usd_df.columns:
        vol_now = _latest_valid(btc_usd_df["Volume"])
        vol_30avg = (
            float(pd.to_numeric(btc_usd_df["Volume"], errors="coerce").tail(30).mean())
            if len(btc_usd_df) >= 30 else None
        )

    plot_df = btc_eur_df.copy()
    plot_df["Close_EUR"] = plot_df["Close"]
    plot_df["MA200_EUR"] = plot_df["Close_EUR"].rolling(200).mean()

    st.markdown("### ₿ Bitcoinin hinta")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        safe_eur_card("Nyt", btc_vals["now"], None, 0)
    with c2:
        safe_eur_card("1 kk", btc_vals["1m"], btc_vals["pct_1m"], 0)
    with c3:
        safe_eur_card("1 v", btc_vals["1y"], btc_vals["pct_1y"], 0)
    with c4:
        safe_eur_card("5 v", btc_vals["5y"], btc_vals["pct_5y"], 0)

    st.divider()

    st.markdown("### 📊 Bitcoinin tunnusluvut")
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.metric("ATH", _fmt_money(ath_eur, "€", 0))
        st.caption(_fmt_money(ath_usd, "$", 0))

    with k2:
        st.metric("ATL", _fmt_money(atl_eur, "€", 0))
        st.caption(_fmt_money(atl_usd, "$", 0))

    with k3:
        st.metric("ATH-drawdown", f"{drawdown_from_ath:+.1f} %" if drawdown_from_ath is not None else "—")
        st.caption("Nykyhinta vs kaikkien aikojen huippu")

    with k4:
        st.metric("30 pv volatiliteetti", f"{vol30:.1f} %" if vol30 is not None else "—")
        st.caption("Annualisoitu")

    if vol_now is not None or vol_30avg is not None:
        k5, k6 = st.columns(2)
        with k5:
            st.metric("Volyymi nyt", _fmt_money(vol_now, "", 0).strip())
            st.caption("Yahoo Finance volume")
        with k6:
            st.metric("Volyymi, 30 pv ka", _fmt_money(vol_30avg, "", 0).strip())
            st.caption("Keskiarvo")

    st.divider()

    t1, t2, t3 = st.tabs(["💹 Hinta", "📉 Drawdown", "📦 Volyymi"])

    with t1:
        render_price_chart_with_extra_lines(
            plot_df,
            title="Bitcoin (€)",
            key="btc_price_period",
            base_col="Close_EUR",
            extra_lines=[
                ("MA200_EUR", "MA200", {}),
            ],
            y_title="EUR",
            options=["1 kk", "1 v", "5 v", "10 v"],
            default="1 v",
            postprocess=_add_halving_lines,
        )

    with t2:
        period = period_selector(
            "Kuvaajan tarkasteluväli",
            key="btc_dd_period",
            options=["1 kk", "1 v", "5 v", "10 v"],
            default="1 v",
        )

        dd = plot_df.dropna(subset=["Close_EUR"]).copy()
        dd["rolling_ath"] = dd["Close_EUR"].cummax()
        dd["drawdown_pct"] = (dd["Close_EUR"] / dd["rolling_ath"] - 1.0) * 100.0
        dd["drawdown_pct"] = dd["drawdown_pct"].clip(upper=0)

        dd_plot = filter_by_period(dd, period, date_col="Date")

        y_min = float(dd_plot["drawdown_pct"].min()) if not dd_plot.empty else -10.0
        y_min = min(y_min - 3.0, -5.0)

        fig = px.line(
            dd_plot,
            x="Date",
            y="drawdown_pct",
            title=f"Bitcoinin drawdown ATH:sta ({period})",
            labels={"Date": "Päivä", "drawdown_pct": "%"},
        )
        fig.update_yaxes(range=[y_min, 5], ticksuffix=" %")
        fig.add_hline(y=0, line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)

    with t3:
        period = period_selector(
            "Kuvaajan tarkasteluväli",
            key="btc_vol_period",
            options=["1 kk", "1 v", "5 v", "10 v"],
            default="1 v",
        )

        if "Volume" not in btc_usd_df.columns or btc_usd_df["Volume"].dropna().empty:
            st.info("Volyymihistoriaa ei löytynyt tästä aineistosta.")
        else:
            vol_plot_df = filter_by_period(btc_usd_df.dropna(subset=["Volume"]).copy(), period, date_col="Date")

            vol_plot_df["Volume_B"] = vol_plot_df["Volume"] / 1_000_000_000

            fig = px.bar(
                vol_plot_df,
                x="Date",
                y="Volume_B",
                title=f"Bitcoinin volyymi ({period})",
                labels={"Date": "Päivä", "Volume_B": "Volyymi (mrd)"},
            )

            fig.update_yaxes(ticksuffix=" mrd")
            st.plotly_chart(fig, use_container_width=True)