# tabs/currency.py
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from services.currency_data import (
    CURRENCY_META,
    fetch_currency_bundle,
    fetch_major_currency_overview,
    generate_ai_summary,
)


def _fmt_num(x: float | None, decimals: int = 4) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x:,.{decimals}f}".replace(",", " ")


def _fmt_pct(x: float | None, decimals: int = 1) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x:+.{decimals}f} %"


def _fmt_money_supply(x: float | None) -> str:
    if x is None or pd.isna(x):
        return "—"

    x = float(x)
    abs_x = abs(x)

    if abs_x >= 1_000_000_000_000:
        return f"{x / 1_000_000_000_000:,.1f} bilj.".replace(",", " ")
    if abs_x >= 1_000_000_000:
        return f"{x / 1_000_000_000:,.1f} mrd.".replace(",", " ")
    if abs_x >= 1_000_000:
        return f"{x / 1_000_000:,.1f} milj.".replace(",", " ")
    return f"{x:,.0f}".replace(",", " ")


@st.cache_data(show_spinner="Haetaan valuuttakursseja (ECB)…")
def load_currency_overview(years: int = 10) -> pd.DataFrame:
    return fetch_major_currency_overview(years=years)


@st.cache_data(show_spinner="Haetaan valuutan tarkemmat tiedot…")
def load_currency_bundle(currency: str, years: int = 10) -> dict:
    return fetch_currency_bundle(currency, years=years)


def render() -> None:
    st.subheader("💱 Valuuttakurssit")
    st.caption(
        "Päivittäiset EUR-pohjaiset valuuttakurssit sekä valitun valuutta-alueen rahamääräindikaattoreita. "
        "Kurssi tarkoittaa, kuinka monta yksikköä kyseistä valuuttaa saa yhdellä eurolla."
    )

    with st.sidebar:
        st.markdown("### 💱 Valuuttatabi")
        selected_currency = st.selectbox(
            "Valitse valuutta",
            list(CURRENCY_META.keys()),
            format_func=lambda c: f"{c} – {CURRENCY_META[c]['name']}",
            key="currency_selected_code",
        )
        years = st.slider("Historiapituus (vuotta)", 3, 15, 10, key="currency_years")

    overview = load_currency_overview(years=years)
    bundle = load_currency_bundle(selected_currency, years=years)

    fx = bundle["fx"]
    money = bundle["money"]
    metrics = bundle["metrics"]

    # KPI-rivi
    k1, k2, k3, k4, k5 = st.columns(5, gap="large")

    with k1:
        st.metric(
            f"{selected_currency} / EUR",
            _fmt_num(metrics.latest_rate, 4),
            _fmt_pct(metrics.ytd_pct),
        )
        if metrics.latest_date is not None:
            st.caption(f"Päivä: {metrics.latest_date.date()}")

    with k2:
        st.metric("Muutos 1 v", _fmt_pct(metrics.change_1y_pct))
        st.caption("Valuuttakurssi")

    with k3:
        st.metric("Muutos 5 v", _fmt_pct(metrics.change_5y_pct))
        st.caption("Valuuttakurssi")

    with k4:
        st.metric("Muutos 10 v", _fmt_pct(metrics.change_10y_pct))
        st.caption("Valuuttakurssi")

    with k5:
        st.metric("Volatiliteetti 1 v", _fmt_pct(metrics.volatility_1y_pct))
        st.caption("Annualisoitu")

    st.divider()

    t1, t2, t3, t4 = st.tabs([
        "📋 Yleiskuva 10 valuutasta",
        "📈 Kurssikehitys",
        "🏦 Rahamäärä",
        "🧠 AI-yhteenveto",
    ])

    with t1:
        st.markdown("#### Suurimmat / seuratuimmat valuutat")
        if overview is None or overview.empty:
            st.warning("Valuuttayhteenvetoa ei saatu.")
        else:
            show = overview.copy()
            for col in ["Nykykurssi", "YTD %", "1v %", "5v %", "10v %", "Volatiliteetti 1v %", "10v min", "10v max"]:
                show[col] = pd.to_numeric(show[col], errors="coerce")

            st.dataframe(
                show,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Nykykurssi": st.column_config.NumberColumn(format="%.4f"),
                    "YTD %": st.column_config.NumberColumn(format="%.1f %%"),
                    "1v %": st.column_config.NumberColumn(format="%.1f %%"),
                    "5v %": st.column_config.NumberColumn(format="%.1f %%"),
                    "10v %": st.column_config.NumberColumn(format="%.1f %%"),
                    "Volatiliteetti 1v %": st.column_config.NumberColumn(format="%.1f %%"),
                    "10v min": st.column_config.NumberColumn(format="%.4f"),
                    "10v max": st.column_config.NumberColumn(format="%.4f"),
                },
            )

    with t2:
        st.markdown(f"#### {selected_currency} – kurssikehitys")
        if fx is None or fx.empty:
            st.warning("Kurssihistoriaa ei saatu.")
        else:
            fig = px.line(
                fx,
                x="Date",
                y="Rate",
                title=f"{selected_currency} / EUR – viimeiset {years} vuotta",
                labels={"Date": "Päivä", "Rate": f"{selected_currency} per EUR"},
            )
            st.plotly_chart(fig, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                st.metric("10 v min", _fmt_num(metrics.min_10y, 4))
            with c2:
                st.metric("10 v max", _fmt_num(metrics.max_10y, 4))

    with t3:
        st.markdown(f"#### {selected_currency} – rahamäärä ja kasvu")
        st.caption("Rahamäärä = broad money / laaja raha. Tiedot ovat vuositasoisia ja viimeisin saatavilla oleva vuosi vaihtelee maittain.")

        if money is None or money.empty:
            st.info("Rahamäärädataa ei saatu tälle valuutalle.")
        else:
            latest_row = money.dropna(how="all", subset=["BroadMoney_LCU", "BroadMoney_GrowthPct", "BroadMoney_GDPPct"]).iloc[-1]

            c1, c2, c3 = st.columns(3, gap="large")
            with c1:
                st.metric("Broad money", _fmt_money_supply(latest_row.get("BroadMoney_LCU")))
                st.caption(f"Vuosi: {int(latest_row['Year'])}")
            with c2:
                st.metric("Broad money kasvu", _fmt_pct(latest_row.get("BroadMoney_GrowthPct")))
                st.caption(f"Vuosi: {int(latest_row['Year'])}")
            with c3:
                st.metric("Broad money / BKT", _fmt_pct(latest_row.get("BroadMoney_GDPPct")))
                st.caption(f"Vuosi: {int(latest_row['Year'])}")

            money_plot = money.melt(
                id_vars=["Year"],
                value_vars=["BroadMoney_GrowthPct", "BroadMoney_GDPPct"],
                var_name="Sarja",
                value_name="Arvo",
            ).dropna()

            if not money_plot.empty:
                fig = px.line(
                    money_plot,
                    x="Year",
                    y="Arvo",
                    color="Sarja",
                    markers=True,
                    title=f"{selected_currency} – broad money -tunnusluvut",
                    labels={"Year": "Vuosi", "Arvo": "Arvo", "Sarja": ""},
                )
                st.plotly_chart(fig, use_container_width=True)

    with t4:
        st.markdown("#### AI-yhteenveto")
        st.info(generate_ai_summary(selected_currency, metrics, money))