# tabs/gold.py
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from services.market_data import fetch_price_history_eur
from services.asset_ui import (
    latest_period_values,
    pct_change,
    filter_by_period,
    period_selector,
    render_price_chart,
)


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def load_gold() -> pd.DataFrame:
    return fetch_price_history_eur("GC=F", period="5y")


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def load_silver() -> pd.DataFrame:
    return fetch_price_history_eur("SI=F", period="5y")


def _fmt_money(x: float | None, decimals: int = 0) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{float(x):,.{decimals}f} €".replace(",", " ")


def _fmt_number(x: float | None, decimals: int = 2) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{float(x):,.{decimals}f}".replace(",", " ")


def _pct_color(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "#6b7280"
    return "#15803d" if value >= 0 else "#b91c1c"

def _pct_text(value: float | None, label: str = "") -> str:
    if value is None or pd.isna(value):
        return "—"
    suffix = f" ({label})" if label else ""
    return f"{value:+.1f} %"


def _pct_html(value: float | None, label: str = "") -> str:
    if value is None or pd.isna(value):
        txt = "—"
    else:
        suffix = f" ({label})" if label else ""
        txt = f"{value:+.1f} %{suffix}"

    return f"""
    <span style="
        color:{_pct_color(value)};
        font-weight:700;
        font-size:1.05rem;
    ">
        {txt}
    </span>
    """


def _price_card(label: str, value: float | None, pct: float | None = None, decimals: int = 0) -> None:
    with st.container(border=True):
        st.caption(label)
        st.markdown(f"## {_fmt_money(value, decimals)}")
        if pct is not None and not pd.isna(pct):
            st.markdown(_pct_html(pct), unsafe_allow_html=True)


def _number_card(label: str, value: float | None, pct: float | None = None, decimals: int = 2, caption: str | None = None) -> None:
    with st.container(border=True):
        st.caption(label)
        st.markdown(f"## {_fmt_number(value, decimals)}")
        if pct is not None and not pd.isna(pct):
            st.markdown(_pct_html(pct), unsafe_allow_html=True)
        if caption:
            st.caption(caption)


def _status_from_pct(pct: float | None) -> tuple[str, str]:
    if pct is None or pd.isna(pct):
        return "⚪", "Ei dataa"
    if pct >= 10:
        return "🟢", "Vahva nousu"
    if pct >= 2:
        return "🟢", "Nousussa"
    if pct > -2:
        return "🟡", "Vakaa"
    if pct > -10:
        return "🟠", "Laskussa"
    return "🔴", "Selvä lasku"


def _status_from_ratio(ratio_vs_mean: float | None) -> tuple[str, str]:
    if ratio_vs_mean is None or pd.isna(ratio_vs_mean):
        return "⚪", "Ei dataa"
    if ratio_vs_mean > 15:
        return "🟠", "Hopea suhteessa halpa"
    if ratio_vs_mean < -15:
        return "🟠", "Hopea suhteessa kallis"
    return "🟡", "Suhde lähellä keskiarvoa"


def _render_signal_cards(gold_vals: dict, silver_vals: dict, ratio_vals: dict, ratio_vs_mean: float | None) -> None:
    gold_icon, gold_status = _status_from_pct(gold_vals.get("pct_1y"))
    silver_icon, silver_status = _status_from_pct(silver_vals.get("pct_1y"))
    ratio_icon, ratio_status = _status_from_ratio(ratio_vs_mean)

    st.markdown("### 📌 Tilannekuva")

    c1, c2, c3 = st.columns(3)

    with c1:
        with st.container(border=True):
            st.markdown(f"### {gold_icon} Kulta")
            st.markdown(f"**Tila:** {gold_status}")
            st.markdown(_pct_html(gold_vals.get("pct_1y"), "1 v"), unsafe_allow_html=True)
            st.caption("Kullan euromääräinen kehitys vuoden aikana.")

    with c2:
        with st.container(border=True):
            st.markdown(f"### {silver_icon} Hopea")
            st.markdown(f"**Tila:** {silver_status}")
            st.markdown(_pct_html(silver_vals.get("pct_1y"), "1 v"), unsafe_allow_html=True)
            st.caption("Hopean euromääräinen kehitys vuoden aikana.")

    with c3:
        with st.container(border=True):
            st.markdown(f"### {ratio_icon} Kulta / hopea")
            st.markdown(f"**Tila:** {ratio_status}")
            st.markdown(_pct_html(ratio_vs_mean, "vs ka."), unsafe_allow_html=True)
            st.caption("Nykyinen suhdeluku verrattuna 5 vuoden keskiarvoon.")


def _render_analysis(gold_vals: dict, silver_vals: dict, ratio_vals: dict, ratio_vs_mean: float | None) -> None:
    gold_1y = gold_vals.get("pct_1y")
    silver_1y = silver_vals.get("pct_1y")
    ratio_now = ratio_vals.get("now")

    parts = []

    if gold_1y is not None:
        if gold_1y > 10:
            parts.append("Kulta on vahvassa nousussa vuoden takaiseen verrattuna, mikä voi kertoa turvasatamakysynnästä tai reaalikorkoihin liittyvästä tuesta.")
        elif gold_1y < -10:
            parts.append("Kullan vuoden kehitys on selvästi negatiivinen, mikä voi viitata riskinottohalun kasvuun tai jalometallien kysynnän heikkenemiseen.")
        else:
            parts.append("Kullan kehitys on vuoden tasolla melko maltillinen.")

    if silver_1y is not None:
        if silver_1y > gold_1y if gold_1y is not None else False:
            parts.append("Hopea on kehittynyt kultaa vahvemmin, mikä voi viitata syklisempään riskinottoon jalometalleissa.")
        elif gold_1y is not None and silver_1y < gold_1y:
            parts.append("Hopea on jäänyt kullasta jälkeen, mikä voi kertoa varovaisemmasta teollisesta kysyntäkuvasta.")

    if ratio_now is not None and ratio_vs_mean is not None:
        if ratio_vs_mean > 15:
            parts.append("Kulta/hopea-suhde on selvästi keskiarvon yläpuolella, eli hopea näyttää suhteessa kultaan halvemmalta kuin viime vuosina keskimäärin.")
        elif ratio_vs_mean < -15:
            parts.append("Kulta/hopea-suhde on selvästi keskiarvon alapuolella, eli hopea on suhteessa kultaan kallistunut.")
        else:
            parts.append("Kulta/hopea-suhde on lähellä viime vuosien keskiarvoa.")

    if not parts:
        parts.append("Analyysia ei voitu muodostaa, koska keskeisiä muutostietoja puuttuu.")

    st.markdown("### 🧠 Jalometallianalyysi")
    with st.container(border=True):
        st.write(" ".join(parts))

    st.info("Tämä ei ole sijoitussuositus. Kulta ja hopea voivat reagoida voimakkaasti korkoihin, dollariin, inflaatio-odotuksiin ja riskisentimenttiin.")


def render() -> None:
    st.subheader("🟨 Kulta & ⬜ Hopea")
    st.caption("Lähde: Yahoo Finance. Euromääräinen hinta muodostetaan USD-hinnoista ja EUR/USD-kurssista.")

    gold_df = load_gold()
    silver_df = load_silver()

    if gold_df is None or gold_df.empty or silver_df is None or silver_df.empty:
        st.error("Dataa ei saatu ladattua.")
        return

    gold_df = gold_df.copy()
    silver_df = silver_df.copy()

    gold_df["Date"] = pd.to_datetime(gold_df["Date"], errors="coerce")
    silver_df["Date"] = pd.to_datetime(silver_df["Date"], errors="coerce")

    gold_df["Close"] = pd.to_numeric(gold_df["Close"], errors="coerce")
    silver_df["Close"] = pd.to_numeric(silver_df["Close"], errors="coerce")

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

    ratio_for_values = ratio_df.rename(columns={"Ratio": "Close"})
    ratio_vals = latest_period_values(ratio_for_values, "Close")

    ratio_mean = float(ratio_df["Ratio"].mean()) if not ratio_df.empty else None
    ratio_vs_mean = pct_change(ratio_vals["now"], ratio_mean) if ratio_mean is not None else None
    ratio_df["LongRunMean"] = ratio_mean

    st.markdown("### 💶 Hinnat euroissa")

    c1, c2 = st.columns(2)


    def _metal_summary_card(
        title: str,
        value: float | None,
        values: dict,
        decimals: int,
    ) -> None:
        with st.container(border=True):
            st.markdown(f"### {title}")
            st.caption("Nykyinen hinta")
            st.markdown(f"# {_fmt_money(value, decimals)}")

            st.divider()

            for label, pct in [
                ("1 kk", values.get("pct_1m")),
                ("1 vuosi", values.get("pct_1y")),
                ("5 vuotta", values.get("pct_5y")),
            ]:
                icon = "↗" if pct is not None and pct >= 0 else "↘"
                color = "#15803d" if pct is not None and pct >= 0 else "#b91c1c"

                st.markdown(
                    f"""
                    <div style="
                        display:flex;
                        justify-content:space-between;
                        align-items:center;
                        padding:0.55rem 0;
                        border-bottom:1px solid #e5e7eb;
                    ">
                        <span style="color:#6b7280;">{icon} {label}</span>
                        <span style="color:{color}; font-weight:700; font-size:1.05rem;">
                            {_pct_text(pct)}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


    with c1:
        _metal_summary_card(
            title="🥇 Kulta",
            value=gold_vals["now"],
            values=gold_vals,
            decimals=0,
        )

    with c2:
        _metal_summary_card(
            title="🥈 Hopea",
            value=silver_vals["now"],
            values=silver_vals,
            decimals=2,
        )

    st.divider()

    tab_gold, tab_silver, tab_ratio, tab_analysis = st.tabs(
        ["🥇 Kulta", "🥈 Hopea", "⚖️ Kulta / hopea", "🧠 Analyysi"]
    )

    
    with tab_gold:
        st.markdown("### 🥇 Kullan hinta")

        render_price_chart(
            gold_df,
            "Kulta (€)",
            key="gold_chart",
            y_col="Close",
            y_title="€",
            options=["1 kk", "1 v", "5 v"],
            default="1 v",
        )

    with tab_silver:
        st.markdown("### 🥈 Hopean hinta")
 
        render_price_chart(
            silver_df,
            "Hopea (€)",
            key="silver_chart",
            y_col="Close",
            y_title="€",
            options=["1 kk", "1 v", "5 v"],
            default="1 v",
        )

    with tab_ratio:
        st.markdown("### ⚖️ Kulta / hopea -suhde")

        c1, c2, c3 = st.columns(3)

        with c1:
            _number_card("Suhde nyt", ratio_vals["now"], None, 2)
        with c2:
            _number_card("5 vuoden keskiarvo", ratio_mean, ratio_vs_mean, 2, "Nykyinen suhde vs keskiarvo")
        with c3:
            _number_card("1 v sitten", ratio_vals["1y"], ratio_vals["pct_1y"], 2)

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

        if "LongRunMean" in ratio_plot_df.columns and ratio_plot_df["LongRunMean"].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=ratio_plot_df["Date"],
                    y=ratio_plot_df["LongRunMean"],
                    mode="lines",
                    name="5 vuoden keskiarvo",
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

    with tab_analysis:
        _render_signal_cards(gold_vals, silver_vals, ratio_vals, ratio_vs_mean)
        st.divider()
        _render_analysis(gold_vals, silver_vals, ratio_vals, ratio_vs_mean)

        






