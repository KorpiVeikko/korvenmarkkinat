# services/currency_view_helpers.py
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from services.currency_data import (
    CURRENCY_META,
    generate_ai_summary,
    get_major_macro_currencies,
    is_major_macro_currency,
)
from services.currency_utils import (
    ANCHOR_CURRENCY,
    build_fx_metrics,
    change_metrics,
    fmt_money_supply,
    fmt_num,
    fmt_pct,
    pct_color_style,
    to_anchor_fx,
)


DISPLAY_START_DATE = pd.Timestamp("2000-01-01")


def _show_debug(title: str, message: str | None) -> None:
    if not message:
        return
    with st.expander(title, expanded=False):
        st.code(message)


def _latest_non_null(df: pd.DataFrame, value_col: str) -> tuple[float | None, pd.Timestamp | None]:
    if df is None or df.empty or value_col not in df.columns:
        return None, None

    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=["Date", value_col]).sort_values("Date")

    if d.empty:
        return None, None

    row = d.iloc[-1]
    return float(row[value_col]), pd.to_datetime(row["Date"])


def _render_money_macro_analysis(
    currency: str,
    money_1y: float | None,
    money_5y: float | None,
    inflation: float | None,
    policy_rate: float | None,
    real_rate: float | None,
) -> None:
    st.markdown("### 🤖 Tulkinta")

    points = []

    if money_1y is not None:
        if money_1y > 6:
            points.append("Rahamäärä kasvaa melko nopeasti, mikä voi pidemmällä aikavälillä lisätä inflaatio- ja valuutan heikkenemispaineita.")
        elif money_1y > 2:
            points.append("Rahamäärä kasvaa maltillisesti, mikä viittaa normaalimpaan luotto- ja rahankiertoon.")
        else:
            points.append("Rahamäärän kasvu on hidasta, mikä voi kertoa tiukemmasta rahoitusympäristöstä.")

    if inflation is not None and policy_rate is not None:
        if policy_rate > inflation:
            points.append("Ohjauskorko on inflaatiota korkeampi, joten rahapolitiikka näyttää reaalisesti melko kireältä.")
        else:
            points.append("Ohjauskorko on inflaatiota matalampi, joten reaalinen korkotaso on edelleen löysähkö.")

    if real_rate is not None:
        if real_rate > 1:
            points.append("Reaalikorko on selvästi positiivinen, mikä tukee säästämistä ja voi hillitä kysyntää.")
        elif real_rate < 0:
            points.append("Reaalikorko on negatiivinen, jolloin inflaatio syö korkotuottoa ja ostovoimaa.")
        else:
            points.append("Reaalikorko on lähellä nollaa, eli rahapolitiikan kiristävä vaikutus on melko neutraali.")

    if money_5y is not None:
        if money_5y > 25:
            points.append("Viiden vuoden rahamäärän kasvu on voimakasta, joten pitkän aikavälin rahamäärätausta on selvästi elvyttävä.")
        elif money_5y > 10:
            points.append("Viiden vuoden rahamäärän kasvu on kohtalaista, mutta ei poikkeuksellisen rajua.")
        else:
            points.append("Viiden vuoden rahamäärän kasvu on maltillista.")

    if not points:
        st.info("Tulkintaa ei voitu muodostaa, koska dataa puuttuu.")
        return

    with st.container(border=True):
        st.markdown(f"**{currency}: rahamäärän, inflaation ja korkojen tilanne**")
        for p in points:
            st.write(f"• {p}")


def render_overview_tab(overview: pd.DataFrame) -> None:
    st.markdown("#### Seurattavat valuutat")
    st.caption(
        "**YTD %** = *year-to-date*, eli muutos vuoden alusta tähän päivään. "
        "Päätaulukossa näytetään YTD, 1 v ja 5 v, koska ne ovat yleensä käytännöllisimmät tarkastelujaksot. "
        "10 vuoden muutos jätettiin pois, koska se jää usein tyhjäksi datan kattavuuden takia eikä yleensä lisää paljon käytännön hyötyä."
    )

    if overview is None or overview.empty:
        st.warning("Valuuttayhteenvetoa ei saatu.")
        return

    show = overview.copy()

    numeric_cols = ["Nykykurssi", "YTD %", "1v %", "5v %", "Volatiliteetti 1v %", "Min", "Max"]
    for col in numeric_cols:
        show[col] = pd.to_numeric(show[col], errors="coerce")

    styled = (
        show[["Koodi", "Valuutta", "Nykykurssi", "YTD %", "1v %", "5v %", "Volatiliteetti 1v %", "Min", "Max", "Viimeisin päivä"]]
        .style
        .format(
            {
                "Nykykurssi": "{:.4f}",
                "YTD %": "{:+.1f} %",
                "1v %": "{:+.1f} %",
                "5v %": "{:+.1f} %",
                "Volatiliteetti 1v %": "{:.1f} %",
                "Min": "{:.4f}",
                "Max": "{:.4f}",
            },
            na_rep="—",
        )
        .map(pct_color_style, subset=["YTD %", "1v %", "5v %"])
    )

    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("#### Ranking")

    rank_metric = st.selectbox(
        "Valitse ranking-mittari",
        ["YTD %", "1v %", "5v %"],
        index=1,
        key="currency_rank_metric",
    )

    rank_df = show[["Koodi", "Valuutta", rank_metric]].copy()
    rank_df[rank_metric] = pd.to_numeric(rank_df[rank_metric], errors="coerce")
    rank_df = rank_df.dropna(subset=[rank_metric]).sort_values(rank_metric, ascending=False)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Vahvimmat**")
        st.dataframe(rank_df.head(5), use_container_width=True, hide_index=True)

    with c2:
        st.markdown("**Heikoimmat**")
        st.dataframe(
            rank_df.tail(5).sort_values(rank_metric, ascending=True),
            use_container_width=True,
            hide_index=True,
        )


def render_fx_tab(
    fx_currency: str,
    years: int,
    load_currency_bundle,
) -> None:
    fx_bundle = load_currency_bundle(fx_currency, years=years)
    anchor_bundle = load_currency_bundle(ANCHOR_CURRENCY, years=years)

    fx = to_anchor_fx(fx_bundle["fx"], anchor_bundle["fx"])
    fx_metrics = build_fx_metrics(fx)

    if fx is None or fx.empty:
        st.warning("Kurssihistoriaa ei saatu.")
        return

    st.markdown(f"#### {fx_currency} / {ANCHOR_CURRENCY}")

    k1, k2, k3, k4 = st.columns(4, gap="large")

    with k1:
        st.metric(
            f"{fx_currency} / {ANCHOR_CURRENCY}",
            fmt_num(fx_metrics.latest_rate, 4),
            f"{fmt_pct(fx_metrics.ytd_pct)} (YTD)" if fx_metrics.ytd_pct is not None else None,
        )
        if fx_metrics.latest_date is not None:
            st.caption(f"Päivä: {fx_metrics.latest_date.date()}")

    with k2:
        st.metric("Muutos 1 v", fmt_pct(fx_metrics.change_1y_pct))
        st.caption("Valuuttakurssi")

    with k3:
        st.metric("Muutos 5 v", fmt_pct(fx_metrics.change_5y_pct))
        st.caption("Valuuttakurssi")

    with k4:
        st.metric("Volatiliteetti 1 v", fmt_pct(fx_metrics.volatility_1y_pct))
        st.caption("Annualisoitu")

    st.divider()

    fig = px.line(
        fx,
        x="Date",
        y="Rate",
        title=f"{fx_currency} / {ANCHOR_CURRENCY} – viimeiset {years} vuotta",
        labels={"Date": "Päivä", "Rate": f"{fx_currency} per {ANCHOR_CURRENCY}"},
    )

    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Min", fmt_num(fx_metrics.min_rate, 4))
    with c2:
        st.metric("Max", fmt_num(fx_metrics.max_rate, 4))


def render_money_macro_tab(
    money_currency: str,
    years: int,
    load_currency_bundle,
) -> None:
    money_bundle = load_currency_bundle(money_currency, years=years)
    money = money_bundle["money"]
    macro = money_bundle["macro"]
    debug = money_bundle.get("debug", {})

    st.markdown(f"#### {money_currency} – rahamäärä, inflaatio ja korot")
    st.caption(
        "Tässä välilehdessä näytetään vain USD ja EUR, jotta mukana voidaan käyttää tuoreempia kuukausisarjoja. "
        "Valuuttakurssit näkyvät edelleen kaikille valuutoille erikseen."
    )

    with st.expander("Mitä broad money tarkoittaa?"):
        st.write(
            "Broad money tarkoittaa laajaa rahamäärää taloudessa. "
            "Se sisältää käteisen, käyttötilit sekä muita melko helposti rahaksi muutettavia talletuksia ja likvidejä varoja. "
            "Karkeasti se kertoo, kuinka paljon rahaa ja ostovoimaa on kierrossa pankkijärjestelmässä."
        )

    if not is_major_macro_currency(money_currency):
        st.info("Tälle valuutalle näytetään vain kurssidata. Rahamäärä- ja makrodata on rajattu USD:iin ja EUR:oon.")
        return

    money_metrics = change_metrics(money, "Date", "BroadMoney_LCU")

    if money is None or money.empty:
        st.info("Rahamäärädataa ei saatu tälle valuutalle.")
        _show_debug("Rahamäärädatan debug", debug.get("money"))
    else:
        latest_candidates = money.dropna(how="all", subset=["BroadMoney_LCU", "BroadMoney_GrowthPct"])
        if latest_candidates.empty:
            st.info("Rahamäärädataa ei saatu tälle valuutalle.")
            _show_debug("Rahamäärädatan debug", debug.get("money"))
        else:
            c1, c2, c3 = st.columns(3, gap="large")

            with c1:
                st.metric(
                    "Broad money",
                    fmt_money_supply(money_metrics.latest_value, money_currency),
                )
                st.caption(
                    f"Päivä: {money_metrics.latest_date.date() if money_metrics.latest_date is not None else '—'}"
                )

            with c2:
                st.metric("Broad money muutos 1 v", fmt_pct(money_metrics.change_1y_pct))
                st.caption("Taso")

            with c3:
                st.metric("Broad money muutos 5 v", fmt_pct(money_metrics.change_5y_pct))
                st.caption("Taso")

            plot_df = money.melt(
                id_vars=["Date"],
                value_vars=["BroadMoney_GrowthPct"],
                var_name="Sarja",
                value_name="Arvo",
            ).dropna()

            if not plot_df.empty:
                plot_df = plot_df[plot_df["Date"] >= DISPLAY_START_DATE].copy()

                fig = px.line(
                    plot_df,
                    x="Date",
                    y="Arvo",
                    color="Sarja",
                    title=f"{money_currency} – broad money -kasvu",
                    labels={"Date": "Päivä", "Arvo": "%", "Sarja": ""},
                )
                st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("#### Inflaatio ja korko")
    st.caption(
        "**Ohjauskorko** on keskuspankin keskeinen korkotaso, jolla se ohjaa rahan hintaa ja talouden aktiivisuutta. "
        "**Reaalikorko-proxy** on tässä yksinkertaistus: ohjauskorko miinus inflaatio. "
        "Positiivinen arvo tarkoittaa karkeasti, että korkotaso on inflaatiota korkeampi, negatiivinen että inflaatio syö korkotuoton."
    )

    if macro is None or macro.empty:
        st.info("Inflaatio- tai korkodataa ei saatu tälle valuutalle.")
        _show_debug("Makrodatan debug", debug.get("macro"))
        return

    inf_val, inf_date = _latest_non_null(macro, "InflationCPI_Pct")
    pol_val, pol_date = _latest_non_null(macro, "PolicyRate_Pct")
    real_val, real_date = _latest_non_null(macro, "RealInterestRate_Pct")

    if inf_val is None and pol_val is None and real_val is None:
        st.info("Inflaatio- tai korkodataa ei saatu tälle valuutalle.")
        _show_debug("Makrodatan debug", debug.get("macro"))
        return

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Inflaatio", fmt_pct(inf_val))
        st.caption(f"Päivä: {inf_date.date()}" if inf_date is not None else "Päivä: —")

    with c2:
        st.metric("Ohjauskorko", fmt_pct(pol_val))
        st.caption(f"Päivä: {pol_date.date()}" if pol_date is not None else "Päivä: —")

    with c3:
        st.metric("Reaalikorko-proxy", fmt_pct(real_val))
        st.caption(f"Päivä: {real_date.date()}" if real_date is not None else "Päivä: —")

    _render_money_macro_analysis(
        currency=money_currency,
        money_1y=money_metrics.change_1y_pct,
        money_5y=money_metrics.change_5y_pct,
        inflation=inf_val,
        policy_rate=pol_val,
        real_rate=real_val,
    )

    macro_plot = macro.melt(
        id_vars=["Date"],
        value_vars=["InflationCPI_Pct", "PolicyRate_Pct", "RealInterestRate_Pct"],
        var_name="Sarja",
        value_name="Arvo",
    ).dropna()

    if not macro_plot.empty:
        macro_plot = macro_plot[macro_plot["Date"] >= DISPLAY_START_DATE].copy()

        fig = px.line(
            macro_plot,
            x="Date",
            y="Arvo",
            color="Sarja",
            title=f"{money_currency} – inflaatio, korko ja reaalikorko-proxy",
            labels={"Date": "Päivä", "Arvo": "%", "Sarja": ""},
        )
        st.plotly_chart(fig, use_container_width=True)

    _show_debug("Makrodatan huomautus", debug.get("macro"))



def currency_format_func(code: str) -> str:
    return f"{code} – {CURRENCY_META[code]['name']}"


def macro_currency_format_func(code: str) -> str:
    return f"{code} – {CURRENCY_META[code]['name']}"


def get_macro_currency_options() -> list[str]:
    return get_major_macro_currencies()