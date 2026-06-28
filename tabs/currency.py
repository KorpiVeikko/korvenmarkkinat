# tabs/currency.py
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st


from services.currency_utils import (
    ANCHOR_CURRENCY,
    build_fx_metrics,
    fmt_num,
    fmt_pct,
    to_anchor_fx,
)
from services.currency_view_helpers import (
    currency_format_func,
    get_macro_currency_options,
    macro_currency_format_func,
    render_money_macro_tab,
    render_overview_tab,
    render_currency_analysis_tab,
)

from services.currency_data import (
    CURRENCY_META,
    fetch_currency_bundle,
    fetch_ecb_fx_series,
    fetch_money_supply_panel,
    fetch_macro_context_panel,
    fetch_central_bank_balance_sheets,
)



@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def load_fx_series(currency: str, years: int = 10) -> pd.DataFrame:
    return fetch_ecb_fx_series(currency, years=years)


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def load_currency_bundle(currency: str, years: int = 10) -> dict:
    return fetch_currency_bundle(currency, years=years)

@st.cache_data(ttl=60 * 60 * 24 * 7, show_spinner="Ladataan rahamäärä- ja makrodataa…")
def load_money_macro_bundle(currency: str, years: int = 10) -> dict:
    fx = fetch_ecb_fx_series(currency, years=years)
    money, money_debug = fetch_money_supply_panel(currency, years=years)
    macro, macro_debug = fetch_macro_context_panel(currency, years=years)

    return {
        "fx": fx,
        "money": money,
        "macro": macro,
        "debug": {
            "money": money_debug,
            "macro": macro_debug,
        },
    }


@st.cache_data(ttl=60 * 60 * 24, show_spinner="Rakennetaan valuuttayhteenveto…")
def load_currency_overview_anchor(years: int = 10) -> pd.DataFrame:
    anchor_fx = load_fx_series(ANCHOR_CURRENCY, years=years)

    rows = []

    for code, meta in CURRENCY_META.items():
        if code == ANCHOR_CURRENCY:
            continue

        fx = load_fx_series(code, years=years)
        fx_anchor = to_anchor_fx(fx, anchor_fx)
        metrics = build_fx_metrics(fx_anchor)

        rows.append(
            {
                "Koodi": code,
                "Valuutta": meta["name"],
                "Nykykurssi": metrics.latest_rate,
                "YTD %": metrics.ytd_pct,
                "1v %": metrics.change_1y_pct,
                "5v %": metrics.change_5y_pct,
                "Volatiliteetti 1v %": metrics.volatility_1y_pct,
                "Min": metrics.min_rate,
                "Max": metrics.max_rate,
                "Viimeisin päivä": metrics.latest_date.date() if metrics.latest_date is not None else None,
            }
        )

    return pd.DataFrame(rows)

@st.cache_data(ttl=60 * 60 * 24, show_spinner="Ladataan keskuspankkien taseita…")
def load_central_bank_balance_sheets(years: int = 10) -> dict:
    df, debug = fetch_central_bank_balance_sheets(years=years)
    return {"data": df, "debug": debug}


def _fx_traffic_light(value: float | None, kind: str) -> tuple[str, str]:
    if value is None or pd.isna(value):
        return "⚪", "Ei dataa"

    v = abs(float(value))

    if kind == "change":
        if v < 5:
            return "🟢", "Maltillinen"
        if v < 15:
            return "🟡", "Selvä liike"
        return "🔴", "Voimakas liike"

    if kind == "volatility":
        if v < 8:
            return "🟢", "Rauhallinen"
        if v < 15:
            return "🟡", "Koholla"
        return "🔴", "Voimakas"

    return "⚪", "Ei luokitusta"


def _fx_score(metrics) -> tuple[int, str, str]:
    score = 0

    if metrics.ytd_pct is not None and abs(metrics.ytd_pct) < 5:
        score += 1

    if metrics.change_1y_pct is not None and abs(metrics.change_1y_pct) < 5:
        score += 1

    if metrics.change_5y_pct is not None and abs(metrics.change_5y_pct) < 15:
        score += 1

    if metrics.volatility_1y_pct is not None and metrics.volatility_1y_pct < 8:
        score += 1

    if score >= 3:
        return score, "🟢", "Vakaa"
    if score >= 2:
        return score, "🟡", "Neutraali / seurattava"
    return score, "🔴", "Heilahteleva"


def _render_fx_summary_card(pair_label: str, metrics) -> None:
    score, icon, status = _fx_score(metrics)

    ytd_icon, ytd_label = _fx_traffic_light(metrics.ytd_pct, "change")
    ch1_icon, ch1_label = _fx_traffic_light(metrics.change_1y_pct, "change")
    ch5_icon, ch5_label = _fx_traffic_light(metrics.change_5y_pct, "change")
    vol_icon, vol_label = _fx_traffic_light(metrics.volatility_1y_pct, "volatility")

    points = []

    if metrics.ytd_pct is not None:
        if metrics.ytd_pct > 0:
            points.append("Valuutta on vahvistunut vuoden alusta.")
        elif metrics.ytd_pct < 0:
            points.append("Valuutta on heikentynyt vuoden alusta.")

    if metrics.change_1y_pct is not None:
        if abs(metrics.change_1y_pct) < 5:
            points.append("Vuoden muutos on ollut maltillinen.")
        else:
            points.append("Vuoden aikana kurssissa on ollut selvä liike.")

    if metrics.volatility_1y_pct is not None:
        if metrics.volatility_1y_pct < 8:
            points.append("Volatiliteetti on ollut rauhallista.")
        else:
            points.append("Volatiliteetti on koholla.")

    with st.container(border=True):
        top1, top2 = st.columns([1.4, 0.8])

        with top1:
            st.markdown(f"### {icon} {pair_label}: {status}")
            st.caption("Yhteenveto valuuttakurssin muutoksesta ja heilunnasta.")

        with top2:
            st.metric("Kurssipisteet", f"{score} / 4")

        k1, k2, k3, k4, k5 = st.columns(5)

        with k1:
            st.caption("Nykykurssi")
            st.markdown(f"### {fmt_num(metrics.latest_rate, 4)}")
            if metrics.latest_date is not None:
                st.caption(f"Päivä: {metrics.latest_date.date()}")

        with k2:
            st.caption("YTD")
            st.markdown(f"### {ytd_icon} {fmt_pct(metrics.ytd_pct)}")
            st.caption(ytd_label)

        with k3:
            st.caption("Muutos 1 v")
            st.markdown(f"### {ch1_icon} {fmt_pct(metrics.change_1y_pct)}")
            st.caption(ch1_label)

        with k4:
            st.caption("Muutos 5 v")
            st.markdown(f"### {ch5_icon} {fmt_pct(metrics.change_5y_pct)}")
            st.caption(ch5_label)

        with k5:
            st.caption("Volatiliteetti 1 v")
            st.markdown(f"### {vol_icon} {fmt_pct(metrics.volatility_1y_pct)}")
            st.caption(vol_label)

        if points:
            st.markdown("**Tulkinta**")
            for p in points:
                st.write(f"• {p}")


def _filter_fx_window(fx: pd.DataFrame, window_label: str) -> pd.DataFrame:
    if fx is None or fx.empty:
        return pd.DataFrame()

    d = fx.copy()
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d = d.dropna(subset=["Date"]).sort_values("Date")

    if d.empty:
        return d

    latest = d["Date"].max()

    years_map = {
        "1 v": 1,
        "3 v": 3,
        "5 v": 5,
        "10 v": 10,
    }

    years_back = years_map.get(window_label, 10)
    cutoff = latest - pd.DateOffset(years=years_back)

    return d[d["Date"] >= cutoff].copy()


def _prepare_fx_plot_df(fx: pd.DataFrame, display_mode: str) -> tuple[pd.DataFrame, str, str]:
    plot_df = fx.copy()

    if display_mode == "Indeksi 100":
        first = pd.to_numeric(plot_df["Rate"], errors="coerce").dropna()
        if first.empty:
            return pd.DataFrame(), "Rate", "Kurssi"

        base = float(first.iloc[0])
        if base == 0:
            return pd.DataFrame(), "Rate", "Kurssi"

        plot_df["Index"] = plot_df["Rate"] / base * 100.0
        return plot_df, "Index", "Indeksi, alku = 100"

    return plot_df, "Rate", "Kurssi"



def render_fx_tab_fast(fx_currency: str, years: int) -> None:
    fx_raw = load_fx_series(fx_currency, years=years)
    anchor_fx = load_fx_series(ANCHOR_CURRENCY, years=years)

    fx = to_anchor_fx(fx_raw, anchor_fx)

    if fx is None or fx.empty:
        st.warning("Kurssihistoriaa ei saatu.")
        return

    metrics = build_fx_metrics(fx)
    pair_label = f"{fx_currency} / {ANCHOR_CURRENCY}"

    st.markdown(f"#### {pair_label} – kurssikehitys")

    _render_fx_summary_card(pair_label, metrics)

    st.divider()
    st.markdown("### 📈 Kuvaaja")

    c1, c2 = st.columns([1, 1])

    with c1:
        window_label = st.radio(
            "Aikajakso",
            ["1 v", "3 v", "5 v", "10 v"],
            index=3,
            horizontal=True,
            key=f"fx_window_{fx_currency}",
        )

    with c2:
        display_mode = st.radio(
            "Näyttötapa",
            ["Kurssi", "Indeksi 100"],
            index=0,
            horizontal=True,
            key=f"fx_display_mode_{fx_currency}",
        )

    plot_source = _filter_fx_window(fx, window_label)
    plot_df, y_col, y_label = _prepare_fx_plot_df(plot_source, display_mode)

    if plot_df is None or plot_df.empty:
        st.info("Kuvaajadataa ei saatu valitulle aikajaksolle.")
        return

    fig = px.line(
        plot_df,
        x="Date",
        y=y_col,
        title=f"{pair_label} – {window_label}, {display_mode.lower()}",
        labels={
            "Date": "Päivä",
            y_col: y_label,
        },
    )

    st.plotly_chart(fig, use_container_width=True)


def render() -> None:
    st.subheader("💱 Valuuttakurssit")
    st.caption(
        "Näkymä käyttää **USD:tä ankkurivaluuttana**. "
        "Kurssi tarkoittaa, kuinka monta yksikköä kyseistä valuuttaa saa yhdellä Yhdysvaltain dollarilla. "
        "Positiivinen muutos tarkoittaa, että yhtä USD:tä kohden saa enemmän kyseistä valuuttaa kuin aiemmin. "
        f"{ANCHOR_CURRENCY} itse on jätetty pois FX-vertailusta, koska se olisi ankkurina aina 1.0."
    )

    years = 10

    fx_codes = [c for c in CURRENCY_META.keys() if c != ANCHOR_CURRENCY]
    macro_codes = get_macro_currency_options()

    default_currency = "EUR" if "EUR" in fx_codes else fx_codes[0]
    default_idx = fx_codes.index(default_currency)

    default_macro_currency = "USD" if "USD" in macro_codes else macro_codes[0]
    default_macro_idx = macro_codes.index(default_macro_currency)

    view = st.radio(
        "Valitse näkymä",
        ["📋 Yleiskuva valuutoista", "📈 Kurssikehitys", "💰 Rahamäärä & makro", "🏛️ Keskuspankkien taseet", "🧠 Analyysi"],
        horizontal=True,
        label_visibility="collapsed",
        key="currency_view",
    )

    st.divider()

    if view == "📋 Yleiskuva valuutoista":
        overview = load_currency_overview_anchor(years=years)
        render_overview_tab(overview)

    elif view == "📈 Kurssikehitys":
        st.markdown("### 📈 Kurssikehitys")

        fx_currency = st.selectbox(
            "Valitse valuutta",
            fx_codes,
            index=default_idx,
            format_func=currency_format_func,
            key="currency_selected_code_fx",
        )

        render_fx_tab_fast(fx_currency=fx_currency, years=years)

    elif view == "💰 Rahamäärä & makro":
        st.markdown("### 🏦 Rahamäärä & makro")

        money_currency = st.selectbox(
            "Valitse valuutta",
            macro_codes,
            index=default_macro_idx,
            format_func=macro_currency_format_func,
            key="currency_selected_code_money",
        )

        render_money_macro_tab(
            money_currency=money_currency,
            years=years,
            load_currency_bundle=load_money_macro_bundle,
        )
    elif view == "🏛️ Keskuspankkien taseet":
        
        st.markdown("### 🏛️ Keskuspankkien taseet")

        with st.expander("Mitä keskuspankin tase tarkoittaa ja miten sitä tulkitaan?", expanded=False):
            st.write(
                "Keskuspankin tase kuvaa keskuspankin varoja ja velkoja. "
                "Taseen varoihin voi kuulua esimerkiksi valtionlainoja, pankkijärjestelmälle annettuja lainoja, "
                "valuuttavarantoja ja muita rahapolitiikan välineitä. Velkapuolella näkyvät esimerkiksi "
                "liikkeessä oleva keskuspankkiraha ja pankkien talletukset keskuspankissa."
            )
            st.write(
                "Kun keskuspankin tase kasvaa, se kertoo usein siitä, että keskuspankki lisää likviditeettiä "
                "rahoitusjärjestelmään esimerkiksi arvopaperiostoilla, lainoituksella tai valuuttainterventioilla. "
                "Kun tase supistuu, keskuspankki yleensä kiristää tai normalisoi rahapolitiikkaa."
            )
            st.write(
                "Taseita ei kannata verrata suoraan nimellistasoina eri valuuttojen välillä, koska sarjat ovat eri valuutoissa "
                "ja eri yksiköissä. Ensimmäisessä vaiheessa tärkeintä on katsoa kunkin keskuspankin oman taseen suuntaa: "
                "kasvaako vai supistuuko tase, ja kuinka nopeasti."
            )

        bundle = load_central_bank_balance_sheets(years=years)
        df = bundle["data"]

        if bundle.get("debug"):
            with st.expander("Tekninen lähdehuomautus", expanded=False):
                st.caption(bundle["debug"])

        if df is None or df.empty:
            st.warning("Keskuspankkien tasedataa ei saatu.")
            return

        latest = (
            df.sort_values("Date")
            .groupby("CentralBank", as_index=False)
            .tail(1)
            .sort_values("CentralBank")
        )

        cols = st.columns(min(4, len(latest)))

        for col, (_, row) in zip(cols, latest.iterrows()):
            with col:
                st.metric(
                    row["Name"],
                    f"{row['Assets']:,.0f}",
                    f"{row['Assets_Change_1Y_Pct']:+.1f} % (1v)"
                    if pd.notna(row["Assets_Change_1Y_Pct"])
                    else None,
                )
                st.caption(f"{row['Unit']} | {pd.to_datetime(row['Date']).date()}")

        st.divider()

        fig = px.line(
            df,
            x="Date",
            y="Assets",
            color="CentralBank",
            title="Keskuspankkien taseet",
            labels={"Date": "Päivä", "Assets": "Tase", "CentralBank": "Keskuspankki"},
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Huomautus yksiköistä", expanded=False):
            st.write(
                "Sarjat ovat alkuperäisissä valuutoissaan ja eri yksiköissä. "
                "Fed on miljoonina dollareina, EKP miljoonina euroina ja BOJ 100 miljoonina jeneinä. "
                "Siksi tasoja ei pidä vielä verrata suoraan toisiinsa; tässä vaiheessa tärkeintä on kunkin keskuspankin oman taseen suunta."
            )

    elif view == "🧠 Analyysi":

        render_currency_analysis_tab(
            years=years,
            load_currency_bundle=load_money_macro_bundle,
            load_fx_series=load_fx_series,
        )