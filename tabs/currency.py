# tabs/currency.py
from __future__ import annotations

import pandas as pd
import streamlit as st

from services.currency_data import (
    CURRENCY_META,
    fetch_currency_bundle,
)
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
    render_fx_tab,
    render_money_macro_tab,
    render_overview_tab,
)


@st.cache_data(show_spinner="Haetaan valuutan tarkemmat tiedot…")
def load_currency_bundle(currency: str, years: int = 10) -> dict:
    return fetch_currency_bundle(currency, years=years)


@st.cache_data(show_spinner="Rakennetaan USD-ankkurinen valuuttayhteenveto…")
def load_currency_overview_anchor(years: int = 10) -> pd.DataFrame:
    anchor_bundle = fetch_currency_bundle(ANCHOR_CURRENCY, years=years)
    anchor_fx = anchor_bundle["fx"]

    rows = []

    for code, meta in CURRENCY_META.items():
        if code == ANCHOR_CURRENCY:
            continue

        bundle = fetch_currency_bundle(code, years=years)
        fx = bundle["fx"]

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


def render() -> None:
    st.subheader("💱 Valuuttakurssit")
    st.caption(
        "Näkymä käyttää **USD:tä ankkurivaluuttana**. "
        "Kurssi tarkoittaa, kuinka monta yksikköä kyseistä valuuttaa saa yhdellä Yhdysvaltain dollarilla. "
        "Positiivinen muutos tarkoittaa, että yhtä USD:tä kohden saa enemmän kyseistä valuuttaa kuin aiemmin. "
        f"{ANCHOR_CURRENCY} itse on jätetty pois FX-vertailusta, koska se olisi ankkurina aina 1.0."
    )

    with st.sidebar:
        #st.markdown("### 💱 Valuuttatabi")
        years = 10
        #years = st.slider("Historiapituus (vuotta)", 3, 15, 10, key="currency_years")

    fx_codes = [c for c in CURRENCY_META.keys() if c != ANCHOR_CURRENCY]
    macro_codes = get_macro_currency_options()

    default_currency = "EUR" if "EUR" in fx_codes else fx_codes[0]
    default_idx = fx_codes.index(default_currency)

    default_macro_currency = "USD" if "USD" in macro_codes else macro_codes[0]
    default_macro_idx = macro_codes.index(default_macro_currency)

    overview = load_currency_overview_anchor(years=years)

    
    t1, t2, t3 = st.tabs([
        "📋 Yleiskuva valuutoista",
        "📈 Kurssikehitys",
        "🏦 Rahamäärä & makro",
        
    ])

    with t1:
        render_overview_tab(overview)

    with t2:
        st.markdown("### 📈 Kurssikehitys")

        fx_currency = st.selectbox(
            "Valitse valuutta",
            fx_codes,
            index=default_idx,
            format_func=currency_format_func,
            key="currency_selected_code_fx",
        )

        render_fx_tab(
            fx_currency=fx_currency,
            years=years,
            load_currency_bundle=load_currency_bundle,
        )

    with t3:
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
            load_currency_bundle=load_currency_bundle,
        )
        

    