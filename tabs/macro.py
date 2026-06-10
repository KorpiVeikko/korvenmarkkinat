from __future__ import annotations

import streamlit as st

from services.macro_data import (
    load_debt_mio_eur,
    load_debt_pct_gdp,
    load_euribor_12m,
    load_gdp_yoy,
    load_household_debt_pct_gdi,
    load_household_debt_pct_gdp,
    load_household_loans_mio,
    load_inflation,
    load_nfc_debt_pct_gdp,
    load_nfc_loans_mio_nac,
    load_private_sector_debt_pct_gdp,
    load_trade_totals,
    load_unemployment,
    load_wages,
    build_trade_balance,
    load_gdp_demand_components_yoy,
)
from services.macro_view_helpers import (
    EXPORT_CFG,
    IMPORT_CFG,
    render_debt_section,
    render_gdp_section,
    render_inflation_pressure_section,
    render_inflation_section,
    render_private_debt_section,
    render_trade_balance_section,
    render_trade_flow_section,
    render_unemployment_section,
    render_wages_section,
    render_interest_section,
)
from services.macro_analysis_view import render_macro_analysis
from services.macro_inflation_pressure import load_inflation_pressure_bundle


YEARS = 8
MONTHS = YEARS * 12


def render() -> None:
    st.subheader("🇫🇮 Makrotalous")
    st.caption(
        "Lähteet: Tilastokeskus / StatFin, Tulli / Uljas, Eurostat ja ECB. "
        "Sisältö: inflaatio, BKT, työttömyys, palkat, vienti, tuonti, kauppatase, "
        "julkinen velka, yksityinen velka ja korot."
    )

    section_options = [
        "📈 Inflaatio",
        "🏛️ BKT",
        "🧑‍💼 Työttömyys",
        "💶 Palkat",
        "🚢 Vienti",
        "📥 Tuonti",
        "⚖️ Kauppatase",
        "🏦 Velka",
        "💳 Korot",
        "🧠 Analyysi",
    ]

    section = st.radio(
        "Valitse makronäkymä",
        section_options,
        horizontal=True,
        key="macro_section",
    )

    st.divider()

    if section == "📈 Inflaatio":
        try:
            pressure_bundle = load_inflation_pressure_bundle()
            render_inflation_pressure_section(pressure_bundle)
        except Exception as e:
            st.warning("Arjen inflaatiopainetta ei saatu ladattua.")
            with st.expander("Tekninen virhe"):
                st.code(repr(e))

        with st.expander("📊 Näytä vanha virallinen inflaatiokuvaaja", expanded=False):
            try:
                infl = load_inflation()
                render_inflation_section(infl, YEARS)
            except Exception as e:
                st.warning("Virallista inflaatiokuvaajaa ei saatu ladattua.")
                with st.expander("Tekninen virhe"):
                    st.code(repr(e))

    elif section == "🏛️ BKT":
        gdp_yoy = load_gdp_yoy()
        gdp_components = load_gdp_demand_components_yoy()
        render_gdp_section(gdp_yoy, YEARS, gdp_components)

    elif section == "🧑‍💼 Työttömyys":
        unemp = load_unemployment()
        render_unemployment_section(unemp, YEARS)

    elif section == "💶 Palkat":
        wages = load_wages()
        render_wages_section(wages, YEARS)

    elif section == "🚢 Vienti":
        render_trade_flow_section(EXPORT_CFG, MONTHS)

    elif section == "📥 Tuonti":
        render_trade_flow_section(IMPORT_CFG, MONTHS)

    elif section == "⚖️ Kauppatase":
        exports_total_df, imports_total_df = load_trade_totals(MONTHS)
        render_trade_balance_section(exports_total_df, imports_total_df, YEARS)

    elif section == "🏦 Velka":
        debt_section = st.radio(
            "Valitse velkanäkymä",
            ["🏦 Julkinen velka", "🏠 Yksityinen velka"],
            horizontal=True,
            key="macro_debt_section",
        )

        if debt_section == "🏦 Julkinen velka":
            debt_pct = load_debt_pct_gdp()
            debt_eur = load_debt_mio_eur()
            render_debt_section(debt_pct, debt_eur, YEARS)

        else:
            hh_debt_pct_gdp = load_household_debt_pct_gdp()
            hh_debt_pct_gdi = load_household_debt_pct_gdi()
            nfc_debt_pct_gdp = load_nfc_debt_pct_gdp()
            private_debt_pct_gdp = load_private_sector_debt_pct_gdp()

            hh_loans_mio, _ = load_household_loans_mio()
            nfc_loans_mio, _ = load_nfc_loans_mio_nac()

            render_private_debt_section(
                household_pct_gdp=hh_debt_pct_gdp,
                household_pct_gdi=hh_debt_pct_gdi,
                nfc_pct_gdp=nfc_debt_pct_gdp,
                private_pct_gdp=private_debt_pct_gdp,
                household_loans_mio=hh_loans_mio,
                nfc_loans_mio=nfc_loans_mio,
                household_loans_debug=None,
                nfc_loans_debug=None,
                years=YEARS,
            )

    elif section == "💳 Korot":
        euribor_12m, _ = load_euribor_12m()
        infl = load_inflation()
        render_interest_section(euribor_12m, infl, YEARS)

    elif section == "🧠 Analyysi":
        infl = load_inflation()
        gdp_yoy = load_gdp_yoy()
        unemp = load_unemployment()
        wages = load_wages()

        debt_pct = load_debt_pct_gdp()
        hh_debt_pct_gdi = load_household_debt_pct_gdi()
        euribor_12m, _ = load_euribor_12m()

        exports_total_df, imports_total_df = load_trade_totals(MONTHS)
        trade_balance_df = build_trade_balance(
            exports_total_df,
            imports_total_df,
        )

        render_macro_analysis(
            inflation_df=infl,
            gdp_df=gdp_yoy,
            unemployment_df=unemp,
            wages_df=wages,
            debt_df=debt_pct,
            trade_balance_df=trade_balance_df,
            household_debt_df=hh_debt_pct_gdi,
            interest_df=euribor_12m,
        )