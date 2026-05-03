from __future__ import annotations

import streamlit as st
import pandas as pd

from services.forest_data import (
    fetch_wood_prices,
    prepare_wood_prices_df,
    fetch_industrial_wood_trade,
    prepare_industrial_wood_trade_df,
    fetch_harvests,
    prepare_harvests_df,
    fetch_wood_use,
    prepare_wood_use_df,
)
from services.forest_view_helpers import (
    render_wood_prices_section,
    render_industrial_wood_trade_section,
    render_harvests_section,
    render_wood_use_section,
)


@st.cache_data(show_spinner="Haetaan metsätalouden aineistoja…")
def load_forest_bundle() -> dict[str, pd.DataFrame]:
    wood_raw = fetch_wood_prices()
    wood_df = prepare_wood_prices_df(wood_raw)

    industrial_raw = fetch_industrial_wood_trade()
    industrial_df = prepare_industrial_wood_trade_df(industrial_raw)

    harvest_raw = fetch_harvests()
    harvest_df = prepare_harvests_df(harvest_raw)

    use_raw = fetch_wood_use()
    use_df = prepare_wood_use_df(use_raw)

    return {
        "wood_raw": wood_raw,
        "wood_df": wood_df,
        "industrial_raw": industrial_raw,
        "industrial_df": industrial_df,
        "harvest_raw": harvest_raw,
        "harvest_df": harvest_df,
        "use_raw": use_raw,
        "use_df": use_df,
    }


def render() -> None:
    st.subheader("🌲 Metsätalous")
    st.caption("Lähteet: Luke / PXWeb – puun hinnat, teollinen puukauppa, hakkuut ja puun käyttö")

    try:
        bundle = load_forest_bundle()
    except Exception as e:
        st.error(f"Metsätalousdatan haku epäonnistui: {e}")
        return

    prices_tab, industrial_tab, harvests_tab, use_tab = st.tabs(
        [
            "🪵 Puun hinnat",
            "🏭 Teollinen puukauppa",
            "🪓 Hakkuut",
            "🏗️ Puun käyttö",
        ]
    )


    with prices_tab:
        render_wood_prices_section(bundle["wood_df"])

    with industrial_tab:
        render_industrial_wood_trade_section(bundle["industrial_df"])

    with harvests_tab:
        render_harvests_section(bundle["harvest_df"])

    with use_tab:
        render_wood_use_section(bundle["use_df"])
