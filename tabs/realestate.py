from __future__ import annotations

import streamlit as st

from services.construction_charts import (
    render_construction_area,
    render_construction_koko_maa,
    render_construction_leading_indicator,
)
from services.construction_pxweb import (
    add_construction_features,
    clean_construction_df,
    fetch_construction_data,
    filter_last_n_years,
)
from services.realestate_charts import (
    render_asunnot_counts_chart,
    render_asunnot_prices_chart,
    render_pelto_alueet_chart,
    render_pelto_koko_maa_chart,
    render_tontti_kauppamaara_chart,
    render_tontti_nelio_area_comparison_chart,
    render_tontti_selected_area_index_chart,
    render_tontti_selected_area_nelio_chart,
)
from services.realestate_helpers import aggregate_trade_counts, calc_quarterly_yoy, latest_and_yoy, latest_value
from services.realestate_pxweb import (
    add_yoy_change_quarterly,
    add_yoy_change_yearly,
    clean_detached_plot_df,
    clean_realestate_df,
    fetch_detached_plot_data,
    fetch_farmland_prices,
    fetch_realestate_counts,
    fetch_realestate_prices,
)


def _render_asunnot_tab() -> None:
    st.subheader("🏠 Asuntokaupat Suomessa")
    st.caption("Lähde: Tilastokeskus / PXWeb")

    try:
        df_counts = add_yoy_change_quarterly(clean_realestate_df(fetch_realestate_counts()))
        df_prices = add_yoy_change_quarterly(clean_realestate_df(fetch_realestate_prices()))
    except Exception as e:
        st.error(f"Asuntodata ei latautunut: {e}")
        return

    c1, c2 = st.columns(2)

    latest_counts, yoy_counts = latest_and_yoy(df_counts)
    with c1:
        st.metric(
            "Asuntokauppojen lukumäärä",
            f"{latest_counts:,.0f}".replace(",", " ") if latest_counts is not None else "—",
            f"{yoy_counts:+.1f} % (1v)" if yoy_counts is not None else None,
        )

    latest_prices, yoy_prices = latest_and_yoy(df_prices)
    with c2:
        st.metric(
            "Uusien asuntojen neliöhinta",
            f"{latest_prices:,.0f} €/m²".replace(",", " ") if latest_prices is not None else "—",
            f"{yoy_prices:+.1f} % (1v)" if yoy_prices is not None else None,
        )

    st.subheader("📊 Asuntokauppojen lukumäärä")
    render_asunnot_counts_chart(df_counts)

    st.subheader("💶 Uusien asuntojen neliöhinta")
    render_asunnot_prices_chart(df_prices)


def _render_pelto_koko_maa(koko_maa_df, series_label: str) -> None:
    st.subheader(f"📈 Peltomaan {series_label} – koko maa")

    if koko_maa_df.empty:
        st.info("Koko maan sarjaa ei löytynyt datasta.")
        return

    latest_val, latest_yoy = latest_and_yoy(koko_maa_df)
    st.metric(
        f"Peltomaan {series_label}",
        f"{latest_val:,.0f} €/ha".replace(",", " ") if latest_val is not None else "—",
        f"{latest_yoy:+.1f} % (1v)" if latest_yoy is not None else None,
    )

    render_pelto_koko_maa_chart(koko_maa_df, series_label)


def _render_pelto_alueet(alue_df, series_label: str) -> None:
    st.subheader(f"🗺️ Peltomaan {series_label} – alueittain")

    if alue_df.empty:
        st.info("Alueellista sarjaa ei löytynyt datasta.")
        return

    render_pelto_alueet_chart(alue_df, series_label)


def _render_peltomaa_tab() -> None:
    st.subheader("🌾 Peltomaa")
    st.caption("Lähde: Luke / PxWeb")

    selected = st.radio(
        "Valitse seurattava sarja",
        options=["Kauppahinta", "Vuokrahinta"],
        horizontal=True,
        key="pelto_series",
    )

    try:
        series_key = "sale" if selected == "Kauppahinta" else "rent"
        series_label = "kauppahinta" if selected == "Kauppahinta" else "vuokrahinta"

        pelto_df = fetch_farmland_prices(series_key)
        pelto_df["Alue"] = pelto_df["Alue"].astype(str).str.strip()
        pelto_df["Alue_norm"] = pelto_df["Alue"].str.lower()

        koko_maa_df = pelto_df[pelto_df["Alue_norm"].isin(["koko maa", "koko_maa"])].copy()
        alue_df = pelto_df[~pelto_df["Alue_norm"].isin(["koko maa", "koko_maa"])].copy()
        alue_df = alue_df[alue_df["Alue"].str.lower() != "ahvenanmaa"].copy()

        koko_maa_df = add_yoy_change_yearly(koko_maa_df)
        alue_df = add_yoy_change_yearly(alue_df)

        _render_pelto_koko_maa(koko_maa_df, series_label)
        _render_pelto_alueet(alue_df, series_label)

    except Exception as e:
        st.error(f"Peltomaadata ei latautunut: {e}")


def _metric_value_and_yoy(df, value_fmt: str = ".1f", suffix: str = "") -> tuple[str, str | None]:
    if df is None or df.empty:
        return "—", None

    d = df.sort_values("Jakso_dt").copy()
    d = calc_quarterly_yoy(d)

    latest, yoy = latest_and_yoy(d)

    value = f"{latest:{value_fmt}}{suffix}" if latest is not None else "—"
    value = value.replace(",", " ")

    delta = f"{yoy:+.1f} % (1v)" if yoy is not None else None
    return value, delta


def _latest_region_nelio_metric(alue_nelio_df, region_name: str) -> tuple[str, str | None]:
    if alue_nelio_df is None or alue_nelio_df.empty:
        return "—", None

    f = alue_nelio_df[
        alue_nelio_df["Alue"].astype(str).str.lower() == region_name.lower()
    ].copy()

    if f.empty:
        return "—", None

    return _metric_value_and_yoy(f, value_fmt=",.0f", suffix=" €/m²")


def _render_tontti_metrics(
    koko_hinta_df,
    koko_real_df,
    alue_nelio_df,
    lkm_total_df,
) -> None:
    st.markdown("#### Koko maa")
    c1, c2, c3 = st.columns(3)

    hinta_value, hinta_delta = _metric_value_and_yoy(koko_hinta_df, value_fmt=",.1f")
    real_value, real_delta = _metric_value_and_yoy(koko_real_df, value_fmt=",.1f")
    lkm_value, lkm_delta = _metric_value_and_yoy(lkm_total_df, value_fmt=",.0f")

    with c1:
        st.metric("Hintaindeksi", hinta_value, hinta_delta)
    with c2:
        st.metric("Reaalihintaindeksi", real_value, real_delta)
    with c3:
        st.metric("Kauppojen lukumäärä", lkm_value, lkm_delta)

    st.markdown("#### Neliöhinnat alueittain")
    c1, c2, c3, c4 = st.columns(4)

    regions = [
        ("Etelä-Suomi", c1),
        ("Länsi-Suomi", c2),
        ("Itä-Suomi", c3),
        ("Pohjois-Suomi", c4),
    ]

    for region, col in regions:
        value, delta = _latest_region_nelio_metric(alue_nelio_df, region)
        with col:
            st.metric(region, value, delta)

def _render_tontti_ai_summary(
    koko_hinta_df,
    koko_real_df,
    alue_nelio_df,
    lkm_total_df,
) -> None:
    hinta_value, hinta_delta = _metric_value_and_yoy(koko_hinta_df, value_fmt=",.1f")
    real_value, real_delta = _metric_value_and_yoy(koko_real_df, value_fmt=",.1f")
    lkm_value, lkm_delta = _metric_value_and_yoy(lkm_total_df, value_fmt=",.0f")

    def _delta_num(delta_text: str | None) -> float | None:
        if not delta_text:
            return None
        try:
            return float(delta_text.split("%")[0].replace("+", "").replace(",", ".").strip())
        except Exception:
            return None

    hinta_yoy = _delta_num(hinta_delta)
    real_yoy = _delta_num(real_delta)
    lkm_yoy = _delta_num(lkm_delta)

    region_rows = []
    for region in ["Etelä-Suomi", "Länsi-Suomi", "Itä-Suomi", "Pohjois-Suomi"]:
        value, delta = _latest_region_nelio_metric(alue_nelio_df, region)
        region_rows.append((region, value, _delta_num(delta)))

    rising_regions = [r for r, _, d in region_rows if d is not None and d > 0]
    falling_regions = [r for r, _, d in region_rows if d is not None and d < 0]

    if hinta_yoy is None and real_yoy is None and lkm_yoy is None:
        st.info("Tonttimarkkinan tulkintaa ei voitu muodostaa puuttuvien muutostietojen vuoksi.")
        return

    parts = []

    if hinta_yoy is not None and real_yoy is not None:
        if hinta_yoy > 0 and real_yoy > 0:
            parts.append("Omakotitalotonttien hintakehitys on koko maan tasolla noususuuntainen myös reaalisesti.")
        elif hinta_yoy > 0 and real_yoy <= 0:
            parts.append("Nimellinen hintaindeksi on noussut, mutta reaalihinta ei ole vahvistunut samalla tavalla.")
        elif hinta_yoy < 0 and real_yoy < 0:
            parts.append("Tonttien hinnat ovat heikentyneet sekä nimellisesti että reaalisesti.")
        else:
            parts.append("Hintakehitys on kaksijakoista: nimellinen ja reaalinen kehitys antavat eri suuntaista viestiä.")

    if lkm_yoy is not None:
        if lkm_yoy > 10:
            parts.append("Kauppamäärä on selvästi vuoden takaista korkeampi, mikä viittaa markkina-aktiivisuuden piristymiseen.")
        elif lkm_yoy < -10:
            parts.append("Kauppamäärä on selvästi vuoden takaista matalampi, mikä kertoo markkina-aktiivisuuden heikkenemisestä.")
        elif lkm_yoy > 0:
            parts.append("Kauppamäärä on hieman vuoden takaista korkeampi.")
        elif lkm_yoy < 0:
            parts.append("Kauppamäärä on hieman vuoden takaista matalampi.")

    if rising_regions and falling_regions:
        parts.append(
            "Alueellinen kehitys on eriytynyttä: neliöhinnat nousevat alueilla "
            f"{', '.join(rising_regions)}, mutta laskevat alueilla {', '.join(falling_regions)}."
        )
    elif rising_regions:
        parts.append(f"Neliöhinnat ovat nousussa kaikilla saatavilla olevilla pääalueilla: {', '.join(rising_regions)}.")
    elif falling_regions:
        parts.append(f"Neliöhinnat ovat laskussa kaikilla saatavilla olevilla pääalueilla: {', '.join(falling_regions)}.")

    st.info(" ".join(parts))

def _render_tontit_tab() -> None:
    st.subheader("🏡 Omakotitalotontit")
    st.caption("Lähde: Tilastokeskus / PXWeb")

    try:
        tontti_df = clean_detached_plot_df(fetch_detached_plot_data())

        if tontti_df.empty:
            st.warning("Omakotitalotonttien dataa ei saatu.")
            return

        hinta_df = tontti_df[tontti_df["Tiedot"] == "Hintaindeksi"].copy()
        real_df = tontti_df[tontti_df["Tiedot"] == "Reaalihintaindeksi"].copy()
        nelio_df = tontti_df[tontti_df["Tiedot"] == "Neliöhinta"].copy()
        lkm_df = tontti_df[tontti_df["Tiedot"] == "Kauppojen lukumäärä"].copy()

        koko_hinta_df = hinta_df[hinta_df["Alue"] == "Koko maa"].copy()
        koko_real_df = real_df[real_df["Alue"] == "Koko maa"].copy()
        koko_nelio_df = nelio_df[nelio_df["Alue"] == "Koko maa"].copy()

        alue_hinta_df = hinta_df[hinta_df["Alue"] != "Koko maa"].copy()
        alue_real_df = real_df[real_df["Alue"] != "Koko maa"].copy()
        alue_nelio_df = nelio_df[nelio_df["Alue"] != "Koko maa"].copy()

        lkm_total_df = aggregate_trade_counts(lkm_df)

        st.markdown("### 📌 Yhteenveto")
        _render_tontti_metrics(koko_hinta_df, koko_real_df, alue_nelio_df, lkm_total_df)
        _render_tontti_ai_summary(koko_hinta_df, koko_real_df, alue_nelio_df, lkm_total_df)

        st.markdown("### 📈 Hintakehitys")
        render_tontti_selected_area_index_chart(
            koko_hinta_df,
            koko_real_df,
            alue_hinta_df,
            alue_real_df,
        )

        st.markdown("### 💶 Hintataso")
        render_tontti_selected_area_nelio_chart(
            koko_nelio_df,
            alue_nelio_df,
        )

        st.markdown("### 🏷️ Markkina-aktiivisuus")
        render_tontti_kauppamaara_chart(lkm_df)

        with st.expander("🗺️ Aluevertailu: neliöhinnat", expanded=False):
            render_tontti_nelio_area_comparison_chart(alue_nelio_df)

    except Exception as e:
        st.error(f"Omakotitalotonttidata ei latautunut: {e}")


def _render_construction_tab() -> None:
    st.subheader("🏗️ Rakentaminen")
    st.caption("Lähde: Tilastokeskus / PXWeb")

    try:
        df = fetch_construction_data()
        df = clean_construction_df(df)
        df = add_construction_features(df)
        df = filter_last_n_years(df, years=10)

        if df.empty:
            st.warning("Rakentamisen dataa ei saatu.")
            return

        render_construction_leading_indicator(df)
        render_construction_koko_maa(df)
        render_construction_area(df)

    except Exception as e:
        st.error(f"Rakentamisdata ei latautunut: {e}")


def render():
    st.subheader("🏡 Kiinteistöt ja rakentaminen")
    
    tab_asunnot, tab_pelto, tab_tontit, tab_rakentaminen = st.tabs(
        ["🏠 Asunnot", "🌾 Peltomaa", "🏡 Omakotitalotontit", "🏗️ Rakentaminen"]
    )

    with tab_asunnot:
        _render_asunnot_tab()

    with tab_pelto:
        _render_peltomaa_tab()

    with tab_tontit:
        _render_tontit_tab()

    with tab_rakentaminen:
        _render_construction_tab()