from __future__ import annotations

import streamlit as st
import pandas as pd

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
    debug_pxweb_metadata, 
    ASUNTOKAUPAT_URL, 
    TONTTI_URL
)

from services.realestate_analysis import build_realestate_analysis_bundle


def _pct_color(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "#6b7280"
    return "#15803d" if value >= 0 else "#b91c1c"


def _pct_html(value: float | None, label: str = "1 v") -> str:
    if value is None or pd.isna(value):
        txt = "—"
    else:
        txt = f"{value:+.1f} % ({label})"

    return f"""
    <span style="
        color:{_pct_color(value)};
        font-weight:700;
        font-size:1.05rem;
    ">
        {txt}
    </span>
    """


def _metric_card(
    title: str,
    value: str,
    pct: float | None = None,
    caption: str | None = None,
) -> None:
    with st.container(border=True):
        st.caption(title)
        st.markdown(f"## {value}")

        if pct is not None and not pd.isna(pct):
            st.markdown(_pct_html(pct), unsafe_allow_html=True)

        if caption:
            st.caption(caption)


def _info_card(title: str, value: str, delta: str | None = None, caption: str | None = None) -> None:
    with st.container(border=True):
        st.metric(title, value, delta)
        if caption:
            st.caption(caption)


def _section_card(title: str, caption: str | None = None) -> None:
    with st.container(border=True):
        st.markdown(f"### {title}")
        if caption:
            st.caption(caption)


def _render_asunnot_tab() -> None:
    st.subheader("🏠 Asuntokaupat Suomessa")
    st.caption("Lähde: Tilastokeskus / PXWeb")

    try:
        df_counts = add_yoy_change_quarterly(clean_realestate_df(fetch_realestate_counts()))
        df_prices = add_yoy_change_quarterly(clean_realestate_df(fetch_realestate_prices()))
    except Exception as e:
        st.error(f"Asuntodata ei latautunut: {e}")
        return

    st.markdown("### 📌 Tilannekuva")

    c1, c2, c3, c4 = st.columns(4)

    latest_counts, yoy_counts = latest_and_yoy(df_counts)
    latest_prices, yoy_prices = latest_and_yoy(df_prices)

    count_status = "🟢 Vilkastuva" if (yoy_counts or 0) > 5 else "🔴 Heikko" if (yoy_counts or 0) < -5 else "🟡 Vakaa"
    price_status = "🟢 Nousussa" if (yoy_prices or 0) > 5 else "🔴 Laskussa" if (yoy_prices or 0) < -5 else "🟡 Vakaa"

    with c1:
        _metric_card(
            "Asuntokauppojen määrä",
            f"{latest_counts:,.0f}".replace(",", " ") if latest_counts is not None else "—",
            yoy_counts,
            "Kvartaali"
        )

    with c2:
        _metric_card(
            "Uusien asuntojen neliöhinta",
            f"{latest_prices:,.0f} €/m²".replace(",", " ") if latest_prices is not None else "—",
            yoy_prices,
            "Koko maa"
        )

    with c3:
        with st.container(border=True):
            st.markdown("### 🏘️ Asuntomarkkina")
            st.markdown(f"**Tila:** {count_status}")
            st.caption("Perustuu asuntokauppojen määrän vuosimuutokseen.")

    with c4:
        with st.container(border=True):
            st.markdown("### 💶 Hintakehitys")
            st.markdown(f"**Tila:** {price_status}")
            st.caption("Perustuu uusien asuntojen neliöhinnan vuosimuutokseen.")

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

        latest_val, latest_yoy = latest_and_yoy(koko_maa_df)

        latest_region_rows = (
            alue_df.sort_values("Vuosi")
            .groupby("Alue", as_index=False)
            .tail(1)
        )

        highest_region = None
        lowest_region = None

        if not latest_region_rows.empty:
            highest_region = latest_region_rows.sort_values("Arvo", ascending=False).iloc[0]
            lowest_region = latest_region_rows.sort_values("Arvo", ascending=True).iloc[0]

        st.markdown("### 📌 Tilannekuva")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            _metric_card(
                f"Peltomaan {series_label}",
                f"{latest_val:,.0f} €/ha".replace(",", " ") if latest_val is not None else "—",
                latest_yoy,
                "Koko maa"
            )

        with c2:
            if highest_region is not None:
                _metric_card(
                    "Korkein alue",
                    highest_region["Alue"],
                    None,
                    f'{highest_region["Arvo"]:,.0f} €/ha'.replace(",", " ")
                )

        with c3:
            if lowest_region is not None:
                _metric_card(
                    "Matalin alue",
                    lowest_region["Alue"],
                    None,
                    f'{lowest_region["Arvo"]:,.0f} €/ha'.replace(",", " ")
                )

        with c4:
            with st.container(border=True):
                if latest_yoy is not None:
                    if latest_yoy > 5:
                        status = "🟢 Vahva nousu"
                    elif latest_yoy < -5:
                        status = "🔴 Heikentyvä"
                    else:
                        status = "🟡 Vakaa"
                else:
                    status = "⚪ Ei dataa"

                st.markdown("### 🌾 Markkinatila")
                st.markdown(f"**Tila:** {status}")
                st.caption("Perustuu koko maan vuosimuutokseen.")

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
    st.markdown("### 📌 Tonttimarkkinan pikakuva")

    hinta_value, hinta_delta = _metric_value_and_yoy(koko_hinta_df, value_fmt=",.1f")
    real_value, real_delta = _metric_value_and_yoy(koko_real_df, value_fmt=",.1f")
    lkm_value, lkm_delta = _metric_value_and_yoy(lkm_total_df, value_fmt=",.0f")

    c1, c2, c3 = st.columns(3)

    with c1:
        _info_card(
            "Hintaindeksi",
            hinta_value,
            hinta_delta,
            "Koko maa, omakotitalotontit",
        )

    with c2:
        _info_card(
            "Reaalihintaindeksi",
            real_value,
            real_delta,
            "Inflaation huomioiva hintakehitys",
        )

    with c3:
        _info_card(
            "Kauppojen määrä",
            lkm_value,
            lkm_delta,
            "Markkina-aktiivisuus",
        )

    st.markdown("### 🗺️ Neliöhinnat alueittain")

    regions = ["Etelä-Suomi", "Länsi-Suomi", "Itä-Suomi", "Pohjois-Suomi"]
    cols = st.columns(4)

    for region, col in zip(regions, cols):
        value, delta = _latest_region_nelio_metric(alue_nelio_df, region)
        with col:
            _info_card(region, value, delta, "Viimeisin €/m²")


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

        _render_tontti_metrics(koko_hinta_df, koko_real_df, alue_nelio_df, lkm_total_df)

        st.markdown("### 🧠 Tulkinta")
        _render_tontti_ai_summary(koko_hinta_df, koko_real_df, alue_nelio_df, lkm_total_df)

        st.divider()

        _section_card(
            "📈 Hintakehitys",
            "Hintaindeksi kertoo nimellisen hintakehityksen. Reaalihintaindeksi näyttää kehityksen inflaation jälkeen.",
        )
        render_tontti_selected_area_index_chart(
            koko_hinta_df,
            koko_real_df,
            alue_hinta_df,
            alue_real_df,
        )

        st.divider()

        _section_card(
            "💶 Hintataso",
            "Neliöhinta kertoo tonttien keskimääräisen hintatason euroina per neliömetri.",
        )
        render_tontti_selected_area_nelio_chart(
            koko_nelio_df,
            alue_nelio_df,
        )

        st.divider()

        _section_card(
            "🏷️ Markkina-aktiivisuus",
            "Kauppamäärä kertoo, kuinka vilkas tonttimarkkina on ollut.",
        )
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

        with st.container(border=True):
            st.markdown("### 🧭 Rakentamisen tilannekuva")
            st.caption(
                "Rakennusluvat toimivat ennakoivana mittarina. "
                "Valmistuneet kertovat toteutuneesta tarjonnasta ja rakenteilla olevat lähiajan tuotannosta."
            )
            render_construction_leading_indicator(df)

        st.divider()

        with st.container(border=True):
            st.markdown("### 📈 Koko maan rakentaminen")
            st.caption("12 kuukauden kertymä tasoittaa kuukausittaista vaihtelua.")
            render_construction_koko_maa(df)

        st.divider()

        with st.container(border=True):
            st.markdown("### 🗺️ Alueellinen rakentaminen")
            st.caption("Valitse maakunta ja vertaile lupia, rakenteilla olevia sekä valmistuneita asuntoja.")
            render_construction_area(df)

    except Exception as e:
        st.error(f"Rakentamisdata ei latautunut: {e}")


@st.cache_data(ttl=60 * 60 * 6, show_spinner="Rakennetaan kiinteistömarkkinan analyysiä…")
def load_realestate_analysis_bundle() -> dict:
    df_counts = add_yoy_change_quarterly(clean_realestate_df(fetch_realestate_counts()))
    df_prices = add_yoy_change_quarterly(clean_realestate_df(fetch_realestate_prices()))

    tontti_df = clean_detached_plot_df(fetch_detached_plot_data())

    construction_df = fetch_construction_data()
    construction_df = clean_construction_df(construction_df)
    construction_df = add_construction_features(construction_df)
    construction_df = filter_last_n_years(construction_df, years=10)

    return build_realestate_analysis_bundle(
        df_counts=df_counts,
        df_prices=df_prices,
        tontti_df=tontti_df,
        construction_df=construction_df,
    )


def _analysis_pct_color(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "#6b7280"
    return "#15803d" if value >= 0 else "#b91c1c"


def _analysis_pct_html(value: float | None, label: str = "1 v") -> str:
    if value is None or pd.isna(value):
        txt = "—"
    else:
        sign = "+" if value >= 0 else ""
        txt = f"{sign}{value:.1f} % ({label})"

    return f"""
    <span style="
        color:{_analysis_pct_color(value)};
        font-weight:700;
        font-size:1.05rem;
    ">
        {txt}
    </span>
    """


def _render_realestate_indicator_card(item: dict) -> None:
    with st.container(border=True):
        st.markdown(f"### {item.get('Ikoni', '⚪')} {item.get('Osa-alue', '')}")
        st.markdown(f"**Tila:** {item.get('Tila', '—')}")
        st.markdown(_analysis_pct_html(item.get("Muutos")), unsafe_allow_html=True)
        st.caption(item.get("Selite", ""))


def _render_realestate_analysis_tab() -> None:
    st.subheader("🧠 Kiinteistömarkkinan analyysi")
    st.caption(
        "Yhdistää asuntojen, tonttien ja rakentamisen datan yhdeksi helposti luettavaksi markkinakuvaksi."
    )

    try:
        bundle = load_realestate_analysis_bundle()
    except Exception as e:
        st.error(f"Kiinteistömarkkinan analyysi ei latautunut: {e}")
        return

    with st.container(border=True):
        st.markdown(f"## {bundle.get('cycle_icon', '⚪')} {bundle.get('cycle_label', 'Ei dataa')}")
        st.write(bundle.get("summary", ""))

    st.divider()

    st.markdown("### 📌 Tilaindikaattorit")

    indicators = bundle.get("indicators", [])
    if indicators:
        for i in range(0, len(indicators), 3):
            cols = st.columns(3)
            for col, item in zip(cols, indicators[i : i + 3]):
                with col:
                    _render_realestate_indicator_card(item)

    st.divider()

    c1, c2, c3 = st.columns(3)

    with c1:
        with st.container(border=True):
            st.markdown("### ✅ Vahvuudet")
            for item in bundle.get("strengths", []):
                st.write(f"• {item}")

    with c2:
        with st.container(border=True):
            st.markdown("### ⚠️ Riskit")
            for item in bundle.get("risks", []):
                st.write(f"• {item}")

    with c3:
        with st.container(border=True):
            st.markdown("### 👀 Seurattavaa")
            for item in bundle.get("watchlist", []):
                st.write(f"• {item}")

    st.divider()

    st.info(
        "Tämä analyysi ei ole sijoitus- tai ostopäätössuositus. Se kokoaa usean datalähteen yhteen: "
        "kauppamäärät kuvaavat kysyntää, hinnat markkinatasoa ja rakennusluvat tulevan tarjonnan suuntaa."
    )

def render():
    st.subheader("🏡 Kiinteistöt ja rakentaminen")

    tab_asunnot, tab_pelto, tab_tontit, tab_rakentaminen, tab_analyysi = st.tabs(
        [
            "🏠 Asunnot",
            "🌾 Peltomaa",
            "🏡 Omakotitalotontit",
            "🏗️ Rakentaminen",
            "🧠 Kiinteistömarkkinan analyysi",
        ]
    )

    with tab_asunnot:
        _render_asunnot_tab()

    with tab_pelto:
        _render_peltomaa_tab()

    with tab_tontit:
        _render_tontit_tab()

    with tab_rakentaminen:
        _render_construction_tab()

    with tab_analyysi:
        _render_realestate_analysis_tab()