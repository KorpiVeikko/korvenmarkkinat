# tabs/energy.py
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from services.energy_pxweb import (
    fetch_electricity_production_consumption,
    fetch_household_electricity_prices,
)
from services.energy_view_helper import (
    agg_for_view,
    build_summary_cards,
    clip_years,
    default_years_back,
    find_series_col,
    fmt_delta,
    get_best_xcol,
    keep_full_years,
    keep_monthly_actual_rows,
    latest_val,
    pct_change,
    pick_consumption_series,
    pick_import_country_series,
    pick_net_import_series,
    pick_total_production_series,
    unit_to_twh,
)
from services.energy_price_helper import (
    build_component_timeseries,
    build_price_summary,
    find_component_col,
    find_consumer_col,
    find_measure_col,
    get_consumer_options,
)
from services.energy_correl import (
    add_change_features,
    build_electricity_features,
    build_price_series,
    merge_price_and_features,
    pairwise_corr_to_target,
    rolling_corr,
)
from services.energy_spot import build_spot_summary, fetch_fi_day_ahead_spot

SHOW_DEBUG_DETAILS = False
SHOW_VERTAILUN_DATA = False

@st.cache_data(show_spinner="Haetaan sähködataa…")
def load_electricity() -> pd.DataFrame:
    return fetch_electricity_production_consumption(start_year=2020, end_year=2025)


@st.cache_data(show_spinner="Haetaan kotitaloussähkön hintadataa…")
def load_household_electricity_prices() -> pd.DataFrame:
    return fetch_household_electricity_prices()

@st.cache_data(show_spinner="Haetaan spot-hintaa…", ttl=900)
def load_spot_prices() -> pd.DataFrame:
    return fetch_fi_day_ahead_spot(hours_back=24 * 14, hours_forward=24)

def direction_text(value: float) -> str:
    if value > 0:
        return "liikkuu samaan suuntaan hinnan kanssa"
    elif value < 0:
        return "liikkuu vastakkaiseen suuntaan hintaan"
    return "ei selkeää suuntaa"


def _corr_text(value: float) -> str:
    v = abs(value)
    if v > 0.7:
        return "vahva"
    elif v > 0.4:
        return "kohtalainen"
    elif v > 0.2:
        return "heikko"
    return "erittäin heikko"

def _render_metric_row(
    *,
    metric_label: str,
    metric_value: str,
    metric_delta: str | None = None,
    caption_left: str | None = None,
    info_lines: list[str] | None = None,
    info_caption: str | None = None,
    left_ratio: int = 1,
    right_ratio: int = 2,
) -> None:
    c1, c2 = st.columns([left_ratio, right_ratio])

    with c1:
        st.metric(metric_label, metric_value, metric_delta)
        if caption_left:
            st.caption(caption_left)

    with c2:
        if info_lines:
            for line in info_lines:
                st.write(line)
        if info_caption:
            st.caption(info_caption)


def _metric_cards(cards: list[dict], columns: int = 5):
    cols = st.columns(columns)
    for i, card in enumerate(cards):
        with cols[i % columns]:
            st.metric(card["label"], card["value"], card.get("delta"))
            year_caption = card.get("year_caption")
            if year_caption:
                st.caption(year_caption)


def _add_zero_line(fig, df: pd.DataFrame):
    if not df.empty and pd.to_numeric(df["Arvo"], errors="coerce").min() < 0:
        fig.add_hline(y=0, line_width=2, line_color="red")
    return fig


def _render_single_series(
    df: pd.DataFrame,
    xcol: str,
    title: str,
    y_label: str,
    time_mode: str,
    add_zero_line: bool = False,
):
    f = df.dropna(subset=["Arvo"]).copy()
    if f.empty:
        st.info("Ei dataa valinnoilla.")
        return

    f = f.sort_values(xcol)

    if time_mode == "Vuosi (summa)":
        fig = px.line(
            f,
            x=xcol,
            y="Arvo",
            title=title,
            labels={xcol: "Aika", "Arvo": y_label},
            markers=True,
        )
        if xcol == "Aika":
            fig.update_xaxes(type="category")
        fig.update_traces(hovertemplate="%{y:.1f}<extra></extra>")
    else:
        fig = px.bar(
            f,
            x=xcol,
            y="Arvo",
            title=title,
            labels={xcol: "Aika", "Arvo": y_label},
        )
        fig.update_traces(hovertemplate="%{y:.1f}<extra></extra>")

    if add_zero_line:
        _add_zero_line(fig, f)

    st.plotly_chart(fig, use_container_width=True)


def _render_multi_series(
    df: pd.DataFrame,
    xcol: str,
    series_col: str,
    title: str,
    y_label: str,
    time_mode: str,
    add_zero_line: bool = False,
    month_barmode: str = "relative",
):
    f = df.dropna(subset=["Arvo"]).copy()
    if f.empty:
        st.info("Ei dataa valinnoilla.")
        return

    f = f.sort_values(xcol)

    if time_mode == "Vuosi (summa)":
        fig = px.line(
            f,
            x=xcol,
            y="Arvo",
            color=series_col,
            title=title,
            labels={xcol: "Aika", "Arvo": y_label, series_col: "Sarja"},
            markers=True,
        )
        if xcol == "Aika":
            fig.update_xaxes(type="category")
        fig.update_traces(hovertemplate="%{fullData.name}: %{y:.1f}<extra></extra>")
    else:
        fig = px.bar(
            f,
            x=xcol,
            y="Arvo",
            color=series_col,
            barmode=month_barmode,
            title=title,
            labels={xcol: "Aika", "Arvo": y_label, series_col: "Sarja"},
        )
        fig.update_traces(hovertemplate="%{fullData.name}: %{y:.1f}<extra></extra>")

    if add_zero_line:
        _add_zero_line(fig, f)

    st.plotly_chart(fig, use_container_width=True)


def _preferred_production_series(all_series: list[str]) -> list[str]:
    wanted_prefixes = ["1.1", "1.2", "1.3", "1.4", "1.5", "1.6"]
    excluded_prefixes = ["1.5.1", "1.5.2", "1.5.3"]

    out: list[str] = []
    for s in all_series:
        su = str(s).strip().upper()

        if su in ("1", "1 SÄHKÖN TUOTANTO"):
            continue

        if any(su.startswith(prefix.upper()) for prefix in excluded_prefixes):
            continue

        if any(su.startswith(prefix.upper()) for prefix in wanted_prefixes):
            out.append(s)

    seen = set()
    unique_out = []
    for x in out:
        if x not in seen:
            unique_out.append(x)
            seen.add(x)
    return unique_out



def _render_consumption(base: pd.DataFrame, series_col: str, time_mode: str):
    st.markdown("### ⚡ Sähkön kokonaiskulutus ja kokonaistuotanto")

    all_series = sorted(base[series_col].dropna().astype(str).unique().tolist())

    consumption_name = pick_consumption_series(all_series)
    production_name = pick_total_production_series(all_series)

    if not consumption_name:
        st.info("Kokonaiskulutuksen sarjaa ei löytynyt.")
        return

    cons = base[base[series_col].astype(str) == str(consumption_name)].copy()
    cons = agg_for_view(cons, series_col, time_mode)

    if production_name:
        prod = base[base[series_col].astype(str) == str(production_name)].copy()
        prod = agg_for_view(prod, series_col, time_mode)
    else:
        st.info("Kokonaistuotannon sarjaa ei löytynyt.")
        return

    cons = unit_to_twh(cons)
    prod = unit_to_twh(prod)

    xcol_cons = get_best_xcol(cons, time_mode=time_mode)
    xcol_prod = get_best_xcol(prod, time_mode=time_mode)

    latest_cons = latest_val(cons)
    latest_prod = latest_val(prod)

    cons_delta = pct_change(cons, 1)
    prod_delta = pct_change(prod, 1)

    delta_label = "1 v" if time_mode == "Vuosi (summa)" else "1 kk"
    period_text = "Viimeisin vuosi" if time_mode == "Vuosi (summa)" else "Viimeisin kuukausi"
    latest_time = cons[xcol_cons].iloc[-1] if not cons.empty else None

    c1, c2 = st.columns(2)
    with c1:
        st.metric(
            "Kokonaiskulutus",
            f"{latest_cons:,.1f} TWh".replace(",", " ") if latest_cons is not None else "—",
            fmt_delta(cons_delta, delta_label),
        )
        if latest_time is not None:
            st.caption(f"{period_text}: {latest_time}")

    with c2:
        st.metric(
            "Kokonaistuotanto",
            f"{latest_prod:,.1f} TWh".replace(",", " ") if latest_prod is not None else "—",
            fmt_delta(prod_delta, delta_label),
        )
        if latest_time is not None:
            st.caption(f"{period_text}: {latest_time}")

    if time_mode == "Kuukausi":
        cons_plot = cons[[xcol_cons, "Arvo"]].copy().rename(columns={"Arvo": "Kulutus"})
        prod_plot = prod[[xcol_prod, "Arvo"]].copy().rename(columns={"Arvo": "Tuotanto"})

        if xcol_cons == xcol_prod:
            merged = pd.merge(cons_plot, prod_plot, on=xcol_cons, how="inner")
            merged = merged.rename(columns={xcol_cons: "Aika"})
        else:
            merged = pd.merge(
                cons_plot,
                prod_plot,
                left_on=xcol_cons,
                right_on=xcol_prod,
                how="inner",
            )
            merged = merged.drop(columns=[xcol_prod], errors="ignore")
            merged = merged.rename(columns={xcol_cons: "Aika"})

        merged = merged.sort_values("Aika")

        fig = px.bar(
            merged,
            x="Aika",
            y="Kulutus",
            title="Kokonaiskulutus ja kokonaistuotanto (kuukausi)",
            labels={"Aika": "Aika", "Kulutus": "Kulutus (TWh)"},
        )
        fig.update_traces(name="Kulutus", hovertemplate="Kulutus: %{y:.1f} TWh<extra></extra>")

        fig.add_scatter(
            x=merged["Aika"],
            y=merged["Tuotanto"],
            mode="lines+markers",
            name="Tuotanto",
            yaxis="y",
            hovertemplate="Tuotanto: %{y:.1f} TWh<extra></extra>",
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        cons_plot = cons[[xcol_cons, "Arvo"]].copy()
        cons_plot["Sarja"] = "Kulutus"

        prod_plot = prod[[xcol_prod, "Arvo"]].copy()
        prod_plot["Sarja"] = "Tuotanto"

        cons_plot = cons_plot.rename(columns={xcol_cons: "Aika"})
        prod_plot = prod_plot.rename(columns={xcol_prod: "Aika"})

        plot_df = pd.concat([cons_plot, prod_plot], ignore_index=True).sort_values("Aika")

        fig = px.line(
            plot_df,
            x="Aika",
            y="Arvo",
            color="Sarja",
            title="Kokonaiskulutus ja kokonaistuotanto (vuosi)",
            labels={"Aika": "Vuosi", "Arvo": "Määrä (TWh)", "Sarja": "Sarja"},
            markers=True,
        )
        fig.update_xaxes(type="category")
        fig.update_traces(hovertemplate="%{fullData.name}: %{y:.1f} TWh<extra></extra>")
        st.plotly_chart(fig, use_container_width=True)


def _render_production(base: pd.DataFrame, series_col: str, time_mode: str):
    st.markdown("### 🏭 Tuotanto menetelmittäin")

    all_series = sorted(base[series_col].dropna().astype(str).unique().tolist())
    production_series = _preferred_production_series(all_series)

    if not production_series:
        st.info("Tuotantosarjoja ei löytynyt.")
        return

    with st.expander("⚙️ Valinnat (tuotanto)", expanded=False):
        chosen = st.multiselect(
            "Näytettävät tuotantosarjat",
            options=production_series,
            default=production_series,
            key="energy_production_series",
        )

    if not chosen:
        st.info("Valitse vähintään yksi tuotantosarja.")
        return

    f = base[base[series_col].astype(str).isin([str(x) for x in chosen])].copy()
    f = agg_for_view(f, series_col, time_mode)
    f = unit_to_twh(f)

    xcol = get_best_xcol(f, time_mode=time_mode)

    _render_multi_series(
        f,
        xcol=xcol,
        series_col=series_col,
        title=f"Tuotanto menetelmittäin ({time_mode.lower()})",
        y_label="Määrä (TWh)",
        time_mode=time_mode,
        month_barmode="stack",
    )


def _render_net_import(base: pd.DataFrame, series_col: str, time_mode: str):
    st.markdown("### 🌍 Nettotuonti")

    series_name = pick_net_import_series(sorted(base[series_col].dropna().astype(str).unique().tolist()))
    if not series_name:
        st.info("Nettotuonnin sarjaa ei löytynyt.")
        return

    f = base[base[series_col].astype(str) == str(series_name)].copy()
    f = agg_for_view(f, series_col, time_mode)
    f = unit_to_twh(f)

    xcol = get_best_xcol(f, time_mode=time_mode)
    latest_delta = pct_change(f, 1)
    delta_label = "1 v" if time_mode == "Vuosi (summa)" else "1 kk"
    period_text = "Viimeisin vuosi" if time_mode == "Vuosi (summa)" else "Viimeisin kuukausi"
    latest_time = f[xcol].iloc[-1] if not f.empty else None
    latest = latest_val(f)

    _render_metric_row(
        metric_label="Viimeisin nettotuonti",
        metric_value=f"{latest:,.1f} TWh".replace(",", " ") if latest is not None else "—",
        metric_delta=fmt_delta(latest_delta, delta_label),
        caption_left=f"{period_text}: {latest_time}",
        info_caption="Punainen 0-viiva näkyy vain, jos sarjassa on negatiivisia arvoja.",
    )

    _render_single_series(
        f,
        xcol=xcol,
        title=f"Nettotuonti ({time_mode.lower()})",
        y_label="Määrä (TWh)",
        time_mode=time_mode,
        add_zero_line=True,
    )


def _render_import_countries(base: pd.DataFrame, series_col: str, time_mode: str):
    st.markdown("### 🌐 Tuonti maittain")

    all_series = sorted(base[series_col].dropna().astype(str).unique().tolist())
    country_series = pick_import_country_series(all_series)
    if not country_series:
        st.info("Maakohtaisia tuontisarjoja ei löytynyt.")
        return

    with st.expander("⚙️ Valinnat (tuontimaat)", expanded=False):
        chosen = st.multiselect(
            "Näytettävät maakohtaiset tuontisarjat",
            options=country_series,
            default=country_series[:4],
            key="energy_import_country_series",
        )

    if not chosen:
        st.info("Valitse vähintään yksi maakohtainen sarja.")
        return

    f = base[base[series_col].astype(str).isin([str(x) for x in chosen])].copy()
    f = agg_for_view(f, series_col, time_mode)
    f = unit_to_twh(f)

    xcol = get_best_xcol(f, time_mode=time_mode)

    _render_multi_series(
        f,
        xcol=xcol,
        series_col=series_col,
        title=f"Tuonti maittain ({time_mode.lower()})",
        y_label="Määrä (TWh)",
        time_mode=time_mode,
        add_zero_line=True,
        month_barmode="relative",
    )

    if time_mode == "Vuosi (summa)":
        st.caption("Vuositason viivakuva helpottaa pienten sarjojen, kuten Norjan, erottumista.")
    else:
        st.caption("Jos Norjan sarja näyttää pieneltä, syy on yleensä pieni nettovirta suhteessa Ruotsiin, ei datanlukuvika.")


def _render_electricity_system():
    st.subheader("⚡ Sähkö (tuotanto, kulutus ja tuonti)")
    st.caption("Lähde: Tilastokeskus / StatFin (PXWeb) – statfin_ehk_pxt_12su")

    with st.expander("⚙️ Yleiset valinnat (sähkö)", expanded=True):
        time_mode = st.radio(
            "Aikataso",
            ["Kuukausi", "Vuosi (summa)"],
            index=0,
            horizontal=True,
            key="energy_time_mode",
        )

    years_back = default_years_back(time_mode)

    raw = load_electricity()
    if raw.empty:
        st.warning("Sähködataa ei saatu ladattua.")
        return

    base = keep_monthly_actual_rows(raw)

    series_col = find_series_col(base)
    if not series_col:
        st.error(f"En löytänyt sarakesaraketta sähködatalle. Sarakkeet: {list(base.columns)}")
        return

    base = clip_years(base, years_back)

    if time_mode == "Vuosi (summa)":
        base = keep_full_years(base, series_col=series_col, min_months=12)

    years_in_data: list[int] = []
    if "Aika_dt" in base.columns and base["Aika_dt"].notna().any():
        years_in_data = sorted(base["Aika_dt"].dt.year.dropna().astype(int).unique().tolist())

    if time_mode == "Kuukausi":
        st.caption(
            "Näkymä käyttää automaattisesti viimeistä 5 vuoden jaksoa. "
            "Yksikkö on TWh. Yhteenvetopalkki näyttää aina viimeisimmän kokonaisen vuoden arvot."
        )
        if 2025 in years_in_data:
            st.caption("Vuosi 2025 voi olla aineistossa mukana keskeneräisenä kuukausisarjana.")
    else:
        st.caption(
            "Näkymä käyttää automaattisesti viimeistä 10 vuoden jaksoa. "
            "Yksikkö on TWh. Yhteenvetopalkki näyttää aina viimeisimmän kokonaisen vuoden arvot."
        )
        st.caption("Vuosisummassa näytetään vain täydet vuodet, joten keskeneräinen 2025 jätetään pois.")

    st.markdown("### 📌 Yhteenveto")
    cards = build_summary_cards(base, series_col)
    if cards:
        _metric_cards(cards, columns=5)

    st.divider()
    _render_consumption(base, series_col, time_mode)
    st.divider()
    _render_production(base, series_col, time_mode)
    st.divider()
    _render_net_import(base, series_col, time_mode)
    st.divider()
    _render_import_countries(base, series_col, time_mode)

    if SHOW_DEBUG_DETAILS:
        with st.expander("🔍 Raakadata (sähköjärjestelmä)"):
            st.dataframe(base.tail(300), use_container_width=True)


def _render_spot_prices():
    st.markdown("### ⚡ Spot-hinta Suomi")
    st.caption(
        "Lähde: ENTSO-E Transparency Platform. Suomen day-ahead tuntihinta. "
        "Hinta ei sisällä sähkön siirtoa, veroja eikä myyjän marginaalia."
    )

    with st.expander("⚙️ Valinnat (spot)", expanded=True):
        view = st.radio(
            "Aikajänne",
            ["48 tuntia", "7 päivää", "14 päivää"],
            index=1,
            horizontal=True,
            key="spot_view_range",
        )

    try:
        df = load_spot_prices()
    except Exception as e:
        st.error(f"Spot-hinnan haku epäonnistui: {e}")
        return

    if df.empty:
        st.info("Spot-hintadataa ei löytynyt.")
        return

    df = df.copy()
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df["Price_snt_kWh"] = pd.to_numeric(df["Price_snt_kWh"], errors="coerce")
    df = df.dropna(subset=["Time", "Price_snt_kWh"]).sort_values("Time")

    if view == "48 tuntia":
        start_time = df["Time"].max() - pd.Timedelta(hours=48)
    elif view == "7 päivää":
        start_time = df["Time"].max() - pd.Timedelta(days=7)
    else:
        start_time = df["Time"].max() - pd.Timedelta(days=14)

    plot_df = df[df["Time"] >= start_time].copy()

    now = pd.Timestamp.now(tz="Europe/Helsinki")
    realized_df = plot_df[plot_df["Time"] <= now].copy()

    if realized_df.empty:
        st.info("Toteutunutta spot-hintadataa ei löytynyt.")
        return

    summary = build_spot_summary(realized_df)

    latest = summary.get("latest")
    delta_24h = summary.get("delta_24h")

    min_price = float(plot_df["Price_snt_kWh"].min()) if not plot_df.empty else None
    max_price = float(plot_df["Price_snt_kWh"].max()) if not plot_df.empty else None
    avg_price = float(plot_df["Price_snt_kWh"].mean()) if not plot_df.empty else None

    latest_time = realized_df["Time"].iloc[-1]
    current_time = latest_time

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Nykyinen spot",
            f"{latest:,.1f} snt/kWh".replace(",", " ") if latest is not None else "—",
            fmt_delta(delta_24h, "24 h") if delta_24h is not None else None,
        )
        if latest_time is not None:
            st.caption(f"Aika: {latest_time:%d.%m.%Y %H:%M}")

    with c2:
        st.metric(
            "Jakson alin",
            f"{min_price:,.1f} snt/kWh".replace(",", " ") if min_price is not None else "—",
        )

    with c3:
        st.metric(
            "Jakson korkein",
            f"{max_price:,.1f} snt/kWh".replace(",", " ") if max_price is not None else "—",
        )

    with c4:
        st.metric(
            "Jakson keskiarvo",
            f"{avg_price:,.1f} snt/kWh".replace(",", " ") if avg_price is not None else "—",
        )

    plot_df["Keskiarvo_24h"] = (
        plot_df["Price_snt_kWh"]
        .rolling(window=24, min_periods=6)
        .mean()
    )

    fig = px.bar(
        plot_df,
        x="Time",
        y="Price_snt_kWh",
        title=f"Suomen spot-hinta ({view})",
        labels={"Time": "Aika", "Price_snt_kWh": "snt/kWh"},
    )

    current_time_plot = pd.to_datetime(current_time).tz_localize(None)

    plot_df["Time_plot"] = pd.to_datetime(plot_df["Time"]).dt.tz_localize(None)

    fig = px.bar(
        plot_df,
        x="Time_plot",
        y="Price_snt_kWh",
        title=f"Suomen spot-hinta ({view})",
        labels={"Time_plot": "Aika", "Price_snt_kWh": "snt/kWh"},
    )

    fig.add_shape(
        type="line",
        x0=current_time_plot,
        x1=current_time_plot,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="red", width=3, dash="dash"),
    )

    fig.add_annotation(
        x=current_time_plot,
        y=1,
        yref="paper",
        text="nykyhetki",
        showarrow=False,
        yanchor="bottom",
    )

    fig.add_scatter(
        x=plot_df["Time"],
        y=plot_df["Keskiarvo_24h"],
        mode="lines",
        name="24 h keskiarvo",
        hovertemplate="24 h keskiarvo: %{y:.1f} snt/kWh<extra></extra>",
    )

    fig.update_traces(
        selector=dict(type="bar"),
        name="Spot-hinta",
        hovertemplate="Spot: %{y:.1f} snt/kWh<br>%{x|%d.%m.%Y %H:%M}<extra></extra>",
    )

    fig.update_layout(
        hovermode="x unified",
        bargap=0.05,
        xaxis_title="Aika",
        yaxis_title="snt/kWh",
    )

    st.plotly_chart(fig, use_container_width=True)


def _render_household_electricity_prices():
    st.markdown("### 🔌 Kotitalouksien sähkön hinta")
    st.caption(
        "Lähde: Tilastokeskus / StatFin (13rb). Aineisto sisältää hintakomponentit: "
        "sähköenergia, verkkopalvelumaksu, verot ja kokonaishinta."
    )

    df = load_household_electricity_prices()
    if df.empty:
        st.info("Kotitaloussähkön hintadataa ei saatu ladattua.")
        return

    consumer_col = find_consumer_col(df)
    component_col = find_component_col(df)
    measure_col = find_measure_col(df)

    if not consumer_col or not component_col:
        st.error(f"En löytänyt tarvittavia hintasarakkeita. Sarakkeet: {list(df.columns)}")
        return

    consumer_options = get_consumer_options(df, consumer_col)
    if not consumer_options:
        st.info("Kuluttajaryhmiä ei löytynyt hintadatasta.")
        return

    with st.expander("⚙️ Valinnat (sähkön hinta)", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            years_option = st.radio(
                "Aikajänne",
                ["5 vuotta", "10 vuotta"],
                index=0,
                horizontal=True,
                key="hh_price_years_option",
            )

            years_back = 5 if years_option == "5 vuotta" else 10
        with c2:
            consumer = st.selectbox(
                "Kuluttajaryhmä",
                consumer_options,
                index=0,
                key="hh_price_consumer",
            )

    comp_df = build_component_timeseries(
        df,
        consumer_col=consumer_col,
        component_col=component_col,
        consumer_value=consumer,
        years_back=years_back,
        measure_col=measure_col,
    )

    if comp_df.empty:
        st.warning("Valitulla kuluttajaryhmällä ei löytynyt hintakomponentteja.")
        st.write("Hintadatan sarakkeet:", list(df.columns))
        st.write("Hintakomponentit:", sorted(df[component_col].dropna().astype(str).unique().tolist()))
        if measure_col:
            st.write("Tieto-sarakkeen arvot:", sorted(df[measure_col].dropna().astype(str).unique().tolist()))
        return

    summary = build_price_summary(comp_df)

    latest_time = summary["latest_time"]
    latest_total = summary["latest_total"]
    latest_energy = summary["latest_energy"]
    latest_transfer = summary["latest_transfer"]
    latest_tax = summary["latest_tax"]
    delta_total = summary["delta_total_1y"]
    delta_energy = summary["delta_energy_1y"]
    delta_transfer = summary["delta_transfer_1y"]
    delta_tax = summary["delta_tax_1y"]

    parts = summary["parts"]
    total_df = parts["Kokonaishinta"]
    energy_df = parts["Energia"]
    transfer_df = parts["Siirto"]
    tax_df = parts["Verot"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            "Kokonaishinta",
            f"{latest_total:,.1f} snt/kWh".replace(",", " ") if latest_total is not None else "—",
            fmt_delta(delta_total, "1 v"),
        )
        if latest_time is not None:
            st.caption(f"Viimeisin kuukausi: {latest_time}")
    with c2:
        st.metric(
            "Energia",
            f"{latest_energy:,.1f} snt/kWh".replace(",", " ") if latest_energy is not None else "—",
            fmt_delta(delta_energy, "1 v"),
        )
    with c3:
        st.metric(
            "Siirto",
            f"{latest_transfer:,.1f} snt/kWh".replace(",", " ") if latest_transfer is not None else "—",
            fmt_delta(delta_transfer, "1 v"),
        )
    with c4:
        st.metric(
            "Verot",
            f"{latest_tax:,.1f} snt/kWh".replace(",", " ") if latest_tax is not None else "—",
            fmt_delta(delta_tax, "1 v"),
        )

    total_plot = total_df.sort_values("Aika_dt").copy()
    total_plot["Trendi_12kk"] = (
        total_plot["Arvo"]
        .rolling(window=12, min_periods=3)
        .mean()
    )

    fig_total = px.line(
        total_plot,
        x="Aika_dt",
        y="Arvo",
        markers=True,
        title=f"Kotitaloussähkön kokonaishinta: {consumer}",
        labels={"Aika_dt": "Aika", "Arvo": "snt/kWh"},
    )
    fig_total.update_traces(
        name="Kokonaishinta",
        hovertemplate="Kokonaishinta: %{y:.1f} snt/kWh<extra></extra>",
    )

    fig_total.add_scatter(
        x=total_plot["Aika_dt"],
        y=total_plot["Trendi_12kk"],
        mode="lines",
        name="Trendi (12 kk ka.)",
        hovertemplate="Trendi: %{y:.1f} snt/kWh<extra></extra>",
    )

    st.plotly_chart(fig_total, use_container_width=True)

    stack_df = pd.concat([energy_df, transfer_df, tax_df], ignore_index=True).sort_values(["Aika_dt", "_Komponentti"])
    fig_stack = px.bar(
        stack_df,
        x="Aika_dt",
        y="Arvo",
        color="_Komponentti",
        barmode="stack",
        title="Sähkölaskun rakenne: energia + siirto + verot",
        labels={"Aika_dt": "Aika", "Arvo": "snt/kWh", "_Komponentti": "Komponentti"},
    )
    fig_stack.update_traces(hovertemplate="%{fullData.name}: %{y:.1f} snt/kWh<extra></extra>")
    st.plotly_chart(fig_stack, use_container_width=True)

    compare_df = pd.concat([energy_df, transfer_df, tax_df], ignore_index=True).copy()
    compare_df["Vuosi"] = compare_df["Aika_dt"].dt.year.astype(str)
    compare_yearly = (
        compare_df.groupby(["Vuosi", "_Komponentti"], as_index=False)["Arvo"]
        .mean()
        .sort_values(["Vuosi", "_Komponentti"])
    )

    fig_compare = px.line(
        compare_yearly,
        x="Vuosi",
        y="Arvo",
        color="_Komponentti",
        markers=True,
        title="Hintakomponentit erikseen (vuosikeskiarvo)",
        labels={"Vuosi": "Vuosi", "Arvo": "snt/kWh", "_Komponentti": "Komponentti"},
    )
    fig_compare.update_xaxes(type="category")
    fig_compare.update_traces(hovertemplate="%{fullData.name}: %{y:.1f} snt/kWh<extra></extra>")
    st.plotly_chart(fig_compare, use_container_width=True)


def _render_prices():
    st.subheader("💶 Sähkön hinnat")

    p1, p2 = st.tabs(["⚡ Spot-hinta", "🔌 Kotitalouksien sähkön hinta"])

    with p1:
        _render_spot_prices()

    with p2:
        _render_household_electricity_prices()


def _render_price_vs_shares():
    st.subheader("🔗 Vertailu (hinta vs osuudet)")
    st.caption(
        "Tässä osiossa verrataan sähkön hinnan kehitystä tuotantorakenteeseen ja tuonnin osuuteen. "
        "Mukana ovat tuulivoima, ydinvoima, vesivoima, aurinkovoima, yhteistuotanto sekä tuonnin osuus."
    )

    elec = load_electricity()
    if elec.empty:
        st.warning("Sähködataa ei saatu ladattua.")
        return

    elec = keep_monthly_actual_rows(elec)
    series_col = find_series_col(elec)
    if not series_col:
        st.error(f"En löytänyt sähködatan sarakesaraketta. Sarakkeet: {list(elec.columns)}")
        return

    price_df = load_household_electricity_prices()
    if price_df.empty:
        st.warning("Kotitaloussähkön hintadataa ei saatu ladattua.")
        return

    consumer_col = find_consumer_col(price_df)
    component_col = find_component_col(price_df)
    measure_col = find_measure_col(price_df)

    if not consumer_col or not component_col:
        st.error(f"En löytänyt tarvittavia hintasarakkeita. Sarakkeet: {list(price_df.columns)}")
        return

    consumer_options = get_consumer_options(price_df, consumer_col)
    if not consumer_options:
        st.info("Kuluttajaryhmiä ei löytynyt hintadatasta.")
        return

    years_back = 10

    preferred_consumer = None
    for candidate in consumer_options:
        text = str(candidate)
        if "2 500" in text and "4 999" in text:
            preferred_consumer = candidate
            break
    if preferred_consumer is None:
        preferred_consumer = consumer_options[0]

    st.caption(f"Käytetty hintasarja: {preferred_consumer}")
    st.caption("Korrelaatio lasketaan Spearmanin menetelmällä. Rullaava korrelaatio käyttää 12 kk ikkunaa.")

    method = "spearman"
    rolling_window = 12

    comp_df = build_component_timeseries(
        price_df,
        consumer_col=consumer_col,
        component_col=component_col,
        consumer_value=preferred_consumer,
        years_back=years_back,
        measure_col=measure_col,
    )

    if comp_df.empty:
        st.warning("Hintakomponentteja ei löytynyt oletuskuluttajaryhmällä.")
        return

    total_price_df = comp_df[comp_df["_Komponentti"] == "Kokonaishinta"].copy()
    price_series = build_price_series(total_price_df)

    if price_series.empty:
        st.warning("Kokonaishinnan kuukausisarjaa ei saatu muodostettua.")
        return

    elec = clip_years(elec, years_back)
    features = build_electricity_features(elec, series_col=series_col, unit="TWh")

    if features.empty:
        st.warning("Sähköfeatureiden muodostus epäonnistui.")
        return

    merged = merge_price_and_features(price_series, features)
    if merged.empty:
        st.warning("Hinnan ja sähköfeatureiden aikaleimat eivät osuneet yhteen.")
        return

    base_features = [
        "wind_share",
        "nuclear_share",
        "hydro_share",
        "solar_share",
        "chp_share",
        "import_share",
    ]
    available_base_features = [c for c in base_features if c in merged.columns]

    merged_changes = add_change_features(merged, available_base_features)

    label_map = {
        "wind_share": "Tuulivoiman osuus",
        "nuclear_share": "Ydinvoiman osuus",
        "hydro_share": "Vesivoiman osuus",
        "solar_share": "Aurinkovoiman osuus",
        "chp_share": "Yhteistuotannon osuus",
        "import_share": "Tuonnin osuus",
    }

    change_features = []
    change_labels = {}

    for c in available_base_features:
        if c.endswith("_share") and f"{c}_diff_pp" in merged_changes.columns:
            change_features.append(f"{c}_diff_pp")
            change_labels[f"{c}_diff_pp"] = f"{label_map[c]}, muutos"
        elif f"{c}_mom_pct" in merged_changes.columns:
            change_features.append(f"{c}_mom_pct")
            change_labels[f"{c}_mom_pct"] = f"{label_map[c]}, muutos"

    level_corr = pairwise_corr_to_target(
        merged,
        target="price",
        features=available_base_features,
        method=method,
    )

    change_corr = pairwise_corr_to_target(
        merged_changes,
        target="price_mom_pct",
        features=change_features,
        method=method,
    )

    

    def _build_bar_df(df: pd.DataFrame, label_lookup: dict[str, str]) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()

        out = df.copy()
        out["Tekijä"] = out["feature"].map(lambda x: label_lookup.get(x, x))
        out["Korrelaatio"] = out["corr"].round(2)
        out["Suunta"] = out["Korrelaatio"].apply(lambda x: "Positiivinen" if x >= 0 else "Negatiivinen")
        out = out.sort_values("Korrelaatio", ascending=True)
        return out

    level_bar = _build_bar_df(level_corr, label_map)
    change_bar = _build_bar_df(change_corr, change_labels)

    st.markdown("### Hinnan kanssa liikkuvat tekijät")
    st.caption("Vihreä = positiivinen yhteys, punainen = negatiivinen yhteys.")

    if level_bar.empty:
        st.info("Korrelaatiota ei saatu laskettua.")
    else:
        fig_level = px.bar(
            level_bar,
            x="Korrelaatio",
            y="Tekijä",
            orientation="h",
            color="Suunta",
            color_discrete_map={"Positiivinen": "#2ca02c", "Negatiivinen": "#d62728"},
            title="Hintatasoon liittyvät tekijät",
        )

        fig_level.add_vline(x=0, line_width=1, line_dash="dash", line_color="gray")

        fig_level.update_layout(
            showlegend=False,
            yaxis_title=None,
            xaxis_title="Korrelaatio",
            bargap=0.25,
            height=420,
            margin=dict(l=20, r=20, t=50, b=20),
        )

        fig_level.update_traces(
            hovertemplate="%{y}: %{x:.2f}<extra></extra>",
            texttemplate="%{x:.2f}",
            textposition="outside",
            cliponaxis=False,
        )

        fig_level.update_xaxes(range=[-1, 1])

        st.plotly_chart(fig_level, use_container_width=True)

        strongest_pos = level_bar.sort_values("Korrelaatio", ascending=False).iloc[0]
        strongest_neg = level_bar.sort_values("Korrelaatio", ascending=True).iloc[0]

        st.info(
            f"Hintaan liittyy vahvimmin positiivisesti **{strongest_pos['Tekijä']}** "
            f"({strongest_pos['Korrelaatio']:.2f}, {_corr_text(strongest_pos['Korrelaatio'])} yhteys) "
            f"ja negatiivisesti **{strongest_neg['Tekijä']}** "
            f"({strongest_neg['Korrelaatio']:.2f}, {_corr_text(strongest_neg['Korrelaatio'])} yhteys)."
        )
        

    st.markdown("### Hinnan muutosten kanssa liikkuvat tekijät")
    st.caption("Näyttää, mitkä muutokset tapahtuvat samaan aikaan hinnan kuukausimuutosten kanssa.")

    if change_bar.empty:
        st.info("Hinnan muutosten korrelaatiota ei saatu laskettua.")
    else:
        fig_change = px.bar(
        change_bar,
        x="Korrelaatio",
        y="Tekijä",
        orientation="h",
        color="Suunta",
        color_discrete_map={"Positiivinen": "#2ca02c", "Negatiivinen": "#d62728"},
        title="Hinnan muutoksiin liittyvät tekijät",
        )

        fig_change.add_vline(x=0, line_width=1, line_dash="dash", line_color="gray")

        fig_change.update_layout(
            showlegend=False,
            yaxis_title=None,
            xaxis_title="Korrelaatio",
            bargap=0.25,
            height=420,
            margin=dict(l=20, r=20, t=50, b=20),
        )

        fig_change.update_traces(
            hovertemplate="%{y}: %{x:.2f}<extra></extra>",
            texttemplate="%{x:.2f}",
            textposition="outside",
            cliponaxis=False,
        )

        fig_change.update_xaxes(range=[-1, 1])

        st.plotly_chart(fig_change, use_container_width=True)

        strongest_change_abs = change_bar.reindex(change_bar["Korrelaatio"].abs().sort_values(ascending=False).index).iloc[0]

        st.info(
            f"Hinnan muutoksiin liittyy eniten **{strongest_change_abs['Tekijä']}** "
            f"({strongest_change_abs['Korrelaatio']:.2f}, "
            f"{direction_text(strongest_change_abs['Korrelaatio'])}, "
            f"{_corr_text(strongest_change_abs['Korrelaatio'])} yhteys)."
        )

    st.markdown("### Seuranta ajan yli")
    st.caption("Valitse tekijä, jonka yhteyttä sähkön hintaan haluat seurata ajan yli.")

    rolling_feature_options = [c for c in available_base_features]
    rolling_feature_labels = {c: label_map.get(c, c) for c in rolling_feature_options}

    if not rolling_feature_options:
        st.info("Rullaavaa korrelaatiota varten ei löytynyt sarjoja.")
    else:
        selected_driver = st.selectbox(
            "Seurattava tekijä",
            rolling_feature_options,
            index=0,
            format_func=lambda x: rolling_feature_labels.get(x, x),
            key="cmp_driver_simple",
        )

        rc = rolling_corr(
            merged,
            x=selected_driver,
            y="price",
            window=rolling_window,
            method=method,
            min_periods=6,
        )

        if rc.empty:
            st.info("Rullaavaa korrelaatiota ei saatu laskettua.")
        else:
            fig_roll = px.line(
                rc,
                x="Month",
                y="corr",
                markers=True,
                title=f"Rullaava korrelaatio: {rolling_feature_labels.get(selected_driver, selected_driver)} vs sähkön hinta",
                labels={"Month": "Aika", "corr": "Korrelaatio"},
            )
            fig_roll.add_hline(y=0, line_width=1, line_dash="dash")
            fig_roll.update_traces(hovertemplate="Korrelaatio: %{y:.2f}<extra></extra>")
            st.plotly_chart(fig_roll, use_container_width=True)
            
    if SHOW_VERTAILUN_DATA:
        with st.expander("🔍 Vertailun data"):
            show_cols = ["Month", "price"] + available_base_features
            extra_cols = ["price_mom_pct"] + change_features
            show_cols = [c for c in show_cols + extra_cols if c in merged_changes.columns]
            st.dataframe(merged_changes[show_cols].round(4), use_container_width=True)


def render():
    st.subheader("☢️ Energia")

    st.caption(
        "Lähteet: Tilastokeskus / StatFin (sähkön tuotanto, kulutus ja hinnat), "
        
    )
    t1, t2, t3 = st.tabs(
        ["⚡ Sähkö (tuotanto/kulutus/tuonti)", "💶 Sähkön hinnat", "🔗 Vertailu (hinta vs osuudet)"]
    )

    with t1:
        _render_electricity_system()

    with t2:
        _render_prices()

    with t3:
        _render_price_vs_shares()