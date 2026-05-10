from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from services.forest_helpers import (
    find_first_matching_column,
    latest_and_yoy,
    rolling_mean,
    rolling_sum,
)


def metric_card(label: str, value: str, delta: str | None = None, caption: str | None = None) -> None:
    with st.container(border=True):
        st.metric(label, value, delta)
        if caption:
            st.caption(caption)





def _format_million_m3_from_thousand(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value / 1000:,.1f} milj. m³".replace(",", " ")


def _latest_year_label(df: pd.DataFrame, prefix: str = "Vuoden") -> str | None:
    if df is None or df.empty or "Date" not in df.columns:
        return None
    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d = d.dropna(subset=["Date"])
    if d.empty:
        return None
    latest_year = int(d["Date"].dt.year.max())
    return f"{prefix} {latest_year}"


def _collapse_series_by_date(df: pd.DataFrame, how: str = "sum") -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Date", "Arvo"])

    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["Arvo"] = pd.to_numeric(out["Arvo"], errors="coerce")
    out = out.dropna(subset=["Date", "Arvo"])

    if out.empty:
        return pd.DataFrame(columns=["Date", "Arvo"])

    agg = "mean" if how == "mean" else "sum"
    return (
        out.groupby("Date", as_index=False)["Arvo"]
        .agg(agg)
        .sort_values("Date")
        .reset_index(drop=True)
    )


def _default_option(options: list[str], preferred_substrings: list[str], fallback_first: bool = True) -> str | None:
    if not options:
        return None

    options_str = [str(x) for x in options]
    for pref in preferred_substrings:
        pref_l = pref.lower()
        for opt in options_str:
            if pref_l in opt.lower():
                return opt

    return options_str[0] if fallback_first else None


def _sort_options_with_total_first(options: list[str], preferred_terms: list[str]) -> list[str]:
    if not options:
        return []

    options_str = [str(x) for x in options]

    preferred: list[str] = []
    others: list[str] = []

    for opt in options_str:
        opt_l = opt.lower()
        if any(term.lower() in opt_l for term in preferred_terms):
            preferred.append(opt)
        else:
            others.append(opt)

    return preferred + others


def render_wood_prices_section(df: pd.DataFrame) -> None:
    st.subheader("🪵 Puun hinnat")
    st.caption("Viikkokantohinnat Luke / PXWeb.")

    week_col = find_first_matching_column(df, ["W", "Viikko", "week"])
    area_col = find_first_matching_column(df, ["MPKH", "Alue", "hinta-alue", "hintaalue"])
    hakt_col = find_first_matching_column(df, ["HAKT", "Hakkuutapa"])
    ptl_col = find_first_matching_column(df, ["PTL", "Puutavaralaji"])

    if not all([week_col, area_col, hakt_col, ptl_col]):
        st.warning(f"Puunhintadatan sarakkeita ei tunnistettu oikein. Sarakkeet: {list(df.columns)}")
        return

    areas = sorted(df[area_col].dropna().astype(str).unique().tolist())
    hakkuutavat = sorted(df[hakt_col].dropna().astype(str).unique().tolist())
    puutavaralajit = sorted(df[ptl_col].dropna().astype(str).unique().tolist())

    # Oletusvalinnat kuvan mukaan
    default_area = _default_option(areas, ["savo-karjala"]) or (areas[0] if areas else None)
    default_hakt = _default_option(hakkuutavat, ["uudistushakkuu"]) or (hakkuutavat[0] if hakkuutavat else None)
    default_ptl = _default_option(puutavaralajit, ["kuusitukki"]) or (puutavaralajit[0] if puutavaralajit else None)

    # Tallennetaan aktiiviset valinnat session stateen vain kerran
    if "forest_prices_selected_area" not in st.session_state:
        st.session_state["forest_prices_selected_area"] = default_area
    if "forest_prices_selected_hakt" not in st.session_state:
        st.session_state["forest_prices_selected_hakt"] = default_hakt
    if "forest_prices_selected_ptl" not in st.session_state:
        st.session_state["forest_prices_selected_ptl"] = default_ptl

    with st.form("forest_prices_form"):
        st.markdown("### ⚙️ Valinnat")
        c1, c2, c3 = st.columns(3)

        chosen_area = c1.selectbox(
            "Alue",
            areas,
            index=areas.index(st.session_state["forest_prices_selected_area"]) if st.session_state["forest_prices_selected_area"] in areas else 0,
        )
        chosen_hakt = c2.selectbox(
            "Hakkuutapa",
            hakkuutavat,
            index=hakkuutavat.index(st.session_state["forest_prices_selected_hakt"]) if st.session_state["forest_prices_selected_hakt"] in hakkuutavat else 0,
        )
        chosen_ptl = c3.selectbox(
            "Puutavaralaji",
            puutavaralajit,
            index=puutavaralajit.index(st.session_state["forest_prices_selected_ptl"]) if st.session_state["forest_prices_selected_ptl"] in puutavaralajit else 0,
        )

        submitted = st.form_submit_button("Hae", use_container_width=False)

    # Päivitä aktiiviset valinnat vasta kun käyttäjä painaa Hae
    if submitted:
        st.session_state["forest_prices_selected_area"] = chosen_area
        st.session_state["forest_prices_selected_hakt"] = chosen_hakt
        st.session_state["forest_prices_selected_ptl"] = chosen_ptl

    active_area = st.session_state["forest_prices_selected_area"]
    active_hakt = st.session_state["forest_prices_selected_hakt"]
    active_ptl = st.session_state["forest_prices_selected_ptl"]

    f = df[
        (df[area_col] == active_area)
        & (df[hakt_col] == active_hakt)
        & (df[ptl_col] == active_ptl)
    ].copy()

    if f.empty:
        st.info("Valinnoilla ei löytynyt dataa.")
        return

    f = f.sort_values("Date").reset_index(drop=True)
    f = rolling_mean(f, "Arvo", window=52)

    latest, yoy = latest_and_yoy(f, "Arvo", periods=52)

    c1, c2 = st.columns(2)
    with c1:
        metric_card(
            "Viimeisin kantohinta",
            f"{latest:,.1f} €/m³".replace(",", " ") if latest is not None else "—",
            f"{yoy:+.1f} % (1 v)" if yoy is not None else None,
            f"{active_area} • {active_hakt} • {active_ptl}",
        )
    with c2:
        latest_week = f[week_col].iloc[-1] if week_col in f.columns and not f.empty else None
        metric_card(
            "Viimeisin viikko",
            str(latest_week) if latest_week is not None else "—",
            None,
            "Puunhintasarjan viimeisin havainto",
        )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=f["Date"], y=f["Arvo"], mode="lines", name="Viikkohinta"))
    if "Arvo_trend" in f.columns:
        fig.add_trace(
            go.Scatter(
                x=f["Date"],
                y=f["Arvo_trend"],
                mode="lines",
                name="52 vk trendi",
                line=dict(width=4),
            )
        )
    fig.update_layout(
        title=f"Kantohinnan kehitys – {active_ptl} / {active_area} / {active_hakt}",
        xaxis_title="Aika",
        yaxis_title="€/m³",
        hovermode="x unified",
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown("#### 🗺️ Aluevertailu viimeisimmällä viikolla")
    latest_date = f["Date"].max()
    latest_week_all = df[df["Date"] == latest_date].copy()
    comp = latest_week_all[
        (latest_week_all[hakt_col] == active_hakt)
        & (latest_week_all[ptl_col] == active_ptl)
    ].copy()

    if not comp.empty:
        comp = (
            comp[[area_col, "Arvo"]]
            .dropna(subset=[area_col])
            .groupby(area_col, as_index=False)["Arvo"]
            .mean()
            .sort_values("Arvo", ascending=False)
        )

        bar = px.bar(
            comp,
            x=area_col,
            y="Arvo",
            title=f"Aluevertailu – {active_ptl} / {active_hakt}",
            labels={area_col: "Alue", "Arvo": "€/m³"},
        )
        bar.update_xaxes(categoryorder="total descending")
        st.plotly_chart(bar, width="stretch")


def render_industrial_wood_trade_section(df: pd.DataFrame) -> None:
    st.subheader("🏭 Teollinen puukauppa")
    st.caption("Teollisuuspuun määrät kuukausitasolla.")

    if df is None or df.empty:
        st.info("Teollisen puukaupan dataa ei löytynyt.")
        return

    info_col = find_first_matching_column(df, ["Tieto", "Tiedot"])
    area_col = find_first_matching_column(df, ["Hinta-alue", "Alue"])

    f = df.copy()

    # Käytetään vain määrä-sarjaa
    if info_col:
        infos = sorted(f[info_col].dropna().astype(str).unique().tolist())
        chosen_info = None

        preferred = [x for x in infos if "määrä" in x.lower()]
        if preferred:
            chosen_info = preferred[0]
        elif infos:
            chosen_info = infos[0]

        if chosen_info is not None:
            f = f[f[info_col].astype(str) == str(chosen_info)].copy()
    else:
        chosen_info = "Määrä (1000 m³)"

    # Aluevalitsin jätetään
    if area_col:
        areas = sorted(f[area_col].dropna().astype(str).unique().tolist())

        koko_maa = [x for x in areas if "koko maa" in str(x).lower()]
        muut = [x for x in areas if "koko maa" not in str(x).lower()]
        areas = koko_maa + muut
        default_area = next((x for x in areas if "koko maa" in str(x).lower()), areas[0])

        with st.expander("⚙️ Valinnat", expanded=True):
            chosen_area = st.selectbox(
                "Hinta-alue",
                areas,
                index=areas.index(default_area),
                key="forest_industrial_area",
            )

        f = f[f[area_col].astype(str) == str(chosen_area)].copy()
    else:
        chosen_area = ""

    if f.empty:
        st.info("Valinnoilla ei löytynyt teollisen puukaupan dataa.")
        return

    # Määräsarja summataan kuukausittain
    monthly_df = _collapse_series_by_date(f, how="sum")
    monthly_df = rolling_sum(monthly_df, "Arvo", window=12)

    latest, yoy = latest_and_yoy(monthly_df, "Arvo", periods=12)

    plot_df = monthly_df.copy()
    plot_df["Arvo_plot"] = plot_df["Arvo"] / 1000
    plot_df["Arvo_12kk_plot"] = (
        plot_df["Arvo_12kk"] / 1000 if "Arvo_12kk" in plot_df.columns else pd.Series(dtype=float)
    )

    metric_card(
        "Viimeisin kuukausihavainto",
        _format_million_m3_from_thousand(latest),
        f"{yoy:+.1f} % (1 v)" if yoy is not None else None,
        f"{chosen_info} • {chosen_area}".strip(" •"),
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_df["Date"],
            y=plot_df["Arvo_plot"],
            mode="lines",
            name="Kuukausisarja",
            yaxis="y1",
        )
    )

    if plot_df["Arvo_12kk_plot"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=plot_df["Date"],
                y=plot_df["Arvo_12kk_plot"],
                mode="lines",
                name="12 kk kertymä",
                line=dict(width=4),
                yaxis="y2",
            )
        )
        fig.update_layout(
            yaxis=dict(title="milj. m³ / kk"),
            yaxis2=dict(
                title="12 kk kertymä (milj. m³)",
                overlaying="y",
                side="right",
                showgrid=False,
            ),
        )
    else:
        fig.update_layout(yaxis=dict(title="milj. m³ / kk"))

    fig.update_layout(
        title="Teollisen puukaupan kehitys",
        xaxis_title="Aika",
        hovermode="x unified",
    )
    st.plotly_chart(fig, width="stretch")


def render_harvests_section(df: pd.DataFrame) -> None:
    st.subheader("🪓 Hakkuut")
    st.caption("Hakkuumäärät alueittain, omistajittain ja puutavaralajeittain.")

    mk_col = find_first_matching_column(df, ["Maakunta", "MK", "Alue"])
    om_col = find_first_matching_column(df, ["Omistajaryhmä", "OM", "Omistaja"])
    ptl_col = find_first_matching_column(df, ["Puutavaralaji", "PTL"])
    pl_col = find_first_matching_column(df, ["Puulaji", "PL"])
    info_col = find_first_matching_column(df, ["Tieto", "Tiedot"])

    if not all([mk_col, om_col, ptl_col, pl_col]):
        st.warning(f"Hakkuudatan sarakkeita ei tunnistettu oikein. Sarakkeet: {list(df.columns)}")
        return

    infos = sorted(df[info_col].dropna().astype(str).unique().tolist()) if info_col else []
    chosen_info = infos[0] if infos else None

    areas = _sort_options_with_total_first(
        sorted(df[mk_col].dropna().astype(str).unique().tolist()),
        ["koko maa", "yhteensä"],
    )
    owners = _sort_options_with_total_first(
        sorted(df[om_col].dropna().astype(str).unique().tolist()),
        ["yhteensä"],
    )
    ptl_vals = _sort_options_with_total_first(
        sorted(df[ptl_col].dropna().astype(str).unique().tolist()),
        ["yhteensä"],
    )
    pl_vals = _sort_options_with_total_first(
        sorted(df[pl_col].dropna().astype(str).unique().tolist()),
        ["yhteensä"],
    )

    default_area = _default_option(areas, ["koko maa"]) or (areas[0] if areas else None)
    default_owner = _default_option(owners, ["yhteensä"]) or (owners[0] if owners else None)
    default_ptl = _default_option(ptl_vals, ["yhteensä"]) or (ptl_vals[0] if ptl_vals else None)
    default_pl = _default_option(pl_vals, ["yhteensä"]) or (pl_vals[0] if pl_vals else None)

    if "forest_harvest_selected_area" not in st.session_state:
        st.session_state["forest_harvest_selected_area"] = default_area
    if "forest_harvest_selected_owner" not in st.session_state:
        st.session_state["forest_harvest_selected_owner"] = default_owner
    if "forest_harvest_selected_ptl" not in st.session_state:
        st.session_state["forest_harvest_selected_ptl"] = default_ptl
    if "forest_harvest_selected_pl" not in st.session_state:
        st.session_state["forest_harvest_selected_pl"] = default_pl

    with st.form("forest_harvests_form"):
        st.markdown("### ⚙️ Valinnat")
        cols = st.columns(4)

        chosen_mk = cols[0].selectbox(
            "Alue",
            areas,
            index=areas.index(st.session_state["forest_harvest_selected_area"]) if st.session_state["forest_harvest_selected_area"] in areas else 0,
        )
        chosen_om = cols[1].selectbox(
            "Omistaja",
            owners,
            index=owners.index(st.session_state["forest_harvest_selected_owner"]) if st.session_state["forest_harvest_selected_owner"] in owners else 0,
        )
        chosen_ptl = cols[2].selectbox(
            "Puutavaralaji",
            ptl_vals,
            index=ptl_vals.index(st.session_state["forest_harvest_selected_ptl"]) if st.session_state["forest_harvest_selected_ptl"] in ptl_vals else 0,
        )
        chosen_pl = cols[3].selectbox(
            "Puulaji",
            pl_vals,
            index=pl_vals.index(st.session_state["forest_harvest_selected_pl"]) if st.session_state["forest_harvest_selected_pl"] in pl_vals else 0,
        )

        submitted = st.form_submit_button("Hae", use_container_width=False)

    if submitted:
        st.session_state["forest_harvest_selected_area"] = chosen_mk
        st.session_state["forest_harvest_selected_owner"] = chosen_om
        st.session_state["forest_harvest_selected_ptl"] = chosen_ptl
        st.session_state["forest_harvest_selected_pl"] = chosen_pl

    active_mk = st.session_state["forest_harvest_selected_area"]
    active_om = st.session_state["forest_harvest_selected_owner"]
    active_ptl = st.session_state["forest_harvest_selected_ptl"]
    active_pl = st.session_state["forest_harvest_selected_pl"]

    f = df.copy()
    if info_col and chosen_info is not None:
        f = f[f[info_col].astype(str) == str(chosen_info)].copy()

    f = f[
        (f[mk_col] == active_mk)
        & (f[om_col] == active_om)
        & (f[ptl_col] == active_ptl)
        & (f[pl_col] == active_pl)
    ].copy()

    if f.empty:
        st.info("Valinnoilla ei löytynyt hakkuudataa.")
        return

    annual_df = _collapse_series_by_date(f, how="sum")
    latest, yoy = latest_and_yoy(annual_df, "Arvo", periods=1)
    latest_label = _latest_year_label(annual_df, prefix="Vuoden")

    metric_card(
        "Viimeisin hakkuumäärä",
        _format_million_m3_from_thousand(latest),
        f"{yoy:+.1f} % (1 v)" if yoy is not None else None,
        f"{latest_label} hakkuumäärä • {active_mk} • {active_om} • {active_ptl} • {active_pl}",
    )

    plot_df = annual_df.copy()
    plot_df["Arvo_milj_m3"] = plot_df["Arvo"] / 1000

    fig = px.line(
        plot_df,
        x="Date",
        y="Arvo_milj_m3",
        markers=True,
        title="Hakkuumäärän kehitys",
        labels={"Date": "Vuosi", "Arvo_milj_m3": "milj. m³"},
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown("#### 🗺️ Aluevertailu viimeisimmällä vuodella")
    latest_date = plot_df["Date"].max()
    comp = df.copy()
    if info_col and chosen_info is not None:
        comp = comp[comp[info_col].astype(str) == str(chosen_info)].copy()

    comp = comp[
        (comp["Date"] == latest_date)
        & (comp[om_col] == active_om)
        & (comp[ptl_col] == active_ptl)
        & (comp[pl_col] == active_pl)
    ].copy()

    if not comp.empty:
        comp = comp[comp[mk_col].astype(str).str.upper() != "KOKO MAA"].copy()
        comp = comp.groupby(mk_col, as_index=False)["Arvo"].sum().sort_values("Arvo", ascending=False)
        comp["Arvo_milj_m3"] = comp["Arvo"] / 1000

        bar = px.bar(
            comp,
            x=mk_col,
            y="Arvo_milj_m3",
            title=f"Aluevertailu – {active_ptl} / {active_pl} / {active_om}",
            labels={mk_col: "Alue", "Arvo_milj_m3": "milj. m³"},
        )
        st.plotly_chart(bar, width="stretch")


def render_wood_use_section(df: pd.DataFrame) -> None:
    st.subheader("🏗️ Puun käyttö")
    st.caption("Puun käyttö koko maassa: raakapuu yhteensä, metsäteollisuus ja energiakäyttö.")

    kt_col = find_first_matching_column(df, ["Käyttötarkoitus", "Käyttötapa", "KT"])
    if kt_col is None:
        st.warning(f"Puun käytön sarakkeita ei tunnistettu oikein. Sarakkeet: {list(df.columns)}")
        return

    f = df.copy()
    f["Date"] = pd.to_datetime(f["Date"], errors="coerce")
    f["Arvo"] = pd.to_numeric(f["Arvo"], errors="coerce")
    f = f.dropna(subset=["Date", "Arvo"]).copy()

    if f.empty:
        st.info("Puun käytön dataa ei löytynyt.")
        return

    labels = sorted(f[kt_col].dropna().astype(str).unique().tolist())

    def pick_label(preferred_terms: list[str]) -> str | None:
        for term in preferred_terms:
            term_l = term.lower()
            for label in labels:
                if term_l in str(label).lower():
                    return label
        return None

    total_label = pick_label(["yhteensä", "raakapuu yhteensä"])
    industry_label = pick_label(["metsäteoll", "raakapuu metsäteollisuus"])
    energy_label = pick_label(["energi", "raakapuu energiakäyttö"])

    if total_label is None or industry_label is None or energy_label is None:
        st.warning(f"Puun käytön sarjoja ei löytynyt odotetusti. Sarjat: {labels}")
        return

    total_df = _collapse_series_by_date(
        f[f[kt_col].astype(str) == str(total_label)].copy(),
        how="mean",
    )
    industry_df = _collapse_series_by_date(
        f[f[kt_col].astype(str) == str(industry_label)].copy(),
        how="mean",
    )
    energy_df = _collapse_series_by_date(
        f[f[kt_col].astype(str) == str(energy_label)].copy(),
        how="mean",
    )

    if total_df.empty:
        st.info("Puun käytön kokonaisdataa ei löytynyt.")
        return

    total_latest, total_yoy = latest_and_yoy(total_df, "Arvo", periods=1)
    industry_latest, industry_yoy = latest_and_yoy(industry_df, "Arvo", periods=1)
    energy_latest, energy_yoy = latest_and_yoy(energy_df, "Arvo", periods=1)

    latest_label = _latest_year_label(total_df, prefix="Vuoden")

    total_df["Arvo_plot"] = total_df["Arvo"] / 1000
    industry_df["Arvo_plot"] = industry_df["Arvo"] / 1000 if not industry_df.empty else pd.Series(dtype=float)
    energy_df["Arvo_plot"] = energy_df["Arvo"] / 1000 if not energy_df.empty else pd.Series(dtype=float)

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card(
            "Raakapuun käyttö yhteensä",
            _format_million_m3_from_thousand(total_latest),
            f"{total_yoy:+.1f} % (1 v)" if total_yoy is not None else None,
            f"{latest_label} • {total_label}",
        )

    with c2:
        metric_card(
            "Metsäteollisuuden puunkäyttö",
            _format_million_m3_from_thousand(industry_latest),
            f"{industry_yoy:+.1f} % (1 v)" if industry_yoy is not None else None,
            f"{latest_label} • {industry_label}",
        )

    with c3:
        metric_card(
            "Energiapuun käyttö",
            _format_million_m3_from_thousand(energy_latest),
            f"{energy_yoy:+.1f} % (1 v)" if energy_yoy is not None else None,
            f"{latest_label} • {energy_label}",
        )

    st.markdown("### 📈 Metsäteollisuus ja energiakäyttö")
    fig_compare = go.Figure()

    if not industry_df.empty:
        fig_compare.add_trace(
            go.Scatter(
                x=industry_df["Date"],
                y=industry_df["Arvo_plot"],
                mode="lines+markers",
                name="Metsäteollisuus",
            )
        )

    if not energy_df.empty:
        fig_compare.add_trace(
            go.Scatter(
                x=energy_df["Date"],
                y=energy_df["Arvo_plot"],
                mode="lines+markers",
                name="Energiakäyttö",
            )
        )

    fig_compare.update_layout(
        title="Puun käytön kehitys – metsäteollisuus ja energiakäyttö",
        xaxis_title="Vuosi",
        yaxis_title="milj. m³",
        hovermode="x unified",
    )
    st.plotly_chart(fig_compare, width="stretch")

    st.markdown("### 📊 Raakapuun käyttö yhteensä")
    fig_total = px.line(
        total_df,
        x="Date",
        y="Arvo_plot",
        markers=True,
        title="Raakapuun käytön kehitys – yhteensä",
        labels={"Date": "Vuosi", "Arvo_plot": "milj. m³"},
    )
    st.plotly_chart(fig_total, width="stretch")

def _stock_pct_color(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "#6b7280"
    return "#15803d" if value >= 0 else "#b91c1c"


def _stock_pct_html(value: float | None) -> str:
    if value is None or pd.isna(value):
        txt = "—"
    else:
        sign = "+" if value >= 0 else ""
        txt = f"{sign}{value:.1f} %"

    return f"""
    <span style="
        color:{_stock_pct_color(value)};
        font-weight:700;
        font-size:1.15rem;
    ">
        {txt}
    </span>
    """


def render_forest_stocks_section(bundle: dict) -> None:
    st.subheader("📈 Metsäyhtiöt")
    st.caption(
        "Suomalaisten metsä- ja metsäteollisuuteen liittyvien pörssiyhtiöiden kurssikehitys. "
        "Muutosluvut ovat prosentteja: vihreä = nousu, punainen = lasku."
    )

    snapshots = bundle.get("snapshots", [])
    normalized = bundle.get("normalized", pd.DataFrame())

    if not snapshots:
        st.info("Metsäyhtiöiden osakedataa ei saatu.")
        return

    for i in range(0, len(snapshots), 3):
        cols = st.columns(3)

        for col, snap in zip(cols, snapshots[i : i + 3]):
            with col:
                with st.container(border=True):
                    st.markdown(f"### {snap['Yhtiö']}")
                    st.caption(f"{snap['Symboli']} • {snap['Kuvaus']}")

                    value = snap.get("Nyt")
                    value_txt = "—" if value is None or pd.isna(value) else f"{value:,.2f} €".replace(",", " ")

                    st.markdown(f"**Kurssi nyt:** {value_txt}")

                    c1, c2 = st.columns(2)
                    with c1:
                        st.caption("1 kk")
                        st.markdown(_stock_pct_html(snap.get("1 kk %")), unsafe_allow_html=True)

                    with c2:
                        st.caption("1 v")
                        st.markdown(_stock_pct_html(snap.get("1 v %")), unsafe_allow_html=True)

    st.divider()

    st.markdown("### 💶 Miten 100 € olisi kehittynyt?")
    st.caption("Kaikki yhtiöt alkavat arvosta 100, jotta kehitystä on helpompi vertailla.")

    if normalized is None or normalized.empty:
        st.info("Vertailukuvaajaa ei voitu muodostaa.")
        return

    fig = px.line(
        normalized,
        x="Date",
        y="Arvo",
        color="Yhtiö",
        title="Metsäyhtiöiden suhteellinen kurssikehitys",
        labels={"Date": "Aika", "Arvo": "Arvo, kun alkuhetki = 100", "Yhtiö": ""},
    )
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, width="stretch")

    with st.expander("Näytä tekninen taulukko"):
        rows = []
        for snap in snapshots:
            rows.append(
                {
                    "Yhtiö": snap["Yhtiö"],
                    "Symboli": snap["Symboli"],
                    "Kurssi": snap["Nyt"],
                    "1 kk %": snap["1 kk %"],
                    "1 v %": snap["1 v %"],
                }
            )

        table = pd.DataFrame(rows)
        st.dataframe(table, use_container_width=True, hide_index=True)


def render_forest_analysis_section(bundle: dict) -> None:
    st.subheader("🧠 Metsäsektorin analyysi")
    st.caption(
        "Yhdistää puumarkkinan, metsäteollisuuden kysynnän, viennin ja metsäyhtiöiden markkinatunnelman yhdeksi tilannekuvaksi."
    )

    if not bundle:
        st.info("Analyysia ei voitu muodostaa.")
        return

    cycle_icon = bundle.get("cycle_icon", "⚪")
    cycle_label = bundle.get("cycle_label", "Ei dataa")
    summary = bundle.get("summary", "")

    with st.container(border=True):
        st.markdown(f"## {cycle_icon} {cycle_label}")
        st.write(summary)

    st.divider()

    st.markdown("### 📌 Tilaindikaattorit")

    indicators = bundle.get("indicators", [])
    if indicators:
        for i in range(0, len(indicators), 3):
            cols = st.columns(3)

            for col, item in zip(cols, indicators[i : i + 3]):
                with col:
                    with st.container(border=True):
                        st.markdown(f"### {item['Ikoni']} {item['Osa-alue']}")
                        st.markdown(f"**Tila:** {item['Tila']}")
                        st.markdown(_stock_pct_html(item.get("Muutos")), unsafe_allow_html=True)
                        st.caption(item.get("Selite", ""))

    st.divider()

    st.markdown("### 🚢 Metsäteollisuuden vienti ja tuonti")

    trade = bundle.get("trade", {})
    trade_df = trade.get("trade_df", pd.DataFrame())

    c1, c2 = st.columns(2)

    with c1:
        metric_card(
            "Metsäteollisuuden vienti, 12 kk",
            _fmt_analysis_money(trade.get("latest_export_12kk")),
            f"{trade.get('export_yoy'):+.1f} % (1 v)" if trade.get("export_yoy") is not None else None,
            "Tulli / Uljas",
        )

    with c2:
        metric_card(
            "Metsäteollisuuden nettovienti, 12 kk",
            _fmt_analysis_money(trade.get("latest_net_12kk")),
            f"{trade.get('net_yoy'):+.1f} % (1 v)" if trade.get("net_yoy") is not None else None,
            "Vienti − tuonti",
        )

    if trade_df is not None and not trade_df.empty:
        plot_df = trade_df.copy()
        plot_df["Vienti_12kk_milj"] = plot_df["Vienti_12kk"] / 1_000_000
        plot_df["Tuonti_12kk_milj"] = plot_df["Tuonti_12kk"] / 1_000_000
        plot_df["Nettovienti_12kk_milj"] = plot_df["Nettovienti_12kk"] / 1_000_000

        trade_plot = plot_df.melt(
            id_vars=["Aika_dt"],
            value_vars=["Vienti_12kk_milj", "Tuonti_12kk_milj", "Nettovienti_12kk_milj"],
            var_name="Sarja",
            value_name="Arvo",
        ).dropna()

        name_map = {
            "Vienti_12kk_milj": "Vienti 12 kk",
            "Tuonti_12kk_milj": "Tuonti 12 kk",
            "Nettovienti_12kk_milj": "Nettovienti 12 kk",
        }
        trade_plot["Sarja"] = trade_plot["Sarja"].map(name_map)

        fig = px.line(
            trade_plot,
            x="Aika_dt",
            y="Arvo",
            color="Sarja",
            title="Metsäteollisuuden ulkomaankauppa, 12 kk liukuva summa",
            labels={"Aika_dt": "Aika", "Arvo": "milj. €", "Sarja": ""},
        )
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, width="stretch")

    st.divider()

    st.markdown("### 🧭 Miten tätä kannattaa tulkita?")
    st.info(
        "Tämä analyysi ei ole sijoitussuositus. Se kokoaa eri lähteistä metsäsektorin suhdannekuvaa: "
        "puun hinnat kertovat raaka-aineen markkinasta, puukauppa ja puunkäyttö teollisuuden aktiivisuudesta, "
        "vienti ulkoisesta kysynnästä ja metsäyhtiöiden osakkeet markkinoiden odotuksista."
    )


def _fmt_analysis_money(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value / 1_000_000:,.0f} milj. €".replace(",", " ")