# tabs/energy.py
import streamlit as st
import pandas as pd
import plotly.express as px

from services.energy_pxweb import (
    fetch_electricity_production_consumption,
    fetch_heating_energy_prices,
)

from services.energy_correl import (
    build_electricity_features,
    build_price_series,
    merge_price_and_features,
    corr_table,
    lag_corr,
    rolling_corr,
)

try:
    from services.energy_pxweb import fetch_household_electricity_prices
    HAS_HH_ELECTRICITY = True
except Exception:
    HAS_HH_ELECTRICITY = False


# -------------------------
# Cache
# -------------------------
@st.cache_data(show_spinner="Haetaan sähkön tuotanto/hankinta -dataa…")
def load_electricity() -> pd.DataFrame:
    return fetch_electricity_production_consumption()


@st.cache_data(show_spinner="Haetaan lämmitysenergian hintadataa…")
def load_heating_prices() -> pd.DataFrame:
    return fetch_heating_energy_prices()


@st.cache_data(show_spinner="Haetaan kotitaloussähkön hintadataa…")
def load_household_electricity_prices() -> pd.DataFrame:
    if not HAS_HH_ELECTRICITY:
        return pd.DataFrame()
    return fetch_household_electricity_prices()


# -------------------------
# Helpers
# -------------------------
def _dedupe_df_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Plotly/narwhals vaatii uniikit sarakenimet."""
    if df is None or df.empty:
        return pd.DataFrame()
    cols = list(df.columns)
    seen: dict[str, int] = {}
    new_cols: list[str] = []
    for c in cols:
        base = str(c)
        if base not in seen:
            seen[base] = 0
            new_cols.append(base)
        else:
            seen[base] += 1
            new_cols.append(f"{base}__{seen[base]}")
    out = df.copy()
    out.columns = new_cols
    return out


def _filter_monthly_values_only(df: pd.DataFrame) -> pd.DataFrame:
    """
    PXWeb-tauluissa on usein dimensio, jossa on sekä kuukausiarvo että kertymä (vuoden alusta).
    Tämä suodattaa pois kertymät ja yrittää jättää vain kuukausiarvon.

    Miksi tärkeä:
    - jos samassa kuukaudessa useita rivejä -> Plotly tekee pystypiikkejä
    - jos summataan -> kertymä aiheuttaa "vuosikumulatiivisen" näköisen nousun
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    # Etsi mahdollinen "tieto/tiedot/suure" -dimensio (tai vastaava)
    dim_cols: list[str] = []
    for c in out.columns:
        cl = str(c).strip().lower()
        if cl in ("tiedot", "tieto", "suure", "mittaustieto", "tietolaji"):
            dim_cols.append(c)
        elif cl.startswith("tiedot") or cl.startswith("tieto") or cl.startswith("suure"):
            dim_cols.append(c)

    if not dim_cols:
        return out

    for col in dim_cols:
        s = out[col].astype(str).str.lower()

        is_cum = (
            s.str.contains("kertym", na=False)
            | s.str.contains("vuoden alusta", na=False)
            | s.str.contains("kumul", na=False)
            | s.str.contains("ytd", na=False)
        )
        tmp = out.loc[~is_cum].copy()

        # Suosi kuukausi/kk-tyyppistä, jos löytyy
        if not tmp.empty:
            s2 = tmp[col].astype(str).str.lower()
            is_month = s2.str.contains("kuukaus", na=False) | s2.str.contains("kk", na=False)
            if is_month.any():
                tmp = tmp.loc[is_month].copy()
            out = tmp

    return out


def _find_series_col(df: pd.DataFrame) -> str | None:
    """
    Palauttaa sarakkeen nimen, joka vastaa 'Sähkön tuotanto/hankinta'.
    Huom: dedup voi tehdä '...__1' -variaatioita.
    """
    if df is None or df.empty:
        return None

    for c in df.columns:
        if str(c).strip().lower() == "sähkön tuotanto/hankinta":
            return c

    for c in df.columns:
        s = str(c).lower()
        if "sähkön tuotanto/hankinta" in s:
            return c

    for c in df.columns:
        s = str(c).lower()
        if "tuotanto" in s and "hankinta" in s:
            return c

    return None


def _time_col(df: pd.DataFrame) -> str:
    if "Aika_dt" in df.columns and df["Aika_dt"].notna().any():
        return "Aika_dt"
    if "Aika" in df.columns:
        return "Aika"
    return df.columns[0]


def _clip_years(df: pd.DataFrame, years_back: int) -> pd.DataFrame:
    f = df.copy()
    if "Aika_dt" in f.columns and f["Aika_dt"].notna().any():
        last = f["Aika_dt"].max()
        start = last - pd.DateOffset(years=int(years_back))
        f = f[f["Aika_dt"] >= start].copy()
    return f

def _keep_full_years(df: pd.DataFrame, series_col: str | None = None, min_months: int = 12) -> pd.DataFrame:
    """
    Poistaa vajaat vuodet (esim. jos years_back leikkaa kesken vuoden).
    Säilyttää vain ne vuodet, joilta löytyy vähintään `min_months` eri kuukautta.

    Jos series_col annettu, tarkistus tehdään per (vuosi, sarja).
    """
    if df is None or df.empty:
        return pd.DataFrame()

    f = _ensure_time_dt(df).copy()
    if "Aika_dt" not in f.columns or not f["Aika_dt"].notna().any():
        return f

    f["_Year"] = f["Aika_dt"].dt.year
    f["_MonthNo"] = f["Aika_dt"].dt.month

    if series_col and series_col in f.columns:
        # Count unique months per year & series
        counts = (
            f.dropna(subset=["_Year", "_MonthNo"])
             .groupby(["_Year", series_col])["_MonthNo"]
             .nunique()
             .reset_index(name="_n_months")
        )
        ok = counts[counts["_n_months"] >= min_months][["_Year", series_col]]
        out = f.merge(ok, on=["_Year", series_col], how="inner")
    else:
        counts = (
            f.dropna(subset=["_Year", "_MonthNo"])
             .groupby("_Year")["_MonthNo"]
             .nunique()
             .reset_index(name="_n_months")
        )
        ok_years = set(counts.loc[counts["_n_months"] >= min_months, "_Year"].tolist())
        out = f[f["_Year"].isin(ok_years)].copy()

    return out.drop(columns=[c for c in ["_Year", "_MonthNo"] if c in out.columns])

def _ensure_time_dt(df: pd.DataFrame) -> pd.DataFrame:
    """Varmistaa että Aika_dt löytyy jos Aika on esim 1992M01."""
    f = df.copy()
    if "Aika_dt" in f.columns and f["Aika_dt"].notna().any():
        return f

    if "Aika" in f.columns:
        s = f["Aika"].astype(str).str.strip()

        def _parse_aika(x: str):
            if "M" in x and len(x) >= 7 and x[:4].isdigit():
                y = x[:4]
                m = x.split("M")[-1][:2]
                if m.isdigit():
                    return f"{y}-{m}-01"
            return x

        f["Aika_dt"] = pd.to_datetime(s.map(_parse_aika), errors="coerce")
    else:
        f["Aika_dt"] = pd.NaT

    return f


def _unit_transform_energy(df: pd.DataFrame, unit: str) -> tuple[pd.DataFrame, str]:
    """GWh -> TWh tarvittaessa."""
    f = df.copy()
    f["Arvo"] = pd.to_numeric(f["Arvo"], errors="coerce")
    if unit == "TWh":
        f["Arvo"] = f["Arvo"] / 1000.0
        return f, "TWh"
    return f, "GWh"


def _render_one_series(df: pd.DataFrame, x: str, title: str, y_label: str):
    """
    Selkeä renderöinti:
    - Kuukausi (datetime): pylväsdiagrammi + 1 pylväs / kk (summa)
    - Muut: viiva
    """
    f = df.dropna(subset=["Arvo"]).copy()
    if f.empty:
        st.info("Ei dataa valinnoilla.")
        return

    if x in f.columns:
        xd = pd.to_datetime(f[x], errors="coerce")
        if xd.notna().any():
            f["_x_dt"] = xd
            f["_Month"] = f["_x_dt"].dt.to_period("M").dt.to_timestamp(how="start")
            f = f.dropna(subset=["_Month"])
            f = f.groupby("_Month", as_index=False)["Arvo"].sum().sort_values("_Month")

            fig = px.bar(
                f,
                x="_Month",
                y="Arvo",
                title=title,
                labels={"_Month": "Aika", "Arvo": y_label},
            )
            fig.update_traces(hovertemplate="%{x|%Y-%m}: %{y}<extra></extra>")
            st.plotly_chart(fig, use_container_width=True)
            return

    f = f.sort_values(x)
    fig = px.line(
        f,
        x=x,
        y="Arvo",
        markers=True,
        title=title,
        labels={x: "Aika", "Arvo": y_label},
    )
    st.plotly_chart(fig, use_container_width=True)


def _is_import_series(name: str) -> bool:
    n = str(name).strip().upper()
    return n.startswith("2 ") or n.startswith("2.") or "NETTOTUONTI" in n


def _is_production_series(name: str) -> bool:
    n = str(name).strip().upper()
    if n.startswith("SSS"):
        return False
    if n.startswith("2 ") or n.startswith("2."):
        return False
    return n.startswith("1")


def _pick_consumption_series(all_series: list[str]) -> str | None:
    for s in all_series:
        if str(s).strip().upper().startswith("SSS"):
            return s
    for s in all_series:
        if "KOKONAISKULUTUS" in str(s).upper():
            return s
    return None


def _agg_time(df: pd.DataFrame, series_col: str, mode: str) -> pd.DataFrame:
    """
    mode: "Kuukausi" (ei aggregointia) tai "Vuosi (summa)"
    """
    f = _ensure_time_dt(df)

    if mode == "Vuosi (summa)" and "Aika_dt" in f.columns and f["Aika_dt"].notna().any():
        f["Vuosi"] = f["Aika_dt"].dt.year.astype("Int64")
        f = (
            f.dropna(subset=["Vuosi"])
            .groupby(["Vuosi", series_col], as_index=False)["Arvo"].sum()
        )
        f["Aika_dt"] = pd.to_datetime(f["Vuosi"].astype(str) + "-01-01", errors="coerce")
        f["Aika"] = f["Vuosi"].astype(int).astype(str)
    return f


# -------------------------
# Electricity (system)
# -------------------------
def _render_electricity_system():
    st.subheader("⚡ Sähkö (tuotanto, kulutus ja tuonti)")
    st.caption("Lähde: Tilastokeskus / StatFin (PXWeb) – statfin_ehk_pxt_12su")

    df = _dedupe_df_cols(load_electricity())
    if df.empty:
        st.warning("Sähködataa ei saatu ladattua.")
        return

    # ✅ suodata pois kertymät/kumulatiiviset
    df = _filter_monthly_values_only(df)

    series_col = _find_series_col(df)
    if not series_col:
        st.error(f"En löytänyt saraketta 'Sähkön tuotanto/hankinta'. Sarakkeet: {list(df.columns)}")
        return

    with st.expander("⚙️ Yleiset valinnat (sähkö)", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            time_mode = st.radio(
                "Aikataso",
                ["Kuukausi", "Vuosi (summa)"],
                index=0,
                horizontal=True,
                key="energy_time_mode",
            )
        with c2:
            unit = st.selectbox("Yksikkö", ["GWh", "TWh"], index=1, key="energy_unit")
        with c3:
            years_back = st.slider("Aikajänne (vuotta)", 5, 30, 10, key="energy_sys_years")

    base = _ensure_time_dt(df.copy())
    base = _clip_years(base, int(years_back))
    if time_mode == "Vuosi (summa)":
        base = _keep_full_years(base, series_col=series_col, min_months=12)
    all_series = sorted(base[series_col].dropna().astype(str).unique().tolist())

    # --- 1) Kokonaiskulutus
    st.markdown("### 🧾 Sähkön kokonaiskulutus")
    cons_name = _pick_consumption_series(all_series)
    if not cons_name:
        st.info("Kokonaiskulutuksen sarjaa (SSS...) ei löytynyt tästä datasetistä.")
    else:
        cons = base[base[series_col].astype(str) == str(cons_name)].copy()
        cons = _agg_time(cons, series_col, time_mode)
        cons, y_unit = _unit_transform_energy(cons, unit)

        _render_one_series(
            cons,
            x=("Aika_dt" if ("Aika_dt" in cons.columns and cons["Aika_dt"].notna().any()) else "Aika"),
            title=f"Kokonaiskulutus ({time_mode.lower()})",
            y_label=f"Määrä ({y_unit})",
        )

    st.divider()

    # --- 2) Tuonti
    st.markdown("### 🌍 Tuontisähkö (nettotuonti ja maat)")
    import_series = [s for s in all_series if _is_import_series(s)]
    if not import_series:
        st.info("Tuontisähkön (2.* / nettotuonti) sarjoja ei löytynyt tästä datasetistä.")
    else:
        with st.expander("⚙️ Valinnat (tuonti)", expanded=True):
            default_imp = import_series[:5] if len(import_series) > 5 else import_series
            chosen_import = st.multiselect(
                "Näytettävät tuontisarjat",
                import_series,
                default=default_imp,
                key="energy_import_series",
            )

        imp = base[base[series_col].astype(str).isin([str(x) for x in chosen_import])].copy()
        imp = _agg_time(imp, series_col, time_mode)
        imp, y_unit = _unit_transform_energy(imp, unit)

        # Kuukausi -> pylväs (selkeä), Vuosi -> viiva (koska sarjoja voi olla monta)
        xcol = "Aika_dt" if ("Aika_dt" in imp.columns and imp["Aika_dt"].notna().any()) else "Aika"
        if time_mode == "Kuukausi" and xcol == "Aika_dt":
            imp2 = imp.dropna(subset=["Arvo"]).copy()
            imp2["_Month"] = pd.to_datetime(imp2[xcol], errors="coerce").dt.to_period("M").dt.to_timestamp(how="start")
            imp2 = imp2.dropna(subset=["_Month"])
            imp2 = imp2.groupby(["_Month", series_col], as_index=False)["Arvo"].sum().sort_values("_Month")

            fig = px.bar(
                imp2,
                x="_Month",
                y="Arvo",
                color=series_col,
                barmode="stack",
                title="Tuontisähkö (kuukausi)",
                labels={"_Month": "Aika", "Arvo": f"Määrä ({y_unit})", series_col: "Sarja"},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.line(
                imp.dropna(subset=["Arvo"]).sort_values(xcol),
                x=xcol,
                y="Arvo",
                color=series_col,
                title=f"Tuontisähkö ({time_mode.lower()})",
                labels={"Arvo": f"Määrä ({y_unit})", series_col: "Sarja"},
            )
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --- 3) Tuotanto menetelmittäin
    st.markdown("### 🏭 Tuotanto menetelmittäin")
    prod_series = [s for s in all_series if _is_production_series(s)]
    if not prod_series:
        st.info("Tuotannon (1.*) sarjoja ei löytynyt tästä datasetistä.")
    else:
        default_candidates = []
        for want in ["1.1", "1.2", "1.3", "1.4", "1.5", "1.6"]:
            for s in prod_series:
                if str(s).strip().startswith(want):
                    default_candidates.append(s)

        default_prod = default_candidates[:8] if default_candidates else (prod_series[:8] if len(prod_series) > 8 else prod_series)

        with st.expander("⚙️ Valinnat (tuotanto)", expanded=True):
            chosen_prod = st.multiselect(
                "Näytettävät tuotantosarjat",
                prod_series,
                default=default_prod,
                key="energy_production_series",
            )

        prod = base[base[series_col].astype(str).isin([str(x) for x in chosen_prod])].copy()
        prod = _agg_time(prod, series_col, time_mode)
        prod, y_unit = _unit_transform_energy(prod, unit)

        xcol = "Aika_dt" if ("Aika_dt" in prod.columns and prod["Aika_dt"].notna().any()) else "Aika"
        if time_mode == "Kuukausi" and xcol == "Aika_dt":
            prod2 = prod.dropna(subset=["Arvo"]).copy()
            prod2["_Month"] = pd.to_datetime(prod2[xcol], errors="coerce").dt.to_period("M").dt.to_timestamp(how="start")
            prod2 = prod2.dropna(subset=["_Month"])
            prod2 = prod2.groupby(["_Month", series_col], as_index=False)["Arvo"].sum().sort_values("_Month")

            fig = px.bar(
                prod2,
                x="_Month",
                y="Arvo",
                color=series_col,
                barmode="stack",
                title="Tuotanto menetelmittäin (kuukausi)",
                labels={"_Month": "Aika", "Arvo": f"Määrä ({y_unit})", series_col: "Sarja"},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.line(
                prod.dropna(subset=["Arvo"]).sort_values(xcol),
                x=xcol,
                y="Arvo",
                color=series_col,
                title=f"Tuotanto menetelmittäin ({time_mode.lower()})",
                labels={"Arvo": f"Määrä ({y_unit})", series_col: "Sarja"},
            )
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("🔍 Raakadata (sähköjärjestelmä)"):
        st.dataframe(base.tail(300), use_container_width=True)


# -------------------------
# Comparison tab: price vs shares
# -------------------------
def _render_price_vs_shares():
    st.subheader("🔗 Vertailu: hinta vs tuotantomuotojen osuudet")
    st.caption(
        "Vertaillaan hintaa ja osuuksia (tuuli/ydin/vesi/aurinko) sekä tuontia. "
        "Korrelaatio kertoo yhteisliikkeestä, ei todista syy–seurausta."
    )

    elec = _dedupe_df_cols(load_electricity())
    if elec.empty:
        st.warning("Sähködataa ei saatu ladattua.")
        return

    elec = _filter_monthly_values_only(elec)

    series_col = _find_series_col(elec)
    if not series_col:
        st.error(f"En löytänyt saraketta 'Sähkön tuotanto/hankinta'. Sarakkeet: {list(elec.columns)}")
        return

    elec = _ensure_time_dt(elec)

    with st.expander("⚙️ Valinnat (vertailu)", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            unit = st.selectbox("Sähkö: yksikkö", ["GWh", "TWh"], index=1, key="cmp_unit")
        with c2:
            years_back = st.slider("Aikajänne (vuotta)", 5, 30, 10, key="cmp_years")
        with c3:
            method = st.radio("Korrelaatiotapa", ["pearson", "spearman"], horizontal=True, index=0, key="cmp_method")
        with c4:
            price_source = st.selectbox(
                "Hintasarja",
                ["Lämmitysenergia (€)", "Kotitaloussähkö (snt/kWh)"],
                index=0,
                key="cmp_price_source",
            )

    elec = _clip_years(elec, int(years_back))

    feats = build_electricity_features(elec, series_col=series_col, unit=unit)
    if feats.empty:
        st.info("Featureiden rakentaminen epäonnistui (ei löytynyt tarvittavia sarjoja).")
        return

    # Hintasarja
    price_series = pd.DataFrame()

    if price_source == "Kotitaloussähkö (snt/kWh)" and not HAS_HH_ELECTRICITY:
        st.warning(
            "Kotitaloussähkön hintadata ei ole käytössä (fetch_household_electricity_prices puuttuu). "
            "Käytetään lämmitysenergian hintaa."
        )
        price_source = "Lämmitysenergia (€)"

    if price_source == "Lämmitysenergia (€)":
        price_df = _dedupe_df_cols(load_heating_prices())
        if price_df.empty:
            st.warning("Lämmitysenergian hintadataa ei saatu ladattua.")
            return

        ignore = {"Arvo", "Aika", "Aika_dt", "Vuosi", "Kuukausi", "Neljannes", "Vuosi_num", "Kuukausi_num"}
        dims = [c for c in price_df.columns if c not in ignore]

        price_one = price_df.copy()
        if len(dims) >= 2:
            building_col = dims[0]
            energy_col = dims[1]

            st.markdown("### 💶 Valitse hintasarja (lämmitysenergia)")
            c1, c2 = st.columns(2)
            with c1:
                b = st.selectbox(
                    "Rakennustyyppi",
                    sorted(price_df[building_col].dropna().astype(str).unique()),
                    key="cmp_building",
                )
            with c2:
                e = st.selectbox(
                    "Energiamuoto",
                    sorted(price_df[energy_col].dropna().astype(str).unique()),
                    key="cmp_energy",
                )

            price_one = price_df[
                (price_df[building_col].astype(str) == str(b)) &
                (price_df[energy_col].astype(str) == str(e))
            ].copy()

        price_one = _ensure_time_dt(price_one)
        price_one = _clip_years(price_one, int(years_back))

        price_series = build_price_series(price_one)
        if price_series.empty:
            st.info("Lämmitysenergian hintasarjaa ei saatu kuukausitasolle.")
            return

        price_label = "Hinta (€)"

    else:
        hh = _dedupe_df_cols(load_household_electricity_prices())
        if hh.empty:
            st.warning("Kotitaloussähkön hintadataa ei saatu ladattua.")
            return

        consumer_col = None
        for c in hh.columns:
            if str(c).strip().lower() == "sähkön kuluttajatyyppi":
                consumer_col = c
                break

        component_col = None
        for c in hh.columns:
            if str(c).strip().lower() in ("tiedot", "hintatiedot", "tieto"):
                component_col = c
                break

        if not consumer_col or not component_col:
            st.error(f"En löytänyt tarvittavia sarakkeita kotitaloussähköstä. Sarakkeet: {list(hh.columns)}")
            return

        st.markdown("### 🔌 Valitse hintasarja (kotitaloussähkö)")
        c1, c2 = st.columns(2)
        with c1:
            consumer = st.selectbox(
                "Kuluttajatyyppi",
                sorted(hh[consumer_col].dropna().astype(str).unique()),
                key="cmp_hh_consumer",
            )
        with c2:
            component = st.selectbox(
                "Komponentti",
                sorted(hh[component_col].dropna().astype(str).unique()),
                key="cmp_hh_component",
            )

        hh_one = hh[
            (hh[consumer_col].astype(str) == str(consumer)) &
            (hh[component_col].astype(str) == str(component))
        ].copy()

        hh_one = _ensure_time_dt(hh_one)
        hh_one = _clip_years(hh_one, int(years_back))

        price_series = build_price_series(hh_one)
        if price_series.empty:
            st.info("Kotitaloussähkön hintasarjaa ei saatu kuukausitasolle.")
            return

        price_label = "Hinta (snt/kWh)"

    merged = merge_price_and_features(price_series, feats)
    if merged.empty:
        st.info("Hinnan ja sähköfeatureiden aikaleimat eivät osuneet yhteen (ei yhteisiä kuukausia).")
        return

    merged = merged.copy()
    share_to_pct = {
        "wind_share": "wind_pct",
        "nuclear_share": "nuclear_pct",
        "hydro_share": "hydro_pct",
        "solar_share": "solar_pct",
        "import_share": "import_pct",
    }
    for src, dst in share_to_pct.items():
        if src in merged.columns:
            merged[dst] = merged[src] * 100.0

    candidates = []
    for c in ["wind_pct", "nuclear_pct", "hydro_pct", "solar_pct", "import_pct", "import_net", "total_consumption"]:
        if c in merged.columns:
            candidates.append(c)

    if not candidates:
        st.info("Vertailtavia sarjoja ei löytynyt tästä aineistosta.")
        return

    label_map = {
        "wind_pct": "Tuulivoiman osuus (%)",
        "nuclear_pct": "Ydinvoiman osuus (%)",
        "hydro_pct": "Vesivoiman osuus (%)",
        "solar_pct": "Aurinkovoiman osuus (%)",
        "import_pct": "Nettotuonnin osuus (%)",
        "import_net": "Nettotuonti",
        "total_consumption": "Kokonaiskulutus",
    }

    defaults = [c for c in ["wind_pct", "nuclear_pct", "hydro_pct", "solar_pct", "import_pct", "import_net"] if c in candidates]
    if not defaults:
        defaults = candidates[:3]

    st.markdown("### 📌 Valitse vertailtavat sarjat")
    chosen_feats = st.multiselect(
        "Sarjat",
        options=candidates,
        default=defaults,
        key="cmp_features",
        format_func=lambda x: label_map.get(x, x),
    )
    if not chosen_feats:
        st.info("Valitse vähintään yksi sarja.")
        return

    ct = corr_table(merged, cols=["price"] + chosen_feats, method=method)
    if ct.empty:
        st.info("Korrelaatiota ei saatu laskettua (liian vähän dataa).")
        return

    st.markdown("### 🔗 Korrelaatiot (kuukausi)")
    ct_show = ct.copy()
    ct_show.index = [("Hinta" if i == "price" else label_map.get(i, i)) for i in ct_show.index]
    ct_show.columns = [("Hinta" if c == "price" else label_map.get(c, c)) for c in ct_show.columns]
    st.dataframe(ct_show.round(2), use_container_width=True)

    st.divider()

    st.markdown("### 🎯 Scatter: hinta vs valittu sarja")
    pick = st.selectbox(
        "Näytä scatter",
        options=chosen_feats,
        key="cmp_scatter_pick",
        format_func=lambda x: label_map.get(x, x),
    )

    fig_sc = px.scatter(
        merged.dropna(subset=["price", pick]),
        x=pick,
        y="price",
        trendline="ols",
        labels={pick: label_map.get(pick, pick), "price": price_label},
        title=f"{price_label} vs {label_map.get(pick, pick)} (trendiviiva)",
    )

    if pick.endswith("_pct"):
        fig_sc.update_xaxes(ticksuffix="%", tickformat=".1f")
        fig_sc.update_traces(
            hovertemplate=f"{label_map.get(pick, pick)}=%{{x:.1f}}%<br>{price_label}=%{{y}}<extra></extra>"
        )
    else:
        fig_sc.update_traces(
            hovertemplate=f"{label_map.get(pick, pick)}=%{{x}}<br>{price_label}=%{{y}}<extra></extra>"
        )

    st.plotly_chart(fig_sc, use_container_width=True)

    st.divider()

    st.markdown("### ⏱️ Viivekorrelaatio (0–12 kk)")
    lc = lag_corr(merged, x=pick, y="price", max_lag=12, method=method)
    fig_lag = px.line(
        lc,
        x="lag_months",
        y="corr",
        markers=True,
        labels={"lag_months": "Viive (kk) — sarja(t-lag) vs hinta(t)", "corr": "Korrelaatio"},
        title=f"Viivekorrelaatio: {label_map.get(pick, pick)} → {price_label}",
    )
    st.plotly_chart(fig_lag, use_container_width=True)

    st.divider()

    st.markdown("### 🔄 Rullaava korrelaatio")
    window = st.slider("Ikkuna (kk)", 12, 60, 24, key="cmp_window")
    rc = rolling_corr(merged, x=pick, y="price", window=int(window), method=method)
    fig_roll = px.line(
        rc,
        x="Month",
        y="corr",
        labels={"Month": "Aika", "corr": f"Rullaava korrelaatio ({window} kk)"},
        title=f"Rullaava korrelaatio: {label_map.get(pick, pick)} vs {price_label}",
    )
    st.plotly_chart(fig_roll, use_container_width=True)

    st.caption(
        "Kuukausiseuranta: kertymä-/kumulatiiviset rivit suodatetaan pois ennen kuvaajia ja featureiden rakentamista."
    )

    with st.expander("🔍 Vertailun yhdistetty data (Month, price, featuret)"):
        show_cols = ["Month", "price"] + chosen_feats
        show_cols = [c for c in show_cols if c in merged.columns]
        st.dataframe(merged[show_cols].tail(300), use_container_width=True)


# -------------------------
# Prices (household)
# -------------------------
def _render_prices():
    st.subheader("💶 Hinnat (kotitaloudet)")
    st.caption("Lämmitysenergia + kotitaloussähkö (energia/siirto/verot/kokonais)")

    p1, p2 = st.tabs(["🔥 Lämmitysenergian hinnat", "🔌 Kotitaloussähkö (komponentit)"])

    # --- Lämmitysenergian hinnat
    with p1:
        st.markdown("### 💶 Lämmitysenergian hinnat (kotitaloudet)")
        st.caption("Lähde: Tilastokeskus / StatFin (PXWeb) – statfin_ehi_pxt_13nl")

        df = _dedupe_df_cols(load_heating_prices())
        if df.empty:
            st.warning("Hintadataa ei saatu ladattua.")
        else:
            xcol = _time_col(df)

            ignore = {"Arvo", "Aika", "Aika_dt", "Vuosi", "Kuukausi", "Neljannes", "Vuosi_num", "Kuukausi_num"}
            dims = [c for c in df.columns if c not in ignore]

            if len(dims) < 2:
                st.error(f"Hintadata ei sisällä tarpeeksi dimensioita. Sarakkeet: {list(df.columns)}")
            else:
                building_col = dims[0]
                energy_col = dims[1]

                with st.expander("⚙️ Valinnat (lämmitys)", expanded=True):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        years_back = st.slider("Aikajänne (vuotta)", 5, 30, 10, key="heat_years")
                    with c2:
                        sel_building = st.selectbox(
                            "Rakennustyyppi",
                            sorted(df[building_col].dropna().astype(str).unique().tolist()),
                            key="heat_building",
                        )
                    with c3:
                        sel_energy = st.selectbox(
                            "Energiamuoto",
                            sorted(df[energy_col].dropna().astype(str).unique().tolist()),
                            key="heat_energy",
                        )

                f = df[
                    (df[building_col].astype(str) == str(sel_building)) &
                    (df[energy_col].astype(str) == str(sel_energy))
                ].copy()

                f = _ensure_time_dt(f)
                f = _clip_years(f, int(years_back))
                f = f.dropna(subset=["Arvo"]).sort_values(xcol)

                if f.empty:
                    st.info("Valinnoilla ei löytynyt hintadataa.")
                else:
                    latest_val = float(pd.to_numeric(f["Arvo"].iloc[-1], errors="coerce"))
                    latest_time = f[xcol].iloc[-1]

                    delta = None
                    if len(f) > 12:
                        then_val = float(pd.to_numeric(f["Arvo"].iloc[-13], errors="coerce"))
                        if then_val and not pd.isna(then_val):
                            delta = (latest_val / then_val - 1) * 100

                    cA, cB = st.columns([1, 2])
                    with cA:
                        st.metric(
                            "Viimeisin",
                            f"{latest_val:,.2f} €".replace(",", " "),
                            f"{delta:+.1f} % (1 v)" if delta is not None else None,
                        )
                        st.caption(f"Aika: {latest_time}")
                    with cB:
                        st.write(f"**Rakennus:** {sel_building}  |  **Energiamuoto:** {sel_energy}")
                        st.caption("Yksikkö riippuu StatFinin sarjasta; tässä näytetään arvot euroina (€).")

                    fig = px.line(
                        f,
                        x=xcol,
                        y="Arvo",
                        markers=True,
                        title=f"Hinta: {sel_building} / {sel_energy}",
                        labels={xcol: "Aika", "Arvo": "Hinta (€)"},
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with st.expander("🔍 Raakadata (lämmitys)"):
                    st.dataframe(df.tail(300), use_container_width=True)

    # --- Kotitaloussähkö (komponentit)
    with p2:
        st.markdown("### 🔌 Kotitalouksien sähkön hinta (energia, siirto, verot, kokonais)")
        st.caption("Lähde: Tilastokeskus / StatFin (PXWeb) – statfin_ehi_pxt_13rb")

        if not HAS_HH_ELECTRICITY:
            st.info(
                "Kotitaloussähkön hintadatan hakufunktiota ei löydy `services/energy_pxweb.py`:stä.\n\n"
                "Jos haluat tämän käyttöön, lisää `fetch_household_electricity_prices()` palvelutiedostoon."
            )
            return

        df = _dedupe_df_cols(load_household_electricity_prices())
        if df.empty:
            st.warning("Kotitaloussähkön hintadataa ei saatu ladattua.")
            return

        xcol = _time_col(df)

        consumer_col = None
        for c in df.columns:
            if str(c).strip().lower() == "sähkön kuluttajatyyppi":
                consumer_col = c
                break

        component_col = None
        for c in df.columns:
            if str(c).strip().lower() in ("tiedot", "hintatiedot", "tieto"):
                component_col = c
                break

        if not consumer_col or not component_col:
            st.error(f"En löytänyt tarvittavia sarakkeita. Sarakkeet: {list(df.columns)}")
            return

        with st.expander("⚙️ Valinnat (kotitaloussähkö)", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                years_back = st.slider("Aikajänne (vuotta)", 5, 30, 10, key="hh_elec_years")
            with c2:
                consumer = st.selectbox(
                    "Kuluttajatyyppi",
                    sorted(df[consumer_col].dropna().astype(str).unique()),
                    key="hh_elec_consumer",
                )
            with c3:
                component = st.selectbox(
                    "Komponentti (Tiedot)",
                    sorted(df[component_col].dropna().astype(str).unique()),
                    key="hh_elec_component",
                )

        f = df[
            (df[consumer_col].astype(str) == str(consumer)) &
            (df[component_col].astype(str) == str(component))
        ].copy()

        f = _ensure_time_dt(f)
        f = _clip_years(f, int(years_back))
        f = f.dropna(subset=["Arvo"]).sort_values(xcol)

        if f.empty:
            st.info("Valinnoilla ei löytynyt dataa.")
            return

        latest_val = float(pd.to_numeric(f["Arvo"].iloc[-1], errors="coerce"))
        latest_time = f[xcol].iloc[-1]

        delta = None
        if len(f) > 12:
            then_val = float(pd.to_numeric(f["Arvo"].iloc[-13], errors="coerce"))
            if then_val and not pd.isna(then_val):
                delta = (latest_val / then_val - 1) * 100

        cA, cB = st.columns([1, 2])
        with cA:
            st.metric(
                "Viimeisin",
                f"{latest_val:,.2f} snt/kWh".replace(",", " "),
                f"{delta:+.1f} % (1 v)" if delta is not None else None,
            )
            st.caption(f"Aika: {latest_time}")

        with cB:
            st.write(f"**Kuluttaja:** {consumer}")
            st.write(f"**Komponentti:** {component}")
            st.caption("Yksikkö: snt/kWh (komponentin mukaan energia/siirto/verot/kokonais).")

        _render_one_series(
            f,
            x=xcol,
            title=f"Sähkön hinta: {consumer} / {component}",
            y_label="Hinta (snt/kWh)",
        )

        with st.expander("🔍 Raakadata (kotitaloussähkö)"):
            st.dataframe(df.tail(300), use_container_width=True)


# -------------------------
# Public render
# -------------------------
def render():
    st.subheader("⚡ Energia")

    t1, t2, t3 = st.tabs(
        ["⚡ Sähkö (tuotanto/kulutus/tuonti)" ,"💶 Hinnat (kotitaloudet)", "🔗 Vertailu (hinta vs osuudet)"]
    )
    with t1:
        _render_electricity_system()
    with t2:
        _render_prices()
    with t3:
        _render_price_vs_shares()












