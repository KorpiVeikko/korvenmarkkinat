from __future__ import annotations

import pandas as pd

from services.forest_helpers import find_first_matching_column
from services.macro_uljas import fetch_exports_products, fetch_imports_products


def _pct(now: float | None, then: float | None) -> float | None:
    if now is None or then is None or then == 0 or pd.isna(now) or pd.isna(then):
        return None
    return (now / then - 1.0) * 100.0


def _latest_and_offset_pct(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    offset: pd.DateOffset,
) -> tuple[float | None, float | None, pd.Timestamp | None]:
    if df is None or df.empty:
        return None, None, None

    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=[date_col, value_col]).sort_values(date_col)

    if d.empty:
        return None, None, None

    latest_date = pd.to_datetime(d.iloc[-1][date_col])
    latest_val = float(d.iloc[-1][value_col])

    target = latest_date - offset
    prev = d[d[date_col] <= target]

    pct = None
    if not prev.empty:
        pct = _pct(latest_val, float(prev.iloc[-1][value_col]))

    return latest_val, pct, latest_date


def _series_by_date(df: pd.DataFrame, value_col: str = "Arvo", how: str = "mean") -> pd.DataFrame:
    if df is None or df.empty or "Date" not in df.columns or value_col not in df.columns:
        return pd.DataFrame(columns=["Date", value_col])

    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=["Date", value_col])

    if d.empty:
        return pd.DataFrame(columns=["Date", value_col])

    agg = "sum" if how == "sum" else "mean"

    return (
        d.groupby("Date", as_index=False)[value_col]
        .agg(agg)
        .sort_values("Date")
        .reset_index(drop=True)
    )


def _pick_wood_price_series(
    wood_df: pd.DataFrame,
    labels_to_match: list[str],
) -> pd.DataFrame:
    if wood_df is None or wood_df.empty:
        return pd.DataFrame()

    ptl_col = find_first_matching_column(
        wood_df,
        ["Puutavaralaji", "PTL"],
    )

    if ptl_col is None:
        return pd.DataFrame()

    f = wood_df.copy()

    matched_frames = []

    labels = f[ptl_col].dropna().astype(str).unique().tolist()

    for wanted in labels_to_match:
        for label in labels:
            if wanted.lower() in label.lower():
                matched_frames.append(
                    f[f[ptl_col].astype(str) == str(label)].copy()
                )

    if not matched_frames:
        return pd.DataFrame()

    combined = pd.concat(matched_frames, ignore_index=True)

    return (
        combined.groupby("Date", as_index=False)["Arvo"]
        .mean()
        .sort_values("Date")
        .reset_index(drop=True)
    )


def _pick_wood_use_series(use_df: pd.DataFrame, terms: list[str]) -> pd.DataFrame:
    if use_df is None or use_df.empty:
        return pd.DataFrame()

    kt_col = find_first_matching_column(use_df, ["Käyttötarkoitus", "Käyttötapa", "KT"])
    if kt_col is None:
        return pd.DataFrame()

    labels = use_df[kt_col].dropna().astype(str).unique().tolist()

    chosen = None
    for term in terms:
        for label in labels:
            if term.lower() in label.lower():
                chosen = label
                break
        if chosen:
            break

    if chosen is None:
        return pd.DataFrame()

    return use_df[use_df[kt_col].astype(str) == str(chosen)].copy()


def _pick_harvest_series(harvest_df: pd.DataFrame) -> pd.DataFrame:
    if harvest_df is None or harvest_df.empty:
        return pd.DataFrame()

    mk_col = find_first_matching_column(harvest_df, ["Maakunta", "MK", "Alue"])
    om_col = find_first_matching_column(harvest_df, ["Omistajaryhmä", "OM", "Omistaja"])
    ptl_col = find_first_matching_column(harvest_df, ["Puutavaralaji", "PTL"])
    pl_col = find_first_matching_column(harvest_df, ["Puulaji", "PL"])
    info_col = find_first_matching_column(harvest_df, ["Tieto", "Tiedot"])

    f = harvest_df.copy()

    def _filter_total(col: str | None) -> None:
        nonlocal f
        if not col or col not in f.columns or f.empty:
            return

        values = f[col].dropna().astype(str).unique().tolist()
        preferred = None

        for term in ["koko maa", "yhteensä", "total"]:
            for value in values:
                if term in value.lower():
                    preferred = value
                    break
            if preferred:
                break

        if preferred is not None:
            f = f[f[col].astype(str) == str(preferred)].copy()

    if info_col and info_col in f.columns:
        infos = f[info_col].dropna().astype(str).unique().tolist()
        preferred_info = None

        for value in infos:
            txt = value.lower()
            if "määrä" in txt or "volume" in txt:
                preferred_info = value
                break

        if preferred_info is not None:
            f = f[f[info_col].astype(str) == str(preferred_info)].copy()

    _filter_total(mk_col)
    _filter_total(om_col)
    _filter_total(ptl_col)
    _filter_total(pl_col)

    return f


def _status_from_pct(pct: float | None, positive_good: bool = True) -> tuple[str, str]:
    if pct is None or pd.isna(pct):
        return "⚪", "Ei dataa"

    v = pct if positive_good else -pct

    if v >= 8:
        return "🟢", "Vahva"
    if v >= 2:
        return "🟢", "Kasvava"
    if v > -2:
        return "🟡", "Vakaa"
    if v > -8:
        return "🟠", "Heikkenevä"
    return "🔴", "Heikko"


def _build_trade_analysis(months: int = 84) -> dict:
    exports_df, exports_debug = fetch_exports_products(months=months, lang="fi")
    imports_df, imports_debug = fetch_imports_products(months=months, lang="fi")

    forest_exports = pd.DataFrame()
    forest_imports = pd.DataFrame()

    if exports_df is not None and not exports_df.empty:
        forest_exports = exports_df[exports_df["Tuoteryhmä"].astype(str) == "Metsäteollisuus"].copy()

    if imports_df is not None and not imports_df.empty:
        forest_imports = imports_df[imports_df["Tuoteryhmä"].astype(str) == "Metsäteollisuus"].copy()

    if not forest_exports.empty:
        e = forest_exports.groupby("Aika_dt", as_index=False)["Vienti_eur"].sum()
    else:
        e = pd.DataFrame(columns=["Aika_dt", "Vienti_eur"])

    if not forest_imports.empty:
        i = forest_imports.groupby("Aika_dt", as_index=False)["Tuonti_eur"].sum()
    else:
        i = pd.DataFrame(columns=["Aika_dt", "Tuonti_eur"])

    trade_df = pd.DataFrame()

    if not e.empty or not i.empty:
        trade_df = pd.merge(e, i, on="Aika_dt", how="outer").sort_values("Aika_dt")
        trade_df["Vienti_eur"] = pd.to_numeric(trade_df["Vienti_eur"], errors="coerce").fillna(0)
        trade_df["Tuonti_eur"] = pd.to_numeric(trade_df["Tuonti_eur"], errors="coerce").fillna(0)
        trade_df["Nettovienti_eur"] = trade_df["Vienti_eur"] - trade_df["Tuonti_eur"]
        trade_df["Vienti_12kk"] = trade_df["Vienti_eur"].rolling(12, min_periods=12).sum()
        trade_df["Tuonti_12kk"] = trade_df["Tuonti_eur"].rolling(12, min_periods=12).sum()
        trade_df["Nettovienti_12kk"] = trade_df["Nettovienti_eur"].rolling(12, min_periods=12).sum()

    latest_export, export_yoy, export_date = _latest_and_offset_pct(
        trade_df.dropna(subset=["Vienti_12kk"]) if not trade_df.empty else trade_df,
        "Aika_dt",
        "Vienti_12kk",
        pd.DateOffset(years=1),
    )

    latest_net, net_yoy, _ = _latest_and_offset_pct(
        trade_df.dropna(subset=["Nettovienti_12kk"]) if not trade_df.empty else trade_df,
        "Aika_dt",
        "Nettovienti_12kk",
        pd.DateOffset(years=1),
    )

    return {
        "trade_df": trade_df,
        "latest_export_12kk": latest_export,
        "export_yoy": export_yoy,
        "latest_net_12kk": latest_net,
        "net_yoy": net_yoy,
        "latest_date": export_date,
        "debug": {
            "exports": exports_debug,
            "imports": imports_debug,
        },
    }


def _build_strengths_risks_watchlist(indicators: list[dict]) -> tuple[list[str], list[str], list[str]]:
    strengths: list[str] = []
    risks: list[str] = []
    watchlist: list[str] = []

    for item in indicators:
        name = item.get("Osa-alue", "")
        pct = item.get("Muutos")

        if pct is None or pd.isna(pct):
            watchlist.append(f"{name}: dataa kannattaa seurata, kun uusi havainto päivittyy.")
            continue

        if pct >= 5:
            strengths.append(f"{name}: kehitys on selvästi positiivinen ({pct:+.1f} %).")
        elif pct <= -5:
            risks.append(f"{name}: kehitys on selvästi negatiivinen ({pct:+.1f} %).")
        else:
            watchlist.append(f"{name}: tilanne on melko vakaa ({pct:+.1f} %).")

    if not strengths:
        strengths.append("Selviä vahvuuksia ei erottunut nykyisestä datasta.")

    if not risks:
        risks.append("Selviä riskisignaaleja ei erottunut nykyisestä datasta.")

    watchlist.append("Seuraa erityisesti, liikkuvatko puun hinnat, puukauppa ja vienti samaan suuntaan vai alkavatko ne eriytyä.")

    return strengths, risks, watchlist


def build_forest_analysis_bundle(
    forest_bundle: dict,
    stocks_bundle: dict | None = None,
    months: int = 84,
) -> dict:
    wood_df = forest_bundle.get("wood_df", pd.DataFrame())
    industrial_df = forest_bundle.get("industrial_df", pd.DataFrame())
    harvest_df = forest_bundle.get("harvest_df", pd.DataFrame())
    use_df = forest_bundle.get("use_df", pd.DataFrame())

    wood_price_series = _series_by_date(wood_df, "Arvo", how="mean")

    pine_spruce_logs_series = _pick_wood_price_series(
        wood_df,
        ["Kuusitukki", "Mäntytukki"],
    )

    pine_spruce_pulp_series = _pick_wood_price_series(
        wood_df,
        ["Kuusikuitupuu", "Mäntykuitupuu"],
    )

    birch_logs_series = _pick_wood_price_series(
        wood_df,
        ["Koivutukki"],
    )

    birch_pulp_series = _pick_wood_price_series(
        wood_df,
        ["Koivukuitupuu"],
    )

    industrial_series = _series_by_date(industrial_df, "Arvo", how="sum")

    harvest_selected = _pick_harvest_series(harvest_df)
    harvest_series = _series_by_date(harvest_selected, "Arvo", how="sum")

    industry_use_df = _pick_wood_use_series(use_df, ["metsäteoll"])
    energy_use_df = _pick_wood_use_series(use_df, ["energi"])
    total_use_df = _pick_wood_use_series(use_df, ["yhteensä", "raakapuu yhteensä"])

    industry_use_series = _series_by_date(industry_use_df, "Arvo", how="mean")
    energy_use_series = _series_by_date(energy_use_df, "Arvo", how="mean")
    total_use_series = _series_by_date(total_use_df, "Arvo", how="mean")

    wood_price_latest, wood_price_yoy, wood_price_date = _latest_and_offset_pct(
        wood_price_series, "Date", "Arvo", pd.DateOffset(years=1)
    )

    logs_latest, logs_yoy, logs_date = _latest_and_offset_pct(
        pine_spruce_logs_series,
        "Date",
        "Arvo",
        pd.DateOffset(years=1),
    )

    pulp_latest, pulp_yoy, pulp_date = _latest_and_offset_pct(
        pine_spruce_pulp_series,
        "Date",
        "Arvo",
        pd.DateOffset(years=1),
    )

    birch_logs_latest, birch_logs_yoy, birch_logs_date = _latest_and_offset_pct(
        birch_logs_series,
        "Date",
        "Arvo",
        pd.DateOffset(years=1),
    )

    birch_pulp_latest, birch_pulp_yoy, birch_pulp_date = _latest_and_offset_pct(
        birch_pulp_series,
        "Date",
        "Arvo",
        pd.DateOffset(years=1),
    )

    industrial_latest, industrial_yoy, industrial_date = _latest_and_offset_pct(
        industrial_series, "Date", "Arvo", pd.DateOffset(years=1)
    )
    harvest_latest, harvest_yoy, harvest_date = _latest_and_offset_pct(
        harvest_series, "Date", "Arvo", pd.DateOffset(years=1)
    )
    industry_use_latest, industry_use_yoy, industry_use_date = _latest_and_offset_pct(
        industry_use_series, "Date", "Arvo", pd.DateOffset(years=1)
    )
    energy_use_latest, energy_use_yoy, energy_use_date = _latest_and_offset_pct(
        energy_use_series, "Date", "Arvo", pd.DateOffset(years=1)
    )
    total_use_latest, total_use_yoy, total_use_date = _latest_and_offset_pct(
        total_use_series, "Date", "Arvo", pd.DateOffset(years=1)
    )

    stock_1m_values = []
    stock_1y_values = []

    if stocks_bundle:
        for snap in stocks_bundle.get("snapshots", []):
            if snap.get("1 kk %") is not None and not pd.isna(snap.get("1 kk %")):
                stock_1m_values.append(float(snap.get("1 kk %")))
            if snap.get("1 v %") is not None and not pd.isna(snap.get("1 v %")):
                stock_1y_values.append(float(snap.get("1 v %")))

    stock_1m_avg = sum(stock_1m_values) / len(stock_1m_values) if stock_1m_values else None
    stock_1y_avg = sum(stock_1y_values) / len(stock_1y_values) if stock_1y_values else None

    trade = _build_trade_analysis(months=months)

    indicators = [
        
        {
            "Osa-alue": "Teollinen puukauppa",
            "Muutos": industrial_yoy,
            "Ikoni": _status_from_pct(industrial_yoy)[0],
            "Tila": _status_from_pct(industrial_yoy)[1],
            "Selite": "Puukaupan määrä suhteessa vuoden takaiseen.",
        },
        {
            "Osa-alue": "Metsäteollisuuden vienti",
            "Muutos": trade["export_yoy"],
            "Ikoni": _status_from_pct(trade["export_yoy"])[0],
            "Tila": _status_from_pct(trade["export_yoy"])[1],
            "Selite": "Metsäteollisuuden 12 kk vientisumma.",
        },
        {
            "Osa-alue": "Metsäyhtiöt",
            "Muutos": stock_1y_avg,
            "Ikoni": _status_from_pct(stock_1y_avg)[0],
            "Tila": _status_from_pct(stock_1y_avg)[1],
            "Selite": "Seurattujen metsäyhtiöiden keskimääräinen 1 vuoden kurssimuutos.",
        },
    ]

    usable_scores = []
    for item in indicators:
        pct = item["Muutos"]
        if pct is None or pd.isna(pct):
            continue
        if pct >= 8:
            usable_scores.append(2)
        elif pct >= 2:
            usable_scores.append(1)
        elif pct > -2:
            usable_scores.append(0)
        elif pct > -8:
            usable_scores.append(-1)
        else:
            usable_scores.append(-2)

    avg_score = sum(usable_scores) / len(usable_scores) if usable_scores else None

    if avg_score is None:
        cycle_label = "Ei riittävästi dataa"
        cycle_icon = "⚪"
    elif avg_score >= 1.0:
        cycle_label = "Vahva / kasvava vaihe"
        cycle_icon = "🟢"
    elif avg_score >= 0.2:
        cycle_label = "Lievä kasvu"
        cycle_icon = "🟢"
    elif avg_score > -0.4:
        cycle_label = "Vakaa / tasaantuva vaihe"
        cycle_icon = "🟡"
    elif avg_score > -1.0:
        cycle_label = "Hidastuva vaihe"
        cycle_icon = "🟠"
    else:
        cycle_label = "Heikko vaihe"
        cycle_icon = "🔴"

    summary_parts = []

    if wood_price_yoy is not None:
        if wood_price_yoy > 5:
            summary_parts.append("Puun hintataso on nousussa, mikä tukee metsänomistajan näkökulmaa mutta voi kiristää teollisuuden kustannuksia.")
        elif wood_price_yoy < -5:
            summary_parts.append("Puun hintataso on laskenut vuoden takaiseen nähden, mikä viittaa kysynnän tai markkinapaineen hellittämiseen.")
        else:
            summary_parts.append("Puun hintataso näyttää melko vakaalta vuoden takaiseen verrattuna.")

    if industrial_yoy is not None:
        if industrial_yoy > 5:
            summary_parts.append("Teollinen puukauppa on piristynyt, mikä viittaa aktiivisempaan raakapuumarkkinaan.")
        elif industrial_yoy < -5:
            summary_parts.append("Teollinen puukauppa on hidastunut, mikä voi kertoa varovaisemmasta ostokysynnästä.")
        else:
            summary_parts.append("Puukaupan määrä on melko lähellä vuoden takaista tasoa.")


    if trade["export_yoy"] is not None:
        if trade["export_yoy"] > 5:
            summary_parts.append("Metsäteollisuuden vienti on vahvistunut 12 kuukauden tarkastelussa.")
        elif trade["export_yoy"] < -5:
            summary_parts.append("Metsäteollisuuden vienti on heikentynyt 12 kuukauden tarkastelussa.")
        else:
            summary_parts.append("Metsäteollisuuden vienti on pysynyt melko vakaana.")

    if stock_1y_avg is not None:
        if stock_1y_avg > 8:
            summary_parts.append("Metsäyhtiöiden osakkeet hinnoittelevat selvästi parempaa markkinanäkymää.")
        elif stock_1y_avg < -8:
            summary_parts.append("Metsäyhtiöiden osakkeet kertovat varovaisesta tai heikentyneestä markkinatunnelmasta.")
        else:
            summary_parts.append("Metsäyhtiöiden osakemarkkinatunnelma on melko neutraali.")

    summary_parts.append(
        "Hakkuut ja puunkäyttö jätetään pääsuhdannearviosta pois, koska niiden viimeisimmät havainnot päivittyvät viiveellä."
    )

    if not summary_parts:
        summary_parts.append("Metsäsektorista ei saatu riittävästi dataa analyysin muodostamiseen.")

    analysis_indicators = [
        {
            "Osa-alue": "Puun hinnat",
            "Muutos": wood_price_yoy,
            "Ikoni": _status_from_pct(wood_price_yoy)[0],
            "Tila": _status_from_pct(wood_price_yoy)[1],
            "Selite": "Keskimääräinen puunhintataso suhteessa vuoden takaiseen.",
        },
        *indicators,
    ]

    strengths, risks, watchlist = _build_strengths_risks_watchlist(analysis_indicators)

    return {
        "cycle_icon": cycle_icon,
        "cycle_label": cycle_label,
        "cycle_score": avg_score,
        "summary": " ".join(summary_parts),
        "strengths": strengths,
        "risks": risks,
        "watchlist": watchlist,
        "price_breakdown": [
            {
                "Nimi": "Keskimääräinen puunhinta",
                "Muutos": wood_price_yoy,
                "Tila": _status_from_pct(wood_price_yoy)[1],
                "Ikoni": _status_from_pct(wood_price_yoy)[0],
            },
            {
                "Nimi": "Havutukit",
                "Muutos": logs_yoy,
                "Tila": _status_from_pct(logs_yoy)[1],
                "Ikoni": _status_from_pct(logs_yoy)[0],
            },
            {
                "Nimi": "Havukuitupuu",
                "Muutos": pulp_yoy,
                "Tila": _status_from_pct(pulp_yoy)[1],
                "Ikoni": _status_from_pct(pulp_yoy)[0],
            },
            {
                "Nimi": "Koivutukki",
                "Muutos": birch_logs_yoy,
                "Tila": _status_from_pct(birch_logs_yoy)[1],
                "Ikoni": _status_from_pct(birch_logs_yoy)[0],
            },
            {
                "Nimi": "Koivukuitupuu",
                "Muutos": birch_pulp_yoy,
                "Tila": _status_from_pct(birch_pulp_yoy)[1],
                "Ikoni": _status_from_pct(birch_pulp_yoy)[0],
            },
        ],
        "indicators": indicators,
        "trade": trade,
        "metrics": {
            "wood_price_latest": wood_price_latest,
            "wood_price_yoy": wood_price_yoy,
            "wood_price_date": wood_price_date,
            "industrial_latest": industrial_latest,
            "industrial_yoy": industrial_yoy,
            "industrial_date": industrial_date,
            "harvest_latest": harvest_latest,
            "harvest_yoy": harvest_yoy,
            "harvest_date": harvest_date,
            "industry_use_latest": industry_use_latest,
            "industry_use_yoy": industry_use_yoy,
            "industry_use_date": industry_use_date,
            "energy_use_latest": energy_use_latest,
            "energy_use_yoy": energy_use_yoy,
            "energy_use_date": energy_use_date,
            "total_use_latest": total_use_latest,
            "total_use_yoy": total_use_yoy,
            "total_use_date": total_use_date,
            "stock_1m_avg": stock_1m_avg,
            "stock_1y_avg": stock_1y_avg,
            "logs_latest": logs_latest,
            "logs_yoy": logs_yoy,
            "logs_date": logs_date,
            "pulp_latest": pulp_latest,
            "pulp_yoy": pulp_yoy,
            "pulp_date": pulp_date,
            "birch_logs_latest": birch_logs_latest,
            "birch_logs_yoy": birch_logs_yoy,
            "birch_logs_date": birch_logs_date,
            "birch_pulp_latest": birch_pulp_latest,
            "birch_pulp_yoy": birch_pulp_yoy,
            "birch_pulp_date": birch_pulp_date,
        },
    }