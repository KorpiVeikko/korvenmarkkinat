from __future__ import annotations

import pandas as pd

from services.macro_pxweb_common import (
    add_quarter_date,
    find_time_code,
    get_px_meta,
    pick_value,
    post_px,
)

ATI_Q_URL = "https://pxdata.stat.fi/PXWeb/api/v1/fi/StatFin/ati/statfin_ati_pxt_14um.px"
ATI_WAGES_Q_URL = "https://pxdata.stat.fi/PXWeb/api/v1/fi/StatFin/ati/statfin_ati_pxt_14uv.px"


def _find_var_code(meta: dict, candidates: list[str]) -> str | None:
    for var in meta.get("variables") or []:
        code = str(var.get("code", ""))
        code_l = code.strip().lower()
        if any(token in code_l for token in candidates):
            return code
    return None


def _resolve_sector_column(df: pd.DataFrame, preferred_code: str | None) -> str | None:
    if preferred_code and preferred_code in df.columns:
        return preferred_code

    for col in df.columns:
        col_l = str(col).lower()
        if "sector" in col_l or "sektori" in col_l:
            return col

    return None


def fetch_wage_level_sector_quarterly() -> pd.DataFrame:
    """
    columns:
      Date, Sector, wage_eur
    """
    meta = get_px_meta(ATI_WAGES_Q_URL)
    if not (meta.get("variables") or []):
        return pd.DataFrame()

    time_code = find_time_code(meta) or "Vuosineljännes"
    sector_code = _find_var_code(meta, ["sector", "sektori"])
    gender_code = _find_var_code(meta, ["gender", "sukupuoli"])
    info_code = _find_var_code(meta, ["information", "tiedot"])

    info_val = pick_value(meta, info_code, ["average monthly earnings", "keskiansiot"], fallback_first=True)
    gender_val = pick_value(meta, gender_code, ["total", "yhteensä", "yhteensa"], fallback_first=True)

    query = {"query": [], "response": {"format": "json-stat2"}}

    if sector_code:
        query["query"].append({"code": sector_code, "selection": {"filter": "all", "values": ["*"]}})
    if gender_code:
        query["query"].append(
            {"code": gender_code, "selection": {"filter": "item", "values": [gender_val] if gender_val else ["*"]}}
        )
    query["query"].append({"code": time_code, "selection": {"filter": "all", "values": ["*"]}})
    if info_code:
        query["query"].append(
            {"code": info_code, "selection": {"filter": "item", "values": [info_val] if info_val else ["*"]}}
        )

    df = add_quarter_date(post_px(ATI_WAGES_Q_URL, query))
    if df.empty:
        return pd.DataFrame()

    sector_col = _resolve_sector_column(df, sector_code)
    df["Sector"] = df[sector_col].astype(str) if sector_col else "Kaikki"

    return (
        df[["Date", "Sector", "Arvo"]]
        .rename(columns={"Arvo": "wage_eur"})
        .dropna(subset=["Date", "wage_eur"])
        .sort_values(["Sector", "Date"])
        .reset_index(drop=True)
    )


def fetch_wage_index_sector_quarterly() -> pd.DataFrame:
    """
    columns:
      Date, Sector, wage_index, real_wage_index
    """
    meta = get_px_meta(ATI_Q_URL)
    if not (meta.get("variables") or []):
        return pd.DataFrame()

    time_code = find_time_code(meta) or "Vuosineljännes"
    sector_code = _find_var_code(meta, ["sector", "sektori"])
    info_code = _find_var_code(meta, ["information", "tiedot"])

    wage_index_val = pick_value(
        meta,
        info_code,
        ["index of wage and salary earnings", "ansiotasoindeksi"],
        fallback_first=False,
    )
    real_index_val = pick_value(
        meta,
        info_code,
        ["index of real wage and salary earnings", "reaaliansio"],
        fallback_first=False,
    )

    frames: list[pd.DataFrame] = []

    for chosen_val, out_name in [
        (wage_index_val, "wage_index"),
        (real_index_val, "real_wage_index"),
    ]:
        if not chosen_val:
            continue

        query = {"query": [], "response": {"format": "json-stat2"}}

        if sector_code:
            query["query"].append({"code": sector_code, "selection": {"filter": "all", "values": ["*"]}})
        query["query"].append({"code": time_code, "selection": {"filter": "all", "values": ["*"]}})
        if info_code:
            query["query"].append({"code": info_code, "selection": {"filter": "item", "values": [chosen_val]}})

        df = add_quarter_date(post_px(ATI_Q_URL, query))
        if df.empty:
            continue

        sector_col = _resolve_sector_column(df, sector_code)
        df["Sector"] = df[sector_col].astype(str) if sector_col else "Kaikki"

        tmp = (
            df[["Date", "Sector", "Arvo"]]
            .rename(columns={"Arvo": out_name})
            .dropna(subset=["Date", out_name])
            .sort_values(["Sector", "Date"])
        )
        frames.append(tmp)

    if not frames:
        return pd.DataFrame()

    out = frames[0]
    for frame in frames[1:]:
        out = pd.merge(out, frame, on=["Date", "Sector"], how="outer")

    return out.sort_values(["Sector", "Date"]).reset_index(drop=True)


def build_wage_panel() -> pd.DataFrame:
    levels = fetch_wage_level_sector_quarterly()
    indexes = fetch_wage_index_sector_quarterly()

    if levels.empty and indexes.empty:
        return pd.DataFrame()

    if levels.empty:
        out = indexes.copy()
    elif indexes.empty:
        out = levels.copy()
    else:
        out = pd.merge(levels, indexes, on=["Date", "Sector"], how="outer")

    out = out.sort_values(["Sector", "Date"]).reset_index(drop=True)

    if "wage_eur" in out.columns:
        out["wage_yoy_pct"] = out.groupby("Sector")["wage_eur"].pct_change(4) * 100

    if "wage_index" in out.columns:
        out["wage_index_yoy_pct"] = out.groupby("Sector")["wage_index"].pct_change(4) * 100

    if "real_wage_index" in out.columns:
        out["real_wage_yoy_pct"] = out.groupby("Sector")["real_wage_index"].pct_change(4) * 100

    return out


def sector_options(df: pd.DataFrame) -> list[str]:
    if df is None or df.empty or "Sector" not in df.columns:
        return []

    vals = sorted(df["Sector"].dropna().astype(str).unique().tolist())

    preferred = [
        "Total economy",
        "Koko kansantalous",
        "Total",
        "Yhteensä",
        "General government",
        "Julkinen sektori",
        "Private sector",
        "Yksityinen sektori",
    ]

    ordered: list[str] = []

    for item in preferred:
        if item in vals and item not in ordered:
            ordered.append(item)

    for item in vals:
        if item not in ordered:
            ordered.append(item)

    return ordered