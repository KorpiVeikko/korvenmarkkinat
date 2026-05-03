from __future__ import annotations

import math

import pandas as pd
import requests
import streamlit as st

from services.macro_pxweb_common import (
    add_time_columns,
    find_time_code,
    get_px_meta,
    merge_on_date,
    pick_value,
    pick_value_no_fallback,
    post_px,
)
from services.macro_uljas import fetch_exports_products, fetch_imports_products
from services.macro_wages_pxweb import build_wage_panel

CPI_YOY_URL = "https://pxdata.stat.fi/PXWeb/api/v1/fi/StatFin/khi/statfin_khi_pxt_122p.px"
GDP_132H_URL = "https://pxdata.stat.fi/PXWeb/api/v1/fi/StatFin/ntp/statfin_ntp_pxt_132h.px"
LFS_135Z_URL = "https://pxdata.stat.fi/PXWeb/api/v1/fi/StatFin/tyti/statfin_tyti_pxt_135z.px"
EUROSTAT_API = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"
ECB_API_BASE = "https://data-api.ecb.europa.eu/service/data"


def fmt(x: float | None, decimals: int = 1, suffix: str = "") -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:,.{decimals}f}".replace(",", " ") + suffix


def fmt_money(x: float | None) -> str:
    if x is None or pd.isna(x):
        return "—"

    x = float(x)

    if abs(x) >= 1_000_000_000:
        return f"{x / 1_000_000_000:,.1f} mrd €".replace(",", " ")

    return f"{x / 1_000_000:,.0f} milj. €".replace(",", " ")


def fmt_millions(x: float | None, decimals: int = 0) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x / 1_000_000:,.{decimals}f} milj. €".replace(",", " ")


def norm(s: str) -> str:
    s = str(s).lower()
    return s.replace("ä", "a").replace("ö", "o").replace("å", "a")


def clip_by_years(df: pd.DataFrame, date_col: str, years: int) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col])

    if out.empty:
        return out

    end = out[date_col].max()
    start = end - pd.DateOffset(years=int(years))
    return out[out[date_col] >= start].copy()


def yoy_delta(series: pd.Series, periods: int) -> float | None:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) <= periods:
        return None
    return float(s.iloc[-1]) - float(s.iloc[-(periods + 1)])


def latest_valid(df: pd.DataFrame, date_col: str, value_col: str) -> tuple[float | None, pd.Timestamp | None]:
    if df is None or df.empty or value_col not in df.columns:
        return None, None

    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=[date_col, value_col]).sort_values(date_col)

    if d.empty:
        return None, None

    return float(d.iloc[-1][value_col]), pd.to_datetime(d.iloc[-1][date_col])


def latest_row_by_date(df: pd.DataFrame, date_col: str, value_col: str) -> pd.Series | None:
    if df is None or df.empty or value_col not in df.columns:
        return None

    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=[date_col, value_col]).sort_values(date_col)

    if d.empty:
        return None
    return d.iloc[-1]


def pct_change_vs_year_ago(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
) -> tuple[float | None, float | None, pd.Timestamp | None]:
    if df is None or df.empty:
        return None, None, None

    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=[date_col, value_col]).sort_values(date_col)

    if d.empty:
        return None, None, None

    latest_row = d.iloc[-1]
    latest_date = pd.to_datetime(latest_row[date_col])
    latest_val = float(latest_row[value_col])

    prev_year_date = latest_date - pd.DateOffset(years=1)
    prev = d[d[date_col] == prev_year_date]

    if prev.empty:
        return latest_val, None, latest_date

    prev_val = float(prev.iloc[-1][value_col])
    pct = None if prev_val == 0 else ((latest_val / prev_val) - 1.0) * 100.0
    return latest_val, pct, latest_date


def to_yearly(df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=[date_col, value_col])

    if d.empty:
        return pd.DataFrame()

    d["Vuosi"] = d[date_col].dt.year
    yearly = d.groupby("Vuosi", as_index=False)[value_col].sum().sort_values("Vuosi")
    yearly["Miljardia"] = yearly[value_col] / 1_000_000_000
    yearly["Miljoonaa"] = yearly[value_col] / 1_000_000
    return yearly


def latest_full_year_change(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
) -> tuple[float | None, float | None, int | None]:
    yearly = to_yearly(df, date_col, value_col)
    if yearly.empty:
        return None, None, None

    latest_row = yearly.iloc[-1]
    latest_year = int(latest_row["Vuosi"])
    latest_val = float(latest_row[value_col])

    prev = yearly[yearly["Vuosi"] == latest_year - 1]
    if prev.empty:
        pct = None
    else:
        prev_val = float(prev.iloc[0][value_col])
        pct = None if prev_val == 0 else ((latest_val / prev_val) - 1.0) * 100.0

    return latest_val, pct, latest_year


def build_total_flow_from_products(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Aika_dt", value_col])

    return (
        df.groupby("Aika_dt", as_index=False)[value_col]
        .sum()
        .sort_values("Aika_dt")
    )


def build_total_exports_from_products(df: pd.DataFrame) -> pd.DataFrame:
    return build_total_flow_from_products(df, "Vienti_eur")


def build_total_imports_from_products(df: pd.DataFrame) -> pd.DataFrame:
    return build_total_flow_from_products(df, "Tuonti_eur")


def build_trade_balance(exports_df: pd.DataFrame, imports_df: pd.DataFrame) -> pd.DataFrame:
    if (exports_df is None or exports_df.empty) and (imports_df is None or imports_df.empty):
        return pd.DataFrame(columns=["Aika_dt", "Vienti_eur", "Tuonti_eur", "Kauppatase_eur"])

    d = pd.merge(exports_df, imports_df, on="Aika_dt", how="outer").sort_values("Aika_dt")
    d["Vienti_eur"] = pd.to_numeric(d["Vienti_eur"], errors="coerce").fillna(0)
    d["Tuonti_eur"] = pd.to_numeric(d["Tuonti_eur"], errors="coerce").fillna(0)
    d["Kauppatase_eur"] = d["Vienti_eur"] - d["Tuonti_eur"]
    return d


def fetch_inflation_yoy() -> pd.DataFrame:
    meta = get_px_meta(CPI_YOY_URL)
    info_code = "Tiedot"
    time_code = find_time_code(meta) or "Kuukausi"
    info_val = pick_value(meta, info_code, ["vuosimuutos", "year-on-year", "%"], fallback_first=True)

    query = {
        "query": [
            {"code": info_code, "selection": {"filter": "item", "values": [info_val] if info_val else ["*"]}},
            {"code": time_code, "selection": {"filter": "all", "values": ["*"]}},
        ],
        "response": {"format": "json-stat2"},
    }
    return add_time_columns(post_px(CPI_YOY_URL, query))


def build_inflation_series(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    f = df.copy()
    f["Arvo"] = pd.to_numeric(f["Arvo"], errors="coerce")
    f = f.dropna(subset=["Aika_dt", "Arvo"]).sort_values("Aika_dt")

    if f.empty:
        return pd.DataFrame()

    return f[["Aika_dt", "Arvo"]].rename(columns={"Aika_dt": "Date", "Arvo": "inflation_yoy"})


def fetch_gdp_growth_yoy() -> pd.DataFrame:
    meta = get_px_meta(GDP_132H_URL)
    variables = meta.get("variables") or []
    if not variables:
        return pd.DataFrame()

    time_code = find_time_code(meta) or "Vuosineljännes"
    tx_code = "Taloustoimi"
    info_code = "Tiedot"

    tx_val = pick_value(meta, tx_code, ["b1gmh", "bruttokansantuote", "gdp"], fallback_first=True)
    info_yoy = (
        pick_value(meta, info_code, ["%", "edellisestä vuodesta"], fallback_first=False)
        or pick_value(meta, info_code, ["edellisestä vuodesta"], fallback_first=False)
        or pick_value(meta, info_code, ["vuosimuutos", "%"], fallback_first=True)
    )

    query = {
        "query": [
            {"code": tx_code, "selection": {"filter": "item", "values": [tx_val] if tx_val else ["*"]}},
            {"code": info_code, "selection": {"filter": "item", "values": [info_yoy] if info_yoy else ["*"]}},
            {"code": time_code, "selection": {"filter": "all", "values": ["*"]}},
        ],
        "response": {"format": "json-stat2"},
    }

    df = add_time_columns(post_px(GDP_132H_URL, query))
    if df is None or df.empty:
        return pd.DataFrame()

    f = df.copy()
    f["Arvo"] = pd.to_numeric(f["Arvo"], errors="coerce")
    f = f.dropna(subset=["Aika_dt", "Arvo"]).sort_values("Aika_dt")

    if f.empty:
        return pd.DataFrame()

    return f[["Aika_dt", "Arvo"]].rename(columns={"Aika_dt": "Date", "Arvo": "gdp_yoy"})


def _fetch_unemployment_series(
    meta: dict,
    info_needles: list[str],
    kausi_needles: list[str] | None,
    value_name: str,
) -> pd.DataFrame:
    variables = meta.get("variables") or []
    if not variables:
        return pd.DataFrame()

    time_code = find_time_code(meta) or variables[-1].get("code", "Kuukausi")
    query_parts: list[dict] = []

    for var in variables:
        code = var.get("code")
        if not code:
            continue

        norm_code = norm(code)

        if code == time_code:
            query_parts.append({"code": code, "selection": {"filter": "all", "values": ["*"]}})
            continue

        if norm_code == "tiedot":
            chosen = pick_value_no_fallback(meta, code, info_needles)
            if not chosen:
                return pd.DataFrame()
            query_parts.append({"code": code, "selection": {"filter": "item", "values": [chosen]}})
            continue

        if "kausi" in norm_code:
            if kausi_needles:
                chosen = pick_value_no_fallback(meta, code, kausi_needles)
                if not chosen:
                    return pd.DataFrame()
                query_parts.append({"code": code, "selection": {"filter": "item", "values": [chosen]}})
            else:
                chosen = pick_value(meta, code, ["kausitasoitettu", "seasonally adjusted", "sa"], fallback_first=True)
                query_parts.append(
                    {"code": code, "selection": {"filter": "item", "values": [chosen] if chosen else ["*"]}}
                )
            continue

        if "sukupu" in norm_code:
            chosen = pick_value(meta, code, ["yhteensä", "yhteensa", "total", "miehet ja naiset"], fallback_first=True)
            query_parts.append(
                {"code": code, "selection": {"filter": "item", "values": [chosen] if chosen else ["*"]}}
            )
            continue

        if "ika" in norm_code:
            chosen = pick_value(meta, code, ["15–74", "15-74", "15 74", "yhteensä", "yhteensa", "total"], fallback_first=True)
            query_parts.append(
                {"code": code, "selection": {"filter": "item", "values": [chosen] if chosen else ["*"]}}
            )
            continue

        chosen = pick_value(meta, code, ["yhteensä", "yhteensa", "total"], fallback_first=True)
        query_parts.append(
            {"code": code, "selection": {"filter": "item", "values": [chosen] if chosen else ["*"]}}
        )

    query = {"query": query_parts, "response": {"format": "json-stat2"}}
    df = add_time_columns(post_px(LFS_135Z_URL, query))

    if df is None or df.empty:
        return pd.DataFrame()

    f = df.copy()
    f["Arvo"] = pd.to_numeric(f["Arvo"], errors="coerce")
    f = f.dropna(subset=["Aika_dt", "Arvo"]).sort_values("Aika_dt")

    if f.empty:
        return pd.DataFrame()

    return (
        f.groupby("Aika_dt", as_index=False)["Arvo"]
        .mean()
        .rename(columns={"Aika_dt": "Date", "Arvo": value_name})
        .sort_values("Date")
    )


def fetch_unemployment_135z() -> pd.DataFrame:
    meta = get_px_meta(LFS_135Z_URL)
    if not (meta.get("variables") or []):
        return pd.DataFrame()

    rate_sa = _fetch_unemployment_series(
        meta,
        info_needles=["työttömyysaste", "unemployment rate"],
        kausi_needles=["kausitasoitettu", "seasonally adjusted", "sa"],
        value_name="unemployment_rate_sa",
    )
    rate_trend = _fetch_unemployment_series(
        meta,
        info_needles=["työttömyysaste", "unemployment rate"],
        kausi_needles=["trendi", "trend"],
        value_name="unemployment_rate_trend",
    )
    level_sa = _fetch_unemployment_series(
        meta,
        info_needles=["työttömät", "unemployed"],
        kausi_needles=["kausitasoitettu", "seasonally adjusted", "sa"],
        value_name="unemployed_1000_sa",
    )

    return merge_on_date([rate_sa, rate_trend, level_sa])


def build_unemployment_series(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy().sort_values("Date").reset_index(drop=True)

    if "unemployment_rate_sa" in out.columns and "unemployment_rate" not in out.columns:
        out["unemployment_rate"] = out["unemployment_rate_sa"]

    if "unemployed_1000_sa" in out.columns and "unemployed_1000" not in out.columns:
        out["unemployed_1000"] = out["unemployed_1000_sa"]

    return out


def _eurostat_timeseries_to_df(payload: dict, value_name: str) -> pd.DataFrame:
    if not isinstance(payload, dict) or "value" not in payload or "dimension" not in payload:
        return pd.DataFrame()

    dim = payload["dimension"]
    time_id = "time" if "time" in dim else (payload.get("id") or [None])[-1]
    time_cat = dim.get(time_id, {}).get("category", {}) if time_id else {}
    time_index = time_cat.get("index", {})
    time_label = time_cat.get("label", {})

    if isinstance(time_index, dict) and time_index:
        time_keys = [k for k, _ in sorted(time_index.items(), key=lambda kv: kv[1])]
    else:
        time_keys = list(time_label.keys())

    values = payload.get("value", {})
    rows = []
    for i, tk in enumerate(time_keys):
        value = values.get(str(i), values.get(i))
        rows.append({"Period": tk, value_name: pd.to_numeric(value, errors="coerce")})

    df = pd.DataFrame(rows).dropna(subset=[value_name])
    if df.empty:
        return df

    if df["Period"].astype(str).str.fullmatch(r"\d{4}").all():
        df["Date"] = pd.to_datetime(df["Period"].astype(str) + "-01-01", errors="coerce")
    else:
        df["Date"] = pd.PeriodIndex(df["Period"], freq="Q").to_timestamp(how="start")

    return df.sort_values("Date").reset_index(drop=True)


def _eurostat_fetch_json(dataset_code: str, params: dict) -> dict:
    response = requests.get(
        f"{EUROSTAT_API}/{dataset_code}",
        params={"lang": "EN", "format": "JSON", **params},
        timeout=45,
    )
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, dict) else {}


def eurostat_fetch_gov_debt_pct_gdp(geo: str = "FI") -> pd.DataFrame:
    payload = _eurostat_fetch_json(
        "gov_10q_ggdebt",
        {
            "freq": "Q",
            "sector": "S13",
            "na_item": "GD",
            "unit": "PC_GDP",
            "geo": geo,
        },
    )
    return _eurostat_timeseries_to_df(payload, value_name="debt_pct_gdp")


def eurostat_fetch_gov_debt_mio_eur(geo: str = "FI") -> pd.DataFrame:
    payload = _eurostat_fetch_json(
        "gov_10q_ggdebt",
        {
            "freq": "Q",
            "sector": "S13",
            "na_item": "GD",
            "unit": "MIO_EUR",
            "geo": geo,
        },
    )
    return _eurostat_timeseries_to_df(payload, value_name="debt_mio_eur")


def eurostat_fetch_private_debt_ratio(dataset_code: str, value_name: str, geo: str = "FI") -> pd.DataFrame:
    payload = _eurostat_fetch_json(dataset_code, {"geo": geo})
    return _eurostat_timeseries_to_df(payload, value_name=value_name)


def eurostat_fetch_private_loans_mio_nac(
    sector: str,
    value_name: str,
    geo: str = "FI",
) -> tuple[pd.DataFrame, dict]:
    debug: dict = {
        "dataset": "tipspd26",
        "geo": geo,
        "sector": sector,
        "unit": "MIO_NAC",
        "freq": "A",
        "ok": False,
        "rows": 0,
        "columns": [],
        "error": None,
        "sample": [],
        "raw_keys": [],
        "ids": [],
        "sizes": [],
        "dimension_keys": [],
    }

    try:
        payload = _eurostat_fetch_json(
            "tipspd26",
            {
                "geo": geo,
                "sector": sector,
                "unit": "MIO_NAC",
                "freq": "A",
            },
        )

        debug["raw_keys"] = list(payload.keys()) if isinstance(payload, dict) else []
        debug["ids"] = payload.get("id", []) if isinstance(payload, dict) else []
        debug["sizes"] = payload.get("size", []) if isinstance(payload, dict) else []
        debug["dimension_keys"] = list((payload.get("dimension") or {}).keys()) if isinstance(payload, dict) else []

        df = _eurostat_timeseries_to_df(payload, value_name=value_name)

        debug["rows"] = len(df)
        debug["columns"] = df.columns.tolist() if not df.empty else []
        debug["sample"] = df.head(5).to_dict(orient="records") if not df.empty else []
        debug["ok"] = not df.empty

        return df, debug

    except Exception as ex:
        debug["error"] = f"{type(ex).__name__}: {ex}"
        return pd.DataFrame(columns=["Date", value_name]), debug


def _parse_ecb_sdmx_json_time_series(payload: dict, value_name: str) -> tuple[pd.DataFrame, dict]:
    debug: dict = {
        "ok": False,
        "error": None,
        "top_keys": [],
        "dataset_count": 0,
        "series_count": 0,
        "observation_dim_count": 0,
        "time_values_count": 0,
        "sample_time_values": [],
        "sample_series_keys": [],
        "rows_before_dropna": 0,
        "rows_after_dropna": 0,
        "sample_rows": [],
    }

    try:
        if not isinstance(payload, dict):
            debug["error"] = "Payload ei ole dict."
            return pd.DataFrame(columns=["Date", value_name]), debug

        debug["top_keys"] = list(payload.keys())

        data_sets = payload.get("dataSets", [])
        debug["dataset_count"] = len(data_sets)

        if not data_sets:
            debug["error"] = "dataSets puuttuu tai on tyhjä."
            return pd.DataFrame(columns=["Date", value_name]), debug

        series_block = data_sets[0].get("series", {})
        debug["series_count"] = len(series_block)
        debug["sample_series_keys"] = list(series_block.keys())[:5]

        obs_dims = payload.get("structure", {}).get("dimensions", {}).get("observation", [])
        debug["observation_dim_count"] = len(obs_dims)

        if not obs_dims:
            debug["error"] = "Observation-dimensiot puuttuvat."
            return pd.DataFrame(columns=["Date", value_name]), debug

        time_values = obs_dims[0].get("values", [])
        debug["time_values_count"] = len(time_values)
        debug["sample_time_values"] = time_values[:5]

        time_map: dict[str, str] = {}
        for idx, item in enumerate(time_values):
            if isinstance(item, dict):
                time_key = item.get("id") or item.get("name")
            else:
                time_key = str(item)
            time_map[str(idx)] = str(time_key)

        rows: list[dict] = []
        for _, series in series_block.items():
            observations = series.get("observations", {})
            for obs_idx, obs_val in observations.items():
                period = time_map.get(str(obs_idx))
                if period is None:
                    continue

                value = None
                if isinstance(obs_val, list) and obs_val:
                    value = obs_val[0]
                elif isinstance(obs_val, (int, float)):
                    value = obs_val

                rows.append(
                    {
                        "Period": str(period),
                        value_name: pd.to_numeric(value, errors="coerce"),
                    }
                )

        debug["rows_before_dropna"] = len(rows)

        df = pd.DataFrame(rows)
        if df.empty:
            debug["error"] = "Rivejä ei muodostunut."
            return pd.DataFrame(columns=["Date", value_name]), debug

        df[value_name] = pd.to_numeric(df[value_name], errors="coerce")
        df = df.dropna(subset=[value_name]).copy()
        debug["rows_after_dropna"] = len(df)
        debug["sample_rows"] = df.head(5).to_dict(orient="records")

        if df.empty:
            debug["error"] = "Kaikki arvot putosivat pois dropna-vaiheessa."
            return pd.DataFrame(columns=["Date", value_name]), debug

        df["Date"] = pd.to_datetime(df["Period"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date").copy()

        if df.empty:
            debug["error"] = "Päivämäärämuunnos epäonnistui."
            return pd.DataFrame(columns=["Date", value_name]), debug

        df["Year"] = df["Date"].dt.year
        df = (
            df.sort_values("Date")
            .groupby("Year", as_index=False)
            .tail(1)[["Date", value_name]]
            .sort_values("Date")
            .reset_index(drop=True)
        )

        debug["ok"] = not df.empty
        return df, debug

    except Exception as ex:
        debug["error"] = f"{type(ex).__name__}: {ex}"
        return pd.DataFrame(columns=["Date", value_name]), debug


def ecb_fetch_household_loans_mio(geo: str = "FI") -> tuple[pd.DataFrame, dict]:
    debug: dict = {
        "source": "ECB",
        "geo": geo,
        "series_key": None,
        "url": None,
        "http_status": None,
        "ok": False,
        "error": None,
        "parse_debug": {},
    }

    candidate_series = [
        f"M.{geo}.N.A.A20.A.1.U2.2250.Z01.E",
    ]

    for series_key in candidate_series:
        url = f"{ECB_API_BASE}/BSI/{series_key}"
        debug["series_key"] = series_key
        debug["url"] = url

        try:
            response = requests.get(
                url,
                headers={"Accept": "application/json"},
                timeout=45,
            )
            debug["http_status"] = response.status_code
            response.raise_for_status()

            payload = response.json()
            df, parse_debug = _parse_ecb_sdmx_json_time_series(payload, "household_loans_mio")
            debug["parse_debug"] = parse_debug

            if not df.empty:
                debug["ok"] = True
                return df, debug

        except Exception as ex:
            debug["error"] = f"{type(ex).__name__}: {ex}"

    return pd.DataFrame(columns=["Date", "household_loans_mio"]), debug


@st.cache_data(show_spinner="Haetaan inflaatio (Tilastokeskus)…")
def load_inflation() -> pd.DataFrame:
    return build_inflation_series(fetch_inflation_yoy())


@st.cache_data(show_spinner="Haetaan BKT YoY (Tilastokeskus)…")
def load_gdp_yoy() -> pd.DataFrame:
    return fetch_gdp_growth_yoy()


@st.cache_data(show_spinner="Haetaan työttömyys (Tilastokeskus)…")
def load_unemployment() -> pd.DataFrame:
    return build_unemployment_series(fetch_unemployment_135z())


@st.cache_data(show_spinner="Haetaan velka/BKT (Eurostat)…")
def load_debt_pct_gdp() -> pd.DataFrame:
    return eurostat_fetch_gov_debt_pct_gdp("FI")


@st.cache_data(show_spinner="Haetaan velan määrä (Eurostat)…")
def load_debt_mio_eur() -> pd.DataFrame:
    return eurostat_fetch_gov_debt_mio_eur("FI")


@st.cache_data(show_spinner="Haetaan kotitalouksien velka/BKT (Eurostat)…")
def load_household_debt_pct_gdp() -> pd.DataFrame:
    return eurostat_fetch_private_debt_ratio("tipspd22", "household_debt_pct_gdp")


@st.cache_data(show_spinner="Haetaan kotitalouksien velka/tulot (Eurostat)…")
def load_household_debt_pct_gdi() -> pd.DataFrame:
    return eurostat_fetch_private_debt_ratio("tipspd40", "household_debt_pct_gdi")


@st.cache_data(show_spinner="Haetaan yritysvelka/BKT (Eurostat)…")
def load_nfc_debt_pct_gdp() -> pd.DataFrame:
    return eurostat_fetch_private_debt_ratio("tipspd30", "nfc_debt_pct_gdp")


@st.cache_data(show_spinner="Haetaan yksityinen velka/BKT (Eurostat)…")
def load_private_sector_debt_pct_gdp() -> pd.DataFrame:
    return eurostat_fetch_private_debt_ratio("tipspd20", "private_debt_pct_gdp")


@st.cache_data(show_spinner="Haetaan kotitalouksien lainakanta (ECB)…")
def load_household_loans_mio() -> tuple[pd.DataFrame, dict]:
    return ecb_fetch_household_loans_mio("FI")


@st.cache_data(show_spinner="Haetaan yritysten lainakanta (Eurostat)…")
def load_nfc_loans_mio_nac() -> tuple[pd.DataFrame, dict]:
    return eurostat_fetch_private_loans_mio_nac("S11", "nfc_loans_mio_nac")


@st.cache_data(show_spinner="Haetaan palkkadata (Tilastokeskus)…")
def load_wages() -> pd.DataFrame:
    return build_wage_panel()


def load_trade_totals(months: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    exports_products_df, _ = fetch_exports_products(months=months, lang="fi")
    imports_products_df, _ = fetch_imports_products(months=months, lang="fi")

    exports_total_df = build_total_exports_from_products(exports_products_df)
    imports_total_df = build_total_imports_from_products(imports_products_df)

    return exports_total_df, imports_total_df