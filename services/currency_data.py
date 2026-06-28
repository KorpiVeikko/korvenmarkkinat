# services/currency_data.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from io import StringIO, BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv
from urllib.parse import urljoin
import re

load_dotenv()

ECB_API_BASE = "https://data-api.ecb.europa.eu/service/data/EXR"
ECB_DATA_API_BASE = "https://data-api.ecb.europa.eu/service/data"
FRED_CSV_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"

FRED_API_BASE = "https://api.stlouisfed.org/fred/series/observations"
FRED_API_KEY = os.getenv("FRED_API_KEY", "")

DATA_CACHE_DIR = Path("data_cache")
FRED_CACHE_DIR = DATA_CACHE_DIR / "fred"

HTTP_CONNECT_TIMEOUT = 4
HTTP_READ_TIMEOUT = 8
HTTP_RETRY_TOTAL = 0

CURRENCY_META: dict[str, dict[str, str | None]] = {
    "EUR": {"name": "Euro", "country": None},
    "USD": {"name": "Yhdysvaltain dollari", "country": "USA"},
    "JPY": {"name": "Japanin jeni", "country": "JPN"},
    "GBP": {"name": "Englannin punta", "country": "GBR"},
    "CHF": {"name": "Sveitsin frangi", "country": "CHE"},
    "CAD": {"name": "Kanadan dollari", "country": "CAN"},
    "AUD": {"name": "Australian dollari", "country": "AUS"},
    "CNY": {"name": "Kiinan juan", "country": "CHN"},
    "SEK": {"name": "Ruotsin kruunu", "country": "SWE"},
    "NOK": {"name": "Norjan kruunu", "country": "NOR"},
    "INR": {"name": "Intian rupia", "country": "IND"},
}

MAJOR_MACRO_CURRENCIES: list[str] = ["USD", "EUR", "JPY", "CNY"]

FRED_SERIES = {
    "USD_M2": "M2SL",
    "USD_CPI": "CPIAUCSL",
    "USD_POLICY": "FEDFUNDS",
    "EUR_HICP": "CP0000EZ19M086NEST",
    "EUR_POLICY": "ECBDFR",
    "FED_ASSETS": "WALCL",
    "ECB_ASSETS": "ECBASSETSW",
    "BOJ_ASSETS": "JPNASSETS",
    
}

ECB_M3_DATASET = "BSI"
ECB_M3_KEY = "M.U2.Y.V.M30.X.1.U2.2300.Z01.E"

OECD_PRICES_URL = "https://sdmx.oecd.org/public/rest/data/OECD.SDD.TPS,DSD_PRICES@DF_PRICES_ALL/all"
OECD_FINMARK_URL = "https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_STES@DF_FINMARK/all"
OECD_MONAGG_URL = "https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_STES@DF_MONAGG/all"

BOJ_API_BASE = "https://www.stat-search.boj.or.jp/api/v1"

BOJ_JPY_M2_DB = "MD02"
BOJ_JPY_M2_CODE = "MAM1NAM2M2MO"

BOJ_JPY_CALL_RATE_DB = "FM01"
BOJ_JPY_CALL_RATE_CODE = "STRDCLUCON"



@dataclass
class FxMetrics:
    currency: str
    currency_name: str
    latest_rate: float | None
    latest_date: pd.Timestamp | None
    change_1y_pct: float | None
    change_5y_pct: float | None
    change_10y_pct: float | None
    ytd_pct: float | None
    volatility_1y_pct: float | None
    min_10y: float | None
    max_10y: float | None


def get_major_macro_currencies() -> list[str]:
    return [c for c in MAJOR_MACRO_CURRENCIES if c in CURRENCY_META]


def is_major_macro_currency(currency: str) -> bool:
    return currency in get_major_macro_currencies()


def _build_session() -> requests.Session:
    session = requests.Session()

    retry = Retry(
        total=HTTP_RETRY_TOTAL,
        connect=HTTP_RETRY_TOTAL,
        read=HTTP_RETRY_TOTAL,
        backoff_factor=0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )

    adapter = HTTPAdapter(max_retries=retry, pool_connections=2, pool_maxsize=2)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (compatible; TaloudenSeuranta/1.0)",
            "Accept": "text/csv,application/json,text/plain,*/*",
        }
    )
    return session


def _empty_fx_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["Date", "Rate", "Currency"])


def _empty_money_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["Date", "Year", "BroadMoney_LCU", "BroadMoney_GrowthPct", "BroadMoney_GDPPct", "Currency"]
    )


def _empty_macro_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["Date", "Year", "CPI_Index", "InflationCPI_Pct", "PolicyRate_Pct", "RealInterestRate_Pct", "Currency"]
    )

def _ensure_macro_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    required = [
        "Date",
        "Year",
        "CPI_Index",
        "InflationCPI_Pct",
        "PolicyRate_Pct",
        "RealInterestRate_Pct",
        "Currency",
    ]

    for col in required:
        if col not in out.columns:
            out[col] = np.nan

    return out


def _build_macro_panel(
    currency: str,
    cpi: pd.DataFrame,
    rate: pd.DataFrame,
    cpi_label: str,
    debug_parts: list[str],
) -> tuple[pd.DataFrame, str | None]:
    if cpi.empty and rate.empty:
        msg = " | ".join(debug_parts) if debug_parts else f"{currency}: inflaatio- ja korkosarjat jäivät tyhjiksi."
        return _empty_macro_df(), msg

    if not cpi.empty:
        cpi = _calc_yoy_from_index(cpi, "Value").rename(
            columns={"Value": "CPI_Index", "YoY_Pct": "InflationCPI_Pct"}
        )
        cpi = _to_month_end(cpi)
    else:
        cpi = pd.DataFrame(columns=["Date", "CPI_Index", "InflationCPI_Pct"])

    if not rate.empty:
        rate = rate.rename(columns={"Value": "PolicyRate_Pct"})
        rate = _to_month_end(rate)
    else:
        rate = pd.DataFrame(columns=["Date", "PolicyRate_Pct"])

    merged = pd.merge(cpi, rate, on="Date", how="outer").sort_values("Date").reset_index(drop=True)

    if merged.empty:
        return _empty_macro_df(), f"{currency}: {cpi_label} ja korko haettiin, mutta yhdistämisen jälkeen ei jäänyt rivejä."

    merged = _ensure_macro_columns(merged)

    merged["Date"] = pd.to_datetime(merged["Date"], errors="coerce")
    merged["CPI_Index"] = pd.to_numeric(merged["CPI_Index"], errors="coerce")
    merged["InflationCPI_Pct"] = pd.to_numeric(merged["InflationCPI_Pct"], errors="coerce")
    merged["PolicyRate_Pct"] = pd.to_numeric(merged["PolicyRate_Pct"], errors="coerce")

    merged["RealInterestRate_Pct"] = merged["PolicyRate_Pct"] - merged["InflationCPI_Pct"]
    merged["Currency"] = currency
    merged["Year"] = merged["Date"].dt.year

    merged = merged.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    if merged.empty:
        return _empty_macro_df(), f"{currency}: makrodata jäi tyhjäksi päivämääräsiivouksen jälkeen."

    return (
        merged[
            [
                "Date",
                "Year",
                "CPI_Index",
                "InflationCPI_Pct",
                "PolicyRate_Pct",
                "RealInterestRate_Pct",
                "Currency",
            ]
        ].copy(),
        " | ".join(debug_parts) if debug_parts else None,
    )


def _empty_metrics(currency: str) -> FxMetrics:
    return FxMetrics(
        currency=currency,
        currency_name=str(CURRENCY_META[currency]["name"]),
        latest_rate=None,
        latest_date=None,
        change_1y_pct=None,
        change_5y_pct=None,
        change_10y_pct=None,
        ytd_pct=None,
        volatility_1y_pct=None,
        min_10y=None,
        max_10y=None,
    )


def _safe_get(url: str, params: dict | None = None, timeout: int | None = None) -> requests.Response:
    session = _build_session()

    read_timeout = HTTP_READ_TIMEOUT if timeout is None else min(int(timeout), HTTP_READ_TIMEOUT)

    try:
        r = session.get(
            url,
            params=params,
            timeout=(HTTP_CONNECT_TIMEOUT, read_timeout),
        )
        r.raise_for_status()
        return r

    except requests.Timeout as e:
        raise requests.Timeout(f"Timeout haussa: {url} params={params} error={e!r}") from e

    except requests.RequestException as e:
        raise requests.RequestException(f"HTTP-haku epäonnistui: {url} params={params} error={e!r}") from e


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lowered = {str(c).strip().lower(): c for c in df.columns}

    for cand in candidates:
        if cand.lower() in lowered:
            return lowered[cand.lower()]

    for col in df.columns:
        col_l = str(col).strip().lower()
        for cand in candidates:
            if cand.lower() in col_l:
                return col

    return None


def _synthetic_eur_series(years: int = 10) -> pd.DataFrame:
    start = pd.Timestamp(date.today().replace(year=date.today().year - years))
    end = pd.Timestamp(date.today())
    dates = pd.date_range(start=start, end=end, freq="B")
    out = pd.DataFrame({"Date": dates, "Rate": 1.0, "Currency": "EUR"})
    return out.reset_index(drop=True)


def _clean_fx_history(df: pd.DataFrame, currency: str) -> pd.DataFrame:
    if df is None or df.empty:
        return _empty_fx_df()

    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["Rate"] = pd.to_numeric(out["Rate"], errors="coerce")
    out = out.dropna(subset=["Date", "Rate"]).sort_values("Date").reset_index(drop=True)
    out["Currency"] = currency

    if out.empty:
        return _empty_fx_df()

    return out[["Date", "Rate", "Currency"]].copy()


def _fred_cache_path(series_id: str) -> Path:
    safe_id = str(series_id).replace("/", "_").replace("\\", "_")
    return FRED_CACHE_DIR / f"{safe_id}.csv"


def _read_fred_cache(series_id: str) -> tuple[pd.DataFrame, str | None]:
    path = _fred_cache_path(series_id)

    if not path.exists():
        return pd.DataFrame(columns=["Date", "Value"]), None

    try:
        df = pd.read_csv(path)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
        df = df.dropna(subset=["Date", "Value"]).sort_values("Date").reset_index(drop=True)

        if df.empty:
            return pd.DataFrame(columns=["Date", "Value"]), None

        return df, f"FRED {series_id}: käytetään välimuistia, koska verkkohaku epäonnistui."

    except Exception as e:
        return pd.DataFrame(columns=["Date", "Value"]), f"FRED {series_id}: välimuistin luku epäonnistui: {e!r}"


def _write_fred_cache(series_id: str, df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return

    try:
        FRED_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        out = df.copy()
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        out["Value"] = pd.to_numeric(out["Value"], errors="coerce")
        out = out.dropna(subset=["Date", "Value"]).sort_values("Date")

        if not out.empty:
            out[["Date", "Value"]].to_csv(_fred_cache_path(series_id), index=False)

    except Exception:
        pass


def _fetch_fred_series(series_id: str) -> tuple[pd.DataFrame, str | None]:
    if not FRED_API_KEY:
        cached_df, cache_msg = _read_fred_cache(series_id)

        if not cached_df.empty:
            return cached_df, "FRED API key puuttuu, käytetään välimuistia."

        return (
            pd.DataFrame(columns=["Date", "Value"]),
            "FRED_API_KEY puuttuu Streamlit Cloud Secrets -asetuksista.",
        )

    try:
        r = _safe_get(
            FRED_API_BASE,
            params={
                "series_id": series_id,
                "api_key": FRED_API_KEY,
                "file_type": "json",
            },
            timeout=8,
        )

        data = r.json()

        observations = data.get("observations", [])

        if not observations:
            cached_df, cache_msg = _read_fred_cache(series_id)

            if not cached_df.empty:
                return cached_df, cache_msg

            return (
                pd.DataFrame(columns=["Date", "Value"]),
                f"FRED {series_id}: observations-lista oli tyhjä.",
            )

        out = pd.DataFrame(
            {
                "Date": pd.to_datetime(
                    [x.get("date") for x in observations],
                    errors="coerce",
                ),
                "Value": pd.to_numeric(
                    [x.get("value") for x in observations],
                    errors="coerce",
                ),
            }
        )

        out = out.dropna(subset=["Date", "Value"]).sort_values("Date").reset_index(drop=True)

        if out.empty:
            cached_df, cache_msg = _read_fred_cache(series_id)

            if not cached_df.empty:
                return cached_df, cache_msg

            return (
                pd.DataFrame(columns=["Date", "Value"]),
                f"FRED {series_id}: JSON-data jäi tyhjäksi siivouksen jälkeen.",
            )

        _write_fred_cache(series_id, out)

        return out, None

    except Exception as e:
        cached_df, cache_msg = _read_fred_cache(series_id)

        if not cached_df.empty:
            return cached_df, cache_msg or f"FRED {series_id}: käytetään välimuistia virheen jälkeen."

        return (
            pd.DataFrame(columns=["Date", "Value"]),
            f"FRED {series_id}: API-haku epäonnistui eikä välimuistia ollut: {e!r}",
        )


def _fetch_ecb_dataset_csv(dataset: str, key: str, start_years: int = 10) -> tuple[pd.DataFrame, str | None]:
    start = date.today().replace(year=date.today().year - start_years)
    url = f"{ECB_DATA_API_BASE}/{dataset}/{key}"

    try:
        r = _safe_get(
            url,
            params={
                "startPeriod": start.isoformat(),
                "format": "csvdata",
            },
            timeout=8,
        )
        text_preview = r.text[:300]
        df = pd.read_csv(StringIO(r.text))

        if df.empty:
            return pd.DataFrame(columns=["Date", "Value"]), f"ECB {dataset}/{key}: CSV tuli, mutta se oli tyhjä."

        time_col = _pick_col(df, ["TIME_PERIOD", "time_period", "date", "time"])
        value_col = _pick_col(df, ["OBS_VALUE", "obs_value", "value"])

        if time_col is None or value_col is None:
            return (
                pd.DataFrame(columns=["Date", "Value"]),
                f"ECB {dataset}/{key}: sarakkeita ei tunnistettu. "
                f"Sarakkeet={list(df.columns)} Vastauksen alku={text_preview!r}",
            )

        out = pd.DataFrame(
            {
                "Date": pd.to_datetime(df[time_col], errors="coerce"),
                "Value": pd.to_numeric(df[value_col], errors="coerce"),
            }
        ).dropna(subset=["Date", "Value"])

        if out.empty:
            return (
                pd.DataFrame(columns=["Date", "Value"]),
                f"ECB {dataset}/{key}: data tuli, mutta siivouksen jälkeen ei jäänyt rivejä. "
                f"Sarakkeet={list(df.columns)} Ensimmäiset rivit={df.head(3).to_dict(orient='records')}",
            )

        return out.sort_values("Date").reset_index(drop=True), None

    except Exception as e:
        return pd.DataFrame(columns=["Date", "Value"]), f"ECB {dataset}/{key}: haku epäonnistui: {e!r}"
    


def _fetch_boj_json(endpoint: str, params: dict | None = None) -> tuple[dict, str | None]:
    url = f"{BOJ_API_BASE}/{endpoint.lstrip('/')}"

    try:
        r = _safe_get(url, params=params or {}, timeout=8)
        return r.json(), None
    except Exception as e:
        return {}, f"BOJ API haku epäonnistui: {url} params={params} error={e!r}"




def _parse_boj_date(value):
    s = str(value).strip()

    if not s or s.lower() in {"nan", "none"}:
        return pd.NaT

    if len(s) == 8 and s.isdigit():
        return pd.to_datetime(s, format="%Y%m%d", errors="coerce")

    if len(s) == 6 and s.isdigit():
        return pd.to_datetime(s + "01", format="%Y%m%d", errors="coerce")

    if len(s) == 4 and s.isdigit():
        return pd.to_datetime(s + "1231", format="%Y%m%d", errors="coerce")

    return pd.to_datetime(s, errors="coerce")


def _clean_boj_timeseries_df(raw: pd.DataFrame, code: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["Date", "Value"])

    d = raw.copy()
    d.columns = [str(c).strip() for c in d.columns]

    date_col = _pick_col(
        d,
        [
            "TIME_PERIOD",
            "time_period",
            "Time",
            "Date",
            "DATE",
            "Period",
            "PERIOD",
        ],
    )

    if date_col is None:
        for col in d.columns:
            c = str(col).strip().lower()
            if (
                "time" in c
                or "date" in c
                or "period" in c
            ) and "update" not in c:
                date_col = col
                break

    if date_col is None:
        return pd.DataFrame(columns=["Date", "Value"])

    value_col = None

    for candidate in [code, "OBS_VALUE", "obs_value", "Value", "VALUE", "value"]:
        if candidate in d.columns:
            value_col = candidate
            break

    if value_col is None:
        best_col = None
        best_count = 0

        ignore = {
            date_col,
            "SERIES_CODE",
            "series_code",
            "NAME_OF_TIME_SERIES",
            "name_of_time_series",
            "UNIT",
            "unit",
            "FREQUENCY",
            "frequency",
            "CATEGORY",
            "category",
            "NOTES",
            "notes",
            "LAST_UPDATE",
            "last_update",
        }

        for col in d.columns:
            if col in ignore:
                continue

            numeric = pd.to_numeric(d[col], errors="coerce")
            count = int(numeric.notna().sum())

            if count > best_count:
                best_count = count
                best_col = col

        value_col = best_col

    if value_col is None:
        return pd.DataFrame(columns=["Date", "Value"])

    out = pd.DataFrame(
        {
            "Date": d[date_col].map(_parse_boj_date),
            "Value": pd.to_numeric(d[value_col], errors="coerce"),
        }
    )

    out = (
        out
        .dropna(subset=["Date", "Value"])
        .sort_values("Date")
        .reset_index(drop=True)
    )

    return out[["Date", "Value"]]


def _fetch_boj_series(
    db: str,
    code: str,
    years: int = 10,
    daily: bool = False,
) -> tuple[pd.DataFrame, str | None]:
    start_year = date.today().year - years - 2
    start_date = f"{start_year}0101" if daily else f"{start_year}01"

    def _parse_nested_values(payload: dict) -> tuple[pd.DataFrame, str | None]:
        rows = payload.get("RESULTSET") or payload.get("resultset") or []
        raw = pd.DataFrame(rows)

        if raw.empty:
            return pd.DataFrame(columns=["Date", "Value"]), "BOJ RESULTSET oli tyhjä."

        if "VALUES" in raw.columns and isinstance(raw.iloc[0]["VALUES"], dict):
            values_obj = raw.iloc[0]["VALUES"]

            dates = (
                values_obj.get("SURVEY_DATES")
                or values_obj.get("DATES")
                or values_obj.get("DATE")
                or []
            )
            values = values_obj.get("VALUES") or []

            out = pd.DataFrame(
                {
                    "Date": [_parse_boj_date(x) for x in dates],
                    "Value": pd.to_numeric(values, errors="coerce"),
                }
            )

            out = (
                out.dropna(subset=["Date", "Value"])
                .sort_values("Date")
                .reset_index(drop=True)
            )

            return out[["Date", "Value"]], None

        out = _clean_boj_timeseries_df(raw, code)
        if out.empty:
            return (
                pd.DataFrame(columns=["Date", "Value"]),
                f"BOJ {db}/{code}: JSON-data jäi tyhjäksi. Sarakkeet={list(raw.columns)}",
            )

        return out, None

    # 1) Ensisijaisesti CSV startDate-parametrilla
    try:
        r = _safe_get(
            f"{BOJ_API_BASE}/getDataCode",
            params={
                "format": "csv",
                "lang": "en",
                "db": db,
                "code": code,
                "startDate": start_date,
            },
            timeout=8,
        )

        raw = pd.read_csv(StringIO(r.text))
        out = _clean_boj_timeseries_df(raw, code)

        if not out.empty:
            return out, None

    except Exception:
        pass

    # 2) JSON startDate-parametrilla
    try:
        payload, msg = _fetch_boj_json(
            "getDataCode",
            params={
                "format": "json",
                "lang": "en",
                "db": db,
                "code": code,
                "startDate": start_date,
            },
        )

        if not msg:
            out, parse_msg = _parse_nested_values(payload)
            if not out.empty:
                return out, None

    except Exception:
        pass

    # 3) BOJ FM01 / päivädata: kokeillaan ilman startDate-parametria
    try:
        payload, msg = _fetch_boj_json(
            "getDataCode",
            params={
                "format": "json",
                "lang": "en",
                "db": db,
                "code": code,
            },
        )

        if msg:
            return pd.DataFrame(columns=["Date", "Value"]), msg

        out, parse_msg = _parse_nested_values(payload)

        if out.empty:
            return pd.DataFrame(columns=["Date", "Value"]), parse_msg

        if daily:
            cutoff = pd.Timestamp(date.today() - pd.DateOffset(years=years))
            out = out[out["Date"] >= cutoff].copy()

        return out.sort_values("Date").reset_index(drop=True), None

    except Exception as e:
        return pd.DataFrame(columns=["Date", "Value"]), f"BOJ {db}/{code}: haku epäonnistui: {e!r}"


def _fetch_jpy_money_panel(years: int = 10) -> tuple[pd.DataFrame, str | None]:
    m2, msg = _fetch_boj_series(
        db=BOJ_JPY_M2_DB,
        code=BOJ_JPY_M2_CODE,
        years=years,
        daily=False,
    )

    if m2.empty:
        fallback, fallback_msg = _fetch_worldbank_money_panel("JPY", years=years)
        return fallback, msg or fallback_msg

    m2 = _to_month_end(m2)
    m2 = m2.rename(columns={"Value": "BroadMoney_LCU"})

    m2["BroadMoney_GrowthPct"] = m2["BroadMoney_LCU"].pct_change(12) * 100.0
    m2["BroadMoney_GDPPct"] = np.nan
    m2["Currency"] = "JPY"
    m2["Year"] = m2["Date"].dt.year

    return (
        m2[
            [
                "Date",
                "Year",
                "BroadMoney_LCU",
                "BroadMoney_GrowthPct",
                "BroadMoney_GDPPct",
                "Currency",
            ]
        ].copy(),
        msg or "JPY M2: Bank of Japan MD02 / Money Stock.",
    )


def _fetch_jpy_macro_panel(years: int = 10) -> tuple[pd.DataFrame, str | None]:
    wb_macro, wb_msg = _fetch_worldbank_macro_panel("JPY", years=years)

    inflation = pd.DataFrame(columns=["Date", "InflationCPI_Pct"])

    if wb_macro is not None and not wb_macro.empty and "InflationCPI_Pct" in wb_macro.columns:
        inflation = (
            wb_macro[["Date", "InflationCPI_Pct"]]
            .dropna(subset=["Date", "InflationCPI_Pct"])
            .copy()
        )
        inflation["Date"] = pd.to_datetime(inflation["Date"], errors="coerce")
        inflation = inflation.dropna(subset=["Date"]).sort_values("Date")

    # BOJ FM01 antaa 400-virheen pitkällä päivähaulla.
    # Terveyskorttia varten riittää tuore korkohavainto.
    rate, rate_msg = _fetch_boj_series(
        db=BOJ_JPY_CALL_RATE_DB,
        code=BOJ_JPY_CALL_RATE_CODE,
        years=years,
        daily=True,
    )

    if rate.empty:
        return (
            wb_macro,
            rate_msg or wb_msg or "JPY: BOJ-korkosarja jäi tyhjäksi, käytetään World Bank -fallbackia.",
        )

    rate = rate.rename(columns={"Value": "PolicyRate_Pct"})
    rate = _to_month_end(rate)

    if not inflation.empty:
        merged = pd.merge_asof(
            rate.sort_values("Date"),
            inflation.sort_values("Date"),
            on="Date",
            direction="backward",
        )
    else:
        merged = rate.copy()
        merged["InflationCPI_Pct"] = np.nan

    merged["CPI_Index"] = np.nan
    merged["PolicyRate_Pct"] = pd.to_numeric(merged["PolicyRate_Pct"], errors="coerce")
    merged["InflationCPI_Pct"] = pd.to_numeric(merged["InflationCPI_Pct"], errors="coerce")
    merged["RealInterestRate_Pct"] = merged["PolicyRate_Pct"] - merged["InflationCPI_Pct"]
    merged["Currency"] = "JPY"
    merged["Year"] = merged["Date"].dt.year

    merged = _ensure_macro_columns(merged)
    merged = (
        merged
        .dropna(subset=["Date"])
        .sort_values("Date")
        .reset_index(drop=True)
    )

    return (
        merged[
            [
                "Date",
                "Year",
                "CPI_Index",
                "InflationCPI_Pct",
                "PolicyRate_Pct",
                "RealInterestRate_Pct",
                "Currency",
            ]
        ].copy(),
        "JPY korko: Bank of Japan FM01 / Uncollateralized Overnight Call Rate. "
        + (wb_msg or "JPY inflaatio: World Bank fallback."),
    )


def _parse_oecd_period(value) -> pd.Timestamp:
    s = str(value).strip()

    if not s or s.lower() in {"nan", "none"}:
        return pd.NaT

    if "Q" in s:
        try:
            return pd.Period(s.replace("-Q", "Q"), freq="Q").end_time.normalize()
        except Exception:
            return pd.NaT

    if len(s) == 7 and s[4] == "-":
        try:
            return pd.Period(s, freq="M").end_time.normalize()
        except Exception:
            return pd.NaT

    if len(s) == 4 and s.isdigit():
        return pd.to_datetime(f"{s}-12-31", errors="coerce")

    return pd.to_datetime(s, errors="coerce")


def _fetch_oecd_csv_table(url: str, start_years: int = 10) -> tuple[pd.DataFrame, str | None]:
    start_year = date.today().year - start_years - 1

    try:
        r = _safe_get(
            url,
            params={
                "startPeriod": str(start_year),
                "dimensionAtObservation": "AllDimensions",
                "format": "csvfilewithlabels",
            },
            timeout=12,
        )
        df = pd.read_csv(StringIO(r.text))
        return df, None
    except Exception as e:
        return pd.DataFrame(), f"OECD-haku epäonnistui: {url} error={e!r}"


def _filter_oecd_china(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    return df[
        (df.get("REF_AREA", pd.Series(dtype=str)).astype(str) == "CHN")
        | (
            df.get("Reference area", pd.Series(dtype=str))
            .astype(str)
            .str.contains("China", case=False, na=False)
        )
    ].copy()


def _fetch_cny_oecd_money_panel(years: int = 10) -> tuple[pd.DataFrame, str | None]:
    df, msg = _fetch_oecd_csv_table(OECD_MONAGG_URL, start_years=years)

    if df.empty:
        return _empty_money_df(), msg or "CNY OECD rahamäärädata jäi tyhjäksi."

    china = _filter_oecd_china(df)

    rows = china[
        (china.get("MEASURE", "").astype(str) == "MABM")
        & (china.get("FREQ", "").astype(str) == "M")
        & (china.get("UNIT_MEASURE", "").astype(str) == "XDC")
    ].copy()

    if rows.empty:
        return _empty_money_df(), "CNY OECD M3 / MABM -sarjaa ei löytynyt."

    rows["Date"] = rows["TIME_PERIOD"].map(_parse_oecd_period)
    rows["BroadMoney_LCU"] = pd.to_numeric(rows["OBS_VALUE"], errors="coerce")

    out = (
        rows[["Date", "BroadMoney_LCU"]]
        .dropna(subset=["Date", "BroadMoney_LCU"])
        .sort_values("Date")
        .reset_index(drop=True)
    )

    out["BroadMoney_GrowthPct"] = out["BroadMoney_LCU"].pct_change(12) * 100.0
    out["BroadMoney_GDPPct"] = np.nan
    out["Currency"] = "CNY"
    out["Year"] = out["Date"].dt.year

    return (
        out[["Date", "Year", "BroadMoney_LCU", "BroadMoney_GrowthPct", "BroadMoney_GDPPct", "Currency"]].copy(),
        "CNY rahamäärä: OECD DF_MONAGG / MABM eli M3.",
    )


def _fetch_cny_oecd_cpi_yoy(years: int = 10) -> tuple[pd.DataFrame, str | None]:
    df, msg = _fetch_oecd_csv_table(OECD_PRICES_URL, start_years=years)

    if df.empty:
        return pd.DataFrame(columns=["Date", "Value"]), msg

    china = _filter_oecd_china(df)

    rows = china[
        (china.get("MEASURE", "").astype(str) == "CPI")
        & (china.get("EXPENDITURE", "").astype(str) == "_T")
        & (china.get("TRANSFORMATION", "").astype(str) == "GY")
        & (china.get("UNIT_MEASURE", "").astype(str) == "PA")
    ].copy()

    if rows.empty:
        return pd.DataFrame(columns=["Date", "Value"]), "CNY OECD CPI YoY -sarjaa ei löytynyt."

    rows["Date"] = rows["TIME_PERIOD"].map(_parse_oecd_period)
    rows["Value"] = pd.to_numeric(rows["OBS_VALUE"], errors="coerce")

    out = rows[["Date", "Value"]].dropna().sort_values("Date").reset_index(drop=True)

    return out, "CNY inflaatio: OECD CPI vuosimuutos."


def _fetch_cny_oecd_rate(years: int = 10, measure: str = "IR3TIB") -> tuple[pd.DataFrame, str | None]:
    df, msg = _fetch_oecd_csv_table(OECD_FINMARK_URL, start_years=years)

    if df.empty:
        return pd.DataFrame(columns=["Date", "Value"]), msg

    china = _filter_oecd_china(df)

    rows = china[
        (china.get("MEASURE", "").astype(str) == measure)
        & (china.get("UNIT_MEASURE", "").astype(str) == "PA")
    ].copy()

    if rows.empty:
        return pd.DataFrame(columns=["Date", "Value"]), f"CNY OECD korkosarjaa ei löytynyt: {measure}"

    rows["Date"] = rows["TIME_PERIOD"].map(_parse_oecd_period)
    rows["Value"] = pd.to_numeric(rows["OBS_VALUE"], errors="coerce")

    out = rows[["Date", "Value"]].dropna().sort_values("Date").reset_index(drop=True)

    return out, f"CNY korko: OECD FINMARK {measure}."


def _fetch_cny_oecd_macro_panel(years: int = 10) -> tuple[pd.DataFrame, str | None]:
    cpi, cpi_msg = _fetch_cny_oecd_cpi_yoy(years=years)
    rate, rate_msg = _fetch_cny_oecd_rate(years=years, measure="IR3TIB")

    debug_parts = [x for x in [cpi_msg, rate_msg] if x]

    if cpi.empty and rate.empty:
        return _empty_macro_df(), " | ".join(debug_parts)

    if not cpi.empty:
        cpi = cpi.rename(columns={"Value": "InflationCPI_Pct"})
        cpi["CPI_Index"] = np.nan
    else:
        cpi = pd.DataFrame(columns=["Date", "CPI_Index", "InflationCPI_Pct"])

    if not rate.empty:
        rate = rate.rename(columns={"Value": "PolicyRate_Pct"})
    else:
        rate = pd.DataFrame(columns=["Date", "PolicyRate_Pct"])

    if not rate.empty and not cpi.empty:
        merged = pd.merge_asof(
            rate.sort_values("Date"),
            cpi[["Date", "CPI_Index", "InflationCPI_Pct"]].sort_values("Date"),
            on="Date",
            direction="backward",
        )
    elif not rate.empty:
        merged = rate.copy()
        merged["CPI_Index"] = np.nan
        merged["InflationCPI_Pct"] = np.nan
    else:
        merged = cpi.copy()
        merged["PolicyRate_Pct"] = np.nan

    merged = _ensure_macro_columns(merged)

    merged["Date"] = pd.to_datetime(merged["Date"], errors="coerce")
    merged["InflationCPI_Pct"] = pd.to_numeric(merged["InflationCPI_Pct"], errors="coerce")
    merged["PolicyRate_Pct"] = pd.to_numeric(merged["PolicyRate_Pct"], errors="coerce")
    merged["RealInterestRate_Pct"] = merged["PolicyRate_Pct"] - merged["InflationCPI_Pct"]
    merged["Currency"] = "CNY"
    merged["Year"] = merged["Date"].dt.year

    merged = merged.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    return (
        merged[[
            "Date",
            "Year",
            "CPI_Index",
            "InflationCPI_Pct",
            "PolicyRate_Pct",
            "RealInterestRate_Pct",
            "Currency",
        ]].copy(),
        " | ".join(debug_parts),
    )






WORLD_BANK_INDICATORS = {
    "BROAD_MONEY_GROWTH": "FM.LBL.BMNY.ZG",
    "INFLATION": "FP.CPI.TOTL.ZG",
    "REAL_INTEREST_RATE": "FR.INR.RINR",
    "LENDING_INTEREST_RATE": "FR.INR.LEND",
}


def _fetch_worldbank_indicator(
    country: str,
    indicator: str,
    years: int = 10,
) -> tuple[pd.DataFrame, str | None]:
    url = f"https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"

    try:
        r = _safe_get(
            url,
            params={
                "format": "json",
                "per_page": 20000,
            },
            timeout=8,
        )

        payload = r.json()

        if not isinstance(payload, list) or len(payload) < 2:
            return pd.DataFrame(columns=["Date", "Year", "Value"]), f"World Bank {country}/{indicator}: vastaus ei ollut odotettu."

        rows = payload[1] or []

        out = pd.DataFrame(
            {
                "Year": pd.to_numeric([x.get("date") for x in rows], errors="coerce"),
                "Value": pd.to_numeric([x.get("value") for x in rows], errors="coerce"),
            }
        )

        out = out.dropna(subset=["Year", "Value"]).copy()
        if out.empty:
            return pd.DataFrame(columns=["Date", "Year", "Value"]), f"World Bank {country}/{indicator}: data jäi tyhjäksi."

        out["Year"] = out["Year"].astype(int)
        min_year = date.today().year - years - 2
        out = out[out["Year"] >= min_year].sort_values("Year").reset_index(drop=True)
        out["Date"] = pd.to_datetime(out["Year"].astype(str) + "-12-31", errors="coerce")

        return out[["Date", "Year", "Value"]], None

    except Exception as e:
        return pd.DataFrame(columns=["Date", "Year", "Value"]), f"World Bank {country}/{indicator}: haku epäonnistui: {e!r}"


def _build_broad_money_index_from_growth(growth_df: pd.DataFrame, currency: str) -> pd.DataFrame:
    if growth_df is None or growth_df.empty:
        return _empty_money_df()

    d = growth_df.copy()
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d["Year"] = pd.to_numeric(d["Year"], errors="coerce")
    d["BroadMoney_GrowthPct"] = pd.to_numeric(d["Value"], errors="coerce")
    d = d.dropna(subset=["Date", "Year", "BroadMoney_GrowthPct"]).sort_values("Date").reset_index(drop=True)

    if d.empty:
        return _empty_money_df()

    index_values = []
    level = 100.0

    for _, row in d.iterrows():
        growth = row["BroadMoney_GrowthPct"]
        if pd.notna(growth):
            level *= 1.0 + float(growth) / 100.0
        index_values.append(level)

    d["BroadMoney_LCU"] = index_values
    d["BroadMoney_GDPPct"] = np.nan
    d["Currency"] = currency

    return d[["Date", "Year", "BroadMoney_LCU", "BroadMoney_GrowthPct", "BroadMoney_GDPPct", "Currency"]].copy()


def _fetch_worldbank_money_panel(currency: str, years: int = 10) -> tuple[pd.DataFrame, str | None]:
    country = CURRENCY_META.get(currency, {}).get("country")
    if not country:
        return _empty_money_df(), f"{currency}: World Bank -maakoodia ei löytynyt."

    growth, msg = _fetch_worldbank_indicator(
        str(country),
        WORLD_BANK_INDICATORS["BROAD_MONEY_GROWTH"],
        years=years,
    )

    if growth.empty:
        return _empty_money_df(), msg or f"{currency}: broad money growth jäi tyhjäksi."

    return _build_broad_money_index_from_growth(growth, currency), msg


def _fetch_worldbank_macro_panel(currency: str, years: int = 10) -> tuple[pd.DataFrame, str | None]:
    country = CURRENCY_META.get(currency, {}).get("country")
    if not country:
        return _empty_macro_df(), f"{currency}: World Bank -maakoodia ei löytynyt."

    inflation, infl_msg = _fetch_worldbank_indicator(
        str(country),
        WORLD_BANK_INDICATORS["INFLATION"],
        years=years,
    )

    real_rate, real_msg = _fetch_worldbank_indicator(
        str(country),
        WORLD_BANK_INDICATORS["REAL_INTEREST_RATE"],
        years=years,
    )

    lending_rate, lend_msg = _fetch_worldbank_indicator(
        str(country),
        WORLD_BANK_INDICATORS["LENDING_INTEREST_RATE"],
        years=years,
    )

    frames = []

    if not inflation.empty:
        frames.append(
            inflation.rename(columns={"Value": "InflationCPI_Pct"})[
                ["Date", "Year", "InflationCPI_Pct"]
            ]
        )

    if not lending_rate.empty:
        frames.append(
            lending_rate.rename(columns={"Value": "PolicyRate_Pct"})[
                ["Date", "Year", "PolicyRate_Pct"]
            ]
        )

    if not real_rate.empty:
        frames.append(
            real_rate.rename(columns={"Value": "RealInterestRate_Pct"})[
                ["Date", "Year", "RealInterestRate_Pct"]
            ]
        )

    if not frames:
        debug = " | ".join(x for x in [infl_msg, real_msg, lend_msg] if x)
        return _empty_macro_df(), debug or f"{currency}: World Bank -makrodata jäi tyhjäksi."

    out = frames[0]
    for frame in frames[1:]:
        out = pd.merge(out, frame, on=["Date", "Year"], how="outer")

    out["CPI_Index"] = np.nan
    out["Currency"] = currency

    out = _ensure_macro_columns(out)
    out = out.sort_values("Date").reset_index(drop=True)

    debug = " | ".join(x for x in [infl_msg, real_msg, lend_msg] if x)

    return (
        out[
            [
                "Date",
                "Year",
                "CPI_Index",
                "InflationCPI_Pct",
                "PolicyRate_Pct",
                "RealInterestRate_Pct",
                "Currency",
            ]
        ].copy(),
        debug or "World Bank -data on vuositasoista. PolicyRate_Pct on tässä lainakorko-proxy, ei varsinainen ohjauskorko.",
    )



def _calc_yoy_from_index(df: pd.DataFrame, value_col: str = "Value") -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=["Date", value_col]).sort_values("Date").reset_index(drop=True)

    if d.empty:
        return pd.DataFrame()

    d["YoY_Pct"] = d[value_col].pct_change(12) * 100.0
    return d


def _to_month_end(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    if d.empty:
        return pd.DataFrame()

    d[date_col] = d[date_col].dt.to_period("M").dt.to_timestamp("M")
    d = d.groupby(date_col, as_index=False).tail(1).sort_values(date_col).reset_index(drop=True)
    return d



def fetch_ecb_fx_series(currency: str, years: int = 10) -> pd.DataFrame:
    if currency == "EUR":
        return _synthetic_eur_series(years=years)

    start = date.today().replace(year=date.today().year - years)
    series_key = f"D.{currency}.EUR.SP00.A"
    url = f"{ECB_API_BASE}/{series_key}"

    params = {
        "startPeriod": start.isoformat(),
        "format": "csvdata",
    }

    try:
        txt = _safe_get(url, params=params, timeout=8).text
    except requests.RequestException:
        return _empty_fx_df()

    try:
        df = pd.read_csv(StringIO(txt))
    except Exception:
        return _empty_fx_df()

    if df.empty:
        return _empty_fx_df()

    time_col = _pick_col(df, ["TIME_PERIOD", "time_period", "date", "time"])
    value_col = _pick_col(df, ["OBS_VALUE", "obs_value", "value", "rate"])

    if time_col is None or value_col is None:
        return _empty_fx_df()

    out = pd.DataFrame(
        {
            "Date": pd.to_datetime(df[time_col], errors="coerce"),
            "Rate": pd.to_numeric(df[value_col], errors="coerce"),
        }
    )

    return _clean_fx_history(out, currency)


def fetch_money_supply_panel(currency: str, years: int = 10) -> tuple[pd.DataFrame, str | None]:
    if not is_major_macro_currency(currency):
        return _empty_money_df(), f"{currency}: rahamäärädata rajattu vain päävaluutoille."

    if currency == "USD":
        m2, msg = _fetch_fred_series(FRED_SERIES["USD_M2"])
        if m2.empty:
            return _empty_money_df(), msg or "USD M2 -sarja jäi tyhjäksi."

        m2 = _to_month_end(m2)
        m2 = m2.rename(columns={"Value": "BroadMoney_LCU"})
        m2["BroadMoney_GrowthPct"] = m2["BroadMoney_LCU"].pct_change(12) * 100.0
        m2["BroadMoney_GDPPct"] = np.nan
        m2["Currency"] = currency
        m2["Year"] = m2["Date"].dt.year

        return m2[
            ["Date", "Year", "BroadMoney_LCU", "BroadMoney_GrowthPct", "BroadMoney_GDPPct", "Currency"]
        ].copy(), msg

    if currency == "EUR":
        m3, msg = _fetch_ecb_dataset_csv(ECB_M3_DATASET, ECB_M3_KEY, start_years=years)
        if m3.empty:
            return _empty_money_df(), msg or "EUR M3 -sarja jäi tyhjäksi."

        m3 = _to_month_end(m3)
        m3 = m3.rename(columns={"Value": "BroadMoney_LCU"})
        m3["BroadMoney_GrowthPct"] = m3["BroadMoney_LCU"].pct_change(12) * 100.0
        m3["BroadMoney_GDPPct"] = np.nan
        m3["Currency"] = currency
        m3["Year"] = m3["Date"].dt.year

        return m3[
            ["Date", "Year", "BroadMoney_LCU", "BroadMoney_GrowthPct", "BroadMoney_GDPPct", "Currency"]
        ].copy(), msg

    if currency == "JPY":
        return _fetch_jpy_money_panel(years=years)
    
    if currency == "CNY":
        return _fetch_cny_oecd_money_panel(years=years)

    return _fetch_worldbank_money_panel(currency, years=years)


def fetch_macro_context_panel(currency: str, years: int = 10) -> tuple[pd.DataFrame, str | None]:
    if not is_major_macro_currency(currency):
        return _empty_macro_df(), f"{currency}: makrodata rajattu vain päävaluutoille."

    if currency == "USD":
        cpi, cpi_msg = _fetch_fred_series(FRED_SERIES["USD_CPI"])
        rate, rate_msg = _fetch_fred_series(FRED_SERIES["USD_POLICY"])

        debug_parts = [x for x in [cpi_msg, rate_msg] if x]

        return _build_macro_panel(
            currency=currency,
            cpi=cpi,
            rate=rate,
            cpi_label="CPI",
            debug_parts=debug_parts,
        )

    if currency == "EUR":
        hicp, hicp_msg = _fetch_fred_series(FRED_SERIES["EUR_HICP"])
        rate, rate_msg = _fetch_fred_series(FRED_SERIES["EUR_POLICY"])

        debug_parts = [x for x in [hicp_msg, rate_msg] if x]

        return _build_macro_panel(
            currency=currency,
            cpi=hicp,
            rate=rate,
            cpi_label="HICP",
            debug_parts=debug_parts,
        )

    if currency == "JPY":
        return _fetch_jpy_macro_panel(years=years)
    
    if currency == "CNY":
        return _fetch_cny_oecd_macro_panel(years=years)

    return _fetch_worldbank_macro_panel(currency, years=years)


CENTRAL_BANK_BALANCE_META = {
    "FED": {
        "name": "Federal Reserve",
        "currency": "USD",
        "series": "FED_ASSETS",
        "unit": "milj. USD",
    },
    "ECB": {
        "name": "Euroopan keskuspankki / Eurosystem",
        "currency": "EUR",
        "series": "ECB_ASSETS",
        "unit": "milj. EUR",
    },
    "BOJ": {
        "name": "Bank of Japan",
        "currency": "JPY",
        "series": "BOJ_ASSETS",
        "unit": "100 milj. JPY",
    },
}


def fetch_central_bank_balance_sheets(years: int = 10) -> tuple[pd.DataFrame, str | None]:
    frames = []
    debug_parts = []

    for bank, meta in CENTRAL_BANK_BALANCE_META.items():
        series_id = FRED_SERIES[meta["series"]]
        df, msg = _fetch_fred_series(series_id)

        if msg:
            debug_parts.append(f"{bank}: {msg}")

        if df.empty:
            continue

        out = df.copy()
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        out["Assets"] = pd.to_numeric(out["Value"], errors="coerce")
        out = out.dropna(subset=["Date", "Assets"]).sort_values("Date")

        cutoff = pd.Timestamp(date.today() - pd.DateOffset(years=years))
        out = out[out["Date"] >= cutoff].copy()

        out["CentralBank"] = bank
        out["Name"] = meta["name"]
        out["Currency"] = meta["currency"]
        out["Unit"] = meta["unit"]

        periods_1y = 52 if bank in {"FED", "ECB"} else 12
        periods_5y = 260 if bank in {"FED", "ECB"} else 60

        out["Assets_Change_1Y_Pct"] = out["Assets"].pct_change(periods_1y) * 100.0
        out["Assets_Change_5Y_Pct"] = out["Assets"].pct_change(periods_5y) * 100.0

        frames.append(
            out[
                [
                    "Date",
                    "CentralBank",
                    "Name",
                    "Currency",
                    "Unit",
                    "Assets",
                    "Assets_Change_1Y_Pct",
                    "Assets_Change_5Y_Pct",
                ]
            ]
        )

    # PBOC lisätään FRED-silmukan jälkeen, ei sen sisällä.
    
    if not frames:
        return pd.DataFrame(), " | ".join(debug_parts) if debug_parts else "Keskuspankkien tasedataa ei saatu."

    return pd.concat(frames, ignore_index=True), " | ".join(debug_parts) if debug_parts else None


def _latest_rate(history: pd.DataFrame) -> tuple[pd.Timestamp | None, float | None]:
    if history is None or history.empty:
        return None, None

    d = history.copy()
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d["Rate"] = pd.to_numeric(d["Rate"], errors="coerce")
    d = d.dropna(subset=["Date", "Rate"]).sort_values("Date")

    if d.empty:
        return None, None

    latest_row = d.iloc[-1]
    return pd.to_datetime(latest_row["Date"]), float(latest_row["Rate"])


def _closest_rate_before_or_on(history: pd.DataFrame, target_date: pd.Timestamp) -> float | None:
    if history is None or history.empty:
        return None

    d = history.copy()
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d["Rate"] = pd.to_numeric(d["Rate"], errors="coerce")
    d = d.dropna(subset=["Date", "Rate"]).sort_values("Date")

    if d.empty:
        return None

    older = d[d["Date"] <= target_date]
    if older.empty:
        return None

    return float(older.iloc[-1]["Rate"])


def _pct_change_from_nearest(history: pd.DataFrame, years_back: int) -> float | None:
    latest_date, latest_rate = _latest_rate(history)
    if latest_date is None or latest_rate is None:
        return None

    target_date = latest_date - pd.DateOffset(years=years_back)
    older_rate = _closest_rate_before_or_on(history, target_date)

    if older_rate is None or older_rate == 0:
        return None

    return ((latest_rate / older_rate) - 1.0) * 100.0


def _ytd_change(history: pd.DataFrame) -> float | None:
    latest_date, latest_rate = _latest_rate(history)
    if latest_date is None or latest_rate is None:
        return None

    year_start = pd.Timestamp(year=latest_date.year, month=1, day=1)
    first_rate = _closest_rate_before_or_on(history, year_start)

    if first_rate is None or first_rate == 0:
        return None

    return ((latest_rate / first_rate) - 1.0) * 100.0


def compute_fx_metrics(history: pd.DataFrame, currency: str) -> FxMetrics:
    if history is None or history.empty:
        return _empty_metrics(currency)

    d = history.copy()
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d["Rate"] = pd.to_numeric(d["Rate"], errors="coerce")
    d = d.dropna(subset=["Date", "Rate"]).sort_values("Date").reset_index(drop=True)

    if d.empty:
        return _empty_metrics(currency)

    latest_date = d["Date"].max()
    latest_rate = float(d.loc[d["Date"] == latest_date, "Rate"].iloc[-1])

    one_year_cutoff = latest_date - pd.DateOffset(years=1)
    hist_1y = d[d["Date"] >= one_year_cutoff].copy()
    hist_1y["ret"] = hist_1y["Rate"].pct_change()
    vol_1y = hist_1y["ret"].std() * np.sqrt(252) * 100 if len(hist_1y) > 10 else None

    return FxMetrics(
        currency=currency,
        currency_name=str(CURRENCY_META[currency]["name"]),
        latest_rate=latest_rate,
        latest_date=latest_date,
        change_1y_pct=_pct_change_from_nearest(d, 1),
        change_5y_pct=_pct_change_from_nearest(d, 5),
        change_10y_pct=_pct_change_from_nearest(d, 10),
        ytd_pct=_ytd_change(d),
        volatility_1y_pct=float(vol_1y) if vol_1y is not None and not pd.isna(vol_1y) else None,
        min_10y=float(d["Rate"].min()) if not d.empty else None,
        max_10y=float(d["Rate"].max()) if not d.empty else None,
    )


def fetch_currency_bundle(currency: str, years: int = 10) -> dict[str, Any]:
    fx = fetch_ecb_fx_series(currency, years=years)
    money, money_debug = fetch_money_supply_panel(currency, years=years)
    macro, macro_debug = fetch_macro_context_panel(currency, years=years)
    metrics = compute_fx_metrics(fx, currency)

    return {
        "fx": fx,
        "money": money,
        "macro": macro,
        "metrics": metrics,
        "debug": {
            "money": money_debug,
            "macro": macro_debug,
        },
    }


def fetch_major_currency_overview(years: int = 10) -> pd.DataFrame:
    rows = []

    for code, meta in CURRENCY_META.items():
        try:
            fx = fetch_ecb_fx_series(code, years=years)
            m = compute_fx_metrics(fx, code)

            rows.append(
                {
                    "Valuutta": code,
                    "Nimi": m.currency_name,
                    "Nykykurssi": m.latest_rate,
                    "Päivä": m.latest_date.date() if m.latest_date is not None else None,
                    "YTD %": m.ytd_pct,
                    "1v %": m.change_1y_pct,
                    "5v %": m.change_5y_pct,
                    "10v %": m.change_10y_pct,
                    "Volatiliteetti 1v %": m.volatility_1y_pct,
                    "10v min": m.min_10y,
                    "10v max": m.max_10y,
                }
            )
        except Exception:
            rows.append(
                {
                    "Valuutta": code,
                    "Nimi": str(meta["name"]),
                    "Nykykurssi": np.nan,
                    "Päivä": None,
                    "YTD %": np.nan,
                    "1v %": np.nan,
                    "5v %": np.nan,
                    "10v %": np.nan,
                    "Volatiliteetti 1v %": np.nan,
                    "10v min": np.nan,
                    "10v max": np.nan,
                }
            )

    return pd.DataFrame(rows)


def generate_ai_summary(
    currency: str,
    anchor_currency: str,
    metrics: FxMetrics,
    money_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    real_fx_proxy_df: pd.DataFrame | None = None,
) -> str:
    name = str(CURRENCY_META[currency]["name"])
    anchor_name = str(CURRENCY_META[anchor_currency]["name"]) if anchor_currency in CURRENCY_META else anchor_currency

    parts: list[str] = []

    if metrics.latest_rate is not None and metrics.latest_date is not None:
        parts.append(
            f"{name} noteerattiin viimeksi tasolla {metrics.latest_rate:.4f} per {anchor_currency} "
            f"({metrics.latest_date.date()})."
        )

    if metrics.change_1y_pct is not None:
        if metrics.change_1y_pct > 3:
            parts.append(f"Valuutta on heikentynyt selvästi viimeisen vuoden aikana {anchor_name.lower()}a vastaan.")
        elif metrics.change_1y_pct < -3:
            parts.append(f"Valuutta on vahvistunut selvästi viimeisen vuoden aikana {anchor_name.lower()}a vastaan.")
        else:
            parts.append(f"Valuutta on liikkunut viimeisen vuoden aikana melko sivuttaisesti {anchor_name.lower()}a vastaan.")

    if metrics.change_5y_pct is not None:
        if metrics.change_5y_pct > 10:
            parts.append("Viiden vuoden kuvassa heikkenemistrendi on ollut selvä.")
        elif metrics.change_5y_pct < -10:
            parts.append("Viiden vuoden kuvassa valuutta on vahvistunut merkittävästi.")
        else:
            parts.append("Viiden vuoden aikajänteellä muutos on ollut melko maltillinen.")

    if metrics.volatility_1y_pct is not None:
        if metrics.volatility_1y_pct > 12:
            parts.append("Kurssivaihtelu on ollut poikkeuksellisen korkeaa.")
        elif metrics.volatility_1y_pct > 7:
            parts.append("Kurssivaihtelu on ollut kohtalaista.")
        else:
            parts.append("Kurssivaihtelu on ollut melko rauhallista.")

    if macro_df is not None and not macro_df.empty:
        d = macro_df.dropna(how="all", subset=["InflationCPI_Pct", "PolicyRate_Pct", "RealInterestRate_Pct"])
        if not d.empty:
            row = d.iloc[-1]
            bits = []

            if pd.notna(row.get("InflationCPI_Pct")):
                bits.append(f"inflaatio oli {row['InflationCPI_Pct']:.1f} %")
            if pd.notna(row.get("PolicyRate_Pct")):
                bits.append(f"ohjauskorko oli {row['PolicyRate_Pct']:.2f} %")
            if pd.notna(row.get("RealInterestRate_Pct")):
                bits.append(f"reaalikorko-proxy oli {row['RealInterestRate_Pct']:.1f} %")

            if bits and pd.notna(row.get("Date")):
                parts.append(
                    f"Tuoreimmassa makrohavainnossa ({pd.to_datetime(row['Date']).date()}) "
                    + " ja ".join(bits)
                    + "."
                )

    if money_df is not None and not money_df.empty:
        d = money_df.dropna(how="all", subset=["BroadMoney_LCU", "BroadMoney_GrowthPct"])
        if not d.empty:
            row = d.iloc[-1]
            bits = []

            if pd.notna(row.get("BroadMoney_GrowthPct")):
                bits.append(f"broad money -kasvu oli {row['BroadMoney_GrowthPct']:.1f} %")

            if bits and pd.notna(row.get("Date")):
                parts.append(
                    f"Tuoreimmassa rahamäärähavainnossa ({pd.to_datetime(row['Date']).date()}) "
                    + " ja ".join(bits)
                    + "."
                )

    if real_fx_proxy_df is not None and not real_fx_proxy_df.empty:
        last = real_fx_proxy_df.iloc[-1]
        if pd.notna(last.get("NominalIndex")) and pd.notna(last.get("RealIndex")):
            gap = float(last["RealIndex"]) - float(last["NominalIndex"])
            if gap > 8:
                parts.append("Inflaatiokorjattuna valuutta näyttää hieman nimellistä kurssia vahvemmalta.")
            elif gap < -8:
                parts.append("Inflaatiokorjattuna valuutta näyttää nimellistä kurssia heikommalta.")
            else:
                parts.append("Inflaatiokorjaus ei muuta kuvaa kovin paljon nimelliseen kurssiin verrattuna.")

    if not parts:
        return "Valitusta valuutasta ei saatu riittävästi dataa yhteenvetoon."

    return " ".join(parts)