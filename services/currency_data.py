# services/currency_data.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ECB_API_BASE = "https://data-api.ecb.europa.eu/service/data/EXR"
ECB_DATA_API_BASE = "https://data-api.ecb.europa.eu/service/data"
FRED_CSV_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"

DATA_CACHE_DIR = Path("data_cache")
FRED_CACHE_DIR = DATA_CACHE_DIR / "fred"

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
    "RUB": {"name": "Venäjän rupla", "country": "RUS"},
}

MAJOR_MACRO_CURRENCIES: list[str] = ["USD", "EUR"]

FRED_SERIES = {
    "USD_M2": "M2SL",
    "USD_CPI": "CPIAUCSL",
    "USD_POLICY": "FEDFUNDS",
    "EUR_HICP": "CP0000EZ19M086NEST",
    "EUR_POLICY": "ECBDFR",
}

ECB_M3_DATASET = "BSI"
ECB_M3_KEY = "M.U2.Y.V.M30.X.1.U2.2300.Z01.E"


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
        total=2,
        connect=2,
        read=2,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
    )

    adapter = HTTPAdapter(max_retries=retry)
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


def _safe_get(url: str, params: dict | None = None, timeout: int = 30) -> requests.Response:
    session = _build_session()
    r = session.get(url, params=params, timeout=(8, timeout))
    r.raise_for_status()
    return r


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
    try:
        r = _safe_get(FRED_CSV_BASE, params={"id": series_id}, timeout=60)
        text_preview = r.text[:300]
        df = pd.read_csv(StringIO(r.text))

        if df.empty:
            cached_df, cache_msg = _read_fred_cache(series_id)
            if not cached_df.empty:
                return cached_df, cache_msg
            return pd.DataFrame(columns=["Date", "Value"]), f"FRED {series_id}: CSV tuli, mutta se oli tyhjä."

        date_col = _pick_col(df, ["DATE", "date", "observation_date"])
        value_col = None
        for c in df.columns:
            if str(c).strip().lower() not in {"date", "observation_date"}:
                value_col = c
                break

        if date_col is None or value_col is None:
            cached_df, cache_msg = _read_fred_cache(series_id)
            if not cached_df.empty:
                return cached_df, cache_msg

            return (
                pd.DataFrame(columns=["Date", "Value"]),
                f"FRED {series_id}: sarakkeita ei tunnistettu. Sarakkeet={list(df.columns)} Vastauksen alku={text_preview!r}",
            )

        out = pd.DataFrame(
            {
                "Date": pd.to_datetime(df[date_col], errors="coerce"),
                "Value": pd.to_numeric(df[value_col], errors="coerce"),
            }
        ).dropna(subset=["Date", "Value"])

        if out.empty:
            cached_df, cache_msg = _read_fred_cache(series_id)
            if not cached_df.empty:
                return cached_df, cache_msg

            return (
                pd.DataFrame(columns=["Date", "Value"]),
                f"FRED {series_id}: data tuli, mutta siivouksen jälkeen ei jäänyt rivejä. "
                f"Sarakkeet={list(df.columns)} Ensimmäiset rivit={df.head(3).to_dict(orient='records')}",
            )

        out = out.sort_values("Date").reset_index(drop=True)
        _write_fred_cache(series_id, out)

        return out, None

    except Exception as e:
        cached_df, cache_msg = _read_fred_cache(series_id)
        if not cached_df.empty:
            return cached_df, cache_msg or f"FRED {series_id}: käytetään välimuistia virheen jälkeen."

        return pd.DataFrame(columns=["Date", "Value"]), f"FRED {series_id}: verkkohaku epäonnistui eikä välimuistia ollut: {e!r}"


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
            timeout=60,
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
        txt = _safe_get(url, params=params, timeout=45).text
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

        return m2[["Date", "Year", "BroadMoney_LCU", "BroadMoney_GrowthPct", "BroadMoney_GDPPct", "Currency"]].copy(), msg

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

        return m3[["Date", "Year", "BroadMoney_LCU", "BroadMoney_GrowthPct", "BroadMoney_GDPPct", "Currency"]].copy(), msg

    return _empty_money_df(), f"{currency}: rahamäärädataa ei ole määritelty."


def fetch_macro_context_panel(currency: str, years: int = 10) -> tuple[pd.DataFrame, str | None]:
    if not is_major_macro_currency(currency):
        return _empty_macro_df(), f"{currency}: makrodata rajattu vain päävaluutoille."

    if currency == "USD":
        cpi, cpi_msg = _fetch_fred_series(FRED_SERIES["USD_CPI"])
        rate, rate_msg = _fetch_fred_series(FRED_SERIES["USD_POLICY"])

        debug_parts = [x for x in [cpi_msg, rate_msg] if x]

        if cpi.empty and rate.empty:
            return _empty_macro_df(), " | ".join(debug_parts) if debug_parts else "USD CPI- ja korkosarjat jäivät tyhjiksi."

        if not cpi.empty:
            cpi = _calc_yoy_from_index(cpi, "Value").rename(
                columns={"Value": "CPI_Index", "YoY_Pct": "InflationCPI_Pct"}
            )
            cpi = _to_month_end(cpi)

        if not rate.empty:
            rate = rate.rename(columns={"Value": "PolicyRate_Pct"})
            rate = _to_month_end(rate)

        merged = pd.merge(cpi, rate, on="Date", how="outer").sort_values("Date").reset_index(drop=True)
        if merged.empty:
            return _empty_macro_df(), "USD: CPI ja korko haettiin, mutta yhdistämisen jälkeen ei jäänyt rivejä."

        merged["RealInterestRate_Pct"] = merged["PolicyRate_Pct"] - merged["InflationCPI_Pct"]
        merged["Currency"] = currency
        merged["Year"] = merged["Date"].dt.year

        return (
            merged[["Date", "Year", "CPI_Index", "InflationCPI_Pct", "PolicyRate_Pct", "RealInterestRate_Pct", "Currency"]].copy(),
            " | ".join(debug_parts) if debug_parts else None,
        )

    if currency == "EUR":
        hicp, hicp_msg = _fetch_fred_series(FRED_SERIES["EUR_HICP"])
        rate, rate_msg = _fetch_fred_series(FRED_SERIES["EUR_POLICY"])

        debug_parts = [x for x in [hicp_msg, rate_msg] if x]

        if hicp.empty and rate.empty:
            return _empty_macro_df(), " | ".join(debug_parts) if debug_parts else "EUR HICP- ja korkosarjat jäivät tyhjiksi."

        if not hicp.empty:
            hicp = _calc_yoy_from_index(hicp, "Value").rename(
                columns={"Value": "CPI_Index", "YoY_Pct": "InflationCPI_Pct"}
            )
            hicp = _to_month_end(hicp)

        if not rate.empty:
            rate = rate.rename(columns={"Value": "PolicyRate_Pct"})
            rate = _to_month_end(rate)

        merged = pd.merge(hicp, rate, on="Date", how="outer").sort_values("Date").reset_index(drop=True)
        if merged.empty:
            return _empty_macro_df(), "EUR: HICP ja korko haettiin, mutta yhdistämisen jälkeen ei jäänyt rivejä."

        merged["RealInterestRate_Pct"] = merged["PolicyRate_Pct"] - merged["InflationCPI_Pct"]
        merged["Currency"] = currency
        merged["Year"] = merged["Date"].dt.year

        return (
            merged[["Date", "Year", "CPI_Index", "InflationCPI_Pct", "PolicyRate_Pct", "RealInterestRate_Pct", "Currency"]].copy(),
            " | ".join(debug_parts) if debug_parts else None,
        )

    return _empty_macro_df(), f"{currency}: makrodataa ei ole määritelty."


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