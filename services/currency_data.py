# services/currency_data.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from io import StringIO
from typing import Any

import numpy as np
import pandas as pd
import requests

ECB_API_BASE = "https://data-api.ecb.europa.eu/service/data/EXR"
WORLD_BANK_API = "https://api.worldbank.org/v2"

# 10 suurta / yleisesti seurattua valuuttaa euroon nähden
CURRENCY_META: dict[str, dict[str, str]] = {
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

WB_INDICATORS = {
    "broad_money_lcu": "FM.LBL.BMNY.CN",
    "broad_money_growth": "FM.LBL.BMNY.ZG",
    "broad_money_gdp": "FM.LBL.BMNY.GD.ZS",
}


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


def _safe_get(url: str, params: dict | None = None, timeout: int = 45) -> requests.Response:
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    for c in df.columns:
        cl = str(c).strip().lower()
        for cand in candidates:
            if cand.lower() in cl:
                return c
    return None


def fetch_ecb_fx_series(currency: str, years: int = 10) -> pd.DataFrame:
    """
    ECB euro reference rate series.
    Series key pattern:
      D.<CURRENCY>.EUR.SP00.A
    Interpretation: units of foreign currency per 1 euro.
    """
    start = date.today().replace(year=date.today().year - years)
    series_key = f"D.{currency}.EUR.SP00.A"
    url = f"{ECB_API_BASE}/{series_key}"

    params = {
        "startPeriod": start.isoformat(),
        "format": "csvdata",
    }

    txt = _safe_get(url, params=params).text
    df = pd.read_csv(StringIO(txt))
    if df.empty:
        return pd.DataFrame(columns=["Date", "Rate", "Currency"])

    time_col = _pick_col(df, ["TIME_PERIOD", "time_period", "date", "time"])
    value_col = _pick_col(df, ["OBS_VALUE", "obs_value", "value", "rate"])

    if time_col is None or value_col is None:
        return pd.DataFrame(columns=["Date", "Rate", "Currency"])

    out = pd.DataFrame({
        "Date": pd.to_datetime(df[time_col], errors="coerce"),
        "Rate": pd.to_numeric(df[value_col], errors="coerce"),
    }).dropna(subset=["Date", "Rate"])

    out["Currency"] = currency
    return out.sort_values("Date").reset_index(drop=True)


def _world_bank_indicator(country: str, indicator: str) -> pd.DataFrame:
    url = f"{WORLD_BANK_API}/country/{country}/indicator/{indicator}"
    params = {"format": "json", "per_page": 2000}
    data = _safe_get(url, params=params).json()

    if not isinstance(data, list) or len(data) < 2 or not isinstance(data[1], list):
        return pd.DataFrame(columns=["Year", "Value"])

    rows = []
    for item in data[1]:
        year = item.get("date")
        value = item.get("value")
        if year is None:
            continue
        rows.append({
            "Year": pd.to_numeric(year, errors="coerce"),
            "Value": pd.to_numeric(value, errors="coerce"),
        })

    df = pd.DataFrame(rows).dropna(subset=["Year"])
    if df.empty:
        return df

    df["Year"] = df["Year"].astype(int)
    return df.sort_values("Year").reset_index(drop=True)


def fetch_money_supply_panel(currency: str, years: int = 10) -> pd.DataFrame:
    country = CURRENCY_META[currency]["country"]

    amount = _world_bank_indicator(country, WB_INDICATORS["broad_money_lcu"]).rename(
        columns={"Value": "BroadMoney_LCU"}
    )
    growth = _world_bank_indicator(country, WB_INDICATORS["broad_money_growth"]).rename(
        columns={"Value": "BroadMoney_GrowthPct"}
    )
    share = _world_bank_indicator(country, WB_INDICATORS["broad_money_gdp"]).rename(
        columns={"Value": "BroadMoney_GDPPct"}
    )

    merged = amount.merge(growth, on="Year", how="outer").merge(share, on="Year", how="outer")
    if merged.empty:
        return merged

    cutoff = date.today().year - years
    merged = merged[merged["Year"] >= cutoff].sort_values("Year").reset_index(drop=True)
    merged["Currency"] = currency
    return merged


def _pct_change_from_nearest(history: pd.DataFrame, years_back: int) -> float | None:
    if history.empty:
        return None
    latest_date = history["Date"].max()
    latest_rate = float(history.loc[history["Date"] == latest_date, "Rate"].iloc[-1])

    target_date = latest_date - pd.DateOffset(years=years_back)
    older = history[history["Date"] <= target_date]
    if older.empty:
        return None

    older_rate = float(older.iloc[-1]["Rate"])
    if older_rate == 0:
        return None

    return ((latest_rate / older_rate) - 1.0) * 100.0


def _ytd_change(history: pd.DataFrame) -> float | None:
    if history.empty:
        return None

    latest_date = history["Date"].max()
    latest_rate = float(history.loc[history["Date"] == latest_date, "Rate"].iloc[-1])

    year_start = pd.Timestamp(year=latest_date.year, month=1, day=1)
    first = history[history["Date"] >= year_start]
    if first.empty:
        first = history[history["Date"].dt.year == latest_date.year]
    if first.empty:
        return None

    first_rate = float(first.iloc[0]["Rate"])
    if first_rate == 0:
        return None

    return ((latest_rate / first_rate) - 1.0) * 100.0


def compute_fx_metrics(history: pd.DataFrame, currency: str) -> FxMetrics:
    if history.empty:
        return FxMetrics(
            currency=currency,
            currency_name=CURRENCY_META[currency]["name"],
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

    latest_date = history["Date"].max()
    latest_rate = float(history.loc[history["Date"] == latest_date, "Rate"].iloc[-1])

    one_year_cutoff = latest_date - pd.DateOffset(years=1)
    hist_1y = history[history["Date"] >= one_year_cutoff].copy()
    hist_1y["ret"] = hist_1y["Rate"].pct_change()
    vol_1y = hist_1y["ret"].std() * np.sqrt(252) * 100 if len(hist_1y) > 10 else None

    return FxMetrics(
        currency=currency,
        currency_name=CURRENCY_META[currency]["name"],
        latest_rate=latest_rate,
        latest_date=latest_date,
        change_1y_pct=_pct_change_from_nearest(history, 1),
        change_5y_pct=_pct_change_from_nearest(history, 5),
        change_10y_pct=_pct_change_from_nearest(history, 10),
        ytd_pct=_ytd_change(history),
        volatility_1y_pct=float(vol_1y) if vol_1y is not None and not pd.isna(vol_1y) else None,
        min_10y=float(history["Rate"].min()) if not history.empty else None,
        max_10y=float(history["Rate"].max()) if not history.empty else None,
    )


def fetch_currency_bundle(currency: str, years: int = 10) -> dict[str, Any]:
    fx = fetch_ecb_fx_series(currency, years=years)
    money = fetch_money_supply_panel(currency, years=years)
    metrics = compute_fx_metrics(fx, currency)

    return {
        "fx": fx,
        "money": money,
        "metrics": metrics,
    }


def fetch_major_currency_overview(years: int = 10) -> pd.DataFrame:
    rows = []
    for code in CURRENCY_META:
        try:
            fx = fetch_ecb_fx_series(code, years=years)
            m = compute_fx_metrics(fx, code)
            rows.append({
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
            })
        except Exception:
            rows.append({
                "Valuutta": code,
                "Nimi": CURRENCY_META[code]["name"],
                "Nykykurssi": np.nan,
                "Päivä": None,
                "YTD %": np.nan,
                "1v %": np.nan,
                "5v %": np.nan,
                "10v %": np.nan,
                "Volatiliteetti 1v %": np.nan,
                "10v min": np.nan,
                "10v max": np.nan,
            })

    return pd.DataFrame(rows)


def generate_ai_summary(currency: str, metrics: FxMetrics, money_df: pd.DataFrame) -> str:
    name = CURRENCY_META[currency]["name"]

    parts: list[str] = []
    if metrics.latest_rate is not None and metrics.latest_date is not None:
        parts.append(
            f"{name} noteerattiin viimeksi tasolla {metrics.latest_rate:.4f} per euro "
            f"({metrics.latest_date.date()})."
        )

    if metrics.change_1y_pct is not None:
        if metrics.change_1y_pct > 3:
            parts.append("Valuutta on vahvistunut selvästi viimeisen vuoden aikana euroa vastaan.")
        elif metrics.change_1y_pct < -3:
            parts.append("Valuutta on heikentynyt selvästi viimeisen vuoden aikana euroa vastaan.")
        else:
            parts.append("Valuutta on liikkunut viimeisen vuoden aikana melko sivuttaisesti euroa vastaan.")

    if metrics.volatility_1y_pct is not None:
        if metrics.volatility_1y_pct > 12:
            parts.append("Kurssivaihtelu on ollut poikkeuksellisen korkeaa.")
        elif metrics.volatility_1y_pct > 7:
            parts.append("Kurssivaihtelu on ollut kohtalaista.")
        else:
            parts.append("Kurssivaihtelu on ollut melko rauhallista.")

    if not money_df.empty:
        latest_money = money_df.dropna(subset=["BroadMoney_GrowthPct", "BroadMoney_GDPPct"], how="all")
        if not latest_money.empty:
            row = latest_money.iloc[-1]
            bits = []
            if pd.notna(row.get("BroadMoney_GrowthPct")):
                bits.append(f"broad money -kasvu oli {row['BroadMoney_GrowthPct']:.1f} %")
            if pd.notna(row.get("BroadMoney_GDPPct")):
                bits.append(f"ja broad money vastasi {row['BroadMoney_GDPPct']:.1f} % BKT:stä")
            if bits:
                parts.append(f"Viimeisimmän saatavilla olevan vuoden ({int(row['Year'])}) rahamääräindikaattorit viittaavat siihen, että " + " ".join(bits) + ".")

    if not parts:
        return "Valitusta valuutasta ei saatu riittävästi dataa yhteenvetoon."

    return " ".join(parts)