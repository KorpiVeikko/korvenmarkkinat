# services/worldbank.py
from __future__ import annotations

import requests
import pandas as pd


WB_BASE = "https://api.worldbank.org/v2"


def fetch_indicator(country_iso3: str, indicator_code: str, per_page: int = 200) -> pd.DataFrame:
    """
    Hakee World Bank API:sta indikaattorin aikasarjan.
    Palauttaa DataFrame: year(int), value(float).
    """
    url = f"{WB_BASE}/country/{country_iso3}/indicator/{indicator_code}"
    params = {"format": "json", "per_page": per_page}

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()

    payload = r.json()
    if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
        return pd.DataFrame(columns=["year", "value"])

    rows = []
    for item in payload[1]:
        year = item.get("date")
        value = item.get("value")
        if year is None:
            continue
        rows.append({"year": int(year), "value": value})

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["year", "value"])

    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"]).sort_values("year").reset_index(drop=True)
    return df


def latest_and_change(df: pd.DataFrame) -> tuple[float | None, float | None, int | None]:
    """
    Palauttaa (latest_value, change_vs_previous_year, latest_year)
    change on latest - previous (ei prosenttia, vaan yksikköä).
    """
    if df is None or df.empty:
        return None, None, None

    latest = df.iloc[-1]
    latest_year = int(latest["year"])
    latest_val = float(latest["value"])

    if len(df) < 2:
        return latest_val, None, latest_year

    prev_val = float(df.iloc[-2]["value"])
    return latest_val, (latest_val - prev_val), latest_year
