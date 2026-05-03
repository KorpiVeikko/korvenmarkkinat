# services/energy_pxweb.py
from __future__ import annotations

import itertools
from typing import Iterable

import pandas as pd
import requests


ELECTRICITY_URL = "https://pxdata.stat.fi/PxWeb/api/v1/fi/StatFin/ehk/statfin_ehk_pxt_12su.px"
HOUSEHOLD_ELECTRICITY_PRICE_URL = "https://pxdata.stat.fi/PxWeb/api/v1/fi/StatFin/ehi/statfin_ehi_pxt_13rb.px"


ELECTRICITY_SERIES_VALUES = [
    "SSS",
    "1",
    "1.1",
    "1.2",
    "1.3",
    "1.4",
    "1.5",
    "1.5.1",
    "1.5.2",
    "1.5.3",
    "1.6",
    "2",
    "2.1",
    "2.2",
    "2.3",
    "2.4",
]

PRICE_COMPONENT_VALUES = ["A", "B", "C", "SSS"]


def _month_values(start_year: int = 2020, end_year: int = 2025) -> list[str]:
    values: list[str] = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            values.append(f"{year}M{month:02d}")
    return values


def _dedupe_columns(cols: Iterable[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []

    for col in cols:
        base = str(col)
        if base not in seen:
            seen[base] = 0
            out.append(base)
        else:
            seen[base] += 1
            out.append(f"{base}__{seen[base]}")
    return out


def _parse_jsonstat2(payload: dict) -> pd.DataFrame:
    if not isinstance(payload, dict) or "value" not in payload or "dimension" not in payload:
        return pd.DataFrame()

    dim = payload["dimension"]
    ids = payload.get("id") or dim.get("id")
    values = payload.get("value")

    if not ids or values is None:
        return pd.DataFrame()

    dim_levels: list[list[str]] = []
    for did in ids:
        d = dim.get(did, {})
        cat = d.get("category") or {}
        idx = cat.get("index")
        labels = cat.get("label") or {}

        if isinstance(idx, dict):
            keys = list(idx.keys())
        elif isinstance(idx, list):
            keys = idx
        else:
            keys = list(labels.keys())

        dim_levels.append([labels.get(k, str(k)) for k in keys])

    combos = list(itertools.product(*dim_levels))
    if len(combos) != len(values):
        return pd.DataFrame()

    cols = _dedupe_columns([str(x) for x in ids])
    df = pd.DataFrame(combos, columns=cols)
    df["Arvo"] = pd.to_numeric(values, errors="coerce")
    df.columns = _dedupe_columns(df.columns)
    return df


def _post_px(url: str, query: dict, timeout: int = 45) -> pd.DataFrame:
    response = requests.post(url, json=query, timeout=timeout)
    response.raise_for_status()
    return _parse_jsonstat2(response.json())


def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    out.columns = _dedupe_columns(out.columns)

    time_col = None
    for col in out.columns:
        if str(col).strip().lower() in ("kuukausi", "aika", "time"):
            time_col = col
            break

    if time_col is None:
        time_col = out.columns[0]

    out["Aika"] = out[time_col].astype(str).str.strip()

    month_match = out["Aika"].str.extract(r"^(?P<y>\d{4})M(?P<m>\d{2})$")
    out["Vuosi_num"] = pd.to_numeric(month_match["y"], errors="coerce")
    out["Kuukausi_num"] = pd.to_numeric(month_match["m"], errors="coerce")
    out["Aika_dt"] = pd.to_datetime(
        out["Vuosi_num"].astype("Int64").astype(str)
        + "-"
        + out["Kuukausi_num"].astype("Int64").astype(str).str.zfill(2)
        + "-01",
        errors="coerce",
    )

    return out


def fetch_electricity_production_consumption(
    start_year: int = 2020,
    end_year: int = 2025,
) -> pd.DataFrame:
    query = {
        "query": [
            {
                "code": "Kuukausi",
                "selection": {
                    "filter": "item",
                    "values": _month_values(start_year, end_year),
                },
            },
            {
                "code": "Sähkön tuotanto/hankinta",
                "selection": {
                    "filter": "item",
                    "values": ELECTRICITY_SERIES_VALUES,
                },
            },
        ],
        "response": {"format": "json-stat2"},
    }

    df = _post_px(ELECTRICITY_URL, query)
    return add_time_columns(df)


def fetch_household_electricity_prices() -> pd.DataFrame:
    query = {
        "query": [
            {
                "code": "Hintakomponentti",
                "selection": {
                    "filter": "item",
                    "values": PRICE_COMPONENT_VALUES,
                },
            }
        ],
        "response": {"format": "json-stat2"},
    }
    df = _post_px(HOUSEHOLD_ELECTRICITY_PRICE_URL, query)
    return add_time_columns(df)
