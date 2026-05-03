from __future__ import annotations

import pandas as pd
import requests
from pyjstat import pyjstat

from services.forest_helpers import (
    find_first_matching_column,
    add_week_sort_key,
    add_week_date,
    add_month_date,
    add_year_date,
)

WOOD_PRICES_URL = "https://statdb.luke.fi:443/PxWeb/api/v1/fi/LUKE/met/metryv/0100_metryv.px"
INDUSTRIAL_WOOD_URL = "https://statdb.luke.fi:443/PxWeb/api/v1/fi/LUKE/met/teokau/kk/0100_teokau.px"
WOOD_USE_URL = "https://statdb.luke.fi:443/PxWeb/api/v1/fi/LUKE/met/puukay/0100_puukay.px"
HARVEST_URL = "https://statdb.luke.fi:443/PxWeb/api/v1/fi/LUKE/met/hakker/0100_hakker.px"


def _fetch_px_table(url: str, query: dict, timeout: int = 60) -> pd.DataFrame:
    r = requests.post(url, json=query, timeout=timeout)
    r.raise_for_status()
    return pyjstat.from_json_stat(r.json(), naming="label")[0]


def _rename_value_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "value" in out.columns:
        out = out.rename(columns={"value": "Arvo"})
    elif "Arvo" not in out.columns:
        for col in out.columns:
            if str(col).lower() == "value":
                out = out.rename(columns={col: "Arvo"})
                break

    out["Arvo"] = pd.to_numeric(out["Arvo"], errors="coerce")
    out = out.dropna(subset=["Arvo"]).copy()
    return out


def fetch_wood_prices() -> pd.DataFrame:
    query = {
        "query": [
            {
                "code": "MPKH",
                "selection": {"filter": "item", "values": ["1", "3", "4", "5", "6", "71", "72", "8"]},
            },
            {
                "code": "HAKT",
                "selection": {"filter": "item", "values": ["8021", "8023", "8022"]},
            },
            {
                "code": "PTL",
                "selection": {"filter": "item", "values": ["N1", "N2", "N3", "N4", "N5", "N6"]},
            },
        ],
        "response": {"format": "json-stat2"},
    }
    df = _fetch_px_table(WOOD_PRICES_URL, query)
    return _rename_value_column(df)


def prepare_wood_prices_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    week_col = find_first_matching_column(out, ["W", "Viikko", "week"])
    if week_col is None:
        return pd.DataFrame()

    out = add_week_sort_key(out, week_col)
    out = out[out["sort_key"] > 0].copy()
    out = add_week_date(out, "sort_key")
    out = out.dropna(subset=["Date", "Arvo"]).sort_values("Date").reset_index(drop=True)
    return out


def fetch_industrial_wood_trade() -> pd.DataFrame:
    query = {
        "query": [],
        "response": {"format": "json-stat2"},
    }
    df = _fetch_px_table(INDUSTRIAL_WOOD_URL, query)
    return _rename_value_column(df)


def prepare_industrial_wood_trade_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    month_col = find_first_matching_column(out, ["Kuukausi", "Aika", "time", "month"])
    if month_col is None:
        return pd.DataFrame()

    out = add_month_date(out, month_col)
    out = out.dropna(subset=["Date", "Arvo"]).sort_values("Date").reset_index(drop=True)
    return out


def fetch_wood_use() -> pd.DataFrame:
    query = {
        "query": [
            {
                "code": "A",
                "selection": {
                    "filter": "item",
                    "values": [
                        "2015", "2016", "2017", "2018", "2019",
                        "2020", "2021", "2022", "2023", "2024",
                    ],
                },
            },
            {
                "code": "MK",
                "selection": {
                    "filter": "item",
                    "values": ["SSS"],
                },
            },
            {
                "code": "KT",
                "selection": {
                    "filter": "item",
                    "values": ["RAAP_YHT", "RAAP_METTEOL", "RAAP_ENERT"],
                },
            },
        ],
        "response": {"format": "json-stat2"},
    }
    df = _fetch_px_table(WOOD_USE_URL, query)
    return _rename_value_column(df)


def prepare_wood_use_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    year_col = find_first_matching_column(out, ["Vuosi", "A", "year"])
    if year_col is None:
        return pd.DataFrame()

    out = add_year_date(out, year_col)
    out = out.dropna(subset=["Date", "Arvo"]).sort_values("Date").reset_index(drop=True)
    return out


def fetch_harvests() -> pd.DataFrame:
    query = {
        "query": [
            {
                "code": "MK",
                "selection": {
                    "filter": "item",
                    "values": [
                        "SSS", "MK01", "MK02", "MK04", "MK05", "MK06", "MK07", "MK08",
                        "MK09", "MK10", "MK11", "MK12", "MK13", "MK14", "MK15", "MK16",
                        "MK17", "MK18", "MK19", "MK21",
                    ],
                },
            },
            {
                "code": "OM",
                "selection": {
                    "filter": "item",
                    "values": ["OM_YHT", "YKS", "METTEOL_VAL"],
                },
            },
            {
                "code": "PTL",
                "selection": {
                    "filter": "item",
                    "values": ["PTL_YHT", "TUK_YHT", "KUI_YHT", "ENER"],
                },
            },
            {
                "code": "PL",
                "selection": {
                    "filter": "item",
                    "values": ["PL_YHT", "MA", "KU", "LE"],
                },
            },
        ],
        "response": {"format": "json-stat2"},
    }
    df = _fetch_px_table(HARVEST_URL, query)
    return _rename_value_column(df)


def prepare_harvests_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    year_col = find_first_matching_column(out, ["Vuosi", "Aika", "year"])
    if year_col is None:
        return pd.DataFrame()

    out = add_year_date(out, year_col)
    out = out.dropna(subset=["Date", "Arvo"]).sort_values("Date").reset_index(drop=True)
    return out