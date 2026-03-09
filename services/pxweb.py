# services/pxweb.py

import requests
import pandas as pd
from pyjstat import pyjstat

PXWEB_URL = "https://pxdata.stat.fi/PxWeb/api/v1/fi/StatFin/ashi/statfin_ashi_pxt_12dd.px"


def _fetch_pxweb(query: dict) -> pd.DataFrame:
    r = requests.post(PXWEB_URL, json=query, timeout=30)
    r.raise_for_status()
    return pyjstat.from_json_stat(r.json(), naming="id")[0]


def fetch_realestate_counts() -> pd.DataFrame:
    """Asuntokauppojen lukumäärä, koko maa"""
    query = {
        "query": [
            {
                "code": "Alue",
                "selection": {
                    "filter": "item",
                    "values": ["ksu"]
                }
            },
            {
                "code": "Tiedot",
                "selection": {
                    "filter": "item",
                    "values": ["lkm_julk_uudet"]
                }
            }
        ],
        "response": {"format": "json-stat2"}
    }

    return _fetch_pxweb(query)


def fetch_realestate_prices() -> pd.DataFrame:
    """Uusien asuntojen keskimääräinen neliöhinta, koko maa"""
    query = {
        "query": [
            {
                "code": "Alue",
                "selection": {
                    "filter": "item",
                    "values": ["ksu"]
                }
            },
            {
                "code": "Tiedot",
                "selection": {
                    "filter": "item",
                    "values": ["keskihinta_uudet"]
                }
            }
        ],
        "response": {"format": "json-stat2"}
    }

    return _fetch_pxweb(query)


def clean_realestate_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={
        "value": "Arvo",
        "Vuosineljännes": "Kvartaali"
    })

    df["Kvartaali"] = pd.Categorical(
        df["Kvartaali"],
        categories=sorted(df["Kvartaali"].unique()),
        ordered=True
    )

    return df.sort_values("Kvartaali")


