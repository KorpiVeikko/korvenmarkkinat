# services/luke_pxweb.py
import re
import requests
import pandas as pd
from pyjstat import pyjstat

WOOD_TABLE_URL = "https://statdb.luke.fi:443/PxWeb/api/v1/fi/LUKE/met/metryv/0100_metryv.px"


def fetch_wood_prices(
    areas=None,
    harvest_types=None,
    assortments=None,
    response_format: str = "json-stat2",
    timeout: int = 60,
) -> pd.DataFrame:
    """
    Hakee LUKE PxWebistä puukaupan kantohinnat viikoittain.
    - areas = MPKH (esim: ["1","3","4","5","6","71","72","8"])
    - harvest_types = HAKT (esim: ["8021","8023","8022"])
    - assortments = PTL (esim: ["N1".."N6"])
    Palauttaa long-muotoisen DataFrame:n (viikko/alue/hakkuutapa/puutavaralaji + Arvo).
    """

    if areas is None:
        areas = ["1", "3", "4", "5", "6", "71", "72", "8"]

    if harvest_types is None:
        harvest_types = ["8021", "8023", "8022"]

    if assortments is None:
        assortments = ["N1", "N2", "N3", "N4", "N5", "N6"]

    query = {
        "query": [
            {
                "code": "MPKH",
                "selection": {"filter": "item", "values": areas},
            },
            {
                "code": "HAKT",
                "selection": {"filter": "item", "values": harvest_types},
            },
            {
                "code": "PTL",
                "selection": {"filter": "item", "values": assortments},
            },
        ],
        "response": {"format": response_format},
    }

    r = requests.post(WOOD_TABLE_URL, json=query, timeout=timeout)
    r.raise_for_status()

    if response_format != "json-stat2":
        raise ValueError(
            "Tämä funktio on toteutettu json-stat2 -muodolle. "
            "Vaihda response_format='json-stat2'."
        )

    df = pyjstat.from_json_stat(r.json(), naming="label")[0]

    # Vakioidaan sarakenimi "Arvo"
    if "value" in df.columns:
        df.rename(columns={"value": "Arvo"}, inplace=True)
    elif "Arvo" not in df.columns:
        # fallback
        for col in df.columns:
            if str(col).lower() == "value":
                df.rename(columns={col: "Arvo"}, inplace=True)
                break

    df["Arvo"] = pd.to_numeric(df["Arvo"], errors="coerce")
    df = df.dropna(subset=["Arvo"]).copy()

    return df


def add_week_sort_key(df: pd.DataFrame, week_col: str) -> pd.DataFrame:
    """
    Lisää sort_key-sarakkeen viikkotekstistä, esim:
    - "2024W05" -> 202405
    - "2024/05" -> 202405
    Jos muoto poikkeaa, yrittää silti parhaan.
    """
    def _to_key(x: str) -> int:
        s = str(x).strip()
        m = re.search(r"(\d{4})\D?W?(\d{1,2})", s, re.IGNORECASE)
        if not m:
            return -1
        y = int(m.group(1))
        w = int(m.group(2))
        return y * 100 + w

    out = df.copy()
    out["sort_key"] = out[week_col].apply(_to_key)
    return out

