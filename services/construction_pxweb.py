from __future__ import annotations

import pandas as pd
import requests
from pyjstat import pyjstat

RAKU_URL = "https://pxdata.stat.fi/PxWeb/api/v1/fi/StatFin/raku/statfin_raku_pxt_156f.px"


def fetch_construction_data() -> pd.DataFrame:
    query = {
        "query": [
            {
                "code": "rakennusvaihe",
                "selection": {"filter": "item", "values": ["1", "2", "3"]},
            },
            {
                "code": "alue",
                "selection": {
                    "filter": "item",
                    "values": [
                        "SSS",
                        "MK01",
                        "MK02",
                        "MK04",
                        "MK05",
                        "MK06",
                        "MK07",
                        "MK08",
                        "MK09",
                        "MK10",
                        "MK11",
                        "MK12",
                        "MK13",
                        "MK14",
                        "MK15",
                        "MK16",
                        "MK17",
                        "MK18",
                        "MK19",
                        "MK21",
                    ],
                },
            },
            {
                "code": "timeperiod",
                "selection": {"filter": "all", "values": ["*"]},
            },
            {
                "code": "rakennusluokitus2018",
                "selection": {"filter": "item", "values": ["01", "02T19"]},
            },
            {
                "code": "ContentCode",
                "selection": {"filter": "item", "values": ["uusiAsuntoLkm"]},
            },
        ],
        "response": {"format": "json-stat2"},
    }

    response = requests.post(RAKU_URL, json=query, timeout=30)
    response.raise_for_status()
    df = pyjstat.from_json_stat(response.json(), naming="id")[0]
    return pd.DataFrame(df)


def clean_construction_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(
            columns=["Vaihe", "Alue", "Aika", "Aika_dt", "Arvo"]
        )

    out = df.rename(
        columns={
            "value": "Arvo",
            "rakennusvaihe": "Vaihe",
            "alue": "Alue",
            "timeperiod": "Aika",
        }
    ).copy()

    vaihe_map = {
        "1": "Rakennusluvat",
        "2": "Rakenteilla",
        "3": "Valmistuneet",
    }

    alue_map = {
        "SSS": "Koko maa",
        "MK01": "Uusimaa",
        "MK02": "Varsinais-Suomi",
        "MK04": "Satakunta",
        "MK05": "Kanta-Häme",
        "MK06": "Pirkanmaa",
        "MK07": "Päijät-Häme",
        "MK08": "Kymenlaakso",
        "MK09": "Etelä-Karjala",
        "MK10": "Etelä-Savo",
        "MK11": "Pohjois-Savo",
        "MK12": "Pohjois-Karjala",
        "MK13": "Keski-Suomi",
        "MK14": "Etelä-Pohjanmaa",
        "MK15": "Pohjanmaa",
        "MK16": "Keski-Pohjanmaa",
        "MK17": "Pohjois-Pohjanmaa",
        "MK18": "Kainuu",
        "MK19": "Lappi",
        "MK21": "Ahvenanmaa",
    }

    out["Vaihe"] = out["Vaihe"].astype(str).map(vaihe_map)
    out["Alue"] = out["Alue"].astype(str).map(alue_map)
    out["Arvo"] = pd.to_numeric(out["Arvo"], errors="coerce")
    out["Aika_dt"] = pd.to_datetime(out["Aika"].astype(str), format="%YM%m", errors="coerce")

    out = out.dropna(subset=["Vaihe", "Alue", "Arvo", "Aika_dt"]).copy()

    # Summataan rakennusluokat yhteen -> yksi piste per kuukausi / alue / vaihe
    out = (
        out.groupby(["Alue", "Vaihe", "Aika", "Aika_dt"], as_index=False)["Arvo"]
        .sum()
        .sort_values(["Alue", "Vaihe", "Aika_dt"])
        .reset_index(drop=True)
    )

    return out


def add_construction_features(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(
            columns=["Vaihe", "Alue", "Aika", "Aika_dt", "Arvo", "Arvo_sum12", "YoY_pct"]
        )

    out = df.copy().sort_values(["Alue", "Vaihe", "Aika_dt"]).reset_index(drop=True)

    out["Arvo_sum12"] = (
        out.groupby(["Alue", "Vaihe"])["Arvo"]
        .transform(lambda s: s.rolling(window=12, min_periods=1).sum())
    )

    out["YoY_pct"] = (
        out.groupby(["Alue", "Vaihe"])["Arvo"]
        .transform(lambda s: s.pct_change(12) * 100.0)
    )

    return out


def filter_last_n_years(df: pd.DataFrame, years: int = 10) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    max_date = df["Aika_dt"].max()
    if pd.isna(max_date):
        return df

    cutoff = max_date - pd.DateOffset(years=years)
    return df[df["Aika_dt"] >= cutoff].copy()