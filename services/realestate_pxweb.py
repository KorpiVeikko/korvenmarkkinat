from __future__ import annotations

import pandas as pd
import requests
from pyjstat import pyjstat

ASUNTOKAUPAT_URL = "https://pxdata.stat.fi/PxWeb/api/v1/fi/StatFin/ashi/statfin_ashi_pxt_12dd.px"
PELTO_URL = "https://statdb.luke.fi/PxWeb/api/v1/fi/LUKE/maa/peltov/0100_peltov.px"
TONTTI_URL = "https://pxdata.stat.fi/PxWeb/api/v1/fi/StatFin/kihi/statfin_kihi_pxt_11jb.px"


def _fetch_pxweb_json(url: str, query: dict) -> dict:
    r = requests.post(url, json=query, timeout=30)
    r.raise_for_status()
    return r.json()


def _fetch_pxweb_df(url: str, query: dict) -> pd.DataFrame:
    data = _fetch_pxweb_json(url, query)
    return pyjstat.from_json_stat(data, naming="id")[0]


def _fetch_pxweb_metadata(url: str) -> dict:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_realestate_counts() -> pd.DataFrame:
    query = {
        "query": [
            {
                "code": "Alue",
                "selection": {"filter": "item", "values": ["ksu"]},
            },
            {
                "code": "Tiedot",
                "selection": {"filter": "item", "values": ["lkm_julk_uudet"]},
            },
        ],
        "response": {"format": "json-stat2"},
    }
    return _fetch_pxweb_df(ASUNTOKAUPAT_URL, query)


def fetch_realestate_prices() -> pd.DataFrame:
    query = {
        "query": [
            {
                "code": "Alue",
                "selection": {"filter": "item", "values": ["ksu"]},
            },
            {
                "code": "Tiedot",
                "selection": {"filter": "item", "values": ["keskihinta_uudet"]},
            },
        ],
        "response": {"format": "json-stat2"},
    }
    return _fetch_pxweb_df(ASUNTOKAUPAT_URL, query)


def clean_realestate_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Kvartaali", "Arvo"])

    out = df.rename(columns={"value": "Arvo", "Vuosineljännes": "Kvartaali"}).copy()

    if "Kvartaali" not in out.columns:
        return pd.DataFrame(columns=["Kvartaali", "Arvo"])

    out["Arvo"] = pd.to_numeric(out["Arvo"], errors="coerce")
    out = out.dropna(subset=["Kvartaali", "Arvo"]).copy()

    out["Kvartaali_sort"] = (
        out["Kvartaali"]
        .astype(str)
        .str.replace("Q1", "01", regex=False)
        .str.replace("Q2", "04", regex=False)
        .str.replace("Q3", "07", regex=False)
        .str.replace("Q4", "10", regex=False)
    )

    return out.sort_values("Kvartaali_sort").drop(columns=["Kvartaali_sort"]).reset_index(drop=True)


def add_yoy_change_quarterly(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Kvartaali", "Arvo", "YoY_pct"])

    out = df.copy()
    out["Arvo"] = pd.to_numeric(out["Arvo"], errors="coerce")
    out["YoY_pct"] = out["Arvo"].pct_change(4) * 100.0
    return out


def fetch_farmland_prices(series_type: str = "sale") -> pd.DataFrame:
    meta = requests.get(PELTO_URL, timeout=30).json()
    variables = meta if isinstance(meta, list) else meta.get("variables", [])

    value_code = None
    region_code = None
    time_code = None
    value_var = None

    for var in variables:
        if not isinstance(var, dict):
            continue
        code = var.get("code")
        if code == "NUTS2":
            region_code = code
        elif var.get("time") is True:
            time_code = code
        else:
            value_code = code
            value_var = var

    if not value_code or not region_code or not time_code or not value_var:
        raise ValueError("Peltomaan metadatasta ei löytynyt tarvittavia kenttiä.")

    values = value_var.get("values", [])
    value_texts = value_var.get("valueTexts", [])
    pairs = list(zip(values, value_texts))

    if series_type == "sale":
        chosen = next((code for code, text in pairs if "ostohinta" in str(text).lower()), None)
    elif series_type == "rent":
        chosen = next((code for code, text in pairs if "vuokrahinta" in str(text).lower()), None)
    else:
        raise ValueError("series_type pitää olla 'sale' tai 'rent'")

    if not chosen:
        raise ValueError(f"Sarjaa '{series_type}' ei löytynyt metadatasta. Saatavilla: {pairs}")

    query = {
        "query": [
            {
                "code": value_code,
                "selection": {"filter": "item", "values": [chosen]},
            },
            {
                "code": region_code,
                "selection": {"filter": "all", "values": ["*"]},
            },
            {
                "code": time_code,
                "selection": {"filter": "all", "values": ["*"]},
            },
        ],
        "response": {"format": "json-stat2"},
    }

    df = _fetch_pxweb_df(PELTO_URL, query)
    df = pd.DataFrame(df).rename(columns={"value": "Arvo", time_code: "Vuosi", region_code: "Alue"})

    area_map = {
        "SSS": "Koko maa",
        "FI1B": "Helsinki-Uusimaa",
        "FI1C": "Etelä-Suomi",
        "FI19": "Länsi-Suomi",
        "FI1D": "Pohjois- ja Itä-Suomi",
        "FI20": "Ahvenanmaa",
    }

    if "Alue" in df.columns:
        df["Alue"] = df["Alue"].astype(str).replace(area_map)

    df["Vuosi"] = pd.to_numeric(df["Vuosi"], errors="coerce")
    df["Arvo"] = pd.to_numeric(df["Arvo"], errors="coerce")
    return df.dropna(subset=["Vuosi", "Arvo"]).sort_values(["Alue", "Vuosi"]).reset_index(drop=True)


def add_yoy_change_yearly(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Vuosi", "Arvo", "YoY_pct"])

    out = df.copy()
    out["Arvo"] = pd.to_numeric(out["Arvo"], errors="coerce")

    if "Alue" in out.columns:
        out = out.sort_values(["Alue", "Vuosi"]).reset_index(drop=True)
        out["YoY_pct"] = out.groupby("Alue")["Arvo"].pct_change(1) * 100.0
    else:
        out = out.sort_values("Vuosi").reset_index(drop=True)
        out["YoY_pct"] = out["Arvo"].pct_change(1) * 100.0

    return out


def fetch_detached_plot_data() -> pd.DataFrame:
    """
    Omakotitalotontit:
    - 01 = Koko maa
    - 08,09,10,11 = Etelä-, Länsi-, Itä- ja Pohjois-Suomi
    - keskihinta = Neliöhinta (EUR/m2)
    """
    query = {
        "query": [
            {
                "code": "Aluejako",
                "selection": {
                    "filter": "item",
                    "values": ["01", "08", "09", "10", "11"],
                },
            },
            {
                "code": "Tiedot",
                "selection": {
                    "filter": "item",
                    "values": [
                        "ketjutettu_lv",
                        "realind_lv",
                        "keskihinta",
                        "lkm_julk",
                    ],
                },
            },
        ],
        "response": {"format": "json-stat2"},
    }

    return pd.DataFrame(_fetch_pxweb_df(TONTTI_URL, query))


def _normalize_detached_area(value: str) -> str:
    s = str(value).strip().lower()

    mapping = {
        "01": "Koko maa",
        "08": "Etelä-Suomi",
        "09": "Länsi-Suomi",
        "10": "Itä-Suomi",
        "11": "Pohjois-Suomi",
        "koko maa": "Koko maa",
        "etelä-suomi": "Etelä-Suomi",
        "länsi-suomi": "Länsi-Suomi",
        "itä-suomi": "Itä-Suomi",
        "pohjois-suomi": "Pohjois-Suomi",
    }

    return mapping.get(s, str(value).strip())


def _normalize_detached_metric(value: str) -> str:
    s = str(value).strip().lower()

    if s == "ketjutettu_lv" or s == "indeksi":
        return "Hintaindeksi"
    if s == "realind_lv" or "reaalihintaindeksi" in s:
        return "Reaalihintaindeksi"
    if s == "keskihinta" or "neliöhinta" in s:
        return "Neliöhinta"
    if s == "lkm_julk" or "kauppojen lukumäärä" in s:
        return "Kauppojen lukumäärä"

    return str(value).strip()


def clean_detached_plot_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Jakso", "Jakso_dt", "Vuosi", "Alue", "Tiedot", "Arvo"])

    out = df.copy()

    rename_map = {
        "Aluejako": "Alue",
        "Vuosineljännes": "Jakso",
        "Tiedot": "Tiedot",
        "value": "Arvo",
    }
    out = out.rename(columns=rename_map)

    required = ["Alue", "Jakso", "Tiedot", "Arvo"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        return pd.DataFrame(columns=["Jakso", "Jakso_dt", "Vuosi", "Alue", "Tiedot", "Arvo"])

    out["Arvo"] = pd.to_numeric(out["Arvo"], errors="coerce")

    out["Vuosi"] = out["Jakso"].astype(str).str.extract(r"(\d{4})", expand=False)
    out["Vuosi"] = pd.to_numeric(out["Vuosi"], errors="coerce")

    out["Alue"] = out["Alue"].map(_normalize_detached_area)
    out["Tiedot"] = out["Tiedot"].map(_normalize_detached_metric)

    out = out.dropna(subset=["Jakso", "Vuosi", "Arvo"]).copy()
    out["Vuosi"] = out["Vuosi"].astype(int)

    out["Jakso_dt"] = pd.PeriodIndex(out["Jakso"].astype(str), freq="Q").to_timestamp()
    out = out.sort_values(["Tiedot", "Alue", "Jakso_dt"]).reset_index(drop=True)

    return out[["Jakso", "Jakso_dt", "Vuosi", "Alue", "Tiedot", "Arvo"]]