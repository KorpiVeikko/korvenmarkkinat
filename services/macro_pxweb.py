# services/macro_pxweb.py
from __future__ import annotations

import itertools
import requests
import pandas as pd


# -------------------------
# PXWeb endpoints (StatFin)
# -------------------------
# Inflaatio: kuluttajahintaindeksin vuosimuutos, kuukausitiedot
CPI_YOY_URL = "https://pxdata.stat.fi/PxWeb/api/v1/fi/StatFin/khi/statfin_khi_pxt_122p.px"

# BKT neljännesvuosittain (Bruttokansantuote ja -tulo sekä tarjonta ja kysyntä), 1990Q1-2025Q4
GDP_Q_URL = "https://pxdata.stat.fi/PxWeb/api/v1/fi/StatFin/ntp/statfin_ntp_pxt_132h.px"


# -------------------------
# JSON-stat2 parser (kuten energy_pxweb)
# -------------------------
def _dedupe_columns(cols: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for c in cols:
        c0 = str(c)
        if c0 not in seen:
            seen[c0] = 1
            out.append(c0)
        else:
            seen[c0] += 1
            out.append(f"{c0}__{seen[c0]}")
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
        lab = cat.get("label") or {}

        if isinstance(idx, dict):
            keys = list(idx.keys())
        elif isinstance(idx, list):
            keys = idx
        else:
            keys = list(lab.keys())

        labels = [lab.get(k, str(k)) for k in keys]
        dim_levels.append(labels)

    combos = list(itertools.product(*dim_levels))
    if len(combos) != len(values):
        return pd.DataFrame()

    cols = _dedupe_columns([str(x) for x in ids])
    df = pd.DataFrame(combos, columns=cols)
    df["Arvo"] = pd.to_numeric(values, errors="coerce")
    df.columns = _dedupe_columns(list(df.columns))
    return df


def _post_px(url: str, query: dict, timeout: int = 45) -> pd.DataFrame:
    r = requests.post(url, json=query, timeout=timeout)
    r.raise_for_status()
    return _parse_jsonstat2(r.json())


def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lisää Aika + Aika_dt tulokseen. Tukee:
    - 2025M01
    - 2025Q4
    - 2025
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    out.columns = _dedupe_columns(list(out.columns))

    # etsi aikadimensio
    time_candidates = [c for c in out.columns if str(c).strip().lower() in ("aika", "time", "kuukausi", "vuosineljännes", "vuosi")]
    time_col = time_candidates[0] if time_candidates else out.columns[0]

    s = out[time_col].astype(str).str.strip()
    out["Aika"] = s

    q = s.str.extract(r"^(?P<y>\d{4})Q(?P<q>\d)$")
    m = s.str.extract(r"^(?P<y>\d{4})M(?P<m>\d{2})$")
    y_only = s.str.extract(r"^(?P<y>\d{4})$")

    if m["y"].notna().any():
        out["Vuosi_num"] = pd.to_numeric(m["y"], errors="coerce")
        out["Kuukausi_num"] = pd.to_numeric(m["m"], errors="coerce")
        out["Aika_dt"] = pd.to_datetime(
            out["Vuosi_num"].astype("Int64").astype(str) + "-" +
            out["Kuukausi_num"].astype("Int64").astype(str).str.zfill(2) + "-01",
            errors="coerce",
        )
    elif q["y"].notna().any():
        out["Vuosi_num"] = pd.to_numeric(q["y"], errors="coerce")
        qn = pd.to_numeric(q["q"], errors="coerce")
        start_month = (qn - 1) * 3 + 1
        out["Aika_dt"] = pd.to_datetime(
            out["Vuosi_num"].astype("Int64").astype(str) + "-" +
            start_month.astype("Int64").astype(str).str.zfill(2) + "-01",
            errors="coerce",
        )
    elif y_only["y"].notna().any():
        out["Vuosi_num"] = pd.to_numeric(y_only["y"], errors="coerce")
        out["Aika_dt"] = pd.to_datetime(out["Vuosi_num"].astype("Int64").astype(str) + "-01-01", errors="coerce")
    else:
        out["Aika_dt"] = pd.to_datetime(s, errors="coerce")

    out.columns = _dedupe_columns(list(out.columns))
    return out


# -------------------------
# Public fetchers
# -------------------------
def fetch_cpi_yoy() -> pd.DataFrame:
    """
    Kuluttajahintaindeksin vuosimuutos (%), kuukausitaso.
    StatFin khi / 122p.
    """
    query = {
        "query": [
            {"code": "Kuukausi", "selection": {"filter": "all", "values": ["*"]}},
            {"code": "Tiedot", "selection": {"filter": "all", "values": ["*"]}},
        ],
        "response": {"format": "json-stat2"},
    }
    df = _post_px(CPI_YOY_URL, query)
    return add_time_columns(df)


def fetch_gdp_quarterly() -> pd.DataFrame:
    """
    Neljännesvuosittainen BKT-taulukko 132h.
    Haetaan vain Taloustoimi = Bruttokansantuote (B1GMH) ja kaikki Tiedot + kaikki neljännekset.
    """
    # Huom: Taloustoimi dimension labelissa on yleensä "B1GMH Bruttokansantuote ..."
    # Käytetään item-filterillä B1GMH-koodi. Usein PxWeb hyväksyy koodit, mutta joskus vaatii labelin.
    # Varmin: haetaan kaikki Taloustoimi ja suodatetaan koodilla/labelilla jälkikäteen — mutta se voi olla iso.
    # Tässä käytetään koodia "B1GMH" (yleinen koodi), ja jos se ei osu, macro.py:ssä on fallback.
    query = {
        "query": [
            {"code": "Vuosineljännes", "selection": {"filter": "all", "values": ["*"]}},
            {"code": "Taloustoimi", "selection": {"filter": "item", "values": ["B1GMH"]}},
            {"code": "Tiedot", "selection": {"filter": "all", "values": ["*"]}},
        ],
        "response": {"format": "json-stat2"},
    }
    df = _post_px(GDP_Q_URL, query)
    return add_time_columns(df)