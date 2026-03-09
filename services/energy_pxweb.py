# services/energy_pxweb.py
from __future__ import annotations

import itertools
import requests
import pandas as pd

# =========================
# 1) SÄHKÖN TUOTANTO / HANKINTA
# =========================
ELECTRICITY_URL = "https://pxdata.stat.fi/PxWeb/api/v1/fi/StatFin/ehk/statfin_ehk_pxt_12su.px"

ELECTRICITY_VALUES = [
    "SSS",
    "1", "1.1", "1.2", "1.3", "1.4",
    "1.5", "1.5.1", "1.5.2", "1.5.3",
    "1.6",
    "2", "2.1", "2.2", "2.3", "2.4",
]

# =========================
# 2) LÄMMITYSENERGIAN HINNAT
# =========================
HEATING_PRICE_URL = "https://pxdata.stat.fi/PxWeb/api/v1/fi/StatFin/ehi/statfin_ehi_pxt_13nl.px"
HEATING_PRICE_QUERY = {"query": [], "response": {"format": "json-stat2"}}

# =========================
# 3) KOTITALOUSSÄHKÖN HINTA (komponentit)
# =========================
HOUSEHOLD_ELECTRICITY_PRICE_URL = "https://pxdata.stat.fi/PxWeb/api/v1/fi/StatFin/ehi/statfin_ehi_pxt_13rb.px"
HOUSEHOLD_ELECTRICITY_PRICE_QUERY = {"query": [], "response": {"format": "json-stat2"}}


def _dedupe_columns(cols: list[str]) -> list[str]:
    """Varmistaa uniikit sarakenimet (Plotly/narwhals vaatii tämän)."""
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

    # ✅ varmistus: vielä kerran dedupe
    df.columns = _dedupe_columns(list(df.columns))
    return df


def _post_px(url: str, query: dict, timeout: int = 45) -> pd.DataFrame:
    r = requests.post(url, json=query, timeout=timeout)
    r.raise_for_status()
    return _parse_jsonstat2(r.json())


def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lisää Aika ja Aika_dt ja varmistaa uniikit sarakkeet."""
    if df.empty:
        return df

    out = df.copy()
    out.columns = _dedupe_columns(list(out.columns))  # ✅ tärkeä

    # etsi aikadimensio
    time_candidates = [c for c in out.columns if str(c).strip().lower() in ("aika", "time")]
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

    out.columns = _dedupe_columns(list(out.columns))  # ✅ varmistus lopuksi
    return out


def fetch_electricity_production_consumption(values: list[str] | None = None) -> pd.DataFrame:
    query = {
        "query": [
            {
                "code": "Sähkön tuotanto/hankinta",
                "selection": {"filter": "item", "values": values or ELECTRICITY_VALUES},
            }
        ],
        "response": {"format": "json-stat2"},
    }
    df = _post_px(ELECTRICITY_URL, query)
    return add_time_columns(df)


def fetch_heating_energy_prices() -> pd.DataFrame:
    df = _post_px(HEATING_PRICE_URL, HEATING_PRICE_QUERY)
    return add_time_columns(df)


def fetch_household_electricity_prices() -> pd.DataFrame:
    df = _post_px(HOUSEHOLD_ELECTRICITY_PRICE_URL, HOUSEHOLD_ELECTRICITY_PRICE_QUERY)
    return add_time_columns(df)






