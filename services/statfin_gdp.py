# services/statfin_gdp.py
from __future__ import annotations

import itertools
import requests
import pandas as pd


# ✅ Käytä PXWeb (isoilla) – tätä muotoa StatFin dokumenteissa/URL:eissa yleisesti käytetään
GDP_Q_URL = "https://pxdata.stat.fi/PXWeb/api/v1/fi/StatFin/ntp/statfin_ntp_pxt_132h.px"


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
    """
    JSON-stat2 → DataFrame (robusti järjestykselle).
    """
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

        # ✅ Tärkeä: jos index on dict (koodi -> järjestysnumero), järjestä sen mukaan.
        if isinstance(idx, dict) and idx:
            keys = [k for k, _ in sorted(idx.items(), key=lambda kv: kv[1])]
        elif isinstance(idx, list) and idx:
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


def fetch_gdp_quarterly_b1gmh_all_tiedot() -> pd.DataFrame:
    """
    Hakee taulukosta 132h:
    - Taloustoimi: B1GMH (Bruttokansantuote markkinahintaan)
    - Vuosineljännes: kaikki
    - Tiedot: kaikki

    Palauttaa sarakkeet: Vuosineljännes, Taloustoimi, Tiedot, Arvo
    """
    query = {
        "query": [
            {"code": "Taloustoimi", "selection": {"filter": "item", "values": ["B1GMH"]}},
            {"code": "Vuosineljännes", "selection": {"filter": "all", "values": ["*"]}},
            {"code": "Tiedot", "selection": {"filter": "all", "values": ["*"]}},
        ],
        "response": {"format": "json-stat2"},
    }
    return _post_px(GDP_Q_URL, query)


def pick_gdp_growth_yoy_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Poimii 'Kausitasoitettu ... volyymin muutos vuodentakaisesta, %' -sarjan.
    Palauttaa: Quarter(str), value(float)
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # Sarakenimet ovat suoraan dimensioiden nimet (suomeksi): Vuosineljännes, Taloustoimi, Tiedot
    needed = {"Vuosineljännes", "Taloustoimi", "Tiedot", "Arvo"}
    if not needed.issubset(set(df.columns)):
        return pd.DataFrame()

    f = df.dropna(subset=["Arvo"]).copy()

    # ✅ Etsi “vuodentakaisesta, %” ja “Kausitasoitettu”
    mask = (
        f["Tiedot"].astype(str).str.contains("vuodentakaisesta", case=False, na=False)
        & f["Tiedot"].astype(str).str.contains("%", case=False, na=False)
        & f["Tiedot"].astype(str).str.contains("Kausitasoit", case=False, na=False)
    )
    f = f[mask].copy()
    if f.empty:
        return pd.DataFrame()

    out = f[["Vuosineljännes", "Arvo"]].copy()
    out = out.rename(columns={"Vuosineljännes": "Quarter", "Arvo": "value"})
    out["Quarter_dt"] = pd.to_datetime(
        out["Quarter"].str.replace("Q", "-Q", regex=False),
        errors="coerce",
    )
    out = out.dropna(subset=["Quarter_dt"]).sort_values("Quarter_dt")
    return out[["Quarter", "Quarter_dt", "value"]]