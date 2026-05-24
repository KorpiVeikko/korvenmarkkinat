from __future__ import annotations

import pandas as pd
import requests
import streamlit as st
from pyjstat import pyjstat


CPI_TOTAL_YOY_URL = "https://pxdata.stat.fi/PXWeb/api/v1/fi/StatFin/khi/statfin_khi_pxt_122p.px"
CPI_INDEX_URL = "https://pxdata.stat.fi/PXWeb/api/v1/fi/StatFin/khi/statfin_khi_pxt_15b5.px"


def _px_get_metadata(url: str) -> list[dict]:
    r = requests.get(url, timeout=30)

    if r.status_code != 200:
        raise RuntimeError(
            f"Metadata-haku epäonnistui: {url}\n"
            f"Status: {r.status_code}\n"
            f"Response: {r.text}"
        )

    meta = r.json()
    return meta if isinstance(meta, list) else meta.get("variables", [])


def _px_post(url: str, query: dict) -> pd.DataFrame:
    r = requests.post(url, json=query, timeout=30)

    if r.status_code != 200:
        raise RuntimeError(
            "PXWeb POST epäonnistui.\n\n"
            f"URL: {url}\n"
            f"Status: {r.status_code}\n\n"
            f"Query:\n{query}\n\n"
            f"Response:\n{r.text}"
        )

    return pd.DataFrame(pyjstat.from_json_stat(r.json(), naming="id")[0])


def _time_var(variables: list[dict]) -> dict:
    for var in variables:
        if var.get("time") is True:
            return var

    raise ValueError("Aikamuuttujaa ei löytynyt.")


def _month_to_date(value: str) -> pd.Timestamp:
    s = str(value).strip()

    if "M" in s:
        return pd.to_datetime(s.replace("M", "-") + "-01", errors="coerce")

    return pd.to_datetime(s, errors="coerce")


def _latest_months(var: dict, months: int = 96) -> list[str]:
    values = [str(x) for x in var.get("values", [])]
    return values[-months:] if len(values) > months else values


def _find_var_by_text(
    variables: list[dict],
    terms: list[str],
    skip: set[str],
) -> dict:
    for var in variables:
        code = str(var.get("code"))

        if code in skip:
            continue

        texts = " ".join(str(x).lower() for x in var.get("valueTexts", []))

        if all(term.lower() in texts for term in terms):
            return var

    raise ValueError(f"Muuttujaa ei löytynyt termeillä: {terms}")


def _find_code(var: dict, *terms: str) -> str | None:
    for code, text in zip(var.get("values", []), var.get("valueTexts", [])):
        txt = str(text).lower()

        if all(term.lower() in txt for term in terms):
            return str(code)

    return None


# lisää aiempaan tiedostoon: sarjoihin myös Indeksi, 3v ja 5v muutos

def _add_long_term_changes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["Indeksi"] = pd.to_numeric(out["Indeksi"], errors="coerce")
    out = out.dropna(subset=["Date", "Sarja", "Indeksi"]).sort_values(["Sarja", "Date"])

    rows = []

    for _, g in out.groupby("Sarja"):
        g = g.sort_values("Date").copy()
        latest = g.iloc[-1]
        latest_date = latest["Date"]
        latest_index = float(latest["Indeksi"])

        def pct_from_years(years: int) -> float | None:
            target = latest_date - pd.DateOffset(years=years)
            prev = g[g["Date"] <= target]
            if prev.empty:
                return None
            prev_index = float(prev.iloc[-1]["Indeksi"])
            if prev_index == 0:
                return None
            return (latest_index / prev_index - 1.0) * 100.0

        row = latest.to_dict()
        row["Muutos_3v"] = pct_from_years(3)
        row["Muutos_5v"] = pct_from_years(5)
        rows.append(row)

    return pd.DataFrame(rows)


def _fetch_category_index(months: int = 108) -> pd.DataFrame:
    variables = _px_get_metadata(CPI_INDEX_URL)

    time_var = _time_var(variables)
    time_code = str(time_var["code"])
    time_values = _latest_months(time_var, months)

    category_var = _find_var_by_text(
        variables,
        terms=["elintarvikkeet"],
        skip={time_code},
    )
    category_code = str(category_var["code"])

    info_var = _find_var_by_text(
        variables,
        terms=["indeksi"],
        skip={time_code, category_code},
    )
    info_code = str(info_var["code"])
    info_value = _find_code(info_var, "indeksi") or str(info_var.get("values", [])[0])

    category_values = category_var.get("values", [])
    first_category_code = str(category_values[0]) if category_values else None

    raw_picks = {
        "Virallinen inflaatio": (
            _find_code(category_var, "kokonaisindeksi")
            or _find_code(category_var, "kuluttajahintaindeksi")
            or _find_code(category_var, "kaikki")
            or first_category_code
        ),
        "Ruokainflaatio": _find_code(category_var, "elintarvikkeet"),
        "Energia": (
            _find_code(category_var, "sähkö", "kaasu")
            or _find_code(category_var, "sähkö")
            or _find_code(category_var, "asuminen", "energia")
        ),
        "Polttoaineet": (
            _find_code(category_var, "polttoaineet")
            or _find_code(category_var, "polttoaine")
        ),
    }
    

    category_picks: dict[str, str] = {}
    used_codes: set[str] = set()

    for name, code in raw_picks.items():
        if code is None:
            continue

        if code in used_codes:
            continue

        category_picks[name] = code
        used_codes.add(code)

    if not category_picks:
        raise ValueError("Kulutusluokkia ei löytynyt 15b5-taulusta.")

    code_to_name = {code: name for name, code in category_picks.items()}

    query = {
        "query": [
            {
                "code": category_code,
                "selection": {
                    "filter": "item",
                    "values": list(category_picks.values()),
                },
            },
            {
                "code": time_code,
                "selection": {
                    "filter": "item",
                    "values": time_values,
                },
            },
            {
                "code": info_code,
                "selection": {
                    "filter": "item",
                    "values": [info_value],
                },
            },
        ],
        "response": {"format": "json-stat2"},
    }

    df = _px_post(CPI_INDEX_URL, query)
    df = df.rename(columns={"value": "Indeksi"})

    df["Date"] = df[time_code].map(_month_to_date)
    df["Sarja"] = df[category_code].astype(str).map(code_to_name)
    df["Indeksi"] = pd.to_numeric(df["Indeksi"], errors="coerce")

    df = df.dropna(subset=["Date", "Sarja", "Indeksi"])
    df = df.sort_values(["Sarja", "Date"]).reset_index(drop=True)

    df["Inflaatio"] = df.groupby("Sarja")["Indeksi"].pct_change(12) * 100.0

    return df[["Date", "Sarja", "Indeksi", "Inflaatio"]].dropna(subset=["Date", "Sarja", "Indeksi"])




@st.cache_data(
    ttl=60 * 60 * 6,
    show_spinner="Haetaan tarkempia inflaatiomittareita…",
)
def load_inflation_pressure_bundle() -> dict:
    try:
        combined = _fetch_category_index()
        latest = _add_long_term_changes(combined)

        return {
            "ok": True,
            "series": combined,
            "latest": latest,
            "error": None,
        }

    except Exception as e:
        return {
            "ok": False,
            "series": pd.DataFrame(),
            "latest": pd.DataFrame(),
            "error": repr(e),
        }