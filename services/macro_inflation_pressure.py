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


def _fetch_total_yoy(months: int = 96) -> pd.DataFrame:
    variables = _px_get_metadata(CPI_TOTAL_YOY_URL)

    time_var = _time_var(variables)
    time_code = str(time_var["code"])
    time_values = _latest_months(time_var, months)

    query = {
        "query": [
            {
                "code": time_code,
                "selection": {
                    "filter": "item",
                    "values": time_values,
                },
            },
            {
                "code": "Tiedot",
                "selection": {
                    "filter": "item",
                    "values": ["Vuosimuutos"],
                },
            },
        ],
        "response": {"format": "json-stat2"},
    }

    df = _px_post(CPI_TOTAL_YOY_URL, query)
    df = df.rename(columns={"value": "Inflaatio"})

    df["Date"] = df[time_code].map(_month_to_date)
    df["Sarja"] = "Virallinen inflaatio"
    df["Inflaatio"] = pd.to_numeric(df["Inflaatio"], errors="coerce")

    return df[["Date", "Sarja", "Inflaatio"]].dropna()


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

    raw_picks = {
        "Ruokainflaatio": _find_code(category_var, "elintarvikkeet"),
        "Energia": (
            _find_code(category_var, "sähkö", "kaasu")
            or _find_code(category_var, "sähkö")
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

    return df[["Date", "Sarja", "Inflaatio"]].dropna()


def _build_household_pressure(series_df: pd.DataFrame) -> pd.DataFrame:
    if series_df is None or series_df.empty:
        return pd.DataFrame(columns=["Date", "Sarja", "Inflaatio"])

    pivot = series_df.pivot_table(
        index="Date",
        columns="Sarja",
        values="Inflaatio",
        aggfunc="last",
    ).sort_index()

    if "Ruokainflaatio" not in pivot.columns:
        return pd.DataFrame(columns=["Date", "Sarja", "Inflaatio"])

    food = pivot["Ruokainflaatio"]
    energy = pivot["Energia"] if "Energia" in pivot.columns else None
    fuel = pivot["Polttoaineet"] if "Polttoaineet" in pivot.columns else None

    if energy is not None and fuel is not None:
        pressure = 0.5 * food + 0.3 * energy + 0.2 * fuel
    elif energy is not None:
        pressure = 0.6 * food + 0.4 * energy
    elif fuel is not None:
        pressure = 0.7 * food + 0.3 * fuel
    else:
        pressure = food

    out = pressure.dropna().reset_index()
    out["Sarja"] = "Kotitalouspaine"
    out = out.rename(columns={0: "Inflaatio"})
    out["Inflaatio"] = pd.to_numeric(out["Inflaatio"], errors="coerce")

    return out[["Date", "Sarja", "Inflaatio"]].dropna()


@st.cache_data(
    ttl=60 * 60 * 6,
    show_spinner="Haetaan tarkempia inflaatiomittareita…",
)
def load_inflation_pressure_bundle() -> dict:
    try:
        total = _fetch_total_yoy()
        categories = _fetch_category_index()
        base = pd.concat([total, categories], ignore_index=True)

        pressure = _build_household_pressure(base)
        combined = pd.concat([base, pressure], ignore_index=True)

        latest = (
            combined.sort_values("Date")
            .groupby("Sarja", as_index=False)
            .tail(1)
            .sort_values("Sarja")
        )

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