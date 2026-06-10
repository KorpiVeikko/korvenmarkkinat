# services/oil_stats.py
from __future__ import annotations

import itertools

import pandas as pd
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


PXWEB_URL = "https://pxdata.stat.fi/PxWeb/api/v1/fi/StatFin/khi/11xx.px"

FUEL_NAME_MATCHES = {
    "Polttoöljy": ["kevyt polttoöljy", "light fuel oil"],
    "Diesel": ["diesel"],
    "Bensiini 95": ["bensiini 95", "petrol 95"],
    "Bensiini 98": ["bensiini 98", "petrol 98"],
}


def _build_session() -> requests.Session:
    session = requests.Session()

    retry = Retry(
        total=2,
        connect=2,
        read=2,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
    )

    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    session.headers.update(
        {"User-Agent": "Mozilla/5.0 (compatible; TaloudenSeuranta/1.0)"}
    )
    return session


def _norm(s: str) -> str:
    return (
        str(s)
        .lower()
        .replace("ä", "a")
        .replace("ö", "o")
        .replace("å", "a")
    )


def _px_selection(value: str | None) -> dict:
    if value is None or value == "*":
        return {"filter": "all", "values": ["*"]}
    return {"filter": "item", "values": [value]}


def _get_meta() -> dict:
    session = _build_session()
    r = session.get(PXWEB_URL, timeout=30)

    if r.status_code != 200:
        raise RuntimeError(
            "Polttoainehintojen metadata-haku epäonnistui.\n\n"
            f"URL: {PXWEB_URL}\n"
            f"Status: {r.status_code}\n"
            f"Response: {r.text[:1000]}"
        )

    payload = r.json()
    return payload if isinstance(payload, dict) else {}


def _find_var_code(meta: dict, needles: list[str]) -> str | None:
    for var in meta.get("variables") or []:
        code = str(var.get("code", ""))
        text = str(var.get("text", ""))
        combined = _norm(f"{code} {text}")

        if any(_norm(n) in combined for n in needles):
            return code

    return None


def _var_by_code(meta: dict, code: str | None) -> dict | None:
    if not code:
        return None

    for var in meta.get("variables") or []:
        if str(var.get("code", "")) == str(code):
            return var

    return None


def _pick_value(meta: dict, var_code: str | None, needles: list[str]) -> str | None:
    var = _var_by_code(meta, var_code)
    if not var:
        return None

    values = var.get("values") or []
    texts = var.get("valueTexts") or []

    if not values:
        return None

    for val, txt in zip(values, texts):
        txt_norm = _norm(txt)
        if any(_norm(n) in txt_norm for n in needles):
            return str(val)

    return None


def _pick_fuel_codes(meta: dict, fuel_code: str) -> dict[str, str]:
    var = _var_by_code(meta, fuel_code)
    if not var:
        return {}

    values = [str(x) for x in var.get("values", [])]
    texts = [str(x) for x in var.get("valueTexts", [])]

    out: dict[str, str] = {}

    for wanted_name, needles in FUEL_NAME_MATCHES.items():
        for val, txt in zip(values, texts):
            txt_norm = _norm(txt)

            if any(_norm(n) in txt_norm for n in needles):
                out[val] = wanted_name
                break

    return out


def _latest_time_values(meta: dict, time_code: str, months: int) -> list[str]:
    var = _var_by_code(meta, time_code)
    if not var:
        return []

    values = [str(x) for x in var.get("values", [])]
    return values[-months:] if len(values) > months else values


def _parse_jsonstat2(payload: dict) -> pd.DataFrame:
    if not isinstance(payload, dict) or "dimension" not in payload:
        return pd.DataFrame()

    dim = payload.get("dimension", {})
    ids = payload.get("id") or dim.get("id")
    values = payload.get("value")

    if not ids or values is None:
        return pd.DataFrame()

    levels: list[list[str]] = []

    for dim_id in ids:
        d = dim.get(dim_id, {})
        cat = d.get("category") or {}
        index = cat.get("index") or {}

        if isinstance(index, dict) and index:
            keys = [k for k, _ in sorted(index.items(), key=lambda kv: kv[1])]
        elif isinstance(index, list) and index:
            keys = index
        else:
            keys = list((cat.get("label") or {}).keys())

        levels.append([str(k) for k in keys])

    combos = list(itertools.product(*levels))

    if isinstance(values, list):
        if len(values) != len(combos):
            return pd.DataFrame()
        value_list = values
    elif isinstance(values, dict):
        value_list = [None] * len(combos)
        for k, v in values.items():
            try:
                idx = int(k)
            except Exception:
                continue
            if 0 <= idx < len(value_list):
                value_list[idx] = v
    else:
        return pd.DataFrame()

    df = pd.DataFrame(combos, columns=[str(x) for x in ids])
    df["Value"] = pd.to_numeric(value_list, errors="coerce")

    return df


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def fetch_finland_fuel_prices_debug(years: int = 5) -> tuple[pd.DataFrame, str | None]:
    try:
        months = max(12, years * 12 + 2)

        meta = _get_meta()

        time_code = _find_var_code(meta, ["kuukausi", "month", "timeperiod_m"])
        fuel_code = _find_var_code(meta, ["hyödyke", "hyodyke", "commodity", "varat", "varor"])
        info_code = _find_var_code(meta, ["tiedot", "information", "contentscode", "uppgifter"])

        if not time_code or not fuel_code or not info_code:
            return (
                pd.DataFrame(),
                "Polttoainehintojen muuttujakoodeja ei löytynyt. "
                f"time_code={time_code}, fuel_code={fuel_code}, info_code={info_code}",
            )

        time_values = _latest_time_values(meta, time_code, months)
        fuel_codes = _pick_fuel_codes(meta, fuel_code)
        info_value = _pick_value(meta, info_code, ["keskihinta", "average price", "genomsnittligt pris"])

        if not time_values:
            return pd.DataFrame(), "Kuukausiarvoja ei löytynyt metadatasta."

        if not fuel_codes:
            return pd.DataFrame(), "Polttoainekoodeja ei löytynyt metadatasta."

        if not info_value:
            return pd.DataFrame(), "Keskihinta-arvoa ei löytynyt metadatasta."

        payload = {
            "query": [
                {
                    "code": time_code,
                    "selection": {
                        "filter": "item",
                        "values": time_values,
                    },
                },
                {
                    "code": fuel_code,
                    "selection": {
                        "filter": "item",
                        "values": list(fuel_codes.keys()),
                    },
                },
                {
                    "code": info_code,
                    "selection": _px_selection(info_value),
                },
            ],
            "response": {"format": "json-stat2"},
        }

        session = _build_session()
        res = session.post(PXWEB_URL, json=payload, timeout=30)

        if res.status_code != 200:
            return (
                pd.DataFrame(),
                "Polttoainehintojen POST epäonnistui.\n\n"
                f"URL: {PXWEB_URL}\n"
                f"Status: {res.status_code}\n\n"
                f"Query:\n{payload}\n\n"
                f"Response:\n{res.text[:1000]}",
            )

        raw = _parse_jsonstat2(res.json())
        if raw.empty:
            return pd.DataFrame(), "Tilastokeskuksen vastausrakenne ei tuottanut rivejä."

        raw["Date"] = pd.to_datetime(
            raw[time_code].astype(str).str.replace("M", "-", regex=False) + "-01",
            errors="coerce",
        )
        raw["Fuel"] = raw[fuel_code].astype(str).map(fuel_codes)
        raw["Value"] = pd.to_numeric(raw["Value"], errors="coerce")

        df = (
            raw.dropna(subset=["Date", "Fuel", "Value"])
            .sort_values(["Fuel", "Date"])
            .reset_index(drop=True)
        )

        if df.empty:
            return pd.DataFrame(), "Polttoainehintadata jäi tyhjäksi siivouksen jälkeen."

        max_date = df["Date"].max()
        cutoff = max_date - pd.DateOffset(years=years)
        df = df[df["Date"] >= cutoff].copy()

        if df.empty:
            return pd.DataFrame(), f"Polttoainehintadata jäi tyhjäksi {years} vuoden rajauksen jälkeen."

        df["Month"] = df["Date"].dt.strftime("%Y-%m")

        return df[["Date", "Month", "Fuel", "Value"]], None

    except requests.exceptions.Timeout as e:
        return pd.DataFrame(), f"Polttoainehintojen haku aikakatkaistiin: {e!r}"

    except requests.exceptions.ConnectionError as e:
        return pd.DataFrame(), f"Polttoainehintojen haku epäonnistui verkkoyhteysvirheeseen: {e!r}"

    except requests.exceptions.HTTPError as e:
        status = getattr(e.response, "status_code", "tuntematon")
        body = ""
        try:
            body = e.response.text[:500]
        except Exception:
            pass
        return pd.DataFrame(), f"Polttoainehintojen haku palautti HTTP-virheen {status}. Vastauksen alku: {body}"

    except Exception as e:
        return pd.DataFrame(), f"Polttoainehintojen haku epäonnistui odottamattomaan virheeseen: {e!r}"


def fetch_finland_fuel_prices(years: int = 5) -> pd.DataFrame:
    df, _ = fetch_finland_fuel_prices_debug(years=years)
    return df