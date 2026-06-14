# services/energy_pxweb.py
from __future__ import annotations

import itertools
from typing import Iterable

import pandas as pd
import requests


ELECTRICITY_URL = "https://pxdata.stat.fi/PxWeb/api/v1/fi/StatFin/ehk/12su.px"
HOUSEHOLD_ELECTRICITY_PRICE_URL = "https://pxdata.stat.fi/PxWeb/api/v1/fi/StatFin/ehi/13rb.px"


ELECTRICITY_SERIES_VALUES = [
    "SSS",
    "1",
    "1.1",
    "1.2",
    "1.3",
    "1.4",
    "1.5",
    "1.5.1",
    "1.5.2",
    "1.5.3",
    "1.6",
    "2",
    "2.1",
    "2.2",
    "2.3",
    "2.4",
]

PRICE_COMPONENT_VALUES = ["A", "B", "C", "SSS"]


def _label_map(meta: dict, var_code: str | None) -> dict[str, str]:
    var = _var_by_code(meta, var_code)
    if not var:
        return {}

    values = [str(x) for x in var.get("values", [])]
    texts = [str(x) for x in var.get("valueTexts", [])]

    return dict(zip(values, texts))


def _pick_electricity_content_value(meta: dict, var_code: str | None) -> str | None:
    var = _var_by_code(meta, var_code)
    if not var:
        return None

    values = [str(x) for x in var.get("values", [])]
    texts = [str(x) for x in var.get("valueTexts", [])]

    for val, txt in zip(values, texts):
        combined = _norm(f"{val} {txt}")
        if "gwh" in combined or "gigawattitunti" in combined:
            return val

    for val, txt in zip(values, texts):
        combined = _norm(f"{val} {txt}")
        if "maara" in combined or "määrä" in combined:
            return val

    return values[0] if values else None


def _norm(s: str) -> str:
    return str(s).lower().replace("ä", "a").replace("ö", "o").replace("å", "a")


def _month_values(start_year: int = 2020, end_year: int = 2025) -> list[str]:
    values: list[str] = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            values.append(f"{year}M{month:02d}")
    return values


def _dedupe_columns(cols: Iterable[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []

    for col in cols:
        base = str(col)
        if base not in seen:
            seen[base] = 0
            out.append(base)
        else:
            seen[base] += 1
            out.append(f"{base}__{seen[base]}")
    return out


def _get_meta(url: str) -> dict:
    r = requests.get(url, timeout=30)

    if r.status_code != 200:
        raise RuntimeError(
            "PXWeb metadata-haku epäonnistui.\n\n"
            f"URL: {url}\n"
            f"Status: {r.status_code}\n"
            f"Response:\n{r.text[:1000]}"
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
        if str(var.get("code")) == str(code):
            return var

    return None


def _pick_existing_values(meta: dict, var_code: str | None, wanted: list[str]) -> list[str]:
    var = _var_by_code(meta, var_code)
    if not var:
        return []

    available = [str(x) for x in var.get("values", [])]
    return [v for v in wanted if v in available]


def _latest_month_values(meta: dict, time_code: str | None, start_year: int, end_year: int) -> list[str]:
    var = _var_by_code(meta, time_code)
    if not var:
        return []

    available = [str(x) for x in var.get("values", [])]
    wanted = set(_month_values(start_year, end_year))
    selected = [v for v in available if v in wanted]

    if selected:
        return selected

    return available[-72:]


def _parse_jsonstat2(payload: dict) -> pd.DataFrame:
    if not isinstance(payload, dict) or "dimension" not in payload:
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
        labels = cat.get("label") or {}

        if isinstance(idx, dict) and idx:
            keys = [k for k, _ in sorted(idx.items(), key=lambda kv: kv[1])]
        elif isinstance(idx, list) and idx:
            keys = idx
        else:
            keys = list(labels.keys())

        # Tärkeää: palautetaan koodit, ei label-tekstit.
        dim_levels.append([str(k) for k in keys])

    combos = list(itertools.product(*dim_levels))

    if isinstance(values, list):
        if len(combos) != len(values):
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

    cols = _dedupe_columns([str(x) for x in ids])
    df = pd.DataFrame(combos, columns=cols)
    df["Arvo"] = pd.to_numeric(value_list, errors="coerce")
    df.columns = _dedupe_columns(df.columns)
    return df


def _post_px(url: str, query: dict, timeout: int = 45) -> pd.DataFrame:
    response = requests.post(url, json=query, timeout=timeout)

    if response.status_code != 200:
        raise RuntimeError(
            "PXWeb POST epäonnistui.\n\n"
            f"URL: {url}\n"
            f"Status: {response.status_code}\n\n"
            f"Query:\n{query}\n\n"
            f"Response:\n{response.text[:1000]}"
        )

    return _parse_jsonstat2(response.json())


def _label_map(meta: dict, var_code: str | None) -> dict[str, str]:
    var = _var_by_code(meta, var_code)
    if not var:
        return {}

    values = [str(x) for x in var.get("values", [])]
    texts = [str(x) for x in var.get("valueTexts", [])]

    return dict(zip(values, texts))


def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    out.columns = _dedupe_columns(out.columns)

    time_col = None
    for col in out.columns:
        if str(col).strip().lower() in (
            "kuukausi",
            "aika",
            "time",
            "timeperiod",
            "timeperiod_m",
        ):
            time_col = col
            break

    if time_col is None:
        time_col = out.columns[0]

    out["Aika"] = out[time_col].astype(str).str.strip()

    month_match = out["Aika"].str.extract(r"^(?P<y>\d{4})M(?P<m>\d{2})$")
    out["Vuosi_num"] = pd.to_numeric(month_match["y"], errors="coerce")
    out["Kuukausi_num"] = pd.to_numeric(month_match["m"], errors="coerce")
    out["Aika_dt"] = pd.to_datetime(
        out["Vuosi_num"].astype("Int64").astype(str)
        + "-"
        + out["Kuukausi_num"].astype("Int64").astype(str).str.zfill(2)
        + "-01",
        errors="coerce",
    )

    return out


def fetch_electricity_production_consumption(
    start_year: int = 2020,
    end_year: int = 2025,
) -> pd.DataFrame:
    meta = _get_meta(ELECTRICITY_URL)

    time_code = _find_var_code(meta, ["kuukausi", "timeperiod", "time"])
    series_code = _find_var_code(meta, ["sähkön tuotanto", "sahkon tuotanto", "hankinta", "electricity"])
    content_code = _find_var_code(meta, ["tiedot", "contentscode", "contentcode"])

    if not time_code or not series_code:
        raise RuntimeError(
            "Sähkötilaston muuttujakoodeja ei löytynyt.\n\n"
            f"time_code={time_code}\n"
            f"series_code={series_code}\n"
            f"content_code={content_code}"
        )

    time_values = _latest_month_values(meta, time_code, start_year, end_year)
    series_values = _pick_existing_values(meta, series_code, ELECTRICITY_SERIES_VALUES)
    content_value = _pick_electricity_content_value(meta, content_code)

    if not time_values or not series_values:
        raise RuntimeError(
            "Sähkötilaston valittuja arvoja ei löytynyt.\n\n"
            f"time_values={time_values[:5]} ... {len(time_values)} kpl\n"
            f"series_values={series_values}"
        )

    query_items = [
        {
            "code": time_code,
            "selection": {"filter": "item", "values": time_values},
        },
        {
            "code": series_code,
            "selection": {"filter": "item", "values": series_values},
        },
    ]

    if content_code and content_value:
        query_items.append(
            {
                "code": content_code,
                "selection": {"filter": "item", "values": [content_value]},
            }
        )

    query = {
        "query": query_items,
        "response": {"format": "json-stat2"},
    }

    df = _post_px(ELECTRICITY_URL, query)
    df = add_time_columns(df)

    series_labels = _label_map(meta, series_code)

    if series_code in df.columns:
        df["Sähkön tuotanto/hankinta"] = (
            df[series_code]
            .astype(str)
            .map(lambda x: series_labels.get(x, x))
        )
        df = df.drop(columns=[series_code], errors="ignore")

    if content_code and content_code in df.columns:
        df = df.rename(columns={content_code: "Tiedot"})

    return df


def fetch_household_electricity_prices() -> pd.DataFrame:
    meta = _get_meta(HOUSEHOLD_ELECTRICITY_PRICE_URL)

    component_code = _find_var_code(meta, ["hintakomponentti", "price component"])
    time_code = _find_var_code(meta, ["kuukausi", "neljännes", "quarter", "timeperiod", "time"])

    if not component_code:
        raise RuntimeError("Sähkön hintakomponentin muuttujakoodia ei löytynyt.")

    component_values = _pick_existing_values(meta, component_code, PRICE_COMPONENT_VALUES)

    if not component_values:
        raise RuntimeError(
            "Sähkön hintakomponenttien arvoja ei löytynyt.\n\n"
            f"component_code={component_code}"
        )

    query_items = [
        {
            "code": component_code,
            "selection": {"filter": "item", "values": component_values},
        }
    ]

    if time_code:
        query_items.append(
            {
                "code": time_code,
                "selection": {"filter": "all", "values": ["*"]},
            }
        )

    query = {
        "query": query_items,
        "response": {"format": "json-stat2"},
    }

    df = _post_px(HOUSEHOLD_ELECTRICITY_PRICE_URL, query)
    df = add_time_columns(df)

    rename_map = {}

    if time_code and time_code in df.columns:
        rename_map[time_code] = "Kuukausi"

    if component_code and component_code in df.columns:
        rename_map[component_code] = "Hintakomponentti"

    if "contentscode" in df.columns:
        rename_map["contentscode"] = "Tiedot"

    energia_cols = [c for c in df.columns if str(c).startswith("energia_")]

    for col in energia_cols:
        if col not in rename_map:
            if "Kuluttajaryhmä" not in rename_map.values():
                rename_map[col] = "Kuluttajaryhmä"
            else:
                rename_map[col] = "Hintaluokka"

    df = df.rename(columns=rename_map)

    # Muutetaan koodit selkokielisiksi teksteiksi.
    if "Hintakomponentti" in df.columns:
        component_labels = _label_map(meta, component_code)
        df["Hintakomponentti"] = df["Hintakomponentti"].astype(str).replace(component_labels)

    if "Kuluttajaryhmä" in df.columns:
        original_col = next((c for c in energia_cols if c in _label_map(meta, c)), None)
        # fallback: käytetään ensimmäisen energia-sarakkeen label-map
        if original_col is None and energia_cols:
            original_col = energia_cols[0]

        group_labels = _label_map(meta, original_col)
        df["Kuluttajaryhmä"] = df["Kuluttajaryhmä"].astype(str).replace(group_labels)

    if "Hintaluokka" in df.columns and len(energia_cols) > 1:
        class_labels = _label_map(meta, energia_cols[1])
        df["Hintaluokka"] = df["Hintaluokka"].astype(str).replace(class_labels)

    if "Tiedot" in df.columns:
        info_labels = _label_map(meta, "contentscode")
        df["Tiedot"] = df["Tiedot"].astype(str).replace(info_labels)

    return df