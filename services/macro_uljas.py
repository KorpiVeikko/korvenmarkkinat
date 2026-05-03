# services/macro_uljas.py
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests
from urllib.parse import quote

ULJAS_API_URL = "https://uljas.tulli.fi/uljas/graph/api.aspx"

ULJAS_SITC_IFILE = "/DATABASE/01 ULKOMAANKAUPPATILASTOT/02 SITC/ULJAS_SITC"
ULJAS_SITC2_IFILE = "/DATABASE/01 ULKOMAANKAUPPATILASTOT/02 SITC/ULJAS_SITC2"

EXPORT_DIRECTION = "2"   # Vienti määrämaittain
IMPORT_DIRECTION = "1"   # Tuonti alkuperämaittain

CONTINENT_CODES = {"10", "21", "22", "31", "32", "41", "42", "43", "50"}

DEFAULT_START = "202001"
DEFAULT_END = "202612"


# ------------------------------------------------------------
# Low-level helpers
# ------------------------------------------------------------
def _verti_escape(s: str) -> str:
    return (
        str(s)
        .replace(" ", "*;")
        .replace("ä", "*228;")
        .replace("ö", "*246;")
        .replace("å", "*229;")
        .replace("Ä", "*196;")
        .replace("Ö", "*214;")
        .replace("Å", "*197;")
    )


def _enc(v: str) -> str:
    return quote(_verti_escape(v), safe="*/;=:+-._()/[]")


def _build_body(params: dict) -> bytes:
    body = "&".join(
        f"{quote(str(k), safe='')}={_enc(str(v))}"
        for k, v in params.items()
        if v is not None
    )
    return body.encode("utf-8")


def _post_text(params: dict, timeout: int = 30) -> str:
    r = requests.post(
        ULJAS_API_URL,
        data=_build_body(params),
        headers={
            "User-Agent": "TaloudenSeuranta/1.0",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        },
        timeout=timeout,
    )
    r.raise_for_status()
    return r.content.decode("utf-8-sig")


def _post_json(params: dict, timeout: int = 30) -> Any:
    txt = _post_text(params, timeout=timeout)
    return json.loads(txt)


# ------------------------------------------------------------
# Metadata helpers
# ------------------------------------------------------------
@dataclass
class ClassItem:
    code: str
    text: str


def class_values(ifile: str, class_name: str, lang: str = "fi") -> list[ClassItem]:
    try:
        data = _post_json(
            {
                "lang": lang,
                "atype": "class",
                "konv": "json",
                "ifile": ifile,
                "class": class_name,
            },
            timeout=30,
        )
    except Exception:
        return []

    out: list[ClassItem] = []
    if isinstance(data, dict):
        items = data.get("classification") or []
        for block in items:
            for row in block.get("class", []) or []:
                code = row.get("code")
                text = row.get("text")
                if code is not None:
                    out.append(
                        ClassItem(
                            code=str(code),
                            text=str(text) if text is not None else str(code),
                        )
                    )
    return out


def _join_value_list(values: list[str]) -> str:
    return '"'.join([str(v) for v in values if str(v).strip()])


def _yyyymm_range(start_yyyymm: str, end_yyyymm: str) -> list[str]:
    start = pd.Period(start_yyyymm, freq="M")
    end = pd.Period(end_yyyymm, freq="M")
    if end < start:
        return []
    return [p.strftime("%Y%m") for p in pd.period_range(start, end, freq="M")]


def _resolve_time_codes(
    months: int | None = None,
    start_yyyymm: str | None = None,
    end_yyyymm: str | None = None,
) -> list[str]:
    if start_yyyymm and end_yyyymm:
        return _yyyymm_range(start_yyyymm, end_yyyymm)

    if months is None:
        months = 24

    months = max(int(months), 1)
    end = pd.Period(DEFAULT_END, freq="M")
    start = end - (months - 1)
    return [p.strftime("%Y%m") for p in pd.period_range(start, end, freq="M")]


# ------------------------------------------------------------
# Parsing helpers
# ------------------------------------------------------------
def _yyyymm_to_dt(s: str) -> pd.Timestamp:
    s = str(s).strip()
    if len(s) == 6 and s.isdigit():
        return pd.to_datetime(f"{s[:4]}-{s[4:6]}-01", errors="coerce")
    return pd.to_datetime(s, errors="coerce")


def _rows_to_df(rows: Any) -> pd.DataFrame:
    if isinstance(rows, dict):
        rows = rows.get("data") or rows.get("rows") or rows.get("dataset") or []

    if not isinstance(rows, list):
        return pd.DataFrame()

    out_rows = []
    for row in rows:
        if not isinstance(row, dict):
            continue

        keys = row.get("keys") or []
        vals = row.get("vals") or []
        value = vals[0] if vals else None

        if len(keys) != 4:
            continue

        out_rows.append(
            {
                "code": str(keys[0]),
                "aika": str(keys[1]),
                "geo": str(keys[2]),
                "suunta": str(keys[3]),
                "value": pd.to_numeric(value, errors="coerce"),
            }
        )

    df = pd.DataFrame(out_rows)
    if df.empty:
        return df

    df["Aika_dt"] = df["aika"].map(_yyyymm_to_dt)
    return df


def _empty_debug(ifiles: list[str], geo_class_name: str, geo_code: str, direction: str, time_codes: list[str]) -> dict:
    return {
        "ok_ifile": None,
        "ok_ifiles": [],
        "why_failed": [],
        "time_codes": time_codes,
        "geo_class_name": geo_class_name,
        "geo_code": geo_code,
        "direction": direction,
        "ifiles": ifiles,
        "years": [],
    }


def _fetch_trade_raw_single(
    *,
    ifile: str,
    time_codes: list[str],
    lang: str,
    sitc_code: str,
    geo_class_name: str,
    geo_code: str,
    direction: str,
    product_class_name: str = "Tavaraluokitus SITC1",
    timeout: int = 60,
) -> tuple[pd.DataFrame, str | None]:
    try:
        data = _post_json(
            {
                "lang": lang,
                "atype": "data",
                "konv": "json",
                "ifile": ifile,
                "select": "codes",
                product_class_name: sitc_code,
                "Aika": _join_value_list(time_codes),
                geo_class_name: geo_code,
                "Suunta": direction,
                "Indikaattorit": "V1",
            },
            timeout=timeout,
        )

        df = _rows_to_df(data)
        if df.empty:
            return pd.DataFrame(), f"{ifile}: kysely palautti tyhjän aineiston"

        df = df.dropna(subset=["Aika_dt", "value"]).copy()
        if df.empty:
            return pd.DataFrame(), f"{ifile}: aineistossa ei ollut numeerisia havaintoja"

        df["ifile"] = ifile
        return df, None

    except Exception as ex:
        return pd.DataFrame(), f"{ifile}: {type(ex).__name__}: {ex}"


def _fetch_trade_raw(
    *,
    months: int,
    lang: str,
    sitc_code: str,
    geo_class_name: str,
    geo_code: str,
    direction: str,
    product_class_name: str = "Tavaraluokitus SITC1",
    timeout: int = 60,
    start_yyyymm: str | None = None,
    end_yyyymm: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    months = max(int(months), 84)
    time_codes = _resolve_time_codes(
        months=months,
        start_yyyymm=start_yyyymm or DEFAULT_START,
        end_yyyymm=end_yyyymm or DEFAULT_END,
    )

    ifiles = [ULJAS_SITC_IFILE, ULJAS_SITC2_IFILE]
    debug = _empty_debug(ifiles, geo_class_name, geo_code, direction, time_codes)

    frames: list[pd.DataFrame] = []
    for ifile in ifiles:
        df, err = _fetch_trade_raw_single(
            ifile=ifile,
            time_codes=time_codes,
            lang=lang,
            sitc_code=sitc_code,
            geo_class_name=geo_class_name,
            geo_code=geo_code,
            direction=direction,
            product_class_name=product_class_name,
            timeout=timeout,
        )
        if err:
            debug["why_failed"].append(err)
        if not df.empty:
            frames.append(df)
            debug["ok_ifiles"].append(ifile)

    if not frames:
        debug["why_failed"].append("Yksikään ifile ei palauttanut havaintoja.")
        return pd.DataFrame(), debug

    out = pd.concat(frames, ignore_index=True)

    out["ifile_rank"] = out["ifile"].map(
        {
            ULJAS_SITC_IFILE: 1,
            ULJAS_SITC2_IFILE: 2,
        }
    ).fillna(0)

    out = (
        out.sort_values(["code", "aika", "geo", "suunta", "ifile_rank"])
        .drop_duplicates(subset=["code", "aika", "geo", "suunta"], keep="last")
        .drop(columns=["ifile_rank"])
        .sort_values(["Aika_dt", "code", "geo"])
        .reset_index(drop=True)
    )

    debug["ok_ifile"] = ", ".join(debug["ok_ifiles"])
    debug["years"] = sorted(out["Aika_dt"].dt.year.dropna().unique().tolist()) if not out.empty else []
    return out, debug


# ------------------------------------------------------------
# Legacy own grouping from SITC2
# ------------------------------------------------------------
FOOD_CODES = {"00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "11", "12"}
FOREST_CODES = {"24", "25", "63", "64"}
CHEM_CODES = {"51", "52", "53", "54", "55", "56", "57", "58", "59"}
ENERGY_CODES = {"33", "34", "35"}
METAL_CODES = {"67", "68", "69"}
MACHINERY_CODES = {"71", "72", "73", "74", "78", "79"}
ELECTRONICS_CODES = {"75", "76", "77"}


def _map_sitc2_to_legacy_group(code: str) -> str:
    code = str(code).strip()

    if code in FOOD_CODES:
        return "Elintarvikkeet & maatalous"
    if code in FOREST_CODES:
        return "Metsäteollisuus"
    if code in CHEM_CODES:
        return "Kemianteollisuus"
    if code in ENERGY_CODES:
        return "Energiatuotteet"
    if code in METAL_CODES:
        return "Metallit & metallituotteet"
    if code in MACHINERY_CODES:
        return "Koneet & laitteet"
    if code in ELECTRONICS_CODES:
        return "Elektroniikka & sähköteollisuus"
    return "Muut"


# ------------------------------------------------------------
# Label helpers
# ------------------------------------------------------------
def _label_map_for_class(class_name: str, lang: str = "fi") -> dict[str, str]:
    out: dict[str, str] = {}

    for ifile in [ULJAS_SITC_IFILE, ULJAS_SITC2_IFILE]:
        items = class_values(ifile, class_name, lang=lang)
        for item in items:
            txt = item.text or item.code
            if ") " in txt:
                txt = txt.split(") ", 1)[1]
            out[item.code] = txt

    return out



def _continent_code_to_name_map(lang: str = "fi") -> dict[str, str]:
    return _label_map_for_class("Maanosat ja ryhmiä", lang=lang)


# ------------------------------------------------------------
# Public fetchers: products
# ------------------------------------------------------------
def fetch_exports_products(months: int = 24, lang: str = "fi") -> tuple[pd.DataFrame, dict]:
    raw, debug = _fetch_trade_raw(
        months=max(months, 84),
        lang=lang,
        sitc_code="=ALL",
        geo_class_name="Maa",
        geo_code="AA",
        direction=EXPORT_DIRECTION,
        product_class_name="Tavaraluokitus SITC2",
        timeout=60,
    )
    if raw.empty:
        return pd.DataFrame(), debug

    raw["code"] = raw["code"].astype(str).str.strip()
    raw = raw[raw["code"].str.fullmatch(r"\d{2}")].copy()
    raw["Tuoteryhmä"] = raw["code"].map(_map_sitc2_to_legacy_group)

    out = (
        raw.groupby(["Aika_dt", "Tuoteryhmä"], as_index=False)["value"]
        .sum()
        .rename(columns={"value": "Vienti_eur"})
        .sort_values(["Aika_dt", "Tuoteryhmä"])
        .reset_index(drop=True)
    )
    debug["years"] = sorted(out["Aika_dt"].dt.year.dropna().unique().tolist()) if not out.empty else []
    return out, debug


def fetch_imports_products(months: int = 24, lang: str = "fi") -> tuple[pd.DataFrame, dict]:
    raw, debug = _fetch_trade_raw(
        months=max(months, 84),
        lang=lang,
        sitc_code="=ALL",
        geo_class_name="Maa",
        geo_code="AA",
        direction=IMPORT_DIRECTION,
        product_class_name="Tavaraluokitus SITC2",
        timeout=60,
    )
    if raw.empty:
        return pd.DataFrame(), debug

    raw["code"] = raw["code"].astype(str).str.strip()
    raw = raw[raw["code"].str.fullmatch(r"\d{2}")].copy()
    raw["Tuoteryhmä"] = raw["code"].map(_map_sitc2_to_legacy_group)

    out = (
        raw.groupby(["Aika_dt", "Tuoteryhmä"], as_index=False)["value"]
        .sum()
        .rename(columns={"value": "Tuonti_eur"})
        .sort_values(["Aika_dt", "Tuoteryhmä"])
        .reset_index(drop=True)
    )
    debug["years"] = sorted(out["Aika_dt"].dt.year.dropna().unique().tolist()) if not out.empty else []
    return out, debug


# ------------------------------------------------------------
# Public fetchers: regions / continents
# ------------------------------------------------------------
def fetch_exports_regions(months: int = 24, lang: str = "fi") -> tuple[pd.DataFrame, dict]:
    raw, debug = _fetch_trade_raw(
        months=max(months, 84),
        lang=lang,
        sitc_code="0-9",
        geo_class_name="Maanosat ja ryhmiä",
        geo_code="=ALL",
        direction=EXPORT_DIRECTION,
        timeout=60,
    )
    if raw.empty:
        return pd.DataFrame(), debug

    raw = raw[raw["geo"].isin(CONTINENT_CODES)].copy()

    geo_map = _continent_code_to_name_map(lang=lang)
    raw["Alue"] = raw["geo"].map(lambda x: geo_map.get(x, x))

    out = (
        raw.groupby(["Aika_dt", "Alue"], as_index=False)["value"]
        .sum()
        .rename(columns={"value": "Vienti_eur"})
        .sort_values(["Aika_dt", "Alue"])
        .reset_index(drop=True)
    )
    debug["years"] = sorted(out["Aika_dt"].dt.year.dropna().unique().tolist()) if not out.empty else []
    return out, debug


def fetch_imports_regions(months: int = 24, lang: str = "fi") -> tuple[pd.DataFrame, dict]:
    raw, debug = _fetch_trade_raw(
        months=max(months, 84),
        lang=lang,
        sitc_code="0-9",
        geo_class_name="Maanosat ja ryhmiä",
        geo_code="=ALL",
        direction=IMPORT_DIRECTION,
        timeout=60,
    )
    if raw.empty:
        return pd.DataFrame(), debug

    raw = raw[raw["geo"].isin(CONTINENT_CODES)].copy()

    geo_map = _continent_code_to_name_map(lang=lang)
    raw["Alue"] = raw["geo"].map(lambda x: geo_map.get(x, x))

    out = (
        raw.groupby(["Aika_dt", "Alue"], as_index=False)["value"]
        .sum()
        .rename(columns={"value": "Tuonti_eur"})
        .sort_values(["Aika_dt", "Alue"])
        .reset_index(drop=True)
    )
    debug["years"] = sorted(out["Aika_dt"].dt.year.dropna().unique().tolist()) if not out.empty else []
    return out, debug


# ------------------------------------------------------------
# Country lists
# ------------------------------------------------------------
def list_export_countries(lang: str = "fi") -> pd.DataFrame:
    rows = []
    for ifile in [ULJAS_SITC_IFILE, ULJAS_SITC2_IFILE]:
        items = class_values(ifile, "Maa", lang=lang)
        for item in items:
            if item.code == "AA":
                continue
            txt = item.text or item.code
            if ") " in txt:
                txt = txt.split(") ", 1)[1]
            rows.append({"code": item.code, "name": txt})

    df = pd.DataFrame(rows).drop_duplicates()
    if df.empty:
        return df
    return df.sort_values("name").reset_index(drop=True)


def list_import_countries(lang: str = "fi") -> pd.DataFrame:
    return list_export_countries(lang=lang)


# ------------------------------------------------------------
# Country details
# ------------------------------------------------------------
def fetch_exports_country_detail(country_code: str, months: int = 24, lang: str = "fi") -> tuple[pd.DataFrame, dict]:
    raw, debug = _fetch_trade_raw(
        months=max(months, 84),
        lang=lang,
        sitc_code="0-9",
        geo_class_name="Maa",
        geo_code=country_code,
        direction=EXPORT_DIRECTION,
        timeout=60,
    )
    if raw.empty:
        return pd.DataFrame(), debug

    out = (
        raw.groupby("Aika_dt", as_index=False)["value"]
        .sum()
        .rename(columns={"value": "Vienti_eur"})
        .sort_values("Aika_dt")
        .reset_index(drop=True)
    )
    debug["years"] = sorted(out["Aika_dt"].dt.year.dropna().unique().tolist()) if not out.empty else []
    return out, debug


def fetch_imports_country_detail(country_code: str, months: int = 24, lang: str = "fi") -> tuple[pd.DataFrame, dict]:
    raw, debug = _fetch_trade_raw(
        months=max(months, 84),
        lang=lang,
        sitc_code="0-9",
        geo_class_name="Maa",
        geo_code=country_code,
        direction=IMPORT_DIRECTION,
        timeout=60,
    )
    if raw.empty:
        return pd.DataFrame(), debug

    out = (
        raw.groupby("Aika_dt", as_index=False)["value"]
        .sum()
        .rename(columns={"value": "Tuonti_eur"})
        .sort_values("Aika_dt")
        .reset_index(drop=True)
    )
    debug["years"] = sorted(out["Aika_dt"].dt.year.dropna().unique().tolist()) if not out.empty else []
    return out, debug


def fetch_exports_country_products(country_code: str, months: int = 24, lang: str = "fi") -> tuple[pd.DataFrame, dict]:
    raw, debug = _fetch_trade_raw(
        months=max(months, 84),
        lang=lang,
        sitc_code="=ALL",
        geo_class_name="Maa",
        geo_code=country_code,
        direction=EXPORT_DIRECTION,
        product_class_name="Tavaraluokitus SITC2",
        timeout=60,
    )
    if raw.empty:
        return pd.DataFrame(), debug

    raw["code"] = raw["code"].astype(str).str.strip()
    raw = raw[raw["code"].str.fullmatch(r"\d{2}")].copy()
    raw["Tuoteryhmä"] = raw["code"].map(_map_sitc2_to_legacy_group)

    out = (
        raw.groupby(["Aika_dt", "Tuoteryhmä"], as_index=False)["value"]
        .sum()
        .rename(columns={"value": "Vienti_eur"})
        .sort_values(["Aika_dt", "Tuoteryhmä"])
        .reset_index(drop=True)
    )
    debug["years"] = sorted(out["Aika_dt"].dt.year.dropna().unique().tolist()) if not out.empty else []
    return out, debug


def fetch_imports_country_products(country_code: str, months: int = 24, lang: str = "fi") -> tuple[pd.DataFrame, dict]:
    raw, debug = _fetch_trade_raw(
        months=max(months, 84),
        lang=lang,
        sitc_code="=ALL",
        geo_class_name="Maa",
        geo_code=country_code,
        direction=IMPORT_DIRECTION,
        product_class_name="Tavaraluokitus SITC2",
        timeout=60,
    )
    if raw.empty:
        return pd.DataFrame(), debug

    raw["code"] = raw["code"].astype(str).str.strip()
    raw = raw[raw["code"].str.fullmatch(r"\d{2}")].copy()
    raw["Tuoteryhmä"] = raw["code"].map(_map_sitc2_to_legacy_group)

    out = (
        raw.groupby(["Aika_dt", "Tuoteryhmä"], as_index=False)["value"]
        .sum()
        .rename(columns={"value": "Tuonti_eur"})
        .sort_values(["Aika_dt", "Tuoteryhmä"])
        .reset_index(drop=True)
    )
    debug["years"] = sorted(out["Aika_dt"].dt.year.dropna().unique().tolist()) if not out.empty else []
    return out, debug