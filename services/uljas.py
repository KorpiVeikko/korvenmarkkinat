# services/uljas.py
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests
from urllib.parse import quote

ULJAS_API_URL = "https://uljas.tulli.fi/uljas/graph/api.aspx"
ULJAS_SITC_IFILE = "/DATABASE/01 ULKOMAANKAUPPATILASTOT/02 SITC/ULJAS_SITC"


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


def _latest_time_codes(months: int, ifile: str = ULJAS_SITC_IFILE, lang: str = "fi") -> list[str]:
    items = class_values(ifile, "Aika", lang=lang)
    codes = [x.code for x in items if x.code]
    return codes[: max(1, int(months))]


def _join_value_list(values: list[str]) -> str:
    # Uljas toimii muodossa 202512"202511"202510...
    return '"'.join([str(v) for v in values if str(v).strip()])


# ------------------------------------------------------------
# Parsing helpers
# ------------------------------------------------------------
def _yyyymm_to_dt(s: str) -> pd.Timestamp:
    s = str(s).strip()
    if len(s) == 6 and s.isdigit():
        return pd.to_datetime(f"{s[:4]}-{s[4:6]}-01", errors="coerce")
    return pd.to_datetime(s, errors="coerce")


def _rows_to_df(rows: Any) -> pd.DataFrame:
    if not isinstance(rows, list):
        return pd.DataFrame()

    out_rows = []
    for row in rows:
        keys = row.get("keys") or []
        vals = row.get("vals") or []
        value = vals[0] if vals else None

        # Odotettu järjestys tässä kuutiossa:
        # [sitc_or_total, aika, maa, suunta]
        if len(keys) != 4:
            continue

        out_rows.append(
            {
                "code": str(keys[0]),
                "aika": str(keys[1]),
                "maa": str(keys[2]),
                "suunta": str(keys[3]),
                "value": pd.to_numeric(value, errors="coerce"),
            }
        )

    df = pd.DataFrame(out_rows)
    if df.empty:
        return df

    df["Aika_dt"] = df["aika"].map(_yyyymm_to_dt)
    return df


# ------------------------------------------------------------
# SITC2 -> teollisuusryhmä
# ------------------------------------------------------------
FOOD_CODES = {
    "00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "11", "12"
}
FOREST_CODES = {
    "24", "25", "63", "64"
}
CHEM_CODES = {
    "51", "52", "53", "54", "55", "56", "57", "58", "59"
}
ENERGY_CODES = {
    "33", "34", "35"
}
METAL_CODES = {
    "67", "68", "69"
}
MACHINERY_CODES = {
    "71", "72", "73", "74", "78", "79"
}
ELECTRONICS_CODES = {
    "75", "76", "77"
}


def _map_sitc2_to_group(code: str) -> str:
    c = str(code).strip()

    if c == "0-9":
        return "Kaikki ryhmät"

    if c in FOOD_CODES:
        return "Elintarvikkeet & maatalous"
    if c in FOREST_CODES:
        return "Metsäteollisuus"
    if c in CHEM_CODES:
        return "Kemianteollisuus"
    if c in ELECTRONICS_CODES:
        return "Elektroniikka & sähköteollisuus"
    if c in MACHINERY_CODES:
        return "Koneet & laitteet"
    if c in METAL_CODES:
        return "Metallit & metallituotteet"
    if c in ENERGY_CODES:
        return "Energiatuotteet"

    return "Muut"


# ------------------------------------------------------------
# Country label helper
# ------------------------------------------------------------
def _country_code_to_name_map(lang: str = "fi") -> dict[str, str]:
    items = class_values(ULJAS_SITC_IFILE, "Maa", lang=lang)
    out: dict[str, str] = {}
    for item in items:
        txt = item.text or item.code
        # poista alku "(2002--.) "
        if ") " in txt:
            txt = txt.split(") ", 1)[1]
        out[item.code] = txt
    return out


# ------------------------------------------------------------
# Public fetchers
# ------------------------------------------------------------
def fetch_exports_products(months: int = 48, lang: str = "fi") -> tuple[pd.DataFrame, dict]:
    """
    df columns:
      Aika_dt, Tuoteryhmä, Vienti_eur
    """
    debug: dict = {
        "tried_ifiles": [ULJAS_SITC_IFILE],
        "ok_ifile": None,
        "why_failed": [],
        "time_codes": [],
    }

    try:
        time_codes = _latest_time_codes(months=months, ifile=ULJAS_SITC_IFILE, lang=lang)
        debug["time_codes"] = time_codes

        if not time_codes:
            debug["why_failed"].append("Aikakoodeja ei saatu haettua class=Aika -kyselystä.")
            return pd.DataFrame(), debug

        data = _post_json(
            {
                "lang": lang,
                "atype": "data",
                "konv": "json",
                "ifile": ULJAS_SITC_IFILE,
                "select": "codes",
                "Tavaraluokitus SITC2": "=ALL",
                "Aika": _join_value_list(time_codes),
                "Maa": "AA",
                "Suunta": "2",
                "Indikaattorit": "V1",
            },
            timeout=60,
        )

        df = _rows_to_df(data)
        if df.empty:
            debug["why_failed"].append("Tuoteryhmäkysely palautti tyhjän aineiston.")
            return pd.DataFrame(), debug

        df = df.dropna(subset=["Aika_dt", "value"]).copy()

        # Poista kokonaissumma 0-9 pinotusta sarjasta
        df = df[df["code"] != "0-9"].copy()

        df["Tuoteryhmä"] = df["code"].map(_map_sitc2_to_group)

        out = (
            df.groupby(["Aika_dt", "Tuoteryhmä"], as_index=False)["value"]
            .sum()
            .rename(columns={"value": "Vienti_eur"})
            .sort_values(["Aika_dt", "Tuoteryhmä"])
        )

        debug["ok_ifile"] = ULJAS_SITC_IFILE
        return out, debug

    except Exception as ex:
        debug["why_failed"].append(f"{type(ex).__name__}: {ex}")
        return pd.DataFrame(), debug


def fetch_exports_regions(months: int = 48, lang: str = "fi") -> tuple[pd.DataFrame, dict]:
    """
    df columns:
      Aika_dt, Alue, Vienti_eur

    Alue = 5 suurinta vientimaata valitulla ajanjaksolla + 'Muut maat'
    """
    debug: dict = {
        "tried_ifiles": [ULJAS_SITC_IFILE],
        "ok_ifile": None,
        "why_failed": [],
        "time_codes": [],
        "top5_codes": [],
        "top5_names": [],
    }

    try:
        time_codes = _latest_time_codes(months=months, ifile=ULJAS_SITC_IFILE, lang=lang)
        debug["time_codes"] = time_codes

        if not time_codes:
            debug["why_failed"].append("Aikakoodeja ei saatu haettua class=Aika -kyselystä.")
            return pd.DataFrame(), debug

        data = _post_json(
            {
                "lang": lang,
                "atype": "data",
                "konv": "json",
                "ifile": ULJAS_SITC_IFILE,
                "select": "codes",
                "Tavaraluokitus SITC2": "0-9",
                "Aika": _join_value_list(time_codes),
                "Maa": "=ALL",
                "Suunta": "2",
                "Indikaattorit": "V1",
            },
            timeout=60,
        )

        df = _rows_to_df(data)
        if df.empty:
            debug["why_failed"].append("Maakysely palautti tyhjän aineiston.")
            return pd.DataFrame(), debug

        df = df.dropna(subset=["Aika_dt", "value"]).copy()

        # Poista kokonaissumma AA, jotta top5 lasketaan oikeista maista
        df = df[df["maa"] != "AA"].copy()

        if df.empty:
            debug["why_failed"].append("Maakyselyn jälkeen data tyhjeni (AA poistettu).")
            return pd.DataFrame(), debug

        country_name_map = _country_code_to_name_map(lang=lang)

        # Top 5 maata koko valitulta ajanjaksolta
        totals = (
            df.groupby("maa", as_index=False)["value"]
            .sum()
            .sort_values("value", ascending=False)
        )

        top5_codes = totals["maa"].head(5).tolist()
        debug["top5_codes"] = top5_codes
        debug["top5_names"] = [country_name_map.get(c, c) for c in top5_codes]

        def bucket_country(code: str) -> str:
            c = str(code).strip()
            if c in top5_codes:
                return country_name_map.get(c, c)
            return "Muut maat"

        df["Alue"] = df["maa"].map(bucket_country)

        out = (
            df.groupby(["Aika_dt", "Alue"], as_index=False)["value"]
            .sum()
            .rename(columns={"value": "Vienti_eur"})
            .sort_values(["Aika_dt", "Alue"])
        )

        debug["ok_ifile"] = ULJAS_SITC_IFILE
        return out, debug

    except Exception as ex:
        debug["why_failed"].append(f"{type(ex).__name__}: {ex}")
        return pd.DataFrame(), debug
    
# ------------------------------------------------------------
# Extra public helpers for detailed export views
# ------------------------------------------------------------
def list_export_countries(lang: str = "fi") -> pd.DataFrame:
    """
    Palauttaa maat dropdownia varten.
    columns: code, name
    """
    items = class_values(ULJAS_SITC_IFILE, "Maa", lang=lang)
    rows = []
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


def fetch_exports_country_detail(country_code: str, months: int = 48, lang: str = "fi") -> tuple[pd.DataFrame, dict]:
    """
    Yhden maan koko tavaravienti kuukausittain.
    columns: Aika_dt, Vienti_eur
    """
    debug: dict = {
        "country_code": country_code,
        "tried_ifiles": [ULJAS_SITC_IFILE],
        "ok_ifile": None,
        "why_failed": [],
        "time_codes": [],
    }

    try:
        time_codes = _latest_time_codes(months=months, ifile=ULJAS_SITC_IFILE, lang=lang)
        debug["time_codes"] = time_codes

        if not time_codes:
            debug["why_failed"].append("Aikakoodeja ei saatu haettua class=Aika -kyselystä.")
            return pd.DataFrame(), debug

        data = _post_json(
            {
                "lang": lang,
                "atype": "data",
                "konv": "json",
                "ifile": ULJAS_SITC_IFILE,
                "select": "codes",
                "Tavaraluokitus SITC2": "0-9",
                "Aika": _join_value_list(time_codes),
                "Maa": country_code,
                "Suunta": "2",
                "Indikaattorit": "V1",
            },
            timeout=60,
        )

        df = _rows_to_df(data)
        if df.empty:
            debug["why_failed"].append("Maan yksityiskohtainen kysely palautti tyhjän aineiston.")
            return pd.DataFrame(), debug

        df = df.dropna(subset=["Aika_dt", "value"]).copy()

        out = (
            df.groupby("Aika_dt", as_index=False)["value"]
            .sum()
            .rename(columns={"value": "Vienti_eur"})
            .sort_values("Aika_dt")
        )

        debug["ok_ifile"] = ULJAS_SITC_IFILE
        return out, debug

    except Exception as ex:
        debug["why_failed"].append(f"{type(ex).__name__}: {ex}")
        return pd.DataFrame(), debug


def fetch_exports_country_products(country_code: str, months: int = 48, lang: str = "fi") -> tuple[pd.DataFrame, dict]:
    """
    Yhden maan vienti teollisuusryhmittäin kuukausittain.
    columns: Aika_dt, Tuoteryhmä, Vienti_eur
    """
    debug: dict = {
        "country_code": country_code,
        "tried_ifiles": [ULJAS_SITC_IFILE],
        "ok_ifile": None,
        "why_failed": [],
        "time_codes": [],
    }

    try:
        time_codes = _latest_time_codes(months=months, ifile=ULJAS_SITC_IFILE, lang=lang)
        debug["time_codes"] = time_codes

        if not time_codes:
            debug["why_failed"].append("Aikakoodeja ei saatu haettua class=Aika -kyselystä.")
            return pd.DataFrame(), debug

        data = _post_json(
            {
                "lang": lang,
                "atype": "data",
                "konv": "json",
                "ifile": ULJAS_SITC_IFILE,
                "select": "codes",
                "Tavaraluokitus SITC2": "=ALL",
                "Aika": _join_value_list(time_codes),
                "Maa": country_code,
                "Suunta": "2",
                "Indikaattorit": "V1",
            },
            timeout=90,
        )

        df = _rows_to_df(data)
        if df.empty:
            debug["why_failed"].append("Maan tuoteryhmäkysely palautti tyhjän aineiston.")
            return pd.DataFrame(), debug

        df = df.dropna(subset=["Aika_dt", "value"]).copy()
        df = df[df["code"] != "0-9"].copy()
        df["Tuoteryhmä"] = df["code"].map(_map_sitc2_to_group)

        out = (
            df.groupby(["Aika_dt", "Tuoteryhmä"], as_index=False)["value"]
            .sum()
            .rename(columns={"value": "Vienti_eur"})
            .sort_values(["Aika_dt", "Tuoteryhmä"])
        )

        debug["ok_ifile"] = ULJAS_SITC_IFILE
        return out, debug

    except Exception as ex:
        debug["why_failed"].append(f"{type(ex).__name__}: {ex}")
        return pd.DataFrame(), debug

# ------------------------------------------------------------
# Import-side helpers
# ------------------------------------------------------------
def list_import_countries(lang: str = "fi") -> pd.DataFrame:
    # Sama maatieto kuin viennissä
    return list_export_countries(lang=lang)


def fetch_imports_products(months: int = 48, lang: str = "fi") -> tuple[pd.DataFrame, dict]:
    """
    df columns:
      Aika_dt, Tuoteryhmä, Tuonti_eur
    """
    debug: dict = {
        "tried_ifiles": [ULJAS_SITC_IFILE],
        "ok_ifile": None,
        "why_failed": [],
        "time_codes": [],
    }

    try:
        time_codes = _latest_time_codes(months=months, ifile=ULJAS_SITC_IFILE, lang=lang)
        debug["time_codes"] = time_codes

        if not time_codes:
            debug["why_failed"].append("Aikakoodeja ei saatu haettua class=Aika -kyselystä.")
            return pd.DataFrame(), debug

        data = _post_json(
            {
                "lang": lang,
                "atype": "data",
                "konv": "json",
                "ifile": ULJAS_SITC_IFILE,
                "select": "codes",
                "Tavaraluokitus SITC2": "=ALL",
                "Aika": _join_value_list(time_codes),
                "Maa": "AA",
                "Suunta": "1",   # tuonti alkuperämaittain
                "Indikaattorit": "V1",
            },
            timeout=60,
        )

        df = _rows_to_df(data)
        if df.empty:
            debug["why_failed"].append("Tuoteryhmäkysely palautti tyhjän aineiston.")
            return pd.DataFrame(), debug

        df = df.dropna(subset=["Aika_dt", "value"]).copy()
        df = df[df["code"] != "0-9"].copy()
        df["Tuoteryhmä"] = df["code"].map(_map_sitc2_to_group)

        out = (
            df.groupby(["Aika_dt", "Tuoteryhmä"], as_index=False)["value"]
            .sum()
            .rename(columns={"value": "Tuonti_eur"})
            .sort_values(["Aika_dt", "Tuoteryhmä"])
        )

        debug["ok_ifile"] = ULJAS_SITC_IFILE
        return out, debug

    except Exception as ex:
        debug["why_failed"].append(f"{type(ex).__name__}: {ex}")
        return pd.DataFrame(), debug


def fetch_imports_regions(months: int = 48, lang: str = "fi") -> tuple[pd.DataFrame, dict]:
    """
    df columns:
      Aika_dt, Alue, Tuonti_eur

    Alue = 5 suurinta tuontimaata valitulla ajanjaksolla + 'Muut maat'
    """
    debug: dict = {
        "tried_ifiles": [ULJAS_SITC_IFILE],
        "ok_ifile": None,
        "why_failed": [],
        "time_codes": [],
        "top5_codes": [],
        "top5_names": [],
    }

    try:
        time_codes = _latest_time_codes(months=months, ifile=ULJAS_SITC_IFILE, lang=lang)
        debug["time_codes"] = time_codes

        if not time_codes:
            debug["why_failed"].append("Aikakoodeja ei saatu haettua class=Aika -kyselystä.")
            return pd.DataFrame(), debug

        data = _post_json(
            {
                "lang": lang,
                "atype": "data",
                "konv": "json",
                "ifile": ULJAS_SITC_IFILE,
                "select": "codes",
                "Tavaraluokitus SITC2": "0-9",
                "Aika": _join_value_list(time_codes),
                "Maa": "=ALL",
                "Suunta": "1",
                "Indikaattorit": "V1",
            },
            timeout=60,
        )

        df = _rows_to_df(data)
        if df.empty:
            debug["why_failed"].append("Maakysely palautti tyhjän aineiston.")
            return pd.DataFrame(), debug

        df = df.dropna(subset=["Aika_dt", "value"]).copy()
        df = df[df["maa"] != "AA"].copy()

        if df.empty:
            debug["why_failed"].append("Maakyselyn jälkeen data tyhjeni (AA poistettu).")
            return pd.DataFrame(), debug

        country_name_map = _country_code_to_name_map(lang=lang)

        totals = (
            df.groupby("maa", as_index=False)["value"]
            .sum()
            .sort_values("value", ascending=False)
        )

        top5_codes = totals["maa"].head(5).tolist()
        debug["top5_codes"] = top5_codes
        debug["top5_names"] = [country_name_map.get(c, c) for c in top5_codes]

        def bucket_country(code: str) -> str:
            c = str(code).strip()
            if c in top5_codes:
                return country_name_map.get(c, c)
            return "Muut maat"

        df["Alue"] = df["maa"].map(bucket_country)

        out = (
            df.groupby(["Aika_dt", "Alue"], as_index=False)["value"]
            .sum()
            .rename(columns={"value": "Tuonti_eur"})
            .sort_values(["Aika_dt", "Alue"])
        )

        debug["ok_ifile"] = ULJAS_SITC_IFILE
        return out, debug

    except Exception as ex:
        debug["why_failed"].append(f"{type(ex).__name__}: {ex}")
        return pd.DataFrame(), debug


def fetch_imports_country_detail(country_code: str, months: int = 48, lang: str = "fi") -> tuple[pd.DataFrame, dict]:
    """
    Yhden maan koko tavaratuonti kuukausittain.
    columns: Aika_dt, Tuonti_eur
    """
    debug: dict = {
        "country_code": country_code,
        "tried_ifiles": [ULJAS_SITC_IFILE],
        "ok_ifile": None,
        "why_failed": [],
        "time_codes": [],
    }

    try:
        time_codes = _latest_time_codes(months=months, ifile=ULJAS_SITC_IFILE, lang=lang)
        debug["time_codes"] = time_codes

        if not time_codes:
            debug["why_failed"].append("Aikakoodeja ei saatu haettua class=Aika -kyselystä.")
            return pd.DataFrame(), debug

        data = _post_json(
            {
                "lang": lang,
                "atype": "data",
                "konv": "json",
                "ifile": ULJAS_SITC_IFILE,
                "select": "codes",
                "Tavaraluokitus SITC2": "0-9",
                "Aika": _join_value_list(time_codes),
                "Maa": country_code,
                "Suunta": "1",
                "Indikaattorit": "V1",
            },
            timeout=60,
        )

        df = _rows_to_df(data)
        if df.empty:
            debug["why_failed"].append("Maan yksityiskohtainen kysely palautti tyhjän aineiston.")
            return pd.DataFrame(), debug

        df = df.dropna(subset=["Aika_dt", "value"]).copy()

        out = (
            df.groupby("Aika_dt", as_index=False)["value"]
            .sum()
            .rename(columns={"value": "Tuonti_eur"})
            .sort_values("Aika_dt")
        )

        debug["ok_ifile"] = ULJAS_SITC_IFILE
        return out, debug

    except Exception as ex:
        debug["why_failed"].append(f"{type(ex).__name__}: {ex}")
        return pd.DataFrame(), debug


def fetch_imports_country_products(country_code: str, months: int = 48, lang: str = "fi") -> tuple[pd.DataFrame, dict]:
    """
    Yhden maan tuonti teollisuusryhmittäin kuukausittain.
    columns: Aika_dt, Tuoteryhmä, Tuonti_eur
    """
    debug: dict = {
        "country_code": country_code,
        "tried_ifiles": [ULJAS_SITC_IFILE],
        "ok_ifile": None,
        "why_failed": [],
        "time_codes": [],
    }

    try:
        time_codes = _latest_time_codes(months=months, ifile=ULJAS_SITC_IFILE, lang=lang)
        debug["time_codes"] = time_codes

        if not time_codes:
            debug["why_failed"].append("Aikakoodeja ei saatu haettua class=Aika -kyselystä.")
            return pd.DataFrame(), debug

        data = _post_json(
            {
                "lang": lang,
                "atype": "data",
                "konv": "json",
                "ifile": ULJAS_SITC_IFILE,
                "select": "codes",
                "Tavaraluokitus SITC2": "=ALL",
                "Aika": _join_value_list(time_codes),
                "Maa": country_code,
                "Suunta": "1",
                "Indikaattorit": "V1",
            },
            timeout=90,
        )

        df = _rows_to_df(data)
        if df.empty:
            debug["why_failed"].append("Maan tuoteryhmäkysely palautti tyhjän aineiston.")
            return pd.DataFrame(), debug

        df = df.dropna(subset=["Aika_dt", "value"]).copy()
        df = df[df["code"] != "0-9"].copy()
        df["Tuoteryhmä"] = df["code"].map(_map_sitc2_to_group)

        out = (
            df.groupby(["Aika_dt", "Tuoteryhmä"], as_index=False)["value"]
            .sum()
            .rename(columns={"value": "Tuonti_eur"})
            .sort_values(["Aika_dt", "Tuoteryhmä"])
        )

        debug["ok_ifile"] = ULJAS_SITC_IFILE
        return out, debug

    except Exception as ex:
        debug["why_failed"].append(f"{type(ex).__name__}: {ex}")
        return pd.DataFrame(), debug