from __future__ import annotations

from datetime import date
from io import StringIO

import pandas as pd
import requests


JODI_ANNUAL_PRIMARY_URL = (
    "https://www.jodidata.org/_resources/files/downloads/"
    "oil-data/annual-csv/primary/{year}.csv"
)


COUNTRY_NAMES = {
    "AE": "United Arab Emirates",
    "AO": "Angola",
    "AR": "Argentina",
    "AU": "Australia",
    "AZ": "Azerbaijan",
    "BR": "Brazil",
    "CA": "Canada",
    "CN": "China",
    "CO": "Colombia",
    "DZ": "Algeria",
    "EC": "Ecuador",
    "EG": "Egypt",
    "GB": "United Kingdom",
    "ID": "Indonesia",
    "IN": "India",
    "IQ": "Iraq",
    "IR": "Iran",
    "KZ": "Kazakhstan",
    "KW": "Kuwait",
    "LY": "Libya",
    "MX": "Mexico",
    "MY": "Malaysia",
    "NG": "Nigeria",
    "NO": "Norway",
    "OM": "Oman",
    "QA": "Qatar",
    "RU": "Russia",
    "SA": "Saudi Arabia",
    "US": "United States",
    "VE": "Venezuela",
}


def _country_name(code: str) -> str:
    normalized = str(code).strip().upper()
    return COUNTRY_NAMES.get(normalized, normalized)


def _load_jodi_year(
    session: requests.Session,
    year: int,
) -> tuple[pd.DataFrame, str | None]:
    url = JODI_ANNUAL_PRIMARY_URL.format(year=year)

    try:
        response = session.get(url, timeout=45)

        if response.status_code == 404:
            return (
                pd.DataFrame(),
                f"JODI-vuositiedostoa {year} ei löytynyt: {url}",
            )

        response.raise_for_status()

        df = pd.read_csv(
            StringIO(response.text),
            dtype=str,
        )

        if df.empty:
            return (
                pd.DataFrame(),
                f"JODI-vuositiedosto {year} oli tyhjä.",
            )

        return df, None

    except Exception as exc:
        return (
            pd.DataFrame(),
            f"JODI-vuoden {year} haku epäonnistui: {exc}",
        )


def fetch_jodi_crude_production_debug(
    years: int = 8,
) -> tuple[pd.DataFrame, str | None]:
    """
    Hakee JODI-Oil-tietokannasta maakohtaisen kuukausittaisen
    raakaöljyn tuotannon.

    Palauttaa sarakkeet:
    - CountryCode
    - Country
    - Date
    - Production_kbd
    - AssessmentCode

    Yksikkö:
    tuhatta barrelia päivässä (kb/d).
    """

    current_year = date.today().year
    first_year = max(2002, current_year - years + 1)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 Talous-dashboard/1.0 "
                "(JODI oil production data)"
            )
        }
    )

    frames: list[pd.DataFrame] = []
    messages: list[str] = []

    for year in range(first_year, current_year + 1):
        year_df, message = _load_jodi_year(session, year)

        if message:
            messages.append(message)

        if year_df is not None and not year_df.empty:
            frames.append(year_df)

    if not frames:
        return (
            pd.DataFrame(),
            "\n".join(messages) or "JODI-tuotantodataa ei saatu.",
        )

    raw = pd.concat(frames, ignore_index=True)

    required_columns = {
        "REF_AREA",
        "TIME_PERIOD",
        "ENERGY_PRODUCT",
        "FLOW_BREAKDOWN",
        "UNIT_MEASURE",
        "OBS_VALUE",
    }

    missing = required_columns.difference(raw.columns)

    if missing:
        return (
            pd.DataFrame(),
            (
                "JODI-aineistosta puuttuu odotettuja sarakkeita: "
                f"{sorted(missing)}. "
                f"Löytyneet sarakkeet: {list(raw.columns)}"
            ),
        )

    filtered = raw[
        (raw["ENERGY_PRODUCT"].astype(str).str.strip().str.upper() == "CRUDEOIL")
        & (
            raw["FLOW_BREAKDOWN"]
            .astype(str)
            .str.strip()
            .str.upper()
            == "INDPROD"
        )
        & (
            raw["UNIT_MEASURE"]
            .astype(str)
            .str.strip()
            .str.upper()
            == "KBD"
        )
    ].copy()

    if filtered.empty:
        return (
            pd.DataFrame(),
            (
                "JODI-aineistosta ei löytynyt "
                "CRUDEOIL / INDPROD / KBD -sarjaa."
            ),
        )

    filtered["Date"] = pd.to_datetime(
        filtered["TIME_PERIOD"],
        format="%Y-%m",
        errors="coerce",
    )

    filtered["Production_kbd"] = pd.to_numeric(
        filtered["OBS_VALUE"],
        errors="coerce",
    )

    if "ASSESSMENT_CODE" in filtered.columns:
        filtered["AssessmentCode"] = pd.to_numeric(
            filtered["ASSESSMENT_CODE"],
            errors="coerce",
        )
    else:
        filtered["AssessmentCode"] = pd.NA

    filtered["CountryCode"] = (
        filtered["REF_AREA"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    filtered["Country"] = filtered["CountryCode"].map(_country_name)

    out = filtered[
        [
            "CountryCode",
            "Country",
            "Date",
            "Production_kbd",
            "AssessmentCode",
        ]
    ].copy()

    out = (
        out.dropna(
            subset=[
                "CountryCode",
                "Country",
                "Date",
                "Production_kbd",
            ]
        )
        .loc[lambda df: df["Production_kbd"] > 0]
        .sort_values(["Country", "Date"])
        .drop_duplicates(
            subset=["CountryCode", "Date"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    if out.empty:
        return (
            pd.DataFrame(),
            "JODI-tuotantosarja jäi tyhjäksi puhdistuksen jälkeen.",
        )


    failed_messages = [
        message
        for message in messages
        if "ei löytynyt" not in message
    ]

    if failed_messages:
        return out, "\n".join(failed_messages)

    return out, None