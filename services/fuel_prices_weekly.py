from __future__ import annotations

from io import BytesIO
import re
from urllib.parse import urljoin

import pandas as pd
import requests


EU_OIL_BULLETIN_PAGE = (
    "https://energy.ec.europa.eu/data-and-analysis/"
    "weekly-oil-bulletin_en"
)

# Varalinkki, jos komission sivun linkin automaattinen tunnistus epäonnistuu.
EU_OIL_HISTORY_FALLBACK_URL = (
    "https://energy.ec.europa.eu/document/download/"
    "906e60ca-8b6a-44e7-8589-652854d2fd3f_en"
    "?filename=Weekly_Oil_Bulletin_Prices_History_maticni_4web.xlsx"
)

SHEET_NAME = "Prices with taxes"

FINLAND_COLUMNS = {
    "FI_price_with_tax_euro95": "95E10",
    "FI_price_with_tax_diesel": "Diesel",
}


def _build_session() -> requests.Session:
    session = requests.Session()

    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 Talous-dashboard/1.0 "
                "(EU Weekly Oil Bulletin data)"
            )
        }
    )

    return session


def _find_history_xlsx_url(
    session: requests.Session,
) -> tuple[str | None, str | None]:
    """
    Etsii komission Weekly Oil Bulletin -sivulta uusimman
    'Price developments 2005 onwards' -Excel-linkin.
    """

    try:
        response = session.get(
            EU_OIL_BULLETIN_PAGE,
            timeout=30,
        )
        response.raise_for_status()

        html = response.text

        # Linkki voi sisältää joko tiedoston varsinaisen nimen tai
        # document/download-polun ja xlsx-tiedostonimen parametrina.
        patterns = [
            r'href="([^"]*Weekly_Oil_Bulletin_Prices_History[^"]*\.xlsx[^"]*)"',
            r'href="([^"]*document/download/[^"]*filename=[^"]*\.xlsx[^"]*)"',
        ]

        for pattern in patterns:
            matches = re.findall(
                pattern,
                html,
                flags=re.IGNORECASE,
            )

            for match in matches:
                url = urljoin(
                    EU_OIL_BULLETIN_PAGE,
                    match.replace("&amp;", "&"),
                )

                if "Prices_History" in url or "price" in url.lower():
                    return url, None

        return (
            None,
            "Komission sivulta ei löytynyt hintahistorian Excel-linkkiä.",
        )

    except Exception as exc:
        return (
            None,
            f"Weekly Oil Bulletin -sivun lukeminen epäonnistui: {exc}",
        )


def _download_history_excel(
    session: requests.Session,
) -> tuple[bytes | None, str | None]:
    discovered_url, discovery_message = _find_history_xlsx_url(
        session
    )

    urls = []

    if discovered_url:
        urls.append(discovered_url)

    if EU_OIL_HISTORY_FALLBACK_URL not in urls:
        urls.append(EU_OIL_HISTORY_FALLBACK_URL)

    messages: list[str] = []

    if discovery_message:
        messages.append(discovery_message)

    for url in urls:
        try:
            response = session.get(
                url,
                timeout=60,
            )
            response.raise_for_status()

            content_type = response.headers.get(
                "Content-Type",
                "",
            ).lower()

            # XLSX on ZIP-pohjainen tiedosto ja alkaa tavallisesti PK-tavuilla.
            looks_like_xlsx = (
                response.content[:2] == b"PK"
                or "spreadsheet" in content_type
                or "excel" in content_type
            )

            if not looks_like_xlsx:
                messages.append(
                    f"Ladattu sisältö ei näyttänyt Excel-tiedostolta: {url}"
                )
                continue

            return response.content, None

        except Exception as exc:
            messages.append(
                f"Excel-tiedoston lataus epäonnistui ({url}): {exc}"
            )

    return None, "\n".join(messages)


def _parse_finland_weekly_prices(
    excel_content: bytes,
) -> tuple[pd.DataFrame, str | None]:
    try:
        raw = pd.read_excel(
            BytesIO(excel_content),
            sheet_name=SHEET_NAME,
            header=None,
        )

    except Exception as exc:
        return (
            pd.DataFrame(),
            f"Weekly Oil Bulletin -Excelin lukeminen epäonnistui: {exc}",
        )

    if raw.empty or len(raw) < 4:
        return (
            pd.DataFrame(),
            "Weekly Oil Bulletin -taulukko oli tyhjä tai liian lyhyt.",
        )

    # Ensimmäinen rivi sisältää vakaat sarjatunnukset.
    header_codes = raw.iloc[0].astype(str).str.strip()

    date_col = 0
    found_columns: dict[str, int] = {}

    for series_code in FINLAND_COLUMNS:
        matches = header_codes[
            header_codes == series_code
        ].index.tolist()

        if matches:
            found_columns[series_code] = int(matches[0])

    missing = [
        code
        for code in FINLAND_COLUMNS
        if code not in found_columns
    ]

    if missing:
        return (
            pd.DataFrame(),
            (
                "Suomen polttoainehintojen sarakkeita ei löytynyt: "
                f"{missing}"
            ),
        )

    frames: list[pd.DataFrame] = []

    for series_code, fuel_name in FINLAND_COLUMNS.items():
        value_col = found_columns[series_code]

        fuel_df = pd.DataFrame(
            {
                "Date": pd.to_datetime(
                    raw.iloc[3:, date_col],
                    errors="coerce",
                ),
                "Price_EUR_1000L": pd.to_numeric(
                    raw.iloc[3:, value_col],
                    errors="coerce",
                ),
            }
        )

        fuel_df["Fuel"] = fuel_name

        # Komission yksikkö on EUR / 1 000 litraa.
        fuel_df["Price_EUR_L"] = (
            fuel_df["Price_EUR_1000L"] / 1_000.0
        )

        frames.append(fuel_df)

    out = pd.concat(
        frames,
        ignore_index=True,
    )

    out = (
        out.dropna(
            subset=[
                "Date",
                "Fuel",
                "Price_EUR_L",
            ]
        )
        .loc[
            lambda df:
            (df["Price_EUR_L"] > 0)
            & (df["Price_EUR_L"] < 10)
        ]
        .sort_values(["Fuel", "Date"])
        .drop_duplicates(
            subset=["Fuel", "Date"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    if out.empty:
        return (
            pd.DataFrame(),
            "Suomen viikkohintasarja jäi tyhjäksi puhdistuksen jälkeen.",
        )

    out["Country"] = "Finland"
    out["Source"] = "European Commission Weekly Oil Bulletin"
    out["Unit"] = "EUR/l"

    return (
        out[
            [
                "Date",
                "Country",
                "Fuel",
                "Price_EUR_L",
                "Price_EUR_1000L",
                "Source",
                "Unit",
            ]
        ],
        None,
    )


def fetch_finland_weekly_fuel_prices_debug(
    years: int | None = 10,
) -> tuple[pd.DataFrame, str | None]:
    """
    Hakee Suomen 95E10-bensiinin ja dieselin viikoittaiset
    kuluttajahinnat Euroopan komission Weekly Oil Bulletinista.

    Palautettavat sarakkeet:
    - Date
    - Country
    - Fuel
    - Price_EUR_L
    - Price_EUR_1000L
    - Source
    - Unit

    Price_EUR_L:
    euroa litralta, verot sisältävä kuluttajahinta.
    """

    session = _build_session()

    excel_content, download_message = _download_history_excel(
        session
    )

    if excel_content is None:
        return (
            pd.DataFrame(),
            download_message
            or "Weekly Oil Bulletin -Exceliä ei saatu ladattua.",
        )

    out, parse_message = _parse_finland_weekly_prices(
        excel_content
    )

    if out.empty:
        return out, parse_message

    if years is not None and years > 0:
        latest_date = out["Date"].max()
        cutoff = latest_date - pd.DateOffset(years=years)

        out = out[
            out["Date"] >= cutoff
        ].copy()

    out = out.reset_index(drop=True)

    return out, None