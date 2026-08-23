from __future__ import annotations

from io import BytesIO
import re

import pandas as pd
import requests


EIA_STEO_MONTHLY_XLSX = "https://www.eia.gov/outlooks/steo/xls/steo_m.xlsx"
EIA_SHEET = "3atab"


def _parse_release_month(raw: pd.DataFrame) -> pd.Timestamp | None:
    """
    Lukee taulukon riviltä STEO:n forecast date -päivämäärän.

    Esimerkiksi:
    Wednesday, July 1, 2026
    """
    try:
        value = raw.iloc[3, 0]
        parsed = pd.to_datetime(value, errors="coerce")

        if pd.isna(parsed):
            return None

        return pd.Timestamp(parsed).to_period("M").to_timestamp()

    except Exception:
        return None


def _build_monthly_dates(raw: pd.DataFrame) -> dict[int, pd.Timestamp]:
    """
    Muodostaa Excelin sarakenumeron ja kuukauden välisen kartan.

    Taulukossa vuosi näkyy vain tammikuun sarakkeessa ja muiden
    kuukausien vuosisolu on tyhjä. Vuosi täytetään eteenpäin.
    """
    dates: dict[int, pd.Timestamp] = {}
    current_year: int | None = None

    for col in range(2, raw.shape[1]):
        year_value = pd.to_numeric(raw.iloc[2, col], errors="coerce")
        month_value = str(raw.iloc[3, col]).strip()

        if pd.notna(year_value):
            current_year = int(year_value)

        if current_year is None:
            continue

        month = pd.to_datetime(month_value, format="%b", errors="coerce")

        if pd.isna(month):
            continue

        dates[col] = pd.Timestamp(
            year=current_year,
            month=int(month.month),
            day=1,
        )

    return dates


def _find_series_row(
    raw: pd.DataFrame,
    series_code: str,
) -> int | None:
    """
    Etsii sarjan sen vakaan EIA-tunnuksen perusteella.

    papr_world = maailman petroleum and other liquids -tuotanto
    patc_world = maailman petroleum and other liquids -kulutus
    """
    codes = raw.iloc[:, 0].astype(str).str.strip()

    matches = raw.index[codes == series_code].tolist()

    if not matches:
        return None

    # 3atabissa papr_world voi esiintyä kahdesti.
    # Ensimmäinen kelvollinen rivi riittää.
    return int(matches[0])


def _extract_monthly_series(
    raw: pd.DataFrame,
    row_index: int,
    date_map: dict[int, pd.Timestamp],
    value_name: str,
) -> pd.DataFrame:
    rows = []

    for col, date in date_map.items():
        value = pd.to_numeric(raw.iloc[row_index, col], errors="coerce")

        if pd.isna(value):
            continue

        rows.append(
            {
                "Date": date,
                value_name: float(value),
            }
        )

    return pd.DataFrame(rows)


def fetch_world_oil_balance_eia_debug() -> tuple[pd.DataFrame, str | None]:
    """
    Hakee EIA STEO:n kuukausittaisen maailman liquid fuels
    -tuotannon ja -kulutuksen.

    Palautettavat sarakkeet:
    - Date
    - Production
    - Consumption
    - Balance
    - Status
    - ReleaseMonth
    - Source
    - Unit

    Yksikkö:
    miljoonaa barrelia päivässä.
    """
    try:
        response = requests.get(
            EIA_STEO_MONTHLY_XLSX,
            timeout=45,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 Talous-dashboard/1.0 "
                    "(world oil balance data)"
                )
            },
        )
        response.raise_for_status()

        excel = pd.ExcelFile(BytesIO(response.content))

        if EIA_SHEET not in excel.sheet_names:
            return (
                pd.DataFrame(),
                (
                    f"EIA STEO -tiedostosta ei löytynyt välilehteä "
                    f"{EIA_SHEET}. Löytyneet välilehdet: {excel.sheet_names}"
                ),
            )

        raw = pd.read_excel(
            excel,
            sheet_name=EIA_SHEET,
            header=None,
        )

        date_map = _build_monthly_dates(raw)

        if not date_map:
            return (
                pd.DataFrame(),
                "EIA STEO -taulukosta ei löytynyt kuukausisarakkeita.",
            )

        production_row = _find_series_row(raw, "papr_world")
        consumption_row = _find_series_row(raw, "patc_world")

        if production_row is None or consumption_row is None:
            return (
                pd.DataFrame(),
                (
                    "EIA STEO -taulukosta eivät löytyneet sarjat "
                    "papr_world ja patc_world."
                ),
            )

        production = _extract_monthly_series(
            raw,
            production_row,
            date_map,
            "Production",
        )

        consumption = _extract_monthly_series(
            raw,
            consumption_row,
            date_map,
            "Consumption",
        )

        out = pd.merge(
            production,
            consumption,
            on="Date",
            how="inner",
        )

        out = out.dropna(
            subset=["Date", "Production", "Consumption"]
        ).sort_values("Date")

        if out.empty:
            return (
                pd.DataFrame(),
                "EIA STEO -sarjojen yhdistämisen jälkeen data oli tyhjä.",
            )

        out["Balance"] = out["Production"] - out["Consumption"]

        release_month = _parse_release_month(raw)

        if release_month is not None:
            # Julkaisukuukausi ja sitä myöhemmät kuukaudet tulkitaan
            # ennusteeksi. Tätä edeltävät kuukaudet ovat historiaa/arvioita.
            out["Status"] = out["Date"].apply(
                lambda date: (
                    "Ennuste"
                    if pd.Timestamp(date) >= release_month
                    else "Historia"
                )
            )
            out["ReleaseMonth"] = release_month
        else:
            out["Status"] = "Ei määritelty"
            out["ReleaseMonth"] = pd.NaT

        out["Source"] = "EIA STEO"
        out["Unit"] = "milj. bbl/pv"

        out = out.reset_index(drop=True)

        return out, None

    except Exception as exc:
        return (
            pd.DataFrame(),
            f"EIA STEO -kuukausidatan haku epäonnistui: {exc}",
        )