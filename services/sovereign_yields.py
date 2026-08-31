
from __future__ import annotations

from io import BytesIO
from typing import Iterable

import numpy as np
import pandas as pd
import requests


DEFAULT_TIMEOUT = 25
DEFAULT_MATURITIES = ("2Y", "5Y", "10Y", "30Y")

# Britannia jätetään tässä kevyessä versiossa tarkoituksella pois.
COUNTRY_META = {
    "US": {
        "name": "Yhdysvallat",
        "flag": "🇺🇸",
        "source": "FRED / Federal Reserve",
    },
    "DE": {
        "name": "Saksa",
        "flag": "🇩🇪",
        "source": "Deutsche Bundesbank",
    },
    "JP": {
        "name": "Japani",
        "flag": "🇯🇵",
        "source": "Ministry of Finance Japan",
    },
}

FRED_SERIES = {
    "2Y": "DGS2",
    "5Y": "DGS5",
    "10Y": "DGS10",
    "30Y": "DGS30",
}

BUNDESBANK_SERIES = {
    "2Y": "D.REN.EUR.A610.000000WT0202.A",
    "5Y": "D.REN.EUR.A620.000000WT0505.A",
    "10Y": "D.REN.EUR.A630.000000WT1010.A",
    "30Y": "D.REN.EUR.A640.000000WT3030.A",
}

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
BUNDESBANK_CSV_URL = (
    "https://api.statistiken.bundesbank.de/rest/data/"
    "BBSSY/{series_key}?format=csv&lang=en"
)
JAPAN_MOF_CSV_URL = (
    "https://www.mof.go.jp/english/policy/jgbs/reference/"
    "interest_rate/historical/jgbcme_all.csv"
)


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Korvenmarkkinat/1.0 sovereign-yields",
        "Accept": "*/*",
    })
    return session


def _empty() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["Date", "Country", "Maturity", "Yield", "Source"]
    )


def _parse_dates(
    values: pd.Series,
    formats: Iterable[str] = (),
    *,
    dayfirst: bool = False,
) -> pd.Series:
    raw = values.astype(str).str.strip()
    result = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")

    for fmt in formats:
        missing = result.isna()
        if not missing.any():
            break
        result.loc[missing] = pd.to_datetime(
            raw.loc[missing],
            format=fmt,
            errors="coerce",
            dayfirst=dayfirst,
        )

    missing = result.isna()
    if missing.any():
        result.loc[missing] = pd.to_datetime(
            raw.loc[missing],
            format="mixed",
            errors="coerce",
            dayfirst=dayfirst,
        )

    return result


def _standardize(
    dates: pd.Series,
    values: pd.Series,
    *,
    country: str,
    maturity: str,
    source: str,
    date_formats: Iterable[str] = (),
    dayfirst: bool = False,
) -> pd.DataFrame:
    out = pd.DataFrame({
        "Date": _parse_dates(
            dates,
            date_formats,
            dayfirst=dayfirst,
        ),
        "Yield": pd.to_numeric(values, errors="coerce"),
    })

    out = (
        out.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["Date", "Yield"])
        .sort_values("Date")
        .drop_duplicates("Date", keep="last")
    )

    out = out.loc[out["Yield"].between(-20.0, 50.0)].copy()
    out["Country"] = country
    out["Maturity"] = maturity
    out["Source"] = source

    return out[["Date", "Country", "Maturity", "Yield", "Source"]].reset_index(drop=True)


def _filter_years(df: pd.DataFrame, years: int) -> pd.DataFrame:
    if df is None or df.empty:
        return _empty()

    latest = df["Date"].max()
    cutoff = latest - pd.DateOffset(years=years)

    return (
        df.loc[df["Date"] >= cutoff]
        .sort_values("Date")
        .reset_index(drop=True)
    )


def _filter_months(df: pd.DataFrame, months: int) -> pd.DataFrame:
    if df is None or df.empty:
        return _empty()

    latest = df["Date"].max()
    cutoff = latest - pd.DateOffset(months=months)

    return (
        df.loc[df["Date"] >= cutoff]
        .sort_values("Date")
        .reset_index(drop=True)
    )


# ==========================================================
# USA
# ==========================================================

def fetch_us_yields(
    *,
    years: int = 10,
    maturities: Iterable[str] = DEFAULT_MATURITIES,
) -> tuple[pd.DataFrame, str | None]:
    session = _session()
    frames = []
    errors = []

    for maturity in maturities:
        series_id = FRED_SERIES.get(maturity)
        if not series_id:
            errors.append(f"USA: tuntematon maturiteetti {maturity}.")
            continue

        try:
            response = session.get(
                FRED_CSV_URL.format(series_id=series_id),
                timeout=DEFAULT_TIMEOUT,
            )
            response.raise_for_status()
            raw = pd.read_csv(BytesIO(response.content))

            if raw.empty or len(raw.columns) < 2:
                raise ValueError(f"FRED {series_id}: tyhjä tai virheellinen CSV.")

            frame = _standardize(
                raw.iloc[:, 0],
                raw.iloc[:, 1],
                country="US",
                maturity=maturity,
                source="FRED / Federal Reserve",
                date_formats=("%Y-%m-%d",),
            )
            frames.append(_filter_years(frame, years))

        except Exception as exc:
            errors.append(f"USA {maturity}: {exc}")

    df = pd.concat(frames, ignore_index=True) if frames else _empty()
    return df, " | ".join(errors) if errors else None


# ==========================================================
# SAKSA
# ==========================================================

def _read_bundesbank_csv(content: bytes) -> pd.DataFrame:
    for sep, decimal in [(";", ","), (",", ".")]:
        try:
            df = pd.read_csv(BytesIO(content), sep=sep, decimal=decimal)
            if len(df.columns) >= 2:
                return df
        except Exception:
            pass
    return pd.DataFrame()


def _find_bundesbank_columns(df: pd.DataFrame) -> tuple[object, object]:
    normalized = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    date_col = next(
        (
            normalized[key]
            for key in ("date", "time_period", "time period", "datum")
            if key in normalized
        ),
        df.columns[0],
    )

    value_col = next(
        (
            normalized[key]
            for key in ("value", "obs_value", "obs value", "wert")
            if key in normalized
        ),
        None,
    )

    if value_col is not None:
        return date_col, value_col

    candidates = []
    for column in df.columns:
        if column == date_col:
            continue
        numeric = pd.to_numeric(
            df[column].astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        )
        candidates.append((int(numeric.notna().sum()), column))

    if not candidates:
        raise ValueError("Bundesbank: arvosaraketta ei löytynyt.")

    candidates.sort(key=lambda item: item[0], reverse=True)
    return date_col, candidates[0][1]


def fetch_germany_yields(
    *,
    years: int = 10,
    maturities: Iterable[str] = DEFAULT_MATURITIES,
) -> tuple[pd.DataFrame, str | None]:
    session = _session()
    frames = []
    errors = []

    for maturity in maturities:
        series_key = BUNDESBANK_SERIES.get(maturity)
        if not series_key:
            errors.append(f"Saksa: tuntematon maturiteetti {maturity}.")
            continue

        try:
            response = session.get(
                BUNDESBANK_CSV_URL.format(series_key=series_key),
                timeout=DEFAULT_TIMEOUT,
            )
            response.raise_for_status()

            raw = _read_bundesbank_csv(response.content)
            if raw.empty:
                raise ValueError("Bundesbank palautti tyhjän CSV:n.")

            date_col, value_col = _find_bundesbank_columns(raw)
            values = (
                raw[value_col]
                .astype(str)
                .str.replace(",", ".", regex=False)
            )

            frame = _standardize(
                raw[date_col],
                values,
                country="DE",
                maturity=maturity,
                source="Deutsche Bundesbank",
                date_formats=("%Y-%m-%d", "%d.%m.%Y"),
                dayfirst=True,
            )
            frames.append(_filter_years(frame, years))

        except Exception as exc:
            errors.append(f"Saksa {maturity}: {exc}")

    df = pd.concat(frames, ignore_index=True) if frames else _empty()
    return df, " | ".join(errors) if errors else None


# ==========================================================
# JAPANI
# ==========================================================

def _load_japan_raw(session: requests.Session) -> pd.DataFrame:
    response = session.get(JAPAN_MOF_CSV_URL, timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()

    raw = pd.read_csv(
        BytesIO(response.content),
        header=1,
        na_values=["-", "", " "],
    )
    raw.columns = [str(column).strip() for column in raw.columns]

    if raw.empty or "Date" not in raw.columns:
        raise ValueError(
            f"MOF CSV:n rakennetta ei tunnistettu. Sarakkeet: {list(raw.columns)}"
        )

    return raw


def fetch_japan_yields(
    *,
    years: int = 10,
    maturities: Iterable[str] = DEFAULT_MATURITIES,
) -> tuple[pd.DataFrame, str | None]:
    session = _session()

    try:
        raw = _load_japan_raw(session)
    except Exception as exc:
        return _empty(), f"Japani: {exc}"

    frames = []
    errors = []

    for maturity in maturities:
        if maturity not in raw.columns:
            errors.append(f"Japani {maturity}: saraketta ei löytynyt.")
            continue

        frame = _standardize(
            raw["Date"],
            raw[maturity],
            country="JP",
            maturity=maturity,
            source="Ministry of Finance Japan",
            date_formats=("%Y/%m/%d", "%Y-%m-%d"),
        )
        frames.append(_filter_years(frame, years))

    df = pd.concat(frames, ignore_index=True) if frames else _empty()
    return df, " | ".join(errors) if errors else None


# ==========================================================
# KEVYT YLEISKUVA
# ==========================================================

def fetch_sovereign_yields_overview(
    *,
    months: int = 14,
) -> tuple[pd.DataFrame, dict[str, str | None]]:
    """
    Kevyt yleiskuvan haku.

    Haetaan vain noin 14 kuukauden aineisto:
    - nykyiset 2Y/5Y/10Y/30Y
    - 10Y 1 kk muutos
    - 10Y-2Y
    - riittävä historia noin vuoden vertailuun.

    Britannia ei kuulu tähän versioon.
    """
    fetchers = {
        "US": fetch_us_yields,
        "DE": fetch_germany_yields,
        "JP": fetch_japan_yields,
    }

    frames = []
    debug = {}

    # Lähdefunktiot suodattavat vuosina, joten 2 vuotta riittää,
    # minkä jälkeen leikataan tarkasti kuukausiin.
    for country, fetcher in fetchers.items():
        df, error = fetcher(years=2)
        df = _filter_months(df, months)

        debug[country] = error

        if not df.empty:
            frames.append(df)

    combined = pd.concat(frames, ignore_index=True) if frames else _empty()

    return (
        combined.sort_values(["Country", "Maturity", "Date"]).reset_index(drop=True),
        debug,
    )


# ==========================================================
# HISTORIA: HAE VAIN VALITTU MAA
# ==========================================================

def fetch_country_yields(
    country: str,
    *,
    years: int = 10,
    maturities: Iterable[str] = DEFAULT_MATURITIES,
) -> tuple[pd.DataFrame, str | None]:
    fetchers = {
        "US": fetch_us_yields,
        "DE": fetch_germany_yields,
        "JP": fetch_japan_yields,
    }

    fetcher = fetchers.get(country)
    if fetcher is None:
        return _empty(), f"Tuntematon maa: {country}"

    return fetcher(
        years=years,
        maturities=maturities,
    )


def fetch_yield_history(
    country: str,
    maturity: str,
    *,
    years: int = 10,
) -> tuple[pd.DataFrame, str | None]:
    """
    Korkokehitys-näkymää varten haetaan vain yksi maa + yksi maturiteetti.
    """
    return fetch_country_yields(
        country,
        years=years,
        maturities=(maturity,),
    )


# ==========================================================
# YHTEENSOPIVUUS VANHAN RAJAPINNAN KANSSA
# ==========================================================

def fetch_sovereign_yields(
    *,
    years: int = 10,
) -> tuple[pd.DataFrame, dict[str, str | None]]:
    """
    Täysi haku kaikille käytössä oleville maille.
    Säilytetään yhteensopivuuden vuoksi, mutta rates.py ei käytä tätä
    ensilatauksessa.
    """
    frames = []
    debug = {}

    for country in ("US", "DE", "JP"):
        df, error = fetch_country_yields(country, years=years)
        debug[country] = error
        if not df.empty:
            frames.append(df)

    combined = pd.concat(frames, ignore_index=True) if frames else _empty()

    return (
        combined.sort_values(["Country", "Maturity", "Date"]).reset_index(drop=True),
        debug,
    )


# ==========================================================
# ANALYYSIAPURIT
# ==========================================================

def build_latest_yield_snapshot(yields_df: pd.DataFrame) -> pd.DataFrame:
    if yields_df is None or yields_df.empty:
        return pd.DataFrame()

    out = yields_df.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["Yield"] = pd.to_numeric(out["Yield"], errors="coerce")
    out = out.dropna(subset=["Date", "Yield", "Country", "Maturity"])

    return (
        out.sort_values("Date")
        .groupby(["Country", "Maturity"], as_index=False)
        .tail(1)
        .sort_values(["Country", "Maturity"])
        .reset_index(drop=True)
    )


def build_yield_curve_snapshot(
    yields_df: pd.DataFrame,
    *,
    country: str,
) -> pd.DataFrame:
    latest = build_latest_yield_snapshot(yields_df)
    if latest.empty:
        return pd.DataFrame()

    out = latest.loc[latest["Country"] == country].copy()
    maturity_order = {"2Y": 2, "5Y": 5, "10Y": 10, "30Y": 30}
    out["MaturityYears"] = out["Maturity"].map(maturity_order)

    return (
        out.dropna(subset=["MaturityYears"])
        .sort_values("MaturityYears")
        .reset_index(drop=True)
    )


def build_10y_2y_spread(
    yields_df: pd.DataFrame,
    *,
    country: str,
) -> pd.DataFrame:
    if yields_df is None or yields_df.empty:
        return pd.DataFrame()

    subset = yields_df.loc[
        (yields_df["Country"] == country)
        & yields_df["Maturity"].isin(["2Y", "10Y"])
    ].copy()

    if subset.empty:
        return pd.DataFrame()

    wide = (
        subset.pivot_table(
            index="Date",
            columns="Maturity",
            values="Yield",
            aggfunc="last",
        )
        .sort_index()
    )

    if "2Y" not in wide.columns or "10Y" not in wide.columns:
        return pd.DataFrame()

    out = wide[["2Y", "10Y"]].dropna().copy()
    out["Spread_10Y_2Y"] = out["10Y"] - out["2Y"]

    return out.reset_index().sort_values("Date").reset_index(drop=True)
