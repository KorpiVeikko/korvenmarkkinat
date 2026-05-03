# services/oil_production.py
from __future__ import annotations

from io import StringIO

import pandas as pd
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


OWID_OIL_CSV_URL = "https://ourworldindata.org/grapher/oil-production-by-country.csv"


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


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("OWID palautti tyhjän taulukon.")

    value_col = None
    for candidate in ["Oil", "oil", "Oil production", "oil production"]:
        if candidate in df.columns:
            value_col = candidate
            break

    if value_col is None:
        raise ValueError(f"Öljyntuotannon arvosaraketta ei löytynyt. Sarakkeet: {list(df.columns)}")

    if "Entity" not in df.columns or "Year" not in df.columns:
        raise ValueError(f"Tarvittavat sarakkeet puuttuvat. Sarakkeet: {list(df.columns)}")

    df = df.rename(
        columns={
            "Entity": "Country",
            "Year": "Year",
            value_col: "Value",
        }
    )

    required = {"Country", "Year", "Value"}
    if not required.issubset(df.columns):
        raise ValueError(f"Tarvittavat sarakkeet puuttuvat nimeämisen jälkeen. Sarakkeet: {list(df.columns)}")

    df = df[["Country", "Year", "Value"]].copy()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

    # Poistetaan alueet, ryhmät ja muut aggregaatit
    exclude_exact = {
        "World",
        "Asia",
        "Europe",
        "North America",
        "South America",
        "Africa",
        "Oceania",
        "European Union (27)",
        "European Union (28)",
        "High-income countries",
        "Upper-middle-income countries",
        "Lower-middle-income countries",
        "Low-income countries",
        "OECD (EI)",
        "Non-OECD (EI)",
        "OPEC (EI)",
        "Non-OPEC (EI)",
        "Middle East (EI)",
        "CIS (EI)",
        "North America (EI)",
        "Europe (EI)",
        "Africa (EI)",
        "Asia Pacific (EI)",
        "European Union",
    }

    df = df[~df["Country"].isin(exclude_exact)]

    # Varmistetaan vielä, että muutkin EI-aggregaatit poistuvat
    df = df[~df["Country"].str.contains(r"\(EI\)", na=False)]

    df = df.dropna(subset=["Country", "Year", "Value"])
    df = df.sort_values(["Year", "Value"], ascending=[True, False]).reset_index(drop=True)

    if df.empty:
        raise ValueError("Öljyntuotantodata jäi tyhjäksi suodatuksen jälkeen.")

    return df


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def fetch_oil_production_debug() -> tuple[pd.DataFrame, str | None]:
    try:
        session = _build_session()
        res = session.get(OWID_OIL_CSV_URL, timeout=25)
        res.raise_for_status()

        raw_df = pd.read_csv(StringIO(res.text))
        df = _normalize(raw_df)

        return df, None

    except requests.exceptions.Timeout as e:
        return pd.DataFrame(), f"Öljyntuotannon haku aikakatkaistiin: {e!r}"

    except requests.exceptions.ConnectionError as e:
        return pd.DataFrame(), f"Öljyntuotannon haku epäonnistui verkkoyhteysvirheeseen: {e!r}"

    except requests.exceptions.HTTPError as e:
        status = getattr(e.response, "status_code", "tuntematon")
        body = ""
        try:
            body = e.response.text[:500]
        except Exception:
            pass
        return pd.DataFrame(), f"Öljyntuotannon haku palautti HTTP-virheen {status}. Vastauksen alku: {body}"

    except pd.errors.ParserError as e:
        return pd.DataFrame(), f"Öljyntuotantodatan CSV-jäsennys epäonnistui: {e!r}"

    except ValueError as e:
        return pd.DataFrame(), f"Öljyntuotantodatan sisältö ei ollut odotettu: {e}"

    except Exception as e:
        return pd.DataFrame(), f"Öljyntuotannon haku epäonnistui odottamattomaan virheeseen: {e!r}"


def fetch_oil_production() -> pd.DataFrame:
    df, _ = fetch_oil_production_debug()
    return df