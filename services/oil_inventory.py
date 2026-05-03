# services/oil_inventory.py
from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# USA weekly crude stocks excluding SPR (legacy series id via EIA API v2)
US_CRUDE_SERIES_URL = "https://api.eia.gov/v2/seriesid/PET.WCESTUS1.W"

# OECD petroleum and other liquids stocks (monthly, EIA API v2)
OECD_API_URL = "https://api.eia.gov/v2/international/data/"


def _build_session() -> requests.Session:
    session = requests.Session()

    retry = Retry(
        total=3,
        connect=3,
        read=3,
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


def _pct_change(now: float | None, then: float | None) -> float | None:
    if now is None or then is None or then == 0:
        return None
    return (now / then - 1.0) * 100.0


def _get_eia_api_key() -> str | None:
    try:
        key = st.secrets.get("EIA_API_KEY")
        if key:
            return str(key)
    except Exception:
        pass

    key = os.getenv("EIA_API_KEY")
    if key:
        return key

    return None


# =========================================================
# USA: Weekly U.S. Ending Stocks excluding SPR of Crude Oil
# =========================================================

def _parse_us_seriesid_json(payload: dict) -> tuple[pd.DataFrame, str | None]:
    try:
        response = payload.get("response", {})
        data = response.get("data", [])

        if not data:
            return pd.DataFrame(), f"USA-API ei palauttanut datapisteitä. Vastauksen avaimet={list(payload.keys())}"

        df = pd.DataFrame(data)

        if "period" not in df.columns or "value" not in df.columns:
            return pd.DataFrame(), f"USA-API:n sarakkeet eivät olleet odotetut. Sarakkeet={list(df.columns)}"

        out = df[["period", "value"]].copy()
        out.columns = ["Date", "Value"]

        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        out["Value"] = pd.to_numeric(out["Value"], errors="coerce")

        out = out.dropna(subset=["Date", "Value"]).sort_values("Date").reset_index(drop=True)

        # Series on thousand barrels -> million barrels
        out["Value"] = out["Value"] / 1000.0

        if out.empty:
            return pd.DataFrame(), "USA-API:n data jäi tyhjäksi siivouksen jälkeen."

        return out, None

    except Exception as e:
        return pd.DataFrame(), f"USA-API-vastauksen jäsentäminen epäonnistui: {e!r}"


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def fetch_us_crude_inventory_history_debug(years: int = 10) -> tuple[pd.DataFrame, str | None]:
    """
    Palauttaa:
      Date, Value
    Yksikkö: million barrels
    """
    api_key = _get_eia_api_key()
    if not api_key:
        return pd.DataFrame(), (
            "USA-varastoihin tarvitaan EIA API -avain. "
            "Aseta EIA_API_KEY Streamlit secretsiin tai ympäristömuuttujaksi."
        )

    try:
        session = _build_session()
        params = {"api_key": api_key}
        res = session.get(US_CRUDE_SERIES_URL, params=params, timeout=30)
        res.raise_for_status()

        payload = res.json()
        df, msg = _parse_us_seriesid_json(payload)
        if df.empty:
            return df, msg

        cutoff = df["Date"].max() - pd.DateOffset(years=years)
        df = df[df["Date"] >= cutoff].copy().reset_index(drop=True)

        return df, None

    except requests.exceptions.Timeout as e:
        return pd.DataFrame(), f"USA-varastohistorian haku aikakatkaistiin: {e!r}"
    except requests.exceptions.ConnectionError as e:
        return pd.DataFrame(), f"USA-varastohistorian haku epäonnistui verkkoyhteysvirheeseen: {e!r}"
    except requests.exceptions.HTTPError as e:
        status = getattr(e.response, "status_code", "tuntematon")
        body = ""
        try:
            body = e.response.text[:500]
        except Exception:
            pass
        return pd.DataFrame(), f"USA-varastohistorian haku palautti HTTP-virheen {status}. Vastauksen alku: {body}"
    except Exception as e:
        return pd.DataFrame(), f"USA-varastohistorian haku epäonnistui odottamattomaan virheeseen: {e!r}"


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def fetch_us_crude_inventory_debug() -> tuple[pd.DataFrame, str | None]:
    hist_df, msg = fetch_us_crude_inventory_history_debug(years=10)
    if hist_df.empty:
        return pd.DataFrame(), msg

    if len(hist_df) < 2:
        return pd.DataFrame(), "USA-varastohistoriassa ei ollut riittävästi havaintoja vertailuun."

    latest = hist_df.iloc[-1]
    prev = hist_df.iloc[-2]

    out = pd.DataFrame(
        {
            "Date": [latest["Date"]],
            "PreviousDate": [prev["Date"]],
            "Value": [float(latest["Value"])],
            "PreviousValue": [float(prev["Value"])],
        }
    )
    out["Change"] = out["Value"] - out["PreviousValue"]
    out["ChangePct"] = out.apply(lambda r: _pct_change(r["Value"], r["PreviousValue"]), axis=1)

    return out, None


# ==========================================
# OECD: Monthly OECD petroleum stocks via API
# ==========================================

def _parse_oecd_api_json(payload: dict) -> tuple[pd.DataFrame, str | None]:
    try:
        response = payload.get("response", {})
        data = response.get("data", [])

        if not data:
            return pd.DataFrame(), f"OECD-API ei palauttanut datapisteitä. Vastauksen avaimet={list(payload.keys())}"

        df = pd.DataFrame(data)

        if "period" not in df.columns or "value" not in df.columns:
            return pd.DataFrame(), f"OECD-API:n sarakkeet eivät olleet odotetut. Sarakkeet={list(df.columns)}"

        out = df[["period", "value"]].copy()
        out.columns = ["Date", "Value"]

        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        out["Value"] = pd.to_numeric(out["Value"], errors="coerce")

        out = out.dropna(subset=["Date", "Value"]).sort_values("Date").reset_index(drop=True)

        if out.empty:
            return pd.DataFrame(), "OECD-API:n data jäi tyhjäksi siivouksen jälkeen."

        return out, None

    except Exception as e:
        return pd.DataFrame(), f"OECD-API-vastauksen jäsentäminen epäonnistui: {e!r}"


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def fetch_oecd_petroleum_stocks_history_debug(years: int = 10) -> tuple[pd.DataFrame, str | None]:
    api_key = _get_eia_api_key()
    if not api_key:
        return pd.DataFrame(), (
            "OECD-varastoihin tarvitaan EIA API -avain. "
            "Aseta EIA_API_KEY Streamlit secretsiin tai ympäristömuuttujaksi."
        )

    try:
        session = _build_session()

        params = {
            "api_key": api_key,
            "frequency": "monthly",
            "data[0]": "value",
            "facets[activityId][]": "5",
            "facets[productId][]": "5",
            "facets[countryRegionId][]": "OECD",
            "facets[unit][]": "MBBL",
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": "0",
            "length": "5000",
        }

        res = session.get(OECD_API_URL, params=params, timeout=30)
        res.raise_for_status()

        payload = res.json()
        df, msg = _parse_oecd_api_json(payload)
        if df.empty:
            return df, msg

        cutoff = df["Date"].max() - pd.DateOffset(years=years)
        df = df[df["Date"] >= cutoff].copy().reset_index(drop=True)

        if df.empty:
            return pd.DataFrame(), "OECD-historiadata jäi tyhjäksi rajauksen jälkeen."

        return df, None

    except requests.exceptions.Timeout as e:
        return pd.DataFrame(), f"OECD-varastohistorian haku aikakatkaistiin: {e!r}"
    except requests.exceptions.ConnectionError as e:
        return pd.DataFrame(), f"OECD-varastohistorian haku epäonnistui verkkoyhteysvirheeseen: {e!r}"
    except requests.exceptions.HTTPError as e:
        status = getattr(e.response, "status_code", "tuntematon")
        body = ""
        try:
            body = e.response.text[:500]
        except Exception:
            pass
        return pd.DataFrame(), f"OECD-varastohistorian haku palautti HTTP-virheen {status}. Vastauksen alku: {body}"
    except Exception as e:
        return pd.DataFrame(), f"OECD-varastohistorian haku epäonnistui odottamattomaan virheeseen: {e!r}"


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def fetch_oecd_petroleum_stocks_debug() -> tuple[pd.DataFrame, str | None]:
    """
    Palauttaa yhden rivin:
      DateLabel, Value, PreviousValue, YearAgoValue, Change, ChangePct, YoYChangePct
    Yksikkö: million barrels
    """
    hist_df, msg = fetch_oecd_petroleum_stocks_history_debug(years=10)
    if hist_df.empty:
        return pd.DataFrame(), msg

    latest = hist_df.iloc[-1]
    prev = hist_df.iloc[-2] if len(hist_df) >= 2 else None

    year_ago_target = latest["Date"] - pd.DateOffset(years=1)
    year_ago_df = hist_df.iloc[(hist_df["Date"] - year_ago_target).abs().argsort()[:1]]
    year_ago = year_ago_df.iloc[0] if not year_ago_df.empty else None

    value = float(latest["Value"])
    previous_value = float(prev["Value"]) if prev is not None else None
    year_ago_value = float(year_ago["Value"]) if year_ago is not None else None

    out = pd.DataFrame(
        {
            "Date": [latest["Date"]],
            "DateLabel": [latest["Date"].strftime("%m.%Y")],
            "Value": [value],
            "PreviousValue": [previous_value],
            "YearAgoValue": [year_ago_value],
        }
    )
    out["Change"] = out["Value"] - out["PreviousValue"]
    out["ChangePct"] = out.apply(lambda r: _pct_change(r["Value"], r["PreviousValue"]), axis=1)
    out["YoYChangePct"] = out.apply(lambda r: _pct_change(r["Value"], r["YearAgoValue"]), axis=1)

    return out, msg