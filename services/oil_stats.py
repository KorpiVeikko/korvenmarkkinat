# services/oil_stats.py
from __future__ import annotations

import pandas as pd
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


PXWEB_URL = "https://pxdata.stat.fi/PxWeb/api/v1/fi/StatFin/khi/statfin_khi_pxt_11xx.px"

FUEL_CODES = {
    "0400500": "Polttoöljy",
    "0700100": "Diesel",
    "0700200": "Bensiini 95",
    "0700300": "Bensiini 98",
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


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def fetch_finland_fuel_prices_debug(years: int = 5) -> tuple[pd.DataFrame, str | None]:
    try:
        months = max(12, years * 12 + 2)

        payload = {
            "query": [
                {
                    "code": "Kuukausi",
                    "selection": {
                        "filter": "top",
                        "values": [str(months)]
                    }
                },
                {
                    "code": "Hyödyke",
                    "selection": {
                        "filter": "item",
                        "values": list(FUEL_CODES.keys())
                    }
                },
                {
                    "code": "Tiedot",
                    "selection": {
                        "filter": "item",
                        "values": ["keskihinta"]
                    }
                }
            ],
            "response": {
                "format": "json-stat2"
            }
        }

        session = _build_session()
        res = session.post(PXWEB_URL, json=payload, timeout=30)
        res.raise_for_status()

        data = res.json()

        values = data.get("value", [])
        dims = data.get("dimension", {})

        if not values or "Kuukausi" not in dims or "Hyödyke" not in dims:
            return pd.DataFrame(), "Tilastokeskuksen vastausrakenne ei ollut odotettu."

        month_info = dims["Kuukausi"]["category"]
        fuel_info = dims["Hyödyke"]["category"]

        month_codes = [k for k, _ in sorted(month_info["index"].items(), key=lambda x: x[1])]
        fuel_codes = [k for k, _ in sorted(fuel_info["index"].items(), key=lambda x: x[1])]

        rows = []
        idx = 0
        for month_code in month_codes:
            for fuel_code in fuel_codes:
                if idx >= len(values):
                    break
                rows.append(
                    {
                        "Month": month_code,
                        "Fuel": FUEL_CODES.get(fuel_code, fuel_code),
                        "Value": values[idx],
                    }
                )
                idx += 1

        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(), "Polttoainehintadata jäi tyhjäksi rakentamisen jälkeen."

        df["Date"] = pd.to_datetime(
            df["Month"].astype(str).str.replace("M", "-", regex=False),
            format="%Y-%m",
            errors="coerce",
        )
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

        df = df.dropna(subset=["Date", "Value"]).sort_values(["Fuel", "Date"]).reset_index(drop=True)

        max_date = df["Date"].max()
        cutoff = max_date - pd.DateOffset(years=years)
        df = df[df["Date"] >= cutoff].copy()

        if df.empty:
            return pd.DataFrame(), "Polttoainehintadata jäi tyhjäksi 5 vuoden rajauksen jälkeen."

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