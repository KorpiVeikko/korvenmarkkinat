# services/market_data.py
from __future__ import annotations

from io import StringIO

import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


FRED_BRENT_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILBRENTEU"


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


def _period_to_days(period: str) -> int | None:
    mapping = {
        "1d": 1,
        "5d": 5,
        "1mo": 31,
        "3mo": 93,
        "6mo": 186,
        "1y": 366,
        "2y": 366 * 2,
        "5y": 366 * 5,
        "10y": 366 * 10,
        "max": None,
    }
    return mapping.get(period, 366 * 5)


def _trim_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    if df.empty or "Date" not in df.columns:
        return df

    days = _period_to_days(period)
    if days is None:
        return df

    max_date = df["Date"].max()
    cutoff = max_date - pd.Timedelta(days=days)
    return df[df["Date"] >= cutoff].copy()


def _make_datetime_naive(series: pd.Series) -> pd.Series:
    s = pd.to_datetime(series, errors="coerce")
    try:
        if getattr(s.dt, "tz", None) is not None:
            s = s.dt.tz_localize(None)
    except Exception:
        pass
    return s


def _normalize_yfinance_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.reset_index()

    if "Date" not in df.columns:
        possible_date_cols = [c for c in df.columns if str(c).lower() in {"date", "datetime"}]
        if possible_date_cols:
            df = df.rename(columns={possible_date_cols[0]: "Date"})

    if "Date" not in df.columns or "Close" not in df.columns:
        return pd.DataFrame()

    df["Date"] = _make_datetime_naive(df["Date"])
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    if "Open" in df.columns:
        df["Open"] = pd.to_numeric(df["Open"], errors="coerce")
    if "High" in df.columns:
        df["High"] = pd.to_numeric(df["High"], errors="coerce")
    if "Low" in df.columns:
        df["Low"] = pd.to_numeric(df["Low"], errors="coerce")
    if "Adj Close" in df.columns:
        df["Adj Close"] = pd.to_numeric(df["Adj Close"], errors="coerce")
    if "Volume" in df.columns:
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")

    keep_cols = [c for c in ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    df = df[keep_cols].dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)

    return df


def _fetch_from_fred_brent_debug(period: str) -> tuple[pd.DataFrame, str | None]:
    try:
        session = _build_session()
        res = session.get(FRED_BRENT_CSV_URL, timeout=20)

        status_code = res.status_code
        content_type = res.headers.get("Content-Type", "")

        res.raise_for_status()

        df = pd.read_csv(StringIO(res.text))

        date_col = None
        for candidate in ["DATE", "observation_date"]:
            if candidate in df.columns:
                date_col = candidate
                break

        value_col = "DCOILBRENTEU" if "DCOILBRENTEU" in df.columns else None

        if not date_col or not value_col:
            return pd.DataFrame(), (
                "Brent-datan rakenne ei ollut odotettu. "
                f"Status={status_code}, Content-Type={content_type}, "
                f"sarakkeet={list(df.columns)}"
            )

        df = df.rename(columns={date_col: "Date", value_col: "Close"})
        df["Date"] = _make_datetime_naive(df["Date"])
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

        df = df.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)
        df = _trim_period(df, period)

        if df.empty:
            return pd.DataFrame(), "Brent-data saatiin, mutta se jäi tyhjäksi siivouksen jälkeen."

        return df, None

    except requests.exceptions.Timeout as e:
        return pd.DataFrame(), f"Brent-haku aikakatkaistiin: {e!r}"
    except requests.exceptions.ConnectionError as e:
        return pd.DataFrame(), f"Brent-haku epäonnistui verkkoyhteysvirheeseen: {e!r}"
    except requests.exceptions.HTTPError as e:
        status = getattr(e.response, "status_code", "tuntematon")
        body = ""
        try:
            body = e.response.text[:500]
        except Exception:
            pass
        return pd.DataFrame(), f"Brent-haku palautti HTTP-virheen {status}. Vastauksen alku: {body}"
    except pd.errors.ParserError as e:
        return pd.DataFrame(), f"Brent-datan CSV-jäsennys epäonnistui: {e!r}"
    except Exception as e:
        return pd.DataFrame(), f"Brent-haku epäonnistui odottamattomaan virheeseen: {e!r}"


def _fetch_from_yahoo_debug(symbol: str, period: str) -> tuple[pd.DataFrame, str | None]:
    try:
        df = yf.download(
            symbol,
            period=period,
            progress=False,
            auto_adjust=False,
            threads=False,
        )

        df = _normalize_yfinance_df(df)
        if not df.empty:
            return df, None

        ticker = yf.Ticker(symbol)
        df2 = ticker.history(period=period, auto_adjust=False)
        df2 = _normalize_yfinance_df(df2)

        if not df2.empty:
            return df2, None

        return pd.DataFrame(), f"Yahoo Finance ei palauttanut dataa symbolille {symbol}."

    except Exception as e:
        return pd.DataFrame(), f"Yahoo Finance -haku epäonnistui symbolille {symbol}: {e!r}"


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def fetch_price_history_debug(symbol: str, period: str = "5y") -> tuple[pd.DataFrame, str | None]:
    """
    Palauttaa:
        (df, virheteksti)

    Brent haetaan FREDistä.
    Muut symbolit haetaan Yahoo Financesta.
    """
    if symbol == "BZ=F":
        return _fetch_from_fred_brent_debug(period=period)

    return _fetch_from_yahoo_debug(symbol=symbol, period=period)


def fetch_price_history(symbol: str, period: str = "5y") -> pd.DataFrame:
    df, _ = fetch_price_history_debug(symbol=symbol, period=period)
    return df


def _merge_asset_with_eurusd(asset_df: pd.DataFrame, eurusd_df: pd.DataFrame) -> pd.DataFrame:
    if asset_df is None or asset_df.empty or eurusd_df is None or eurusd_df.empty:
        return pd.DataFrame()

    a = asset_df.copy()
    f = eurusd_df.copy()

    if "Date" not in a.columns or "Close" not in a.columns:
        return pd.DataFrame()
    if "Date" not in f.columns or "Close" not in f.columns:
        return pd.DataFrame()

    a["Date"] = pd.to_datetime(a["Date"], errors="coerce")
    f["Date"] = pd.to_datetime(f["Date"], errors="coerce")

    a["Close"] = pd.to_numeric(a["Close"], errors="coerce")
    f["Close"] = pd.to_numeric(f["Close"], errors="coerce")

    a = a.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)
    f = f.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)

    if a.empty or f.empty:
        return pd.DataFrame()

    fx_small = f[["Date", "Close"]].rename(columns={"Close": "EURUSD"}).copy()

    merged = pd.merge_asof(
        a.sort_values("Date"),
        fx_small.sort_values("Date"),
        on="Date",
        direction="backward",
    )

    merged["Close"] = pd.to_numeric(merged["Close"], errors="coerce")
    merged["EURUSD"] = pd.to_numeric(merged["EURUSD"], errors="coerce")
    merged["Close_EUR"] = merged["Close"] / merged["EURUSD"]

    keep_cols = ["Date", "Close_EUR"]
    if "Volume" in merged.columns:
        keep_cols.append("Volume")

    out = merged[keep_cols].copy()
    out = out.rename(columns={"Close_EUR": "Close"})
    out = out.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)
    return out


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def fetch_price_history_eur_debug(symbol_usd: str, period: str = "5y") -> tuple[pd.DataFrame, str | None]:
    """
    Hakee USD-noteeratun sarjan ja muuntaa sen euroiksi EURUSD=X-kurssilla.
    """
    asset_df, asset_msg = fetch_price_history_debug(symbol_usd, period=period)
    if asset_df is None or asset_df.empty:
        return pd.DataFrame(), asset_msg or f"Dataa ei saatu symbolille {symbol_usd}."

    eurusd_df, fx_msg = fetch_price_history_debug("EURUSD=X", period=period)
    if eurusd_df is None or eurusd_df.empty:
        return pd.DataFrame(), fx_msg or "EURUSD-dataa ei saatu."

    out = _merge_asset_with_eurusd(asset_df, eurusd_df)
    if out.empty:
        return pd.DataFrame(), f"EUR-muunnos epäonnistui symbolille {symbol_usd}."

    return out, None


def fetch_price_history_eur(symbol_usd: str, period: str = "5y") -> pd.DataFrame:
    df, _ = fetch_price_history_eur_debug(symbol_usd=symbol_usd, period=period)
    return df





