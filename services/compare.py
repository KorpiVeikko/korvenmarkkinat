# services/compare.py
from __future__ import annotations

import pandas as pd
import streamlit as st

from services.market_data import fetch_price_history, fetch_price_history_eur


EUR_ASSETS: dict[str, str] = {
    "Bitcoin": "BTC-USD",
    "Kulta": "GC=F",
    "Hopea": "SI=F",
}

NATIVE_ASSETS: dict[str, list[str]] = {
    "S&P 500": ["^GSPC", "SPY"],
    "Nasdaq": ["^IXIC", "QQQ"],
    "Suomi (OMXH25)": ["^OMXH25", "OMXH25.HE", "^OMXHPI", "^HEX25"],
}


def _empty_snapshot(status: str = "missing", symbol: str | None = None, currency: str = "native") -> dict:
    return {
        "now": None,
        "m1": None,
        "y1": None,
        "df": pd.DataFrame(),
        "status": status,
        "symbol": symbol,
        "currency": currency,
    }


def _clean_price_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "Close" not in df.columns:
        return pd.DataFrame()

    out = df.copy()

    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        out = out.dropna(subset=["Date"]).sort_values("Date")

    out["Close"] = pd.to_numeric(out["Close"], errors="coerce")
    out = out.dropna(subset=["Close"]).reset_index(drop=True)

    if "Date" in out.columns:
        out = out[["Date", "Close"]].copy()

    return out


@st.cache_data(ttl=60 * 10, show_spinner=False)
def _cached_native_history(symbol: str, period: str = "5y") -> pd.DataFrame:
    try:
        return _clean_price_df(fetch_price_history(symbol, period=period))
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60 * 10, show_spinner=False)
def _cached_eur_history(symbol: str, period: str = "5y") -> pd.DataFrame:
    try:
        return _clean_price_df(fetch_price_history_eur(symbol, period=period))
    except Exception:
        return pd.DataFrame()


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
    merged["Close"] = merged["Close"] / merged["EURUSD"]

    merged = merged.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)
    return merged[["Date", "Close"]].copy()


def _history_eur_with_fallback(symbol: str, period: str = "5y") -> tuple[pd.DataFrame, str]:
    """
    Palauttaa (df, status)
    status:
      - ok
      - fallback_fx
      - missing
    """
    df = _cached_eur_history(symbol, period=period)
    if not df.empty:
        return df, "ok"

    asset_df = _cached_native_history(symbol, period=period)
    fx_df = _cached_native_history("EURUSD=X", period=period)

    merged = _merge_asset_with_eurusd(asset_df, fx_df)
    merged = _clean_price_df(merged)

    if not merged.empty:
        return merged, "fallback_fx"

    return pd.DataFrame(), "missing"


def _first_history(symbols: list[str], period: str = "5y") -> tuple[str | None, pd.DataFrame, str]:
    """
    Palauttaa:
      (käytetty_symboli, df, status)

    status:
      - ok
      - fallback_symbol
      - missing
    """
    for idx, symbol in enumerate(symbols):
        df = _cached_native_history(symbol, period=period)
        if not df.empty:
            status = "ok" if idx == 0 else "fallback_symbol"
            return symbol, df, status

    return None, pd.DataFrame(), "missing"


def _closest_value_before_or_on(df: pd.DataFrame, target_date: pd.Timestamp) -> float | None:
    if df is None or df.empty or "Date" not in df.columns or "Close" not in df.columns:
        return None

    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d["Close"] = pd.to_numeric(d["Close"], errors="coerce")
    d = d.dropna(subset=["Date", "Close"]).sort_values("Date")

    if d.empty:
        return None

    eligible = d[d["Date"] <= target_date]
    if eligible.empty:
        return None

    return float(eligible.iloc[-1]["Close"])


def _pct_vs_offset(df: pd.DataFrame, offset: pd.DateOffset) -> float | None:
    if df is None or df.empty or "Date" not in df.columns or "Close" not in df.columns:
        return None

    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d["Close"] = pd.to_numeric(d["Close"], errors="coerce")
    d = d.dropna(subset=["Date", "Close"]).sort_values("Date")

    if d.empty:
        return None

    now = float(d.iloc[-1]["Close"])
    latest_date = pd.to_datetime(d.iloc[-1]["Date"])
    then = _closest_value_before_or_on(d, latest_date - offset)

    if then is None or then == 0:
        return None

    return (now / then - 1) * 100


def _build_snapshot(df: pd.DataFrame, status: str = "ok", symbol: str | None = None, currency: str = "native") -> dict:
    if df is None or df.empty or "Close" not in df.columns:
        return _empty_snapshot(status=status, symbol=symbol, currency=currency)

    d = _clean_price_df(df)
    if d.empty:
        return _empty_snapshot(status=status, symbol=symbol, currency=currency)

    return {
        "now": float(d.iloc[-1]["Close"]),
        "m1": _pct_vs_offset(d, pd.DateOffset(months=1)),
        "y1": _pct_vs_offset(d, pd.DateOffset(years=1)),
        "df": d[["Date", "Close"]].copy(),
        "status": status,
        "symbol": symbol,
        "currency": currency,
    }


def build_market_compare(period: str = "5y") -> dict[str, dict]:
    out: dict[str, dict] = {}

    for name, symbol in EUR_ASSETS.items():
        df, status = _history_eur_with_fallback(symbol, period=period)
        out[name] = _build_snapshot(
            df,
            status=status,
            symbol=symbol,
            currency="EUR",
        )

    for name, symbols in NATIVE_ASSETS.items():
        used_symbol, df, status = _first_history(symbols, period=period)
        out[name] = _build_snapshot(
            df,
            status=status,
            symbol=used_symbol,
            currency="native",
        )

    return out


def _returns_series(df: pd.DataFrame, days: int) -> pd.Series | None:
    if df is None or df.empty or "Close" not in df.columns:
        return None

    d = df.copy()

    if "Date" in d.columns:
        d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
        d = d.dropna(subset=["Date"]).sort_values("Date").set_index("Date")

    close = pd.to_numeric(d["Close"], errors="coerce").dropna()
    if close.empty:
        return None

    rets = close.pct_change().dropna()
    if len(rets) < 30:
        return None

    return rets.tail(days)


def correlation_matrix(snaps: dict[str, dict], days: int = 252) -> pd.DataFrame:
    series: dict[str, pd.Series] = {}

    for name, snap in snaps.items():
        rets = _returns_series(snap.get("df"), days=days)
        if rets is not None and not rets.empty:
            series[name] = rets

    if not series:
        return pd.DataFrame()

    df_ret = pd.DataFrame(series).dropna(how="any")
    if df_ret.shape[0] < 30:
        return pd.DataFrame()

    return df_ret.corr()