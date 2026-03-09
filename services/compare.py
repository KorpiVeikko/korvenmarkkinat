# services/compare.py
from __future__ import annotations

import pandas as pd

from services.market_data import fetch_price_history


def _first_history(symbols: list[str], period: str = "2y") -> tuple[str | None, pd.DataFrame]:
    """
    Kokeilee symboleja järjestyksessä ja palauttaa ensimmäisen toimivan.
    """
    for s in symbols:
        try:
            df = fetch_price_history(s, period=period)
            if df is not None and not df.empty and "Close" in df.columns:
                df = df.dropna(subset=["Close"]).copy()
                if not df.empty:
                    return s, df
        except Exception:
            pass
    return None, pd.DataFrame()


def _now_m1_y1(df: pd.DataFrame) -> dict:
    """
    Palauttaa now-hinnan ja 1kk/1v muutokset (%).
    Oletus: df on päivätasoa (pörssipäiviä).
    """
    if df is None or df.empty or "Close" not in df.columns:
        return {"now": None, "m1": None, "y1": None, "df": pd.DataFrame()}

    df = df.dropna(subset=["Close"]).copy()
    if df.empty:
        return {"now": None, "m1": None, "y1": None, "df": pd.DataFrame()}

    now = float(df.iloc[-1]["Close"])

    def pct(back: int) -> float | None:
        if len(df) <= back:
            return None
        then = float(df.iloc[-(back + 1)]["Close"])
        if then == 0:
            return None
        return (now / then - 1) * 100

    return {
        "now": now,
        "m1": pct(21),     # ~1 kk
        "y1": pct(252),    # ~1 v
        "df": df[["Date", "Close"]].copy() if "Date" in df.columns else df.copy(),
    }


def build_market_compare(period: str = "2y") -> dict[str, dict]:
    """
    Rakentaa vertailudatan dashboardiin.
    """
    # Yahoo-tickerit voivat vaihdella, joten Suomelle annetaan useampi vaihtoehto.
    assets: dict[str, list[str]] = {
        "Bitcoin": ["BTC-USD"],
        "Kulta": ["GC=F"],
        "Hopea": ["SI=F"],
        "S&P 500": ["^GSPC"],
        "Nasdaq": ["^IXIC"],
        "Suomi (OMXH25)": ["^OMXH25", "OMXH25.HE", "^OMXHPI", "^HEX25"],  # kokeile nämä
    }

    out: dict[str, dict] = {}
    for name, symbols in assets.items():
        used_symbol, df = _first_history(symbols, period=period)
        snap = _now_m1_y1(df)
        snap["symbol"] = used_symbol
        out[name] = snap

    return out


def correlation_matrix(snaps: dict[str, dict], days: int = 252) -> pd.DataFrame:
    """
    Korrelaatio päivätuotoista (log/ pct - sama idea; käytetään pct_change).
    """
    series = {}
    for name, s in snaps.items():
        df = s.get("df")
        if df is None or df.empty:
            continue
        d = df.copy()

        if "Date" in d.columns:
            d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
            d = d.dropna(subset=["Date"]).sort_values("Date")
            d = d.set_index("Date")

        if "Close" not in d.columns:
            continue

        close = pd.to_numeric(d["Close"], errors="coerce").dropna()
        if close.empty:
            continue

        rets = close.pct_change().dropna()
        if len(rets) < 30:
            continue

        series[name] = rets.tail(days)

    if not series:
        return pd.DataFrame()

    df_ret = pd.DataFrame(series).dropna(how="any")
    if df_ret.shape[0] < 30:
        return pd.DataFrame()

    return df_ret.corr()
