from __future__ import annotations

import numpy as np
import pandas as pd


def clean_price_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["Close"] = pd.to_numeric(out["Close"], errors="coerce")

    if "Volume" in out.columns:
        out["Volume"] = pd.to_numeric(out["Volume"], errors="coerce")

    return (
        out.dropna(subset=["Date", "Close"])
        .sort_values("Date")
        .reset_index(drop=True)
    )


def latest_valid(series: pd.Series) -> float | None:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.iloc[-1])


def add_ma200(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Close_EUR"] = out["Close"]
    out["MA200_EUR"] = out["Close_EUR"].rolling(200).mean()
    return out


def calc_volatility_30d(df: pd.DataFrame) -> float | None:
    if df is None or len(df) < 30:
        return None

    ret = pd.to_numeric(df["Close"], errors="coerce").pct_change()
    return float(ret.tail(30).std() * np.sqrt(365) * 100)


def calc_volume_stats(df: pd.DataFrame) -> tuple[float | None, float | None]:
    if df is None or df.empty or "Volume" not in df.columns:
        return None, None

    vol = pd.to_numeric(df["Volume"], errors="coerce")
    vol_now = latest_valid(vol)
    vol_30avg = float(vol.tail(30).mean()) if len(vol.dropna()) >= 30 else None

    return vol_now, vol_30avg


def normalize_price(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    out[price_col] = pd.to_numeric(out[price_col], errors="coerce")
    out = out.dropna(subset=[price_col])

    if out.empty:
        return out

    first = float(out.iloc[0][price_col])
    if first == 0:
        return out

    out["Normalized"] = out[price_col] / first * 100.0
    return out


def eth_btc_ratio(btc_df: pd.DataFrame, eth_df: pd.DataFrame) -> pd.DataFrame:
    btc = btc_df[["Date", "Close"]].rename(columns={"Close": "BTC"})
    eth = eth_df[["Date", "Close"]].rename(columns={"Close": "ETH"})

    out = pd.merge(btc, eth, on="Date", how="inner")
    out = out.dropna(subset=["BTC", "ETH"])
    out = out[out["BTC"] != 0].copy()

    out["ETH_BTC"] = out["ETH"] / out["BTC"]
    return out


def drawdown_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.dropna(subset=["Close_EUR"]).copy()
    out["rolling_ath"] = out["Close_EUR"].cummax()
    out["drawdown_pct"] = (out["Close_EUR"] / out["rolling_ath"] - 1.0) * 100.0
    out["drawdown_pct"] = out["drawdown_pct"].clip(upper=0)
    return out