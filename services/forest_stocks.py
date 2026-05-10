from __future__ import annotations

import pandas as pd

from services.market_data import fetch_price_history


FOREST_STOCKS = {
    "UPM-Kymmene": {
        "symbol": "UPM.HE",
        "description": "Sellu, paperi, tarrat, energia ja biomateriaalit",
    },
    "Stora Enso": {
        "symbol": "STERV.HE",
        "description": "Pakkausmateriaalit, biomateriaalit ja puutuotteet",
    },
    "Metsä Board": {
        "symbol": "METSB.HE",
        "description": "Kartongit ja pakkausmateriaalit",
    },
    "Valmet": {
        "symbol": "VALMT.HE",
        "description": "Sellu-, paperi- ja energiateollisuuden teknologia",
    },
    "Ponsse": {
        "symbol": "PON1V.HE",
        "description": "Metsäkoneet",
    },
}


def _clean_stock_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "Close" not in df.columns:
        return pd.DataFrame(columns=["Date", "Close"])

    out = df.copy()

    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    else:
        out = out.reset_index()
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")

    out["Close"] = pd.to_numeric(out["Close"], errors="coerce")
    out = out.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)

    return out[["Date", "Close"]].copy()


def _closest_value_before_or_on(df: pd.DataFrame, target_date: pd.Timestamp) -> float | None:
    if df is None or df.empty:
        return None

    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d["Close"] = pd.to_numeric(d["Close"], errors="coerce")
    d = d.dropna(subset=["Date", "Close"]).sort_values("Date")

    older = d[d["Date"] <= target_date]
    if older.empty:
        return None

    return float(older.iloc[-1]["Close"])


def _pct_change_from_offset(df: pd.DataFrame, offset: pd.DateOffset) -> float | None:
    if df is None or df.empty:
        return None

    d = df.copy().sort_values("Date")
    latest_date = pd.to_datetime(d.iloc[-1]["Date"])
    latest_value = float(d.iloc[-1]["Close"])

    old_value = _closest_value_before_or_on(d, latest_date - offset)
    if old_value is None or old_value == 0:
        return None

    return (latest_value / old_value - 1.0) * 100.0


def build_forest_stocks_bundle(period: str = "5y") -> dict:
    snapshots = []
    normalized_frames = []

    for name, meta in FOREST_STOCKS.items():
        symbol = meta["symbol"]
        df = _clean_stock_df(fetch_price_history(symbol, period=period))

        if df.empty:
            snapshots.append(
                {
                    "Yhtiö": name,
                    "Symboli": symbol,
                    "Kuvaus": meta["description"],
                    "Nyt": None,
                    "1 kk %": None,
                    "1 v %": None,
                    "Data": pd.DataFrame(),
                }
            )
            continue

        latest = float(df.iloc[-1]["Close"])

        snapshots.append(
            {
                "Yhtiö": name,
                "Symboli": symbol,
                "Kuvaus": meta["description"],
                "Nyt": latest,
                "1 kk %": _pct_change_from_offset(df, pd.DateOffset(months=1)),
                "1 v %": _pct_change_from_offset(df, pd.DateOffset(years=1)),
                "Data": df,
            }
        )

        norm = df.copy()
        if not norm.empty and norm["Close"].iloc[0] != 0:
            norm["Arvo"] = norm["Close"] / norm["Close"].iloc[0] * 100.0
            norm["Yhtiö"] = name
            normalized_frames.append(norm[["Date", "Yhtiö", "Arvo"]])

    normalized = (
        pd.concat(normalized_frames, ignore_index=True)
        if normalized_frames
        else pd.DataFrame(columns=["Date", "Yhtiö", "Arvo"])
    )

    return {
        "snapshots": snapshots,
        "normalized": normalized,
    }