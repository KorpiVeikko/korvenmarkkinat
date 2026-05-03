from __future__ import annotations

import pandas as pd


def latest_and_yoy(
    df: pd.DataFrame | None,
    value_col: str = "Arvo",
    yoy_col: str = "YoY_pct",
):
    if df is None or df.empty:
        return None, None

    latest = df.iloc[-1][value_col] if value_col in df.columns else None
    yoy = df.iloc[-1][yoy_col] if yoy_col in df.columns else None
    return latest, yoy


def latest_value(df: pd.DataFrame | None, value_col: str = "Arvo"):
    if df is None or df.empty or value_col not in df.columns:
        return None
    return df.iloc[-1][value_col]


def calc_quarterly_yoy(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Jakso_dt", "Arvo", "YoY_pct"])

    out = df.sort_values("Jakso_dt").copy()
    out["Arvo"] = pd.to_numeric(out["Arvo"], errors="coerce")
    out["YoY_pct"] = out["Arvo"].pct_change(4) * 100.0
    return out


def aggregate_trade_counts(lkm_df: pd.DataFrame | None) -> pd.DataFrame:
    if lkm_df is None or lkm_df.empty:
        return pd.DataFrame(columns=["Jakso_dt", "Arvo"])

    return (
        lkm_df.groupby("Jakso_dt", as_index=False)["Arvo"]
        .sum()
        .sort_values("Jakso_dt")
        .reset_index(drop=True)
    )