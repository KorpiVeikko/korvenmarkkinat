# services/currency_utils.py
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd


ANCHOR_CURRENCY = "USD"


def fmt_num(x: float | None, decimals: int = 4) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x:,.{decimals}f}".replace(",", " ")


def fmt_pct(x: float | None, decimals: int = 1) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x:+.{decimals}f} %"


def fmt_money_supply(value: float | None, currency: str) -> str:
    if value is None or pd.isna(value):
        return "—"

    v = float(value)

    # USD (FRED M2SL): miljardia USD
    if currency == "USD":
        return f"{v / 1_000:.1f} bilj. USD"

    # EUR (ECB M3): miljoonaa EUR
    if currency == "EUR":
        return f"{v / 1_000_000:.1f} bilj. EUR"

    # JPY (BOJ): 100 miljoonaa JPY
    if currency == "JPY":
        return f"{v / 10_000:.1f} bilj. JPY"

    # CNY (OECD): miljoonaa CNY
    # 8 600 000 -> 8.6 bilj. CNY
    if currency == "CNY":
        return f"{v / 1_000_000:.1f} bilj. CNY"

    return f"{v:,.0f} {currency}"


def pct_change(now: float | None, then: float | None) -> float | None:
    if now is None or then is None or then == 0:
        return None
    return (now / then - 1.0) * 100.0


def closest_before_or_on(
    df: pd.DataFrame,
    target_date: pd.Timestamp,
    date_col: str,
    value_col: str,
) -> float | None:
    if df is None or df.empty:
        return None

    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=[date_col, value_col]).sort_values(date_col)

    if d.empty:
        return None

    eligible = d[d[date_col] <= target_date]
    if eligible.empty:
        return None

    return float(eligible.iloc[-1][value_col])


def to_anchor_fx(fx_df: pd.DataFrame, anchor_df: pd.DataFrame) -> pd.DataFrame:
    """
    Muuntaa ECB:n EUR-pohjaisen sarjan USD-ankkuriseksi.

    Lähtö:
      Rate = valuuttaa per EUR

    Tulos:
      Rate = valuuttaa per USD
           = (valuuttaa per EUR) / (USD per EUR)
    """
    if fx_df is None or fx_df.empty or anchor_df is None or anchor_df.empty:
        return pd.DataFrame()

    a = fx_df.copy()
    b = anchor_df.copy()

    a["Date"] = pd.to_datetime(a["Date"], errors="coerce")
    b["Date"] = pd.to_datetime(b["Date"], errors="coerce")

    a["Rate"] = pd.to_numeric(a["Rate"], errors="coerce")
    b["Rate"] = pd.to_numeric(b["Rate"], errors="coerce")

    a = a.dropna(subset=["Date", "Rate"]).sort_values("Date").reset_index(drop=True)
    b = b.dropna(subset=["Date", "Rate"]).sort_values("Date").reset_index(drop=True)

    if a.empty or b.empty:
        return pd.DataFrame()

    b = b.rename(columns={"Rate": "AnchorRate"})

    merged = pd.merge_asof(
        a[["Date", "Rate"]].sort_values("Date"),
        b[["Date", "AnchorRate"]].sort_values("Date"),
        on="Date",
        direction="backward",
    )

    merged["Rate"] = pd.to_numeric(merged["Rate"], errors="coerce")
    merged["AnchorRate"] = pd.to_numeric(merged["AnchorRate"], errors="coerce")
    merged["Rate"] = merged["Rate"] / merged["AnchorRate"]

    merged = merged.dropna(subset=["Date", "Rate"]).sort_values("Date").reset_index(drop=True)
    return merged[["Date", "Rate"]].copy()


def build_fx_metrics(fx_df: pd.DataFrame) -> SimpleNamespace:
    empty = SimpleNamespace(
        latest_rate=None,
        latest_date=None,
        ytd_pct=None,
        change_1y_pct=None,
        change_5y_pct=None,
        change_10y_pct=None,
        volatility_1y_pct=None,
        min_rate=None,
        max_rate=None,
    )

    if fx_df is None or fx_df.empty:
        return empty

    d = fx_df.copy()
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d["Rate"] = pd.to_numeric(d["Rate"], errors="coerce")
    d = d.dropna(subset=["Date", "Rate"]).sort_values("Date").reset_index(drop=True)

    if d.empty:
        return empty

    latest_date = pd.to_datetime(d.iloc[-1]["Date"])
    latest_rate = float(d.iloc[-1]["Rate"])

    year_start = pd.Timestamp(year=latest_date.year, month=1, day=1)
    rate_ytd = closest_before_or_on(d, year_start, "Date", "Rate")
    rate_1y = closest_before_or_on(d, latest_date - pd.DateOffset(years=1), "Date", "Rate")
    rate_5y = closest_before_or_on(d, latest_date - pd.DateOffset(years=5), "Date", "Rate")
    rate_10y = closest_before_or_on(d, latest_date - pd.DateOffset(years=10), "Date", "Rate")

    ret = d["Rate"].pct_change().dropna()
    ret_1y = ret.tail(252)
    vol_1y = ret_1y.std() * (252**0.5) * 100 if not ret_1y.empty else None

    return SimpleNamespace(
        latest_rate=latest_rate,
        latest_date=latest_date,
        ytd_pct=pct_change(latest_rate, rate_ytd),
        change_1y_pct=pct_change(latest_rate, rate_1y),
        change_5y_pct=pct_change(latest_rate, rate_5y),
        change_10y_pct=pct_change(latest_rate, rate_10y),
        volatility_1y_pct=vol_1y,
        min_rate=float(d["Rate"].min()) if not d.empty else None,
        max_rate=float(d["Rate"].max()) if not d.empty else None,
    )


def pct_color_style(val) -> str:
    try:
        if val is None or pd.isna(val):
            return ""
        val = float(val)
        if val > 0:
            return "color: #1a7f37;"
        if val < 0:
            return "color: #d93025;"
        return ""
    except Exception:
        return ""


def change_metrics(df: pd.DataFrame, date_col: str, value_col: str) -> SimpleNamespace:
    empty = SimpleNamespace(
        latest_date=None,
        latest_value=None,
        change_1y_pct=None,
        change_5y_pct=None,
    )

    if df is None or df.empty or date_col not in df.columns or value_col not in df.columns:
        return empty

    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=[date_col, value_col]).sort_values(date_col)

    if d.empty:
        return empty

    latest_date = pd.to_datetime(d.iloc[-1][date_col])
    latest_value = float(d.iloc[-1][value_col])

    value_1y = closest_before_or_on(d, latest_date - pd.DateOffset(years=1), date_col, value_col)
    value_5y = closest_before_or_on(d, latest_date - pd.DateOffset(years=5), date_col, value_col)

    return SimpleNamespace(
        latest_date=latest_date,
        latest_value=latest_value,
        change_1y_pct=pct_change(latest_value, value_1y),
        change_5y_pct=pct_change(latest_value, value_5y),
    )


def build_real_fx_proxy(
    nominal_fx_df: pd.DataFrame,
    home_macro_df: pd.DataFrame,
    anchor_macro_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Kuukausitason CPI-oikaistu proxy:

      RealRateProxy = NominalRate * (AnchorCPI / HomeCPI)

    Ei ole virallinen REER, vaan yksinkertainen proxy.
    """
    if nominal_fx_df is None or nominal_fx_df.empty:
        return pd.DataFrame()

    fx = nominal_fx_df.copy()
    fx["Date"] = pd.to_datetime(fx["Date"], errors="coerce")
    fx["Rate"] = pd.to_numeric(fx["Rate"], errors="coerce")
    fx = fx.dropna(subset=["Date", "Rate"]).sort_values("Date").reset_index(drop=True)

    if fx.empty:
        return pd.DataFrame()

    fx["Month"] = fx["Date"].dt.to_period("M").dt.to_timestamp("M")
    fx_monthly = (
        fx.sort_values("Date")
        .groupby("Month", as_index=False)
        .tail(1)[["Month", "Rate"]]
        .rename(columns={"Month": "Date"})
        .sort_values("Date")
        .reset_index(drop=True)
    )

    home = home_macro_df.copy() if home_macro_df is not None else pd.DataFrame()
    anchor = anchor_macro_df.copy() if anchor_macro_df is not None else pd.DataFrame()

    for df in [home, anchor]:
        if not df.empty and "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    if home.empty or anchor.empty or "CPI_Index" not in home.columns or "CPI_Index" not in anchor.columns:
        return pd.DataFrame()

    home = home[["Date", "CPI_Index"]].rename(columns={"CPI_Index": "HomeCPI"}).dropna().sort_values("Date")
    anchor = anchor[["Date", "CPI_Index"]].rename(columns={"CPI_Index": "AnchorCPI"}).dropna().sort_values("Date")

    merged = (
        fx_monthly.merge(home, on="Date", how="inner")
        .merge(anchor, on="Date", how="inner")
        .dropna(subset=["Rate", "HomeCPI", "AnchorCPI"])
        .sort_values("Date")
        .reset_index(drop=True)
    )

    if merged.empty:
        return pd.DataFrame()

    merged["RealRateProxy"] = merged["Rate"] * (merged["AnchorCPI"] / merged["HomeCPI"])

    nominal_base = float(merged.iloc[0]["Rate"])
    real_base = float(merged.iloc[0]["RealRateProxy"])

    if nominal_base == 0 or real_base == 0:
        return pd.DataFrame()

    merged["NominalIndex"] = merged["Rate"] / nominal_base * 100.0
    merged["RealIndex"] = merged["RealRateProxy"] / real_base * 100.0

    return merged[["Date", "Rate", "RealRateProxy", "NominalIndex", "RealIndex"]].copy()