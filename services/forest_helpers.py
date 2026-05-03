from __future__ import annotations

import re
import pandas as pd


def find_first_matching_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = list(df.columns)
    lower_map = {str(c).lower(): c for c in cols}

    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]

    for col in cols:
        col_l = str(col).lower()
        for cand in candidates:
            if cand.lower() in col_l:
                return col

    return None


def add_week_sort_key(df: pd.DataFrame, week_col: str) -> pd.DataFrame:
    def _to_key(x: str) -> int:
        s = str(x).strip()
        m = re.search(r"(\d{4})\D?W?(\d{1,2})", s, re.IGNORECASE)
        if not m:
            m = re.search(r"(\d{4})\D+(\d{1,2})", s)
        if not m:
            return -1
        y = int(m.group(1))
        w = int(m.group(2))
        return y * 100 + w

    out = df.copy()
    out["sort_key"] = out[week_col].apply(_to_key)
    return out


def week_key_to_date(key: int) -> pd.Timestamp:
    key = int(key)
    year = key // 100
    week = key % 100
    try:
        return pd.Timestamp.fromisocalendar(year, week, 1)
    except Exception:
        return pd.NaT


def add_week_date(df: pd.DataFrame, sort_key_col: str = "sort_key") -> pd.DataFrame:
    out = df.copy()
    out["Date"] = out[sort_key_col].apply(week_key_to_date)
    return out


def add_month_date(df: pd.DataFrame, month_col: str) -> pd.DataFrame:
    out = df.copy()
    s = out[month_col].astype(str).str.strip()

    month_match = s.str.extract(r"^(?P<y>\d{4})M(?P<m>\d{2})$")
    if month_match["y"].notna().any():
        out["Date"] = pd.to_datetime(
            month_match["y"] + "-" + month_match["m"] + "-01",
            errors="coerce",
        )
        return out

    out["Date"] = pd.to_datetime(s, errors="coerce")
    return out


def add_year_date(df: pd.DataFrame, year_col: str) -> pd.DataFrame:
    out = df.copy()
    out["Date"] = pd.to_datetime(out[year_col].astype(str) + "-01-01", errors="coerce")
    return out


def pct_change_vs_periods(series: pd.Series, periods: int) -> float | None:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) <= periods:
        return None

    latest = float(s.iloc[-1])
    prev = float(s.iloc[-(periods + 1)])
    if prev == 0:
        return None
    return ((latest / prev) - 1.0) * 100.0


def latest_value(df: pd.DataFrame, value_col: str = "Arvo") -> float | None:
    if df is None or df.empty or value_col not in df.columns:
        return None

    d = df.copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=[value_col])
    if d.empty:
        return None
    return float(d.iloc[-1][value_col])


def latest_and_yoy(df: pd.DataFrame, value_col: str = "Arvo", periods: int = 12) -> tuple[float | None, float | None]:
    if df is None or df.empty or value_col not in df.columns:
        return None, None

    d = df.copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=[value_col])
    if d.empty:
        return None, None

    latest = float(d.iloc[-1][value_col])
    yoy = pct_change_vs_periods(d[value_col], periods=periods)
    return latest, yoy


def rolling_mean(df: pd.DataFrame, value_col: str = "Arvo", window: int = 12) -> pd.DataFrame:
    out = df.copy()
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    out[f"{value_col}_trend"] = out[value_col].rolling(window=window, min_periods=max(2, window // 2)).mean()
    return out


def rolling_sum(df: pd.DataFrame, value_col: str = "Arvo", window: int = 12) -> pd.DataFrame:
    out = df.copy()
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    out[f"{value_col}_12kk"] = out[value_col].rolling(window=window, min_periods=window).sum()
    return out