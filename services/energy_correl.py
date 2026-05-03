# services/energy_correl.py
from __future__ import annotations

import numpy as np
import pandas as pd


def _to_monthly(df: pd.DataFrame, time_col: str = "Aika_dt") -> pd.DataFrame:
    f = df.copy()
    if time_col not in f.columns:
        return pd.DataFrame()

    f = f.dropna(subset=[time_col]).copy()
    if f.empty:
        return pd.DataFrame()

    dt = pd.to_datetime(f[time_col], errors="coerce")
    f = f.loc[dt.notna()].copy()
    if f.empty:
        return pd.DataFrame()

    f["Month"] = dt.loc[dt.notna()].dt.to_period("M").dt.to_timestamp(how="start")
    f = f.dropna(subset=["Month"])
    return f


def _col_like(columns: list[str], *needles: str) -> str | None:
    for c in columns:
        cu = str(c).upper()
        if all(n.upper() in cu for n in needles):
            return c
    return None


def build_electricity_features(
    electricity_df: pd.DataFrame,
    series_col: str,
    unit: str = "TWh",
) -> pd.DataFrame:
    """
    Rakentaa kuukausittaiset featuret sähködatasta vertailua varten.

    Palauttaa:
    - total_consumption
    - total_production
    - import_net
    - import_share
    - wind_share
    - nuclear_share
    - hydro_share
    - solar_share
    - chp_share
    """
    f = _to_monthly(electricity_df, "Aika_dt")
    if f.empty:
        return pd.DataFrame()

    if "Arvo" not in f.columns or series_col not in f.columns:
        return pd.DataFrame()

    f["Arvo"] = pd.to_numeric(f["Arvo"], errors="coerce")
    f = f.dropna(subset=["Arvo"])
    if f.empty:
        return pd.DataFrame()

    if unit == "TWh":
        f["Arvo"] = f["Arvo"] / 1000.0

    p = (
        f.groupby(["Month", series_col], as_index=False)["Arvo"]
        .sum()
        .pivot(index="Month", columns=series_col, values="Arvo")
        .sort_index()
    )
    if p.empty:
        return pd.DataFrame()

    cols = [str(c) for c in p.columns]

    total_consumption_col = (
        _col_like(cols, "SSS")
        or _col_like(cols, "KOKONAISKULUTUS")
    )
    total_production_col = (
        _col_like(cols, "1 SÄHKÖN TUOTANTO")
        or _col_like(cols, "KOKONAISTUOTANTO")
    )
    import_col = (
        _col_like(cols, "NETTOTUONTI")
        or _col_like(cols, "2 SÄHKÖN")
    )

    wind_col = _col_like(cols, "TUULI")
    nuclear_col = _col_like(cols, "YDIN")
    hydro_col = _col_like(cols, "VESI")
    solar_col = _col_like(cols, "AURINKO")
    chp_col = (
        _col_like(cols, "YHTEISTUOTANTO YHTEENSÄ")
        or _col_like(cols, "1.5 YHTEISTUOTANTO")
    )

    out = pd.DataFrame(index=p.index)

    if total_consumption_col:
        out["total_consumption"] = p[total_consumption_col]
    if total_production_col:
        out["total_production"] = p[total_production_col]
    if import_col:
        out["import_net"] = p[import_col]

    denom = None
    if "total_production" in out.columns:
        denom = out["total_production"].replace(0, np.nan)
    elif "total_consumption" in out.columns:
        denom = out["total_consumption"].replace(0, np.nan)

    if "total_consumption" in out.columns and "import_net" in out.columns:
        out["import_share"] = out["import_net"] / out["total_consumption"].replace(0, np.nan)

    if denom is not None:
        if wind_col:
            out["wind_share"] = p[wind_col] / denom
        if nuclear_col:
            out["nuclear_share"] = p[nuclear_col] / denom
        if hydro_col:
            out["hydro_share"] = p[hydro_col] / denom
        if solar_col:
            out["solar_share"] = p[solar_col] / denom
        if chp_col:
            out["chp_share"] = p[chp_col] / denom

    out = out.reset_index().rename(columns={"index": "Month"})
    return out


def build_price_series(price_df: pd.DataFrame, value_col: str = "Arvo") -> pd.DataFrame:
    f = _to_monthly(price_df, "Aika_dt")
    if f.empty:
        return pd.DataFrame()

    if value_col not in f.columns:
        return pd.DataFrame()

    f[value_col] = pd.to_numeric(f[value_col], errors="coerce")
    f = f.dropna(subset=[value_col])
    if f.empty:
        return pd.DataFrame()

    s = f.groupby("Month", as_index=False)[value_col].mean()
    s = s.rename(columns={value_col: "price"})
    return s


def merge_price_and_features(price_series: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    if price_series is None or features is None or price_series.empty or features.empty:
        return pd.DataFrame()
    return pd.merge(price_series, features, on="Month", how="inner").sort_values("Month")


def add_change_features(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Lisää muutosfeaturet:
    - price_mom_pct: hinnan kuukausimuutos %
    - *_diff_pp: osuuksien muutos prosenttiyksikköinä
    - *_mom_pct: tasomuuttujien kuukausimuutos %
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy().sort_values("Month")

    if "price" in out.columns:
        out["price_mom_pct"] = out["price"].pct_change() * 100.0

    for col in cols:
        if col not in out.columns:
            continue

        if col.endswith("_share"):
            out[f"{col}_diff_pp"] = (out[col] - out[col].shift(1)) * 100.0
        else:
            out[f"{col}_mom_pct"] = out[col].pct_change() * 100.0

    return out


def corr_table(df: pd.DataFrame, cols: list[str], method: str = "pearson") -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    x = df[cols].dropna()
    if x.empty:
        return pd.DataFrame()

    return x.corr(method=method)


def pairwise_corr_to_target(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    method: str = "pearson",
) -> pd.DataFrame:
    if df is None or df.empty or target not in df.columns:
        return pd.DataFrame()

    rows = []
    for feat in features:
        if feat not in df.columns:
            continue

        tmp = df[[target, feat]].dropna()
        if len(tmp) < 8:
            corr = np.nan
        else:
            corr = tmp.corr(method=method).iloc[0, 1]

        rows.append({"feature": feat, "corr": corr, "n_obs": len(tmp)})

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["abs_corr"] = out["corr"].abs()
    return out.sort_values("abs_corr", ascending=False).drop(columns=["abs_corr"])


def rolling_corr(
    df: pd.DataFrame,
    x: str,
    y: str,
    window: int = 12,
    method: str = "pearson",
    min_periods: int | None = None,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    base = df[["Month", x, y]].dropna().sort_values("Month").copy()
    if base.empty:
        return pd.DataFrame()

    s = base.set_index("Month")[[x, y]].copy()

    if method.lower() == "spearman":
        s = s.rank()

    if min_periods is None:
        min_periods = max(6, window // 2)

    rc = s[x].rolling(window=window, min_periods=min_periods).corr(s[y])

    out = rc.reset_index()
    out.columns = ["Month", "corr"]
    return out.dropna(subset=["corr"])