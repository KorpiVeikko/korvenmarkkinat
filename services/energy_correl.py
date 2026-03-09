# services/energy_correl.py
from __future__ import annotations

import pandas as pd
import numpy as np


def _to_monthly(df: pd.DataFrame, time_col: str = "Aika_dt") -> pd.DataFrame:
    """
    Varmistaa kuukausitason: käyttää Aika_dt ja ankkuroidaan kuukauden alkuun.
    Huom: ei käytetä to_timestamp("MS"), koska joissain pandas-versioissa se ei ole tuettu.
    """
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


def build_electricity_features(
    electricity_df: pd.DataFrame,
    series_col: str,
    unit: str = "GWh",
) -> pd.DataFrame:
    """
    Rakentaa kuukausittaiset featuret sähködatasta vertailua varten.

    Palauttaa (jos löydettävissä):
    - total_consumption: kokonaiskulutus (GWh/TWh)
    - import_net: nettotuonti (GWh/TWh)
    - import_share: nettotuonnin osuus kokonaiskulutuksesta (0–1)
    - wind_share, nuclear_share, hydro_share, solar_share: tuotantomuotojen osuus kokonaiskulutuksesta (0–1)

    Huom: absoluuttisia tuotantomääriä ei palauteta (vain osuudet) selkeyden vuoksi.
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
        f.groupby(["Month", series_col], as_index=False)["Arvo"].sum()
        .pivot(index="Month", columns=series_col, values="Arvo")
        .sort_index()
    )
    if p.empty:
        return pd.DataFrame()

    def col_like(*needles: str) -> str | None:
        cols = [str(c) for c in p.columns]
        for c in cols:
            cu = c.upper()
            ok = True
            for n in needles:
                if n.upper() not in cu:
                    ok = False
                    break
            if ok:
                return c
        return None

    total_col = col_like("SSS") or col_like("KOKONAISKULUTUS")
    import_col = col_like("NETTOTUONTI") or col_like("2 SÄHKÖN")

    wind_col = col_like("TUULI")
    nuclear_col = col_like("YDIN")
    hydro_col = col_like("VESI")
    solar_col = col_like("AURINKO")

    out = pd.DataFrame(index=p.index)

    if total_col:
        out["total_consumption"] = p[total_col]
    if import_col:
        out["import_net"] = p[import_col]

    if "total_consumption" in out.columns:
        denom = out["total_consumption"].replace(0, np.nan)

        # ✅ tuonnin osuus (netto) kokonaiskulutuksesta
        if "import_net" in out.columns:
            out["import_share"] = out["import_net"] / denom

        if wind_col:
            out["wind_share"] = p[wind_col] / denom
        if nuclear_col:
            out["nuclear_share"] = p[nuclear_col] / denom
        if hydro_col:
            out["hydro_share"] = p[hydro_col] / denom
        if solar_col:
            out["solar_share"] = p[solar_col] / denom

    if out.empty:
        return pd.DataFrame()

    out = out.reset_index().rename(columns={"index": "Month"})
    return out


def build_price_series(price_df: pd.DataFrame, value_col: str = "Arvo") -> pd.DataFrame:
    """
    Yhtenäinen kuukausisarja hinnalle: palauttaa Month + price
    """
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
    m = pd.merge(price_series, features, on="Month", how="inner").sort_values("Month")
    return m


def corr_table(df: pd.DataFrame, cols: list[str], method: str = "pearson") -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    x = df[cols].dropna()
    if x.empty:
        return pd.DataFrame()
    return x.corr(method=method)


def lag_corr(
    df: pd.DataFrame,
    x: str,
    y: str,
    max_lag: int = 12,
    method: str = "pearson",
) -> pd.DataFrame:
    """
    lag > 0 tarkoittaa: x(t-lag) vs y(t) (x "johtaa" y:tä)
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = []
    base = df[["Month", x, y]].dropna().sort_values("Month")
    if base.empty:
        return pd.DataFrame()

    for lag in range(0, max_lag + 1):
        shifted = base.copy()
        shifted[x] = shifted[x].shift(lag)
        tmp = shifted[[x, y]].dropna()
        if len(tmp) < 8:
            r = np.nan
        else:
            r = tmp.corr(method=method).iloc[0, 1]
        out.append({"lag_months": lag, "corr": r})

    return pd.DataFrame(out)


def rolling_corr(
    df: pd.DataFrame,
    x: str,
    y: str,
    window: int = 24,
    method: str = "pearson",
) -> pd.DataFrame:
    """
    Rullaava korrelaatio (ikkuna kk).
    Huom: Rolling.corr() ei tue kaikissa pandas-versioissa method-parametria.
    Spearman = Pearson(rank(x), rank(y))
    """
    if df is None or df.empty:
        return pd.DataFrame()

    base = df[["Month", x, y]].dropna().sort_values("Month").copy()
    if base.empty:
        return pd.DataFrame()

    s = base.set_index("Month")[[x, y]].copy()

    if method.lower() == "spearman":
        s = s.rank()

    rc = s[x].rolling(window=window).corr(s[y])

    out = rc.reset_index()
    out.columns = ["Month", "corr"]
    return out