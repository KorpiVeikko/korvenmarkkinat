# services/energy_view_helper.py
from __future__ import annotations

import pandas as pd


def default_years_back(time_mode: str) -> int:
    return 10 if time_mode == "Vuosi (summa)" else 5


def find_series_col(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        c = str(col).strip().lower()
        if c == "sähkön tuotanto/hankinta":
            return col

    for col in df.columns:
        c = str(col).lower()
        if "sähkön tuotanto/hankinta" in c:
            return col

    return None


def find_measure_col(df: pd.DataFrame) -> str | None:
    candidates: list[str] = []
    for col in df.columns:
        c = str(col).strip().lower()
        if c in ("tiedot", "tieto", "suure", "mittaustieto", "tietolaji"):
            candidates.append(col)
        elif c.startswith("tiedot") or c.startswith("tieto") or c.startswith("suure"):
            candidates.append(col)
    return candidates[0] if candidates else None


def get_best_xcol(df: pd.DataFrame, time_mode: str | None = None) -> str:
    if time_mode == "Vuosi (summa)" and "Aika" in df.columns:
        return "Aika"
    if "Aika_dt" in df.columns and df["Aika_dt"].notna().any():
        return "Aika_dt"
    if "Aika" in df.columns:
        return "Aika"
    return df.columns[0]


def keep_monthly_actual_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Poistaa muutos- ja kertymärivit ja suosii kuukausiarvoja.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    measure_col = find_measure_col(out)
    if not measure_col:
        return out

    s = out[measure_col].astype(str).str.lower()

    drop_mask = (
        s.str.contains("kertym", na=False)
        | s.str.contains("vuoden alusta", na=False)
        | s.str.contains("kumul", na=False)
        | s.str.contains("muutos", na=False)
        | s.str.contains("%", na=False)
        | s.str.contains("indeksi", na=False)
    )
    out = out.loc[~drop_mask].copy()

    if out.empty:
        return out

    s2 = out[measure_col].astype(str).str.lower()
    prefer_month = s2.str.contains("kuukaus", na=False) | s2.str.contains("kk", na=False)
    if prefer_month.any():
        out = out.loc[prefer_month].copy()

    return out


def clip_years(df: pd.DataFrame, years_back: int) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    if "Aika_dt" in out.columns and out["Aika_dt"].notna().any():
        last = out["Aika_dt"].max()
        start = last - pd.DateOffset(years=int(years_back))
        out = out[out["Aika_dt"] >= start].copy()
    return out


def keep_full_years(df: pd.DataFrame, series_col: str | None = None, min_months: int = 12) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    if "Aika_dt" not in out.columns or not out["Aika_dt"].notna().any():
        return out

    out["_Year"] = out["Aika_dt"].dt.year
    out["_Month"] = out["Aika_dt"].dt.month

    if series_col and series_col in out.columns:
        counts = (
            out.dropna(subset=["_Year", "_Month"])
            .groupby(["_Year", series_col])["_Month"]
            .nunique()
            .reset_index(name="_n")
        )
        ok = counts[counts["_n"] >= min_months][["_Year", series_col]]
        out = out.merge(ok, on=["_Year", series_col], how="inner")
    else:
        counts = (
            out.dropna(subset=["_Year", "_Month"])
            .groupby("_Year")["_Month"]
            .nunique()
            .reset_index(name="_n")
        )
        ok_years = set(counts.loc[counts["_n"] >= min_months, "_Year"].tolist())
        out = out[out["_Year"].isin(ok_years)].copy()

    return out.drop(columns=[c for c in ["_Year", "_Month"] if c in out.columns])


def unit_to_twh(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Arvo"] = pd.to_numeric(out["Arvo"], errors="coerce") / 1000.0
    return out


def agg_for_view(df: pd.DataFrame, series_col: str, time_mode: str) -> pd.DataFrame:
    out = df.copy()

    if time_mode == "Vuosi (summa)":
        out["Vuosi"] = out["Aika_dt"].dt.year.astype("Int64")
        out = (
            out.dropna(subset=["Vuosi"])
            .groupby(["Vuosi", series_col], as_index=False)["Arvo"]
            .sum()
        )
        out["Aika_dt"] = pd.to_datetime(out["Vuosi"].astype(str) + "-01-01", errors="coerce")
        out["Aika"] = out["Vuosi"].astype(int).astype(str)

    return out


def latest_val(df: pd.DataFrame) -> float | None:
    if df is None or df.empty:
        return None
    s = pd.to_numeric(df["Arvo"], errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.iloc[-1])


def pct_change(df: pd.DataFrame, periods_back: int) -> float | None:
    if df is None or df.empty:
        return None

    s = pd.to_numeric(df["Arvo"], errors="coerce").dropna()
    if len(s) <= periods_back:
        return None

    latest = float(s.iloc[-1])
    prev = float(s.iloc[-(periods_back + 1)])
    if prev == 0 or pd.isna(prev):
        return None

    return (latest / prev - 1.0) * 100.0


def pct_change_year_span(df: pd.DataFrame, span_years: int = 5) -> float | None:
    """
    Esim. 2024 vs 2020 = 5 vuoden muutos.
    """
    if df is None or df.empty or "Vuosi" not in df.columns or "Arvo" not in df.columns:
        return None

    f = df.dropna(subset=["Vuosi", "Arvo"]).sort_values("Vuosi").copy()
    if f.empty:
        return None

    latest_year = int(f["Vuosi"].iloc[-1])
    latest_val_ = float(f["Arvo"].iloc[-1])

    target_year = latest_year - (span_years - 1)
    prev_rows = f[f["Vuosi"] == target_year]
    if prev_rows.empty:
        return None

    prev_val = float(prev_rows["Arvo"].iloc[-1])
    if prev_val == 0 or pd.isna(prev_val):
        return None

    return (latest_val_ / prev_val - 1.0) * 100.0


def fmt_delta(delta: float | None, label: str | None = None) -> str | None:
    if delta is None:
        return None
    if label:
        return f"{delta:+.1f} % ({label})"
    return f"{delta:+.1f} %"


def series_contains(s: str, *parts: str) -> bool:
    su = str(s).upper()
    return all(p.upper() in su for p in parts)


def first_match(items: list[str], *predicates) -> str | None:
    for pred in predicates:
        for item in items:
            if pred(item):
                return item
    return None


def pick_consumption_series(all_series: list[str]) -> str | None:
    return first_match(
        all_series,
        lambda s: str(s).upper().startswith("SSS"),
        lambda s: "KOKONAISKULUTUS" in str(s).upper(),
    )


def pick_total_production_series(all_series: list[str]) -> str | None:
    return first_match(
        all_series,
        lambda s: str(s).strip().upper().startswith("1 "),
        lambda s: str(s).strip().upper() == "1",
        lambda s: "KOKONAISTUOTANTO" in str(s).upper(),
        lambda s: "SÄHKÖN TUOTANTO YHTEENSÄ" in str(s).upper(),
    )


def pick_net_import_series(all_series: list[str]) -> str | None:
    return first_match(
        all_series,
        lambda s: "NETTOTUONTI" in str(s).upper(),
        lambda s: str(s).strip().upper() == "2",
        lambda s: str(s).strip().upper().startswith("2 "),
    )


def pick_import_country_series(all_series: list[str]) -> list[str]:
    out = []
    for s in all_series:
        su = str(s).upper()
        if not (su.startswith("2.") or su.startswith("2 ")):
            continue
        if "NETTOTUONTI" in su:
            continue
        out.append(s)
    return out


def pick_production_series(all_series: list[str]) -> list[str]:
    out = []
    for s in all_series:
        su = str(s).upper()
        if su.startswith("1") and not su.startswith("1.5.1."):
            out.append(s)
    return out


def pick_key_production_series(all_series: list[str]) -> list[str]:
    production = pick_production_series(all_series)
    selected: list[str] = []
    used: set[str] = set()

    keyword_groups = [
        ("YDIN",),
        ("VESI",),
        ("TUULI",),
        ("AURINKO",),
        ("LÄMPÖ",),
        ("LAMPÖ",),
        ("MUU",),
    ]

    for keys in keyword_groups:
        for s in production:
            if s in used:
                continue
            if series_contains(s, *keys):
                selected.append(s)
                used.add(s)
                break

    if not selected:
        selected = production[:6]

    return selected


def annual_series_from_one(base: pd.DataFrame, series_col: str, series_name: str | None) -> pd.DataFrame:
    if not series_name:
        return pd.DataFrame()

    f = base[base[series_col].astype(str) == str(series_name)].copy()
    if f.empty:
        return pd.DataFrame()

    f["Vuosi"] = f["Aika_dt"].dt.year
    out = f.groupby("Vuosi", as_index=False)["Arvo"].sum().sort_values("Vuosi")
    out["Arvo"] = pd.to_numeric(out["Arvo"], errors="coerce") / 1000.0
    out["Aika"] = out["Vuosi"].astype(str)
    out["Aika_dt"] = pd.to_datetime(out["Vuosi"].astype(str) + "-01-01", errors="coerce")
    return out


def annual_series_from_many(base: pd.DataFrame, series_col: str, series_names: list[str]) -> pd.DataFrame:
    if not series_names:
        return pd.DataFrame()

    f = base[base[series_col].astype(str).isin([str(x) for x in series_names])].copy()
    if f.empty:
        return pd.DataFrame()

    f["Vuosi"] = f["Aika_dt"].dt.year
    out = f.groupby("Vuosi", as_index=False)["Arvo"].sum().sort_values("Vuosi")
    out["Arvo"] = pd.to_numeric(out["Arvo"], errors="coerce") / 1000.0
    out["Aika"] = out["Vuosi"].astype(str)
    out["Aika_dt"] = pd.to_datetime(out["Vuosi"].astype(str) + "-01-01", errors="coerce")
    return out


def annual_share_series(num_df: pd.DataFrame, den_df: pd.DataFrame) -> pd.DataFrame:
    if num_df is None or den_df is None or num_df.empty or den_df.empty:
        return pd.DataFrame()

    out = pd.merge(
        num_df[["Vuosi", "Arvo"]].rename(columns={"Arvo": "num"}),
        den_df[["Vuosi", "Arvo"]].rename(columns={"Arvo": "den"}),
        on="Vuosi",
        how="inner",
    ).sort_values("Vuosi")

    out = out[out["den"] != 0].copy()
    if out.empty:
        return pd.DataFrame()

    out["Arvo"] = (out["num"] / out["den"]) * 100.0
    out["Aika"] = out["Vuosi"].astype(str)
    out["Aika_dt"] = pd.to_datetime(out["Vuosi"].astype(str) + "-01-01", errors="coerce")
    return out


def build_summary_cards(base: pd.DataFrame, series_col: str) -> list[dict]:
    """
    Yhteenveto näyttää aina viimeisimmän kokonaisen vuoden arvot.
    Tuuli- ja ydinvoiman osuudet lasketaan kokonaistuotannosta,
    jotta ne vastaavat tuotanto menetelmittäin (%) -kuvaajaa.
    """
    all_series = sorted(base[series_col].dropna().astype(str).unique().tolist())

    consumption_name = pick_consumption_series(all_series)
    total_production_name = pick_total_production_series(all_series)
    net_import_name = pick_net_import_series(all_series)
    wind_name = first_match(all_series, lambda s: series_contains(s, "TUULI"))
    nuclear_name = first_match(all_series, lambda s: series_contains(s, "YDIN"))

    consumption_annual = annual_series_from_one(base, series_col, consumption_name)
    total_production_annual = annual_series_from_one(base, series_col, total_production_name)
    net_import_annual = annual_series_from_one(base, series_col, net_import_name)
    wind_annual = annual_series_from_one(base, series_col, wind_name)
    nuclear_annual = annual_series_from_one(base, series_col, nuclear_name)

    # ✅ Osuudet suhteessa kokonaistuotantoon, ei kulutukseen
    wind_share_annual = annual_share_series(wind_annual, total_production_annual)
    nuclear_share_annual = annual_share_series(nuclear_annual, total_production_annual)

    cards: list[dict] = []

    for label, df, suffix in [
        ("Kokonaiskulutus", consumption_annual, "TWh"),
        ("Kokonaistuotanto", total_production_annual, "TWh"),
        ("Nettotuonti", net_import_annual, "TWh"),
        ("Tuulivoiman osuus", wind_share_annual, "%"),
        ("Ydinvoiman osuus", nuclear_share_annual, "%"),
    ]:
        if df.empty:
            continue

        latest = latest_val(df)
        year = int(df["Vuosi"].iloc[-1])
        change_5y = pct_change_year_span(df, span_years=5)

        if latest is None:
            continue

        if suffix == "%":
            value = f"{latest:.1f} %"
        else:
            value = f"{latest:,.1f} {suffix}".replace(",", " ")

        cards.append(
            {
                "label": label,
                "value": value,
                "delta": fmt_delta(change_5y, "5 v"),
                "year_caption": f"Vuosi {year}",
            }
        )

    return cards