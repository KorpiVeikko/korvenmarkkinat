# tabs/macro.py
from __future__ import annotations

import itertools
import math
import requests
import pandas as pd
import streamlit as st
import plotly.express as px

from services.uljas import (
    fetch_exports_products,
    fetch_exports_regions,
    fetch_exports_country_detail,
    fetch_exports_country_products,
    list_export_countries,
    fetch_imports_products,
    fetch_imports_regions,
    fetch_imports_country_products,
    fetch_imports_country_detail,
    list_import_countries
)

# ============================================================
# StatFin PxWeb endpoints
# ============================================================
CPI_YOY_URL = "https://pxdata.stat.fi/PXWeb/api/v1/fi/StatFin/khi/statfin_khi_pxt_122p.px"
GDP_132H_URL = "https://pxdata.stat.fi/PXWeb/api/v1/fi/StatFin/ntp/statfin_ntp_pxt_132h.px"
LFS_135Z_URL = "https://pxdata.stat.fi/PXWeb/api/v1/fi/StatFin/tyti/statfin_tyti_pxt_135z.px"

# ============================================================
# Eurostat (velka/BKT ja velan määrä)
# ============================================================
EUROSTAT_API = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"


# ============================================================
# JSON-stat2 parsing + metadata GET
# ============================================================
def _dedupe_columns(cols: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for c in cols:
        c0 = str(c)
        if c0 not in seen:
            seen[c0] = 1
            out.append(c0)
        else:
            seen[c0] += 1
            out.append(f"{c0}__{seen[c0]}")
    return out


def _parse_jsonstat2(payload: dict) -> pd.DataFrame:
    if not isinstance(payload, dict) or "value" not in payload or "dimension" not in payload:
        return pd.DataFrame()

    dim = payload["dimension"]
    ids = payload.get("id") or dim.get("id")
    values = payload.get("value")
    if not ids or values is None:
        return pd.DataFrame()

    dim_levels: list[list[str]] = []
    for did in ids:
        d = dim.get(did, {})
        cat = d.get("category") or {}
        idx = cat.get("index")
        lab = cat.get("label") or {}

        if isinstance(idx, dict) and idx:
            keys = [k for k, _ in sorted(idx.items(), key=lambda kv: kv[1])]
        elif isinstance(idx, list) and idx:
            keys = idx
        else:
            keys = list(lab.keys())

        labels = [lab.get(k, str(k)) for k in keys]
        dim_levels.append(labels)

    combos = list(itertools.product(*dim_levels))
    if len(combos) != len(values):
        return pd.DataFrame()

    cols = _dedupe_columns([str(x) for x in ids])
    df = pd.DataFrame(combos, columns=cols)
    df["Arvo"] = pd.to_numeric(values, errors="coerce")
    df.columns = _dedupe_columns(list(df.columns))
    return df


def _post_px(url: str, query: dict, timeout: int = 45) -> pd.DataFrame:
    r = requests.post(url, json=query, timeout=timeout)
    r.raise_for_status()
    return _parse_jsonstat2(r.json())


def _get_px_meta(url: str, timeout: int = 45) -> dict:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    return j if isinstance(j, dict) else {}


def _find_time_code(meta: dict) -> str | None:
    vars_ = meta.get("variables") or []
    for v in vars_:
        if v.get("time") is True:
            return v.get("code")
        if str(v.get("type", "")).lower() in ("t", "time"):
            return v.get("code")
    for v in vars_:
        c = str(v.get("code", "")).strip().lower()
        if c in ("kuukausi", "vuosineljännes", "neljännes", "vuosi", "aika", "time"):
            return v.get("code")
    return vars_[-1].get("code") if vars_ else None


def _pick_value(meta: dict, var_code: str, want_contains_any: list[str], fallback_first: bool = True) -> str | None:
    vars_ = meta.get("variables") or []
    var = None
    for v in vars_:
        if str(v.get("code", "")).strip().lower() == var_code.strip().lower():
            var = v
            break
    if var is None:
        return None

    values = var.get("values") or []
    texts = var.get("valueTexts") or []
    if not values:
        return None
    if not texts or len(values) != len(texts):
        return values[0] if fallback_first else None

    wants = [w.strip().lower() for w in want_contains_any if w and w.strip()]
    for i, t in enumerate(texts):
        tl = str(t).lower()
        if any(w in tl for w in wants):
            return values[i]
    return values[0] if fallback_first else None


def _add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    out.columns = _dedupe_columns(list(out.columns))

    candidates = []
    for c in out.columns:
        cl = str(c).strip().lower()
        if cl in ("kuukausi", "vuosineljännes", "neljännes", "aika", "time", "vuosi", "quarter"):
            candidates.append(c)
    time_col = candidates[0] if candidates else out.columns[0]

    s = out[time_col].astype(str).str.strip()
    out["Aika"] = s

    m = s.str.extract(r"^(?P<y>\d{4})M(?P<m>\d{2})$")
    q = s.str.extract(r"^(?P<y>\d{4})Q(?P<q>\d)$")
    y_only = s.str.extract(r"^(?P<y>\d{4})$")

    if m["y"].notna().any():
        out["Vuosi_num"] = pd.to_numeric(m["y"], errors="coerce")
        out["Kuukausi_num"] = pd.to_numeric(m["m"], errors="coerce")
        out["Aika_dt"] = pd.to_datetime(
            out["Vuosi_num"].astype("Int64").astype(str) + "-" +
            out["Kuukausi_num"].astype("Int64").astype(str).str.zfill(2) + "-01",
            errors="coerce",
        )
    elif q["y"].notna().any():
        out["Vuosi_num"] = pd.to_numeric(q["y"], errors="coerce")
        qn = pd.to_numeric(q["q"], errors="coerce")
        start_month = (qn - 1) * 3 + 1
        out["Aika_dt"] = pd.to_datetime(
            out["Vuosi_num"].astype("Int64").astype(str) + "-" +
            start_month.astype("Int64").astype(str).str.zfill(2) + "-01",
            errors="coerce",
        )
    elif y_only["y"].notna().any():
        out["Vuosi_num"] = pd.to_numeric(y_only["y"], errors="coerce")
        out["Aika_dt"] = pd.to_datetime(out["Vuosi_num"].astype("Int64").astype(str) + "-01-01", errors="coerce")
    else:
        out["Aika_dt"] = pd.to_datetime(s, errors="coerce")

    return out


# ============================================================
# General helpers
# ============================================================
def _fmt(x: float | None, decimals: int = 1, suffix: str = "") -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:,.{decimals}f}".replace(",", " ") + suffix


def _fmt_millions(x: float | None, decimals: int = 0) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x / 1_000_000:,.{decimals}f} milj. €".replace(",", " ")


def _clip_by_years(df: pd.DataFrame, date_col: str, years: int) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    f = df.copy()
    f[date_col] = pd.to_datetime(f[date_col], errors="coerce")
    f = f.dropna(subset=[date_col])
    if f.empty:
        return f
    end = f[date_col].max()
    start = end - pd.DateOffset(years=int(years))
    return f[f[date_col] >= start].copy()


def _kpi_card(label: str, value: str, delta: str | None = None, caption: str | None = None):
    with st.container(border=True):
        st.metric(label, value, delta)
        if caption:
            st.caption(caption)


def _norm(s: str) -> str:
    s = str(s).lower()
    return s.replace("ä", "a").replace("ö", "o").replace("å", "a")


def _yoy_delta(series: pd.Series, periods: int) -> float | None:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) <= periods:
        return None
    last = float(s.iloc[-1])
    then = float(s.iloc[-(periods + 1)])
    return last - then


def _pct_change_vs_year_ago(df: pd.DataFrame, date_col: str, value_col: str) -> tuple[float | None, float | None, pd.Timestamp | None]:
    if df is None or df.empty:
        return None, None, None

    f = df.copy()
    f[date_col] = pd.to_datetime(f[date_col], errors="coerce")
    f[value_col] = pd.to_numeric(f[value_col], errors="coerce")
    f = f.dropna(subset=[date_col, value_col]).sort_values(date_col)

    if f.empty:
        return None, None, None

    latest_row = f.iloc[-1]
    latest_date = pd.to_datetime(latest_row[date_col])
    latest_val = float(latest_row[value_col])

    prev_year_date = latest_date - pd.DateOffset(years=1)
    prev = f[f[date_col] == prev_year_date]

    if prev.empty:
        return latest_val, None, latest_date

    prev_val = float(prev.iloc[-1][value_col])
    if prev_val == 0:
        pct = None
    else:
        pct = ((latest_val / prev_val) - 1.0) * 100.0

    return latest_val, pct, latest_date



def _build_total_exports_from_products(dfp: pd.DataFrame) -> pd.DataFrame:
    if dfp is None or dfp.empty:
        return pd.DataFrame(columns=["Aika_dt", "Vienti_eur"])
    return (
        dfp.groupby("Aika_dt", as_index=False)["Vienti_eur"]
        .sum()
        .sort_values("Aika_dt")
    )


def _to_yearly(df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=[date_col, value_col])

    if d.empty:
        return pd.DataFrame()

    d["Vuosi"] = d[date_col].dt.year

    yearly = (
        d.groupby("Vuosi", as_index=False)[value_col]
        .sum()
        .sort_values("Vuosi")
    )

    yearly["Miljardia"] = yearly[value_col] / 1_000_000_000
    yearly["Miljoonaa"] = yearly[value_col] / 1_000_000
    return yearly


def _line_export_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str):
    yearly = _to_yearly(df, x_col, y_col)

    if yearly is None or yearly.empty:
        st.info("Ei dataa näytettäväksi.")
        return

    latest_year = yearly["Vuosi"].max()

    fig = px.line(
        yearly,
        x="Vuosi",
        y="Miljardia",
        markers=True,
        title=title + " (vuositaso)",
        labels={"Vuosi": "Vuosi", "Miljardia": "Vienti (mrd €)"},
    )

    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>Vienti: %{y:.2f} mrd €<extra></extra>"
    )

    last_point = yearly[yearly["Vuosi"] == latest_year]

    fig.add_scatter(
        x=last_point["Vuosi"],
        y=last_point["Miljardia"],
        mode="markers",
        marker=dict(size=14),
        name="Viimeisin vuosi",
    )

    fig.update_layout(
        yaxis_title="Vienti (miljardia €)",
        xaxis_title="Vuosi",
        legend_title="",
    )

    st.plotly_chart(fig, use_container_width=True)

def _fmt_money(x: float | None) -> str:

    if x is None or pd.isna(x):
        return "—"

    x = float(x)

    if abs(x) >= 1_000_000_000:
        val = x / 1_000_000_000
        return f"{val:,.1f} mrd €".replace(",", " ")

    val = x / 1_000_000
    return f"{val:,.0f} milj. €".replace(",", " ")

def _build_total_imports_from_products(dfp: pd.DataFrame) -> pd.DataFrame:
    if dfp is None or dfp.empty:
        return pd.DataFrame(columns=["Aika_dt", "Tuonti_eur"])
    return (
        dfp.groupby("Aika_dt", as_index=False)["Tuonti_eur"]
        .sum()
        .sort_values("Aika_dt")
    )

def _build_trade_balance(exports_df: pd.DataFrame, imports_df: pd.DataFrame) -> pd.DataFrame:
    if (exports_df is None or exports_df.empty) and (imports_df is None or imports_df.empty):
        return pd.DataFrame(columns=["Aika_dt", "Vienti_eur", "Tuonti_eur", "Kauppatase_eur"])

    d = pd.merge(exports_df, imports_df, on="Aika_dt", how="outer").sort_values("Aika_dt")
    d["Vienti_eur"] = pd.to_numeric(d["Vienti_eur"], errors="coerce").fillna(0)
    d["Tuonti_eur"] = pd.to_numeric(d["Tuonti_eur"], errors="coerce").fillna(0)
    d["Kauppatase_eur"] = d["Vienti_eur"] - d["Tuonti_eur"]
    return d


def _line_trade_chart(exports_df: pd.DataFrame, imports_df: pd.DataFrame, title: str):
    exp_y = _to_yearly(exports_df, "Aika_dt", "Vienti_eur")
    imp_y = _to_yearly(imports_df, "Aika_dt", "Tuonti_eur")

    if exp_y.empty and imp_y.empty:
        st.info("Ei dataa näytettäväksi.")
        return

    exp_y = exp_y.rename(columns={"Miljardia": "Arvo_mrd"})
    exp_y["Sarja"] = "Vienti"
    exp_y = exp_y[["Vuosi", "Arvo_mrd", "Sarja"]]

    imp_y = imp_y.rename(columns={"Miljardia": "Arvo_mrd"})
    imp_y["Sarja"] = "Tuonti"
    imp_y = imp_y[["Vuosi", "Arvo_mrd", "Sarja"]]

    plot_df = pd.concat([exp_y, imp_y], ignore_index=True).sort_values(["Sarja", "Vuosi"])

    fig = px.line(
        plot_df,
        x="Vuosi",
        y="Arvo_mrd",
        color="Sarja",
        markers=True,
        title=title + " (vuositaso)",
        labels={"Vuosi": "Vuosi", "Arvo_mrd": "Arvo (mrd €)", "Sarja": ""},
    )

    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>Arvo: %{y:.2f} mrd €<extra></extra>"
    )

    fig.update_layout(
        yaxis_title="Arvo (miljardia €)",
        xaxis_title="Vuosi",
    )

    st.plotly_chart(fig, use_container_width=True)

def _latest_full_year_change(df: pd.DataFrame, date_col: str, value_col: str) -> tuple[float | None, float | None, int | None]:
    yearly = _to_yearly(df, date_col, value_col)

    if yearly is None or yearly.empty:
        return None, None, None

    latest_row = yearly.iloc[-1]
    latest_year = int(latest_row["Vuosi"])
    latest_val = float(latest_row[value_col])

    prev = yearly[yearly["Vuosi"] == latest_year - 1]
    if prev.empty:
        pct = None
    else:
        prev_val = float(prev.iloc[0][value_col])
        pct = None if prev_val == 0 else ((latest_val / prev_val) - 1.0) * 100.0

    return latest_val, pct, latest_year

def _metric_billions_pct_year(label: str, latest_val: float | None, pct: float | None, year: int | None):
    with st.container(border=True):
        st.metric(
            label,
            _fmt_money(latest_val),
            f"{pct:+.1f} % vs. edellinen vuosi" if pct is not None else None,
        )
        if year is not None:
            st.caption(f"Viimeisin kokonainen vuosi: {year}")


# ============================================================
# 1) Inflaatio (kk, YoY)
# ============================================================
def fetch_inflation_yoy() -> pd.DataFrame:
    meta = _get_px_meta(CPI_YOY_URL)
    info_code = "Tiedot"
    time_code = _find_time_code(meta) or "Kuukausi"
    info_val = _pick_value(meta, info_code, ["vuosimuutos", "year-on-year", "%"], fallback_first=True)

    query = {
    "query": [
        {"code": info_code, "selection": {"filter": "item", "values": [info_val] if info_val else ["*"]}},
        {"code": time_code, "selection": {"filter": "all", "values": ["*"]}},
    ],
    "response": {"format": "json-stat2"},
    }
    return _add_time_columns(_post_px(CPI_YOY_URL, query))


def build_inflation_series(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    f = df.copy()
    f["Arvo"] = pd.to_numeric(f["Arvo"], errors="coerce")
    f = f.dropna(subset=["Aika_dt", "Arvo"]).sort_values("Aika_dt")
    if f.empty:
        return pd.DataFrame()
    return f[["Aika_dt", "Arvo"]].rename(columns={"Aika_dt": "Date", "Arvo": "inflation_yoy"})


# ============================================================
# 2) BKT YoY (132h) – % YoY
# ============================================================
def fetch_gdp_growth_yoy() -> pd.DataFrame:
    meta = _get_px_meta(GDP_132H_URL)
    vars_ = meta.get("variables") or []
    if not vars_:
        return pd.DataFrame()

    time_code = _find_time_code(meta) or "Vuosineljännes"
    tx_code = "Taloustoimi"
    info_code = "Tiedot"

    tx_val = _pick_value(meta, tx_code, ["b1gmh", "bruttokansantuote", "gdp"], fallback_first=True)
    info_yoy = (
        _pick_value(meta, info_code, ["%", "edellisestä vuodesta"], fallback_first=False)
        or _pick_value(meta, info_code, ["edellisestä vuodesta"], fallback_first=False)
        or _pick_value(meta, info_code, ["vuosimuutos", "%"], fallback_first=True)
    )

    query = {
        "query": [
            {"code": tx_code, "selection": {"filter": "item", "values": [tx_val] if tx_val else ["*"]}},
            {"code": info_code, "selection": {"filter": "item", "values": [info_yoy] if info_yoy else ["*"]}},
            {"code": time_code, "selection": {"filter": "all", "values": ["*"]}},
        ],
        "response": {"format": "json-stat2"},
    }

    df = _add_time_columns(_post_px(GDP_132H_URL, query))
    if df is None or df.empty:
        return pd.DataFrame()

    f = df.copy()
    f["Arvo"] = pd.to_numeric(f["Arvo"], errors="coerce")
    f = f.dropna(subset=["Aika_dt", "Arvo"]).sort_values("Aika_dt")
    if f.empty:
        return pd.DataFrame()

    return f[["Aika_dt", "Arvo"]].rename(columns={"Aika_dt": "Date", "Arvo": "gdp_yoy"})


# ============================================================
# 3) Työttömyys (135z)
# ============================================================
def fetch_unemployment_135z() -> pd.DataFrame:
    meta = _get_px_meta(LFS_135Z_URL)
    vars_ = meta.get("variables") or []
    if not vars_:
        return pd.DataFrame()

    time_code = _find_time_code(meta) or (vars_[-1].get("code", "Kuukausi"))
    query_parts: list[dict] = []

    for v in vars_:
        code = v.get("code")
        if not code:
            continue

        if code == time_code:
            query_parts.append({"code": code, "selection": {"filter": "all", "values": ["*"]}})
            continue

        cl = _norm(code)

        if cl == "tiedot":
            values = v.get("values") or []
            texts = v.get("valueTexts") or []
            if values and texts and len(values) == len(texts):
                rate_vals = []
                unemp_vals = []
                for val, txt in zip(values, texts):
                    t = _norm(txt)
                    if ("tyottomyysaste" in t) or ("unemployment rate" in t):
                        rate_vals.append(val)
                    if ("tyottom" in t) or ("unemployed" in t):
                        unemp_vals.append(val)

                chosen = []
                if rate_vals:
                    chosen.append(rate_vals[0])
                if unemp_vals and unemp_vals[0] not in chosen:
                    chosen.append(unemp_vals[0])

                query_parts.append({"code": code, "selection": {"filter": "item", "values": chosen or ["*"]}})
            else:
                query_parts.append({"code": code, "selection": {"filter": "all", "values": ["*"]}})
            continue

        if "kausi" in cl:
            chosen = _pick_value(meta, code, ["kausitasoitettu", "seasonally adjusted", "sa"], fallback_first=True)
            query_parts.append({"code": code, "selection": {"filter": "item", "values": [chosen] if chosen else ["*"]}})
            continue

        if "sukupu" in cl:
            chosen = _pick_value(meta, code, ["yhteensa", "total", "miehet ja naiset"], fallback_first=True)
            query_parts.append({"code": code, "selection": {"filter": "item", "values": [chosen] if chosen else ["*"]}})
            continue

        if "ika" in cl:
            chosen = _pick_value(meta, code, ["15–74", "15-74", "15 74", "yhteensa", "total"], fallback_first=True)
            query_parts.append({"code": code, "selection": {"filter": "item", "values": [chosen] if chosen else ["*"]}})
            continue

        chosen = _pick_value(meta, code, ["yhteensa", "total"], fallback_first=True)
        query_parts.append({"code": code, "selection": {"filter": "item", "values": [chosen] if chosen else ["*"]}})

    query = {"query": query_parts, "response": {"format": "json-stat2"}}
    df = _post_px(LFS_135Z_URL, query)
    return _add_time_columns(df)


def build_unemployment_series(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    f = df.copy()
    f["Arvo"] = pd.to_numeric(f["Arvo"], errors="coerce")
    f = f.dropna(subset=["Aika_dt", "Arvo"]).sort_values("Aika_dt")
    if f.empty:
        return pd.DataFrame()

    tiedot_col = None
    for c in f.columns:
        if _norm(c).startswith("tiedot"):
            tiedot_col = c
            break

    if tiedot_col is None:
        out = f.groupby("Aika_dt", as_index=False)["Arvo"].mean().rename(
            columns={"Aika_dt": "Date", "Arvo": "unemployment_rate"}
        )
        out["unemployed_1000"] = pd.NA
        return out

    f["_t"] = f[tiedot_col].astype(str).map(_norm)

    rate_mask = f["_t"].str.contains("tyottomyysaste") | f["_t"].str.contains("unemployment rate")
    unemp_mask = f["_t"].str.contains("tyottom") | f["_t"].str.contains("unemployed")

    rate = f[rate_mask].copy()
    unemp = f[unemp_mask & (~rate_mask)].copy()

    rate_s = (
        rate.groupby("Aika_dt", as_index=False)["Arvo"].mean()
        .rename(columns={"Aika_dt": "Date", "Arvo": "unemployment_rate"})
    )
    unemp_s = (
        unemp.groupby("Aika_dt", as_index=False)["Arvo"].mean()
        .rename(columns={"Aika_dt": "Date", "Arvo": "unemployed_1000"})
    )

    out = pd.merge(rate_s, unemp_s, on="Date", how="outer").sort_values("Date")
    return out


# ============================================================
# 5) Eurostat: velka/BKT (%) + velan määrä (milj. €)
# ============================================================
def _eurostat_timeseries_to_df(j: dict, value_name: str) -> pd.DataFrame:
    if "value" not in j or "dimension" not in j:
        return pd.DataFrame()

    dim = j["dimension"]
    time_id = "time" if "time" in dim else (j.get("id") or [None])[-1]
    time_cat = dim.get(time_id, {}).get("category", {}) if time_id else {}
    time_index = time_cat.get("index", {})
    time_label = time_cat.get("label", {})

    if isinstance(time_index, dict) and time_index:
        time_keys = [k for k, _ in sorted(time_index.items(), key=lambda kv: kv[1])]
    else:
        time_keys = list(time_label.keys())

    values = j.get("value", {})
    out = []
    for i, tk in enumerate(time_keys):
        v = values.get(str(i))
        if v is None:
            v = values.get(i)
        out.append({"Quarter": tk, value_name: pd.to_numeric(v, errors="coerce")})

    df = pd.DataFrame(out).dropna(subset=[value_name])
    if df.empty:
        return df

    df["Date"] = pd.PeriodIndex(df["Quarter"], freq="Q").to_timestamp(how="start")
    return df.sort_values("Date")


def eurostat_fetch_gov_debt_pct_gdp(geo: str = "FI") -> pd.DataFrame:
    url = f"{EUROSTAT_API}/gov_10q_ggdebt"
    params = {
        "lang": "EN",
        "format": "JSON",
        "freq": "Q",
        "sector": "S13",
        "na_item": "GD",
        "unit": "PC_GDP",
        "geo": geo,
    }
    r = requests.get(url, params=params, timeout=45)
    r.raise_for_status()
    return _eurostat_timeseries_to_df(r.json(), value_name="debt_pct_gdp")


def eurostat_fetch_gov_debt_mio_eur(geo: str = "FI") -> pd.DataFrame:
    url = f"{EUROSTAT_API}/gov_10q_ggdebt"
    params = {
        "lang": "EN",
        "format": "JSON",
        "freq": "Q",
        "sector": "S13",
        "na_item": "GD",
        "unit": "MIO_EUR",
        "geo": geo,
    }
    r = requests.get(url, params=params, timeout=45)
    r.raise_for_status()
    return _eurostat_timeseries_to_df(r.json(), value_name="debt_mio_eur")


# ============================================================
# Streamlit caches
# ============================================================
@st.cache_data(show_spinner="Haetaan inflaatio (Tilastokeskus)…")
def load_inflation() -> pd.DataFrame:
    return build_inflation_series(fetch_inflation_yoy())


@st.cache_data(show_spinner="Haetaan BKT YoY (Tilastokeskus)…")
def load_gdp_yoy() -> pd.DataFrame:
    return fetch_gdp_growth_yoy()


@st.cache_data(show_spinner="Haetaan työttömyys (Tilastokeskus)…")
def load_unemployment() -> pd.DataFrame:
    return build_unemployment_series(fetch_unemployment_135z())


@st.cache_data(show_spinner="Haetaan velka/BKT (Eurostat)…")
def load_debt_pct_gdp() -> pd.DataFrame:
    return eurostat_fetch_gov_debt_pct_gdp("FI")


@st.cache_data(show_spinner="Haetaan velan määrä (Eurostat)…")
def load_debt_mio_eur() -> pd.DataFrame:
    return eurostat_fetch_gov_debt_mio_eur("FI")


@st.cache_data(show_spinner="Haetaan vientimaat (Uljas)…")
def load_export_countries() -> pd.DataFrame:
    return list_export_countries(lang="fi")


@st.cache_data(show_spinner="Haetaan maan vientisarja (Uljas)…")
def load_export_country_detail(country_code: str, months: int) -> pd.DataFrame:
    df, _ = fetch_exports_country_detail(country_code=country_code, months=months, lang="fi")
    return df


@st.cache_data(show_spinner="Haetaan maan tuoteryhmävienti (Uljas)…")
def load_export_country_products(country_code: str, months: int) -> pd.DataFrame:
    df, _ = fetch_exports_country_products(country_code=country_code, months=months, lang="fi")
    return df


@st.cache_data(show_spinner="Haetaan tuontimaat (Uljas)…")
def load_import_countries() -> pd.DataFrame:
    return list_import_countries(lang="fi")


@st.cache_data(show_spinner="Haetaan maan tuontisarja (Uljas)…")
def load_import_country_detail(country_code: str, months: int) -> pd.DataFrame:
    df, _ = fetch_imports_country_detail(country_code=country_code, months=months, lang="fi")
    return df


@st.cache_data(show_spinner="Haetaan maan tuoteryhmätuonti (Uljas)…")
def load_import_country_products(country_code: str, months: int) -> pd.DataFrame:
    df, _ = fetch_imports_country_products(country_code=country_code, months=months, lang="fi")
    return df


# ============================================================
# UI sections
# ============================================================
def _section_inflation(df: pd.DataFrame, years: int):
    st.subheader("📈 Inflaatio (YoY, %, kuukausi)")
    st.caption("YoY = muutos verrattuna edellisvuoden samaan kuukauteen. Negatiivinen inflaatio = deflaatio.")

    if df is None or df.empty:
        st.warning("Inflaatiodataa ei saatu.")
        return

    d = _clip_by_years(df, "Date", years)
    fig = px.bar(d, x="Date", y="inflation_yoy", labels={"Date": "Kuukausi", "inflation_yoy": "Inflaatio (YoY, %)"})
    fig.update_yaxes(ticksuffix=" %", zeroline=True)
    st.plotly_chart(fig, use_container_width=True, key="macro_inflation_bar")


def _section_gdp_yoy(df: pd.DataFrame, years: int):
    st.subheader("🏛️ BKT:n kasvu (YoY, kvartaali)")
    st.caption("YoY = volyymin muutos verrattuna edellisvuoden vastaavaan neljännekseen (%).")

    if df is None or df.empty:
        st.warning("BKT YoY -dataa ei saatu.")
        return

    d = _clip_by_years(df, "Date", years)
    d["gdp_yoy"] = pd.to_numeric(d["gdp_yoy"], errors="coerce")
    d = d.dropna(subset=["Date", "gdp_yoy"]).sort_values("Date")
    if d.empty:
        st.warning("BKT YoY -sarja on tyhjä valitulla aikajänteellä.")
        return

    fig = px.line(
        d,
        x="Date",
        y="gdp_yoy",
        markers=True,
        labels={"Date": "Kvartaali", "gdp_yoy": "BKT YoY (%)"},
        title="BKT:n kasvuvauhti (YoY, %)",
    )
    fig.update_yaxes(ticksuffix=" %", zeroline=True)
    st.plotly_chart(fig, use_container_width=True, key="macro_gdp_yoy_line")


def _section_unemployment(df: pd.DataFrame, years: int):
    st.subheader("🧑‍💼 Työttömyys (kuukausi)")
    st.caption(
        "Työttömyysaste (%) = työttömien osuus työvoimasta. "
        "Työvoima = työlliset + työttömät. "
        "Työttömät (1000 hlö) kuvaa määrää."
    )

    if df is None or df.empty:
        st.warning("Työttömyysdataa ei saatu.")
        return

    d = _clip_by_years(df, "Date", years)

    latest_date = pd.to_datetime(d["Date"].dropna().iloc[-1]).date() if d["Date"].notna().any() else None
    latest_rate = (
        float(pd.to_numeric(d.get("unemployment_rate"), errors="coerce").dropna().iloc[-1])
        if "unemployment_rate" in d.columns and d["unemployment_rate"].notna().any()
        else None
    )
    latest_level = (
        float(pd.to_numeric(d.get("unemployed_1000"), errors="coerce").dropna().iloc[-1])
        if "unemployed_1000" in d.columns and d["unemployed_1000"].notna().any()
        else None
    )

    cA, cB = st.columns(2, gap="large")
    with cA:
        st.metric("Työttömyysaste (%)", _fmt(latest_rate, 2, " %"))
        if latest_date:
            st.caption(f"Kuukausi: {latest_date}")
    with cB:
        st.metric("Työttömät (1000 hlö)", _fmt(latest_level, 0, ""))
        if latest_date:
            st.caption(f"Kuukausi: {latest_date}")

    st.divider()

    c1, c2 = st.columns(2, gap="large")

    with c1:
        if "unemployment_rate" in d.columns and d["unemployment_rate"].notna().any():
            fig = px.line(
                d.dropna(subset=["unemployment_rate"]),
                x="Date",
                y="unemployment_rate",
                markers=True,
                labels={"Date": "Kuukausi", "unemployment_rate": "%"},
                title="Työttömyysaste (%)",
            )
            fig.update_yaxes(ticksuffix=" %", zeroline=True)
            st.plotly_chart(fig, use_container_width=True, key="macro_unemp_rate_line")
        else:
            st.info("Työttömyysaste (%) ei löytynyt tästä aineistosta.")

    with c2:
        if "unemployed_1000" in d.columns and d["unemployed_1000"].notna().any():
            fig = px.line(
                d.dropna(subset=["unemployed_1000"]),
                x="Date",
                y="unemployed_1000",
                markers=True,
                labels={"Date": "Kuukausi", "unemployed_1000": "1000 henkilöä"},
                title="Työttömät (1000 henkilöä)",
            )
            st.plotly_chart(fig, use_container_width=True, key="macro_unemp_level_line")
        else:
            st.info("Työttömien määrä (1000 hlö) ei löytynyt tästä aineistosta.")


def _section_exports_uljas(months: int = 48):
    st.subheader("🚢 Vienti – rakenne (Tulli / Uljas)")
    st.caption(
        "Tavaravienti SITC-luokituksella. Tuoteryhmät on koottu SITC2-koodeista teollisuusaloiksi. "
        "Maanäkymässä pinottu pääkuva näyttää 5 suurinta vientimaata sekä ryhmän 'Muut maat'."
    )

    t1, t2 = st.tabs(["📦 Tuoteryhmät", "🌍 Maat"])

    dfp, dbg_p = fetch_exports_products(months=months, lang="fi")
    dfr, dbg_r = fetch_exports_regions(months=months, lang="fi")

    with t1:
        if dfp is None or dfp.empty:
            st.warning("Tuoteryhmävientiä ei saatu ladattua (Uljas).")
            with st.expander("🔍 Debug", expanded=True):
                st.write("**Onnistunut ifile:**", dbg_p.get("ok_ifile"))
                st.write("**Aikakoodit:**", dbg_p.get("time_codes"))
                st.write("**Virhesyyt:**")
                st.code("\n".join(dbg_p.get("why_failed", [])) or "—")
        else:
            fig = px.bar(
                dfp,
                x="Aika_dt",
                y="Vienti_eur",
                color="Tuoteryhmä",
                barmode="stack",
                title="Vienti pinottuna tuoteryhmittäin",
                labels={"Aika_dt": "Aika", "Vienti_eur": "Vienti (€)", "Tuoteryhmä": "Tuoteryhmä"},
            )
            st.plotly_chart(fig, use_container_width=True, key="exports_products_uljas_ok")

            st.divider()
            st.markdown("#### 🔎 Tuoteryhmän tarkastelu")

            groups = sorted(dfp["Tuoteryhmä"].dropna().unique().tolist())
            selected_group = st.selectbox(
                "Valitse tarkasteltava tuoteryhmä",
                groups,
                key="macro_export_group_detail",
            )

            grp_df = (
                dfp[dfp["Tuoteryhmä"] == selected_group]
                .copy()
                .sort_values("Aika_dt")
            )

            latest_group, pct_group, latest_group_year = _latest_full_year_change(grp_df, "Aika_dt", "Vienti_eur")
            _metric_billions_pct_year(selected_group, latest_group, pct_group, latest_group_year)

            _line_export_chart(
                grp_df,
                "Aika_dt",
                "Vienti_eur",
                f"{selected_group} – viennin kehitys",
            )

    with t2:
        if dfr is None or dfr.empty:
            st.warning("Maavientiä ei saatu ladattua (Uljas).")
            with st.expander("🔍 Debug", expanded=True):
                st.write("**Onnistunut ifile:**", dbg_r.get("ok_ifile"))
                st.write("**Aikakoodit:**", dbg_r.get("time_codes"))
                st.write("**Top 5 maat:**", dbg_r.get("top5_names"))
                st.write("**Virhesyyt:**")
                st.code("\n".join(dbg_r.get("why_failed", [])) or "—")
        else:
            fig = px.bar(
                dfr,
                x="Aika_dt",
                y="Vienti_eur",
                color="Alue",
                barmode="stack",
                title="Vienti pinottuna maaryhmittäin",
                labels={"Aika_dt": "Aika", "Vienti_eur": "Vienti (€)", "Alue": "Maa / ryhmä"},
            )
            st.plotly_chart(fig, use_container_width=True, key="exports_regions_uljas_ok")

            if dbg_r.get("top5_names"):
                st.caption("Suurimmat vientimaat tällä aikajaksolla: " + ", ".join(dbg_r["top5_names"]))

            st.divider()
            st.markdown("#### 🔎 Maan tarkastelu")

            countries_df = load_export_countries()
            if countries_df is None or countries_df.empty:
                st.info("Maalistaa ei saatu haettua.")
            else:
                country_options = countries_df["name"].tolist()
                selected_country_name = st.selectbox(
                    "Valitse tarkasteltava maa",
                    country_options,
                    key="macro_export_country_detail_name",
                )

                selected_country_code = (
                    countries_df.loc[countries_df["name"] == selected_country_name, "code"]
                    .iloc[0]
                )

                country_total_df = load_export_country_detail(selected_country_code, months)
                country_prod_df = load_export_country_products(selected_country_code, months)

                latest_country, pct_country, latest_country_year = _latest_full_year_change(
                    country_total_df, "Aika_dt", "Vienti_eur"
                )
                _metric_billions_pct_year(selected_country_name, latest_country, pct_country, latest_country_year)

                _line_export_chart(
                    country_total_df,
                    "Aika_dt",
                    "Vienti_eur",
                    f"{selected_country_name} – viennin kehitys",
                )

                st.divider()
                st.markdown("#### 🔎 Tuoteryhmä kyseiseen maahan")

                if country_prod_df is None or country_prod_df.empty:
                    st.info("Maan tuoteryhmävientiä ei saatu ladattua.")
                else:
                    country_groups = sorted(country_prod_df["Tuoteryhmä"].dropna().unique().tolist())
                    selected_country_group = st.selectbox(
                        "Valitse tuoteryhmä",
                        country_groups,
                        key="macro_export_country_group_detail",
                    )

                    country_group_df = (
                        country_prod_df[country_prod_df["Tuoteryhmä"] == selected_country_group]
                        .copy()
                        .sort_values("Aika_dt")
                    )

                    latest_country_group, pct_country_group, latest_country_group_year = _latest_full_year_change(
                        country_group_df, "Aika_dt", "Vienti_eur"
                    )
                    _metric_billions_pct_year(
                        f"{selected_country_group} → {selected_country_name}",
                        latest_country_group,
                        pct_country_group,
                        latest_country_group_year,
                    )

                    _line_export_chart(
                        country_group_df,
                        "Aika_dt",
                        "Vienti_eur",
                        f"{selected_country_group} – viennin kehitys maahan {selected_country_name}",
                    )

def _section_imports_uljas(months: int = 48):
    st.subheader("📥 Tuonti – rakenne (Tulli / Uljas)")
    st.caption(
        "Tavaratuonti SITC-luokituksella. Tuoteryhmät on koottu SITC2-koodeista teollisuusaloiksi. "
        "Maanäkymässä pinottu pääkuva näyttää 5 suurinta tuontimaata sekä ryhmän 'Muut maat'."
    )

    t1, t2 = st.tabs(["📦 Tuoteryhmät", "🌍 Maat"])

    dfp, dbg_p = fetch_imports_products(months=months, lang="fi")
    dfr, dbg_r = fetch_imports_regions(months=months, lang="fi")

    with t1:
        if dfp is None or dfp.empty:
            st.warning("Tuonnin tuoteryhmäaineistoa ei saatu ladattua (Uljas).")
            with st.expander("🔍 Debug", expanded=True):
                st.write("**Onnistunut ifile:**", dbg_p.get("ok_ifile"))
                st.write("**Aikakoodit:**", dbg_p.get("time_codes"))
                st.write("**Virhesyyt:**")
                st.code("\n".join(dbg_p.get("why_failed", [])) or "—")
        else:
            fig = px.bar(
                dfp,
                x="Aika_dt",
                y="Tuonti_eur",
                color="Tuoteryhmä",
                barmode="stack",
                title="Tuonti pinottuna tuoteryhmittäin",
                labels={"Aika_dt": "Aika", "Tuonti_eur": "Tuonti (€)", "Tuoteryhmä": "Tuoteryhmä"},
            )
            st.plotly_chart(fig, use_container_width=True, key="imports_products_uljas_ok")

            st.divider()
            st.markdown("#### 🔎 Tuoteryhmän tarkastelu")

            groups = sorted(dfp["Tuoteryhmä"].dropna().unique().tolist())
            selected_group = st.selectbox(
                "Valitse tarkasteltava tuoteryhmä",
                groups,
                key="macro_import_group_detail",
            )

            grp_df = (
                dfp[dfp["Tuoteryhmä"] == selected_group]
                .copy()
                .sort_values("Aika_dt")
            )

            latest_group, pct_group, latest_group_year = _latest_full_year_change(grp_df, "Aika_dt", "Tuonti_eur")
            _metric_billions_pct_year(selected_group, latest_group, pct_group, latest_group_year)

            _line_export_chart(
                grp_df,
                "Aika_dt",
                "Tuonti_eur",
                f"{selected_group} – tuonnin kehitys",
            )

    with t2:
        if dfr is None or dfr.empty:
            st.warning("Maatuonnin aineistoa ei saatu ladattua (Uljas).")
            with st.expander("🔍 Debug", expanded=True):
                st.write("**Onnistunut ifile:**", dbg_r.get("ok_ifile"))
                st.write("**Aikakoodit:**", dbg_r.get("time_codes"))
                st.write("**Top 5 maat:**", dbg_r.get("top5_names"))
                st.write("**Virhesyyt:**")
                st.code("\n".join(dbg_r.get("why_failed", [])) or "—")
        else:
            fig = px.bar(
                dfr,
                x="Aika_dt",
                y="Tuonti_eur",
                color="Alue",
                barmode="stack",
                title="Tuonti pinottuna maaryhmittäin",
                labels={"Aika_dt": "Aika", "Tuonti_eur": "Tuonti (€)", "Alue": "Maa / ryhmä"},
            )
            st.plotly_chart(fig, use_container_width=True, key="imports_regions_uljas_ok")

            if dbg_r.get("top5_names"):
                st.caption("Suurimmat tuontimaat tällä aikajaksolla: " + ", ".join(dbg_r["top5_names"]))

            st.divider()
            st.markdown("#### 🔎 Maan tarkastelu")

            countries_df = load_import_countries()
            if countries_df is None or countries_df.empty:
                st.info("Maalistaa ei saatu haettua.")
            else:
                country_options = countries_df["name"].tolist()
                selected_country_name = st.selectbox(
                    "Valitse tarkasteltava maa",
                    country_options,
                    key="macro_import_country_detail_name",
                )

                selected_country_code = (
                    countries_df.loc[countries_df["name"] == selected_country_name, "code"]
                    .iloc[0]
                )

                country_total_df = load_import_country_detail(selected_country_code, months)
                country_prod_df = load_import_country_products(selected_country_code, months)

                latest_country, pct_country, latest_country_year = _latest_full_year_change(
                    country_total_df, "Aika_dt", "Tuonti_eur"
                )
                _metric_billions_pct_year(selected_country_name, latest_country, pct_country, latest_country_year)

                _line_export_chart(
                    country_total_df,
                    "Aika_dt",
                    "Tuonti_eur",
                    f"{selected_country_name} – tuonnin kehitys",
                )

                st.divider()
                st.markdown("#### 🔎 Tuoteryhmä kyseisestä maasta")

                if country_prod_df is None or country_prod_df.empty:
                    st.info("Maan tuoteryhmätuontia ei saatu ladattua.")
                else:
                    country_groups = sorted(country_prod_df["Tuoteryhmä"].dropna().unique().tolist())
                    selected_country_group = st.selectbox(
                        "Valitse tuoteryhmä",
                        country_groups,
                        key="macro_import_country_group_detail",
                    )

                    country_group_df = (
                        country_prod_df[country_prod_df["Tuoteryhmä"] == selected_country_group]
                        .copy()
                        .sort_values("Aika_dt")
                    )

                    latest_country_group, pct_country_group, latest_country_group_year = _latest_full_year_change(
                        country_group_df, "Aika_dt", "Tuonti_eur"
                    )
                    _metric_billions_pct_year(
                        f"{selected_country_group} ← {selected_country_name}",
                        latest_country_group,
                        pct_country_group,
                        latest_country_group_year,
                    )

                    _line_export_chart(
                        country_group_df,
                        "Aika_dt",
                        "Tuonti_eur",
                        f"{selected_country_group} – tuonnin kehitys maasta {selected_country_name}",
                    )


def _section_trade_balance(exports_total_df: pd.DataFrame, imports_total_df: pd.DataFrame, years: int):
    st.subheader("⚖️ Kauppatase")
    st.caption("Kauppatase = tavaravienti − tavaratuonti. Kuvaaja näyttää viennin ja tuonnin vuosikehityksen samalla kuvalla.")

    trade_df = _build_trade_balance(exports_total_df, imports_total_df)
    trade_df = _clip_by_years(trade_df, "Aika_dt", years)

    if trade_df is None or trade_df.empty:
        st.warning("Kauppatasedataa ei saatu.")
        return

    latest_balance, pct_balance, latest_balance_date = _pct_change_vs_year_ago(trade_df, "Aika_dt", "Kauppatase_eur")

    _kpi_card(
        "Kauppatase",
        _fmt_money(latest_balance),
        f"{pct_balance:+.1f} % (1 v)" if pct_balance is not None else None,
        f"Kuukausi: {latest_balance_date.date()}" if latest_balance_date is not None else None,
    )

    _line_trade_chart(exports_total_df, imports_total_df, "Kokonaisviennin ja -tuonnin kehitys")


def _section_debt(debt_pct: pd.DataFrame, debt_eur: pd.DataFrame, years: int):
    st.subheader("🏦 Julkinen velka")
    st.caption("Eurostat: bruttovelka (S13) kvartaaleittain sekä % BKT:stä että milj. euroina.")

    latest_pct = None
    latest_pct_date = None
    if debt_pct is not None and not debt_pct.empty:
        x = debt_pct.dropna(subset=["debt_pct_gdp"]).sort_values("Date")
        if not x.empty:
            latest_pct = float(x["debt_pct_gdp"].iloc[-1])
            latest_pct_date = pd.to_datetime(x["Date"].iloc[-1]).date()

    latest_eur = None
    latest_eur_date = None
    if debt_eur is not None and not debt_eur.empty:
        y = debt_eur.dropna(subset=["debt_mio_eur"]).sort_values("Date")
        if not y.empty:
            latest_eur = float(y["debt_mio_eur"].iloc[-1])
            latest_eur_date = pd.to_datetime(y["Date"].iloc[-1]).date()

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.metric("Velka / BKT", _fmt(latest_pct, 1, " %"))
        if latest_pct_date:
            st.caption(f"Kvartaali: {latest_pct_date}")
    with c2:
        st.metric("Velka (milj. €)", _fmt_millions(latest_eur * 1_000_000 if latest_eur is not None else None, 0))
        if latest_eur_date:
            st.caption(f"Kvartaali: {latest_eur_date}")

    st.divider()

    d1 = _clip_by_years(debt_pct, "Date", years) if debt_pct is not None else pd.DataFrame()
    d2 = _clip_by_years(debt_eur, "Date", years) if debt_eur is not None else pd.DataFrame()

    cc1, cc2 = st.columns(2, gap="large")
    with cc1:
        if d1 is None or d1.empty:
            st.info("Velka/BKT -sarjaa ei saatu.")
        else:
            fig = px.line(
                d1,
                x="Date",
                y="debt_pct_gdp",
                markers=True,
                labels={"Date": "Kvartaali", "debt_pct_gdp": "% BKT:stä"},
                title="Velka / BKT (%)",
            )
            fig.update_yaxes(ticksuffix=" %", zeroline=True)
            st.plotly_chart(fig, use_container_width=True, key="macro_debt_pct_line")
    with cc2:
        if d2 is None or d2.empty:
            st.info("Velan euromäärää ei saatu.")
        else:
            fig = px.line(
                d2,
                x="Date",
                y="debt_mio_eur",
                markers=True,
                labels={"Date": "Kvartaali", "debt_mio_eur": "milj. €"},
                title="Velka (milj. €)",
            )
            st.plotly_chart(fig, use_container_width=True, key="macro_debt_eur_line")


# ============================================================
# Public render
# ============================================================
def render() -> None:
    st.subheader("🇫🇮 Makrotalous (ajantasainen sijoittajan näkymä)")
    st.caption("Inflaatio (kk), BKT YoY (kv), työttömyys (kk), vienti (kk, tuoteryhmät & alueet), velka (kv).")

    with st.sidebar:
        st.markdown("### ⚙️ Aikajänne")
        years = st.slider("Näytettävä aikajänne (vuotta)", 2, 25, 8, key="macro_years")

    infl = load_inflation()
    gdp_yoy = load_gdp_yoy()
    unemp = load_unemployment()

    months = int(years * 12)

    exports_products_df, _ = fetch_exports_products(months=months, lang="fi")
    exports_total_df = _build_total_exports_from_products(exports_products_df)
    exp_last, exp_pct_y, exp_date = _pct_change_vs_year_ago(exports_total_df, "Aika_dt", "Vienti_eur")

    imports_products_df, _ = fetch_imports_products(months=months, lang="fi")
    imports_total_df = _build_total_imports_from_products(imports_products_df)
    imp_last, imp_pct_y, imp_date = _pct_change_vs_year_ago(imports_total_df, "Aika_dt", "Tuonti_eur")

    trade_balance_df = _build_trade_balance(exports_total_df, imports_total_df)
    bal_last, bal_pct_y, bal_date = _pct_change_vs_year_ago(trade_balance_df, "Aika_dt", "Kauppatase_eur")

    debt_pct = load_debt_pct_gdp()
    debt_eur = load_debt_mio_eur()

    st.markdown("### 📌 Kooste")
    k1, k2, k3, k4 = st.columns(4, gap="large")
    k5, k6, k7 = st.columns(3, gap="large")

    if infl is not None and not infl.empty:
        infl_vals = pd.to_numeric(infl["inflation_yoy"], errors="coerce").dropna()
        infl_last = float(infl_vals.iloc[-1]) if len(infl_vals) else None
        infl_delta_y = _yoy_delta(infl["inflation_yoy"], periods=12)
        infl_date = pd.to_datetime(infl["Date"].iloc[-1]).date() if infl["Date"].notna().any() else None
    else:
        infl_last = infl_delta_y = None
        infl_date = None

    with k1:
        _kpi_card(
            "Inflaatio (YoY, kk)",
            _fmt(infl_last, 2, " %"),
            f"{infl_delta_y:+.2f} %-yks. (1 v)" if infl_delta_y is not None else None,
            f"Kuukausi: {infl_date}" if infl_date else None,
        )

    if gdp_yoy is not None and not gdp_yoy.empty:
        g_vals = pd.to_numeric(gdp_yoy["gdp_yoy"], errors="coerce").dropna()
        g_last = float(g_vals.iloc[-1]) if len(g_vals) else None
        g_delta_y = _yoy_delta(gdp_yoy["gdp_yoy"], periods=4)
        g_date = pd.to_datetime(gdp_yoy["Date"].iloc[-1]).date() if gdp_yoy["Date"].notna().any() else None
    else:
        g_last = g_delta_y = None
        g_date = None

    with k2:
        _kpi_card(
            "BKT YoY",
            _fmt(g_last, 2, " %"),
            f"{g_delta_y:+.2f} %-yks. (1 v)" if g_delta_y is not None else None,
            f"Kvartaali: {g_date}" if g_date else None,
        )

    if unemp is not None and not unemp.empty and "unemployment_rate" in unemp.columns:
        u_vals = pd.to_numeric(unemp["unemployment_rate"], errors="coerce").dropna()
        u_last = float(u_vals.iloc[-1]) if len(u_vals) else None
        u_delta_y = _yoy_delta(unemp["unemployment_rate"], periods=12)
        u_date = pd.to_datetime(unemp["Date"].iloc[-1]).date() if unemp["Date"].notna().any() else None
    else:
        u_last = u_delta_y = None
        u_date = None

    with k3:
        _kpi_card(
            "Työttömyys (%)",
            _fmt(u_last, 2, " %"),
            f"{u_delta_y:+.2f} %-yks. (1 v)" if u_delta_y is not None else None,
            f"Kuukausi: {u_date}" if u_date else None,
        )

    latest_pct = None
    latest_eur = None
    d_date = None

    if debt_pct is not None and not debt_pct.empty:
        tmp = debt_pct.dropna(subset=["debt_pct_gdp"]).sort_values("Date")
        if not tmp.empty:
            latest_pct = float(tmp["debt_pct_gdp"].iloc[-1])
            d_date = pd.to_datetime(tmp["Date"].iloc[-1]).date()

    if debt_eur is not None and not debt_eur.empty:
        tmp2 = debt_eur.dropna(subset=["debt_mio_eur"]).sort_values("Date")
        if not tmp2.empty:
            latest_eur = float(tmp2["debt_mio_eur"].iloc[-1])
            d_date = d_date or pd.to_datetime(tmp2["Date"].iloc[-1]).date()

    with k4:
        cap = f"Kvartaali: {d_date}" if d_date else None
        if latest_pct is not None:
            cap = (cap + f" | Velka/BKT: {_fmt(latest_pct, 1, ' %')}") if cap else f"Velka/BKT: {_fmt(latest_pct, 1, ' %')}"
        _kpi_card(
            "Julkinen velka",
            _fmt_money(latest_eur * 1_000_000 if latest_eur is not None else None),
            None,
            cap,
    )

    with k5:
        _kpi_card(
            "Tavaravienti",
            _fmt_money(exp_last),
            f"{exp_pct_y:+.1f} % (1 v)" if exp_pct_y is not None else None,
            f"Kuukausi: {exp_date.date()}" if exp_date is not None else None,
    )

    with k6:
        _kpi_card(
            "Tavaratuonti",
            _fmt_money(imp_last),
            f"{imp_pct_y:+.1f} % (1 v)" if imp_pct_y is not None else None,
            f"Kuukausi: {imp_date.date()}" if imp_date is not None else None,
    )

    with k7:
        _kpi_card(
            "Kauppatase",
            _fmt_money(bal_last),
            f"{bal_pct_y:+.1f} % (1 v)" if bal_pct_y is not None else None,
            f"Kuukausi: {bal_date.date()}" if bal_date is not None else None,
    )

    st.divider()

    t1, t2, t3, t4, t5, t6, t7 = st.tabs(
    ["📈 Inflaatio (kk)", "🏛️ BKT YoY (kv)", "🧑‍💼 Työttömyys (kk)", "🚢 Vienti (kk)", "📥 Tuonti (kk)", "⚖️ Kauppatase", "🏦 Velka (kv)"]
)

    with t1:
        _section_inflation(infl, years)

    with t2:
        _section_gdp_yoy(gdp_yoy, years)

    with t3:
        _section_unemployment(unemp, years)

    with t4:
        _section_exports_uljas(months=months)

    with t5:
        _section_imports_uljas(months=months)

    with t6:
        _section_trade_balance(exports_total_df, imports_total_df, years)

    with t7:
        _section_debt(debt_pct, debt_eur, years)

   


   