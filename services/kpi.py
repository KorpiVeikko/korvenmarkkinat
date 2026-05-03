# services/kpi.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
import itertools

import pandas as pd
import requests

from services.compare import build_market_compare
from services.macro_uljas import fetch_exports_products, fetch_imports_products

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


CPI_YOY_URL = "https://pxdata.stat.fi/PXWeb/api/v1/fi/StatFin/khi/statfin_khi_pxt_122p.px"
LFS_135Z_URL = "https://pxdata.stat.fi/PXWeb/api/v1/fi/StatFin/tyti/statfin_tyti_pxt_135z.px"
EUROSTAT_API = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"


def _build_session() -> requests.Session:
    session = requests.Session()

    retry = Retry(
        total=2,
        connect=2,
        read=2,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
    )

    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    session.headers.update(
        {"User-Agent": "Mozilla/5.0 (compatible; TaloudenSeuranta/1.0)"}
    )

    return session

# -------------------------
# General helpers
# -------------------------
def _fmt(x: float, decimals: int = 2) -> str:
    if x is None or pd.isna(x):
        return "–"
    return f"{x:,.{decimals}f}".replace(",", " ")


def _fmt_money(x_eur: float | None) -> str:
    if x_eur is None or pd.isna(x_eur):
        return "–"

    x = float(x_eur)
    ax = abs(x)

    if ax >= 1_000_000_000:
        return f"{x / 1_000_000_000:,.1f}".replace(",", " ") + " mrd €"
    return f"{x / 1_000_000:,.0f}".replace(",", " ") + " milj. €"


def _safe_float(x) -> Optional[float]:
    try:
        if x is None or pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def _pct_change(now: float, then: float) -> Optional[float]:
    if then is None or now is None or then == 0:
        return None
    return (now / then - 1) * 100


def _norm(s: str) -> str:
    s = str(s).lower()
    return s.replace("ä", "a").replace("ö", "o").replace("å", "a")


def _build_item(name: str, value: str, delta: str = "", sub: str = "") -> Dict[str, Any]:
    return {
        "name": name,
        "value": value,
        "delta": delta,
        "sub": sub,
    }


# -------------------------
# JSON-stat / PxWeb helpers
# -------------------------
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
    session = _build_session()
    r = session.post(url, json=query, timeout=(10, timeout))
    r.raise_for_status()
    return _parse_jsonstat2(r.json())


def _get_px_meta(url: str, timeout: int = 45) -> dict:
    session = _build_session()
    r = session.get(url, timeout=(10, timeout))
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
            out["Vuosi_num"].astype("Int64").astype(str)
            + "-"
            + out["Kuukausi_num"].astype("Int64").astype(str).str.zfill(2)
            + "-01",
            errors="coerce",
        )
    elif q["y"].notna().any():
        out["Vuosi_num"] = pd.to_numeric(q["y"], errors="coerce")
        qn = pd.to_numeric(q["q"], errors="coerce")
        start_month = (qn - 1) * 3 + 1
        out["Aika_dt"] = pd.to_datetime(
            out["Vuosi_num"].astype("Int64").astype(str)
            + "-"
            + start_month.astype("Int64").astype(str).str.zfill(2)
            + "-01",
            errors="coerce",
        )
    elif y_only["y"].notna().any():
        out["Vuosi_num"] = pd.to_numeric(y_only["y"], errors="coerce")
        out["Aika_dt"] = pd.to_datetime(
            out["Vuosi_num"].astype("Int64").astype(str) + "-01-01",
            errors="coerce",
        )
    else:
        out["Aika_dt"] = pd.to_datetime(s, errors="coerce")

    return out


# -------------------------
# Current macro fetchers
# -------------------------
def fetch_inflation_now() -> tuple[Optional[float], Optional[float], str]:
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

    df = _add_time_columns(_post_px(CPI_YOY_URL, query))
    if df.empty:
        return None, None, ""

    f = df.copy()
    f["Arvo"] = pd.to_numeric(f["Arvo"], errors="coerce")
    f = f.dropna(subset=["Aika_dt", "Arvo"]).sort_values("Aika_dt")
    if f.empty:
        return None, None, ""

    latest_val = float(f["Arvo"].iloc[-1])
    latest_date = pd.to_datetime(f["Aika_dt"].iloc[-1])

    prev = f[f["Aika_dt"] == latest_date - pd.DateOffset(years=1)]
    delta = None
    if not prev.empty:
        delta = latest_val - float(prev["Arvo"].iloc[-1])

    return latest_val, delta, f"Kuukausi {latest_date.date()}"


def fetch_unemployment_now() -> tuple[Optional[float], Optional[float], str]:
    meta = _get_px_meta(LFS_135Z_URL)
    vars_ = meta.get("variables") or []
    if not vars_:
        return None, None, ""

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
            chosen = []
            if values and texts and len(values) == len(texts):
                for val, txt in zip(values, texts):
                    t = _norm(txt)
                    if "tyottomyysaste" in t or "unemployment rate" in t:
                        chosen = [val]
                        break
            query_parts.append({"code": code, "selection": {"filter": "item", "values": chosen or ["*"]}})
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

    df = _add_time_columns(
        _post_px(LFS_135Z_URL, {"query": query_parts, "response": {"format": "json-stat2"}})
    )
    if df.empty:
        return None, None, ""

    f = df.copy()
    f["Arvo"] = pd.to_numeric(f["Arvo"], errors="coerce")
    f = f.dropna(subset=["Aika_dt", "Arvo"]).sort_values("Aika_dt")
    if f.empty:
        return None, None, ""

    latest_val = float(f["Arvo"].iloc[-1])
    latest_date = pd.to_datetime(f["Aika_dt"].iloc[-1])

    prev = f[f["Aika_dt"] == latest_date - pd.DateOffset(years=1)]
    delta = None
    if not prev.empty:
        delta = latest_val - float(prev["Arvo"].iloc[-1])

    return latest_val, delta, f"Kuukausi {latest_date.date()}"


def fetch_public_debt_now() -> tuple[Optional[float], Optional[float], str]:
    url = f"{EUROSTAT_API}/gov_10q_ggdebt"
    params = {
        "lang": "EN",
        "format": "JSON",
        "freq": "Q",
        "sector": "S13",
        "na_item": "GD",
        "unit": "MIO_EUR",
        "geo": "FI",
    }

    session = _build_session()
    r = session.get(url, params=params, timeout=(10, 45))
    r.raise_for_status()
    j = r.json()

    if "value" not in j or "dimension" not in j:
        return None, None, ""

    dim = j["dimension"]
    time_cat = dim.get("time", {}).get("category", {})
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
        out.append({"Quarter": tk, "Arvo": pd.to_numeric(v, errors="coerce")})

    df = pd.DataFrame(out).dropna(subset=["Arvo"])
    if df.empty:
        return None, None, ""

    df["Date"] = pd.PeriodIndex(df["Quarter"], freq="Q").to_timestamp(how="start")
    df = df.sort_values("Date")

    latest_val_mio = float(df["Arvo"].iloc[-1])
    latest_date = pd.to_datetime(df["Date"].iloc[-1])

    prev = df[df["Date"] == latest_date - pd.DateOffset(years=1)]
    pct = None
    if not prev.empty:
        prev_val = float(prev["Arvo"].iloc[-1])
        if prev_val != 0:
            pct = _pct_change(latest_val_mio, prev_val)

    return latest_val_mio * 1_000_000, pct, f"Kvartaali {latest_date.date()}"


def fetch_trade_balance_now(months: int = 24) -> tuple[Optional[float], Optional[float], str]:
    exp_df, _ = fetch_exports_products(months=months, lang="fi")
    imp_df, _ = fetch_imports_products(months=months, lang="fi")

    if exp_df is None or exp_df.empty or imp_df is None or imp_df.empty:
        return None, None, ""

    exp_total = exp_df.groupby("Aika_dt", as_index=False)["Vienti_eur"].sum().sort_values("Aika_dt")
    imp_total = imp_df.groupby("Aika_dt", as_index=False)["Tuonti_eur"].sum().sort_values("Aika_dt")

    df = pd.merge(exp_total, imp_total, on="Aika_dt", how="outer").sort_values("Aika_dt")
    df["Vienti_eur"] = pd.to_numeric(df["Vienti_eur"], errors="coerce").fillna(0)
    df["Tuonti_eur"] = pd.to_numeric(df["Tuonti_eur"], errors="coerce").fillna(0)
    df["Kauppatase_eur"] = df["Vienti_eur"] - df["Tuonti_eur"]
    df = df.dropna(subset=["Aika_dt", "Kauppatase_eur"])

    if df.empty:
        return None, None, ""

    latest_val = float(df.iloc[-1]["Kauppatase_eur"])
    latest_date = pd.to_datetime(df.iloc[-1]["Aika_dt"])

    prev = df[df["Aika_dt"] == latest_date - pd.DateOffset(years=1)]
    pct = None
    if not prev.empty:
        prev_val = float(prev.iloc[-1]["Kauppatase_eur"])
        if prev_val != 0:
            pct = _pct_change(latest_val, prev_val)

    return latest_val, pct, f"Kuukausi {latest_date.date()}"


def _build_market_item_from_snap(name: str, snap: dict, decimals: int = 2) -> Dict[str, Any]:
    latest = snap.get("now")
    pct1m = snap.get("m1")

    if latest is None or pd.isna(latest):
        return _build_item(name, "–", "Data puuttuu", "")

    return _build_item(
        name=name,
        value=_fmt(float(latest), decimals) + " €",
        delta=f"{pct1m:+.1f} % (1 kk)" if pct1m is not None and not pd.isna(pct1m) else "",
        sub="Muutos vs 1 kk sitten",
    )

# -------------------------
# Public builder
# -------------------------


def build_kpi_items() -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []

    macro_fetchers = [
        (
            "Julkinen velka",
            lambda: fetch_public_debt_now(),
            lambda latest_val, delta, sub: _build_item(
                "Julkinen velka",
                _fmt_money(latest_val),
                f"{delta:+.1f}% (1 v)" if delta is not None else "",
                sub,
            ),
        ),
        (
            "Inflaatio",
            lambda: fetch_inflation_now(),
            lambda latest_val, delta, sub: _build_item(
                "Inflaatio",
                f"{_fmt(latest_val, 2)} %",
                f"{delta:+.2f} %-yks (1 v)" if delta is not None else "",
                sub,
            ),
        ),
        (
            "Työttömyys",
            lambda: fetch_unemployment_now(),
            lambda latest_val, delta, sub: _build_item(
                "Työttömyys",
                f"{_fmt(latest_val, 2)} %",
                f"{delta:+.2f} %-yks (1 v)" if delta is not None else "",
                sub,
            ),
        ),
    ]

    for name, fetcher, builder in macro_fetchers:
        try:
            latest_val, delta, sub = fetcher()
            items.append(builder(latest_val, delta, sub))
        except Exception:
            items.append(_build_item(name, "–", "Data puuttuu", ""))

    try:
        snaps = build_market_compare(period="5y")
        items.append(_build_market_item_from_snap("Kulta", snaps.get("Kulta", {}), 0))
        items.append(_build_market_item_from_snap("Hopea", snaps.get("Hopea", {}), 2))
        items.append(_build_market_item_from_snap("Bitcoin", snaps.get("Bitcoin", {}), 0))
    except Exception:
        items.append(_build_item("Kulta", "–", "Data puuttuu", ""))
        items.append(_build_item("Hopea", "–", "Data puuttuu", ""))
        items.append(_build_item("Bitcoin", "–", "Data puuttuu", ""))

    return items


