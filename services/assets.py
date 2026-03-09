# services/assets.py
from __future__ import annotations

import math
from dataclasses import dataclass
import pandas as pd
import streamlit as st


# -----------------------------
# Datamalli
# -----------------------------
@dataclass
class AssetSpec:
    name: str
    ticker: str
    unit: str = "USD"
    decimals: int = 0
    fx_pair: str = "EURUSD=X"
    show_eur: bool = True
    icon: str = ""


# -----------------------------
# Apurit
# -----------------------------
def _safe_float(x) -> float | None:
    try:
        if x is None:
            return None
        if isinstance(x, float) and math.isnan(x):
            return None
        return float(x)
    except Exception:
        return None


def _pct_change(now: float | None, then: float | None) -> float | None:
    if now is None or then is None:
        return None
    if then == 0:
        return None
    return (now / then - 1) * 100.0


def _fmt_num(x: float | None, decimals: int = 0) -> str:
    if x is None:
        return "–"
    return f"{x:,.{decimals}f}".replace(",", " ")


def _delta_str(pct: float | None) -> str | None:
    if pct is None:
        return None
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f} %"


def _fx_usd_per_eur(fetch_price_history) -> float | None:
    try:
        fx = fetch_price_history("EURUSD=X", period="6mo")
        if fx is None or fx.empty or "Close" not in fx.columns:
            return None
        fx = fx.dropna(subset=["Close"])
        if fx.empty:
            return None
        return float(fx.iloc[-1]["Close"])
    except Exception:
        return None


def _compute_now_m1_y1(fetch_price_history, ticker: str, period: str = "4y") -> dict:
    try:
        df = fetch_price_history(ticker, period=period)
    except Exception as e:
        st.error(f"Datan haku epäonnistui: {e}")
        return {"df": pd.DataFrame(), "now": None, "m1": None, "y1": None, "p_m1": None, "p_y1": None}

    if df is None or df.empty or "Close" not in df.columns:
        return {"df": pd.DataFrame(), "now": None, "m1": None, "y1": None, "p_m1": None, "p_y1": None}

    df = df.dropna(subset=["Close"]).copy()
    if df.empty:
        return {"df": pd.DataFrame(), "now": None, "m1": None, "y1": None, "p_m1": None, "p_y1": None}

    now = _safe_float(df.iloc[-1]["Close"])
    m1 = _safe_float(df.iloc[-21]["Close"]) if len(df) > 21 else None
    y1 = _safe_float(df.iloc[-252]["Close"]) if len(df) > 252 else None

    return {
        "df": df,
        "now": now,
        "m1": m1,
        "y1": y1,
        "p_m1": _pct_change(now, m1),
        "p_y1": _pct_change(now, y1),
    }


# -----------------------------
# UI-renderöinti
# -----------------------------
def render_asset_panel(fetch_price_history, spec: AssetSpec):
    """
    Palauttaa AINA (df, usd_per_eur)
    """

    snap = _compute_now_m1_y1(fetch_price_history, spec.ticker, period="4y")
    df = snap["df"]

    # ✅ TÄRKEÄ KORJAUS
    if df is None or df.empty:
        st.warning("Markkinadataa ei saatu.")
        return pd.DataFrame(), None

    usd_per_eur = _fx_usd_per_eur(fetch_price_history) if spec.show_eur else None

    def usd_to_eur(usd: float | None) -> float | None:
        if usd is None or usd_per_eur is None or usd_per_eur == 0:
            return None
        return usd / usd_per_eur

    # ---------- Hintatasot ----------
    #st.markdown("### 📊 Hintatasot ja muutokset")
    st.markdown(f"#### {spec.icon} {spec.name}".strip())

    c_now, c_m1, c_y1 = st.columns(3)

    with c_now:
        st.metric("Nyt", f"{_fmt_num(snap['now'], spec.decimals)} {spec.unit}")
        if spec.show_eur:
            eur = usd_to_eur(snap["now"])
            if eur is not None:
                st.caption(f"≈ {_fmt_num(eur, spec.decimals)} €")

    with c_m1:
        st.metric("1 kk", f"{_fmt_num(snap['m1'], spec.decimals)} {spec.unit}")
        ds = _delta_str(snap["p_m1"])
        if ds:
            st.metric("", "", ds)

    with c_y1:
        st.metric("1 v", f"{_fmt_num(snap['y1'], spec.decimals)} {spec.unit}")
        ds = _delta_str(snap["p_y1"])
        if ds:
            st.metric("", "", ds)

    st.divider()
    # st.markdown("### 📈 Hintakehitys")

    return df, usd_per_eur
