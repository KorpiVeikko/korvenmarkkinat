# services/kpi.py
from __future__ import annotations

from typing import List, Dict, Any, Optional
import pandas as pd

from services.finance import fetch_government_debt
from services.market_data import fetch_price_history
from services.worldbank import fetch_indicator, latest_and_change


def _fmt(x: float, decimals: int = 2) -> str:
    if x is None or pd.isna(x):
        return "–"
    return f"{x:,.{decimals}f}".replace(",", " ")


def _fmt_milj_eur(x_eur: float) -> str:
    if x_eur is None or pd.isna(x_eur):
        return "–"
    return f"{x_eur/1_000_000:,.0f}".replace(",", " ") + " milj. €"


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


def _latest_close(symbol: str, period: str = "6mo") -> tuple[Optional[float], Optional[float]]:
    """
    Palauttaa (latest_close, pct_change_1m).
    1 kk ~ 21 kaupankäyntipäivää.
    """
    df = fetch_price_history(symbol, period=period)
    if df is None or df.empty or "Close" not in df.columns:
        return None, None

    df = df.dropna(subset=["Close"]).copy()
    if df.empty:
        return None, None

    latest = _safe_float(df.iloc[-1]["Close"])
    if latest is None:
        return None, None

    if len(df) > 21:
        prev = _safe_float(df.iloc[-21]["Close"])
        pct = _pct_change(latest, prev)
    else:
        pct = None

    return latest, pct


def build_kpi_items() -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []

    # -------------------------
    # Valtion velka (vuosi)
    # -------------------------
    try:
        df = fetch_government_debt()
        df["pricingDate"] = pd.to_datetime(df["pricingDate"], errors="coerce")
        df = df.dropna(subset=["pricingDate", "nominalAmount"])

        yearly = df.groupby(df["pricingDate"].dt.year)["nominalAmount"].sum().sort_index()
        latest_year = int(yearly.index.max())
        latest_val = _safe_float(yearly.loc[latest_year])

        prev_year = latest_year - 1 if (latest_year - 1) in yearly.index else None
        prev_val = _safe_float(yearly.loc[prev_year]) if prev_year is not None else None

        value = _fmt_milj_eur(latest_val)

        delta = ""
        sub = f"Vuosi {latest_year}"
        if prev_val is not None and latest_val is not None:
            diff = latest_val - prev_val
            pct = _pct_change(latest_val, prev_val)
            # selite deltaan mukaan:
            if pct is not None:
                delta = f"{diff/1_000_000:+,.0f}".replace(",", " ") + f" milj. € ({pct:+.1f}%) (1 v)"
            else:
                delta = f"{diff/1_000_000:+,.0f}".replace(",", " ") + " milj. € (1 v)"
            sub = f"Vuosi {latest_year} · Muutos vs {prev_year}"

        items.append({"name": "Valtion velka", "value": value, "delta": delta, "sub": sub})

    except Exception:
        items.append({"name": "Valtion velka", "value": "–", "delta": "Data puuttuu", "sub": ""})

    # -------------------------
    # World Bank (vuosi)
    # -------------------------
    def wb_item(name: str, code: str, unit: str, delta_unit: str):
        try:
            df = fetch_indicator("FIN", code)
            if df is None or df.empty:
                return {"name": name, "value": "–", "delta": "Data puuttuu", "sub": ""}

            latest_val, change, latest_year = latest_and_change(df)
            latest_val = _safe_float(latest_val)
            change = _safe_float(change)

            value = f"{_fmt(latest_val, 2)}{unit}"
            sub = f"Vuosi {latest_year}"

            delta = ""
            if change is not None and latest_year is not None:
                delta = f"{change:+.2f}{delta_unit} (1 v)"
                sub = f"Vuosi {latest_year} · Muutos vs {latest_year - 1}"

            return {"name": name, "value": value, "delta": delta, "sub": sub}
        except Exception:
            return {"name": name, "value": "–", "delta": "Data puuttuu", "sub": ""}

    items.append(wb_item("Inflaatio", "FP.CPI.TOTL.ZG", " %", " %-yks"))
    items.append(wb_item("Vienti (% BKT)", "NE.EXP.GNFS.ZS", " %", " %-yks"))
    items.append(wb_item("Työttömyys", "SL.UEM.TOTL.ZS", " %", " %-yks"))

    # -------------------------
    # Markkinat (1 kk)
    # -------------------------
    def mkt_item(name: str, symbol: str, decimals: int = 2):
        latest, pct1m = _latest_close(symbol, period="6mo")
        if latest is None:
            return {"name": name, "value": "–", "delta": "Data puuttuu", "sub": ""}

        value = _fmt(latest, decimals)
        delta = ""
        sub = "Muutos vs 1 kk sitten"
        if pct1m is not None:
            delta = f"{pct1m:+.1f} % (1 kk)"

        return {"name": name, "value": value, "delta": delta, "sub": sub}

    items.append(mkt_item("Kulta", "GC=F", 0))
    items.append(mkt_item("Hopea", "SI=F", 2))
    items.append(mkt_item("Bitcoin", "BTC-USD", 0))

    return items


