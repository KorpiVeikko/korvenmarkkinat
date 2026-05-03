# services/energy_price_helper.py
from __future__ import annotations

import pandas as pd

from services.energy_view_helper import clip_years, latest_val, pct_change


COMPONENT_NAME_MAP = {
    "Sähköenergia(veroton)": "Energia",
    "Sähköenergia (veroton)": "Energia",
    "Electric energy (excluding tax)": "Energia",

    "Verkkopalvelumaksu(veroton)": "Siirto",
    "Verkkopalvelumaksu (veroton)": "Siirto",
    "Distribution price (excluding tax)": "Siirto",

    "Verot (sähkövero ja alv.)": "Verot",
    "Verot (sähkövero ja alv.) ": "Verot",
    "Taxes (electricity tax and VAT)": "Verot",

    "Kokonaishinta": "Kokonaishinta",
    "Total price": "Kokonaishinta",
}

ALLOWED_COMPONENTS = ["Energia", "Siirto", "Verot", "Kokonaishinta"]


def find_consumer_col(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        c = str(col).strip().lower()
        if c in ("sähkön kuluttajatyyppi", "type of consumer"):
            return col
    return None


def find_component_col(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        c = str(col).strip().lower()
        if c in ("hintakomponentti", "price component"):
            return col
    return None


def find_measure_col(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        c = str(col).strip().lower()
        if c in ("tiedot", "tieto", "information", "data"):
            return col
    return None


def get_consumer_options(df: pd.DataFrame, consumer_col: str) -> list[str]:
    if df is None or df.empty or consumer_col not in df.columns:
        return []
    return sorted(df[consumer_col].dropna().astype(str).unique().tolist())


def _normalize_component_name(value: str) -> str:
    key = str(value).strip()
    return COMPONENT_NAME_MAP.get(key, key)


def _filter_price_level_only(df: pd.DataFrame, measure_col: str | None) -> pd.DataFrame:
    """
    Pidetään vain varsinainen hintataso, ei vuosimuutossarjaa (%).
    """
    if df is None or df.empty:
        return pd.DataFrame()

    if not measure_col or measure_col not in df.columns:
        return df.copy()

    s = df[measure_col].astype(str).str.strip().str.lower()

    is_price_level = (
        s.eq("hinta (snt/kwh)")
        | s.eq("price (c/kwh)")
        | (
            s.str.contains("hinta", na=False)
            & ~s.str.contains("muutos", na=False)
            & ~s.str.contains("%", na=False)
        )
    )

    out = df[is_price_level].copy()
    return out


def build_component_timeseries(
    df: pd.DataFrame,
    *,
    consumer_col: str,
    component_col: str,
    consumer_value: str,
    years_back: int,
    measure_col: str | None = None,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    f = df[df[consumer_col].astype(str) == str(consumer_value)].copy()
    f = _filter_price_level_only(f, measure_col)
    f = clip_years(f, years_back)
    f = f.dropna(subset=["Arvo", "Aika_dt"]).copy()

    f["_Komponentti"] = f[component_col].astype(str).map(_normalize_component_name)
    f = f[f["_Komponentti"].isin(ALLOWED_COMPONENTS)].copy()

    if f.empty:
        return pd.DataFrame()

    # Jos samalle kuukaudelle/komponentille on useampi rivi, otetaan keskiarvo
    f = (
        f.groupby(["Aika_dt", "_Komponentti"], as_index=False)["Arvo"]
        .mean()
        .sort_values(["Aika_dt", "_Komponentti"])
    )

    return f


def split_price_components(comp_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if comp_df is None or comp_df.empty:
        return {name: pd.DataFrame() for name in ALLOWED_COMPONENTS}

    return {
        "Energia": comp_df[comp_df["_Komponentti"] == "Energia"].copy(),
        "Siirto": comp_df[comp_df["_Komponentti"] == "Siirto"].copy(),
        "Verot": comp_df[comp_df["_Komponentti"] == "Verot"].copy(),
        "Kokonaishinta": comp_df[comp_df["_Komponentti"] == "Kokonaishinta"].copy(),
    }


def build_price_summary(comp_df: pd.DataFrame) -> dict:
    parts = split_price_components(comp_df)

    total_df = parts["Kokonaishinta"]
    energy_df = parts["Energia"]
    transfer_df = parts["Siirto"]
    tax_df = parts["Verot"]

    latest_time = total_df["Aika_dt"].iloc[-1] if not total_df.empty else None

    return {
        "latest_time": latest_time,
        "latest_total": latest_val(total_df),
        "latest_energy": latest_val(energy_df),
        "latest_transfer": latest_val(transfer_df),
        "latest_tax": latest_val(tax_df),
        "delta_total_1y": pct_change(total_df, 12),
        "delta_energy_1y": pct_change(energy_df, 12),
        "delta_transfer_1y": pct_change(transfer_df, 12),
        "delta_tax_1y": pct_change(tax_df, 12),
        "parts": parts,
    }