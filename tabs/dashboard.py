# tabs/dashboard.py
from __future__ import annotations

import pandas as pd
import streamlit as st

from services.compare import build_market_compare, correlation_matrix

@st.cache_data(ttl=60 * 30, show_spinner=False)
def load_dashboard_market_data(period: str = "5y") -> dict:
    return build_market_compare(period=period)


HERO_IMAGE_PATH = "tabs/assets/dashboard_hero.png"
CORR_KEEP = ["Suomi (OMXH25)", "Bitcoin", "Kulta", "Hopea", "S&P 500", "Nasdaq"]


def _fmt(x, decimals: int = 0, suffix: str = "") -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "–"
    return f"{x:,.{decimals}f}{suffix}".replace(",", " ")


def _fmt_now(name: str, value: float | None, snaps: dict) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "–"

    currency = snaps.get(name, {}).get("currency")
    if currency == "EUR":
        decimals = 0 if name == "Bitcoin" else 2
        return _fmt(value, decimals, " €")

    return _fmt(value, 2)


def _badge_delta(pct: float | None) -> str:
    if pct is None or (isinstance(pct, float) and pd.isna(pct)):
        return "–"
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%"


def _build_market_table(snaps: dict) -> pd.DataFrame:
    rows = []
    for name, snap in snaps.items():
        rows.append(
            {
                "Nimi": name,
                "Nyt": snap.get("now"),
                "1 kk": snap.get("m1"),
                "1 v": snap.get("y1"),
            }
        )

    return pd.DataFrame(rows).dropna(subset=["Nimi"])


def _top_bottom_summary(df: pd.DataFrame) -> tuple[pd.Series | None, pd.Series | None]:
    if df is None or df.empty:
        return None, None

    ranked = df.dropna(subset=["1 kk"]).sort_values("1 kk", ascending=False)
    if ranked.empty:
        return None, None

    top = ranked.iloc[0] if len(ranked) >= 1 else None
    bottom = ranked.iloc[-1] if len(ranked) >= 2 else None
    return top, bottom


def _corr_summary(corr: pd.DataFrame) -> str:
    if corr is None or corr.empty:
        return "Korrelaatiota ei voitu laskea (liian vähän yhteistä dataa)."

    c = corr.copy()
    for col in c.columns:
        c.loc[col, col] = pd.NA

    stacked = c.stack(dropna=True)
    if stacked.empty:
        return "Korrelaatiosta ei saatu tiivistettävää (puuttuvia pareja)."

    (a, b), val = max(stacked.items(), key=lambda kv: abs(kv[1]))

    parts = [f"Vahvin yhteisliike: **{a} ↔ {b}** (corr {val:+.2f})."]

    if "S&P 500" in corr.columns and "Nasdaq" in corr.columns:
        sp_nq = corr.loc["S&P 500", "Nasdaq"]
        if pd.notna(sp_nq):
            if sp_nq > 0.85:
                parts.append("Osakkeet liikkuvat hyvin yhdessä (S&P 500 ↔ Nasdaq korkea).")
            elif sp_nq < 0.5:
                parts.append("Osakkeiden sisälläkin hajontaa (S&P 500 ↔ Nasdaq maltillinen).")

    if "Bitcoin" in corr.columns and "S&P 500" in corr.columns:
        btc_sp = corr.loc["Bitcoin", "S&P 500"]
        if pd.notna(btc_sp):
            if btc_sp > 0.4:
                parts.append("BTC on viime aikoina käyttäytynyt osakeriskin suuntaan (positiivinen yhteys S&P:hen).")
            elif btc_sp < 0.1:
                parts.append("BTC on ollut melko irti osakkeista (heikko yhteys S&P:hen).")

    if "Kulta" in corr.columns and "S&P 500" in corr.columns:
        gold_sp = corr.loc["Kulta", "S&P 500"]
        if pd.notna(gold_sp):
            if gold_sp < 0.0:
                parts.append("Kulta on toiminut selkeämmin hajauttavana (negatiivinen yhteys osakkeisiin).")
            elif gold_sp < 0.2:
                parts.append("Kullan ja osakkeiden yhteys on heikko → hajautushyötyä.")
            else:
                parts.append("Kullan ja osakkeiden yhteys on ollut yllättävänkin positiivinen viime jaksolla.")

    return " ".join(parts)


def _render_header():
    st.markdown("## Tervetuloa seuraamaan korvenmarkkinoita")
    st.divider()

    try:
        st.image(HERO_IMAGE_PATH, use_container_width=True)
    except Exception:
        st.info(f"Hero-kuvaa ei löytynyt polusta: {HERO_IMAGE_PATH}")

    st.divider()


def _render_market_signal(df: pd.DataFrame, corr: pd.DataFrame):
    st.subheader("📰 Markkinasähke (vertailu)")

    top, bottom = _top_bottom_summary(df)
    corr_text = _corr_summary(corr)

    c1, c2, c3 = st.columns([1.3, 1.3, 2])

    with c1:
        if top is not None:
            st.markdown("**📈 Kuukauden vahvin**")
            st.write(f"{top['Nimi']} ({_badge_delta(top['1 kk'])})")

    with c2:
        if bottom is not None:
            st.markdown("**📉 Kuukauden heikoin**")
            st.write(f"{bottom['Nimi']} ({_badge_delta(bottom['1 kk'])})")

    with c3:
        st.markdown("**🧠 Tulkinta (korr + fiilis)**")
        st.write(corr_text)

    st.divider()


def _render_market_table(df: pd.DataFrame, snaps: dict):
    st.subheader("📊 Vertailu (muutos %)")

    df_view = df.copy()
    df_view["Nyt"] = df_view.apply(lambda r: _fmt_now(r["Nimi"], r["Nyt"], snaps), axis=1)
    df_view["1 kk"] = df_view["1 kk"].apply(_badge_delta)
    df_view["1 v"] = df_view["1 v"].apply(_badge_delta)

    st.dataframe(
        df_view[["Nimi", "Nyt", "1 kk", "1 v"]],
        use_container_width=True,
        hide_index=True,
    )

    st.divider()


def _render_correlation(corr: pd.DataFrame):
    st.subheader("🔗 Korrelaatio (päivätuotot, ~1 v)")

    if corr is None or corr.empty:
        st.caption("Korrelaatiota ei voitu laskea (liian vähän yhteistä dataa).")
    else:
        keep = [c for c in CORR_KEEP if c in corr.columns]
        corr_small = corr.loc[keep, keep]
        st.dataframe(corr_small.round(2), use_container_width=True)

    st.caption("Markkinadata: Yahoo Finance / EUR-muunnos EURUSD-kurssilla samoilla periaatteilla kuin muissa tabeissa.")


def render():
    _render_header()

    snaps = load_dashboard_market_data(period="5y")
    df = _build_market_table(snaps)

    if df.empty:
        st.warning("Markkinadataa ei saatu.")
        return

    corr = correlation_matrix(snaps, days=252)

    _render_market_signal(df, corr)
    _render_market_table(df, snaps)
    _render_correlation(corr)



