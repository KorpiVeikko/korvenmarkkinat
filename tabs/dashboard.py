# tabs/dashboard.py
import streamlit as st
import pandas as pd

from services.compare import build_market_compare, correlation_matrix

HERO_IMAGE_PATH = "tabs/assets/dashboard_hero.png"


def _fmt(x, decimals=0, suffix=""):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "–"
    return f"{x:,.{decimals}f}{suffix}".replace(",", " ")


def _badge_delta(pct: float | None) -> str:
    if pct is None or (isinstance(pct, float) and pd.isna(pct)):
        return "–"
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%"


def _corr_summary(corr: pd.DataFrame) -> str:
    """
    Tiivis tulkinta korrelaatiomatriisista.
    """
    if corr is None or corr.empty:
        return "Korrelaatiota ei voitu laskea (liian vähän yhteistä dataa)."

    # Poistetaan diagonaali ja etsitään vahvin pari
    c = corr.copy()
    for col in c.columns:
        c.loc[col, col] = pd.NA

    # Etsi maksimi absoluuttinen korrelaatio
    stacked = c.stack(dropna=True)
    if stacked.empty:
        return "Korrelaatiosta ei saatu tiivistettävää (puuttuvia pareja)."

    # Vahvin absoluuttinen
    (a, b), val = max(stacked.items(), key=lambda kv: abs(kv[1]))

    parts = []
    parts.append(f"Vahvin yhteisliike: **{a} ↔ {b}** (corr {val:+.2f}).")

    # Osakekeskittyneisyys (S&P vs Nasdaq)
    if "S&P 500" in corr.columns and "Nasdaq" in corr.columns:
        sp_nq = corr.loc["S&P 500", "Nasdaq"]
        if pd.notna(sp_nq):
            if sp_nq > 0.85:
                parts.append("Osakkeet liikkuvat hyvin yhdessä (S&P 500 ↔ Nasdaq korkea).")
            elif sp_nq < 0.5:
                parts.append("Osakkeiden sisälläkin hajontaa (S&P 500 ↔ Nasdaq maltillinen).")

    # BTC vs osakkeet / kulta
    if "Bitcoin" in corr.columns and "S&P 500" in corr.columns:
        v = corr.loc["Bitcoin", "S&P 500"]
        if pd.notna(v):
            if v > 0.4:
                parts.append("BTC on viime aikoina käyttäytynyt osakeriskin suuntaan (positiivinen yhteys S&P:hen).")
            elif v < 0.1:
                parts.append("BTC on ollut melko irti osakkeista (heikko yhteys S&P:hen).")

    if "Kulta" in corr.columns and "S&P 500" in corr.columns:
        v = corr.loc["Kulta", "S&P 500"]
        if pd.notna(v):
            if v < 0.0:
                parts.append("Kulta on toiminut selkeämmin hajauttavana (negatiivinen yhteys osakkeisiin).")
            elif v < 0.2:
                parts.append("Kullan ja osakkeiden yhteys on heikko → hajautushyötyä.")
            else:
                parts.append("Kullan ja osakkeiden yhteys on ollut yllättävänkin positiivinen viime jaksolla.")

    return " ".join(parts)


def render():
    st.markdown("## 📊 Suomen talouden seuranta")
    st.caption("Makrotalous, kiinteistöt ja markkinat – kooste")

    # Hero-kuva
    try:
        st.image(HERO_IMAGE_PATH, use_container_width=True)
    except Exception:
        st.info(f"Hero-kuvaa ei löytynyt polusta: {HERO_IMAGE_PATH}")

    st.divider()

    # ---------- MARKKINASÄHKE ----------
    st.subheader("📰 Markkinasähke (vertailu)")

    snaps = build_market_compare()

    rows = []
    for name, s in snaps.items():
        rows.append({"Nimi": name, "Nyt": s.get("now"), "1 kk": s.get("m1"), "1 v": s.get("y1")})

    df = pd.DataFrame(rows).dropna(subset=["Nimi"])
    if df.empty:
        st.warning("Markkinadataa ei saatu (Yahoo Finance).")
        return

    # Topit 1 kk muutoksella
    df_rank = df.dropna(subset=["1 kk"]).sort_values("1 kk", ascending=False)
    top = df_rank.iloc[0] if len(df_rank) > 0 else None
    bot = df_rank.iloc[-1] if len(df_rank) > 1 else None

    # Korrelaatio ja tulkinta
    corr = correlation_matrix(snaps, days=252)
    corr_text = _corr_summary(corr)

    c1, c2, c3 = st.columns([1.3, 1.3, 2])

    with c1:
        if top is not None:
            st.markdown("**📈 Kuukauden vahvin**")
            st.write(f"{top['Nimi']} ({_badge_delta(top['1 kk'])})")
    with c2:
        if bot is not None:
            st.markdown("**📉 Kuukauden heikoin**")
            st.write(f"{bot['Nimi']} ({_badge_delta(bot['1 kk'])})")
    with c3:
        st.markdown("**🧠 Tulkinta (korr + fiilis)**")
        st.write(corr_text)

    st.divider()

    # ---------- VERTAILUTAULUKKO ----------
    st.subheader("📊 Vertailu (muutos %)")

    df_view = df.copy()
    df_view["Nyt"] = df_view["Nyt"].apply(lambda x: _fmt(x, 2))
    df_view["1 kk"] = df_view["1 kk"].apply(_badge_delta)
    df_view["1 v"] = df_view["1 v"].apply(_badge_delta)

    st.dataframe(
        df_view[["Nimi", "Nyt", "1 kk", "1 v"]],
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # ---------- KORRELAATIO ----------
    st.subheader("🔗 Korrelaatio (päivätuotot, ~1 v)")
    if corr.empty:
        st.caption("Korrelaatiota ei voitu laskea (liian vähän yhteistä dataa).")
    else:
        keep = [c for c in ["Suomi (OMXH25)", "Bitcoin", "Kulta", "Hopea", "S&P 500", "Nasdaq"] if c in corr.columns]
        corr_small = corr.loc[keep, keep]
        st.dataframe(corr_small.round(2), use_container_width=True)

    st.caption("Markkinadata: Yahoo Finance. Sähke on tiivis kooste (ei sijoitusneuvo).")




