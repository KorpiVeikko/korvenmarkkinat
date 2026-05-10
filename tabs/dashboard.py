# tabs/dashboard.py
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from services.compare import build_market_compare, correlation_matrix


@st.cache_data(ttl=60 * 30, show_spinner=False)
def load_dashboard_market_data(period: str = "5y") -> dict:
    return build_market_compare(period=period)


HERO_IMAGE_PATH = "tabs/assets/dashboard_hero.png"
CORR_KEEP = ["Suomi (OMXH25)", "Bitcoin", "Kulta", "Hopea", "S&P 500", "Nasdaq"]


ASSET_DESCRIPTIONS = {
    "Bitcoin": "Korkean riskin digitaalinen omaisuuserä",
    "Kulta": "Perinteinen turvasatama",
    "Hopea": "Jalometalli, mutta kultaa syklisempi",
    "S&P 500": "Laaja USA-osakemarkkina",
    "Nasdaq": "Teknologia- ja kasvuyhtiöpainotteinen",
    "Suomi (OMXH25)": "Suomen suurimmat pörssiyhtiöt",
}


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


def _pct_text(pct: float | None) -> str:
    if pct is None or (isinstance(pct, float) and pd.isna(pct)):
        return "–"
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f} %"


def _pct_color(pct: float | None) -> str:
    if pct is None or (isinstance(pct, float) and pd.isna(pct)):
        return "#6b7280"
    return "#15803d" if pct >= 0 else "#b91c1c"


def _pct_html(pct: float | None, size: str = "1.0rem") -> str:
    return f"""
    <span style="
        color:{_pct_color(pct)};
        font-weight:700;
        font-size:{size};
    ">
        {_pct_text(pct)}
    </span>
    """


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

    top = ranked.iloc[0]
    bottom = ranked.iloc[-1] if len(ranked) >= 2 else None
    return top, bottom


def _corr_label(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "ei riittävästi dataa"
    v = abs(float(value))
    if v >= 0.80:
        return "liikkuvat lähes samaan tahtiin"
    if v >= 0.50:
        return "liikkuvat usein samaan suuntaan"
    if v >= 0.25:
        return "yhteys on kohtalainen"
    return "liikkuvat melko erillään"


def _corr_summary(corr: pd.DataFrame) -> str:
    if corr is None or corr.empty:
        return "Korrelaatiota ei voitu laskea, koska yhteistä dataa oli liian vähän."

    c = corr.copy()
    for col in c.columns:
        c.loc[col, col] = pd.NA

    stacked = c.stack(dropna=True)
    if stacked.empty:
        return "Korrelaatiosta ei saatu tiivistettävää."

    (a, b), val = max(stacked.items(), key=lambda kv: abs(kv[1]))

    return f"Vahvin yhteisliike on parilla **{a} ↔ {b}**. Ne {_corr_label(val)}."


def _normalize_growth(snaps: dict, start_value: float = 100.0) -> pd.DataFrame:
    frames = []

    for name, snap in snaps.items():
        df = snap.get("df")
        if df is None or df.empty or "Date" not in df.columns or "Close" not in df.columns:
            continue

        d = df.copy()
        d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
        d["Close"] = pd.to_numeric(d["Close"], errors="coerce")
        d = d.dropna(subset=["Date", "Close"]).sort_values("Date")

        if d.empty or d["Close"].iloc[0] == 0:
            continue

        d["Arvo"] = d["Close"] / d["Close"].iloc[0] * start_value
        d["Kohde"] = name
        frames.append(d[["Date", "Kohde", "Arvo"]])

    if not frames:
        return pd.DataFrame(columns=["Date", "Kohde", "Arvo"])

    return pd.concat(frames, ignore_index=True)


def _render_header() -> None:
    st.markdown("## Tervetuloa seuraamaan korvenmarkkinoita")
    st.caption(
        "Selkokielinen markkinanäkymä: mitä on noussut, mitä on laskenut ja miten eri omaisuuserät liikkuvat suhteessa toisiinsa."
    )
    st.divider()

    try:
        st.image(HERO_IMAGE_PATH, use_container_width=True)
    except Exception:
        st.info(f"Hero-kuvaa ei löytynyt polusta: {HERO_IMAGE_PATH}")

    st.divider()


def _render_market_signal(df: pd.DataFrame, corr: pd.DataFrame) -> None:
    st.subheader("📰 Markkinapulssi")

    top, bottom = _top_bottom_summary(df)
    corr_text = _corr_summary(corr)

    c1, c2, c3 = st.columns([1.1, 1.1, 2])

    with c1:
        with st.container(border=True):
            st.markdown("**📈 Kuukauden vahvin**")
            if top is not None:
                st.markdown(f"### {top['Nimi']}")
                st.markdown(_pct_html(top["1 kk"], "1.4rem"), unsafe_allow_html=True)
            else:
                st.write("–")

    with c2:
        with st.container(border=True):
            st.markdown("**📉 Kuukauden heikoin**")
            if bottom is not None:
                st.markdown(f"### {bottom['Nimi']}")
                st.markdown(_pct_html(bottom["1 kk"], "1.4rem"), unsafe_allow_html=True)
            else:
                st.write("–")

    with c3:
        with st.container(border=True):
            st.markdown("**🧠 Lyhyt tulkinta**")
            st.write(corr_text)
            st.caption("Korrelaatio kertoo, liikkuvatko kohteet usein samaan vai eri suuntaan.")

    st.divider()


def _render_market_cards(df: pd.DataFrame, snaps: dict) -> None:
    st.subheader("📊 Vertailu")
    st.caption("Muutosluvut ovat prosentteja. Vihreä tarkoittaa nousua ja punainen laskua.")

    rows = df.to_dict(orient="records")

    for i in range(0, len(rows), 3):
        cols = st.columns(3)

        for col, row in zip(cols, rows[i : i + 3]):
            name = row["Nimi"]
            now = _fmt_now(name, row["Nyt"], snaps)
            m1 = row["1 kk"]
            y1 = row["1 v"]

            with col:
                with st.container(border=True):
                    st.markdown(f"### {name}")
                    st.caption(ASSET_DESCRIPTIONS.get(name, ""))

                    st.markdown(f"**Nyt:** {now}")

                    c1, c2 = st.columns(2)
                    with c1:
                        st.caption("1 kk")
                        st.markdown(_pct_html(m1, "1.25rem"), unsafe_allow_html=True)
                    with c2:
                        st.caption("1 v")
                        st.markdown(_pct_html(y1, "1.25rem"), unsafe_allow_html=True)

    st.divider()


def _render_normalized_growth_chart(snaps: dict) -> None:
    st.subheader("💶 Miten 100 € olisi kehittynyt?")
    st.caption(
        "Kaikki sarjat alkavat arvosta 100. Tämä tekee eri kohteiden vertailusta helpompaa, vaikka niiden hinnat ovat eri tasoilla."
    )

    chart_df = _normalize_growth(snaps, start_value=100.0)

    if chart_df.empty:
        st.info("Kehityskuvaajaa ei voitu muodostaa.")
        return

    fig = px.line(
        chart_df,
        x="Date",
        y="Arvo",
        color="Kohde",
        title="Suhteellinen kehitys valitulla ajanjaksolla",
        labels={
            "Date": "Päivä",
            "Arvo": "Arvo, kun alkuhetki = 100",
            "Kohde": "",
        },
    )

    st.plotly_chart(fig, use_container_width=True)
    st.divider()


def _render_correlation_story(corr: pd.DataFrame) -> None:
    st.subheader("🔗 Miten kohteet liikkuvat yhdessä?")
    st.caption(
        "Korrelaatio auttaa ymmärtämään hajautusta. Korkea luku tarkoittaa, että kohteet heiluvat usein samaan suuntaan. "
        "Matala luku tarkoittaa, että ne liikkuvat enemmän erillään."
    )

    if corr is None or corr.empty:
        st.info("Korrelaatiota ei voitu laskea, koska yhteistä dataa oli liian vähän.")
        return

    pairs = [
        ("S&P 500", "Nasdaq", "USA-osakkeet"),
        ("Kulta", "Hopea", "Jalometallit"),
        ("Bitcoin", "S&P 500", "Bitcoin vs. osakeriski"),
        ("Kulta", "S&P 500", "Kulta vs. osakkeet"),
        ("Suomi (OMXH25)", "S&P 500", "Suomi vs. USA"),
    ]

    cols = st.columns(2)
    idx = 0

    for a, b, title in pairs:
        if a not in corr.columns or b not in corr.columns:
            continue

        val = corr.loc[a, b]
        if pd.isna(val):
            continue

        with cols[idx % 2]:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.markdown(f"{a} ↔ {b}")
                st.markdown(f"### {val:+.2f}")
                st.caption(_corr_label(float(val)))

        idx += 1

    with st.expander("Näytä tekninen korrelaatiotaulukko"):
        keep = [c for c in CORR_KEEP if c in corr.columns]
        corr_small = corr.loc[keep, keep]
        st.dataframe(corr_small.round(2), use_container_width=True)

    st.caption("Markkinadata: Yahoo Finance / EUR-muunnos EURUSD-kurssilla samoilla periaatteilla kuin muissa tabeissa.")


def render() -> None:
    _render_header()

    snaps = load_dashboard_market_data(period="5y")
    df = _build_market_table(snaps)

    if df.empty:
        st.warning("Markkinadataa ei saatu.")
        return

    corr = correlation_matrix(snaps, days=252)

    _render_market_signal(df, corr)
    _render_market_cards(df, snaps)
    _render_normalized_growth_chart(snaps)
    _render_correlation_story(corr)


