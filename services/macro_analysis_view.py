from __future__ import annotations

import pandas as pd
import streamlit as st

from services.macro_data import latest_row_by_date
from services.macro_view_helpers import fmt, fmt_money


def _latest(df: pd.DataFrame, value_col: str) -> float | None:
    if df is None or df.empty or value_col not in df.columns:
        return None

    date_col = "Date" if "Date" in df.columns else "Aika_dt" if "Aika_dt" in df.columns else None
    if date_col is None:
        return None

    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=[date_col, value_col]).sort_values(date_col)

    if d.empty:
        return None

    return float(d.iloc[-1][value_col])


def _latest_pct(df: pd.DataFrame, value_col: str, periods: int = 12) -> float | None:
    if df is None or df.empty or value_col not in df.columns:
        return None

    s = pd.to_numeric(df[value_col], errors="coerce").dropna()
    if len(s) <= periods:
        return None

    latest = float(s.iloc[-1])
    prev = float(s.iloc[-(periods + 1)])

    if prev == 0 or pd.isna(prev):
        return None

    return (latest / prev - 1.0) * 100.0


def _pct_color(value: float | None, positive_good: bool = True) -> str:
    if value is None or pd.isna(value):
        return "#6b7280"

    v = value if positive_good else -value
    return "#15803d" if v >= 0 else "#b91c1c"


def _pct_html(value: float | None, label: str = "", positive_good: bool = True) -> str:
    if value is None or pd.isna(value):
        txt = "—"
    else:
        txt = f"{value:+.1f} %"
        if label:
            txt += f" ({label})"

    return f"""
    <span style="
        color:{_pct_color(value, positive_good)};
        font-weight:700;
        font-size:1.05rem;
    ">
        {txt}
    </span>
    """


def _status_from_indicator(
    value: float | None,
    *,
    kind: str,
) -> tuple[str, str, int]:
    if value is None or pd.isna(value):
        return "⚪", "Ei dataa", 0

    if kind == "inflation":
        if 1.0 <= value <= 3.0:
            return "🟢", "Hyvä", 1
        if 0.0 <= value < 1.0 or 3.0 < value <= 5.0:
            return "🟡", "Seurattava", 0
        return "🔴", "Heikko", -1

    if kind == "gdp":
        if value >= 1.5:
            return "🟢", "Kasvava", 1
        if value >= 0:
            return "🟡", "Hidas kasvu", 0
        return "🔴", "Supistuva", -1

    if kind == "unemployment":
        if value < 7:
            return "🟢", "Hyvä", 1
        if value <= 9:
            return "🟡", "Koholla", 0
        return "🔴", "Heikko", -1

    if kind == "real_wages":
        if value >= 1:
            return "🟢", "Paranee", 1
        if value > -1:
            return "🟡", "Vakaa", 0
        return "🔴", "Heikkenee", -1

    if kind == "debt":
        if value < 70:
            return "🟢", "Kohtuullinen", 1
        if value <= 90:
            return "🟡", "Koholla", 0
        return "🔴", "Korkea", -1

    if kind == "trade":
        if value > 0:
            return "🟢", "Ylijäämäinen", 1
        if value > -1_000_000_000:
            return "🟠", "Lievä alijäämä", -1
        return "🔴", "Alijäämäinen", -2

    return "⚪", "Ei luokitusta", 0


def _cycle_state(score: float | None) -> tuple[str, str]:
    if score is None:
        return "⚪", "Ei riittävästi dataa"

    if score >= 0.7:
        return "🟢", "Vahva / kasvava vaihe"
    if score >= 0.2:
        return "🟡", "Vakaa kasvu"
    if score > -0.5:
        return "🟠", "Heikentyvä suhdanne"
    return "🔴", "Taantumariski koholla"


def _indicator_card(item: dict) -> None:
    with st.container(border=True):
        st.markdown(f"### {item['Ikoni']} {item['Nimi']}")
        st.markdown(f"**Tila:** {item['Tila']}")
        st.markdown(f"## {item['Arvo']}")
        if item.get("Muutos") is not None:
            st.markdown(
                _pct_html(
                    item["Muutos"],
                    item.get("MuutosLabel", ""),
                    item.get("PositiveGood", True),
                ),
                unsafe_allow_html=True,
            )
        st.caption(item.get("Selite", ""))


def _build_lists(indicators: list[dict]) -> tuple[list[str], list[str], list[str]]:
    strengths: list[str] = []
    risks: list[str] = []
    watchlist: list[str] = []

    for item in indicators:
        name = item["Nimi"]
        score = item.get("Score", 0)
        tila = item.get("Tila", "Ei dataa")

        if score > 0:
            strengths.append(f"{name}: {tila}.")
        elif score < 0:
            risks.append(f"{name}: {tila}.")
        else:
            watchlist.append(f"{name}: {tila}.")

    if not strengths:
        strengths.append("Selviä vahvuuksia ei erotu nykyisestä datasta.")

    if not risks:
        risks.append("Selviä riskisignaaleja ei erotu nykyisestä datasta.")

    if not watchlist:
        watchlist.append("Seuraa, vahvistuuko BKT:n kasvu ja helpottaako työttömyys.")

    return strengths, risks, watchlist


def render_macro_analysis(
    inflation_df: pd.DataFrame,
    gdp_df: pd.DataFrame,
    unemployment_df: pd.DataFrame,
    wages_df: pd.DataFrame,
    debt_df: pd.DataFrame,
    trade_balance_df: pd.DataFrame,
) -> None:
    st.subheader("🧠 Makrotalouden analyysi")
    st.caption("Yhteenveto Suomen makrotalouden keskeisistä mittareista.")

    inflation_now = _latest(inflation_df, "inflation_yoy")
    gdp_now = _latest(gdp_df, "gdp_yoy")
    unemployment_now = _latest(unemployment_df, "unemployment_rate_sa")
    debt_now = _latest(debt_df, "debt_pct_gdp")
    trade_now = _latest(trade_balance_df, "Kauppatase_eur")

    real_wage_now = _latest(wages_df, "real_wage_index")
    real_wage_yoy = _latest_pct(wages_df, "real_wage_index", periods=4)

    inflation_icon, inflation_status, inflation_score = _status_from_indicator(inflation_now, kind="inflation")
    gdp_icon, gdp_status, gdp_score = _status_from_indicator(gdp_now, kind="gdp")
    unemp_icon, unemp_status, unemp_score = _status_from_indicator(unemployment_now, kind="unemployment")
    wage_icon, wage_status, wage_score = _status_from_indicator(real_wage_yoy, kind="real_wages")
    debt_icon, debt_status, debt_score = _status_from_indicator(debt_now, kind="debt")
    trade_icon, trade_status, trade_score = _status_from_indicator(trade_now, kind="trade")

    scores = [inflation_score, gdp_score, unemp_score, wage_score, debt_score, trade_score]
    score_avg = sum(scores) / len(scores) if scores else None
    cycle_icon, cycle_label = _cycle_state(score_avg)

    summary_parts = []

    if inflation_now is not None:
        if 1 <= inflation_now <= 3:
            summary_parts.append("Inflaatio on tavoitteen kannalta maltillisella alueella.")
        elif inflation_now > 5:
            summary_parts.append("Inflaatio on edelleen selvästi koholla.")
        else:
            summary_parts.append("Inflaatio on matala, mikä voi kertoa kysynnän vaimeudesta.")

    if gdp_now is not None:
        if gdp_now > 1:
            summary_parts.append("BKT:n kasvu tukee kokonaiskuvaa.")
        elif gdp_now >= 0:
            summary_parts.append("BKT kasvaa, mutta tahti on hidas.")
        else:
            summary_parts.append("BKT supistuu, mikä painaa suhdannekuvaa.")

    if unemployment_now is not None:
        if unemployment_now > 9:
            summary_parts.append("Työttömyys on korkea ja on analyysin keskeinen riskitekijä.")
        elif unemployment_now > 7:
            summary_parts.append("Työttömyys on koholla.")
        else:
            summary_parts.append("Työllisyystilanne tukee talouskuvaa.")

    if real_wage_yoy is not None:
        if real_wage_yoy > 0:
            summary_parts.append("Reaaliansiot ovat nousussa, mikä tukee ostovoimaa.")
        else:
            summary_parts.append("Reaaliansiot eivät vielä tue ostovoimaa vahvasti.")

    if debt_now is not None and debt_now > 85:
        summary_parts.append("Julkinen velkasuhde on korkealla tasolla.")

    if trade_now is not None and trade_now < 0:
        summary_parts.append("Kauppatase on alijäämäinen, mikä heikentää ulkoisen tasapainon kuvaa.")

    if not summary_parts:
        summary_parts.append("Analyysia ei voitu muodostaa, koska keskeisiä mittareita puuttuu.")

    with st.container(border=True):
        st.markdown(f"## {cycle_icon} {cycle_label}")
        st.write(" ".join(summary_parts))

    st.divider()

    st.markdown("### 📌 Tilaindikaattorit")

    indicators = [
        {
            "Nimi": "Inflaatio",
            "Arvo": fmt(inflation_now, 1, " %"),
            "Ikoni": inflation_icon,
            "Tila": inflation_status,
            "Score": inflation_score,
            "Selite": "Kuluttajahintojen vuosimuutos.",
        },
        {
            "Nimi": "BKT YoY",
            "Arvo": fmt(gdp_now, 1, " %"),
            "Ikoni": gdp_icon,
            "Tila": gdp_status,
            "Score": gdp_score,
            "Selite": "BKT:n vuosikasvu.",
        },
        {
            "Nimi": "Työttömyys",
            "Arvo": fmt(unemployment_now, 1, " %"),
            "Ikoni": unemp_icon,
            "Tila": unemp_status,
            "Score": unemp_score,
            "Selite": "Kausitasoitettu työttömyysaste.",
        },
        {
            "Nimi": "Reaaliansiot",
            "Arvo": fmt(real_wage_now, 1),
            "Muutos": real_wage_yoy,
            "MuutosLabel": "1 v",
            "Ikoni": wage_icon,
            "Tila": wage_status,
            "Score": wage_score,
            "Selite": "Ostovoiman kehitys palkkaindeksin perusteella.",
        },
        {
            "Nimi": "Velka / BKT",
            "Arvo": fmt(debt_now, 1, " %"),
            "Ikoni": debt_icon,
            "Tila": debt_status,
            "Score": debt_score,
            "Selite": "Julkisen velan suhde BKT:hen.",
        },
        {
            "Nimi": "Kauppatase",
            "Arvo": fmt_money(trade_now),
            "Ikoni": trade_icon,
            "Tila": trade_status,
            "Score": trade_score,
            "Selite": "Tavaravienti miinus tavaratuonti.",
        },
    ]

    for i in range(0, len(indicators), 3):
        cols = st.columns(3)
        for col, item in zip(cols, indicators[i : i + 3]):
            with col:
                _indicator_card(item)

    st.divider()

    strengths, risks, watchlist = _build_lists(indicators)

    c1, c2, c3 = st.columns(3)

    with c1:
        with st.container(border=True):
            st.markdown("### ✅ Vahvuudet")
            for item in strengths:
                st.write(f"• {item}")

    with c2:
        with st.container(border=True):
            st.markdown("### ⚠️ Riskit")
            for item in risks:
                st.write(f"• {item}")

    with c3:
        with st.container(border=True):
            st.markdown("### 👀 Seurattavaa")
            for item in watchlist:
                st.write(f"• {item}")

    st.info(
        "Tämä analyysi on sääntöpohjainen tilannekuva. Se ei ole ennuste, vaan kokoaa "
        "sovelluksen datasta keskeiset suhdannesignaalit yhteen näkymään."
    )