from __future__ import annotations

import pandas as pd
import streamlit as st

from services.macro_view_helpers import fmt, fmt_money



def _latest(df: pd.DataFrame, value_col: str):
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


def _latest_date(df: pd.DataFrame, value_col: str) -> str:
    if df is None or df.empty or value_col not in df.columns:
        return ""

    date_col = "Date" if "Date" in df.columns else "Aika_dt" if "Aika_dt" in df.columns else None
    if date_col is None:
        return ""

    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=[date_col, value_col]).sort_values(date_col)

    if d.empty:
        return ""

    return str(pd.to_datetime(d.iloc[-1][date_col]).date())


def _latest_trade_date(df: pd.DataFrame) -> str:
    if df is None or df.empty or "Aika_dt" not in df.columns:
        return ""

    dt = pd.to_datetime(df["Aika_dt"], errors="coerce").dropna()
    if dt.empty:
        return ""

    return str(dt.max().date())


def _pct_change(df: pd.DataFrame, value_col: str, periods: int) -> float | None:
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


def _delta_points(df: pd.DataFrame, value_col: str, periods: int) -> float | None:
    if df is None or df.empty or value_col not in df.columns:
        return None

    s = pd.to_numeric(df[value_col], errors="coerce").dropna()
    if len(s) <= periods:
        return None

    return float(s.iloc[-1] - s.iloc[-(periods + 1)])


def _pct_color(value: float | None, positive_good: bool = True) -> str:
    if value is None or pd.isna(value):
        return "#6b7280"

    v = value if positive_good else -value
    return "#15803d" if v >= 0 else "#b91c1c"


def _pct_html(value: float | None, label: str = "1 v", positive_good: bool = True) -> str:
    if value is None or pd.isna(value):
        txt = "—"
    else:
        if "%-yks." in label:
            txt = f"{value:+.1f} {label}"
        else:
            txt = f"{value:+.1f} % ({label})"

    return f"""
    <span style="
        color:{_pct_color(value, positive_good=positive_good)};
        font-weight:700;
        font-size:1.05rem;
    ">
        {txt}
    </span>
    """


def _status_from_metric(name: str, value: float | None) -> tuple[str, str]:
    if value is None or pd.isna(value):
        return "⚪", "Ei dataa"

    if name == "Inflaatio":
        if 1.0 <= value <= 3.0:
            return "🟢", "Tavoitealueella"
        if value <= 0.0:
            return "🟠", "Deflaatioriski"
        if value <= 5.0:
            return "🟡", "Koholla"
        return "🔴", "Liian korkea"

    if name == "BKT":
        if value >= 2.0:
            return "🟢", "Vahva kasvu"
        if value >= 0.5:
            return "🟢", "Kasvussa"
        if value >= 0.0:
            return "🟡", "Hidas kasvu"
        return "🔴", "Supistuu"

    if name == "Työttömyys":
        if value < 7.0:
            return "🟢", "Matala"
        if value < 9.0:
            return "🟡", "Koholla"
        return "🔴", "Korkea"

    if name == "Reaaliansiot":
        if value >= 2.0:
            return "🟢", "Ostovoima vahvistuu"
        if value >= 0.0:
            return "🟡", "Lievästi plussalla"
        return "🔴", "Ostovoima heikkenee"

    if name == "Velka":
        if value < 60.0:
            return "🟢", "Maltillinen"
        if value < 90.0:
            return "🟡", "Koholla"
        return "🔴", "Korkea"

    if name == "Kauppatase":
        if value > 0:
            return "🟢", "Ylijäämäinen"
        return "🔴", "Alijäämäinen"

    return "⚪", "Ei luokitusta"


def _macro_state_score(
    inflation_now: float | None,
    gdp_now: float | None,
    unemployment_now: float | None,
    real_wage_yoy: float | None,
    trade_now: float | None,
) -> tuple[int, str, str]:
    score = 0

    if inflation_now is not None:
        if 1 <= inflation_now <= 3:
            score += 1
        elif inflation_now > 5 or inflation_now <= 0:
            score -= 1

    if gdp_now is not None:
        if gdp_now > 1:
            score += 1
        elif gdp_now < 0:
            score -= 1

    if unemployment_now is not None:
        if unemployment_now < 7:
            score += 1
        elif unemployment_now > 9:
            score -= 1

    if real_wage_yoy is not None:
        if real_wage_yoy > 0:
            score += 1
        elif real_wage_yoy < -1:
            score -= 1

    if trade_now is not None:
        if trade_now > 0:
            score += 1
        else:
            score -= 1

    if score >= 3:
        return score, "🟢", "Vahva / paraneva suhdanne"
    if score >= 1:
        return score, "🟡", "Vakaa mutta epätasainen suhdanne"
    if score == 0:
        return score, "🟠", "Heikentyvä suhdanne"
    return score, "🔴", "Heikko suhdanne"


def _build_lists(indicators: list[dict]) -> tuple[list[str], list[str], list[str]]:
    strengths: list[str] = []
    risks: list[str] = []
    watchlist: list[str] = []

    for item in indicators:
        name = item["name"]
        status = item["status"]
        value = item.get("value")
        change = item.get("change")
        positive_good = item.get("positive_good", True)

        if value is None and change is None:
            watchlist.append(f"{name}: dataa ei saatu analyysiin.")
            continue

        if "Vahva" in status or "Kasvussa" in status or "Tavoitealueella" in status or "Ylijäämäinen" in status:
            strengths.append(f"{name}: {status.lower()}.")
        elif "Korkea" in status or "Supistuu" in status or "heikkenee" in status.lower() or "Alijäämäinen" in status:
            risks.append(f"{name}: {status.lower()}.")

        if change is not None:
            good_change = change >= 0 if positive_good else change <= 0
            if good_change:
                strengths.append(f"{name}: vuosimuutos tukee näkymää ({change:+.1f} %).")
            else:
                risks.append(f"{name}: vuosimuutos heikentää kuvaa ({change:+.1f} %).")

    if not strengths:
        strengths.append("Selviä vahvuuksia ei erottunut nykyisestä datasta.")

    if not risks:
        risks.append("Selviä riskisignaaleja ei erottunut nykyisestä datasta.")

    watchlist.append("Seuraa erityisesti BKT:n, työttömyyden ja reaaliansioiden yhteissuuntaa.")
    watchlist.append("Kauppatase kannattaa tulkita yhdessä viennin ja tuonnin kehityksen kanssa.")
    watchlist.append("Velka/BKT-suhteen nousu on ongelmallisinta silloin, jos BKT-kasvu jää samalla heikoksi.")

    return strengths, risks, watchlist


def _indicator_card(item: dict) -> None:
    with st.container(border=True):
        st.markdown(f"## {item.get('icon', '⚪')} {item['name']}")
        st.markdown(f"**Tila:** {item['status']}")

        st.metric(
            item["label"],
            item["formatted_value"],
        )

        st.markdown(
            _pct_html(
                item.get("change"),
                item.get("change_label", "1 v"),
                positive_good=item.get("positive_good", True),
            ),
            unsafe_allow_html=True,
        )

        if item.get("caption"):
            st.caption(item["caption"])



def _household_status(
    wage_yoy: float | None,
    inflation_now: float | None,
    unemployment_now: float | None,
    household_debt_pct_gdi: float | None,
) -> tuple[str, str, float | None]:

    if (
        wage_yoy is None
        or inflation_now is None
    ):
        return "⚪", "Ei dataa", None

    score = 0.0

    # Ostovoima
    purchasing_power = wage_yoy - inflation_now
    score += purchasing_power

    # Työttömyys rasittaa
    if unemployment_now is not None:
        if unemployment_now > 9:
            score -= 1.5
        elif unemployment_now > 7:
            score -= 0.5
        else:
            score += 0.5

    # Kotitalouksien velka
    if household_debt_pct_gdi is not None:
        if household_debt_pct_gdi > 140:
            score -= 1.5
        elif household_debt_pct_gdi > 120:
            score -= 0.5
        else:
            score += 0.5

    if score >= 2:
        return "🟢", "Vahvistuu", score

    if score <= -1:
        return "🔴", "Heikkenee", score

    return "🟡", "Paineessa", score



def render_macro_analysis(
    inflation_df: pd.DataFrame,
    gdp_df: pd.DataFrame,
    unemployment_df: pd.DataFrame,
    wages_df: pd.DataFrame,
    debt_df: pd.DataFrame,
    trade_balance_df: pd.DataFrame,
    household_debt_df: pd.DataFrame,
) -> None:
    st.subheader("🧠 Makrotalouden analyysi")
    st.caption("Yhteenveto Suomen makrotalouden keskeisistä mittareista.")

    inflation_now = _latest(inflation_df, "inflation_yoy")
    gdp_now = _latest(gdp_df, "gdp_yoy")
    unemployment_now = _latest(unemployment_df, "unemployment_rate_sa")
    debt_now = _latest(debt_df, "debt_pct_gdp")
    trade_now = _latest(trade_balance_df, "Kauppatase_eur")
    household_debt_now = _latest(household_debt_df, "household_debt_pct_gdi")

    wage_yoy = _latest(wages_df, "wage_index_yoy_pct")

    household_debt_now = _latest(
        household_debt_df,
        "household_debt_pct_gdi",
    )

    hh_icon, hh_status, household_signal = _household_status(
        wage_yoy=wage_yoy,
        inflation_now=inflation_now,
        unemployment_now=unemployment_now,
        household_debt_pct_gdi=household_debt_now,
    )

    inflation_change = _delta_points(inflation_df, "inflation_yoy", periods=12)
    gdp_change = _delta_points(gdp_df, "gdp_yoy", periods=4)
    unemployment_change = _delta_points(unemployment_df, "unemployment_rate_sa", periods=12)
    debt_change = _pct_change(debt_df, "debt_pct_gdp", periods=1)

    trade_change = None
    trade_change_good = None

    if trade_balance_df is not None and not trade_balance_df.empty and "Kauppatase_eur" in trade_balance_df.columns:
        trade_series = pd.to_numeric(trade_balance_df["Kauppatase_eur"], errors="coerce").dropna()

        if len(trade_series) > 12:
            latest_trade = float(trade_series.iloc[-1])
            prev_trade = float(trade_series.iloc[-13])

            if prev_trade != 0:
                raw_change = (latest_trade / prev_trade - 1.0) * 100.0

                # Kauppataseessa suurempi euromääräinen arvo on parempi:
                # -200 milj. € on parempi kuin -500 milj. €
                # +300 milj. € on parempi kuin +100 milj. €
                trade_change_good = latest_trade > prev_trade

                # Jos alijäämä kasvaa, näytetään muutos negatiivisena.
                trade_change = abs(raw_change) if trade_change_good else -abs(raw_change)

    score, state_icon, state_label = _macro_state_score(
        inflation_now=inflation_now,
        gdp_now=gdp_now,
        unemployment_now=unemployment_now,
        real_wage_yoy=household_signal,
        trade_now=trade_now,
    )

    indicators = [
        {
            "name": "Inflaatio",
            "label": "Inflaatio YoY",
            "value": inflation_now,
            "formatted_value": fmt(inflation_now, 1, " %"),
            "change": inflation_change,
            "change_label": "%-yks. (1 v)",
            "positive_good": False,
            "icon": _status_from_metric("Inflaatio", inflation_now)[0],
            "status": _status_from_metric("Inflaatio", inflation_now)[1],
            "caption": f"Viimeisin havainto: {_latest_date(inflation_df, 'inflation_yoy')}",
        },
        {
            "name": "BKT",
            "label": "BKT YoY",
            "value": gdp_now,
            "formatted_value": fmt(gdp_now, 1, " %"),
            "change": gdp_change,
            "change_label": "%-yks. (1 v)",
            "positive_good": True,
            "icon": _status_from_metric("BKT", gdp_now)[0],
            "status": _status_from_metric("BKT", gdp_now)[1],
            "caption": f"Viimeisin havainto: {_latest_date(gdp_df, 'gdp_yoy')}",
        },
        {
            "name": "Työttömyys",
            "label": "Työttömyysaste",
            "value": unemployment_now,
            "formatted_value": fmt(unemployment_now, 1, " %"),
            "change": unemployment_change,
            "change_label": "%-yks. (1 v)",
            "positive_good": False,
            "icon": _status_from_metric("Työttömyys", unemployment_now)[0],
            "status": _status_from_metric("Työttömyys", unemployment_now)[1],
            "caption": f"Viimeisin havainto: {_latest_date(unemployment_df, 'unemployment_rate_sa')}",
        },
        {
            "name": "Kotitaloudet",
            "label": "Ostovoimasignaali",
            "value": household_signal,
            "formatted_value": hh_status,
            "change": household_signal,
            "change_label": "palkat − inflaatio * velka %",
            "positive_good": True,
            "icon": hh_icon,
            "status": hh_status,
            "caption": (
                f"Palkat {fmt(wage_yoy, 1, ' %')} • "
                f"inflaatio {fmt(inflation_now, 1, ' %')} • "
                f"velka {fmt(household_debt_now, 0, ' % GDI')}"
            ),
        },
        {
            "name": "Velka / BKT",
            "label": "Julkinen velka",
            "value": debt_now,
            "formatted_value": fmt(debt_now, 1, " %"),
            "change": debt_change,
            "change_label": "1 v",
            "positive_good": False,
            "icon": _status_from_metric("Velka", debt_now)[0],
            "status": _status_from_metric("Velka", debt_now)[1],
            "caption": f"Viimeisin havainto: {_latest_date(debt_df, 'debt_pct_gdp')}",
        },
        {
            "name": "Kauppatase",
            "label": "Kauppatase",
            "value": trade_now,
            "formatted_value": fmt_money(trade_now),
            "change": trade_change,
            "change_label": "1 v",
            "positive_good": trade_change_good,
            "icon": _status_from_metric("Kauppatase", trade_now)[0],
            "status": _status_from_metric("Kauppatase", trade_now)[1],
            "caption": f"Viimeisin havainto: {_latest_trade_date(trade_balance_df)}",
        },
    ]

    st.markdown("### 📌 Tilaindikaattorit")

    with st.container(border=True):
        st.markdown(f"## {state_icon} {state_label}")
        st.write(f"Suhdannepisteet: **{score} / 5**")
        st.caption(
            "Pisteytys perustuu inflaatioon, BKT-kasvuun, työttömyyteen, kotitalouksien ostovoimasignaaliin ja kauppataseeseen."
        )

    c1, c2, c3 = st.columns(3)
    with c1:
        _indicator_card(indicators[0])
    with c2:
        _indicator_card(indicators[1])
    with c3:
        _indicator_card(indicators[2])

    c4, c5, c6 = st.columns(3)
    with c4:
        _indicator_card(indicators[3])
    with c5:
        _indicator_card(indicators[4])
    with c6:
        _indicator_card(indicators[5])

    st.divider()

    strengths, risks, watchlist = _build_lists(indicators)

    st.markdown("### 🧭 Tulkinta")

    a, b, c = st.columns(3)

    with a:
        with st.container(border=True):
            st.markdown("### ✅ Vahvuudet")
            for item in strengths[:5]:
                st.write(f"• {item}")

    with b:
        with st.container(border=True):
            st.markdown("### ⚠️ Riskit")
            for item in risks[:5]:
                st.write(f"• {item}")

    with c:
        with st.container(border=True):
            st.markdown("### 👀 Seuraa seuraavaksi")
            for item in watchlist[:5]:
                st.write(f"• {item}")

    st.divider()

    st.info(
        "Analyysi on sääntöpohjainen yhteenveto eri lähteiden makrodatasta. "
        "Eri mittarit päivittyvät eri aikatauluissa, joten havaintopäivämäärät kannattaa huomioida tulkinnassa."
    )