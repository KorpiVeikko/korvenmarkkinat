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


def _pct_html(
    value: float | None,
    label: str = "",
    positive_good: bool = True,
) -> str:

    if value is None or pd.isna(value):
        return ""

    is_positive = value >= 0
    good = is_positive if positive_good else not is_positive

    color = "#15803d" if good else "#b91c1c"

    if "muutos 1 v" in label:
        txt = fmt_money(value)

        if value > 0:
            txt = f"+{txt}"

        txt = f"{txt} ({label})"

    elif "kotitaloussignaali" in label:
        txt = f"{value:+.1f} pistettä"

    elif "%-yks." in label:
        txt = f"{value:+.1f} {label}"

    else:
        txt = f"{value:+.1f} % ({label})"

    return f"""
    <div style="
        color:{color};
        font-weight:700;
        margin-top:0.4rem;
    ">
        {txt}
    </div>
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

            if item.get("change_label") == "muutos 1 v":
                change_txt = fmt_money(change)
                if change > 0:
                    change_txt = f"+{change_txt}"
            elif "%-yks." in item.get("change_label", ""):
                change_txt = f"{change:+.1f} %-yks."
            else:
                change_txt = f"{change:+.1f} %"

            if good_change:
                strengths.append(f"{name}: vuosimuutos tukee näkymää ({change_txt}).")
            else:
                risks.append(f"{name}: vuosimuutos heikentää kuvaa ({change_txt}).")

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
    euribor_12m: float | None,
) -> tuple[str, str, float | None]:

    if wage_yoy is None or inflation_now is None:
        return "⚪", "Ei dataa", None

    score = 0.0

    purchasing_power = wage_yoy - inflation_now
    score += purchasing_power

    if unemployment_now is not None:
        if unemployment_now > 9:
            score -= 1.5
        elif unemployment_now > 7:
            score -= 0.5
        else:
            score += 0.5

    if household_debt_pct_gdi is not None:
        if household_debt_pct_gdi > 140:
            score -= 1.5
        elif household_debt_pct_gdi > 120:
            score -= 0.5
        else:
            score += 0.5

    if euribor_12m is not None:
        if euribor_12m > 4.0:
            score -= 1.5
        elif euribor_12m > 2.5:
            score -= 1.0
        elif euribor_12m > 1.5:
            score -= 0.5
        else:
            score += 0.5

    if score >= 2:
        return "🟢", "Vahvistuu", score
    if score <= -1:
        return "🔴", "Heikkenee", score
    return "🟡", "Paineessa", score


def _latest_full_year_trade_stats(df: pd.DataFrame) -> dict:
    if df is None or df.empty or "Aika_dt" not in df.columns or "Kauppatase_eur" not in df.columns:
        return {}

    d = df.copy()
    d["Aika_dt"] = pd.to_datetime(d["Aika_dt"], errors="coerce")
    d["Kauppatase_eur"] = pd.to_numeric(d["Kauppatase_eur"], errors="coerce")
    d = d.dropna(subset=["Aika_dt", "Kauppatase_eur"]).sort_values("Aika_dt")

    if d.empty:
        return {}

    d["Vuosi"] = d["Aika_dt"].dt.year
    d["Kuukausi"] = d["Aika_dt"].dt.month

    counts = d.groupby("Vuosi")["Kuukausi"].nunique().reset_index(name="kk_lkm")
    full_years = counts[counts["kk_lkm"] >= 12]["Vuosi"].tolist()

    if not full_years:
        return {}

    latest_year = max(full_years)
    prev_year = latest_year - 1

    yearly = (
        d[d["Vuosi"].isin([latest_year, prev_year])]
        .groupby("Vuosi", as_index=False)["Kauppatase_eur"]
        .sum()
        .sort_values("Vuosi")
    )

    latest_row = yearly[yearly["Vuosi"] == latest_year]
    if latest_row.empty:
        return {}

    latest_val = float(latest_row.iloc[0]["Kauppatase_eur"])

    change = None
    prev_row = yearly[yearly["Vuosi"] == prev_year]
    if not prev_row.empty:
        prev_val = float(prev_row.iloc[0]["Kauppatase_eur"])
        change = latest_val - prev_val

    return {
        "year": latest_year,
        "value": latest_val,
        "change": change,
    }


def render_macro_analysis(
    inflation_df: pd.DataFrame,
    gdp_df: pd.DataFrame,
    unemployment_df: pd.DataFrame,
    wages_df: pd.DataFrame,
    debt_df: pd.DataFrame,
    trade_balance_df: pd.DataFrame,
    household_debt_df: pd.DataFrame,
    interest_df: pd.DataFrame,
) -> None:
    st.subheader("🧠 Makrotalouden analyysi")
    st.caption("Yhteenveto Suomen makrotalouden keskeisistä mittareista.")

    inflation_now = _latest(inflation_df, "inflation_yoy")
    gdp_now = _latest(gdp_df, "gdp_yoy")
    unemployment_now = _latest(unemployment_df, "unemployment_rate_sa")
    debt_now = _latest(debt_df, "debt_pct_gdp")
    
    trade_stats = _latest_full_year_trade_stats(trade_balance_df)
    trade_now = trade_stats.get("value")
    trade_change = trade_stats.get("change")
    trade_year = trade_stats.get("year")

    household_debt_now = _latest(household_debt_df, "household_debt_pct_gdi")
    euribor_now = _latest(interest_df, "euribor_12m")
    euribor_change = _delta_points(interest_df, "euribor_12m", periods=12)

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
        euribor_12m=euribor_now,
    )

    inflation_change = _delta_points(inflation_df, "inflation_yoy", periods=12)
    gdp_change = _delta_points(gdp_df, "gdp_yoy", periods=4)
    unemployment_change = _delta_points(unemployment_df, "unemployment_rate_sa", periods=12)
    debt_change = _pct_change(debt_df, "debt_pct_gdp", periods=1)

    
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
            "label": "Kotitaloussignaali",
            "value": household_signal,
            "formatted_value": f"{household_signal:+.1f} pistettä" if household_signal is not None else "—",
            "change": None,
            "change_label": "",
            "positive_good": True,
            "icon": hh_icon,
            "status": hh_status,
            "caption": (
                f"Palkat {fmt(wage_yoy, 1, ' %')} • "
                f"inflaatio {fmt(inflation_now, 1, ' %')} • "
                f"työttömyys {fmt(unemployment_now, 1, ' %')} • "
                f"velka {fmt(household_debt_now, 0, ' % GDI')} • "
                f"Euribor 12 kk {fmt(euribor_now, 2, ' %')}"
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
            "change_label": "muutos 1 v",
            "positive_good": True,
            "icon": _status_from_metric("Kauppatase", trade_now)[0],
            "status": _status_from_metric("Kauppatase", trade_now)[1],
            "caption": f"Viimeisin täysi vuosi: {trade_year}" if trade_year else "",
        },
        {
            "name": "Korkopaine",
            "label": "12 kk Euribor",
            "value": euribor_now,
            "formatted_value": fmt(euribor_now, 2, " %"),
            "change": euribor_change,
            "change_label": "%-yks. (1 v)",
            "positive_good": False,
            "icon": "🔴" if euribor_now is not None and euribor_now > 3 else "🟡" if euribor_now is not None and euribor_now > 1.5 else "🟢",
            "status": "Korkea" if euribor_now is not None and euribor_now > 3 else "Koholla" if euribor_now is not None and euribor_now > 1.5 else "Kevyt",
            "caption": f"Viimeisin havainto: {_latest_date(interest_df, 'euribor_12m')}",
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