# services/currency_view_helpers.py
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from services.currency_data import (
    CURRENCY_META,
    generate_ai_summary,
    get_major_macro_currencies,
    is_major_macro_currency,
)
from services.currency_utils import (
    ANCHOR_CURRENCY,
    build_fx_metrics,
    change_metrics,
    fmt_money_supply,
    fmt_num,
    fmt_pct,
    pct_color_style,
    to_anchor_fx,
)


DISPLAY_START_DATE = pd.Timestamp("2000-01-01")


def _latest_non_null_with_date(df: pd.DataFrame, column: str):
    if (
        df is None
        or df.empty
        or column not in df.columns
    ):
        return None, None

    tmp = (
        df.dropna(subset=[column])
        .sort_values("Date")
    )

    if tmp.empty:
        return None, None

    row = tmp.iloc[-1]

    return row[column], row["Date"]


def _analysis_snapshot(currency: str, years: int, load_currency_bundle) -> dict:
    bundle = load_currency_bundle(currency, years=years)

    comparison_anchor = "EUR" if currency == ANCHOR_CURRENCY else ANCHOR_CURRENCY
    anchor_bundle = load_currency_bundle(comparison_anchor, years=years)

    fx = to_anchor_fx(
        bundle.get("fx", pd.DataFrame()),
        anchor_bundle.get("fx", pd.DataFrame()),
    )
    fx_metrics = build_fx_metrics(fx)

    money = bundle.get("money", pd.DataFrame())
    macro = bundle.get("macro", pd.DataFrame())

    money_metrics = change_metrics(money, "Date", "BroadMoney_LCU")

    inflation, _ = _latest_non_null(macro, "InflationCPI_Pct")
    policy_rate, _ = _latest_non_null(macro, "PolicyRate_Pct")
    real_rate, _ = _latest_non_null(macro, "RealInterestRate_Pct")

    return {
        "currency": currency,
        "anchor": comparison_anchor,
        "fx_metrics": fx_metrics,
        "money": money,
        "macro": macro,
        "money_1y": money_metrics.change_1y_pct,
        "money_5y": money_metrics.change_5y_pct,
        "inflation": inflation,
        "policy_rate": policy_rate,
        "real_rate": real_rate,
        "fx_1y": fx_metrics.change_1y_pct,
        "fx_vol": fx_metrics.volatility_1y_pct,
    }


def _score_currency_snapshot(s: dict) -> int:
    score = 0

    fx_1y = s.get("fx_1y")
    fx_vol = s.get("fx_vol")
    money_1y = s.get("money_1y")
    inflation = s.get("inflation")
    real_rate = s.get("real_rate")

    # Sama logiikka kuin render_currency_health_card:
    # valuutta saa FX-pisteen vain jos se on vahvistunut selvästi.
    if fx_1y is not None:
        if fx_1y < -3:
            score += 1

    if fx_vol is not None:
        if fx_vol < 8:
            score += 1

    if money_1y is not None:
        if money_1y < 6:
            score += 1

    if inflation is not None:
        if 0 < inflation < 3:
            score += 1

    if real_rate is not None:
        if real_rate > 0:
            score += 1

    return score



def _render_currency_comparison_table(rows: list[dict]) -> None:
    table = pd.DataFrame(
        [
            {
                "Valuutta": r["currency"],
                "FX 1v %": r["fx_1y"],
                "Volatiliteetti 1v %": r["fx_vol"],
                "Rahamäärä 1v %": r["money_1y"],
                "Rahamäärä 5v %": r["money_5y"],
                "Inflaatio %": r["inflation"],
                "Ohjauskorko %": r["policy_rate"],
                "Reaalikorko %": r["real_rate"],
                "Pisteet": f"{_score_currency_snapshot(r)} / 5",
            }
            for r in rows
        ]
    )

    st.markdown("### 📊 Valuuttavertailu")

    styled = (
        table.style.format(
            {
                "FX 1v %": "{:+.1f}",
                "Volatiliteetti 1v %": "{:.1f}",
                "Rahamäärä 1v %": "{:+.1f}",
                "Rahamäärä 5v %": "{:+.1f}",
                "Inflaatio %": "{:+.1f}",
                "Ohjauskorko %": "{:+.1f}",
                "Reaalikorko %": "{:+.1f}",
            },
            na_rep="—",
        )
    )

    st.dataframe(styled, width="stretch", hide_index=True)


def _render_currency_health_ranking(rows: list[dict]) -> None:
    if not rows:
        return

    rank = pd.DataFrame(
        [
            {
                "Valuutta": r["currency"],
                "Pisteet": _score_currency_snapshot(r),
                "Inflaatio %": r.get("inflation"),
                "Reaalikorko %": r.get("real_rate"),
                "Rahamäärä 1v %": r.get("money_1y"),
                "FX 1v %": r.get("fx_1y"),
            }
            for r in rows
        ]
    )

    rank = rank.sort_values(["Pisteet", "Reaalikorko %"], ascending=[False, False])

    st.markdown("### 🏆 Valuuttojen ranking")

    styled = rank.style.format(
        {
            "Inflaatio %": "{:+.1f}",
            "Reaalikorko %": "{:+.1f}",
            "Rahamäärä 1v %": "{:+.1f}",
            "FX 1v %": "{:+.1f}",
        },
        na_rep="—",
    )

    st.dataframe(styled, width="stretch", hide_index=True)


def _render_multi_currency_interpretation(rows: list[dict]) -> None:
    st.markdown("### 🧭 Tulkinta")

    if not rows:
        st.info("Tulkintaa ei voitu muodostaa, koska valuuttadataa ei ole.")
        return

    scored = []
    for r in rows:
        scored.append(
            {
                **r,
                "score": _score_currency_snapshot(r),
            }
        )

    ranked = sorted(scored, key=lambda x: x["score"], reverse=True)
    best = ranked[0]
    weakest = ranked[-1]

    points: list[str] = []

    points.append(
        f"Kokonaispisteissä vahvin valuutta on {best['currency']} "
        f"({best['score']}/5) ja heikoin {weakest['currency']} "
        f"({weakest['score']}/5)."
    )

    cny = next((x for x in scored if x["currency"] == "CNY"), None)
    jpy = next((x for x in scored if x["currency"] == "JPY"), None)
    usd = next((x for x in scored if x["currency"] == "USD"), None)
    eur = next((x for x in scored if x["currency"] == "EUR"), None)

    if cny:
        cny_notes = []
        if cny.get("inflation") is not None and cny["inflation"] < 3:
            cny_notes.append("inflaatio on matala")
        if cny.get("real_rate") is not None and cny["real_rate"] > 0:
            cny_notes.append("reaalikorko on positiivinen")
        if cny.get("fx_vol") is not None and cny["fx_vol"] < 8:
            cny_notes.append("valuuttakurssin vaihtelu on rauhallista")
        if cny_notes:
            points.append(
                "CNY näyttää mittareilla vahvalta: "
                + ", ".join(cny_notes)
                + "."
            )

    if jpy:
        jpy_notes = []
        if jpy.get("fx_1y") is not None and jpy["fx_1y"] > 3:
            jpy_notes.append("jeni on heikentynyt suhteessa USD:hen")
        if jpy.get("real_rate") is not None and jpy["real_rate"] < 0:
            jpy_notes.append("reaalikorko on negatiivinen")
        if jpy.get("money_1y") is not None and jpy["money_1y"] < 6:
            jpy_notes.append("rahamäärän kasvu on maltillista")
        if jpy_notes:
            points.append(
                "JPY on kaksijakoinen: "
                + ", ".join(jpy_notes)
                + "."
            )

    if usd and eur:
        usd_score = _score_currency_snapshot(usd)
        eur_score = _score_currency_snapshot(eur)

        if usd_score > eur_score:
            points.append(f"USD saa euroon verrattuna paremman kokonaislukeman ({usd_score}/5 vs. {eur_score}/5).")
        elif eur_score > usd_score:
            points.append(f"EUR saa dollariin verrattuna paremman kokonaislukeman ({eur_score}/5 vs. {usd_score}/5).")
        else:
            points.append(f"USD ja EUR ovat kokonaispisteissä tasoissa ({usd_score}/5).")

        if usd.get("real_rate") is not None and eur.get("real_rate") is not None:
            if usd["real_rate"] > eur["real_rate"]:
                points.append("USD:n reaalikorko on euroa korkeampi, mikä tukee dollaria korkoeron näkökulmasta.")
            elif eur["real_rate"] > usd["real_rate"]:
                points.append("EUR:n reaalikorko on dollaria korkeampi, mikä tukee euroa korkoeron näkökulmasta.")

        if usd.get("inflation") is not None and eur.get("inflation") is not None:
            if usd["inflation"] > eur["inflation"]:
                points.append("Yhdysvaltain inflaatio on euroaluetta korkeampi, mikä heikentää dollarin ostovoimatulkintaa.")
            elif eur["inflation"] > usd["inflation"]:
                points.append("Euroalueen inflaatio on Yhdysvaltoja korkeampi, mikä heikentää euron ostovoimatulkintaa.")

    with st.container(border=True):
        for p in points:
            st.write(f"• {p}")

def _show_debug(title: str, message: str | None) -> None:
    if not message:
        return
    with st.expander(title, expanded=False):
        st.code(message)


def _latest_non_null(df: pd.DataFrame, value_col: str) -> tuple[float | None, pd.Timestamp | None]:
    if df is None or df.empty or value_col not in df.columns:
        return None, None

    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=["Date", value_col]).sort_values("Date")

    if d.empty:
        return None, None

    row = d.iloc[-1]
    return float(row[value_col]), pd.to_datetime(row["Date"])


def _render_money_macro_analysis(
    currency: str,
    money_1y: float | None,
    money_5y: float | None,
    inflation: float | None,
    policy_rate: float | None,
    real_rate: float | None,
) -> None:
    score, icon, status = _macro_score(money_1y, inflation, real_rate)

    st.markdown("### 🧭 Nykytilanne")

    points = []

    if money_1y is not None:
        if money_1y < 6:
            points.append("Rahamäärän kasvu on maltillista.")
        elif money_1y < 10:
            points.append("Rahamäärän kasvu on koholla.")
        else:
            points.append("Rahamäärä kasvaa nopeasti.")

    if inflation is not None:
        if 0 <= inflation < 3:
            points.append("Inflaatio on hallinnassa.")
        elif inflation < 5:
            points.append("Inflaatio on koholla.")
        else:
            points.append("Inflaatio on korkea.")

    if real_rate is not None:
        if real_rate > 0:
            points.append("Reaalikorko on positiivinen, mikä viittaa kiristävään rahapolitiikkaan.")
        elif real_rate > -1:
            points.append("Reaalikorko on lähellä nollaa.")
        else:
            points.append("Reaalikorko on negatiivinen, mikä viittaa yhä kevyempään rahapolitiikkaan.")

    if money_5y is not None:
        if money_5y > 25:
            points.append("Viiden vuoden rahamäärän kasvu on voimakasta.")
        elif money_5y > 10:
            points.append("Viiden vuoden rahamäärän kasvu on kohtalaista.")
        else:
            points.append("Viiden vuoden rahamäärän kasvu on maltillista.")

    with st.container(border=True):
        st.markdown(f"### {icon} {currency}: {status}")
        st.metric("Makropisteet", f"{score} / 3")

        if points:
            for p in points:
                st.write(f"• {p}")
        else:
            st.info("Tulkintaa ei voitu muodostaa, koska dataa puuttuu.")


def render_currency_health_card(
    currency: str,
    fx_metrics,
    money_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    show_title: bool = True,
) -> None:
    score = 0
    max_score = 5
    positives: list[str] = []
    risks: list[str] = []

    if fx_metrics.change_1y_pct is not None:
        if fx_metrics.change_1y_pct < -3:
            score += 1
            positives.append("Valuutta on vahvistunut.")
        elif fx_metrics.change_1y_pct > 3:
            risks.append("Valuutta on heikentynyt.")

    if fx_metrics.volatility_1y_pct is not None:
        if fx_metrics.volatility_1y_pct < 8:
            score += 1
            positives.append("Kurssivaihtelu on rauhallista.")
        elif fx_metrics.volatility_1y_pct > 12:
            risks.append("Kurssivaihtelu on voimakasta.")

    money_growth = None
    money_date = None

    if (
        money_df is not None
        and not money_df.empty
        and "BroadMoney_GrowthPct" in money_df.columns
    ):
        m = money_df.dropna(subset=["BroadMoney_GrowthPct"]).sort_values("Date")

        if not m.empty:
            money_growth = float(m.iloc[-1]["BroadMoney_GrowthPct"])
            money_date = pd.to_datetime(m.iloc[-1]["Date"], errors="coerce")

            if money_growth < 6:
                score += 1
                positives.append("Rahamäärän kasvu on maltillista.")
            elif money_growth > 10:
                risks.append("Rahamäärä kasvaa nopeasti.")

    inflation = None
    inflation_date = None
    real_rate = None
    real_rate_date = None

    if macro_df is not None and not macro_df.empty:
        inflation, inflation_date = _latest_non_null_with_date(
            macro_df,
            "InflationCPI_Pct",
        )

        real_rate, real_rate_date = _latest_non_null_with_date(
            macro_df,
            "RealInterestRate_Pct",
        )

        if inflation is not None:
            if 0 < inflation < 3:
                score += 1
                positives.append("Inflaatio on hallinnassa.")
            elif inflation > 5:
                risks.append("Inflaatio on korkealla.")

        if real_rate is not None:
            if real_rate > 0:
                score += 1
                positives.append("Reaalikorko on positiivinen.")
            elif real_rate < -1:
                risks.append("Reaalikorko on selvästi negatiivinen.")

    fx_date = getattr(fx_metrics, "latest_date", None)
    fx_date = pd.to_datetime(fx_date, errors="coerce") if fx_date is not None else None

    inflation_date = pd.to_datetime(inflation_date, errors="coerce") if inflation_date is not None else None
    real_rate_date = pd.to_datetime(real_rate_date, errors="coerce") if real_rate_date is not None else None

    if score >= 4:
        icon, status = "🟢", "Vahva"
    elif score >= 2:
        icon, status = "🟡", "Neutraali"
    else:
        icon, status = "🔴", "Paineessa"

    if show_title:
        st.markdown("### 🧭 Valuutan terveysmittari")

    with st.container(border=True):
        top1, top2, top3 = st.columns([1.4, 0.8, 1.0])

        with top1:
            st.markdown(f"### {icon} {currency}")
            st.caption(f"Tila: {status}")

        with top2:
            st.metric("Pisteet", f"{score}/{max_score}")

        with top3:
            st.metric("Inflaatio", fmt_pct(inflation))
            if inflation_date is not None and not pd.isna(inflation_date):
                st.caption(inflation_date.strftime("%Y-%m"))

        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.caption("FX 1 v")
            st.write(fmt_pct(fx_metrics.change_1y_pct))
            if fx_date is not None and not pd.isna(fx_date):
                st.caption(fx_date.strftime("%Y-%m-%d"))

        with m2:
            st.caption("Volatiliteetti")
            st.write(fmt_pct(fx_metrics.volatility_1y_pct))
            if fx_date is not None and not pd.isna(fx_date):
                st.caption(fx_date.strftime("%Y-%m-%d"))

        with m3:
            st.caption("Rahamäärä")
            st.write(fmt_pct(money_growth))
            if money_date is not None and not pd.isna(money_date):
                st.caption(money_date.strftime("%Y-%m"))

        with m4:
            st.caption("Reaalikorko")
            st.write(fmt_pct(real_rate))
            if real_rate_date is not None and not pd.isna(real_rate_date):
                st.caption(real_rate_date.strftime("%Y-%m"))

        note_items = []

        if positives:
            note_items.append("✅ " + positives[0])

        if risks:
            note_items.append("⚠️ " + risks[0])

        if note_items:
            st.caption(" | ".join(note_items))


def render_overview_tab(overview: pd.DataFrame) -> None:
    st.markdown("#### Seurattavat valuutat")
    st.caption(
        "**YTD %** = *year-to-date*, eli muutos vuoden alusta tähän päivään. "
        "Päätaulukossa näytetään YTD, 1 v ja 5 v, koska ne ovat yleensä käytännöllisimmät tarkastelujaksot. "
        "10 vuoden muutos jätettiin pois, koska se jää usein tyhjäksi datan kattavuuden takia eikä yleensä lisää paljon käytännön hyötyä."
    )

    if overview is None or overview.empty:
        st.warning("Valuuttayhteenvetoa ei saatu.")
        return

    show = overview.copy()

    numeric_cols = ["Nykykurssi", "YTD %", "1v %", "5v %", "Volatiliteetti 1v %", "Min", "Max"]
    for col in numeric_cols:
        show[col] = pd.to_numeric(show[col], errors="coerce")

    styled = (
        show[["Koodi", "Valuutta", "Nykykurssi", "YTD %", "1v %", "5v %", "Volatiliteetti 1v %", "Min", "Max", "Viimeisin päivä"]]
        .style
        .format(
            {
                "Nykykurssi": "{:.4f}",
                "YTD %": "{:+.1f} %",
                "1v %": "{:+.1f} %",
                "5v %": "{:+.1f} %",
                "Volatiliteetti 1v %": "{:.1f} %",
                "Min": "{:.4f}",
                "Max": "{:.4f}",
            },
            na_rep="—",
        )
        .map(pct_color_style, subset=["YTD %", "1v %", "5v %"])
    )

    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("#### Ranking")

    rank_metric = st.selectbox(
        "Valitse ranking-mittari",
        ["YTD %", "1v %", "5v %"],
        index=1,
        key="currency_rank_metric",
    )

    rank_df = show[["Koodi", "Valuutta", rank_metric]].copy()
    rank_df[rank_metric] = pd.to_numeric(rank_df[rank_metric], errors="coerce")
    rank_df = rank_df.dropna(subset=[rank_metric]).sort_values(rank_metric, ascending=False)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Vahvimmat**")
        st.dataframe(rank_df.head(5), use_container_width=True, hide_index=True)

    with c2:
        st.markdown("**Heikoimmat**")
        st.dataframe(
            rank_df.tail(5).sort_values(rank_metric, ascending=True),
            use_container_width=True,
            hide_index=True,
        )


def render_fx_tab(
    fx_currency: str,
    years: int,
    load_currency_bundle,
) -> None:
    fx_bundle = load_currency_bundle(fx_currency, years=years)
    anchor_bundle = load_currency_bundle(ANCHOR_CURRENCY, years=years)

    fx = to_anchor_fx(fx_bundle["fx"], anchor_bundle["fx"])
    fx_metrics = build_fx_metrics(fx)

    if fx is None or fx.empty:
        st.warning("Kurssihistoriaa ei saatu.")
        return

    st.markdown(f"#### {fx_currency} / {ANCHOR_CURRENCY}")

    k1, k2, k3, k4 = st.columns(4, gap="large")

    with k1:
        st.metric(
            f"{fx_currency} / {ANCHOR_CURRENCY}",
            fmt_num(fx_metrics.latest_rate, 4),
            f"{fmt_pct(fx_metrics.ytd_pct)} (YTD)" if fx_metrics.ytd_pct is not None else None,
        )
        if fx_metrics.latest_date is not None:
            st.caption(f"Päivä: {fx_metrics.latest_date.date()}")

    with k2:
        st.metric("Muutos 1 v", fmt_pct(fx_metrics.change_1y_pct))
        st.caption("Valuuttakurssi")

    with k3:
        st.metric("Muutos 5 v", fmt_pct(fx_metrics.change_5y_pct))
        st.caption("Valuuttakurssi")

    with k4:
        st.metric("Volatiliteetti 1 v", fmt_pct(fx_metrics.volatility_1y_pct))
        st.caption("Annualisoitu")

    st.divider()

    fig = px.line(
        fx,
        x="Date",
        y="Rate",
        title=f"{fx_currency} / {ANCHOR_CURRENCY} – viimeiset {years} vuotta",
        labels={"Date": "Päivä", "Rate": f"{fx_currency} per {ANCHOR_CURRENCY}"},
    )

    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Min", fmt_num(fx_metrics.min_rate, 4))
    with c2:
        st.metric("Max", fmt_num(fx_metrics.max_rate, 4))


def _traffic_light(value: float | None, kind: str) -> tuple[str, str]:
    if value is None or pd.isna(value):
        return "⚪", "Ei dataa"

    if kind == "money_growth":
        if value < 6:
            return "🟢", "Maltillinen"
        if value < 10:
            return "🟡", "Koholla"
        return "🔴", "Nopea"

    if kind == "inflation":
        if 0 <= value < 3:
            return "🟢", "Hallinnassa"
        if value < 5:
            return "🟡", "Koholla"
        return "🔴", "Korkea"

    if kind == "policy_rate":
        if value >= 4:
            return "🟢", "Kireä"
        if value >= 1:
            return "🟡", "Neutraali"
        return "🔴", "Kevyt"

    if kind == "real_rate":
        if value > 0:
            return "🟢", "Positiivinen"
        if value > -1:
            return "🟡", "Lähellä nollaa"
        return "🔴", "Negatiivinen"

    return "⚪", "Ei luokitusta"


def _macro_score(
    money_1y: float | None,
    inflation: float | None,
    real_rate: float | None,
) -> tuple[int, str, str]:
    score = 0

    if money_1y is not None and money_1y < 6:
        score += 1

    if inflation is not None and 0 <= inflation < 3:
        score += 1

    if real_rate is not None and real_rate > 0:
        score += 1

    if score >= 3:
        return score, "🟢", "Vahva / terve"
    if score >= 2:
        return score, "🟡", "Neutraali / seurattava"
    return score, "🔴", "Paineessa"


def _metric_card(
    title: str,
    value: float | None,
    date_value,
    kind: str,
) -> None:
    icon, label = _traffic_light(value, kind)

    st.metric(title, fmt_pct(value))
    st.caption(f"{icon} {label}")

    if date_value is not None and not pd.isna(date_value):
        st.caption(f"Päivä: {pd.to_datetime(date_value).strftime('%Y-%m')}")
    else:
        st.caption("Päivä: —")


def _render_money_macro_summary_card(
    currency: str,
    money_metrics,
    inflation: float | None,
    inflation_date,
    policy_rate: float | None,
    policy_rate_date,
    real_rate: float | None,
    real_rate_date,
) -> None:
    score, icon, status = _macro_score(
        money_metrics.change_1y_pct,
        inflation,
        real_rate,
    )

    money_icon, money_label = _traffic_light(money_metrics.change_1y_pct, "money_growth")
    infl_icon, infl_label = _traffic_light(inflation, "inflation")
    policy_icon, policy_label = _traffic_light(policy_rate, "policy_rate")
    real_icon, real_label = _traffic_light(real_rate, "real_rate")

    points = []

    if money_metrics.change_1y_pct is not None:
        if money_metrics.change_1y_pct < 6:
            points.append("✅ Rahamäärän kasvu on maltillista.")
        elif money_metrics.change_1y_pct < 10:
            points.append("⚠️ Rahamäärän kasvu on koholla.")
        else:
            points.append("🔴 Rahamäärä kasvaa nopeasti.")

    if inflation is not None:
        if 0 <= inflation < 3:
            points.append("✅ Inflaatio on hallinnassa.")
        elif inflation < 5:
            points.append("⚠️ Inflaatio on koholla.")
        else:
            points.append("🔴 Inflaatio on korkea.")

    if real_rate is not None:
        if real_rate > 0:
            points.append("✅ Reaalikorko on positiivinen.")
        elif real_rate > -1:
            points.append("⚠️ Reaalikorko on lähellä nollaa.")
        else:
            points.append("🔴 Reaalikorko on negatiivinen.")

    with st.container(border=True):
        top1, top2 = st.columns([1.4, 0.8])

        with top1:
            st.markdown(f"### {icon} {currency}: {status}")
            st.caption("Yhteenveto rahamäärästä, inflaatiosta ja korkotasosta.")

        with top2:
            st.metric("Makropisteet", f"{score} / 3")

        k1, k2, k3, k4, k5 = st.columns(5)

        with k1:
            st.caption("Broad money")
            st.markdown(f"### {fmt_money_supply(money_metrics.latest_value, currency)}")
            if money_metrics.latest_date is not None:
                st.caption(f"Päivä: {money_metrics.latest_date.date()}")

        with k2:
            st.caption("Rahamäärä 1 v")
            st.markdown(f"### {money_icon} {fmt_pct(money_metrics.change_1y_pct)}")
            st.caption(money_label)

        with k3:
            st.caption("Inflaatio")
            st.markdown(f"### {infl_icon} {fmt_pct(inflation)}")
            st.caption(infl_label)
            if inflation_date is not None and not pd.isna(inflation_date):
                st.caption(pd.to_datetime(inflation_date).strftime("%Y-%m"))

        with k4:
            st.caption("Ohjauskorko")
            st.markdown(f"### {policy_icon} {fmt_pct(policy_rate)}")
            st.caption(policy_label)
            if policy_rate_date is not None and not pd.isna(policy_rate_date):
                st.caption(pd.to_datetime(policy_rate_date).strftime("%Y-%m"))

        with k5:
            st.caption("Reaalikorko")
            st.markdown(f"### {real_icon} {fmt_pct(real_rate)}")
            st.caption(real_label)
            if real_rate_date is not None and not pd.isna(real_rate_date):
                st.caption(pd.to_datetime(real_rate_date).strftime("%Y-%m"))

        if points:
            st.markdown("**Tulkinta**")
            for p in points:
                st.write(f"• {p}")


def render_money_macro_tab(
    money_currency: str,
    years: int,
    load_currency_bundle,
) -> None:
    money_bundle = load_currency_bundle(money_currency, years=years)
    money = money_bundle["money"]
    macro = money_bundle["macro"]
    debug = money_bundle.get("debug", {})

    st.markdown(f"#### {money_currency} – rahamäärä, inflaatio ja korot")
    st.caption(
        "Näkymä kokoaa rahamäärän, inflaation, ohjauskoron ja reaalikoron samaan yhteenvetokorttiin. "
        "Kuvaajat ovat kortin alapuolella historiallisen kehityksen tarkastelua varten."
    )

    with st.expander("Mitä mittarit tarkoittavat?", expanded=False):
        st.write(
            "**Broad money** tarkoittaa laajaa rahamäärää taloudessa. "
            "Se sisältää käteisen, käyttötilit sekä muita helposti rahaksi muutettavia talletuksia ja likvidejä varoja."
        )
        st.write(
            "**Ohjauskorko** on keskuspankin keskeinen korkotaso, jolla se ohjaa rahan hintaa ja talouden aktiivisuutta."
        )
        st.write(
            "**Reaalikorko-proxy** on tässä yksinkertaistus: ohjauskorko miinus inflaatio. "
            "Positiivinen arvo viittaa kireämpään rahapolitiikkaan, negatiivinen kevyempään."
        )

    if not is_major_macro_currency(money_currency):
        st.info("Tälle valuutalle ei ole vielä riittävän laadukasta rahamäärä- ja makrodataa.")
        return

    money_metrics = change_metrics(money, "Date", "BroadMoney_LCU")

    if money is None or money.empty:
        st.info("Rahamäärädataa ei saatu tälle valuutalle.")
        _show_debug("Rahamäärädatan debug", debug.get("money"))
        return

    if macro is None or macro.empty:
        st.info("Inflaatio- tai korkodataa ei saatu tälle valuutalle.")
        _show_debug("Makrodatan debug", debug.get("macro"))
        return

    inf_val, inf_date = _latest_non_null(macro, "InflationCPI_Pct")
    pol_val, pol_date = _latest_non_null(macro, "PolicyRate_Pct")
    real_val, real_date = _latest_non_null(macro, "RealInterestRate_Pct")

    _render_money_macro_summary_card(
        currency=money_currency,
        money_metrics=money_metrics,
        inflation=inf_val,
        inflation_date=inf_date,
        policy_rate=pol_val,
        policy_rate_date=pol_date,
        real_rate=real_val,
        real_rate_date=real_date,
    )

    st.divider()
    st.markdown("### 📈 Kuvaajat")

    plot_df = money.melt(
        id_vars=["Date"],
        value_vars=["BroadMoney_GrowthPct"],
        var_name="Sarja",
        value_name="Arvo",
    ).dropna()

    if not plot_df.empty:
        plot_df = plot_df[plot_df["Date"] >= DISPLAY_START_DATE].copy()

        fig = px.line(
            plot_df,
            x="Date",
            y="Arvo",
            color="Sarja",
            title=f"{money_currency} – broad money -kasvu",
            labels={"Date": "Päivä", "Arvo": "%", "Sarja": ""},
        )
        st.plotly_chart(fig, use_container_width=True)

    macro_plot = macro.melt(
        id_vars=["Date"],
        value_vars=["InflationCPI_Pct", "PolicyRate_Pct", "RealInterestRate_Pct"],
        var_name="Sarja",
        value_name="Arvo",
    ).dropna()

    if not macro_plot.empty:
        macro_plot = macro_plot[macro_plot["Date"] >= DISPLAY_START_DATE].copy()

        fig = px.line(
            macro_plot,
            x="Date",
            y="Arvo",
            color="Sarja",
            title=f"{money_currency} – inflaatio, korko ja reaalikorko-proxy",
            labels={"Date": "Päivä", "Arvo": "%", "Sarja": ""},
        )
        st.plotly_chart(fig, use_container_width=True)

    _show_debug("Rahamäärädatan huomautus", debug.get("money"))
    _show_debug("Makrodatan huomautus", debug.get("macro"))


def render_currency_analysis_tab(
    years: int,
    load_currency_bundle,
    load_fx_series,
) -> None:
    st.markdown("### 🧠 Valuutta-analyysi")
    st.caption(
        "Analyysi vertailee valuuttoja valuuttakurssin, rahamäärän, inflaation ja korkotason perusteella. "
        "Mukana ovat valuutat, joille on saatu riittävän ajantasaiset avoimet data­lähteet."
    )

    analysis_currencies = ["USD", "EUR", "JPY", "CNY"]

    snapshots: list[dict] = []

    with st.spinner("Rakennetaan valuuttojen terveyskortteja…"):
        for code in analysis_currencies:
            if code not in CURRENCY_META:
                continue

            snap = _analysis_snapshot(code, years, load_currency_bundle)
            snapshots.append(snap)

    st.markdown("### 🧭 Valuuttojen terveysmittarit")

    cols = st.columns(2)

    for i, snap in enumerate(snapshots):
        code = snap["currency"]
        anchor = snap["anchor"]

        with cols[i % 2]:
            render_currency_health_card(
                currency=f"{code} / {anchor}",
                fx_metrics=snap["fx_metrics"],
                money_df=snap["money"],
                macro_df=snap["macro"],
                show_title=False,
            )

    st.divider()

    _render_currency_health_ranking(snapshots)

    st.divider()

    _render_currency_comparison_table(snapshots)

    st.divider()

    _render_multi_currency_interpretation(snapshots)

    

def currency_format_func(code: str) -> str:
    return f"{code} – {CURRENCY_META[code]['name']}"


def macro_currency_format_func(code: str) -> str:
    return f"{code} – {CURRENCY_META[code]['name']}"


def get_macro_currency_options() -> list[str]:
    return [c for c in ["USD", "EUR", "JPY", "CNY"] if c in CURRENCY_META]