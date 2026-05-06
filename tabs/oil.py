from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from services.market_data import fetch_price_history_debug
from services.oil_inventory import (
    fetch_oecd_petroleum_stocks_debug,
    fetch_oecd_petroleum_stocks_history_debug,
    fetch_us_crude_inventory_debug,
    fetch_us_crude_inventory_history_debug,
)
from services.oil_production import fetch_oil_production_debug
from services.oil_stats import fetch_finland_fuel_prices_debug


SHOW_DEBUG_DETAILS = False


def _show_source_message(message: str | None, title: str = "Tekninen huomautus"):
    if not message:
        return
    if SHOW_DEBUG_DETAILS:
        with st.expander(title, expanded=False):
            st.code(message)
    else:
        st.warning(message)


def _pct(now: float | None, then: float | None) -> float | None:
    if now is None or then is None or then == 0:
        return None
    return (now / then - 1.0) * 100.0


def _fmt_pct(x: float | None, decimals: int = 1) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x:+.{decimals}f} %"


def _metric_delta(x: float | None, decimals: int = 1) -> str | None:
    if x is None or pd.isna(x):
        return None
    return f"{x:+.{decimals}f} %"


def _pct_vs_year_ago(df: pd.DataFrame, date_col: str, value_col: str) -> float | None:
    if df is None or df.empty:
        return None

    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=[date_col, value_col]).sort_values(date_col)

    if d.empty:
        return None

    latest_date = d.iloc[-1][date_col]
    latest_val = float(d.iloc[-1][value_col])

    target_date = latest_date - pd.DateOffset(years=1)
    prev = d[d[date_col] <= target_date]

    if prev.empty:
        return None

    prev_val = float(prev.iloc[-1][value_col])
    return _pct(latest_val, prev_val)


def _render_price_tab():
    st.subheader("💵 Öljyn hinta ja Suomen polttoainehinnat")

    st.markdown("### Brent-raakaöljy")

    with st.spinner("Haetaan Brent-raakaöljyn markkinadataa..."):
        oil_df, oil_msg = fetch_price_history_debug("BZ=F", period="5y")

    if oil_df.empty:
        st.error("Brent-dataa ei saatu haettua verkosta.")
        _show_source_message(oil_msg, "Brent-haun tekninen virhe")
    else:
        oil_df = oil_df.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)

        latest = float(oil_df.iloc[-1]["Close"])
        year_pct = _pct_vs_year_ago(oil_df, "Date", "Close")

        st.metric(
            "Viimeisin hinta",
            f"{latest:,.2f} USD".replace(",", " "),
            f"{_metric_delta(year_pct, 1)} (1 v)" if year_pct is not None else None,
        )

        st.caption(f"Viimeisin markkinadata: {oil_df.iloc[-1]['Date'].date()}")

        fig = px.line(
            oil_df,
            x="Date",
            y="Close",
            title="Brent-raakaöljyn hinta (USD / barreli, 5 v)",
            labels={"Date": "Aika", "Close": "USD / barreli"},
        )
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        if oil_msg:
            _show_source_message(oil_msg, "Brent-haun huomautus")

    st.divider()

    st.markdown("### Suomen polttoainehinnat")

    with st.spinner("Haetaan Suomen polttoainehintadataa..."):
        fuel_df, fuel_msg = fetch_finland_fuel_prices_debug(years=5)

    if fuel_df.empty:
        st.error("Polttoainehintadataa ei saatu haettua verkosta.")
        _show_source_message(fuel_msg, "Polttoainehintojen tekninen virhe")
    else:
        fuel_df = fuel_df.dropna(subset=["Date", "Value"]).sort_values(["Fuel", "Date"]).reset_index(drop=True)

        fuel_with_yoy = (
            fuel_df.sort_values("Date")
            .groupby("Fuel", group_keys=False)
            .apply(lambda g: g.assign(YearlyChangePct=_pct_vs_year_ago(g, "Date", "Value")))
            .reset_index(drop=True)
        )

        latest_rows = (
            fuel_with_yoy.sort_values("Date")
            .groupby("Fuel", as_index=False)
            .tail(1)
            .sort_values("Fuel")
        )

        cols = st.columns(len(latest_rows))
        for i, (_, row) in enumerate(latest_rows.iterrows()):
            delta = row.get("YearlyChangePct")

            cols[i].metric(
                row["Fuel"],
                f"{row['Value']:.2f} €/l",
                f"{_metric_delta(delta, 1)} (1 v)" if delta is not None and not pd.isna(delta) else None,
            )

        latest_date = fuel_df["Date"].max()
        st.caption(f"Viimeisin polttoainehintadata: {latest_date.date()}")

        fig = px.line(
            fuel_df,
            x="Date",
            y="Value",
            color="Fuel",
            title="Polttoaineiden hintakehitys Suomessa (5 v)",
            labels={"Date": "Aika", "Value": "€/l", "Fuel": "Polttoaine"},
        )
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        if fuel_msg:
            _show_source_message(fuel_msg, "Polttoainehintojen huomautus")


def _render_production_tab():
    st.subheader("🌍 Raakaöljyn tuotanto maittain")

    with st.spinner("Haetaan raakaöljyn tuotantodataa..."):
        prod_df, prod_msg = fetch_oil_production_debug()

    if prod_df.empty:
        st.error("Raakaöljyn tuotantodataa ei saatu haettua verkosta.")
        _show_source_message(prod_msg, "Öljyntuotannon tekninen virhe")
        return

    prod_df = prod_df.dropna(subset=["Country", "Year", "Value"]).copy()
    latest_year = int(prod_df["Year"].max())

    latest_prod = (
        prod_df[(prod_df["Year"] == latest_year) & (prod_df["Value"] > 0)]
        .sort_values("Value", ascending=False)
        .head(10)
        .copy()
    )

        # KPI-kortit: viimeisimmät tuotantomäärät ja muutos edellisvuodesta
    prev_year = latest_year - 1

    prev_prod = (
        prod_df[prod_df["Year"] == prev_year][["Country", "Value"]]
        .rename(columns={"Value": "PrevValue"})
    )

    latest_cards = latest_prod.merge(prev_prod, on="Country", how="left")
    latest_cards["YoYChangePct"] = (
        (latest_cards["Value"] / latest_cards["PrevValue"] - 1.0) * 100.0
    )

    st.markdown("### 📌 Tuotannon yhteenveto")

    st.caption("Yksikkö: tuhatta barrelia päivässä (kb/d). Muutos verrattuna edelliseen vuoteen.")
    
    cols = st.columns(5)

    for i, (_, row) in enumerate(latest_cards.head(6).iterrows()):
        with cols[i % 5]:
            delta = None
            if pd.notna(row.get("YoYChangePct")):
                delta = f"{row['YoYChangePct']:+.1f} % (1 v)"

            st.metric(
                label=row["Country"],
                value=f"{row['Value']:,.0f} kb/d".replace(",", " "),
                delta=delta,
            )

    fig_top = px.bar(
        latest_prod,
        x="Country",
        y="Value",
        title=f"Suurimmat öljyntuottajamaat ({latest_year})",
        labels={"Country": "Maa", "Value": "Tuotanto"},
    )
    st.plotly_chart(fig_top, use_container_width=True)

    min_year = latest_year - 19
    recent_prod_df = prod_df[prod_df["Year"] >= min_year].copy()

    selectable_countries = (
        recent_prod_df[recent_prod_df["Year"] == latest_year]
        .loc[lambda d: d["Value"] > 0, "Country"]
        .dropna()
        .sort_values()
        .unique()
        .tolist()
    )

    default_countries = [
        c for c in ["United States", "Saudi Arabia", "Russia", "Canada", "Iraq", "Iran"]
        if c in selectable_countries
    ]

    selected_countries = st.multiselect(
        "Valitse maat tuotannon trendikuvaajaan",
        options=selectable_countries,
        default=default_countries,
    )

    if selected_countries:
        trend_df = recent_prod_df[recent_prod_df["Country"].isin(selected_countries)].copy()

        fig_trend = px.line(
            trend_df,
            x="Year",
            y="Value",
            color="Country",
            title=f"Raakaöljyn tuotannon kehitys valituissa maissa ({min_year}–{latest_year})",
            labels={"Year": "Vuosi", "Value": "Tuotanto", "Country": "Maa"},
        )
        fig_trend.update_layout(hovermode="x unified")
        st.plotly_chart(fig_trend, use_container_width=True)

    if prod_msg:
        _show_source_message(prod_msg, "Öljyntuotannon huomautus")


def _render_inventory_tab():
    st.subheader("📦 Öljyvarastot")

    with st.spinner("Haetaan USA:n öljyvarastotietoa..."):
        us_df, us_msg = fetch_us_crude_inventory_debug()

    if us_df.empty:
        st.error("USA:n öljyvarastotietoa ei saatu.")
        _show_source_message(us_msg, "USA-varastojen tekninen virhe")
    else:
        st.markdown("### USA: kaupalliset raakaöljyvarastot")

        row = us_df.iloc[-1]
        c1, c2 = st.columns(2)

        c1.metric(
            "Varasto nyt",
            f"{row['Value']:.1f} milj. bbl",
            f"{row['Change']:+.1f} milj. bbl",
        )
        c2.metric("Muutos % (1 vko)", _fmt_pct(row["ChangePct"], 2))

        st.caption(f"Viimeisin varastodata: {pd.to_datetime(row['Date']).strftime('%d.%m.%Y')}")

        hist_df, hist_msg = fetch_us_crude_inventory_history_debug(years=10)

        if hist_df.empty:
            st.warning("USA-varastojen historiallista dataa ei saatu.")
            _show_source_message(hist_msg, "USA-varastohistorian tekninen virhe")
        else:
            fig = px.line(
                hist_df,
                x="Date",
                y="Value",
                title="USA:n kaupalliset raakaöljyvarastot (10 v)",
                labels={"Date": "Aika", "Value": "milj. bbl"},
            )
            fig.update_layout(hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

        if us_msg:
            _show_source_message(us_msg, "USA-varastojen huomautus")

    st.divider()

    with st.spinner("Haetaan OECD-varastotietoa..."):
        oecd_df, oecd_msg = fetch_oecd_petroleum_stocks_debug()

    if oecd_df.empty:
        st.warning("OECD-varastotietoa ei saatu.")
        _show_source_message(oecd_msg, "OECD-varastojen tekninen virhe")
    else:
        st.markdown("### OECD: petroleum stocks")

        row = oecd_df.iloc[-1]
        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Varasto nyt",
            f"{row['Value'] / 1000:.3f} mrd bbl",
            f"{row['Change'] / 1000:+.3f} mrd bbl",
        )
        c2.metric("Muutos % (1 kk)", _fmt_pct(row["ChangePct"], 2))
        c3.metric("Muutos % (1 v)", _fmt_pct(row["YoYChangePct"], 2))

        st.caption(f"Viimeisin varastodata: {row['DateLabel']}")

        hist_df, hist_msg = fetch_oecd_petroleum_stocks_history_debug(years=10)

        if hist_df.empty:
            st.warning("OECD-varastojen historiallista dataa ei saatu.")
            _show_source_message(hist_msg, "OECD-varastohistorian tekninen virhe")
        else:
            plot_df = hist_df.copy()
            plot_df["Value_Billion"] = plot_df["Value"] / 1000.0

            fig = px.line(
                plot_df,
                x="Date",
                y="Value_Billion",
                title="OECD petroleum stocks (10 v)",
                labels={"Date": "Aika", "Value_Billion": "mrd bbl"},
            )
            fig.update_layout(hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

        if oecd_msg:
            _show_source_message(oecd_msg, "OECD-varastojen huomautus")


def render():
    st.subheader("🛢 Öljy & polttoaineet")
    st.caption(
        "Lähteet: Yahoo Finance (Brent), Tilastokeskus / Traficom (polttoaineet), "
        "EIA (USA varastot, OECD varastot) ja Our World in Data (tuotanto)."
    )

    view = st.radio(
        "Valitse näkymä",
        ["💵 Hinta", "🌍 Tuotanto", "📦 Varastot"],
        horizontal=True,
        label_visibility="collapsed",
        key="oil_view",
    )

    st.divider()

    if view == "💵 Hinta":
        _render_price_tab()

    elif view == "🌍 Tuotanto":
        _render_production_tab()

    elif view == "📦 Varastot":
        _render_inventory_tab()