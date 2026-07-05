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
from services.oil_demand import fetch_world_oil_demand_debug


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


def _summary_card(
    title: str,
    value: str,
    changes: list[tuple[str, float | None]],
) -> None:
    with st.container(border=True):
        st.markdown(f"### {title}")
        st.caption("Viimeisin arvo")
        st.markdown(f"## {value}")

        st.divider()

        for label, pct in changes:
            icon = "↗" if pct is not None and not pd.isna(pct) and pct >= 0 else "↘"
            color = "#15803d" if pct is not None and not pd.isna(pct) and pct >= 0 else "#b91c1c"

            st.markdown(
                f"""
                <div style="
                    display:flex;
                    justify-content:space-between;
                    align-items:center;
                    padding:0.45rem 0;
                    border-bottom:1px solid #e5e7eb;
                ">
                    <span style="color:#6b7280;">{icon} {label}</span>
                    <span style="color:{color}; font-weight:700;">{_fmt_pct(pct)}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


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


def _pct_vs_period(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    *,
    months: int = 0,
    years: int = 0,
) -> float | None:

    if df is None or df.empty:
        return None

    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=[date_col, value_col]).sort_values(date_col)

    if d.empty:
        return None

    latest_date = d.iloc[-1][date_col]
    latest_value = float(d.iloc[-1][value_col])

    target = latest_date - pd.DateOffset(months=months, years=years)

    prev = d[d[date_col] <= target]

    if prev.empty:
        return None

    previous_value = float(prev.iloc[-1][value_col])

    return _pct(latest_value, previous_value)



def _status_from_pct(value: float | None) -> tuple[str, str]:
    if value is None or pd.isna(value):
        return "⚪", "Ei dataa"
    if value >= 10:
        return "🟢", "Vahva nousu"
    if value >= 2:
        return "🟢", "Nousussa"
    if value > -2:
        return "🟡", "Vakaa"
    if value > -10:
        return "🟠", "Laskussa"
    return "🔴", "Selvä lasku"


def _render_analysis_signal_cards(
    brent_yoy: float | None,
    usa_inventory_yoy: float | None,
    oecd_inventory_yoy: float | None,
    production_yoy: float | None,
) -> None:
    brent_icon, brent_status = _status_from_pct(brent_yoy)
    prod_icon, prod_status = _status_from_pct(production_yoy)

    if usa_inventory_yoy is None and oecd_inventory_yoy is None:
        inventory_icon, inventory_status = "⚪", "Ei dataa"
    else:
        avg_inventory = pd.Series([usa_inventory_yoy, oecd_inventory_yoy]).dropna().mean()
        if avg_inventory < -5:
            inventory_icon, inventory_status = "🟠", "Varastot supistuvat"
        elif avg_inventory > 5:
            inventory_icon, inventory_status = "🟢", "Varastot kasvavat"
        else:
            inventory_icon, inventory_status = "🟡", "Varastot melko vakaat"

    st.markdown("### 📌 Tilannekuva")

    c1, c2, c3 = st.columns(3)

    with c1:
        with st.container(border=True):
            st.markdown(f"### {brent_icon} Brent")
            st.markdown(f"**Tila:** {brent_status}")
            st.markdown(_fmt_pct(brent_yoy), unsafe_allow_html=True)
            st.caption("Brent-raakaöljyn 1 vuoden muutos.")

    with c2:
        with st.container(border=True):
            st.markdown(f"### {prod_icon} Tuotanto")
            st.markdown(f"**Tila:** {prod_status}")
            st.markdown(_fmt_pct(production_yoy), unsafe_allow_html=True)
            st.caption("Suurimpien tuottajamaiden keskimääräinen vuosimuutos.")

    with c3:
        with st.container(border=True):
            st.markdown(f"### {inventory_icon} Varastot")
            st.markdown(f"**Tila:** {inventory_status}")
            st.caption(f"USA 1 v: {_fmt_pct(usa_inventory_yoy)}")
            st.caption(f"OECD 1 v: {_fmt_pct(oecd_inventory_yoy)}")


def _render_oil_analysis():
    st.subheader("🧠 Öljy- ja polttoaineanalyysi")

    with st.spinner("Haetaan analyysidataa..."):
        oil_df, oil_msg = fetch_price_history_debug("BZ=F", period="5y")
        prod_df, prod_msg = fetch_oil_production_debug()
        us_hist_df, us_hist_msg = fetch_us_crude_inventory_history_debug(years=10)
        oecd_hist_df, oecd_hist_msg = fetch_oecd_petroleum_stocks_history_debug(years=10)

    brent_yoy = None
    production_yoy = None
    usa_inventory_yoy = None
    oecd_inventory_yoy = None

    if oil_df is not None and not oil_df.empty:
        brent_yoy = _pct_vs_period(oil_df, "Date", "Close", years=1)

    if prod_df is not None and not prod_df.empty:
        prod = prod_df.copy()
        prod["Year"] = pd.to_numeric(prod["Year"], errors="coerce")
        prod["Value"] = pd.to_numeric(prod["Value"], errors="coerce")
        prod = prod.dropna(subset=["Country", "Year", "Value"])
        prod["Year"] = prod["Year"].astype(int)

        latest_year = int(prod["Year"].max())
        prev_year = latest_year - 1

        latest_top = (
            prod[(prod["Year"] == latest_year) & (prod["Value"] > 0)]
            .sort_values("Value", ascending=False)
            .head(5)
        )

        prev = (
            prod[prod["Year"] == prev_year][["Country", "Value"]]
            .rename(columns={"Value": "PrevValue"})
        )

        merged = latest_top.merge(prev, on="Country", how="left")
        merged["YoY"] = (merged["Value"] / merged["PrevValue"] - 1.0) * 100.0

        if merged["YoY"].notna().any():
            production_yoy = float(merged["YoY"].mean())

    if us_hist_df is not None and not us_hist_df.empty:
        usa_inventory_yoy = _pct_vs_period(us_hist_df, "Date", "Value", years=1)

    if oecd_hist_df is not None and not oecd_hist_df.empty:
        oecd_inventory_yoy = _pct_vs_period(oecd_hist_df, "Date", "Value", years=1)

    _render_analysis_signal_cards(
        brent_yoy=brent_yoy,
        usa_inventory_yoy=usa_inventory_yoy,
        oecd_inventory_yoy=oecd_inventory_yoy,
        production_yoy=production_yoy,
    )

    st.divider()

    parts = []

    if brent_yoy is not None:
        if brent_yoy > 10:
            parts.append("Brent-raakaöljyn hinta on noussut selvästi vuoden aikana, mikä voi viitata vahvempaan kysyntään, tarjontarajoitteisiin tai geopoliittisiin riskipreemioihin.")
        elif brent_yoy < -10:
            parts.append("Brent-raakaöljyn hinta on laskenut selvästi vuoden aikana, mikä voi kertoa kysynnän heikkenemisestä tai tarjonnan riittävyydestä.")
        else:
            parts.append("Brent-raakaöljyn hinta on vuoden tasolla melko maltillisessa muutoksessa.")

    if production_yoy is not None:
        if production_yoy > 2:
            parts.append("Suurimpien tuottajamaiden tuotanto on kasvanut, mikä voi lisätä tarjontaa ja hillitä hintapaineita.")
        elif production_yoy < -2:
            parts.append("Suurimpien tuottajamaiden tuotanto on supistunut, mikä voi kiristää tarjontaa ja tukea öljyn hintaa.")
        else:
            parts.append("Suurimpien tuottajamaiden tuotanto on pysynyt melko vakaana.")

    inventory_values = [x for x in [usa_inventory_yoy, oecd_inventory_yoy] if x is not None and not pd.isna(x)]

    if inventory_values:
        avg_inventory = float(pd.Series(inventory_values).mean())

        if avg_inventory < -5:
            parts.append("Varastot ovat supistuneet vuoden aikana, mikä voi viitata markkinan kiristymiseen.")
        elif avg_inventory > 5:
            parts.append("Varastot ovat kasvaneet vuoden aikana, mikä voi kertoa tarjonnan runsaudesta tai kysynnän vaimeudesta.")
        else:
            parts.append("Varastot ovat vuoden tasolla melko vakaat, joten varastodata ei yksin viittaa poikkeuksellisen kireään tai löysään markkinaan.")

    if brent_yoy is not None and inventory_values:
        avg_inventory = float(pd.Series(inventory_values).mean())

        if brent_yoy > 5 and avg_inventory < 0:
            parts.append("Hinnan nousu yhdessä laskevien varastojen kanssa tukee tulkintaa kireämmästä öljymarkkinasta.")
        elif brent_yoy < -5 and avg_inventory > 0:
            parts.append("Hinnan lasku yhdessä kasvavien varastojen kanssa viittaa pehmeämpään markkinatilanteeseen.")
        elif brent_yoy > 5 and avg_inventory > 0:
            parts.append("Hinta on noussut, vaikka varastot ovat kasvaneet. Tämä voi viitata siihen, että markkina hinnoittelee muita tekijöitä, kuten geopoliittisia riskejä.")
    
    if not parts:
        parts.append("Analyysia ei voitu muodostaa, koska keskeisiä öljydataeriä puuttuu.")

    st.markdown("### 🧠 Öljymarkkina-analyysi")

    with st.container(border=True):
        st.write(" ".join(parts))

    st.info(
        "Tämä ei ole sijoitussuositus. Öljyn ja polttoaineiden hinnat voivat muuttua nopeasti "
        "kysynnän, tarjonnan, valuuttakurssien, verotuksen, varastojen ja geopoliittisten riskien vuoksi."
    )

    for msg in [oil_msg, prod_msg, us_hist_msg, oecd_hist_msg]:
        if msg:
            _show_source_message(msg)


def _render_price_tab():
    st.subheader("💵 Öljyn hinta ja Suomen polttoainehinnat")

    # ----------------------------------------------------------
    # Brent
    # ----------------------------------------------------------

    with st.spinner("Haetaan Brent-raakaöljyn markkinadataa..."):
        oil_df, oil_msg = fetch_price_history_debug("BZ=F", period="5y")

    # ----------------------------------------------------------
    # Polttoaineet
    # ----------------------------------------------------------

    with st.spinner("Haetaan Suomen polttoainehintadataa..."):
        fuel_df, fuel_msg = fetch_finland_fuel_prices_debug(years=5)

    if oil_df.empty or fuel_df.empty:
        st.error("Hintadataa ei saatu haettua.")
        return

    oil_df = (
        oil_df.dropna(subset=["Date", "Close"])
        .sort_values("Date")
        .reset_index(drop=True)
    )

    fuel_df = (
        fuel_df.dropna(subset=["Date", "Value"])
        .sort_values(["Fuel", "Date"])
        .reset_index(drop=True)
    )

    latest_brent = float(oil_df.iloc[-1]["Close"])
    brent_yoy = _pct_vs_year_ago(oil_df, "Date", "Close")

    latest_rows = (
        fuel_df.sort_values("Date")
        .groupby("Fuel", as_index=False)
        .tail(1)
    )

    fuel_cards = {}

    for _, row in latest_rows.iterrows():
        fuel = row["Fuel"]

        hist = fuel_df[fuel_df["Fuel"] == fuel]

        fuel_cards[fuel] = {
            "value": row["Value"],
            "pct": _pct_vs_year_ago(hist, "Date", "Value"),
        }

    st.markdown("### 💶 Hintayhteenveto")

    c1, c2, c3, _ = st.columns([0.28, 0.28, 0.28, 0.16])

    with c1:
        _summary_card(
            "🛢 Brent",
            f"{latest_brent:.2f} USD",
            [
                ("1 vuosi", brent_yoy),
            ],
        )

    with c2:
        gasoline_key = None

        for name in fuel_cards:
            name_lower = str(name).lower()
            if "95" in name_lower or "bensiini" in name_lower or "petrol" in name_lower:
                gasoline_key = name
                break

        if gasoline_key:
            _summary_card(
                "⛽ Bensiini",
                f"{fuel_cards[gasoline_key]['value']:.2f} €/l",
                [
                    ("1 vuosi", fuel_cards[gasoline_key]["pct"]),
                ],
            )
        else:
            st.warning("Bensiinikorttia ei voitu muodostaa, koska 95E10/bensiini-riviä ei löytynyt datasta.")

    with c3:
        diesel_key = None

        for name in fuel_cards:
            if "Diesel" in name:
                diesel_key = name
                break

        if diesel_key:
            _summary_card(
                "🚚 Diesel",
                f"{fuel_cards[diesel_key]['value']:.2f} €/l",
                [
                    ("1 vuosi", fuel_cards[diesel_key]["pct"]),
                ],
            )

    st.divider()

    st.caption(
        f"Viimeisin markkinadata: "
        f"{max(oil_df['Date'].max(), fuel_df['Date'].max()).date()}"
    )

    # ----------------------------------------------------------
    # Brent-kuvaaja
    # ----------------------------------------------------------

    fig = px.line(
        oil_df,
        x="Date",
        y="Close",
        title="Brent-raakaöljyn hinta (USD/barreli)",
        labels={
            "Date": "Päivä",
            "Close": "USD/barreli",
        },
    )

    fig.update_layout(
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ----------------------------------------------------------
    # Polttoainekuvaaja
    # ----------------------------------------------------------

    fig = px.line(
        fuel_df,
        x="Date",
        y="Value",
        color="Fuel",
        title="Suomen polttoainehinnat",
        labels={
            "Date": "Päivä",
            "Value": "€/l",
            "Fuel": "Polttoaine",
        },
    )

    fig.update_layout(
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

    if oil_msg:
        _show_source_message(oil_msg)

    if fuel_msg:
        _show_source_message(fuel_msg)


def _render_production_tab():
    st.subheader("🌍 Öljyntuotanto ja kysyntä")

    with st.spinner("Haetaan öljyntuotantodataa..."):
        prod_df, msg = fetch_oil_production_debug()

    if prod_df is None or prod_df.empty:
        st.error("Öljyntuotantodataa ei saatu.")
        return

    prod_df = prod_df.copy()
    prod_df["Year"] = pd.to_numeric(prod_df["Year"], errors="coerce")
    prod_df["Value"] = pd.to_numeric(prod_df["Value"], errors="coerce")
    prod_df = prod_df.dropna(subset=["Country", "Year", "Value"])
    prod_df["Year"] = prod_df["Year"].astype(int)

    latest_year = int(prod_df["Year"].max())
    prev_year = latest_year - 1

    latest_prod = (
        prod_df[(prod_df["Year"] == latest_year) & (prod_df["Value"] > 0)]
        .sort_values("Value", ascending=False)
        .head(10)
        .copy()
    )

    prev_prod = (
        prod_df[prod_df["Year"] == prev_year][["Country", "Value"]]
        .rename(columns={"Value": "PrevValue"})
    )

    latest_cards = latest_prod.merge(prev_prod, on="Country", how="left")
    latest_cards["YoYChangePct"] = (
        (latest_cards["Value"] / latest_cards["PrevValue"] - 1.0) * 100.0
    )

    st.markdown("### 📊 Tuotannon yhteenveto")
    st.caption("Yksikkö: tuhatta barrelia päivässä (kb/d). Muutos verrattuna edelliseen vuoteen.")

    c1, c2, c3, _ = st.columns([0.28, 0.28, 0.28, 0.16])

    for col, (_, row) in zip([c1, c2, c3], latest_cards.head(3).iterrows()):
        with col:
            _summary_card(
                title=row["Country"],
                value=f"{row['Value']:,.0f} kb/d".replace(",", " "),
                changes=[
                    ("1 vuosi", row["YoYChangePct"] if pd.notna(row["YoYChangePct"]) else None),
                ],
            )

    st.divider()

    fig_top = px.bar(
        latest_prod,
        x="Country",
        y="Value",
        title=f"Suurimmat öljyntuottajamaat ({latest_year})",
        labels={"Country": "Maa", "Value": "Tuotanto (kb/d)"},
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
            labels={"Year": "Vuosi", "Value": "Tuotanto (kb/d)", "Country": "Maa"},
        )
        fig_trend.update_layout(hovermode="x unified")
        st.plotly_chart(fig_trend, use_container_width=True)

    if msg:
        _show_source_message(msg)

    st.divider()

    st.markdown("### 🌍 Maailman öljyn kysyntä")

    with st.spinner("Haetaan maailman öljyn kysyntädataa..."):
        demand_df, demand_msg = fetch_world_oil_demand_debug()

    if demand_df is None or demand_df.empty:
        st.warning("Maailman öljyn kysyntädataa ei saatu.")
        if demand_msg:
            _show_source_message(demand_msg)
        return

    demand_df = demand_df.copy()
    demand_df["Year"] = pd.to_numeric(demand_df["Year"], errors="coerce")
    demand_df["Value"] = pd.to_numeric(demand_df["Value"], errors="coerce")
    demand_df = demand_df.dropna(subset=["Year", "Value"]).sort_values("Year")
    demand_df["Year"] = demand_df["Year"].astype(int)

    demand_yoy = _pct_vs_period(
        demand_df,
        "Year",
        "Value",
        years=1,
    )

    demand_5y = _pct_vs_period(
        demand_df,
        "Year",
        "Value",
        years=5,
    )

    demand_status = "Kasvaa" if demand_yoy is not None and demand_yoy > 0 else "Laskee"

    c1, c2 = st.columns([0.28, 0.72])

    with c1:
        _summary_card(
            title="🌍 Kysyntä",
            value=demand_status,
            changes=[
                ("1 vuosi", demand_yoy),
                ("5 vuotta", demand_5y),
            ],
        )

    with c2:
        fig = px.line(
            demand_df,
            x="Year",
            y="Value",
            title="Maailman öljyn kysynnän kehitys",
            labels={
                "Year": "Vuosi",
                "Value": "Energiankulutus",
            },
        )
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Huomio: Tämä sarja kuvaa maailman öljyn energiankulutusta "
        "(Our World in Data), ei suoraan öljyn kulutusta miljoonina barreleina päivässä. "
        "Sarja soveltuu erityisesti pitkän aikavälin kysyntätrendin tarkasteluun."
    )

    if demand_msg:
        _show_source_message(demand_msg)


def _render_inventory_tab():
    st.subheader("📦 Öljyvarastot")

    # ----------------------------------------------------------
    # USA varastot
    # ----------------------------------------------------------

    with st.spinner("Haetaan USA:n öljyvarastotietoa..."):
        us_df, us_msg = fetch_us_crude_inventory_debug()

    with st.spinner("Haetaan USA:n öljyvarastohistoriaa..."):
        us_hist_df, us_hist_msg = fetch_us_crude_inventory_history_debug(years=10)

    # ----------------------------------------------------------
    # OECD varastot
    # ----------------------------------------------------------

    with st.spinner("Haetaan OECD-varastotietoa..."):
        oecd_df, oecd_msg = fetch_oecd_petroleum_stocks_debug()

    with st.spinner("Haetaan OECD-varastohistoriaa..."):
        oecd_hist_df, oecd_hist_msg = fetch_oecd_petroleum_stocks_history_debug(years=10)

    st.markdown("### 📌 Varastojen yhteenveto")

    c1, c2, _ = st.columns([0.34, 0.34, 0.32])

    # USA kortti
    with c1:
        if us_df is None or us_df.empty:
            st.warning("USA-varastokorttia ei voitu muodostaa.")
        else:
            us_row = us_df.iloc[-1]

            _summary_card(
                title="🇺🇸 USA raakaöljy",
                value=f"{us_row['Value']:.1f} milj. bbl",
                changes=[
                    ("1 kk", _pct_vs_period(us_hist_df, "Date", "Value", months=1)),
                    ("1 vuosi", _pct_vs_period(us_hist_df, "Date", "Value", years=1)),
                    ("5 vuotta", _pct_vs_period(us_hist_df, "Date", "Value", years=5)),
                ],
            )

            st.caption(f"Viimeisin data: {pd.to_datetime(us_row['Date']).strftime('%d.%m.%Y')}")

    # OECD kortti
    with c2:
        if oecd_df is None or oecd_df.empty:
            st.warning("OECD-varastokorttia ei voitu muodostaa.")
        else:
            oecd_row = oecd_df.iloc[-1]

            _summary_card(
                title="🌐 OECD petroleum",
                value=f"{oecd_row['Value'] / 1000:.3f} mrd bbl",
                changes=[
                    ("1 kk", _pct_vs_period(oecd_hist_df, "Date", "Value", months=1)),
                    ("1 vuosi", _pct_vs_period(oecd_hist_df, "Date", "Value", years=1)),
                    ("5 vuotta", _pct_vs_period(oecd_hist_df, "Date", "Value", years=5)),
                ],
            )

            if "DateLabel" in oecd_row:
                st.caption(f"Viimeisin data: {oecd_row['DateLabel']}")

    st.divider()

    # ----------------------------------------------------------
    # USA kuvaaja
    # ----------------------------------------------------------

    st.markdown("### 🇺🇸 USA: kaupalliset raakaöljyvarastot")

    if us_hist_df is None or us_hist_df.empty:
        st.warning("USA-varastojen historiallista dataa ei saatu.")
        _show_source_message(us_hist_msg, "USA-varastohistorian tekninen virhe")
    else:
        fig = px.line(
            us_hist_df,
            x="Date",
            y="Value",
            title="USA:n kaupalliset raakaöljyvarastot (10 v)",
            labels={
                "Date": "Päivä",
                "Value": "milj. bbl",
            },
        )

        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ----------------------------------------------------------
    # OECD kuvaaja
    # ----------------------------------------------------------

    st.markdown("### 🌐 OECD: petroleum stocks")

    if oecd_hist_df is None or oecd_hist_df.empty:
        st.warning("OECD-varastojen historiallista dataa ei saatu.")
        _show_source_message(oecd_hist_msg, "OECD-varastohistorian tekninen virhe")
    else:
        plot_df = oecd_hist_df.copy()
        plot_df["Value_Billion"] = plot_df["Value"] / 1000.0

        fig = px.line(
            plot_df,
            x="Date",
            y="Value_Billion",
            title="OECD petroleum stocks (10 v)",
            labels={
                "Date": "Päivä",
                "Value_Billion": "mrd bbl",
            },
        )

        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    if us_msg:
        _show_source_message(us_msg, "USA-varastojen huomautus")

    if oecd_msg:
        _show_source_message(oecd_msg, "OECD-varastojen huomautus")


def render():
    st.subheader("🛢 Öljy & polttoaineet")
    st.caption(
        "Lähteet: Yahoo Finance (Brent), Tilastokeskus / Traficom (polttoaineet), "
        "EIA (USA varastot, OECD varastot) ja Our World in Data (tuotanto)."
    )

    tab_price, tab_production, tab_inventory, tab_analysis = st.tabs(
        ["💵 Hinta", "🌍 Tuotanto", "📦 Varastot", "🧠 Analyysi"]
    )

    with tab_price:
        _render_price_tab()

    with tab_production:
        _render_production_tab()

    with tab_inventory:
        _render_inventory_tab()

    with tab_analysis:
        _render_oil_analysis()