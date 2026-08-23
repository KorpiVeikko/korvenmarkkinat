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

from services.oil_demand_eia import fetch_world_oil_balance_eia_debug
from services.oil_production_jodi import fetch_jodi_crude_production_debug
from services.fuel_prices_weekly import (
    fetch_finland_weekly_fuel_prices_debug,
)
from services.fuel_forecast_xgb import (
    calculate_all_fuel_forecasts_xgb,
)

from services.diesel_v7 import (
    calculate_diesel_v7_forecast,
)



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


def _fmt_oil_rate(
    value: float | None,
    source_unit: str = "million_bbl_per_day",
    decimals: int = 1,
) -> str:
    if value is None or pd.isna(value):
        return "—"

    value = float(value)

    if source_unit == "kb_per_day":
        if abs(value) >= 1_000:
            formatted = f"{value / 1_000:.{decimals}f}".replace(".", ",")
            return f"{formatted} milj. bbl/pv"

        return f"{value:,.0f} kb/d".replace(",", " ")

    formatted = f"{value:.{decimals}f}".replace(".", ",")
    return f"{formatted} milj. bbl/pv"


def _fmt_fuel_price(
    value: float | None,
) -> str:
    if value is None or pd.isna(value):
        return "—"

    return (
        f"{float(value):.3f}"
        .replace(".", ",")
        + " €/l"
    )


def _fmt_cents(
    value: float | None,
) -> str:
    if value is None or pd.isna(value):
        return "—"

    return (
        f"{float(value):+.1f}"
        .replace(".", ",")
        + " snt/l"
    )


def _forecast_color(
    direction: str,
) -> str:
    if direction == "Nousupaine":
        return "#b91c1c"

    if direction == "Laskupaine":
        return "#15803d"

    return "#6b7280"



def _country_flag(country: str) -> str:
    flags = {
        "United States": "🇺🇸",
        "Saudi Arabia": "🇸🇦",
        "Russia": "🇷🇺",
        "Canada": "🇨🇦",
        "Iraq": "🇮🇶",
        "China": "🇨🇳",
        "Brazil": "🇧🇷",
        "United Arab Emirates": "🇦🇪",
        "Iran": "🇮🇷",
        "Kuwait": "🇰🇼",
        "Norway": "🇳🇴",
        "Mexico": "🇲🇽",
        "Kazakhstan": "🇰🇿",
        "Nigeria": "🇳🇬",
        "Libya": "🇱🇾",
    }

    return flags.get(country, "🛢")



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



def _render_forecast_tab():
    st.subheader("🔮 Suomen polttoaineiden hintaennuste")

    st.caption(
        "Neljän viikon ennuste perustuu Brent-raakaöljyn kehitykseen, "
        "Suomen polttoainehintojen omaan historiaan ja kausivaihteluun. "
        "Dieselin ennuste käyttää lisäksi markkinatilanteeseen mukautuvaa "
        "kalibrointia. Ennuste on mallipohjainen arvio, ei hintalupaus."
    )

    # ==========================================================
    # DATA
    # ==========================================================

    with st.spinner("Haetaan ennustemallin aineistoa..."):
        brent_df, brent_msg = fetch_price_history_debug(
            "BZ=F",
            period="10y",
        )

        weekly_fuel_df, weekly_fuel_msg = (
            fetch_finland_weekly_fuel_prices_debug(
                years=10,
            )
        )

    if brent_df is None or brent_df.empty:
        st.error("Brent-dataa ei saatu ennustetta varten.")
        _show_source_message(
            brent_msg,
            "Brent-datan huomautus",
        )
        return

    if weekly_fuel_df is None or weekly_fuel_df.empty:
        st.error(
            "Suomen viikoittaista polttoainehintadataa ei saatu."
        )
        _show_source_message(
            weekly_fuel_msg,
            "Polttoainedatan huomautus",
        )
        return

    # ==========================================================
    # 95E10 + DIESEL V6 -POHJA
    # ==========================================================

    try:
        (
            summaries,
            backtests,
            _importances,
            _contributions,
            messages,
        ) = calculate_all_fuel_forecasts_xgb(
            brent_df=brent_df,
            eurusd_df=None,
            weekly_fuel_df=weekly_fuel_df,
            us_inventory_df=None,
            crack_df=None,
            fuels=["95E10", "Diesel"],
            forecast_horizon_weeks=4,
        )
    except Exception as exc:
        st.error("Polttoaine-ennusteen muodostaminen epäonnistui.")
        if SHOW_DEBUG_DETAILS:
            st.exception(exc)
        return

    if not summaries:
        st.error("Polttoaine-ennusteita ei voitu muodostaa.")
        for message in messages:
            _show_source_message(message)
        return

    gasoline_summary = summaries.get("95E10")
    diesel_v6_summary = summaries.get("Diesel")
    diesel_v6_backtest = backtests.get("Diesel", pd.DataFrame())

    diesel_summary = None
    diesel_v7_backtest = pd.DataFrame()

    if diesel_v6_summary and not diesel_v6_backtest.empty:
        try:
            (
                diesel_summary,
                diesel_v7_backtest,
                diesel_v7_message,
            ) = calculate_diesel_v7_forecast(
                v6_summary=diesel_v6_summary,
                diesel_backtest_df=diesel_v6_backtest,
                weekly_fuel_df=weekly_fuel_df,
            )

            if diesel_v7_message:
                _show_source_message(
                    diesel_v7_message,
                    "Diesel-ennusteen tekninen huomautus",
                )

        except Exception as exc:
            if SHOW_DEBUG_DETAILS:
                st.exception(exc)
            diesel_summary = None

    if diesel_summary is None:
        # Turvallinen fallback: näkymä toimii edelleen V6:lla,
        # mutta käyttäjälle kerrotaan ettei mukautettua Diesel-ennustetta saatu.
        diesel_summary = diesel_v6_summary
        if diesel_summary:
            st.warning(
                "Dieselin mukautettua ennustetta ei saatu muodostettua. "
                "Näytetään perusennuste."
            )

    # ==========================================================
    # ENNUSTEKORTIT
    # ==========================================================

    st.markdown("### Neljän viikon näkymä")

    def render_forecast_card(
        column,
        *,
        title: str,
        icon: str,
        summary: dict | None,
        show_regime: bool = False,
    ) -> None:
        with column:
            if not summary:
                st.warning(
                    f"{title}-ennustetta ei voitu muodostaa."
                )
                return

            current_price = summary.get("latest_fuel_price")
            predicted_price = summary.get("predicted_price")
            change_cents = summary.get("predicted_change_cents")
            change_pct = summary.get("predicted_change_pct")
            direction = str(summary.get("direction", "Ei arviota"))
            confidence = str(summary.get("confidence", "Varovainen"))
            interval_low = summary.get("interval_low_price")
            interval_high = summary.get("interval_high_price")

            direction_color = _forecast_color(direction)

            with st.container(border=True):
                st.markdown(f"### {icon} {title}")
                st.caption("Nykyinen viikkohinta")
                st.markdown(f"## {_fmt_fuel_price(current_price)}")

                st.divider()

                st.markdown(
                    f"""
                    <div style="display:flex;justify-content:space-between;gap:1rem;padding:0.35rem 0;">
                        <span style="color:#6b7280;">4 viikon arvio</span>
                        <span style="font-weight:700;">{_fmt_fuel_price(predicted_price)}</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;gap:1rem;padding:0.35rem 0;">
                        <span style="color:#6b7280;">Arvioitu muutos</span>
                        <span style="font-weight:700;color:{direction_color};">{_fmt_cents(change_cents)}</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;gap:1rem;padding:0.35rem 0;">
                        <span style="color:#6b7280;">Muutos prosentteina</span>
                        <span style="font-weight:700;color:{direction_color};">{_fmt_pct(change_pct, 1)}</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;gap:1rem;padding:0.35rem 0;">
                        <span style="color:#6b7280;">Hintapaine</span>
                        <span style="font-weight:700;color:{direction_color};">{direction}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if interval_low is not None and interval_high is not None:
                    st.markdown(
                        f"""
                        <div style="display:flex;justify-content:space-between;gap:1rem;padding:0.35rem 0;">
                            <span style="color:#6b7280;">Arvioitu vaihteluväli</span>
                            <span style="font-weight:700;text-align:right;">
                                {_fmt_fuel_price(interval_low)} – {_fmt_fuel_price(interval_high)}
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown(
                    f"""
                    <div style="display:flex;justify-content:space-between;gap:1rem;padding:0.35rem 0;">
                        <span style="color:#6b7280;">Epävarmuus</span>
                        <span style="font-weight:700;">{confidence}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if show_regime:
                    st.divider()

                    trend = summary.get(
                        "trend_regime",
                        "Ei luokitusta",
                    )
                    volatility = summary.get(
                        "volatility_regime",
                        "Ei luokitusta",
                    )

                    st.caption("Nykyinen markkinatilanne")
                    st.markdown(
                        f"**Trendi:** {trend}  \n"
                        f"**Volatiliteetti:** {volatility}"
                    )

    c1, c2 = st.columns(2)

    render_forecast_card(
        c1,
        title="95E10",
        icon="⛽",
        summary=gasoline_summary,
        show_regime=False,
    )

    render_forecast_card(
        c2,
        title="Diesel",
        icon="🚚",
        summary=diesel_summary,
        show_regime=True,
    )

    # ==========================================================
    # KÄYTTÄJÄLLE HYÖDYLLINEN HISTORIAVERTAILU
    # ==========================================================

    st.divider()
    st.markdown("### Ennusteiden historiallinen osuvuus")

    selected_fuel = st.segmented_control(
        "Näytettävä polttoaine",
        options=["95E10", "Diesel"],
        default="Diesel",
        key="fuel_forecast_history_fuel",
        label_visibility="collapsed",
    )

    if selected_fuel == "Diesel":
        history_df = diesel_v7_backtest.copy()
        prediction_column = "PredictedPrice_V7"
    else:
        history_df = backtests.get("95E10", pd.DataFrame()).copy()
        prediction_column = "PredictedPrice"

    if (
        history_df is not None
        and not history_df.empty
        and prediction_column in history_df.columns
    ):
        chart_df = history_df[
            ["ForecastDate", "ActualPrice", prediction_column]
        ].copy()

        chart_df["ForecastDate"] = pd.to_datetime(
            chart_df["ForecastDate"],
            errors="coerce",
        )

        # Käyttäjänäkymässä riittää viimeiset noin kaksi vuotta.
        if not chart_df["ForecastDate"].dropna().empty:
            cutoff = chart_df["ForecastDate"].max() - pd.DateOffset(years=2)
            chart_df = chart_df.loc[
                chart_df["ForecastDate"] >= cutoff
            ]

        chart_long = chart_df.rename(
            columns={
                "ForecastDate": "Date",
                "ActualPrice": "Toteutunut",
                prediction_column: "Ennuste",
            }
        ).melt(
            id_vars="Date",
            value_vars=["Toteutunut", "Ennuste"],
            var_name="Sarja",
            value_name="Hinta",
        )

        fig = px.line(
            chart_long,
            x="Date",
            y="Hinta",
            color="Sarja",
            title=f"{selected_fuel}: ennuste vs toteutunut hinta",
            labels={
                "Date": "Ajankohta",
                "Hinta": "€/l",
                "Sarja": "Sarja",
            },
        )

        fig.update_layout(
            hovermode="x unified",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        if selected_fuel == "Diesel":
            mae = diesel_summary.get("walk_forward_mae_cents")
            observations = diesel_summary.get("test_observations")
        else:
            mae = gasoline_summary.get("walk_forward_mae_cents") if gasoline_summary else None
            observations = gasoline_summary.get("test_observations") if gasoline_summary else None

        note_parts = []
        if mae is not None and not pd.isna(mae):
            note_parts.append(
                f"historiallinen keskimääräinen absoluuttinen virhe noin {float(mae):.1f} snt/l"
            )
        if observations:
            note_parts.append(
                f"vertailussa {int(observations)} out-of-sample-havaintoa"
            )

        if note_parts:
            st.caption("; ".join(note_parts) + ".")

    else:
        st.info(
            "Historiallista ennustevertailua ei saatu muodostettua."
        )

    st.info(
        "Ennuste kuvaa mallin arvioimaa hintapainetta noin neljän viikon päähän. "
        "Polttoaineiden vähittäishintoihin vaikuttavat lisäksi muun muassa verotus, "
        "jakelumarginaalit ja paikallinen kilpailu, joita malli ei ennusta erikseen."
    )

    for message in messages:
        if message:
            _show_source_message(message)



def _render_oil_analysis():
    st.subheader("🧠 Öljy- ja polttoaineanalyysi")

    with st.spinner("Haetaan analyysidataa..."):
        oil_df, oil_msg = fetch_price_history_debug(
            "BZ=F",
            period="5y",
        )

        prod_df, prod_msg = fetch_jodi_crude_production_debug(
            years=3,
        )

        eia_df, eia_msg = fetch_world_oil_balance_eia_debug()

        us_hist_df, us_hist_msg = (
            fetch_us_crude_inventory_history_debug(years=10)
        )

        oecd_hist_df, oecd_hist_msg = (
            fetch_oecd_petroleum_stocks_history_debug(years=10)
        )

    brent_yoy = None
    production_yoy = None
    demand_yoy = None
    market_balance = None
    usa_inventory_yoy = None
    oecd_inventory_yoy = None

    # ==========================================================
    # BRENT
    # ==========================================================

    if oil_df is not None and not oil_df.empty:
        brent_yoy = _pct_vs_period(
            oil_df,
            "Date",
            "Close",
            years=1,
        )

    # ==========================================================
    # JODI-TUOTANTO
    # ==========================================================

    if prod_df is not None and not prod_df.empty:
        prod = prod_df.copy()

        prod["Date"] = pd.to_datetime(
            prod["Date"],
            errors="coerce",
        )

        prod["Production_kbd"] = pd.to_numeric(
            prod["Production_kbd"],
            errors="coerce",
        )

        prod = (
            prod.dropna(
                subset=[
                    "Country",
                    "Date",
                    "Production_kbd",
                ]
            )
            .loc[lambda df: df["Production_kbd"] > 0]
            .sort_values(["Country", "Date"])
            .reset_index(drop=True)
        )

        date_counts = (
            prod.groupby("Date")["Country"]
            .nunique()
            .sort_index()
        )

        suitable_dates = date_counts[date_counts >= 10]

        if not suitable_dates.empty:
            latest_comparison_date = suitable_dates.index.max()

            latest_country_data = (
                prod[prod["Date"] == latest_comparison_date]
                .sort_values(
                    "Production_kbd",
                    ascending=False,
                )
                .head(5)
            )

            country_changes = []

            for country in latest_country_data["Country"]:
                country_df = prod[
                    prod["Country"] == country
                ].copy()

                country_yoy = _pct_vs_period(
                    country_df,
                    "Date",
                    "Production_kbd",
                    years=1,
                )

                if country_yoy is not None:
                    country_changes.append(country_yoy)

            if country_changes:
                production_yoy = float(
                    pd.Series(country_changes).mean()
                )

    # ==========================================================
    # EIA:N MAAILMAN KYSYNTÄ JA TASAPAINO
    # ==========================================================

    if eia_df is not None and not eia_df.empty:
        eia = eia_df.copy()

        eia["Date"] = pd.to_datetime(
            eia["Date"],
            errors="coerce",
        )

        for column in [
            "Production",
            "Consumption",
            "Balance",
        ]:
            eia[column] = pd.to_numeric(
                eia[column],
                errors="coerce",
            )

        eia = (
            eia.dropna(
                subset=[
                    "Date",
                    "Production",
                    "Consumption",
                    "Balance",
                ]
            )
            .sort_values("Date")
            .reset_index(drop=True)
        )

        status = (
            eia["Status"]
            .astype(str)
            .str.strip()
            .str.lower()
            if "Status" in eia.columns
            else pd.Series(
                "historia",
                index=eia.index,
            )
        )

        eia_history = eia[
            status.isin(
                [
                    "historia",
                    "historical",
                    "history",
                    "actual",
                ]
            )
        ].copy()

        if not eia_history.empty:
            demand_yoy = _pct_vs_period(
                eia_history,
                "Date",
                "Consumption",
                years=1,
            )

            market_balance = float(
                eia_history.iloc[-1]["Balance"]
            )

    # ==========================================================
    # VARASTOT
    # ==========================================================

    if us_hist_df is not None and not us_hist_df.empty:
        usa_inventory_yoy = _pct_vs_period(
            us_hist_df,
            "Date",
            "Value",
            years=1,
        )

    if oecd_hist_df is not None and not oecd_hist_df.empty:
        oecd_inventory_yoy = _pct_vs_period(
            oecd_hist_df,
            "Date",
            "Value",
            years=1,
        )

    # ==========================================================
    # TILANNEKUVA
    # ==========================================================

    _render_analysis_signal_cards(
        brent_yoy=brent_yoy,
        usa_inventory_yoy=usa_inventory_yoy,
        oecd_inventory_yoy=oecd_inventory_yoy,
        production_yoy=production_yoy,
    )

    st.divider()

    parts: list[str] = []

    # ==========================================================
    # BRENT-ANALYYSI
    # ==========================================================

    if brent_yoy is not None:
        if brent_yoy > 10:
            parts.append(
                "Brent-raakaöljyn hinta on noussut selvästi "
                "vuoden aikana. Tämä voi viitata vahvistuneeseen "
                "kysyntään, tarjonnan rajoitteisiin tai kohonneeseen "
                "geopoliittiseen riskipreemioon."
            )
        elif brent_yoy < -10:
            parts.append(
                "Brent-raakaöljyn hinta on laskenut selvästi "
                "vuoden aikana. Tämä voi kertoa kysynnän "
                "heikkenemisestä tai tarjonnan runsaudesta."
            )
        else:
            parts.append(
                "Brent-raakaöljyn vuosimuutos on maltillinen, "
                "eikä hinta yksin osoita voimakasta "
                "kysyntä- tai tarjontashokkia."
            )

    # ==========================================================
    # TUOTANTO
    # ==========================================================

    if production_yoy is not None:
        if production_yoy > 2:
            parts.append(
                "JODI:n suurimpien raportoineiden tuottajamaiden "
                "tuotanto on kasvanut vuoden takaisesta. "
                "Tarjonnan kasvu voi hillitä öljyn hintapaineita."
            )
        elif production_yoy < -2:
            parts.append(
                "JODI:n suurimpien raportoineiden tuottajamaiden "
                "tuotanto on supistunut vuoden takaisesta. "
                "Tarjonnan väheneminen voi tukea öljyn hintaa."
            )
        else:
            parts.append(
                "JODI:n suurimpien raportoineiden tuottajamaiden "
                "tuotanto on pysynyt melko vakaana."
            )

    # ==========================================================
    # KYSYNTÄ
    # ==========================================================

    if demand_yoy is not None:
        if demand_yoy > 2:
            parts.append(
                "EIA STEO -aineiston mukaan maailman öljyn "
                "kysyntä on kasvanut selvästi vuoden aikana."
            )
        elif demand_yoy < -2:
            parts.append(
                "EIA STEO -aineiston mukaan maailman öljyn "
                "kysyntä on heikentynyt vuoden aikana."
            )
        else:
            parts.append(
                "Maailman öljyn kysynnän vuosimuutos on ollut "
                "melko maltillinen."
            )

    # ==========================================================
    # TUOTANTO–KYSYNTÄTASAPAINO
    # ==========================================================

    if market_balance is not None:
        if market_balance > 1:
            parts.append(
                f"Maailman tuotanto ylittää kysynnän noin "
                f"{market_balance:.1f} miljoonalla barrelilla "
                "päivässä. Tämä viittaa ylitarjontaan ja voi "
                "rajoittaa hinnan nousua."
            )
        elif market_balance < -1:
            parts.append(
                f"Maailman kysyntä ylittää tuotannon noin "
                f"{abs(market_balance):.1f} miljoonalla barrelilla "
                "päivässä. Tämä viittaa markkinan alijäämään ja "
                "voi tukea öljyn hintaa."
            )
        else:
            parts.append(
                "Maailman tuotanto ja kysyntä ovat EIA:n mukaan "
                "melko lähellä tasapainoa."
            )

    # ==========================================================
    # VARASTOT
    # ==========================================================

    inventory_values = [
        value
        for value in [
            usa_inventory_yoy,
            oecd_inventory_yoy,
        ]
        if value is not None and not pd.isna(value)
    ]

    if inventory_values:
        avg_inventory = float(
            pd.Series(inventory_values).mean()
        )

        if avg_inventory < -5:
            parts.append(
                "USA:n ja OECD-maiden varastot ovat keskimäärin "
                "supistuneet vuoden aikana, mikä tukee tulkintaa "
                "kireämmästä markkinasta."
            )
        elif avg_inventory > 5:
            parts.append(
                "USA:n ja OECD-maiden varastot ovat keskimäärin "
                "kasvaneet vuoden aikana, mikä viittaa tarjonnan "
                "runsauteen tai kysynnän vaimeuteen."
            )
        else:
            parts.append(
                "USA:n ja OECD-maiden varastot ovat vuoden "
                "tasolla melko vakaat."
            )

    # ==========================================================
    # YHDISTELMÄTULKINTA
    # ==========================================================

    if (
        brent_yoy is not None
        and market_balance is not None
    ):
        if brent_yoy > 5 and market_balance < 0:
            parts.append(
                "Hinnan nousu yhdessä tuotantoalijäämän kanssa "
                "vahvistaa tulkintaa kireästä öljymarkkinasta."
            )
        elif brent_yoy < -5 and market_balance > 0:
            parts.append(
                "Hinnan lasku yhdessä ylitarjonnan kanssa "
                "vahvistaa tulkintaa pehmeästä öljymarkkinasta."
            )
        elif brent_yoy > 5 and market_balance > 0:
            parts.append(
                "Hinta on noussut ylitarjonnasta huolimatta. "
                "Markkina voi tällöin hinnoitella esimerkiksi "
                "geopoliittisia riskejä tai tulevia "
                "tarjontarajoitteita."
            )

    if not parts:
        parts.append(
            "Analyysia ei voitu muodostaa, koska keskeisiä "
            "öljymarkkinan datasarjoja puuttuu."
        )

    st.markdown("### 🧠 Öljymarkkina-analyysi")

    with st.container(border=True):
        st.write(" ".join(parts))

    st.info(
        "Tämä ei ole sijoitussuositus. Öljyn ja polttoaineiden "
        "hintoihin vaikuttavat kysyntä, tuotanto, varastot, "
        "valuuttakurssit, verotus sekä geopoliittiset riskit."
    )

    for message, title in [
        (oil_msg, "Brent-datan huomautus"),
        (prod_msg, "JODI-tuotantodatan huomautus"),
        (eia_msg, "EIA STEO -huomautus"),
        (us_hist_msg, "USA-varastodatan huomautus"),
        (oecd_hist_msg, "OECD-varastodatan huomautus"),
    ]:
        if message:
            _show_source_message(message, title)


def _render_price_tab():
    st.subheader("💵 Öljyn hinta ja Suomen polttoainehinnat")

    # ==========================================================
    # DATAN HAKU
    # ==========================================================

    with st.spinner("Haetaan Brent-raakaöljyn markkinadataa..."):
        oil_df, oil_msg = fetch_price_history_debug(
            "BZ=F",
            period="10y",
        )

    with st.spinner("Haetaan Suomen viikoittaisia polttoainehintoja..."):
        weekly_fuel_df, weekly_fuel_msg = (
            fetch_finland_weekly_fuel_prices_debug(
                years=10,
            )
        )

    if oil_df is None or oil_df.empty:
        st.error("Brent-raakaöljyn hintadataa ei saatu.")

        if oil_msg:
            _show_source_message(
                oil_msg,
                "Brent-datan tekninen huomautus",
            )

        return

    if weekly_fuel_df is None or weekly_fuel_df.empty:
        st.error(
            "Suomen viikoittaista polttoainehintadataa ei saatu."
        )

        if weekly_fuel_msg:
            _show_source_message(
                weekly_fuel_msg,
                "Polttoainedatan tekninen huomautus",
            )

        return

    # ==========================================================
    # BRENT-DATAN PUHDISTUS
    # ==========================================================

    oil_df = oil_df.copy()

    oil_df["Date"] = pd.to_datetime(
        oil_df["Date"],
        errors="coerce",
    )

    oil_df["Close"] = pd.to_numeric(
        oil_df["Close"],
        errors="coerce",
    )

    oil_df = (
        oil_df.dropna(subset=["Date", "Close"])
        .loc[lambda frame: frame["Close"] > 0]
        .sort_values("Date")
        .reset_index(drop=True)
    )

    if oil_df.empty:
        st.error(
            "Brent-data jäi tyhjäksi puhdistuksen jälkeen."
        )
        return

    # ==========================================================
    # VIIKKOHINTOJEN PUHDISTUS
    # ==========================================================

    weekly_fuel_df = weekly_fuel_df.copy()

    weekly_fuel_df["Date"] = pd.to_datetime(
        weekly_fuel_df["Date"],
        errors="coerce",
    )

    weekly_fuel_df["Price_EUR_L"] = pd.to_numeric(
        weekly_fuel_df["Price_EUR_L"],
        errors="coerce",
    )

    weekly_fuel_df["Fuel"] = (
        weekly_fuel_df["Fuel"]
        .astype(str)
        .str.strip()
    )

    weekly_fuel_df = (
        weekly_fuel_df.dropna(
            subset=[
                "Date",
                "Fuel",
                "Price_EUR_L",
            ]
        )
        .loc[
            lambda frame:
            (frame["Price_EUR_L"] > 0)
            & (frame["Price_EUR_L"] < 10)
        ]
        .sort_values(["Fuel", "Date"])
        .drop_duplicates(
            subset=["Fuel", "Date"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    petrol_weekly = (
        weekly_fuel_df[
            weekly_fuel_df["Fuel"] == "95E10"
        ]
        .copy()
        .sort_values("Date")
        .reset_index(drop=True)
    )

    diesel_weekly = (
        weekly_fuel_df[
            weekly_fuel_df["Fuel"] == "Diesel"
        ]
        .copy()
        .sort_values("Date")
        .reset_index(drop=True)
    )

    if petrol_weekly.empty and diesel_weekly.empty:
        st.error(
            "Viikkodatasta ei löytynyt 95E10- tai dieselhintaa."
        )
        return

    # ==========================================================
    # TUNNUSLUVUT
    # ==========================================================

    latest_brent = float(
        oil_df.iloc[-1]["Close"]
    )

    brent_1m = _pct_vs_period(
        oil_df,
        "Date",
        "Close",
        months=1,
    )

    brent_1y = _pct_vs_period(
        oil_df,
        "Date",
        "Close",
        years=1,
    )

    brent_5y = _pct_vs_period(
        oil_df,
        "Date",
        "Close",
        years=5,
    )

    petrol_latest = (
        float(
            petrol_weekly.iloc[-1][
                "Price_EUR_L"
            ]
        )
        if not petrol_weekly.empty
        else None
    )

    diesel_latest = (
        float(
            diesel_weekly.iloc[-1][
                "Price_EUR_L"
            ]
        )
        if not diesel_weekly.empty
        else None
    )

    petrol_1m = (
        _pct_vs_period(
            petrol_weekly,
            "Date",
            "Price_EUR_L",
            months=1,
        )
        if not petrol_weekly.empty
        else None
    )

    petrol_1y = (
        _pct_vs_period(
            petrol_weekly,
            "Date",
            "Price_EUR_L",
            years=1,
        )
        if not petrol_weekly.empty
        else None
    )

    petrol_5y = (
        _pct_vs_period(
            petrol_weekly,
            "Date",
            "Price_EUR_L",
            years=5,
        )
        if not petrol_weekly.empty
        else None
    )

    diesel_1m = (
        _pct_vs_period(
            diesel_weekly,
            "Date",
            "Price_EUR_L",
            months=1,
        )
        if not diesel_weekly.empty
        else None
    )

    diesel_1y = (
        _pct_vs_period(
            diesel_weekly,
            "Date",
            "Price_EUR_L",
            years=1,
        )
        if not diesel_weekly.empty
        else None
    )

    diesel_5y = (
        _pct_vs_period(
            diesel_weekly,
            "Date",
            "Price_EUR_L",
            years=5,
        )
        if not diesel_weekly.empty
        else None
    )

    # ==========================================================
    # HINTAYHTEENVETO
    # ==========================================================

    st.markdown("### 💶 Hintayhteenveto")

    c1, c2, c3, _ = st.columns(
        [0.28, 0.28, 0.28, 0.16]
    )

    with c1:
        _summary_card(
            "🛢 Brent",
            f"{latest_brent:.2f} USD",
            [
                ("1 kk", brent_1m),
                ("1 vuosi", brent_1y),
                ("5 vuotta", brent_5y),
            ],
        )

        st.caption(
            "Viimeisin markkinapäivä: "
            f"{oil_df['Date'].max().strftime('%d.%m.%Y')}"
        )

    with c2:
        if petrol_latest is None:
            st.warning(
                "95E10-hintaa ei löytynyt."
            )
        else:
            _summary_card(
                "⛽ 95E10",
                _fmt_fuel_price(
                    petrol_latest
                ),
                [
                    ("1 kk", petrol_1m),
                    ("1 vuosi", petrol_1y),
                    ("5 vuotta", petrol_5y),
                ],
            )

            st.caption(
                "Viimeisin viikkohinta: "
                f"{petrol_weekly['Date'].max().strftime('%d.%m.%Y')}"
            )

    with c3:
        if diesel_latest is None:
            st.warning(
                "Dieselhintaa ei löytynyt."
            )
        else:
            _summary_card(
                "🚚 Diesel",
                _fmt_fuel_price(
                    diesel_latest
                ),
                [
                    ("1 kk", diesel_1m),
                    ("1 vuosi", diesel_1y),
                    ("5 vuotta", diesel_5y),
                ],
            )

            st.caption(
                "Viimeisin viikkohinta: "
                f"{diesel_weekly['Date'].max().strftime('%d.%m.%Y')}"
            )

    st.caption(
        "Suomen polttoainehinnat ovat verollisia viikoittaisia "
        "kuluttajahintoja. Lähde: Euroopan komission "
        "Weekly Oil Bulletin."
    )

    st.divider()

    # ==========================================================
    # BRENT-KUVAAJA
    # ==========================================================

    st.markdown("### 🛢 Brent-raakaöljy")

    brent_period = st.segmented_control(
        "Brent-kuvaajan tarkasteluväli",
        options=[
            "1 v",
            "5 v",
            "10 v",
        ],
        default="5 v",
        key="oil_price_brent_period",
    )

    brent_period = (
        brent_period
        or "5 v"
    )

    brent_years = {
        "1 v": 1,
        "5 v": 5,
        "10 v": 10,
    }

    brent_cutoff = (
        oil_df["Date"].max()
        - pd.DateOffset(
            years=brent_years[
                brent_period
            ]
        )
    )

    brent_plot_df = oil_df[
        oil_df["Date"] >= brent_cutoff
    ].copy()

    fig_brent = px.line(
        brent_plot_df,
        x="Date",
        y="Close",
        title=(
            "Brent-raakaöljyn hinta "
            f"({brent_period})"
        ),
        labels={
            "Date": "Päivä",
            "Close": "USD/barreli",
        },
    )

    fig_brent.update_layout(
        hovermode="x unified",
    )

    fig_brent.update_yaxes(
        title_text="USD/barreli",
    )

    st.plotly_chart(
        fig_brent,
        use_container_width=True,
    )

    st.divider()

    # ==========================================================
    # SUOMEN POLTTOAINEHINTOJEN KUVAAJA
    # ==========================================================

    st.markdown("### ⛽ Suomen polttoaineiden viikkohinnat")

    fuel_period = st.segmented_control(
        "Polttoainekuvaajan tarkasteluväli",
        options=[
            "1 v",
            "5 v",
            "10 v",
        ],
        default="5 v",
        key="oil_price_fuel_period",
    )

    fuel_period = (
        fuel_period
        or "5 v"
    )

    fuel_years = {
        "1 v": 1,
        "5 v": 5,
        "10 v": 10,
    }

    fuel_cutoff = (
        weekly_fuel_df["Date"].max()
        - pd.DateOffset(
            years=fuel_years[
                fuel_period
            ]
        )
    )

    fuel_plot_df = weekly_fuel_df[
        (
            weekly_fuel_df["Date"]
            >= fuel_cutoff
        )
        & (
            weekly_fuel_df["Fuel"].isin(
                [
                    "95E10",
                    "Diesel",
                ]
            )
        )
    ].copy()

    fuel_plot_df["Polttoaine"] = (
        fuel_plot_df["Fuel"].map(
            {
                "95E10": "95E10",
                "Diesel": "Diesel",
            }
        )
    )

    if fuel_plot_df.empty:
        st.warning(
            "Polttoaineiden viikkokuvaajaa ei voitu muodostaa."
        )
    else:
        fig_fuels = px.line(
            fuel_plot_df,
            x="Date",
            y="Price_EUR_L",
            color="Polttoaine",
            title=(
                "Suomen verolliset polttoainehinnat "
                f"({fuel_period})"
            ),
            labels={
                "Date": "Viikko",
                "Price_EUR_L": "€/l",
                "Polttoaine": "Polttoaine",
            },
        )

        fig_fuels.update_layout(
            hovermode="x unified",
        )

        fig_fuels.update_yaxes(
            title_text="€/l",
            tickformat=".2f",
        )

        st.plotly_chart(
            fig_fuels,
            use_container_width=True,
        )

    st.info(
        "Hinta- ja ennustevälilehdet käyttävät nyt samaa Suomen "
        "viikoittaista polttoainehintasarjaa. Ennustemalli oppii "
        "toteutuneista verollisista pumppuhinnoista, joten voimassa "
        "oleva verotus sisältyy historialliseen aineistoon."
    )

    if oil_msg:
        _show_source_message(
            oil_msg,
            "Brent-datan huomautus",
        )

    if weekly_fuel_msg:
        _show_source_message(
            weekly_fuel_msg,
            "Viikkohintadatan huomautus",
        )



def _render_production_tab():
    st.subheader("🌍 Öljyntuotanto ja maailman kysyntä")

    with st.spinner("Haetaan JODI:n maakohtaista tuotantodataa..."):
        jodi_df, jodi_msg = fetch_jodi_crude_production_debug(years=8)

    with st.spinner("Haetaan EIA STEO -tuotanto- ja kysyntädataa..."):
        eia_df, eia_msg = fetch_world_oil_balance_eia_debug()

    if jodi_df is None or jodi_df.empty:
        st.error("JODI:n maakohtaista öljyntuotantodataa ei saatu.")
        _show_source_message(jodi_msg, "JODI-haun tekninen virhe")
        return

    # ==========================================================
    # JODI-DATAN PUHDISTUS
    # ==========================================================

    jodi_df = jodi_df.copy()

    jodi_df["Date"] = pd.to_datetime(
        jodi_df["Date"],
        errors="coerce",
    )

    jodi_df["Production_kbd"] = pd.to_numeric(
        jodi_df["Production_kbd"],
        errors="coerce",
    )

    jodi_df = (
        jodi_df
        .dropna(subset=["Country", "Date", "Production_kbd"])
        .loc[lambda df: df["Production_kbd"] > 0]
        .sort_values(["Country", "Date"])
        .reset_index(drop=True)
    )

    # ==========================================================
    # EIA STEO -DATAN PUHDISTUS
    # ==========================================================

    eia_latest_actual = None
    eia_latest_forecast = None
    historical_rows = pd.DataFrame()
    forecast_rows = pd.DataFrame()

    if eia_df is not None and not eia_df.empty:
        eia_df = eia_df.copy()

        eia_df["Date"] = pd.to_datetime(
            eia_df["Date"],
            errors="coerce",
        )

        for column in ["Production", "Consumption", "Balance"]:
            eia_df[column] = pd.to_numeric(
                eia_df[column],
                errors="coerce",
            )

        eia_df = (
            eia_df
            .dropna(
                subset=[
                    "Date",
                    "Production",
                    "Consumption",
                    "Balance",
                ]
            )
            .sort_values("Date")
            .reset_index(drop=True)
        )

        if "Status" not in eia_df.columns:
            eia_df["Status"] = "Historia"

        normalized_status = (
            eia_df["Status"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        historical_rows = eia_df[
            normalized_status.isin(
                ["historia", "historical", "history", "actual"]
            )
        ].copy()

        forecast_rows = eia_df[
            normalized_status.isin(
                ["ennuste", "forecast", "projection"]
            )
        ].copy()

        if not historical_rows.empty:
            eia_latest_actual = historical_rows.iloc[-1]

        if not forecast_rows.empty:
            eia_latest_forecast = forecast_rows.iloc[-1]

    # ==========================================================
    # MAAKOHTAISET TUNNUSLUVUT
    # ==========================================================

    # JODI-aineiston uusin kuukausi toimii tuoreuden vertailukohtana.
    jodi_latest_date = pd.to_datetime(jodi_df["Date"].max())
    freshness_cutoff = jodi_latest_date - pd.DateOffset(months=6)


    def country_stats(
        country: str,
        require_fresh: bool = False,
    ) -> dict | None:
        country_df = (
            jodi_df[jodi_df["Country"] == country]
            .copy()
            .sort_values("Date")
        )

        if country_df.empty:
            return None

        latest_row = country_df.iloc[-1]
        latest_date = pd.to_datetime(latest_row["Date"])

        if require_fresh and latest_date < freshness_cutoff:
            return None

        return {
            "country": country,
            "date": latest_date,
            "value": float(latest_row["Production_kbd"]),
            "pct_1m": _pct_vs_period(
                country_df,
                "Date",
                "Production_kbd",
                months=1,
            ),
            "pct_1y": _pct_vs_period(
                country_df,
                "Date",
                "Production_kbd",
                years=1,
            ),
            "pct_5y": _pct_vs_period(
                country_df,
                "Date",
                "Production_kbd",
                years=5,
            ),
        }


    usa = country_stats(
        "United States",
        require_fresh=True,
    )

    saudi = country_stats(
        "Saudi Arabia",
        require_fresh=True,
    )

    russia_all = country_stats(
        "Russia",
        require_fresh=False,
    )

    russia = country_stats(
        "Russia",
        require_fresh=True,
    )

    third_country_name = "Russia"
    third_country_stats = russia

    # Jos Venäjän data on liian vanhaa tai puuttuu, valitaan seuraavaksi
    # suurin tuoretta dataa raportoiva maa.
    if third_country_stats is None:
        latest_rows_by_country = (
            jodi_df.sort_values("Date")
            .groupby("Country", as_index=False)
            .tail(1)
            .copy()
        )

        fresh_country_rows = latest_rows_by_country[
            pd.to_datetime(latest_rows_by_country["Date"])
            >= freshness_cutoff
        ].copy()

        excluded_countries = {
            "United States",
            "Saudi Arabia",
            "Russia",
        }

        replacement_rows = (
            fresh_country_rows[
                ~fresh_country_rows["Country"].isin(excluded_countries)
            ]
            .sort_values("Production_kbd", ascending=False)
        )

        if not replacement_rows.empty:
            third_country_name = str(
                replacement_rows.iloc[0]["Country"]
            )
            third_country_stats = country_stats(
                third_country_name,
                require_fresh=True,
            )

    # ==========================================================
    # TILANNEKUVA
    # ==========================================================

    st.markdown("### 📊 Tilannekuva")

    st.caption(
        "Maakohtainen tuotanto perustuu JODI:n kuukausidataan. "
        "Maailman kysyntä perustuu EIA STEO -aineistoon. "
        "Yksikkö on miljoonaa barrelia päivässä."
    )

    c1, c2, c3, c4 = st.columns(4)

    def render_country_card(
        column,
        title: str,
        stats: dict | None,
    ) -> None:
        with column:
            if stats is None:
                st.warning(f"{title}-dataa ei löytynyt.")
                return

            changes = []

            if stats["pct_1m"] is not None:
                changes.append(("1 kk", stats["pct_1m"]))

            if stats["pct_1y"] is not None:
                changes.append(("1 vuosi", stats["pct_1y"]))

            if stats["pct_5y"] is not None:
                changes.append(("5 vuotta", stats["pct_5y"]))

            _summary_card(
                title,
                _fmt_oil_rate(
                    stats["value"],
                    source_unit="kb_per_day",
                ),
                changes,
            )

            st.caption(
                f"Viimeisin kuukausi: "
                f"{pd.to_datetime(stats['date']).strftime('%m/%Y')}"
            )

    render_country_card(c1, "🇺🇸 USA", usa)
    render_country_card(c2, "🇸🇦 Saudi-Arabia", saudi)
    render_country_card(
        c3,
        f"{_country_flag(third_country_name)} {third_country_name}",
        third_country_stats,
    )
    if russia_all is not None:
        russia_latest_date = pd.to_datetime(russia_all["date"])

        if russia_latest_date < freshness_cutoff:
            st.info(
                "Venäjän JODI-tuotantosarja päättyy "
                f"{russia_latest_date.strftime('%m/%Y')}. "
                "Sarjaa ei näytetä aloituskortissa eikä oletuskuvaajassa, "
                "koska tieto on yli kuusi kuukautta vanhaa. "
                "Raportointikatkos ajoittuu Venäjän ja Ukrainan välisen "
                "konfliktin ja sitä seuranneiden pakotteiden aikaan, "
                "mutta JODI-aineisto ei yksin kerro katkokselle tarkkaa syytä."
            )

    with c4:
        if eia_latest_actual is None:
            st.warning("Maailman kysyntädataa ei löytynyt.")
        else:
            demand_changes = []

            demand_1y = _pct_vs_period(
                historical_rows,
                "Date",
                "Consumption",
                years=1,
            )

            demand_5y = _pct_vs_period(
                historical_rows,
                "Date",
                "Consumption",
                years=5,
            )

            if demand_1y is not None:
                demand_changes.append(("1 vuosi", demand_1y))

            if demand_5y is not None:
                demand_changes.append(("5 vuotta", demand_5y))

            _summary_card(
                "🌍 Kysyntä",
                _fmt_oil_rate(
                    eia_latest_actual["Consumption"]
                ),
                demand_changes,
            )

            st.caption(
                "Viimeisin historiallinen/arvioitu kuukausi: "
                f"{pd.to_datetime(eia_latest_actual['Date']).strftime('%m/%Y')}"
            )

    st.divider()

    # ==========================================================
    # MAAILMAN TUOTANTO VS KYSYNTÄ
    # ==========================================================

    st.markdown("### 🌍 Maailman tuotanto vs kysyntä")

    if eia_latest_actual is None:
        st.warning(
            "EIA STEO -tuotanto–kysyntävertailua ei voitu muodostaa."
        )
    else:
        b1, b2, b3, _ = st.columns(
            [0.28, 0.28, 0.28, 0.16]
        )

        production_changes = []
        consumption_changes = []

        production_1y = _pct_vs_period(
            historical_rows,
            "Date",
            "Production",
            years=1,
        )

        consumption_1y = _pct_vs_period(
            historical_rows,
            "Date",
            "Consumption",
            years=1,
        )

        if production_1y is not None:
            production_changes.append(
                ("1 vuosi", production_1y)
            )

        if consumption_1y is not None:
            consumption_changes.append(
                ("1 vuosi", consumption_1y)
            )

        with b1:
            _summary_card(
                "🛢 Maailman tuotanto",
                _fmt_oil_rate(
                    eia_latest_actual["Production"]
                ),
                production_changes,
            )

        with b2:
            _summary_card(
                "🌍 Maailman kysyntä",
                _fmt_oil_rate(
                    eia_latest_actual["Consumption"]
                ),
                consumption_changes,
            )

        with b3:
            balance = float(eia_latest_actual["Balance"])
            balance_label = (
                "Ylitarjonta"
                if balance > 0
                else "Alijäämä"
            )

            with st.container(border=True):
                st.markdown("### ⚖️ Erotus")
                st.caption("Tuotanto − kysyntä")
                st.markdown(
                    f"## {_fmt_oil_rate(balance)}"
                )

                st.divider()

                st.markdown(
                    f"**Markkinatasapaino:** {balance_label}"
                )

        latest_actual_date = pd.to_datetime(
            eia_latest_actual["Date"]
        ).strftime("%m/%Y")

        st.caption(
            f"Viimeisin historiallinen/arvioitu kuukausi: "
            f"{latest_actual_date}. "
            "Erotus = maailman tuotanto − maailman kysyntä."
        )

        if eia_latest_forecast is not None:
            latest_forecast_date = pd.to_datetime(
                eia_latest_forecast["Date"]
            ).strftime("%m/%Y")

            st.caption(
                f"EIA STEO -ennuste ulottuu kuukauteen "
                f"{latest_forecast_date}."
            )

        balance_long = eia_df.melt(
            id_vars=["Date", "Status"],
            value_vars=["Production", "Consumption"],
            var_name="Series",
            value_name="Value",
        )

        balance_long["Series"] = balance_long["Series"].map(
            {
                "Production": "Tuotanto",
                "Consumption": "Kysyntä",
            }
        )

        fig_balance = px.line(
            balance_long,
            x="Date",
            y="Value",
            color="Series",
            line_dash="Status",
            title=(
                "Maailman liquid fuels -tuotanto ja kysyntä "
                "(EIA STEO)"
            ),
            labels={
                "Date": "Kuukausi",
                "Value": "Milj. bbl/pv",
                "Series": "Sarja",
                "Status": "Tietotyyppi",
            },
        )

        fig_balance.update_layout(
            hovermode="x unified",
            legend_title_text="Sarja / tietotyyppi",
        )

        fig_balance.update_yaxes(
            title_text="Milj. bbl/pv",
            tickformat=".1f",
        )

        fig_balance.update_xaxes(
            title_text="Kuukausi",
            dtick="M6",
            tickformat="%m/%Y",
        )

        st.plotly_chart(
            fig_balance,
            use_container_width=True,
        )

    st.divider()

    # ==========================================================
    # SUURIMMAT TUOTTAJAT
    # ==========================================================

    st.markdown("### 📈 Suurimmat öljyntuottajamaat")

    date_counts = (
        jodi_df.groupby("Date")["Country"]
        .nunique()
        .sort_index()
    )

    suitable_dates = date_counts[date_counts >= 10]

    if suitable_dates.empty:
        comparison_date = jodi_df["Date"].max()
    else:
        comparison_date = suitable_dates.index.max()

    latest_comparison = (
        jodi_df[jodi_df["Date"] == comparison_date]
        .sort_values("Production_kbd", ascending=False)
        .head(10)
        .copy()
    )

    if latest_comparison.empty:
        st.warning(
            "Suurimpien tuottajien vertailua ei voitu muodostaa."
        )
    else:
        latest_comparison["Production_mbd"] = (
            latest_comparison["Production_kbd"] / 1_000
        )

        top10 = latest_comparison.sort_values(
            "Production_mbd",
            ascending=True,
        )

        fig_top = px.bar(
            top10,
            x="Production_mbd",
            y="Country",
            orientation="h",
            title=(
                "Suurimmat JODI-tuottajamaat "
                f"({comparison_date.strftime('%m/%Y')})"
            ),
            labels={
                "Country": "Maa",
                "Production_mbd": "Milj. bbl/pv",
            },
        )

        fig_top.update_xaxes(
            title_text="Milj. bbl/pv",
        )

        st.plotly_chart(
            fig_top,
            use_container_width=True,
        )

    st.divider()

    # ==========================================================
    # TUOTANNON KUUKAUSITRENDI
    # ==========================================================

    st.markdown("### 📈 Tuotannon kehitys")

    selectable_countries = sorted(
        country
        for country in jodi_df["Country"].dropna().unique()
        if len(str(country)) > 2
    )

    default_country_candidates = [
        "United States",
        "Saudi Arabia",
        third_country_name,
    ]

    default_countries = []

    for country in default_country_candidates:
        stats = country_stats(
            country,
            require_fresh=True,
        )

        if (
            stats is not None
            and country in selectable_countries
            and country not in default_countries
        ):
            default_countries.append(country)

    selected_countries = st.multiselect(
        "Valitse maat tuotannon trendikuvaajaan",
        options=selectable_countries,
        default=default_countries,
        key="jodi_production_countries",
    )

    period_years = st.segmented_control(
        "Tarkasteluväli",
        options=["2 v", "5 v", "8 v"],
        default="5 v",
        key="jodi_production_period",
    )

    period_years = period_years or "5 v"

    years_map = {
        "2 v": 2,
        "5 v": 5,
        "8 v": 8,
    }

    cutoff = (
        jodi_df["Date"].max()
        - pd.DateOffset(
            years=years_map[period_years]
        )
    )

    if selected_countries:
        trend_df = jodi_df[
            (jodi_df["Country"].isin(selected_countries))
            & (jodi_df["Date"] >= cutoff)
        ].copy()

        trend_df["Production_mbd"] = (
            trend_df["Production_kbd"] / 1_000
        )

        fig_trend = px.line(
            trend_df,
            x="Date",
            y="Production_mbd",
            color="Country",
            title=(
                "Kuukausittainen raakaöljyntuotanto "
                f"({period_years})"
            ),
            labels={
                "Date": "Kuukausi",
                "Production_mbd": "Milj. bbl/pv",
                "Country": "Maa",
            },
        )

        fig_trend.update_layout(
            hovermode="x unified",
        )

        fig_trend.update_yaxes(
            title_text="Milj. bbl/pv",
        )

        fig_trend.update_xaxes(
            title_text="Kuukausi",
            tickformat="%m/%Y",
        )

        st.plotly_chart(
            fig_trend,
            use_container_width=True,
        )
    else:
        st.info(
            "Valitse vähintään yksi maa tuotantokuvaajaan."
        )

    if jodi_msg:
        _show_source_message(
            jodi_msg,
            "JODI-tuotantodatan huomautus",
        )

    if eia_msg:
        _show_source_message(
            eia_msg,
            "EIA STEO -huomautus",
        )



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
        "Lähteet: Yahoo Finance (Brent), Tilastokeskus / Traficom "
        "(Suomen polttoainehinnat), JODI (maakohtainen kuukausituotanto) "
        "sekä EIA (varastot ja STEO:n maailman tuotanto-, kysyntä- "
        "ja ennustesarjat)."
    )

    tab_price, tab_production, tab_inventory, tab_forecast, tab_analysis = st.tabs(
        [
            "💵 Hinta",
            "🌍 Tuotanto",
            "📦 Varastot",
            "🔮 Ennuste",
            "🧠 Analyysi",
        ]
    )

    with tab_price:
        _render_price_tab()

    with tab_production:
        _render_production_tab()

    with tab_inventory:
        _render_inventory_tab()

    with tab_forecast:
        _render_forecast_tab()

    with tab_analysis:
        _render_oil_analysis()