
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from services.sovereign_yields import (
    COUNTRY_META,
    build_10y_2y_spread,
    build_latest_yield_snapshot,
    build_yield_curve_snapshot,
    fetch_country_yields,
    fetch_sovereign_yields_overview,
    fetch_yield_history,
)


MATURITY_ORDER = {
    "2Y": 2,
    "5Y": 5,
    "10Y": 10,
    "30Y": 30,
}

MATURITY_LABELS = {
    "2Y": "2 vuotta",
    "5Y": "5 vuotta",
    "10Y": "10 vuotta",
    "30Y": "30 vuotta",
}

# Britannia jätetään toistaiseksi pois suorituskyvyn vuoksi.
COUNTRY_ORDER = ["US", "DE", "JP"]


@st.cache_data(
    ttl=60 * 60 * 6,
    show_spinner=False,
)
def load_rates_overview() -> dict:
    df, debug = fetch_sovereign_yields_overview(months=14)
    return {"data": df, "debug": debug}


@st.cache_data(
    ttl=60 * 60 * 12,
    show_spinner=False,
)
def load_rate_history(
    country: str,
    maturity: str,
    years: int,
) -> dict:
    df, error = fetch_yield_history(
        country,
        maturity,
        years=years,
    )
    return {"data": df, "error": error}


@st.cache_data(
    ttl=60 * 60 * 12,
    show_spinner=False,
)
def load_country_curve_history(
    country: str,
    years: int = 2,
) -> dict:
    df, error = fetch_country_yields(
        country,
        years=years,
        maturities=("2Y", "5Y", "10Y", "30Y"),
    )
    return {"data": df, "error": error}


def _country_label(country: str) -> str:
    meta = COUNTRY_META.get(country, {})
    return f"{meta.get('flag', '')} {meta.get('name', country)}".strip()


def _maturity_label(maturity: str) -> str:
    return MATURITY_LABELS.get(maturity, maturity)


def _fmt_yield(value) -> str:
    if value is None or pd.isna(value):
        return "–"
    return f"{float(value):.2f} %"


def _fmt_bp(value) -> str:
    if value is None or pd.isna(value):
        return "–"
    value = float(value)
    if abs(value) < 0.05:
        value = 0.0
    return f"{value:+.0f} bp"


def _yield_change_bp(
    df: pd.DataFrame,
    *,
    country: str,
    maturity: str,
    months: int,
) -> float | None:
    subset = df.loc[
        (df["Country"] == country)
        & (df["Maturity"] == maturity)
    ].copy()

    if subset.empty:
        return None

    subset["Date"] = pd.to_datetime(subset["Date"], errors="coerce")
    subset["Yield"] = pd.to_numeric(subset["Yield"], errors="coerce")
    subset = subset.dropna(subset=["Date", "Yield"]).sort_values("Date")

    if subset.empty:
        return None

    latest = subset.iloc[-1]
    target = latest["Date"] - pd.DateOffset(months=months)
    old = subset.loc[subset["Date"] <= target]

    if old.empty:
        return None

    return (float(latest["Yield"]) - float(old.iloc[-1]["Yield"])) * 100.0


def _curve_status(spread: float | None) -> tuple[str, str]:
    if spread is None or pd.isna(spread):
        return "⚪", "Ei dataa"

    bp = float(spread) * 100.0

    if bp > 25:
        return "🟢", "Nouseva"
    if bp < -25:
        return "🔴", "Käänteinen"
    return "🟡", "Lähes tasainen"


def _latest_country_date(df: pd.DataFrame, country: str):
    subset = df.loc[df["Country"] == country]
    if subset.empty:
        return None

    dates = pd.to_datetime(subset["Date"], errors="coerce").dropna()
    return dates.max() if not dates.empty else None


def _filter_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date"]).sort_values("Date")

    if out.empty:
        return out

    years = {"1 v": 1, "3 v": 3, "5 v": 5, "10 v": 10}[period]
    cutoff = out["Date"].max() - pd.DateOffset(years=years)

    return out.loc[out["Date"] >= cutoff].reset_index(drop=True)


def _curve_at_date(
    df: pd.DataFrame,
    *,
    country: str,
    target_date: pd.Timestamp,
) -> pd.DataFrame:
    subset = df.loc[df["Country"] == country].copy()

    if subset.empty:
        return pd.DataFrame()

    subset["Date"] = pd.to_datetime(subset["Date"], errors="coerce")
    subset["Yield"] = pd.to_numeric(subset["Yield"], errors="coerce")
    subset = subset.dropna(subset=["Date", "Yield"])

    rows = []

    for maturity, maturity_years in MATURITY_ORDER.items():
        part = subset.loc[
            (subset["Maturity"] == maturity)
            & (subset["Date"] <= target_date)
        ].sort_values("Date")

        if part.empty:
            continue

        row = part.iloc[-1]
        rows.append({
            "Date": row["Date"],
            "Country": country,
            "Maturity": maturity,
            "MaturityYears": maturity_years,
            "Yield": float(row["Yield"]),
        })

    return pd.DataFrame(rows)


def _render_overview(df: pd.DataFrame) -> None:
    st.markdown("### 🌍 Korkomarkkinoiden yleiskuva")
    st.caption(
        "Valtionlainojen tuotot Yhdysvalloissa, Saksassa ja Japanissa. "
        "Korkojen muutokset esitetään korkopisteinä (100 bp = 1 %-yksikkö)."
    )

    latest = build_latest_yield_snapshot(df)

    if latest.empty:
        st.warning("Viimeisimpiä korkotietoja ei saatu.")
        return

    cards = st.columns(len(COUNTRY_ORDER))

    for col, country in zip(cards, COUNTRY_ORDER):
        country_latest = latest.loc[latest["Country"] == country]
        row_10y = country_latest.loc[country_latest["Maturity"] == "10Y"]

        latest_10y = (
            float(row_10y.iloc[-1]["Yield"])
            if not row_10y.empty
            else None
        )

        change_1m = _yield_change_bp(
            df,
            country=country,
            maturity="10Y",
            months=1,
        )
        latest_date = _latest_country_date(df, country)

        with col:
            with st.container(border=True):
                st.markdown(f"#### {_country_label(country)}")
                st.metric(
                    "10 vuoden korko",
                    _fmt_yield(latest_10y),
                    _fmt_bp(change_1m) if change_1m is not None else None,
                )
                st.caption("Muutos 1 kk")
                if latest_date is not None:
                    st.caption(f"Data: {latest_date.date()}")

    st.divider()

    rows = []

    for country in COUNTRY_ORDER:
        country_latest = latest.loc[latest["Country"] == country]
        values = {
            row["Maturity"]: row["Yield"]
            for _, row in country_latest.iterrows()
        }

        spread_df = build_10y_2y_spread(df, country=country)
        spread = (
            float(spread_df.iloc[-1]["Spread_10Y_2Y"])
            if not spread_df.empty
            else None
        )

        icon, status = _curve_status(spread)
        latest_date = _latest_country_date(df, country)

        rows.append({
            "Maa": _country_label(country),
            "2Y": values.get("2Y"),
            "5Y": values.get("5Y"),
            "10Y": values.get("10Y"),
            "30Y": values.get("30Y"),
            "10Y 1 kk (bp)": _yield_change_bp(
                df,
                country=country,
                maturity="10Y",
                months=1,
            ),
            "10Y–2Y (bp)": spread * 100.0 if spread is not None else None,
            "Käyrä": f"{icon} {status}",
            "Data": latest_date.date() if latest_date is not None else None,
        })

    overview = pd.DataFrame(rows)

    st.dataframe(
        overview,
        use_container_width=True,
        hide_index=True,
        column_config={
            "2Y": st.column_config.NumberColumn("2Y", format="%.2f %%"),
            "5Y": st.column_config.NumberColumn("5Y", format="%.2f %%"),
            "10Y": st.column_config.NumberColumn("10Y", format="%.2f %%"),
            "30Y": st.column_config.NumberColumn("30Y", format="%.2f %%"),
            "10Y 1 kk (bp)": st.column_config.NumberColumn(
                "10Y 1 kk", format="%+.0f bp"
            ),
            "10Y–2Y (bp)": st.column_config.NumberColumn(
                "10Y–2Y", format="%+.0f bp"
            ),
        },
    )

    st.caption(
        "Tuottokäyrä: yli +25 bp = nouseva, −25…+25 bp = lähes tasainen, "
        "alle −25 bp = käänteinen."
    )


def _render_history() -> None:
    st.markdown("### 📈 Korkokehitys")

    c1, c2 = st.columns(2)

    with c1:
        country = st.selectbox(
            "Valitse maa",
            COUNTRY_ORDER,
            format_func=_country_label,
            key="rates_history_country",
        )

    with c2:
        maturity = st.selectbox(
            "Valitse maturiteetti",
            list(MATURITY_ORDER),
            index=2,
            format_func=_maturity_label,
            key="rates_history_maturity",
        )

    period = st.radio(
        "Aikajakso",
        ["1 v", "3 v", "5 v", "10 v"],
        index=3,
        horizontal=True,
        key="rates_history_period",
    )

    years = {"1 v": 1, "3 v": 3, "5 v": 5, "10 v": 10}[period]

    with st.spinner(
        f"Ladataan {_country_label(country)} – {_maturity_label(maturity)}…"
    ):
        bundle = load_rate_history(country, maturity, years)

    df = bundle["data"]

    if df is None or df.empty:
        st.warning(bundle["error"] or "Korkohistoriaa ei saatu.")
        return

    subset = _filter_period(df, period)
    latest = subset.iloc[-1]

    change_1m = _yield_change_bp(
        df, country=country, maturity=maturity, months=1
    )
    change_1y = _yield_change_bp(
        df, country=country, maturity=maturity, months=12
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nykyinen korko", _fmt_yield(latest["Yield"]))
    c2.metric("Muutos 1 kk", _fmt_bp(change_1m))
    c3.metric("Muutos 1 v", _fmt_bp(change_1y))
    c4.metric(
        f"{period} vaihteluväli",
        f"{subset['Yield'].min():.2f}–{subset['Yield'].max():.2f} %",
    )

    st.caption(f"Viimeisin havainto: {pd.to_datetime(latest['Date']).date()}")

    fig = px.line(
        subset,
        x="Date",
        y="Yield",
        title=(
            f"{_country_label(country)} – "
            f"{_maturity_label(maturity)} valtionlainan tuotto"
        ),
        labels={"Date": "Päivä", "Yield": "Tuotto (%)"},
    )
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(ticksuffix=" %")
    st.plotly_chart(fig, use_container_width=True)


def _render_curve() -> None:
    st.markdown("### 〽️ Tuottokäyrä")
    st.caption(
        "Nykyinen 2Y/5Y/10Y/30Y-tuottokäyrä verrattuna noin vuoden takaiseen."
    )

    country = st.selectbox(
        "Valitse maa",
        COUNTRY_ORDER,
        format_func=_country_label,
        key="rates_curve_country",
    )

    with st.spinner(f"Ladataan {_country_label(country)} tuottokäyrä…"):
        bundle = load_country_curve_history(country, years=2)

    df = bundle["data"]

    if df is None or df.empty:
        st.warning(bundle["error"] or "Tuottokäyrää ei saatu.")
        return

    current_curve = build_yield_curve_snapshot(df, country=country)

    if current_curve.empty:
        st.info("Tuottokäyrää ei saatu valitulle maalle.")
        return

    current_date = pd.to_datetime(current_curve["Date"]).max()
    previous_curve = _curve_at_date(
        df,
        country=country,
        target_date=current_date - pd.DateOffset(years=1),
    )

    spread_df = build_10y_2y_spread(df, country=country)
    current_spread = (
        float(spread_df.iloc[-1]["Spread_10Y_2Y"])
        if not spread_df.empty
        else None
    )

    icon, status = _curve_status(current_spread)

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "10Y–2Y",
        _fmt_bp(current_spread * 100.0) if current_spread is not None else "–",
    )
    c2.metric("Käyrän muoto", f"{icon} {status}")
    c3.metric("Viimeisin data", str(current_date.date()))

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=current_curve["MaturityYears"],
            y=current_curve["Yield"],
            mode="lines+markers",
            name=f"Nykyinen ({current_date.date()})",
        )
    )

    if not previous_curve.empty:
        previous_date = pd.to_datetime(previous_curve["Date"]).max()
        fig.add_trace(
            go.Scatter(
                x=previous_curve["MaturityYears"],
                y=previous_curve["Yield"],
                mode="lines+markers",
                name=f"Noin 1 v sitten ({previous_date.date()})",
            )
        )

    fig.update_layout(
        title=f"{_country_label(country)} – tuottokäyrä",
        xaxis_title="Maturiteetti",
        yaxis_title="Tuotto (%)",
        hovermode="x unified",
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=[2, 5, 10, 30],
        ticktext=["2Y", "5Y", "10Y", "30Y"],
    )
    fig.update_yaxes(ticksuffix=" %")
    st.plotly_chart(fig, use_container_width=True)

    table = current_curve[["Maturity", "Yield"]].copy()
    table.columns = ["Maturiteetti", "Nykyinen"]

    if not previous_curve.empty:
        old = previous_curve[["Maturity", "Yield"]].copy()
        old.columns = ["Maturiteetti", "1 v sitten"]
        table = table.merge(old, on="Maturiteetti", how="left")
        table["Muutos (bp)"] = (
            table["Nykyinen"] - table["1 v sitten"]
        ) * 100.0

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Nykyinen": st.column_config.NumberColumn(
                "Nykyinen", format="%.2f %%"
            ),
            "1 v sitten": st.column_config.NumberColumn(
                "1 v sitten", format="%.2f %%"
            ),
            "Muutos (bp)": st.column_config.NumberColumn(
                "Muutos", format="%+.0f bp"
            ),
        },
    )


def render() -> None:
    st.subheader("📈 Korot ja velkakirjat")
    st.caption(
        "Valtionlainojen korot, korkojen kehitys ja tuottokäyrät "
        "Yhdysvalloissa, Saksassa ja Japanissa."
    )

    view = st.radio(
        "Valitse näkymä",
        ["🌍 Yleiskuva", "📈 Korkokehitys", "〽️ Tuottokäyrä"],
        horizontal=True,
        label_visibility="collapsed",
        key="rates_view",
    )

    st.divider()

    # Tärkein optimointi:
    # yleiskuvan dataa ei ladata lainkaan, jos käyttäjä avaa suoraan
    # historia- tai tuottokäyränäkymän.
    if view == "🌍 Yleiskuva":
        with st.spinner("Ladataan korkomarkkinoiden yleiskuva…"):
            bundle = load_rates_overview()

        df = bundle["data"]

        if df is None or df.empty:
            st.warning("Korkomarkkinoiden yleiskuvaa ei saatu ladattua.")
        else:
            _render_overview(df)

        debug = bundle.get("debug", {})
        errors = {
            country: message
            for country, message in debug.items()
            if message
        }

        if errors:
            with st.expander("Tekninen lähdehuomautus", expanded=False):
                for country, message in errors.items():
                    st.caption(f"{_country_label(country)}: {message}")

    elif view == "📈 Korkokehitys":
        _render_history()

    elif view == "〽️ Tuottokäyrä":
        _render_curve()

    with st.expander("Datalähteet", expanded=False):
        st.markdown(
            """
- **Yhdysvallat:** FRED / Federal Reserve
- **Saksa:** Deutsche Bundesbank
- **Japani:** Ministry of Finance Japan

Britannia on jätetty tästä versiosta toistaiseksi pois, jotta korko-osio
latautuu nopeasti ilman Bank of Englandin raskaan arkistotiedoston käsittelyä.
            """
        )

        st.caption(
            "Maiden virallisten korkosarjojen laskentamenetelmät eivät ole "
            "täysin identtisiä. Maiden välisiä korkoeroja tulee siksi tulkita "
            "markkinaindikaattoreina."
        )
