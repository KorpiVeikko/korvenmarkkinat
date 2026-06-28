# tabs/bitcoin.py
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from services.market_data import fetch_price_history, fetch_price_history_eur
from services.asset_ui import (
    latest_period_values,
    pct_change,
    filter_by_period,
    period_selector,
    render_price_chart_with_extra_lines,
)


HALVING_DATES = [
    ("2012-11-28", "Halving 2012"),
    ("2016-07-09", "Halving 2016"),
    ("2020-05-11", "Halving 2020"),
    ("2024-04-20", "Halving 2024"),
]


def _fmt_money(x: float | None, currency: str = "€", decimals: int = 0) -> str:
    if x is None or pd.isna(x):
        return "—"

    x = float(x)
    ax = abs(x)

    if ax >= 1_000_000_000_000:
        return f"{x / 1_000_000_000_000:,.2f}".replace(",", " ") + f" bilj. {currency}"
    if ax >= 1_000_000_000:
        return f"{x / 1_000_000_000:,.2f}".replace(",", " ") + f" mrd {currency}"
    if ax >= 1_000_000:
        return f"{x / 1_000_000:,.2f}".replace(",", " ") + f" milj. {currency}"

    return f"{x:,.{decimals}f}".replace(",", " ") + f" {currency}"


def _latest_valid(series: pd.Series) -> float | None:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.iloc[-1])


def _pct_color(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "#6b7280"
    return "#15803d" if value >= 0 else "#b91c1c"


def _pct_text(value: float | None, label: str = "") -> str:
    if value is None or pd.isna(value):
        return "—"
    suffix = f" ({label})" if label else ""
    return f"{value:+.1f} %{suffix}"


def _pct_html(value: float | None, label: str = "") -> str:
    return f"""
    <span style="
        color:{_pct_color(value)};
        font-weight:700;
        font-size:1.05rem;
    ">
        {_pct_text(value, label)}
    </span>
    """


def _price_card(label: str, value: float | None, pct: float | None = None, caption: str | None = None) -> None:
    with st.container(border=True):
        st.caption(label)
        st.markdown(f"## {_fmt_money(value, '€', 0)}")
        if pct is not None and not pd.isna(pct):
            st.markdown(_pct_html(pct), unsafe_allow_html=True)
        if caption:
            st.caption(caption)


def _status_from_drawdown(drawdown: float | None) -> tuple[str, str]:
    if drawdown is None or pd.isna(drawdown):
        return "⚪", "Ei dataa"

    if drawdown > -10:
        return "🟢", "Lähellä huippuja"
    if drawdown > -25:
        return "🟡", "Normaali korjausliike"
    if drawdown > -50:
        return "🟠", "Selvä laskuvaihe"
    return "🔴", "Syvä drawdown"


def _status_from_trend(price: float | None, ma200: float | None) -> tuple[str, str]:
    if price is None or ma200 is None or pd.isna(price) or pd.isna(ma200):
        return "⚪", "Ei dataa"

    if price > ma200 * 1.15:
        return "🟢", "Vahva nousutrendi"
    if price > ma200:
        return "🟢", "Nousutrendi"
    if price > ma200 * 0.85:
        return "🟡", "Lähellä trendirajaa"
    return "🔴", "Alle 200 pv trendin"


def _add_halving_lines(fig, plot_df: pd.DataFrame, _period: str):
    if plot_df is None or plot_df.empty:
        return fig

    min_date = pd.to_datetime(plot_df["Date"], errors="coerce").min()
    max_date = pd.to_datetime(plot_df["Date"], errors="coerce").max()

    for date_str, label in HALVING_DATES:
        d = pd.to_datetime(date_str)
        if min_date <= d <= max_date:
            fig.add_vline(x=d, line_dash="dash", opacity=0.6)
            fig.add_annotation(
                x=d,
                y=1,
                yref="paper",
                text=label,
                showarrow=False,
                yanchor="bottom",
                textangle=-90,
            )

    return fig


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def load_btc_eur() -> pd.DataFrame:
    return fetch_price_history_eur("BTC-USD", period="10y")


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def load_btc_usd() -> pd.DataFrame:
    return fetch_price_history("BTC-USD", period="10y")


def _render_signal_cards(
    btc_vals: dict,
    drawdown_from_ath: float | None,
    vol30: float | None,
    ma200_latest: float | None,
) -> None:
    now = btc_vals.get("now")
    trend_icon, trend_label = _status_from_trend(now, ma200_latest)
    dd_icon, dd_label = _status_from_drawdown(drawdown_from_ath)

    st.markdown("### 📌 Tilannekuva")

    c1, c2, c3 = st.columns(3)

    with c1:
        with st.container(border=True):
            st.markdown(f"### {trend_icon} Trendi")
            st.markdown(f"**Tila:** {trend_label}")
            st.caption("Vertailu 200 päivän liukuvaan keskiarvoon.")
            if now is not None and ma200_latest is not None:
                diff = (now / ma200_latest - 1.0) * 100.0
                st.markdown(_pct_html(diff, "vs MA200"), unsafe_allow_html=True)

    with c2:
        with st.container(border=True):
            st.markdown(f"### {dd_icon} ATH-etäisyys")
            st.markdown(f"**Tila:** {dd_label}")
            st.caption("Kuinka kaukana nykyhinta on kaikkien aikojen huipusta.")
            st.markdown(_pct_html(drawdown_from_ath, "ATH"), unsafe_allow_html=True)

    with c3:
        with st.container(border=True):
            st.markdown("### ⚡ Volatiliteetti")
            if vol30 is None or pd.isna(vol30):
                st.markdown("**Tila:** Ei dataa")
                st.markdown("—")
            elif vol30 > 90:
                st.markdown("**Tila:** Erittäin heiluvainen")
                st.markdown(f"**{vol30:.1f} %**")
            elif vol30 > 55:
                st.markdown("**Tila:** Heiluvainen")
                st.markdown(f"**{vol30:.1f} %**")
            else:
                st.markdown("**Tila:** Rauhallisempi")
                st.markdown(f"**{vol30:.1f} %**")
            st.caption("30 päivän annualisoitu volatiliteetti.")


def _render_analysis(
    btc_vals: dict,
    drawdown_from_ath: float | None,
    vol30: float | None,
    ma200_latest: float | None,
    vol_now: float | None,
    vol_30avg: float | None,
) -> None:
    now = btc_vals.get("now")
    pct_1m = btc_vals.get("pct_1m")
    pct_1y = btc_vals.get("pct_1y")

    parts = []

    if now is not None and ma200_latest is not None:
        if now > ma200_latest:
            parts.append("Bitcoin on 200 päivän keskiarvon yläpuolella, mikä viittaa teknisesti vahvempaan trendiin.")
        else:
            parts.append("Bitcoin on 200 päivän keskiarvon alapuolella, mikä kertoo varovaisemmasta trendikuvasta.")

    if pct_1m is not None:
        if pct_1m > 10:
            parts.append("Lyhyen aikavälin 1 kuukauden kehitys on selvästi positiivinen.")
        elif pct_1m < -10:
            parts.append("Lyhyen aikavälin 1 kuukauden kehitys on selvästi negatiivinen.")
        else:
            parts.append("Viimeisen kuukauden liike on maltillisempi.")

    if pct_1y is not None:
        if pct_1y > 40:
            parts.append("Vuoden kehitys on erittäin vahva, mutta samalla korjausliikkeiden riski voi kasvaa.")
        elif pct_1y < -20:
            parts.append("Vuoden kehitys on heikko, mikä kertoo markkinan varovaisuudesta.")
        else:
            parts.append("Vuoden kehitys on neutraalimpi eikä yksin kerro vahvasta suhdannevaiheesta.")

    if drawdown_from_ath is not None:
        if drawdown_from_ath > -10:
            parts.append("Hinta on lähellä huippuja, joten markkina hinnoittelee optimistista näkymää.")
        elif drawdown_from_ath < -40:
            parts.append("Hinta on selvästi huippujen alapuolella, mikä kertoo voimakkaasta riskin vähentämisestä tai aiemmasta ylikuumenemisesta.")

    if vol30 is not None:
        if vol30 > 80:
            parts.append("Volatiliteetti on korkea, joten lyhyen aikavälin heilunta voi olla voimakasta.")
        elif vol30 < 45:
            parts.append("Volatiliteetti on bitcoinille verrattain rauhallinen.")

    if vol_now is not None and vol_30avg is not None and vol_30avg != 0:
        vol_diff = (vol_now / vol_30avg - 1.0) * 100.0

        if vol_diff > 30:
            parts.append("Kaupankäyntivolyymi on selvästi 30 päivän keskiarvoa korkeampi, mikä kertoo markkinakiinnostuksen tai myynti-/ostopaineen kasvusta.")
        elif vol_diff < -30:
            parts.append("Kaupankäyntivolyymi on selvästi 30 päivän keskiarvoa matalampi, mikä voi viitata vaisumpaan markkina-aktiivisuuteen.")
        else:
            parts.append("Kaupankäyntivolyymi on melko lähellä 30 päivän keskiarvoa.")

    if not parts:
        parts.append("Analyysia ei voitu muodostaa, koska keskeisiä tunnuslukuja puuttuu.")

    st.markdown("### 🧠 Bitcoin-analyysi")
    with st.container(border=True):
        st.write(" ".join(parts))

    st.info(
        "Tämä ei ole sijoitussuositus. Bitcoin on korkean riskin omaisuuserä, "
        "jonka hinta voi muuttua nopeasti."
    )


def render() -> None:
    st.subheader("₿ Bitcoin")
    st.caption("Lähde: Yahoo Finance. Euromääräinen hinta muodostetaan BTC/USD- ja EUR/USD-sarjoista.")

    btc_eur_df = load_btc_eur()
    btc_usd_df = load_btc_usd()

    if btc_eur_df is None or btc_eur_df.empty or btc_usd_df is None or btc_usd_df.empty:
        st.error("Bitcoin-dataa ei saatu.")
        return

    btc_eur_df = btc_eur_df.copy()
    btc_usd_df = btc_usd_df.copy()

    btc_eur_df["Date"] = pd.to_datetime(btc_eur_df["Date"], errors="coerce")
    btc_usd_df["Date"] = pd.to_datetime(btc_usd_df["Date"], errors="coerce")

    btc_eur_df["Close"] = pd.to_numeric(btc_eur_df["Close"], errors="coerce")
    btc_usd_df["Close"] = pd.to_numeric(btc_usd_df["Close"], errors="coerce")

    btc_eur_df = btc_eur_df.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)
    btc_usd_df = btc_usd_df.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)

    latest_date = max(btc_eur_df["Date"].max(), btc_usd_df["Date"].max())
    st.caption(f"Viimeisin markkinadata: {latest_date.date()}")

    btc_vals = latest_period_values(btc_eur_df, "Close")

    ath_usd = float(pd.to_numeric(btc_usd_df["Close"], errors="coerce").max())
    atl_usd = float(pd.to_numeric(btc_usd_df["Close"], errors="coerce").min())
    ath_eur = float(pd.to_numeric(btc_eur_df["Close"], errors="coerce").max())
    atl_eur = float(pd.to_numeric(btc_eur_df["Close"], errors="coerce").min())

    drawdown_from_ath = pct_change(btc_vals["now"], ath_eur)

    btc_eur_df["ret"] = btc_eur_df["Close"].pct_change()
    vol30 = btc_eur_df["ret"].tail(30).std() * np.sqrt(365) * 100 if len(btc_eur_df) >= 30 else None

    vol_now = None
    vol_30avg = None
    if "Volume" in btc_usd_df.columns:
        btc_usd_df["Volume"] = pd.to_numeric(btc_usd_df["Volume"], errors="coerce")
        vol_now = _latest_valid(btc_usd_df["Volume"])
        vol_30avg = float(btc_usd_df["Volume"].tail(30).mean()) if len(btc_usd_df) >= 30 else None

    plot_df = btc_eur_df.copy()
    plot_df["Close_EUR"] = plot_df["Close"]
    plot_df["MA200_EUR"] = plot_df["Close_EUR"].rolling(200).mean()
    ma200_latest = _latest_valid(plot_df["MA200_EUR"])

    st.markdown("### 💶 Hinta euroissa")

    left, _ = st.columns([0.42, 0.58])

    with left:
        with st.container(border=True):
            st.markdown("### ₿ Bitcoin")
            st.caption("Nykyinen hinta")
            st.markdown(f"# {_fmt_money(btc_vals['now'], '€', 0)}")

            st.divider()

            for label, pct in [
                ("1 kk", btc_vals.get("pct_1m")),
                ("1 vuosi", btc_vals.get("pct_1y")),
                ("5 vuotta", btc_vals.get("pct_5y")),
            ]:
                icon = "↗" if pct is not None and pct >= 0 else "↘"
                color = "#15803d" if pct is not None and pct >= 0 else "#b91c1c"

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
                        <span style="color:{color}; font-weight:700;">{_pct_text(pct)}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.divider()


    tab_price, tab_drawdown, tab_volume, tab_analysis = st.tabs(
        ["💹 Hinta", "📉 Drawdown", "📦 Volyymi", "🧠 Analyysi"]
    )

    
    with tab_price:
        render_price_chart_with_extra_lines(
            plot_df,
            title="Bitcoin (€)",
            key="btc_price_period",
            base_col="Close_EUR",
            extra_lines=[
                ("MA200_EUR", "MA200", {}),
            ],
            y_title="EUR",
            options=["1 kk", "1 v", "5 v", "10 v"],
            default="1 v",
            postprocess=_add_halving_lines,
        )

   

    with tab_drawdown:
        st.markdown("### 📉 Drawdown")

        c1, c2, c3 = st.columns(3)

        with c1:
            _price_card("ATH", ath_eur, None)
            st.caption(_fmt_money(ath_usd, "$", 0))

        with c2:
            _price_card("ATL", atl_eur, None)
            st.caption(_fmt_money(atl_usd, "$", 0))

        with c3:
            with st.container(border=True):
                st.caption("ATH-drawdown")
                st.markdown(
                    f"## {drawdown_from_ath:+.1f} %"
                    if drawdown_from_ath is not None
                    else "## —"
                )
                st.caption("Nykyhinta vs kaikkien aikojen huippu")

        period = period_selector(
            "Kuvaajan tarkasteluväli",
            key="btc_dd_period",
            options=["1 kk", "1 v", "5 v", "10 v"],
            default="1 v",
        )

        dd = plot_df.dropna(subset=["Close_EUR"]).copy()
        dd["rolling_ath"] = dd["Close_EUR"].cummax()
        dd["drawdown_pct"] = (dd["Close_EUR"] / dd["rolling_ath"] - 1.0) * 100.0
        dd["drawdown_pct"] = dd["drawdown_pct"].clip(upper=0)

        dd_plot = filter_by_period(dd, period, date_col="Date")

        y_min = float(dd_plot["drawdown_pct"].min()) if not dd_plot.empty else -10.0
        y_min = min(y_min - 3.0, -5.0)

        fig = px.line(
            dd_plot,
            x="Date",
            y="drawdown_pct",
            title=f"Bitcoinin drawdown ATH:sta ({period})",
            labels={"Date": "Päivä", "drawdown_pct": "%"},
        )
        fig.update_yaxes(range=[y_min, 5], ticksuffix=" %")
        fig.add_hline(y=0, line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)

    with tab_volume:
        st.markdown("### 📦 Volyymi")

        v1, v2 = st.columns(2)

        with v1:
            with st.container(border=True):
                st.caption("Volyymi nyt")
                st.markdown(f"## {_fmt_money(vol_now, '', 0).strip()}")
                st.caption("Yahoo Finance volume")

        with v2:
            with st.container(border=True):
                st.caption("Volyymi, 30 pv ka")
                st.markdown(f"## {_fmt_money(vol_30avg, '', 0).strip()}")
                st.caption("Keskiarvo")

        period = period_selector(
            "Kuvaajan tarkasteluväli",
            key="btc_vol_period",
            options=["1 kk", "1 v", "5 v", "10 v"],
            default="1 v",
        )

        if "Volume" not in btc_usd_df.columns or btc_usd_df["Volume"].dropna().empty:
            st.info("Volyymihistoriaa ei löytynyt tästä aineistosta.")
        else:
            vol_plot_df = filter_by_period(
                btc_usd_df.dropna(subset=["Volume"]).copy(),
                period,
                date_col="Date",
            )

            vol_plot_df["Volume_B"] = vol_plot_df["Volume"] / 1_000_000_000

            fig = px.bar(
                vol_plot_df,
                x="Date",
                y="Volume_B",
                title=f"Bitcoinin volyymi ({period})",
                labels={"Date": "Päivä", "Volume_B": "Volyymi (mrd)"},
            )

            fig.update_yaxes(ticksuffix=" mrd")
            st.plotly_chart(fig, use_container_width=True)

    with tab_analysis:

        _render_signal_cards(
            btc_vals=btc_vals,
            drawdown_from_ath=drawdown_from_ath,
            vol30=vol30,
            ma200_latest=ma200_latest,
        )

        st.divider()

        _render_analysis(
            btc_vals=btc_vals,
            drawdown_from_ath=drawdown_from_ath,
            vol30=vol30,
            ma200_latest=ma200_latest,
            vol_now=vol_now,
            vol_30avg=vol_30avg,
        )