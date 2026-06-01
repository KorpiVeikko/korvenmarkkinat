from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from services.macro_data import (
    build_total_flow_from_products,
    build_trade_balance,
    clip_by_years,
    fmt,
    fmt_money,
    latest_row_by_date,
)
from services.macro_uljas import (
    fetch_exports_products,
    fetch_exports_regions,
    fetch_imports_products,
    fetch_imports_regions,
)

EXPORT_CFG = {
    "key": "export",
    "title": "🚢 Vienti – rakenne (Tulli / Uljas)",
    "caption": "Tavaravienti SITC-luokituksella kuukausitasolla vuosille 2020–2026.",
    "value_col": "Vienti_eur",
    "group_col": "Tuoteryhmä",
    "region_col": "Alue",
    "metric_label": "Tavaravienti",
    "flow_name": "viennin",
    "regions_title": "Vienti maanosittain",
    "products_title": "Vienti tuoteryhmittäin",
    "region_detail_title": "#### 🔎 Maanosan tarkastelu",
    "group_detail_title": "#### 🔎 Tuoteryhmän tarkastelu",
    "group_select_label": "Valitse tarkasteltava tuoteryhmä",
    "region_select_label": "Valitse tarkasteltava maanosa",
    "products_empty_msg": "Tuoteryhmävientiä ei saatu ladattua (Uljas).",
    "regions_empty_msg": "Maanosavientiä ei saatu ladattua (Uljas).",
    "fetch_products": fetch_exports_products,
    "fetch_regions": fetch_exports_regions,
}

IMPORT_CFG = {
    "key": "import",
    "title": "📥 Tuonti – rakenne (Tulli / Uljas)",
    "caption": "Tavaratuonti SITC-luokituksella kuukausitasolla vuosille 2020–2026.",
    "value_col": "Tuonti_eur",
    "group_col": "Tuoteryhmä",
    "region_col": "Alue",
    "metric_label": "Tavaratuonti",
    "flow_name": "tuonnin",
    "regions_title": "Tuonti maanosittain",
    "products_title": "Tuonti tuoteryhmittäin",
    "region_detail_title": "#### 🔎 Maanosan tarkastelu",
    "group_detail_title": "#### 🔎 Tuoteryhmän tarkastelu",
    "group_select_label": "Valitse tarkasteltava tuoteryhmä",
    "region_select_label": "Valitse tarkasteltava maanosa",
    "products_empty_msg": "Tuonnin tuoteryhmäaineistoa ei saatu ladattua (Uljas).",
    "regions_empty_msg": "Maanosatuonnin aineistoa ei saatu ladattua (Uljas).",
    "fetch_products": fetch_imports_products,
    "fetch_regions": fetch_imports_regions,
}


def kpi_card(label: str, value: str, delta: str | None = None, caption: str | None = None) -> None:
    with st.container(border=True):
        st.metric(label, value, delta)
        if caption:
            st.caption(caption)


def show_debug_info(debug: dict) -> None:
    with st.expander("🔍 Debug", expanded=True):
        st.write("**Onnistunut ifile:**", debug.get("ok_ifile"))
        st.write("**Aikakoodit:**", debug.get("time_codes"))
        st.write("**Vuodet:**", debug.get("years"))
        st.write("**Virhesyyt:**")
        st.code("\n".join(debug.get("why_failed", [])) or "—")


def _yoy_delta(series: pd.Series, periods: int) -> float | None:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) <= periods:
        return None
    last = float(s.iloc[-1])
    then = float(s.iloc[-(periods + 1)])
    return last - then


def _add_zero_line(fig) -> None:
    fig.add_hline(y=0, line_color="red", line_width=1.5, opacity=0.9)


def fmt_delta_pct(val: float | None) -> str | None:
    if val is None or pd.isna(val):
        return None
    return f"{val:+.1f} %"


def _fmt_mio_eur(x: float | None, decimals: int = 0) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x:,.{decimals}f} milj. €".replace(",", " ")


def format_source_date(dt: pd.Timestamp | None, freq: str = "year") -> str:
    if dt is None or pd.isna(dt):
        return ""

    if freq == "year":
        return f"Vuosi {dt.year}"

    if freq == "month":
        months = [
            "tammikuu", "helmikuu", "maaliskuu", "huhtikuu",
            "toukokuu", "kesäkuu", "heinäkuu", "elokuu",
            "syyskuu", "lokakuu", "marraskuu", "joulukuu",
        ]
        return f"{months[dt.month - 1]} {dt.year}"

    return str(dt.date())


def yoy_pct_change(df: pd.DataFrame, date_col: str, value_col: str) -> float | None:
    if df is None or df.empty:
        return None

    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=[date_col, value_col]).sort_values(date_col)

    if len(d) < 2:
        return None

    latest = d.iloc[-1]
    prev = d.iloc[-2]

    if prev[value_col] == 0:
        return None

    return ((latest[value_col] / prev[value_col]) - 1.0) * 100.0


def _full_year_stats_from_mixed_series(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
) -> tuple[float | None, float | None, int | None]:
    if df is None or df.empty:
        return None, None, None

    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=[date_col, value_col])

    if d.empty:
        return None, None, None

    d["Vuosi"] = d[date_col].dt.year
    d["Kuukausi"] = d[date_col].dt.month

    counts = d.groupby("Vuosi")["Kuukausi"].nunique().reset_index(name="kk_lkm")
    full_years = counts[counts["kk_lkm"] >= 12]["Vuosi"].tolist()

    if not full_years:
        return None, None, None

    latest_full_year = max(full_years)

    yearly = d.groupby("Vuosi", as_index=False)[value_col].sum().sort_values("Vuosi")

    latest_row = yearly[yearly["Vuosi"] == latest_full_year]
    if latest_row.empty:
        return None, None, None

    latest_val = float(latest_row.iloc[0][value_col])

    prev_row = yearly[yearly["Vuosi"] == latest_full_year - 1]
    pct = None
    if not prev_row.empty:
        prev_val = float(prev_row.iloc[0][value_col])
        if prev_val != 0:
            pct = ((latest_val / prev_val) - 1.0) * 100.0

    return latest_val, pct, int(latest_full_year)


def _prepare_plot_df(df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=[date_col, value_col]).sort_values(date_col)
    return d


def _to_million_eur_plot_df(df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    d = _prepare_plot_df(df, date_col, value_col)
    if d.empty:
        return d

    d = d.copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce") / 1_000_000
    return d


def _required_points_for_full_year(d: pd.DataFrame, date_col: str) -> int:
    counts = d.groupby(d[date_col].dt.year)[date_col].nunique()
    if counts.empty:
        return 1

    max_count = int(counts.max())
    if max_count >= 12:
        return 12
    if max_count >= 4:
        return 4
    return 1


def _to_yearly_sum_df(df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    d = _prepare_plot_df(df, date_col, value_col)
    if d.empty:
        return d

    d["Vuosi"] = d[date_col].dt.year
    counts = d.groupby("Vuosi")[date_col].nunique().reset_index(name="havaintoja")
    required_points = _required_points_for_full_year(d, date_col)
    full_years = counts[counts["havaintoja"] >= required_points]["Vuosi"].tolist()

    yearly = d.groupby("Vuosi", as_index=False)[value_col].sum().sort_values("Vuosi")

    if full_years:
        yearly = yearly[yearly["Vuosi"].isin(full_years)].copy()

    return yearly.reset_index(drop=True)


def _to_yearly_mean_df(df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    d = _prepare_plot_df(df, date_col, value_col)
    if d.empty:
        return d

    d["Vuosi"] = d[date_col].dt.year
    counts = d.groupby("Vuosi")[date_col].nunique().reset_index(name="havaintoja")
    required_points = _required_points_for_full_year(d, date_col)
    full_years = counts[counts["havaintoja"] >= required_points]["Vuosi"].tolist()

    yearly = (
        d.groupby("Vuosi", as_index=False)[value_col]
        .mean()
        .sort_values("Vuosi")
    )

    if full_years:
        yearly = yearly[yearly["Vuosi"].isin(full_years)].copy()

    return yearly.reset_index(drop=True)


def _stacked_bar_chart(
    df: pd.DataFrame,
    date_col: str,
    category_col: str,
    value_col: str,
    title: str,
    key: str,
) -> None:
    if df is None or df.empty:
        st.info("Ei dataa näytettäväksi.")
        return

    d = _to_million_eur_plot_df(df, date_col, value_col)
    d = d.dropna(subset=[category_col])

    if d.empty:
        st.info("Ei dataa näytettäväksi.")
        return

    d["Aika_label"] = d[date_col].dt.strftime("%Y-%m")

    fig = px.bar(
        d,
        x="Aika_label",
        y=value_col,
        color=category_col,
        barmode="stack",
        title=title,
        labels={"Aika_label": "Aika", value_col: "Arvo (milj. €)", category_col: category_col},
    )
    fig.update_layout(xaxis_title="Aika", yaxis_title="Arvo (milj. €)")
    st.plotly_chart(fig, width="stretch", key=key)


def _combined_monthly_and_trend_chart(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    title: str,
    y_label: str,
    key: str,
) -> None:
    if df is None or df.empty:
        st.info("Ei dataa näytettäväksi.")
        return

    d = _prepare_plot_df(df, date_col, value_col)
    if d.empty:
        st.info("Ei dataa näytettäväksi.")
        return

    d = d.copy()
    d["monthly_milj_eur"] = pd.to_numeric(d[value_col], errors="coerce") / 1_000_000
    d["rolling_12m_sum_milj_eur"] = (
        pd.to_numeric(d[value_col], errors="coerce").rolling(window=12, min_periods=12).sum() / 1_000_000
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=d[date_col],
            y=d["monthly_milj_eur"],
            mode="lines+markers",
            name="Kuukausisarja",
            line=dict(width=2),
            yaxis="y1",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=d[date_col],
            y=d["rolling_12m_sum_milj_eur"],
            mode="lines",
            name="12 kk liukuva summa",
            line=dict(width=4),
            yaxis="y2",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Aika",
        yaxis=dict(
            title="Kuukausisarja (milj. €)",
            side="left",
        ),
        yaxis2=dict(
            title="12 kk liukuva summa (milj. €)",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        legend_title="Sarja",
        hovermode="x unified",
    )

    st.plotly_chart(fig, width="stretch", key=key)


def _yearly_line_chart(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    title: str,
    y_label: str,
    key: str,
) -> None:
    yearly = _to_yearly_sum_df(df, date_col, value_col)
    if yearly is None or yearly.empty:
        st.info("Ei vuositason dataa näytettäväksi.")
        return

    yearly = yearly.copy()
    yearly[value_col] = pd.to_numeric(yearly[value_col], errors="coerce") / 1_000_000
    yearly["Vuosi_label"] = yearly["Vuosi"].astype(str)

    fig = px.line(
        yearly,
        x="Vuosi_label",
        y=value_col,
        markers=True,
        title=title,
        labels={"Vuosi_label": "Vuosi", value_col: y_label},
    )
    fig.update_layout(xaxis_title="Vuosi", yaxis_title=y_label)
    st.plotly_chart(fig, width="stretch", key=key)


def _latest_value_and_date(df: pd.DataFrame, date_col: str, value_col: str) -> tuple[float | None, pd.Timestamp | None]:
    if df is None or df.empty or value_col not in df.columns:
        return None, None

    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=[date_col, value_col]).sort_values(date_col)

    if d.empty:
        return None, None

    return float(d.iloc[-1][value_col]), pd.to_datetime(d.iloc[-1][date_col])

def _gdp_status(value: float | None) -> tuple[str, str]:
    if value is None or pd.isna(value):
        return "⚪", "Ei dataa"
    if value >= 1.5:
        return "🟢", "Kasvu vahvistuu"
    if value >= 0:
        return "🟡", "Hidas kasvu"
    return "🔴", "Talous supistuu"


def _component_status(value: float | None, name: str) -> tuple[str, str]:
    if value is None or pd.isna(value):
        return "⚪", "Ei dataa"

    if name == "Investoinnit" and value < 0:
        return "🔴", "Investoinnit laskussa"

    if value >= 1:
        return "🟢", "Tukee kasvua"
    if value >= 0:
        return "🟡", "Heikko tuki"
    return "🔴", "Jarruttaa kasvua"


def _build_gdp_driver_text(
    gdp_now: float | None,
    private_consumption: float | None,
    investments: float | None,
) -> str:
    parts = []

    if gdp_now is not None:
        if gdp_now < 0:
            parts.append("BKT on supistumassa, joten talouden kokonaiskuva on heikko.")
        elif gdp_now < 1:
            parts.append("BKT kasvaa vain hitaasti, eli talous on lähellä nollakasvua.")
        else:
            parts.append("BKT kasvaa selvästi, mikä viittaa vahvempaan suhdannekuvaan.")

    if private_consumption is not None:
        if private_consumption > 1:
            parts.append("Yksityinen kulutus tukee kasvua.")
        elif private_consumption >= 0:
            parts.append("Yksityinen kulutus kasvaa vain maltillisesti.")
        else:
            parts.append("Yksityinen kulutus jarruttaa taloutta.")

    if investments is not None:
        if investments < 0:
            parts.append("Investointien lasku on selvä riskisignaali tulevalle kasvulle.")
        elif investments < 1:
            parts.append("Investoinnit ovat vaisut, mikä voi rajoittaa tulevaa kasvua.")
        else:
            parts.append("Investoinnit tukevat talouden kasvupohjaa.")

    if not parts:
        return "BKT:n ajureista ei saatu riittävästi dataa tulkinnan muodostamiseen."

    return " ".join(parts)


def _unemployment_status(value: float | None) -> tuple[str, str]:
    if value is None or pd.isna(value):
        return "⚪", "Ei dataa"

    if value < 7:
        return "🟢", "Matala"
    if value < 9:
        return "🟡", "Koholla"
    return "🔴", "Korkea"


def _unemployment_trend_status(delta_pp: float | None) -> tuple[str, str]:
    if delta_pp is None or pd.isna(delta_pp):
        return "⚪", "Ei dataa"

    if delta_pp <= -0.5:
        return "🟢", "Paranee"
    if delta_pp < 0.5:
        return "🟡", "Vakaa"
    return "🔴", "Heikkenee"


def _build_unemployment_text(
    latest_rate: float | None,
    rate_delta_1y: float | None,
    latest_level: float | None,
    level_delta_1y: float | None,
) -> str:
    parts = []

    if latest_rate is not None:
        if latest_rate >= 9:
            parts.append("Työttömyysaste on korkea, mikä kertoo työmarkkinan selvästä paineesta.")
        elif latest_rate >= 7:
            parts.append("Työttömyysaste on koholla, mutta ei vielä erittäin heikolla tasolla.")
        else:
            parts.append("Työttömyysaste on matala, mikä tukee kotitalouksien ostovoimaa.")

    if rate_delta_1y is not None:
        if rate_delta_1y >= 0.5:
            parts.append("Työttömyysaste on noussut vuoden aikana, eli työmarkkinatilanne on heikentynyt.")
        elif rate_delta_1y <= -0.5:
            parts.append("Työttömyysaste on laskenut vuoden aikana, mikä viittaa työmarkkinan paranemiseen.")
        else:
            parts.append("Työttömyysaste on pysynyt melko vakaana vuoden takaiseen nähden.")

    if level_delta_1y is not None:
        if level_delta_1y > 5:
            parts.append("Työttömien henkilöiden määrä on kasvanut selvästi.")
        elif level_delta_1y < -5:
            parts.append("Työttömien henkilöiden määrä on vähentynyt selvästi.")
        else:
            parts.append("Työttömien henkilöiden määrä on muuttunut vain maltillisesti.")

    if not parts:
        return "Työttömyyden tulkintaa ei voitu muodostaa puuttuvien tietojen vuoksi."

    return " ".join(parts)


def _wage_power_status(real_yoy: float | None) -> tuple[str, str]:
    if real_yoy is None or pd.isna(real_yoy):
        return "⚪", "Ei dataa"

    if real_yoy >= 1:
        return "🟢", "Ostovoima paranee"
    if real_yoy > -1:
        return "🟡", "Ostovoima vakaa"
    return "🔴", "Ostovoima heikkenee"


def _pct_change_over_years(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    years: int,
) -> float | None:
    if df is None or df.empty or value_col not in df.columns:
        return None

    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=[date_col, value_col]).sort_values(date_col)

    if d.empty:
        return None

    latest_date = d.iloc[-1][date_col]
    latest_val = float(d.iloc[-1][value_col])

    target = latest_date - pd.DateOffset(years=years)
    prev = d[d[date_col] <= target]

    if prev.empty:
        return None

    prev_val = float(prev.iloc[-1][value_col])
    if prev_val == 0:
        return None

    return (latest_val / prev_val - 1.0) * 100.0


def _build_wage_text(
    wage_yoy: float | None,
    index_yoy: float | None,
    real_yoy: float | None,
    wage_5y: float | None,
    real_5y: float | None,
) -> str:
    parts = []

    if wage_yoy is not None:
        if wage_yoy > 3:
            parts.append("Nimellispalkat kasvavat selvästi.")
        elif wage_yoy > 0:
            parts.append("Nimellispalkat kasvavat maltillisesti.")
        else:
            parts.append("Nimellispalkat eivät kasva, mikä heikentää tulokehitystä.")

    if real_yoy is not None:
        if real_yoy > 1:
            parts.append("Reaaliansiot ovat nousussa, eli palkat kasvavat hintoja nopeammin.")
        elif real_yoy > -1:
            parts.append("Reaaliansiot ovat melko vakaat, eli ostovoima ei juuri muutu.")
        else:
            parts.append("Reaaliansiot laskevat, eli hintojen nousu syö palkkakehitystä.")

    if wage_5y is not None and real_5y is not None:
        if wage_5y > real_5y + 5:
            parts.append("Viiden vuoden tarkastelussa nimellispalkat ovat nousseet selvästi enemmän kuin reaaliansiot, mikä kertoo inflaation painaneen ostovoimaa.")
        elif real_5y > 0:
            parts.append("Viiden vuoden tarkastelussa ostovoima on vahvistunut.")
        else:
            parts.append("Viiden vuoden tarkastelussa ostovoima on ollut paineessa.")

    if not parts:
        return "Palkkakehityksen tulkintaa ei voitu muodostaa puuttuvien tietojen vuoksi."

    return " ".join(parts)


def _trade_flow_status(pct: float | None) -> tuple[str, str]:
    if pct is None or pd.isna(pct):
        return "⚪", "Ei dataa"
    if pct >= 5:
        return "🟢", "Selvä kasvu"
    if pct >= 0:
        return "🟡", "Lievä kasvu"
    if pct > -5:
        return "🟠", "Lievä lasku"
    return "🔴", "Selvä lasku"


def _latest_full_year_group_stats(
    df: pd.DataFrame,
    date_col: str,
    group_col: str,
    value_col: str,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=[date_col, group_col, value_col])

    if d.empty:
        return pd.DataFrame()

    d["Vuosi"] = d[date_col].dt.year
    d["Kuukausi"] = d[date_col].dt.month

    counts = d.groupby("Vuosi")["Kuukausi"].nunique().reset_index(name="kk_lkm")
    full_years = counts[counts["kk_lkm"] >= 12]["Vuosi"].tolist()

    if not full_years:
        return pd.DataFrame()

    latest_year = max(full_years)
    prev_year = latest_year - 1

    yearly = (
        d[d["Vuosi"].isin([latest_year, prev_year])]
        .groupby(["Vuosi", group_col], as_index=False)[value_col]
        .sum()
    )

    latest = yearly[yearly["Vuosi"] == latest_year][[group_col, value_col]].rename(
        columns={value_col: "latest"}
    )
    prev = yearly[yearly["Vuosi"] == prev_year][[group_col, value_col]].rename(
        columns={value_col: "prev"}
    )

    out = latest.merge(prev, on=group_col, how="left")
    out["change_eur"] = out["latest"] - out["prev"]
    out["change_pct"] = ((out["latest"] / out["prev"]) - 1.0) * 100.0
    out.loc[out["prev"].isna() | (out["prev"] == 0), "change_pct"] = None
    out["share_pct"] = out["latest"] / out["latest"].sum() * 100.0
    out["Vuosi"] = latest_year

    return out.sort_values("latest", ascending=False).reset_index(drop=True)


def _trade_balance_status(balance_eur: float | None, change_eur: float | None) -> tuple[str, str]:
    if balance_eur is None or pd.isna(balance_eur):
        return "⚪", "Ei dataa"

    if balance_eur >= 0 and (change_eur is None or change_eur >= 0):
        return "🟢", "Ylijäämäinen / paranee"

    if balance_eur >= 0 and change_eur is not None and change_eur < 0:
        return "🟡", "Ylijäämäinen / heikkenee"

    if balance_eur < 0 and change_eur is not None and change_eur >= 0:
        return "🟡", "Alijäämäinen / paranee"

    return "🔴", "Alijäämäinen / heikkenee"


def _latest_full_year_trade_balance_stats(d: pd.DataFrame) -> dict:
    if d is None or d.empty:
        return {}

    x = d.copy()
    x["Aika_dt"] = pd.to_datetime(x["Aika_dt"], errors="coerce")

    for col in ["Vienti_eur", "Tuonti_eur", "Kauppatase_eur"]:
        x[col] = pd.to_numeric(x[col], errors="coerce")

    x = x.dropna(subset=["Aika_dt", "Vienti_eur", "Tuonti_eur", "Kauppatase_eur"])
    if x.empty:
        return {}

    x["Vuosi"] = x["Aika_dt"].dt.year
    x["Kuukausi"] = x["Aika_dt"].dt.month

    counts = x.groupby("Vuosi")["Kuukausi"].nunique().reset_index(name="kk_lkm")
    full_years = counts[counts["kk_lkm"] >= 12]["Vuosi"].tolist()

    if not full_years:
        return {}

    latest_year = max(full_years)
    prev_year = latest_year - 1

    yearly = (
        x[x["Vuosi"].isin([latest_year, prev_year])]
        .groupby("Vuosi", as_index=False)[["Vienti_eur", "Tuonti_eur", "Kauppatase_eur"]]
        .sum()
        .sort_values("Vuosi")
    )

    latest = yearly[yearly["Vuosi"] == latest_year]
    if latest.empty:
        return {}

    latest_row = latest.iloc[0]

    prev = yearly[yearly["Vuosi"] == prev_year]

    balance_change = None
    export_change = None
    import_change = None

    if not prev.empty:
        prev_row = prev.iloc[0]
        balance_change = float(latest_row["Kauppatase_eur"] - prev_row["Kauppatase_eur"])
        export_change = float(latest_row["Vienti_eur"] - prev_row["Vienti_eur"])
        import_change = float(latest_row["Tuonti_eur"] - prev_row["Tuonti_eur"])

    export_import_ratio = None
    if float(latest_row["Tuonti_eur"]) != 0:
        export_import_ratio = float(latest_row["Vienti_eur"] / latest_row["Tuonti_eur"] * 100.0)

    return {
        "year": int(latest_year),
        "exports": float(latest_row["Vienti_eur"]),
        "imports": float(latest_row["Tuonti_eur"]),
        "balance": float(latest_row["Kauppatase_eur"]),
        "balance_change": balance_change,
        "export_change": export_change,
        "import_change": import_change,
        "export_import_ratio": export_import_ratio,
    }


def _fmt_trade_change_eur(x: float | None) -> str | None:
    if x is None or pd.isna(x):
        return None

    sign = "+" if x > 0 else ""
    return f"{sign}{fmt_money(x)} (1 v)"


def _trade_balance_text(stats: dict) -> str:
    if not stats:
        return "Kauppataseen tulkintaa ei voitu muodostaa puuttuvien tietojen vuoksi."

    balance = stats.get("balance")
    balance_change = stats.get("balance_change")
    export_change = stats.get("export_change")
    import_change = stats.get("import_change")

    parts = []

    if balance is not None:
        if balance >= 0:
            parts.append("Kauppatase on ylijäämäinen eli vienti ylittää tuonnin.")
        else:
            parts.append("Kauppatase on alijäämäinen eli tuonti ylittää viennin.")

    if balance_change is not None:
        if balance_change > 0:
            parts.append("Kauppatase on parantunut vuodentakaiseen verrattuna.")
        elif balance_change < 0:
            parts.append("Kauppatase on heikentynyt vuodentakaiseen verrattuna.")
        else:
            parts.append("Kauppatase on lähes ennallaan vuodentakaiseen verrattuna.")

    if export_change is not None and import_change is not None:
        if export_change > 0 and import_change < 0:
            parts.append("Tilannetta parantaa se, että vienti kasvaa ja tuonti pienenee.")
        elif export_change < 0 and import_change > 0:
            parts.append("Tilannetta heikentää se, että vienti pienenee ja tuonti kasvaa.")
        elif export_change > import_change:
            parts.append("Vienti kehittyy tuontia vahvemmin, mikä tukee kauppatasetta.")
        elif import_change > export_change:
            parts.append("Tuonti kehittyy vientiä vahvemmin, mikä painaa kauppatasetta.")

    return " ".join(parts)


def _trade_flow_text(
    flow_name: str,
    total_pct: float | None,
    top_driver: str | None,
    top_drag: str | None,
) -> str:

    if flow_name.lower().startswith("vien"):
        subject = "Vienti"
    elif flow_name.lower().startswith("tuon"):
        subject = "Tuonti"
    else:
        subject = flow_name.capitalize()

    parts = []

    if total_pct is not None:
        if total_pct >= 5:
            parts.append(f"{subject} on selvässä kasvussa.")
        elif total_pct >= 0:
            parts.append(f"{subject} on lievässä kasvussa.")
        elif total_pct > -5:
            parts.append(f"{subject} on lievässä laskussa.")
        else:
            parts.append(f"{subject} on selvässä laskussa.")

    if top_driver:
        parts.append(f"Suurin positiivinen ajuri on {top_driver}.")

    if top_drag:
        parts.append(f"Suurin heikentäjä on {top_drag}.")

    return " ".join(parts)


def _debt_status(debt_pct: float | None, change_pct: float | None) -> tuple[str, str]:
    if debt_pct is None or pd.isna(debt_pct):
        return "⚪", "Ei dataa"

    if debt_pct < 60:
        return "🟢", "Maltillinen"

    if debt_pct < 90:
        if change_pct is not None and change_pct > 3:
            return "🟠", "Koholla / kasvaa"
        return "🟡", "Koholla"

    if change_pct is not None and change_pct > 3:
        return "🔴", "Korkea / kasvaa"

    return "🟠", "Korkea"


def _debt_direction_status(change_pct: float | None) -> tuple[str, str]:
    if change_pct is None or pd.isna(change_pct):
        return "⚪", "Ei dataa"

    if change_pct < -1:
        return "🟢", "Velkasuhde laskee"
    if change_pct <= 1:
        return "🟡", "Melko vakaa"
    return "🔴", "Velkasuhde kasvaa"


def _build_debt_text(
    debt_pct: float | None,
    pct_yoy: float | None,
    debt_eur: float | None,
    eur_yoy: float | None,
) -> str:
    parts = []

    if debt_pct is not None:
        if debt_pct < 60:
            parts.append("Julkinen velka suhteessa BKT:hen on maltillisella tasolla.")
        elif debt_pct < 90:
            parts.append("Julkinen velka suhteessa BKT:hen on koholla.")
        else:
            parts.append("Julkinen velka suhteessa BKT:hen on korkealla tasolla.")

    if pct_yoy is not None:
        if pct_yoy > 3:
            parts.append("Velkasuhde on kasvanut selvästi vuodessa, mikä heikentää julkisen talouden liikkumavaraa.")
        elif pct_yoy > 0:
            parts.append("Velkasuhde on noussut hieman vuodessa.")
        elif pct_yoy < -1:
            parts.append("Velkasuhde on laskenut, mikä tukee julkisen talouden kestävyyttä.")
        else:
            parts.append("Velkasuhde on pysynyt melko vakaana.")

    if eur_yoy is not None:
        if eur_yoy > 5:
            parts.append("Euromääräinen velka kasvaa nopeasti.")
        elif eur_yoy > 0:
            parts.append("Euromääräinen velka kasvaa edelleen.")
        elif eur_yoy < 0:
            parts.append("Euromääräinen velka on laskenut.")

    if not parts:
        return "Velkatilanteen tulkintaa ei voitu muodostaa puuttuvien tietojen vuoksi."

    return " ".join(parts)


def _private_debt_status(value: float | None, kind: str = "general") -> tuple[str, str]:
    if value is None or pd.isna(value):
        return "⚪", "Ei dataa"

    if kind == "household_gdi":
        if value < 100:
            return "🟢", "Maltillinen"
        if value < 140:
            return "🟡", "Koholla"
        return "🔴", "Korkea"

    if value < 80:
        return "🟢", "Maltillinen"
    if value < 120:
        return "🟡", "Koholla"
    return "🔴", "Korkea"


def _debt_change_pp(
    df: pd.DataFrame | None,
    date_col: str,
    value_col: str,
    periods: int = 1,
) -> float | None:
    if df is None or df.empty or value_col not in df.columns:
        return None

    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=[date_col, value_col]).sort_values(date_col)

    if len(d) <= periods:
        return None

    return float(d.iloc[-1][value_col] - d.iloc[-(periods + 1)][value_col])


def _private_debt_text(
    hh_gdi: float | None,
    hh_gdi_change: float | None,
    nfc_gdp: float | None,
    nfc_change: float | None,
    private_gdp: float | None,
    private_change: float | None,
    hh_loans_yoy: float | None,
    nfc_loans_yoy: float | None,
) -> str:
    parts = []

    if hh_gdi is not None:
        if hh_gdi >= 140:
            parts.append("Kotitalouksien velka suhteessa tuloihin on korkealla tasolla.")
        elif hh_gdi >= 100:
            parts.append("Kotitalouksien velka suhteessa tuloihin on koholla.")
        else:
            parts.append("Kotitalouksien velka suhteessa tuloihin on maltillisella tasolla.")

    if hh_gdi_change is not None:
        if hh_gdi_change > 2:
            parts.append("Kotitalouksien velkasuhde on noussut vuodessa.")
        elif hh_gdi_change < -2:
            parts.append("Kotitalouksien velkasuhde on laskenut vuodessa.")
        else:
            parts.append("Kotitalouksien velkasuhde on pysynyt melko vakaana.")

    if nfc_gdp is not None:
        if nfc_gdp >= 120:
            parts.append("Yritysvelka suhteessa BKT:hen on korkealla tasolla.")
        elif nfc_gdp >= 80:
            parts.append("Yritysvelka suhteessa BKT:hen on koholla.")
        else:
            parts.append("Yritysvelka suhteessa BKT:hen on maltillisella tasolla.")

    if private_gdp is not None and private_change is not None:
        if private_change > 2:
            parts.append("Yksityisen sektorin kokonaisvelka on noususuunnassa.")
        elif private_change < -2:
            parts.append("Yksityisen sektorin kokonaisvelka on laskusuunnassa.")

    if hh_loans_yoy is not None:
        if hh_loans_yoy > 5:
            parts.append("Kotitalouksien lainakanta kasvaa nopeasti.")
        elif hh_loans_yoy > 0:
            parts.append("Kotitalouksien lainakanta kasvaa maltillisesti.")
        elif hh_loans_yoy < 0:
            parts.append("Kotitalouksien lainakanta supistuu.")

    if nfc_loans_yoy is not None:
        if nfc_loans_yoy > 5:
            parts.append("Yritysten lainakanta kasvaa nopeasti.")
        elif nfc_loans_yoy > 0:
            parts.append("Yritysten lainakanta kasvaa maltillisesti.")
        elif nfc_loans_yoy < 0:
            parts.append("Yritysten lainakanta supistuu.")

    if not parts:
        return "Yksityisen velan tulkintaa ei voitu muodostaa puuttuvien tietojen vuoksi."

    return " ".join(parts)


def render_index_explainer(
    title: str = "ℹ️ Miten indeksejä luetaan?",
    include_real: bool = True,
) -> None:
    with st.expander(title, expanded=False):
        st.markdown(
            """
**Indeksi** kuvaa muutosta suhteessa perusvuoteen tai perusjaksoon.

- **Ansiotasoindeksi / palkkaindeksi** kuvaa palkkojen nimellistä kehitystä.
- **Reaaliansioindeksi / reaalipalkkaindeksi** kuvaa palkkojen kehitystä ostovoimalla mitattuna, eli inflaation vaikutus on poistettu.
"""
        )
        if include_real:
            st.markdown(
                """
Yksinkertaistettuna:
- jos **nimellinen palkkaindeksi** nousee enemmän kuin **hinnat**, myös reaaliansiot nousevat
- jos hinnat nousevat palkkoja nopeammin, **reaaliansiot laskevat**, vaikka nimellispalkat nousisivatkin
"""
            )

        st.markdown(
            """
Käytännön tulkinta:
- **nimellinen indeksi** = paljonko eurot palkkakuitissa muuttuvat
- **reaali-indeksi** = paljonko ostovoima oikeasti muuttuu
"""
        )


def render_inflation_section(df: pd.DataFrame, years: int) -> None:
    st.subheader("📈 Inflaatio (YoY, %, kuukausi)")
    st.caption("YoY = muutos verrattuna edellisvuoden samaan kuukauteen. Negatiivinen inflaatio = deflaatio.")

    if df is None or df.empty:
        st.warning("Inflaatiodataa ei saatu.")
        return

    vals = pd.to_numeric(df["inflation_yoy"], errors="coerce").dropna()
    infl_last = float(vals.iloc[-1]) if len(vals) else None
    infl_delta_y = _yoy_delta(df["inflation_yoy"], periods=12)
    infl_date = pd.to_datetime(df["Date"].iloc[-1]).date() if df["Date"].notna().any() else None

    kpi_card(
        "Inflaatio (YoY, kk)",
        fmt(infl_last, 2, " %"),
        f"{infl_delta_y:+.2f} %-yks. (1 v)" if infl_delta_y is not None else None,
        f"Kuukausi: {infl_date}" if infl_date else None,
    )

    st.divider()

    d = clip_by_years(df, "Date", years)
    fig = px.bar(d, x="Date", y="inflation_yoy", labels={"Date": "Kuukausi", "inflation_yoy": "Inflaatio (YoY, %)"})
    fig.update_yaxes(ticksuffix=" %", zeroline=True)
    st.plotly_chart(fig, width="stretch", key="macro_inflation_bar")


def render_gdp_section(
    df: pd.DataFrame,
    years: int,
    components_df: pd.DataFrame | None = None,
) -> None:
    st.subheader("🏛️ BKT ja talouskasvu")
    st.caption(
        "BKT YoY = volyymin muutos verrattuna edellisvuoden vastaavaan neljännekseen. "
        "Kysyntäerät näyttävät, mistä talouskasvu tai heikkous tulee."
    )

    if df is None or df.empty:
        st.warning("BKT YoY -dataa ei saatu.")
        return

    d_all = df.copy()
    d_all["Date"] = pd.to_datetime(d_all["Date"], errors="coerce")
    d_all["gdp_yoy"] = pd.to_numeric(d_all["gdp_yoy"], errors="coerce")
    d_all = d_all.dropna(subset=["Date", "gdp_yoy"]).sort_values("Date")

    if d_all.empty:
        st.warning("BKT YoY -sarja on tyhjä.")
        return

    g_last = float(d_all["gdp_yoy"].iloc[-1])
    g_delta_y = _yoy_delta(d_all["gdp_yoy"], periods=4)
    g_date = pd.to_datetime(d_all["Date"].iloc[-1]).date()

    last_4q_avg = float(d_all["gdp_yoy"].tail(4).mean()) if len(d_all) >= 4 else None
    last_5y_avg = float(d_all["gdp_yoy"].tail(20).mean()) if len(d_all) >= 20 else None

    comp_latest = pd.DataFrame()
    if components_df is not None and not components_df.empty:
        comp_latest = (
            components_df.copy()
            .assign(Date=lambda x: pd.to_datetime(x["Date"], errors="coerce"))
            .dropna(subset=["Date", "Komponentti", "gdp_component_yoy"])
            .sort_values("Date")
            .groupby("Komponentti", as_index=False)
            .tail(1)
        )

    private_consumption = None
    investments = None

    private_consumption_delta = None
    investments_delta = None

    if components_df is not None and not components_df.empty:

        comp_df = components_df.copy()
        comp_df["Date"] = pd.to_datetime(comp_df["Date"], errors="coerce")
        comp_df["gdp_component_yoy"] = pd.to_numeric(
            comp_df["gdp_component_yoy"],
            errors="coerce",
        )

        # Yksityinen kulutus
        pc_df = (
            comp_df[comp_df["Komponentti"] == "Yksityinen kulutus"]
            .dropna(subset=["Date", "gdp_component_yoy"])
            .sort_values("Date")
        )

        if not pc_df.empty:
            private_consumption = float(pc_df.iloc[-1]["gdp_component_yoy"])

            if len(pc_df) > 4:
                private_consumption_delta = (
                    float(pc_df.iloc[-1]["gdp_component_yoy"])
                    -  float(pc_df.iloc[-5]["gdp_component_yoy"])
                )

        # Investoinnit
        inv_df = (
            comp_df[comp_df["Komponentti"] == "Investoinnit"]
            .dropna(subset=["Date", "gdp_component_yoy"])
            .sort_values("Date")
        )

        if not inv_df.empty:
            investments = float(inv_df.iloc[-1]["gdp_component_yoy"])

            if len(inv_df) > 4:
                investments_delta = (
                    float(inv_df.iloc[-1]["gdp_component_yoy"])
                    -  float(inv_df.iloc[-5]["gdp_component_yoy"])
                )

    gdp_icon, gdp_status = _gdp_status(g_last)
    pc_icon, pc_status = _component_status(private_consumption, "Yksityinen kulutus")
    inv_icon, inv_status = _component_status(investments, "Investoinnit")

    st.markdown("### 📌 Tilaindikaattorit")

    c1, c2, c3, c4 = st.columns(4, gap="large")

    with c1:
        with st.container(border=True):
            st.markdown(f"### {gdp_icon} BKT YoY")
            st.markdown(f"**Tila:** {gdp_status}")
            st.metric(
                "Kasvuvauhti",
                fmt(g_last, 2, " %"),
                f"{g_delta_y:+.2f} %-yks. (1 v)" if g_delta_y is not None else None,
            )
            st.caption(f"Kvartaali: {g_date}")

    with c2:
        with st.container(border=True):
            avg_icon, avg_status = _gdp_status(last_4q_avg)
            st.markdown(f"### {avg_icon} BKT 4Q keskiarvo")
            st.markdown(f"**Tila:** {avg_status}")
            st.metric("Keskiarvo", fmt(last_4q_avg, 2, " %"))
            st.caption("Viimeisten 4 neljänneksen keskimääräinen kasvuvauhti.")

    with c3:
        with st.container(border=True):
            st.markdown(f"### {pc_icon} Yksityinen kulutus")
            st.markdown(f"**Tila:** {pc_status}")
            st.metric(
                "YoY-muutos",
                fmt(private_consumption, 2, " %"),
                (
                    f"{private_consumption_delta:+.2f} %-yks. (1 v)"
                    if private_consumption_delta is not None
                    else None
                ),
            )
            st.caption("Muutos verrattuna vuoden takaiseen kasvuvauhtiin.")

    with c4:
        with st.container(border=True):
            st.markdown(f"### {inv_icon} Investoinnit")
            st.markdown(f"**Tila:** {inv_status}")
            st.metric(
                "YoY-muutos",
                fmt(investments, 2, " %"),
                (
                    f"{investments_delta:+.2f} %-yks. (1 v)"
                    if investments_delta is not None
                    else None
                ),
            )
            if investments is not None and investments < 0:
                st.warning("Investoinnit ovat laskussa. Tämä voi heikentää tulevaa kasvupohjaa.")
            else:
                st.caption("Muutos verrattuna vuoden takaiseen kasvuvauhtiin.")

    st.markdown("### 🧭 Mikä vetää taloutta juuri nyt?")
    with st.container(border=True):
        st.write(
            _build_gdp_driver_text(
                gdp_now=g_last,
                private_consumption=private_consumption,
                investments=investments,
            )
        )

    if last_5y_avg is not None:
        st.caption(f"BKT:n keskimääräinen YoY-kasvu viimeisen 5 vuoden aikana: **{last_5y_avg:.2f} %**")

    st.divider()

    d = clip_by_years(d_all, "Date", years)

    tab_growth, tab_components = st.tabs(["📈 BKT:n kasvuvauhti", "🧩 Kysyntäerät"])

    with tab_growth:
        fig = px.line(
            d,
            x="Date",
            y="gdp_yoy",
            markers=True,
            labels={"Date": "Kvartaali", "gdp_yoy": "BKT YoY (%)"},
            title="BKT:n kasvuvauhti (YoY, %)",
        )
        fig.update_yaxes(ticksuffix=" %", zeroline=True)
        _add_zero_line(fig)
        st.plotly_chart(fig, width="stretch", key="macro_gdp_yoy_line")

    with tab_components:
        if components_df is None or components_df.empty:
            st.info("BKT:n kysyntäerien dataa ei saatu.")
            return

        cdf = components_df.copy()
        cdf["Date"] = pd.to_datetime(cdf["Date"], errors="coerce")
        cdf["gdp_component_yoy"] = pd.to_numeric(cdf["gdp_component_yoy"], errors="coerce")
        cdf = cdf.dropna(subset=["Date", "Komponentti", "gdp_component_yoy"])
        cdf = clip_by_years(cdf, "Date", years)

        if cdf.empty:
            st.info("Kysyntäerien dataa ei löytynyt valitulle aikavälille.")
            return

        preferred = [
            "BKT",
            "Yksityinen kulutus",
            "Julkinen kulutus",
            "Investoinnit",
            "Vienti",
            "Tuonti",
        ]

        selected_components = st.multiselect(
            "Valitse näytettävät kysyntäerät",
            options=[x for x in preferred if x in cdf["Komponentti"].unique()],
            default=[x for x in ["BKT", "Yksityinen kulutus", "Investoinnit", "Vienti"] if x in cdf["Komponentti"].unique()],
            key="macro_gdp_components_select",
        )

        plot_df = cdf[cdf["Komponentti"].isin(selected_components)].copy()

        if plot_df.empty:
            st.info("Valitse vähintään yksi kysyntäerä.")
            return

        fig_comp = px.line(
            plot_df,
            x="Date",
            y="gdp_component_yoy",
            color="Komponentti",
            markers=True,
            title="BKT:n kysyntäerät – YoY-muutos",
            labels={
                "Date": "Kvartaali",
                "gdp_component_yoy": "YoY (%)",
                "Komponentti": "Erä",
            },
        )
        fig_comp.update_yaxes(ticksuffix=" %", zeroline=True)
        _add_zero_line(fig_comp)
        fig_comp.update_layout(hovermode="x unified")
        st.plotly_chart(fig_comp, width="stretch", key="macro_gdp_components_line")



def render_unemployment_section(df: pd.DataFrame, years: int) -> None:
    st.subheader("🧑‍💼 Työttömyys ja työmarkkina")
    st.caption(
        "Työttömyysaste (%) = työttömien osuus työvoimasta. "
        "Kausitasoitettu sarja kertoo nykytilasta ja trendi näyttää pidemmän suunnan."
    )

    if df is None or df.empty:
        st.warning("Työttömyysdataa ei saatu.")
        return

    d = clip_by_years(df, "Date", years).copy()
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d = d.dropna(subset=["Date"]).sort_values("Date")

    if d.empty:
        st.warning("Työttömyysdata on tyhjä valitulla aikavälillä.")
        return

    latest_date = pd.to_datetime(d["Date"].iloc[-1]).date()

    latest_rate = (
        float(pd.to_numeric(d.get("unemployment_rate_sa"), errors="coerce").dropna().iloc[-1])
        if "unemployment_rate_sa" in d.columns and d["unemployment_rate_sa"].notna().any()
        else None
    )

    rate_delta_1y = (
        _yoy_delta(d["unemployment_rate_sa"], periods=12)
        if "unemployment_rate_sa" in d.columns
        else None
    )

    latest_level = (
        float(pd.to_numeric(d.get("unemployed_1000_sa"), errors="coerce").dropna().iloc[-1])
        if "unemployed_1000_sa" in d.columns and d["unemployed_1000_sa"].notna().any()
        else None
    )

    level_delta_1y = None
    if "unemployed_1000_sa" in d.columns:
        s = pd.to_numeric(d["unemployed_1000_sa"], errors="coerce").dropna()
        if len(s) > 12:
            latest = float(s.iloc[-1])
            prev = float(s.iloc[-13])
            if prev != 0:
                level_delta_1y = (latest / prev - 1.0) * 100.0

    trend_latest = None
    trend_delta_1y = None

    if "unemployment_rate_trend" in d.columns and d["unemployment_rate_trend"].notna().any():
        trend_s = pd.to_numeric(d["unemployment_rate_trend"], errors="coerce").dropna()

        if not trend_s.empty:
            trend_latest = float(trend_s.iloc[-1])

        if len(trend_s) > 12:
            trend_delta_1y = float(trend_s.iloc[-1] - trend_s.iloc[-13])

    rate_12m_avg = None
    if "unemployment_rate_sa" in d.columns:
        rate_s = pd.to_numeric(d["unemployment_rate_sa"], errors="coerce").dropna()
        if len(rate_s) >= 12:
            rate_12m_avg = float(rate_s.tail(12).mean())

    rate_icon, rate_status = _unemployment_status(latest_rate)
    trend_icon, trend_status = _unemployment_trend_status(rate_delta_1y)

    st.markdown("### 📌 Tilaindikaattorit")

    c1, c2, c3, c4 = st.columns(4, gap="large")

    with c1:
        with st.container(border=True):
            st.markdown(f"### {rate_icon} Työttömyysaste")
            st.markdown(f"**Tila:** {rate_status}")
            st.metric(
                "Kausitasoitettu",
                fmt(latest_rate, 2, " %"),
                f"{rate_delta_1y:+.2f} %-yks. (1 v)" if rate_delta_1y is not None else None,
            )
            st.caption(f"Kuukausi: {latest_date}")

    with c2:
        with st.container(border=True):
            st.markdown("### 👥 Työttömät")
            st.metric(
                "1000 henkilöä",
                fmt(latest_level, 0, ""),
                f"{level_delta_1y:+.1f} % (1 v)" if level_delta_1y is not None else None,
            )
            st.caption(f"Kuukausi: {latest_date}")

    with c3:
        with st.container(border=True):
            st.markdown(f"### {trend_icon} Suunta")
            st.markdown(f"**Tila:** {trend_status}")
            st.metric(
                "Muutos vuodessa",
                f"{rate_delta_1y:+.2f} %-yks." if rate_delta_1y is not None else "—",
            )
            st.caption("Työttömyysasteen muutos vuoden takaiseen.")

    with c4:
        with st.container(border=True):
            avg_icon, avg_status = _unemployment_status(rate_12m_avg)
            st.markdown(f"### {avg_icon} 12 kk keskiarvo")
            st.markdown(f"**Tila:** {avg_status}")
            st.metric("Keskiarvo", fmt(rate_12m_avg, 2, " %"))
            st.caption("Viimeisten 12 kuukauden keskimääräinen työttömyysaste.")

    if latest_rate is not None and latest_rate >= 9:
        st.warning("Työttömyysaste on korkealla tasolla. Tämä voi painaa kotitalouksien kulutusta ja heikentää suhdannekuvaa.")

    st.markdown("### 🧭 Tulkinta")
    with st.container(border=True):
        st.write(
            _build_unemployment_text(
                latest_rate=latest_rate,
                rate_delta_1y=rate_delta_1y,
                latest_level=latest_level,
                level_delta_1y=level_delta_1y,
            )
        )

    st.divider()

    tab_monthly, tab_yearly = st.tabs(["📊 Kuukausitaso", "📈 Vuositaso"])

    with tab_monthly:
        if "unemployment_rate_sa" in d.columns and d["unemployment_rate_sa"].notna().any():
            dd = d.dropna(subset=["Date", "unemployment_rate_sa"]).copy()

            fig = go.Figure()
            fig.add_bar(
                x=dd["Date"],
                y=dd["unemployment_rate_sa"],
                name="Kausitasoitettu",
            )

            if "unemployment_rate_trend" in d.columns and d["unemployment_rate_trend"].notna().any():
                dt = d.dropna(subset=["Date", "unemployment_rate_trend"]).copy()
                fig.add_scatter(
                    x=dt["Date"],
                    y=dt["unemployment_rate_trend"],
                    mode="lines+markers",
                    name="Trendi",
                )

            fig.update_layout(
                title="Työttömyysaste (%) – kuukausitaso",
                xaxis_title="Kuukausi",
                yaxis_title="%",
                hovermode="x unified",
            )
            fig.update_yaxes(ticksuffix=" %", zeroline=True)
            st.plotly_chart(fig, width="stretch", key="macro_unemp_rate_combo")
        else:
            st.info("Työttömyysaste (%) ei löytynyt tästä aineistosta.")

    with tab_yearly:
        if "unemployment_rate_sa" in d.columns and d["unemployment_rate_sa"].notna().any():
            dd = d.dropna(subset=["Date", "unemployment_rate_sa"]).copy()
            yearly_rate = _to_yearly_mean_df(dd, "Date", "unemployment_rate_sa")

            if not yearly_rate.empty:
                fig_year = px.line(
                    yearly_rate,
                    x="Vuosi",
                    y="unemployment_rate_sa",
                    markers=True,
                    title="Työttömyysaste (%) – vuositaso",
                    labels={"Vuosi": "Vuosi", "unemployment_rate_sa": "%"},
                )
                fig_year.update_yaxes(ticksuffix=" %", zeroline=True)
                st.plotly_chart(fig_year, width="stretch", key="macro_unemp_rate_year_line")
            else:
                st.info("Vuositason työttömyyssarjaa ei voitu muodostaa.")
        else:
            st.info("Työttömyysaste (%) ei löytynyt tästä aineistosta.")


def render_wages_section(df: pd.DataFrame, years: int) -> None:
    st.subheader("💶 Palkat ja ostovoima")
    st.caption(
        "Palkat perustuvat Tilastokeskuksen ansiotasoindeksiin ja kokoaikaisten palkansaajien keskiansioihin. "
        "Reaaliansioindeksi kertoo, miten palkkojen ostovoima kehittyy inflaation jälkeen."
    )

    if df is None or df.empty:
        st.warning("Palkkadataa ei saatu.")
        return

    sectors = df["Sector"].dropna().astype(str).unique().tolist()

    preferred_order = [
        "Koko kansantalous",
        "Total economy",
        "Yhteensä",
        "Total",
    ]

    ordered_sectors: list[str] = []
    for item in preferred_order:
        if item in sectors and item not in ordered_sectors:
            ordered_sectors.append(item)

    for item in sorted(sectors):
        if item not in ordered_sectors:
            ordered_sectors.append(item)

    selected_sector = st.selectbox(
        "Valitse sektori",
        ordered_sectors,
        index=0 if ordered_sectors else None,
        key="macro_wages_sector",
    )

    d = df[df["Sector"] == selected_sector].copy()
    d = clip_by_years(d, "Date", years)

    if d.empty:
        st.info("Valitulle sektorille ei löytynyt dataa.")
        return

    latest_wage_row = latest_row_by_date(d, "Date", "wage_eur")
    latest_index_row = latest_row_by_date(d, "Date", "wage_index")
    latest_real_row = latest_row_by_date(d, "Date", "real_wage_index")

    wage_now = float(latest_wage_row["wage_eur"]) if latest_wage_row is not None else None
    wage_yoy = (
        float(latest_wage_row["wage_yoy_pct"])
        if latest_wage_row is not None and pd.notna(latest_wage_row.get("wage_yoy_pct"))
        else None
    )
    wage_date = pd.to_datetime(latest_wage_row["Date"]).date() if latest_wage_row is not None else None

    index_now = float(latest_index_row["wage_index"]) if latest_index_row is not None else None
    index_yoy = (
        float(latest_index_row["wage_index_yoy_pct"])
        if latest_index_row is not None and pd.notna(latest_index_row.get("wage_index_yoy_pct"))
        else None
    )

    real_now = float(latest_real_row["real_wage_index"]) if latest_real_row is not None else None
    real_yoy = (
        float(latest_real_row["real_wage_yoy_pct"])
        if latest_real_row is not None and pd.notna(latest_real_row.get("real_wage_yoy_pct"))
        else None
    )

    wage_5y = _pct_change_over_years(d, "Date", "wage_eur", 5)
    index_5y = _pct_change_over_years(d, "Date", "wage_index", 5)
    real_5y = _pct_change_over_years(d, "Date", "real_wage_index", 5)

    power_icon, power_status = _wage_power_status(real_yoy)

    st.markdown("### 📌 Tilaindikaattorit")

    c1, c2, c3, c4 = st.columns(4, gap="large")

    with c1:
        with st.container(border=True):
            st.markdown("### 💶 Kuukausipalkka")
            st.metric(
                "Keskimääräinen",
                fmt(wage_now, 0, " €"),
                f"{wage_yoy:+.1f} % (1 v)" if wage_yoy is not None else None,
            )
            st.caption(
                f"5 v: {wage_5y:+.1f} % • Neljännes: {wage_date}"
                if wage_5y is not None and wage_date
                else f"Neljännes: {wage_date}" if wage_date else None
            )

    with c2:
        with st.container(border=True):
            st.markdown("### 📈 Ansiotaso")
            st.metric(
                "Indeksi",
                fmt(index_now, 1),
                f"{index_yoy:+.1f} % (1 v)" if index_yoy is not None else None,
            )
            st.caption(
                f"5 v: {index_5y:+.1f} % • Sektori: {selected_sector}"
                if index_5y is not None
                else f"Sektori: {selected_sector}"
            )

    with c3:
        with st.container(border=True):
            st.markdown("### 🛒 Reaaliansiot")
            st.metric(
                "Indeksi",
                fmt(real_now, 1),
                f"{real_yoy:+.1f} % (1 v)" if real_yoy is not None else None,
            )
            st.caption(
                f"5 v: {real_5y:+.1f} % • Inflaatio huomioitu"
                if real_5y is not None
                else "Inflaatio huomioitu"
            )

    with c4:
        with st.container(border=True):
            st.markdown(f"### {power_icon} Ostovoima")
            st.markdown(f"**Tila:** {power_status}")
            st.metric(
                "Reaaliansioiden muutos",
                f"{real_yoy:+.1f} %" if real_yoy is not None else "—",
            )
            if real_yoy is not None and real_yoy < -1:
                st.warning("Ostovoima heikkenee, koska reaaliansiot laskevat.")
            else:
                st.caption("Perustuu reaaliansioiden vuosimuutokseen.")

    st.markdown("### 🧭 Tulkinta")
    with st.container(border=True):
        st.write(
            _build_wage_text(
                wage_yoy=wage_yoy,
                index_yoy=index_yoy,
                real_yoy=real_yoy,
                wage_5y=wage_5y,
                real_5y=real_5y,
            )
        )

    render_index_explainer("ℹ️ Mitä palkkaindeksi ja reaaliansioindeksi tarkoittavat?")

    st.divider()

    tabs = st.tabs(["💶 Kuukausipalkka", "📈 Ansiotaso vs reaaliansiot", "🛒 Ostovoima 5 v"])

    with tabs[0]:
        plot_df = d.dropna(subset=["Date", "wage_eur"]).copy()
        if plot_df.empty:
            st.info("Kuukausipalkkadataa ei löytynyt.")
        else:
            fig = px.line(
                plot_df,
                x="Date",
                y="wage_eur",
                markers=True,
                title=f"Keskimääräinen kuukausipalkka – {selected_sector}",
                labels={"Date": "Neljännes", "wage_eur": "Euroa / kk"},
            )
            st.plotly_chart(fig, width="stretch", key="macro_wages_level_line")

    with tabs[1]:
        idx_df = d.copy().sort_values("Date")
        has_wage_index = "wage_index" in idx_df.columns and idx_df["wage_index"].notna().any()
        has_real_index = "real_wage_index" in idx_df.columns and idx_df["real_wage_index"].notna().any()

        if not has_wage_index and not has_real_index:
            st.info("Indeksidataa ei löytynyt.")
        else:
            fig = go.Figure()

            if has_wage_index:
                tmp = idx_df.dropna(subset=["Date", "wage_index"]).copy()
                fig.add_trace(
                    go.Scatter(
                        x=tmp["Date"],
                        y=tmp["wage_index"],
                        mode="lines+markers",
                        name="Ansiotasoindeksi",
                    )
                )

            if has_real_index:
                tmp = idx_df.dropna(subset=["Date", "real_wage_index"]).copy()
                fig.add_trace(
                    go.Scatter(
                        x=tmp["Date"],
                        y=tmp["real_wage_index"],
                        mode="lines+markers",
                        name="Reaaliansioindeksi",
                    )
                )

            fig.update_layout(
                title=f"Ansiotasoindeksi ja reaaliansioindeksi – {selected_sector}",
                xaxis_title="Neljännes",
                yaxis_title="Indeksi",
                hovermode="x unified",
            )
            st.plotly_chart(fig, width="stretch", key="macro_wages_index_compare_line")

    with tabs[2]:
        power_df = d.dropna(subset=["Date", "real_wage_index"]).copy()
        power_df = clip_by_years(power_df, "Date", 5)

        if power_df.empty:
            st.info("Ostovoimakuvaajaa ei voitu muodostaa.")
        else:
            first = pd.to_numeric(power_df["real_wage_index"], errors="coerce").dropna().iloc[0]
            power_df["Ostovoima_100"] = power_df["real_wage_index"] / first * 100.0

            fig = px.line(
                power_df,
                x="Date",
                y="Ostovoima_100",
                markers=True,
                title=f"Ostovoiman kehitys viimeisen 5 vuoden aikana – {selected_sector}",
                labels={"Date": "Neljännes", "Ostovoima_100": "Indeksi, alku = 100"},
            )
            st.plotly_chart(fig, width="stretch", key="macro_wages_purchasing_power_5y")


def render_trade_flow_section(cfg: dict, months: int) -> None:
    st.subheader(cfg["title"])
    st.caption(cfg["caption"])

    products_df, products_debug = cfg["fetch_products"](months=max(months, 84), lang="fi")
    total_df = build_total_flow_from_products(products_df, cfg["value_col"])

    latest_val, pct_val, latest_year = _full_year_stats_from_mixed_series(
        total_df,
        "Aika_dt",
        cfg["value_col"],
    )

    icon, status = _trade_flow_status(pct_val)

    product_stats = _latest_full_year_group_stats(
        products_df,
        date_col="Aika_dt",
        group_col=cfg["group_col"],
        value_col=cfg["value_col"],
    )

    top_product = None
    top_product_share = None
    top_driver = None
    top_drag = None

    if not product_stats.empty:
        top_row = product_stats.sort_values("latest", ascending=False).iloc[0]
        top_product = str(top_row[cfg["group_col"]])
        top_product_share = float(top_row["share_pct"])

        driver_df = product_stats.dropna(subset=["change_eur"]).sort_values("change_eur", ascending=False)
        drag_df = product_stats.dropna(subset=["change_eur"]).sort_values("change_eur", ascending=True)

        if not driver_df.empty and float(driver_df.iloc[0]["change_eur"]) > 0:
            top_driver = str(driver_df.iloc[0][cfg["group_col"]])

        if not drag_df.empty and float(drag_df.iloc[0]["change_eur"]) < 0:
            top_drag = str(drag_df.iloc[0][cfg["group_col"]])

    st.markdown("### 📌 Tilaindikaattorit")

    c1, c2, c3, c4 = st.columns(4, gap="large")

    with c1:
        with st.container(border=True):
            st.markdown(f"### {icon} {cfg['metric_label']}")
            st.markdown(f"**Tila:** {status}")
            st.metric(
                "Viimeisin täysi vuosi",
                fmt_money(latest_val),
                f"{pct_val:+.1f} % (1 v)" if pct_val is not None else None,
            )
            st.caption(f"Vuosi: {latest_year}" if latest_year is not None else None)

    with c2:
        with st.container(border=True):
            st.markdown("### 📦 Suurin tuoteryhmä")
            st.metric(
                top_product or "—",
                f"{top_product_share:.1f} %" if top_product_share is not None else "—",
            )
            st.caption("Osuus viimeisimmästä täydestä vuodesta.")

    with c3:
        with st.container(border=True):
            st.markdown("### 🟢 Suurin ajuri")
            st.metric(top_driver or "—", "")
            st.caption("Suurin positiivinen euromääräinen muutos vuodessa.")

    with c4:
        with st.container(border=True):
            st.markdown("### 🔴 Suurin heikentäjä")
            st.metric(top_drag or "—", "")
            st.caption("Suurin negatiivinen euromääräinen muutos vuodessa.")

    st.markdown("### 🧭 Tulkinta")
    with st.container(border=True):
        st.write(
            _trade_flow_text(
                flow_name=cfg["flow_name"],
                total_pct=pct_val,
                top_driver=top_driver,
                top_drag=top_drag,
            )
        )

    st.divider()

    tab_products, tab_regions = st.tabs(["📦 Tuoteryhmät", "🌍 Maanosat"])
    regions_df, regions_debug = cfg["fetch_regions"](months=max(months, 84), lang="fi")

    with tab_products:
        if products_df is None or products_df.empty:
            st.warning(cfg["products_empty_msg"])
            show_debug_info(products_debug)
        else:
            if not product_stats.empty:
                with st.expander("📊 Tuoteryhmien viimeisin täysi vuosi", expanded=False):
                    show_df = product_stats.copy()
                    show_df["Arvo"] = show_df["latest"].apply(fmt_money)
                    show_df["Muutos €"] = show_df["change_eur"].apply(fmt_money)
                    show_df["Muutos %"] = show_df["change_pct"].map(
                        lambda x: f"{x:+.1f} %" if pd.notna(x) else "—"
                    )
                    show_df["Osuus"] = show_df["share_pct"].map(
                        lambda x: f"{x:.1f} %" if pd.notna(x) else "—"
                    )

                    st.dataframe(
                        show_df[[cfg["group_col"], "Arvo", "Muutos €", "Muutos %", "Osuus"]],
                        width="stretch",
                        hide_index=True,
                    )

            _stacked_bar_chart(
                products_df,
                date_col="Aika_dt",
                category_col=cfg["group_col"],
                value_col=cfg["value_col"],
                title=cfg["products_title"],
                key=f"{cfg['key']}_products_mixed",
            )

            st.divider()
            st.markdown(cfg["group_detail_title"])

            groups = sorted(products_df[cfg["group_col"]].dropna().unique().tolist())
            selected_group = st.selectbox(
                cfg["group_select_label"],
                groups,
                key=f"macro_{cfg['key']}_group_detail",
            )

            group_df = products_df[products_df[cfg["group_col"]] == selected_group].copy().sort_values("Aika_dt")

            group_latest_val, group_pct_val, group_latest_year = _full_year_stats_from_mixed_series(
                group_df, "Aika_dt", cfg["value_col"]
            )

            group_icon, group_status = _trade_flow_status(group_pct_val)

            with st.container(border=True):
                st.markdown(f"### {group_icon} {selected_group}")
                st.markdown(f"**Tila:** {group_status}")
                st.metric(
                    "Viimeisin täysi vuosi",
                    fmt_money(group_latest_val),
                    f"{group_pct_val:+.1f} % (1 v)" if group_pct_val is not None else None,
                )
                st.caption(
                    f"Viimeisin täysi vuosi: {group_latest_year}"
                    if group_latest_year is not None
                    else None
                )

            _combined_monthly_and_trend_chart(
                group_df,
                date_col="Aika_dt",
                value_col=cfg["value_col"],
                title=f"{selected_group} – {cfg['flow_name']} kehitys",
                y_label="Arvo (milj. €)",
                key=f"{cfg['key']}_group_combined_line",
            )

    with tab_regions:
        if regions_df is None or regions_df.empty:
            st.warning(cfg["regions_empty_msg"])
            show_debug_info(regions_debug)
        else:
            region_stats = _latest_full_year_group_stats(
                regions_df,
                date_col="Aika_dt",
                group_col=cfg["region_col"],
                value_col=cfg["value_col"],
            )

            if not region_stats.empty:
                with st.expander("🌍 Maanosien viimeisin täysi vuosi", expanded=False):
                    show_df = region_stats.copy()
                    show_df["Arvo"] = show_df["latest"].apply(fmt_money)
                    show_df["Muutos €"] = show_df["change_eur"].apply(fmt_money)
                    show_df["Muutos %"] = show_df["change_pct"].map(
                        lambda x: f"{x:+.1f} %" if pd.notna(x) else "—"
                    )
                    show_df["Osuus"] = show_df["share_pct"].map(
                        lambda x: f"{x:.1f} %" if pd.notna(x) else "—"
                    )

                    st.dataframe(
                        show_df[[cfg["region_col"], "Arvo", "Muutos €", "Muutos %", "Osuus"]],
                        width="stretch",
                        hide_index=True,
                    )

            _stacked_bar_chart(
                regions_df,
                date_col="Aika_dt",
                category_col=cfg["region_col"],
                value_col=cfg["value_col"],
                title=cfg["regions_title"],
                key=f"{cfg['key']}_regions_mixed",
            )

            st.divider()
            st.markdown(cfg["region_detail_title"])

            regions = sorted(regions_df[cfg["region_col"]].dropna().unique().tolist())
            selected_region = st.selectbox(
                cfg["region_select_label"],
                regions,
                key=f"macro_{cfg['key']}_region_detail",
            )

            region_df = regions_df[regions_df[cfg["region_col"]] == selected_region].copy().sort_values("Aika_dt")

            region_latest_val, region_pct_val, region_latest_year = _full_year_stats_from_mixed_series(
                region_df, "Aika_dt", cfg["value_col"]
            )

            region_icon, region_status = _trade_flow_status(region_pct_val)

            with st.container(border=True):
                st.markdown(f"### {region_icon} {selected_region}")
                st.markdown(f"**Tila:** {region_status}")
                st.metric(
                    "Viimeisin täysi vuosi",
                    fmt_money(region_latest_val),
                    f"{region_pct_val:+.1f} % (1 v)" if region_pct_val is not None else None,
                )
                st.caption(
                    f"Viimeisin täysi vuosi: {region_latest_year}"
                    if region_latest_year is not None
                    else None
                )

            _combined_monthly_and_trend_chart(
                region_df,
                date_col="Aika_dt",
                value_col=cfg["value_col"],
                title=f"{selected_region} – {cfg['flow_name']} kehitys",
                y_label="Arvo (milj. €)",
                key=f"{cfg['key']}_region_combined_line",
            )

def render_trade_balance_section(exports_total_df: pd.DataFrame, imports_total_df: pd.DataFrame, years: int) -> None:
    st.subheader("⚖️ Kauppatase")
    st.caption("Kauppatase = tavaravienti − tavaratuonti. Positiivinen luku tarkoittaa ylijäämää, negatiivinen alijäämää.")

    trade_df = build_trade_balance(exports_total_df, imports_total_df)
    if trade_df is None or trade_df.empty:
        st.warning("Kauppatasedataa ei saatu.")
        return

    d = trade_df.copy()
    d["Aika_dt"] = pd.to_datetime(d["Aika_dt"], errors="coerce")

    for col in ["Vienti_eur", "Tuonti_eur", "Kauppatase_eur"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    d = d.dropna(subset=["Aika_dt", "Vienti_eur", "Tuonti_eur", "Kauppatase_eur"]).sort_values("Aika_dt")

    if d.empty:
        st.warning("Kauppatasedataa ei saatu.")
        return

    stats = _latest_full_year_trade_balance_stats(d)

    balance = stats.get("balance")
    balance_change = stats.get("balance_change")
    exports = stats.get("exports")
    imports = stats.get("imports")
    export_import_ratio = stats.get("export_import_ratio")
    latest_year = stats.get("year")

    icon, status = _trade_balance_status(balance, balance_change)

    st.markdown("### 📌 Tilaindikaattorit")

    c1, c2, c3, c4 = st.columns(4, gap="large")

    with c1:
        with st.container(border=True):
            st.markdown(f"### {icon} Kauppatase")
            st.markdown(f"**Tila:** {status}")
            st.metric(
                "Viimeisin täysi vuosi",
                fmt_money(balance),
                _fmt_trade_change_eur(balance_change),
            )
            st.caption(f"Vuosi: {latest_year}" if latest_year is not None else None)

    with c2:
        with st.container(border=True):
            st.markdown("### 🚢 Vienti")
            st.metric("Arvo", fmt_money(exports))
            st.caption("Viimeisin täysi vuosi.")

    with c3:
        with st.container(border=True):
            st.markdown("### 📥 Tuonti")
            st.metric("Arvo", fmt_money(imports))
            st.caption("Viimeisin täysi vuosi.")

    with c4:
        with st.container(border=True):
            ratio_icon = "🟢" if export_import_ratio is not None and export_import_ratio >= 100 else "🔴"
            st.markdown(f"### {ratio_icon} Vienti / tuonti")
            st.metric(
                "Suhde",
                f"{export_import_ratio:.1f} %" if export_import_ratio is not None else "—",
            )
            st.caption("Yli 100 % = vienti suurempi kuin tuonti.")

    st.markdown("### 🧭 Tulkinta")
    with st.container(border=True):
        st.write(_trade_balance_text(stats))

    st.divider()

    plot_df = clip_by_years(d.copy(), "Aika_dt", years)
    plot_df["Kauppatase_milj_eur"] = plot_df["Kauppatase_eur"] / 1_000_000

    tab_month, tab_year, tab_compare = st.tabs(
        ["📉 Kuukausisarja", "📈 Vuositaso", "🚢 Vienti vs tuonti"]
    )

    with tab_month:
        fig_month = go.Figure()
        fig_month.add_trace(
            go.Scatter(
                x=plot_df["Aika_dt"],
                y=plot_df["Kauppatase_milj_eur"],
                mode="lines+markers",
                name="Kuukausisarja",
                line=dict(width=2),
            )
        )
        fig_month.update_layout(
            title="Kauppatase – kuukausisarja",
            xaxis_title="Aika",
            yaxis_title="Kauppatase (milj. €)",
            hovermode="x unified",
        )
        _add_zero_line(fig_month)
        st.plotly_chart(fig_month, width="stretch", key="macro_trade_balance_monthly_line")

    with tab_year:
        yearly = _to_yearly_sum_df(d, "Aika_dt", "Kauppatase_eur")
        if yearly is not None and not yearly.empty:
            yearly = yearly.copy()
            yearly["Kauppatase_milj_eur"] = pd.to_numeric(yearly["Kauppatase_eur"], errors="coerce") / 1_000_000
            yearly["Vuosi_label"] = yearly["Vuosi"].astype(str)

            fig_year = px.line(
                yearly,
                x="Vuosi_label",
                y="Kauppatase_milj_eur",
                markers=True,
                title="Kauppatase – vuositaso",
                labels={
                    "Vuosi_label": "Vuosi",
                    "Kauppatase_milj_eur": "Kauppatase (milj. €)",
                },
            )
            fig_year.update_xaxes(type="category")
            _add_zero_line(fig_year)
            st.plotly_chart(fig_year, width="stretch", key="macro_trade_balance_yearly_line")
        else:
            st.info("Vuositason kauppatasedataa ei saatu muodostettua.")

    with tab_compare:
        compare_df = clip_by_years(d.copy(), "Aika_dt", years)
        compare_df["Aika_label"] = compare_df["Aika_dt"].dt.strftime("%Y-%m")
        compare_df["Vienti_milj_eur"] = pd.to_numeric(compare_df["Vienti_eur"], errors="coerce") / 1_000_000
        compare_df["Tuonti_milj_eur"] = pd.to_numeric(compare_df["Tuonti_eur"], errors="coerce") / 1_000_000

        fig_compare = go.Figure()
        fig_compare.add_trace(
            go.Scatter(
                x=compare_df["Aika_dt"],
                y=compare_df["Vienti_milj_eur"],
                mode="lines+markers",
                name="Vienti",
            )
        )
        fig_compare.add_trace(
            go.Scatter(
                x=compare_df["Aika_dt"],
                y=compare_df["Tuonti_milj_eur"],
                mode="lines+markers",
                name="Tuonti",
            )
        )
        fig_compare.update_layout(
            title="Vienti ja tuonti samassa kuvaajassa",
            xaxis_title="Aika",
            yaxis_title="Arvo (milj. €)",
            hovermode="x unified",
        )
        st.plotly_chart(fig_compare, width="stretch", key="macro_trade_compare_line")


def render_debt_section(debt_pct: pd.DataFrame, debt_eur: pd.DataFrame, years: int) -> None:
    st.subheader("🏦 Julkinen velka")
    st.caption(
        "Eurostat: julkisyhteisöjen bruttovelka kvartaaleittain. "
        "Velka/BKT kertoo velan suhteesta talouden kokoon."
    )

    latest_pct = None
    latest_pct_date = None
    pct_yoy = None

    if debt_pct is not None and not debt_pct.empty and "debt_pct_gdp" in debt_pct.columns:
        x = debt_pct.copy()
        x["Date"] = pd.to_datetime(x["Date"], errors="coerce")
        x["debt_pct_gdp"] = pd.to_numeric(x["debt_pct_gdp"], errors="coerce")
        x = x.dropna(subset=["Date", "debt_pct_gdp"]).sort_values("Date")

        if not x.empty:
            latest_pct = float(x["debt_pct_gdp"].iloc[-1])
            latest_pct_date = pd.to_datetime(x["Date"].iloc[-1]).date()

            if len(x) > 4:
                prev_val = float(x["debt_pct_gdp"].iloc[-5])
                pct_yoy = latest_pct - prev_val

    latest_eur = None
    latest_eur_date = None
    eur_yoy = None

    if debt_eur is not None and not debt_eur.empty and "debt_mio_eur" in debt_eur.columns:
        y = debt_eur.copy()
        y["Date"] = pd.to_datetime(y["Date"], errors="coerce")
        y["debt_mio_eur"] = pd.to_numeric(y["debt_mio_eur"], errors="coerce")
        y = y.dropna(subset=["Date", "debt_mio_eur"]).sort_values("Date")

        if not y.empty:
            latest_eur = float(y["debt_mio_eur"].iloc[-1])
            latest_eur_date = pd.to_datetime(y["Date"].iloc[-1]).date()

            if len(y) > 4:
                prev_val = float(y["debt_mio_eur"].iloc[-5])
                if prev_val != 0:
                    eur_yoy = ((latest_eur / prev_val) - 1.0) * 100.0

    debt_icon, debt_status = _debt_status(latest_pct, pct_yoy)
    direction_icon, direction_status = _debt_direction_status(pct_yoy)

    st.markdown("### 📌 Tilaindikaattorit")

    c1, c2, c3 = st.columns(3, gap="large")

    with c1:
        with st.container(border=True):
            st.markdown(f"### {debt_icon} Velka / BKT")
            st.markdown(f"**Tila:** {debt_status}")
            st.metric(
                "Velkasuhde",
                f"{latest_pct:,.1f} %".replace(",", " ") if latest_pct is not None else "—",
                f"{pct_yoy:+.1f} %-yks. (1 v)" if pct_yoy is not None else None,
            )
            st.caption(
                f"Lähde: Eurostat • {format_source_date(pd.to_datetime(latest_pct_date), 'year')}"
                if latest_pct_date
                else "Lähde: Eurostat"
            )

    with c2:
        with st.container(border=True):
            st.markdown("### 💶 Velka euroina")
            debt_eur_text = "—"
            if latest_eur is not None:
                debt_eur_text = f"{latest_eur:,.0f} milj. €".replace(",", " ")

            st.metric(
                "Bruttovelka",
                debt_eur_text,
                f"{eur_yoy:+.1f} % (1 v)" if eur_yoy is not None else None,
            )
            st.caption(
                f"Lähde: Eurostat • {format_source_date(pd.to_datetime(latest_eur_date), 'year')}"
                if latest_eur_date
                else "Lähde: Eurostat"
            )

    with c3:
        with st.container(border=True):
            st.markdown(f"### {direction_icon} Suunta")
            st.markdown(f"**Tila:** {direction_status}")
            st.metric(
                "Velkasuhteen muutos",
                f"{pct_yoy:+.1f} %-yks." if pct_yoy is not None else "—",
            )
            st.caption("Muutos verrattuna vuoden takaiseen kvartaaliin.")

    if latest_pct is not None and latest_pct >= 90:
        st.warning("Velka/BKT on korkealla tasolla. Tämä voi kaventaa julkisen talouden liikkumavaraa.")

    st.markdown("### 🧭 Tulkinta")
    with st.container(border=True):
        st.write(
            _build_debt_text(
                debt_pct=latest_pct,
                pct_yoy=pct_yoy,
                debt_eur=latest_eur,
                eur_yoy=eur_yoy,
            )
        )

    st.divider()

    d1 = debt_pct.copy() if debt_pct is not None else pd.DataFrame()
    d2 = debt_eur.copy() if debt_eur is not None else pd.DataFrame()

    if not d1.empty:
        d1["Date"] = pd.to_datetime(d1["Date"], errors="coerce")
        d1["debt_pct_gdp"] = pd.to_numeric(d1["debt_pct_gdp"], errors="coerce")
        d1 = d1.dropna(subset=["Date", "debt_pct_gdp"]).sort_values("Date")
        d1 = clip_by_years(d1, "Date", years)

    if not d2.empty:
        d2["Date"] = pd.to_datetime(d2["Date"], errors="coerce")
        d2["debt_mio_eur"] = pd.to_numeric(d2["debt_mio_eur"], errors="coerce")
        d2 = d2.dropna(subset=["Date", "debt_mio_eur"]).sort_values("Date")
        d2 = clip_by_years(d2, "Date", years)

    tab_ratio, tab_eur = st.tabs(["📈 Velka / BKT", "💶 Velka euroina"])

    with tab_ratio:
        if d1.empty:
            st.info("Velka/BKT -sarjaa ei saatu.")
        else:
            fig = px.line(
                d1,
                x="Date",
                y="debt_pct_gdp",
                markers=True,
                labels={"Date": "Kvartaali", "debt_pct_gdp": "% BKT:stä"},
                title="Velka / BKT (%)",
            )
            fig.update_yaxes(ticksuffix=" %")
            st.plotly_chart(fig, width="stretch", key="macro_debt_pct_line")

            yearly = _to_yearly_mean_df(d1, "Date", "debt_pct_gdp")
            if not yearly.empty:
                fig_year = px.line(
                    yearly,
                    x="Vuosi",
                    y="debt_pct_gdp",
                    markers=True,
                    title="Velka / BKT (%) – vuositaso",
                    labels={"Vuosi": "Vuosi", "debt_pct_gdp": "% BKT:stä"},
                )
                fig_year.update_yaxes(ticksuffix=" %")
                st.plotly_chart(fig_year, width="stretch", key="macro_debt_pct_year_line")

    with tab_eur:
        if d2.empty:
            st.info("Velan euromäärää ei saatu.")
        else:
            fig = px.line(
                d2,
                x="Date",
                y="debt_mio_eur",
                markers=True,
                labels={"Date": "Kvartaali", "debt_mio_eur": "milj. €"},
                title="Velka (milj. €)",
            )
            st.plotly_chart(fig, width="stretch", key="macro_debt_eur_line")

            _yearly_line_chart(
                d2,
                date_col="Date",
                value_col="debt_mio_eur",
                title="Velka (milj. €) – vuositaso",
                y_label="milj. €",
                key="macro_debt_eur_year_line",
            )



def render_private_debt_section(
    household_pct_gdp: pd.DataFrame,
    household_pct_gdi: pd.DataFrame,
    nfc_pct_gdp: pd.DataFrame,
    private_pct_gdp: pd.DataFrame,
    household_loans_mio: pd.DataFrame,
    nfc_loans_mio: pd.DataFrame,
    household_loans_debug: dict | None,
    nfc_loans_debug: dict | None,
    years: int,
) -> None:
    st.subheader("🏠 Yksityinen velka")
    st.caption(
        "Yksityinen velka kertoo kotitalouksien ja yritysten velkaantumisesta. "
        "Kotitalouksille tulosuhde on usein BKT-suhdetta havainnollisempi."
    )

    hh_gdp_val, hh_gdp_date = _latest_value_and_date(household_pct_gdp, "Date", "household_debt_pct_gdp")
    hh_gdi_val, hh_gdi_date = _latest_value_and_date(household_pct_gdi, "Date", "household_debt_pct_gdi")
    nfc_gdp_val, nfc_gdp_date = _latest_value_and_date(nfc_pct_gdp, "Date", "nfc_debt_pct_gdp")
    private_gdp_val, private_gdp_date = _latest_value_and_date(private_pct_gdp, "Date", "private_debt_pct_gdp")

    hh_gdp_change = _debt_change_pp(household_pct_gdp, "Date", "household_debt_pct_gdp", periods=1)
    hh_gdi_change = _debt_change_pp(household_pct_gdi, "Date", "household_debt_pct_gdi", periods=1)
    nfc_gdp_change = _debt_change_pp(nfc_pct_gdp, "Date", "nfc_debt_pct_gdp", periods=1)
    private_gdp_change = _debt_change_pp(private_pct_gdp, "Date", "private_debt_pct_gdp", periods=1)

    hh_loans_latest, hh_loans_date = _latest_value_and_date(household_loans_mio, "Date", "household_loans_mio")
    nfc_loans_latest, nfc_loans_date = _latest_value_and_date(nfc_loans_mio, "Date", "nfc_loans_mio_nac")

    hh_loans_yoy = yoy_pct_change(household_loans_mio, "Date", "household_loans_mio")
    nfc_loans_yoy = yoy_pct_change(nfc_loans_mio, "Date", "nfc_loans_mio_nac")

    hh_icon, hh_status = _private_debt_status(hh_gdi_val, kind="household_gdi")
    nfc_icon, nfc_status = _private_debt_status(nfc_gdp_val)
    total_icon, total_status = _private_debt_status(private_gdp_val)

    st.markdown("### 📌 Tilaindikaattorit")

    c1, c2, c3, c4 = st.columns(4, gap="large")

    with c1:
        with st.container(border=True):
            st.markdown(f"### {hh_icon} Kotitaloudet / tulot")
            st.markdown(f"**Tila:** {hh_status}")
            st.metric(
                "Velka suhteessa tuloihin",
                fmt(hh_gdi_val, 1, " %"),
                f"{hh_gdi_change:+.1f} %-yks. (1 v)" if hh_gdi_change is not None else None,
            )
            st.caption(
                f"Lähde: Eurostat • {format_source_date(hh_gdi_date, 'year')}"
                if hh_gdi_date is not None
                else "Lähde: Eurostat"
            )

    with c2:
        with st.container(border=True):
            st.markdown(f"### {nfc_icon} Yritykset / BKT")
            st.markdown(f"**Tila:** {nfc_status}")
            st.metric(
                "Velka suhteessa BKT:hen",
                fmt(nfc_gdp_val, 1, " %"),
                f"{nfc_gdp_change:+.1f} %-yks. (1 v)" if nfc_gdp_change is not None else None,
            )
            st.caption(
                f"Lähde: Eurostat • {format_source_date(nfc_gdp_date, 'year')}"
                if nfc_gdp_date is not None
                else "Lähde: Eurostat"
            )

    with c3:
        with st.container(border=True):
            st.markdown(f"### {total_icon} Yksityinen / BKT")
            st.markdown(f"**Tila:** {total_status}")
            st.metric(
                "Yksityinen velka",
                fmt(private_gdp_val, 1, " %"),
                f"{private_gdp_change:+.1f} %-yks. (1 v)" if private_gdp_change is not None else None,
            )
            st.caption(
                f"Lähde: Eurostat • {format_source_date(private_gdp_date, 'year')}"
                if private_gdp_date is not None
                else "Lähde: Eurostat"
            )

    with c4:
        with st.container(border=True):
            st.markdown("### 💶 Kotitalouslainat")
            st.metric(
                "Lainakanta",
                _fmt_mio_eur(hh_loans_latest),
                f"{hh_loans_yoy:+.1f} % (1 v)" if hh_loans_yoy is not None else None,
            )
            st.caption(
                f"Lähde: ECB • {format_source_date(hh_loans_date, 'month')}"
                if hh_loans_date is not None
                else "Lähde: ECB"
            )

    if hh_gdi_val is not None and hh_gdi_val >= 140:
        st.warning("Kotitalouksien velka suhteessa tuloihin on korkealla tasolla. Korkojen nousu voi rasittaa kotitalouksia nopeasti.")

    st.markdown("### 🧭 Tulkinta")
    with st.container(border=True):
        st.write(
            _private_debt_text(
                hh_gdi=hh_gdi_val,
                hh_gdi_change=hh_gdi_change,
                nfc_gdp=nfc_gdp_val,
                nfc_change=nfc_gdp_change,
                private_gdp=private_gdp_val,
                private_change=private_gdp_change,
                hh_loans_yoy=hh_loans_yoy,
                nfc_loans_yoy=nfc_loans_yoy,
            )
        )

    st.divider()

    ratio_frames = []

    if household_pct_gdp is not None and not household_pct_gdp.empty:
        d = household_pct_gdp.copy()
        d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
        d["Arvo"] = pd.to_numeric(d["household_debt_pct_gdp"], errors="coerce")
        d["Sarja"] = "Kotitaloudet / BKT"
        ratio_frames.append(d[["Date", "Arvo", "Sarja"]])

    if household_pct_gdi is not None and not household_pct_gdi.empty:
        d = household_pct_gdi.copy()
        d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
        d["Arvo"] = pd.to_numeric(d["household_debt_pct_gdi"], errors="coerce")
        d["Sarja"] = "Kotitaloudet / tulot"
        ratio_frames.append(d[["Date", "Arvo", "Sarja"]])

    if nfc_pct_gdp is not None and not nfc_pct_gdp.empty:
        d = nfc_pct_gdp.copy()
        d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
        d["Arvo"] = pd.to_numeric(d["nfc_debt_pct_gdp"], errors="coerce")
        d["Sarja"] = "Yritykset / BKT"
        ratio_frames.append(d[["Date", "Arvo", "Sarja"]])

    if private_pct_gdp is not None and not private_pct_gdp.empty:
        d = private_pct_gdp.copy()
        d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
        d["Arvo"] = pd.to_numeric(d["private_debt_pct_gdp"], errors="coerce")
        d["Sarja"] = "Yksityinen velka / BKT"
        ratio_frames.append(d[["Date", "Arvo", "Sarja"]])

    tab_ratios, tab_loans = st.tabs(["📈 Velkasuhteet", "💶 Lainakannat"])

    with tab_ratios:
        if ratio_frames:
            ratio_df = pd.concat(ratio_frames, ignore_index=True)
            ratio_df = clip_by_years(ratio_df, "Date", years)
            ratio_df = ratio_df.dropna(subset=["Date", "Arvo"]).sort_values("Date")

            if not ratio_df.empty:
                fig_ratio = px.line(
                    ratio_df,
                    x="Date",
                    y="Arvo",
                    color="Sarja",
                    markers=True,
                    title="Yksityisen velan suhdeluvut",
                    labels={"Date": "Vuosi", "Arvo": "%", "Sarja": "Sarja"},
                )
                fig_ratio.update_yaxes(ticksuffix=" %")
                fig_ratio.update_layout(hovermode="x unified")
                st.plotly_chart(fig_ratio, width="stretch", key="macro_private_debt_ratio_line")
            else:
                st.info("Yksityisen velan suhdelukuja ei löytynyt.")
        else:
            st.info("Yksityisen velan suhdelukuja ei löytynyt.")

    with tab_loans:
        loan_frames = []

        if household_loans_mio is not None and not household_loans_mio.empty:
            d = household_loans_mio.copy()
            d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
            d["Arvo"] = pd.to_numeric(d["household_loans_mio"], errors="coerce")
            d["Sarja"] = "Kotitaloudet"
            loan_frames.append(d[["Date", "Arvo", "Sarja"]])

        if nfc_loans_mio is not None and not nfc_loans_mio.empty:
            d = nfc_loans_mio.copy()
            d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
            d["Arvo"] = pd.to_numeric(d["nfc_loans_mio_nac"], errors="coerce")
            d["Sarja"] = "Yritykset"
            loan_frames.append(d[["Date", "Arvo", "Sarja"]])

        if loan_frames:
            loan_df = pd.concat(loan_frames, ignore_index=True)
            loan_df = clip_by_years(loan_df, "Date", years)
            loan_df = loan_df.dropna(subset=["Date", "Arvo"]).sort_values("Date")

            if not loan_df.empty:
                fig_loans = px.line(
                    loan_df,
                    x="Date",
                    y="Arvo",
                    color="Sarja",
                    markers=True,
                    title="Lainakannat sektoreittain",
                    labels={"Date": "Vuosi", "Arvo": "milj. €", "Sarja": "Sarja"},
                )
                fig_loans.update_layout(hovermode="x unified")
                st.plotly_chart(fig_loans, width="stretch", key="macro_private_loans_line")
            else:
                st.info("Lainakantadataa ei löytynyt.")
        else:
            st.info("Lainakantadataa ei löytynyt.")
            


def _inflation_status(value: float | None) -> tuple[str, str]:
    if value is None or pd.isna(value):
        return "⚪", "Ei dataa"

    if value < 2:
        return "🟢", "Vakaa"
    if value < 5:
        return "🟡", "Paineita"
    return "🔴", "Korkea paine"



def render_inflation_pressure_section(bundle: dict) -> None:
    st.subheader("🧺 Arjen inflaatiopaine")
    st.caption("Virallinen inflaatio, arjen hintapaineet ja pidemmän aikavälin hintojen nousu.")

    def _long_inflation_status(value: float | None) -> tuple[str, str]:
        if value is None or pd.isna(value):
            return "⚪", "Ei dataa"

        if value < 10:
            return "🟢", "Maltillinen"
        if value < 25:
            return "🟡", "Selvä hintojen nousu"
        return "🔴", "Voimakas hintojen nousu"

    if not bundle or not bundle.get("ok"):
        st.warning("Tarkempia inflaatiomittareita ei saatu ladattua.")
        if bundle and bundle.get("error"):
            with st.expander("Tekninen virhe"):
                st.code(bundle["error"])
        return

    latest = bundle.get("latest", pd.DataFrame())
    series_df = bundle.get("series", pd.DataFrame())

    if latest.empty:
        st.info("Inflaatiomittareita ei löytynyt.")
        return

    preferred_order = [
        "Virallinen inflaatio",
        "Ruokainflaatio",
        "Energia",
        "Polttoaineet",
    ]

    latest = latest.copy()
    latest["order"] = latest["Sarja"].map({x: i for i, x in enumerate(preferred_order)}).fillna(99)
    latest = latest.sort_values("order")

    st.markdown("### 📌 Nykyinen vuosimuutos")

    cols = st.columns(min(4, len(latest)))

    for i, (_, row) in enumerate(latest.iterrows()):
        value = row.get("Inflaatio")
        value = float(value) if pd.notna(value) else None
        icon, status = _inflation_status(value)

        with cols[i % len(cols)]:
            with st.container(border=True):
                st.markdown(f"### {icon} {row['Sarja']}")
                st.markdown(f"**Tila:** {status}")
                st.metric("Vuosimuutos", f"{value:.1f} %" if value is not None else "—")
                st.caption(f"Viimeisin: {pd.to_datetime(row['Date']).date()}")

    st.markdown("### ⏳ Pidemmän aikavälin hintamuutos")

    cols = st.columns(min(4, len(latest)))

    for i, (_, row) in enumerate(latest.iterrows()):
        m3 = row.get("Muutos_3v")
        m5 = row.get("Muutos_5v")

        m3_val = float(m3) if pd.notna(m3) else None
        m5_val = float(m5) if pd.notna(m5) else None

        icon, status = _long_inflation_status(m5_val)

        with cols[i % len(cols)]:
            with st.container(border=True):
                st.markdown(f"### {icon} {row['Sarja']}")
                st.markdown(f"**Tila:** {status}")
                st.metric(
                    "5 v muutos",
                    f"{m5_val:+.1f} %" if m5_val is not None else "—",
                )
                st.caption(
                    f"3 v: {m3_val:+.1f} %" if m3_val is not None else "3 v: —"
                )

    st.divider()

    plot_df = series_df.copy()
    plot_df["Date"] = pd.to_datetime(plot_df["Date"], errors="coerce")
    plot_df["Indeksi"] = pd.to_numeric(plot_df["Indeksi"], errors="coerce")
    plot_df = plot_df.dropna(subset=["Date", "Sarja", "Indeksi"])

    if plot_df.empty:
        st.info("Pitkän aikavälin indeksikuvaajaa ei voitu muodostaa.")
        return

    plot_df = plot_df[plot_df["Sarja"].isin(preferred_order)].copy()

    start_date = plot_df["Date"].max() - pd.DateOffset(years=5)
    plot_df = plot_df[plot_df["Date"] >= start_date].copy()

    norm_frames = []

    for name, g in plot_df.groupby("Sarja"):
        g = g.sort_values("Date").copy()
        first_values = g["Indeksi"].dropna()

        if first_values.empty:
            continue

        first = float(first_values.iloc[0])

        if first == 0:
            continue

        g["Indeksi_100"] = g["Indeksi"] / first * 100.0
        norm_frames.append(g)

    if not norm_frames:
        st.info("Normalisoitua indeksikuvaajaa ei voitu muodostaa.")
        return

    norm_df = pd.concat(norm_frames, ignore_index=True)
    norm_df["order"] = norm_df["Sarja"].map({x: i for i, x in enumerate(preferred_order)}).fillna(99)
    norm_df = norm_df.sort_values(["order", "Date"])

    fig = px.line(
        norm_df,
        x="Date",
        y="Indeksi_100",
        color="Sarja",
        title="Hintojen kehitys viimeisen 5 vuoden aikana, alku = 100",
        labels={
            "Date": "Aika",
            "Indeksi_100": "Indeksi, alku = 100",
            "Sarja": "Mittari",
        },
    )

    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, width="stretch")