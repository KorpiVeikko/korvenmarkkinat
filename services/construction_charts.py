from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st


def _latest_and_yoy_from_series(series: pd.Series, periods: int = 12) -> tuple[float | None, float | None]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None, None

    latest = float(s.iloc[-1])

    if len(s) <= periods:
        return latest, None

    prev = float(s.iloc[-(periods + 1)])
    if prev == 0:
        return latest, None

    yoy = (latest / prev - 1.0) * 100.0
    return latest, yoy


def _fmt_delta(yoy: float | None) -> str | None:
    if yoy is None:
        return None
    return f"{yoy:+.1f} % (1v)"


def _latest_stage_value_and_yoy(koko: pd.DataFrame, stage: str) -> tuple[float | None, float | None]:
    f = koko[koko["Vaihe"] == stage].copy()
    if f.empty or "Arvo_sum12" not in f.columns:
        return None, None

    f = f.sort_values("Aika_dt")
    return _latest_and_yoy_from_series(f["Arvo_sum12"], periods=12)


def _ratio_series(koko: pd.DataFrame) -> pd.DataFrame:
    if koko.empty or "Arvo_sum12" not in koko.columns:
        return pd.DataFrame()

    p = (
        koko.dropna(subset=["Aika_dt", "Vaihe", "Arvo_sum12"])
        .pivot_table(
            index="Aika_dt",
            columns="Vaihe",
            values="Arvo_sum12",
            aggfunc="last",
        )
        .reset_index()
        .sort_values("Aika_dt")
    )

    if "Rakennusluvat" not in p.columns or "Valmistuneet" not in p.columns:
        return pd.DataFrame()

    p = p[p["Valmistuneet"] != 0].copy()
    p["Luvat_valmistuneet_suhde"] = p["Rakennusluvat"] / p["Valmistuneet"]
    return p


def render_construction_leading_indicator(df: pd.DataFrame) -> None:
    st.subheader("🧭 Leading indicator – rakennusluvat → tuleva tarjonta")

    koko = df[df["Alue"] == "Koko maa"].copy()
    if koko.empty:
        st.info("Koko maan dataa ei löytynyt.")
        return

    if "Arvo_sum12" not in koko.columns:
        st.info("12 kk kertymää ei löytynyt datasta.")
        return

    permits, permits_yoy = _latest_stage_value_and_yoy(koko, "Rakennusluvat")
    started, started_yoy = _latest_stage_value_and_yoy(koko, "Rakenteilla")
    completed, completed_yoy = _latest_stage_value_and_yoy(koko, "Valmistuneet")

    ratio_df = _ratio_series(koko)
    ratio, ratio_yoy = (
        _latest_and_yoy_from_series(ratio_df["Luvat_valmistuneet_suhde"], periods=12)
        if not ratio_df.empty
        else (None, None)
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Rakennusluvat (12 kk)",
            f"{permits:,.0f}".replace(",", " ") if permits is not None else "—",
            _fmt_delta(permits_yoy),
        )

    with c2:
        st.metric(
            "Rakenteilla (12 kk)",
            f"{started:,.0f}".replace(",", " ") if started is not None else "—",
            _fmt_delta(started_yoy),
        )

    with c3:
        st.metric(
            "Valmistuneet (12 kk)",
            f"{completed:,.0f}".replace(",", " ") if completed is not None else "—",
            _fmt_delta(completed_yoy),
        )

    with c4:
        st.metric(
            "Luvat / valmistuneet",
            f"{ratio:.2f}" if ratio is not None else "—",
            _fmt_delta(ratio_yoy),
        )

    if ratio is None:
        st.info("Suhdelukua ei voitu laskea.")
        return

    if ratio > 1.15:
        msg = (
            "Rakennuslupia on 12 kuukauden kertymässä selvästi enemmän kuin valmistuneita asuntoja. "
            "Tämä viittaa siihen, että tuleva tarjonta voi vahvistua lähivuosina."
        )
    elif ratio < 0.85:
        msg = (
            "Rakennuslupia on 12 kuukauden kertymässä selvästi vähemmän kuin valmistuneita asuntoja. "
            "Tämä viittaa siihen, että tuleva tarjonta voi heikentyä, ellei lupamäärä piristy."
        )
    else:
        msg = (
            "Rakennuslupien ja valmistuneiden asuntojen 12 kuukauden kertymä on melko tasapainossa. "
            "Tuleva tarjonta näyttää tällä hetkellä suhteellisen vakaalta."
        )

    st.info(msg)


def render_construction_koko_maa(df: pd.DataFrame) -> None: 

    koko = df[df["Alue"] == "Koko maa"].copy()
    if koko.empty:
        st.info("Dataa ei löytynyt.")
        return

    plot_df = koko.dropna(subset=["Arvo_sum12"]).copy()
    if plot_df.empty:
        st.info("12 kk kertymädataa ei löytynyt.")
        return

    fig = px.line(
        plot_df,
        x="Aika_dt",
        y="Arvo_sum12",
        color="Vaihe",
        title="Rakennusluvat, rakenteilla ja valmistuneet – koko maa (12 kk kertymä)",
        labels={
            "Aika_dt": "Aika",
            "Arvo_sum12": "Asuntoja (12 kk kertymä)",
            "Vaihe": "Vaihe",
        },
    )

    st.plotly_chart(fig, width="stretch")


def render_construction_area(df: pd.DataFrame) -> None:

    alueet = sorted([a for a in df["Alue"].dropna().unique() if a != "Koko maa"])
    if not alueet:
        st.info("Aluedataa ei löytynyt.")
        return

    selected = st.selectbox("Valitse alue", alueet, key="construction_area_select")

    dff = df[df["Alue"] == selected].copy()
    dff = dff.dropna(subset=["Arvo_sum12"]).copy()

    if dff.empty:
        st.info("Valitulle alueelle ei löytynyt dataa.")
        return

    fig = px.line(
        dff,
        x="Aika_dt",
        y="Arvo_sum12",
        color="Vaihe",
        title=f"Rakennusluvat, rakenteilla ja valmistuneet – {selected} (12 kk kertymä)",
        labels={
            "Aika_dt": "Aika",
            "Arvo_sum12": "Asuntoja (12 kk kertymä)",
            "Vaihe": "Vaihe",
        },
    )

    st.plotly_chart(fig, width="stretch")