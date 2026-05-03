from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from services.realestate_helpers import calc_quarterly_yoy, latest_and_yoy


def add_latest_value_annotation(
    fig: go.Figure,
    df: pd.DataFrame,
    value_fmt: str = ".1f",
    x_col: str = "Jakso_dt",
    y_col: str = "Arvo",
    yshift: int = 0,
) -> None:
    if df is None or df.empty:
        return

    last_row = df.sort_values(x_col).iloc[-1]
    x_val = last_row[x_col]
    y_val = last_row[y_col]

    fig.add_annotation(
        x=x_val,
        y=y_val,
        text=f"{y_val:{value_fmt}}",
        showarrow=False,
        xanchor="left",
        yanchor="middle",
        xshift=10,
        yshift=yshift,
        bgcolor="rgba(255,255,255,0.75)",
        bordercolor="rgba(0,0,0,0.15)",
        borderwidth=1,
    )


def render_asunnot_counts_chart(df_counts: pd.DataFrame) -> None:
    fig = px.bar(
        df_counts,
        x="Kvartaali",
        y="Arvo",
        title="Asuntokauppojen lukumäärä kvartaaleittain",
        labels={"Arvo": "Kauppojen lukumäärä", "Kvartaali": "Kvartaali"},
    )
    st.plotly_chart(fig, width="stretch")


def render_asunnot_prices_chart(df_prices: pd.DataFrame) -> None:
    fig = px.line(
        df_prices,
        x="Kvartaali",
        y="Arvo",
        markers=True,
        title="Uusien asuntojen keskimääräinen neliöhinta",
        labels={"Arvo": "€/m²", "Kvartaali": "Kvartaali"},
    )
    st.plotly_chart(fig, width="stretch")


def render_pelto_koko_maa_chart(koko_maa_df: pd.DataFrame, series_label: str) -> None:
    fig = px.line(
        koko_maa_df,
        x="Vuosi",
        y="Arvo",
        markers=True,
        title=f"Peltomaan {series_label} – koko maa",
        labels={"Arvo": "€/ha", "Vuosi": "Vuosi"},
    )
    st.plotly_chart(fig, width="stretch")


def render_pelto_alueet_chart(alue_df: pd.DataFrame, series_label: str) -> None:
    fig = px.line(
        alue_df,
        x="Vuosi",
        y="Arvo",
        color="Alue",
        markers=True,
        title=f"Peltomaan {series_label} alueittain",
        labels={"Arvo": "€/ha", "Vuosi": "Vuosi", "Alue": "Alue"},
    )
    st.plotly_chart(fig, width="stretch")


def render_tontti_koko_maa_index_chart(koko_hinta_df: pd.DataFrame, koko_real_df: pd.DataFrame) -> None:
    st.subheader("📈 Hintaindeksi ja reaalihintaindeksi – koko maa")

    if koko_hinta_df.empty and koko_real_df.empty:
        st.info("Koko maan hintaindeksin tai reaalihintaindeksin dataa ei löytynyt.")
        return

    fig = go.Figure()

    if not koko_hinta_df.empty:
        fig.add_trace(
            go.Scatter(
                x=koko_hinta_df["Jakso_dt"],
                y=koko_hinta_df["Arvo"],
                mode="lines",
                name="Hintaindeksi",
            )
        )
        add_latest_value_annotation(fig, koko_hinta_df, value_fmt=".1f", yshift=12)

    if not koko_real_df.empty:
        fig.add_trace(
            go.Scatter(
                x=koko_real_df["Jakso_dt"],
                y=koko_real_df["Arvo"],
                mode="lines",
                name="Reaalihintaindeksi",
            )
        )
        add_latest_value_annotation(fig, koko_real_df, value_fmt=".1f", yshift=-12)

    fig.update_layout(
        title="Omakotitalotonttien hintaindeksi ja reaalihintaindeksi – koko maa",
        xaxis_title="Aika",
        yaxis_title="Indeksi",
        margin=dict(r=120),
    )
    st.plotly_chart(fig, width="stretch")


def render_tontti_koko_maa_nelio_chart(koko_nelio_df: pd.DataFrame) -> None:
    st.subheader("💶 Neliöhinta – koko maa")

    if koko_nelio_df.empty:
        st.info("Koko maan neliöhintadataa ei löytynyt.")
        return

    df = koko_nelio_df.sort_values("Jakso_dt").copy()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["Jakso_dt"],
            y=df["Arvo"],
            mode="lines",
            name="Neliöhinta",
        )
    )

    add_latest_value_annotation(fig, df, value_fmt=".0f")

    fig.update_layout(
        title="Omakotitalotonttien neliöhinta – koko maa",
        xaxis_title="Aika",
        yaxis_title="€/m²",
        margin=dict(r=100),
    )
    st.plotly_chart(fig, width="stretch")


def render_tontti_selected_area_index_chart(
    koko_hinta_df: pd.DataFrame,
    koko_real_df: pd.DataFrame,
    alue_hinta_df: pd.DataFrame,
    alue_real_df: pd.DataFrame,
) -> None:
    alueet = sorted(
        set(alue_hinta_df["Alue"].dropna().unique()) | set(alue_real_df["Alue"].dropna().unique())
    )

    options = ["Koko maa"] + [a for a in alueet if str(a) != "Koko maa"]

    selected_area = st.selectbox(
        "Valitse alue indeksikuvaajaan",
        options=options,
        index=0,
        key="tontti_index_area_select",
    )

    if selected_area == "Koko maa":
        hinta = koko_hinta_df.copy().sort_values("Jakso_dt")
        reaali = koko_real_df.copy().sort_values("Jakso_dt")
    else:
        hinta = alue_hinta_df[alue_hinta_df["Alue"] == selected_area].copy().sort_values("Jakso_dt")
        reaali = alue_real_df[alue_real_df["Alue"] == selected_area].copy().sort_values("Jakso_dt")

    if hinta.empty and reaali.empty:
        st.info("Valitulle alueelle ei löytynyt indeksidataa.")
        return

    fig = go.Figure()

    if not hinta.empty:
        fig.add_trace(
            go.Scatter(
                x=hinta["Jakso_dt"],
                y=hinta["Arvo"],
                mode="lines",
                name="Hintaindeksi",
            )
        )
        add_latest_value_annotation(fig, hinta, value_fmt=".1f", yshift=12)

    if not reaali.empty:
        fig.add_trace(
            go.Scatter(
                x=reaali["Jakso_dt"],
                y=reaali["Arvo"],
                mode="lines",
                name="Reaalihintaindeksi",
            )
        )
        add_latest_value_annotation(fig, reaali, value_fmt=".1f", yshift=-12)

    fig.update_layout(
        title=f"Omakotitalotonttien hintaindeksi ja reaalihintaindeksi – {selected_area}",
        xaxis_title="Aika",
        yaxis_title="Indeksi",
        margin=dict(r=120),
    )
    st.plotly_chart(fig, width="stretch")


def render_tontti_selected_area_nelio_chart(
    koko_nelio_df: pd.DataFrame,
    alue_nelio_df: pd.DataFrame,
) -> None:
    alueet = sorted(alue_nelio_df["Alue"].dropna().unique().tolist())
    options = ["Koko maa"] + [a for a in alueet if str(a) != "Koko maa"]

    selected_area = st.selectbox(
        "Valitse alue neliöhintakuvaajaan",
        options=options,
        index=0,
        key="tontti_nelio_area_select",
    )

    fig = go.Figure()

    koko = koko_nelio_df.copy().sort_values("Jakso_dt")
    if not koko.empty:
        fig.add_trace(
            go.Scatter(
                x=koko["Jakso_dt"],
                y=koko["Arvo"],
                mode="lines",
                name="Koko maa",
                line=dict(width=3),
            )
        )
        if selected_area == "Koko maa":
            add_latest_value_annotation(fig, koko, value_fmt=".0f", yshift=0)

    if selected_area != "Koko maa":
        alue = alue_nelio_df[alue_nelio_df["Alue"] == selected_area].copy().sort_values("Jakso_dt")
        if not alue.empty:
            fig.add_trace(
                go.Scatter(
                    x=alue["Jakso_dt"],
                    y=alue["Arvo"],
                    mode="lines",
                    name=selected_area,
                )
            )
            add_latest_value_annotation(fig, alue, value_fmt=".0f", yshift=0)

    fig.update_layout(
        title=f"Omakotitalotonttien neliöhinta – {selected_area}",
        xaxis_title="Aika",
        yaxis_title="€/m²",
        margin=dict(r=100),
    )
    st.plotly_chart(fig, width="stretch")


def render_tontti_nelio_area_comparison_chart(alue_nelio_df: pd.DataFrame) -> None:
    st.subheader("🗺️ Neliöhinta alueittain")

    if alue_nelio_df.empty:
        st.info("Alueellista neliöhintadataa ei löytynyt.")
        return

    df = alue_nelio_df.copy().sort_values(["Alue", "Jakso_dt"])

    latest_points = df.groupby("Alue", as_index=False).tail(1).copy()
    latest_points = latest_points.sort_values("Arvo", ascending=False)

    bar = px.bar(
        latest_points,
        x="Alue",
        y="Arvo",
        title="Omakotitalotonttien viimeisin neliöhinta alueittain",
        labels={"Alue": "Alue", "Arvo": "€/m²"},
    )
    bar.update_xaxes(categoryorder="total descending")
    st.plotly_chart(bar, width="stretch")

    st.subheader("📌 Alueellinen neliöhinta – viimeisin arvo ja muutos vuodessa")

    metric_frames = []
    for area, group in df.groupby("Alue"):
        group_yoy = calc_quarterly_yoy(group)
        group_yoy["Alue"] = area
        metric_frames.append(group_yoy)

    if not metric_frames:
        return

    metric_df = pd.concat(metric_frames, ignore_index=True)
    latest_rows = metric_df.groupby("Alue", as_index=False).tail(1).sort_values("Alue")

    cols = st.columns(min(4, max(1, len(latest_rows))))
    for i, (_, row) in enumerate(latest_rows.iterrows()):
        area_df = metric_df[metric_df["Alue"] == row["Alue"]]
        latest_val, latest_yoy = latest_and_yoy(area_df)
        with cols[i % len(cols)]:
            st.metric(
                row["Alue"],
                f"{latest_val:,.0f} €/m²".replace(",", " ") if latest_val is not None else "—",
                f"{latest_yoy:+.1f} %" if latest_yoy is not None else None,
            )


def render_tontti_kauppamaara_chart(lkm_df: pd.DataFrame) -> None:
    if lkm_df.empty:
        st.info("Kauppojen lukumäärän dataa ei löytynyt.")
        return

    agg_lkm = (
        lkm_df.groupby("Jakso_dt", as_index=False)["Arvo"]
        .sum()
        .sort_values("Jakso_dt")
    )

    fig = px.bar(
        agg_lkm,
        x="Jakso_dt",
        y="Arvo",
        title="Omakotitalotonttien kauppojen lukumäärä",
        labels={"Arvo": "Lukumäärä", "Jakso_dt": "Aika"},
    )
    st.plotly_chart(fig, width="stretch")