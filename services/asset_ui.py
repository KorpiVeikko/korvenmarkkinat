# services/asset_ui.py
from __future__ import annotations

from typing import Callable

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def pct_change(now, then):
    if now is None or then is None or then == 0:
        return None
    return (now / then - 1) * 100


def safe_eur_card(label: str, value, pct=None, decimals: int = 0):
    if value is None or pd.isna(value):
        st.metric(label, "–")
        return

    delta = f"{pct:+.1f} %" if pct is not None and not pd.isna(pct) else None
    st.metric(label, f"{float(value):,.{decimals}f} €".replace(",", " "), delta)


def safe_number_card(
    label: str,
    value,
    pct=None,
    decimals: int = 2,
    caption: str | None = None,
):
    if value is None or pd.isna(value):
        st.metric(label, "–")
    else:
        delta = f"{pct:+.1f} %" if pct is not None and not pd.isna(pct) else None
        st.metric(label, f"{float(value):,.{decimals}f}".replace(",", " "), delta)

    if caption:
        st.caption(caption)


def _closest_value_before_or_on(
    df: pd.DataFrame,
    target_date: pd.Timestamp,
    value_col: str,
    date_col: str = "Date",
) -> float | None:
    if df is None or df.empty or date_col not in df.columns or value_col not in df.columns:
        return None

    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=[date_col, value_col]).sort_values(date_col)

    older = d[d[date_col] <= target_date]
    if older.empty:
        return None

    return float(older.iloc[-1][value_col])


def latest_period_values(df: pd.DataFrame, value_col: str = "Close") -> dict:
    empty = {
        "now": None,
        "1m": None,
        "1y": None,
        "5y": None,
        "pct_1m": None,
        "pct_1y": None,
        "pct_5y": None,
    }

    if df is None or df.empty or value_col not in df.columns or "Date" not in df.columns:
        return empty

    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=["Date", value_col]).sort_values("Date").reset_index(drop=True)

    if d.empty:
        return empty

    now = float(d.iloc[-1][value_col])
    latest_date = pd.to_datetime(d.iloc[-1]["Date"])

    v1m = _closest_value_before_or_on(d, latest_date - pd.DateOffset(months=1), value_col)
    v1y = _closest_value_before_or_on(d, latest_date - pd.DateOffset(years=1), value_col)
    v5y = _closest_value_before_or_on(d, latest_date - pd.DateOffset(years=5), value_col)

    return {
        "now": now,
        "1m": v1m,
        "1y": v1y,
        "5y": v5y,
        "pct_1m": pct_change(now, v1m),
        "pct_1y": pct_change(now, v1y),
        "pct_5y": pct_change(now, v5y),
    }


def filter_by_period(df: pd.DataFrame, period: str, date_col: str = "Date") -> pd.DataFrame:
    if df is None or df.empty or date_col not in df.columns:
        return pd.DataFrame()

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col]).sort_values(date_col)

    if out.empty:
        return out

    max_date = out[date_col].max()

    if period == "1 kk":
        cutoff = max_date - pd.DateOffset(months=1)
    elif period == "1 v":
        cutoff = max_date - pd.DateOffset(years=1)
    elif period == "5 v":
        cutoff = max_date - pd.DateOffset(years=5)
    else:
        cutoff = max_date - pd.DateOffset(years=10)

    return out[out[date_col] >= cutoff].copy()


def period_selector(
    label: str,
    key: str,
    options: list[str] | None = None,
    default: str = "1 v",
) -> str:
    if options is None:
        options = ["1 kk", "1 v", "5 v"]

    period = st.segmented_control(
        label,
        options=options,
        default=default,
        key=key,
    )
    return period or default


def build_line_figure(
    df: pd.DataFrame,
    base_col: str,
    title: str,
    date_col: str = "Date",
    y_title: str = "",
    base_name: str | None = None,
    extra_lines: list[tuple[str, str, dict]] | None = None,
):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df[date_col],
            y=df[base_col],
            mode="lines",
            name=base_name or title,
        )
    )

    if extra_lines:
        for col_name, trace_name, trace_kwargs in extra_lines:
            if col_name in df.columns and df[col_name].notna().any():
                fig.add_trace(
                    go.Scatter(
                        x=df[date_col],
                        y=df[col_name],
                        mode="lines",
                        name=trace_name,
                        **trace_kwargs,
                    )
                )

    fig.update_layout(
        title=title,
        xaxis_title="Päivä",
        yaxis_title=y_title,
        legend_title_text="Sarja" if extra_lines else "",
    )
    return fig


def render_price_chart(
    df: pd.DataFrame,
    title: str,
    key: str,
    y_col: str = "Close",
    date_col: str = "Date",
    y_title: str = "€",
    options: list[str] | None = None,
    default: str = "1 v",
    postprocess: Callable | None = None,
):
    period = period_selector("Kuvaajan tarkasteluväli", key=key, options=options, default=default)
    plot_df = filter_by_period(df, period, date_col=date_col)

    fig = build_line_figure(
        plot_df,
        base_col=y_col,
        title=f"{title} ({period})",
        date_col=date_col,
        y_title=y_title,
        base_name=title,
    )

    if postprocess is not None:
        fig = postprocess(fig, plot_df, period)

    st.plotly_chart(fig, use_container_width=True)
    return period, plot_df, fig


def render_price_chart_with_extra_lines(
    df: pd.DataFrame,
    title: str,
    key: str,
    base_col: str,
    extra_lines: list[tuple[str, str, dict]] | None = None,
    date_col: str = "Date",
    y_title: str = "€",
    options: list[str] | None = None,
    default: str = "1 v",
    postprocess: Callable | None = None,
):
    period = period_selector("Kuvaajan tarkasteluväli", key=key, options=options, default=default)
    plot_df = filter_by_period(df, period, date_col=date_col)

    fig = build_line_figure(
        plot_df,
        base_col=base_col,
        title=f"{title} ({period})",
        date_col=date_col,
        y_title=y_title,
        base_name=title,
        extra_lines=extra_lines,
    )

    if postprocess is not None:
        fig = postprocess(fig, plot_df, period)

    st.plotly_chart(fig, use_container_width=True)
    return period, plot_df, fig