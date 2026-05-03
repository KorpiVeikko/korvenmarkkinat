import streamlit as st
import plotly.express as px


def chart_with_period_selector(df, title: str, key: str):
    period = st.segmented_control(
        "Kuvaajan tarkasteluväli",
        options=["1 kk", "1 v", "5 v"],
        default="1 v",
        key=key,
    )

    df_plot = df.copy()

    if period == "1 kk":
        df_plot = df_plot.tail(30)
    elif period == "1 v":
        df_plot = df_plot.tail(365)

    fig = px.line(df_plot, x="Date", y="Close", title=title)
    st.plotly_chart(fig, use_container_width=True)