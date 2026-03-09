# tabs/forest.py
import streamlit as st
import pandas as pd
import plotly.express as px

DATA_PATH = "data/forest_data.xlsx"


def render():
    st.subheader("🌲 Metsämaan kauppahinta (€/ha)")
    st.caption("Lähde: Maanmittauslaitos / KHR – yli 10 ha metsätilat")

    # -----------------------------
    # DATA
    # -----------------------------
    try:
        df = pd.read_excel(DATA_PATH, skiprows=1)

        df.columns = [
            "Vuosi",
            "Maakuntakoodi",
            "Maakunta",
            "Kauppojen_lkm",
            "Pinta_ala_ha",
            "Mediaani_eur_ha",
            "Keskihinta_eur_ha",
            "Keskihajonta_eur_ha"
        ]

        df["Vuosi"] = df["Vuosi"].astype(int)
        df["Keskihinta_eur_ha"] = pd.to_numeric(
            df["Keskihinta_eur_ha"], errors="coerce"
        )

        df = df.dropna(subset=["Maakunta", "Vuosi"])

    except Exception as e:
        st.error(f"Metsädata ei latautunut: {e}")
        return

    # -----------------------------
    # VALINNAT
    # -----------------------------
    maakunnat = sorted(df["Maakunta"].unique())
    latest_year = df["Vuosi"].max()

    st.markdown("### 🎯 Valitse tarkastelu")

    selected_region = st.selectbox(
        "Maakunta (yksityiskohtainen tarkastelu)",
        maakunnat,
        index=maakunnat.index("Uusimaa") if "Uusimaa" in maakunnat else 0
    )

    compare_regions = st.multiselect(
        "Vertailuun (valitse 1–3 maakuntaa)",
        maakunnat,
        default=[selected_region],
        max_selections=3
    )

    # -----------------------------
    # YKSITTÄINEN MAAKUNTA
    # -----------------------------
    st.markdown(f"### 📈 Kehitys: {selected_region}")

    region_df = df[df["Maakunta"] == selected_region]

    fig_region = px.line(
        region_df,
        x="Vuosi",
        y="Keskihinta_eur_ha",
        markers=True,
        title=f"Metsämaan keskihinta €/ha – {selected_region}"
    )

    st.plotly_chart(fig_region, use_container_width=True)

    # -----------------------------
    # VERTAILU: VALITUT MAAKUNNAT
    # -----------------------------
    if len(compare_regions) > 1:
        st.markdown("### 🔍 Vertailu: valitut maakunnat")

        compare_df = df[df["Maakunta"].isin(compare_regions)]

        fig_compare = px.line(
            compare_df,
            x="Vuosi",
            y="Keskihinta_eur_ha",
            color="Maakunta",
            markers=True
        )

        st.plotly_chart(fig_compare, use_container_width=True)

    # -----------------------------
    # KOONTI: VIIMEISIN VUOSI
    # -----------------------------
    st.markdown(f"### 📊 Maakuntien vertailu ({latest_year})")

    latest_df = (
        df[df["Vuosi"] == latest_year]
        .sort_values("Keskihinta_eur_ha", ascending=False)
    )

    fig_bar = px.bar(
        latest_df,
        x="Maakunta",
        y="Keskihinta_eur_ha",
        title=f"Metsämaan keskihinta €/ha ({latest_year})",
    )

    fig_bar.update_layout(
        xaxis_tickangle=-45
    )

    st.plotly_chart(fig_bar, use_container_width=True)

    # -----------------------------
    # RAAKADATA
    # -----------------------------
    with st.expander("🔍 Näytä raakadata"):
        st.dataframe(df, use_container_width=True)






 


