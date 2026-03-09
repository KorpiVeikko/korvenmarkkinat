import streamlit as st

from tabs import oil, gold, realestate, bitcoin, forest, macro, wood, dashboard, energy, currency
from services.kpi import build_kpi_items


def kpi_panel(items: list[dict]):
    # Kehys (Streamlit 1.25+): jos ei tue, vaihda -> st.container()
    try:
        box = st.container(border=True)
    except TypeError:
        box = st.container()

    with box:
        st.markdown("### 📌 Kooste")

        for it in items:
            st.markdown(f"**{it.get('name','')}**")

            sub = it.get("sub")
            if sub:
                st.caption(sub)

            value = it.get("value", "–")
            delta = it.get("delta") or None

            # st.metric näyttää deltan vihreä/punainen automaattisesti
            st.metric(label="", value=value, delta=delta)

            st.divider()


def top_hero():
    logo_path = "tabs/assets/korvenmarkkinat_logo.png"
    left, mid, right = st.columns([1, 1.2, 1])
    with mid:
        try:
            st.image(logo_path, use_container_width=True)
        except Exception:
            st.warning("Hero-kuvaa ei löytynyt (korvenmarkkinat_logo.png).")
    st.divider()


st.set_page_config(page_title="Korvenmarkkinat 🇫🇮", layout="wide")

top_hero()

tabs = st.tabs([   
    "🏠 Dashboard",
    "💱 Valuuttakurssit",
    "📊 Makrotalous",
    "🪙 Kulta ja hopea",
    "₿ Bitcoin",
    "🛢 Raakaöljy",   
    "🏠 Kiinteistöt",   
    "🌲 Metsäkiinteistöt",    
    "🪵 Puuhinta",
    "☢️ Energia",
])

with tabs[0]:
    left, right = st.columns([3, 1], gap="large")
    with left:
        dashboard.render()
    with right:
        kpi_panel(build_kpi_items())

with tabs[1]:
    currency.render()
with tabs[2]:
    macro.render()
with tabs[3]:
    gold.render()
with tabs[4]:
    bitcoin.render()
with tabs[5]:
    oil.render()
with tabs[6]:
    realestate.render()
with tabs[7]:
    forest.render()
with tabs[8]:
    wood.render()
with tabs[9]:
    energy.render()    
















