import streamlit as st

from tabs import oil, gold, realestate, bitcoin, macro, wood, dashboard, energy, currency
from services.kpi import build_kpi_items
from services.error_utils import safe_render, safe_value

st.set_page_config(
    page_title="Korvenmarkkinat",
    layout="wide"
)

SHOW_DEBUG_DETAILS = False

@st.cache_data(ttl=60 * 30, show_spinner=False)
def load_kpi_items_cached():
    return build_kpi_items()

def kpi_panel(items: list[dict]):
    try:
        box = st.container(border=True)
    except TypeError:
        box = st.container()

    with box:
        st.markdown("### 📌 Kooste")

        if not items:
            st.info("Koostedataa ei saatu ladattua.")
            return

        for it in items:
            st.markdown(f"**{it.get('name', '')}**")

            sub = it.get("sub")
            if sub:
                st.caption(sub)

            value = it.get("value", "–")
            delta = it.get("delta") or None

            st.metric(label="", value=value, delta=delta)
            st.divider()


def top_hero():
    logo_path = "tabs/assets/korvenmarkkinat_logo.png"
    left, mid, right = st.columns([1, 1.2, 1])

    with mid:
        try:
            st.image(logo_path, use_container_width=True)
        except Exception:
            st.warning("Logo-kuvaa ei löytynyt.")

    st.divider()

@st.dialog("🌲 Tervetuloa Korvenmarkkinoille")
def welcome_dialog():
    st.markdown(
        """
<style>
div[data-testid="stDialog"] {
    backdrop-filter: blur(6px);
}

div[data-testid="stDialog"] > div {
    border-radius: 22px;
    padding: 1.5rem;
    animation: fadeInScale 0.35s ease-out;
}

@keyframes fadeInScale {
    from {
        opacity: 0;
        transform: scale(0.94);
    }
    to {
        opacity: 1;
        transform: scale(1);
    }
}
</style>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
Korvenmarkkinat auttaa ymmärtämään, missä taloudessa mennään – ja mitä se tarkoittaa sinulle käytännössä.

Sovellus on suunniteltu erityisesti metsänomistajille, kiinteistö- ja asuntokauppoja suunnitteleville sekä kaikille, jotka seuraavat Suomen energiamarkkinoita tai yleistä talouskehitystä.

🌲 Metsänomistajille Korvenmarkkinat tarjoaa ajankohtaista tietoa puun hinnoista ja metsäteollisuuden kehityksestä. Näiden avulla voit hahmottaa markkinatilannetta paremmin ja tehdä perustellumpia päätöksiä esimerkiksi puukaupan ajoituksesta.

🏡 Asunto- ja kiinteistökauppaa suunnitteleville sovellus tuo yhteen keskeisiä tekijöitä, kuten korkotason, inflaation ja hintakehityksen. Näin saat kokonaiskuvan siitä, millaisessa markkinaympäristössä olet tekemässä suuria päätöksiä.

⚡ Energiasta kiinnostuneille näkymät Suomen sähkön tuotantoon, kulutukseen ja spot-hintaan auttavat ymmärtämään hintojen vaihtelua ja markkinan dynamiikkaa – oli kyse sitten arjen sähkölaskusta tai laajemmasta energiapolitiikasta.

🌍 Makrotalouden ja markkinoiden seuraajille Korvenmarkkinat kokoaa yhteen keskeiset tunnusluvut: rahamäärän kehityksen, inflaation, korot, valuuttakurssit sekä raaka-aineiden hinnat kuten öljy, kulta ja bitcoin. Näiden avulla saat paremman kokonaiskuvan talouden suunnasta ja taustalla vaikuttavista voimista.

Korvenmarkkinat ei ole vain datan näyttämistä – se on työkalu parempaan ymmärrykseen ja päätöksentekoon.
"""
    )

    st.info("Tiedot ovat suuntaa-antavia eivätkä ole sijoitus-, vero- tai metsänhoitoneuvontaa.")

    if st.button("Aloita käyttö", type="primary", use_container_width=True):
        st.session_state["welcome_seen"] = True
        st.rerun()

st.set_page_config(page_title="Korvenmarkkinat 🇫🇮", layout="wide")

top_hero()

if not st.session_state.get("welcome_seen", False):
    welcome_dialog()

PAGE_OPTIONS = [
    "🏠 Dashboard",
    "💱 Valuuttakurssit",
    "📊 Makrotalous",
    "🪙 Kulta ja hopea",
    "₿ Bitcoin",
    "🛢 Öljy ja polttoaineet",
    "🏠 Kiinteistöt ja rakentaminen",
    "🌲 Metsätalous",
    "☢️ Energia",
]

with st.sidebar:
    st.markdown("## ☰ Valikko")
    st.caption("Valitse sovelluksen näkymä")

    page = st.radio(
        label="",
        options=PAGE_OPTIONS,
        index=0,
        label_visibility="collapsed",
    )

    st.divider()
    st.caption("Korvenmarkkinat 🇫🇮")

if page == "🏠 Dashboard":
    left, right = st.columns([3, 1], gap="large")

    with left:
        safe_render("Dashboard", dashboard.render, show_details=SHOW_DEBUG_DETAILS)

    with right:
        kpi_items = safe_value(
            "Kooste",
            load_kpi_items_cached,
            fallback=[],
            show_details=SHOW_DEBUG_DETAILS,
        )
        kpi_panel(kpi_items)

elif page == "💱 Valuuttakurssit":
    safe_render("Valuuttakurssit", currency.render, show_details=SHOW_DEBUG_DETAILS)

elif page == "📊 Makrotalous":
    safe_render("Makrotalous", macro.render, show_details=SHOW_DEBUG_DETAILS)

elif page == "🪙 Kulta ja hopea":
    safe_render("Kulta ja hopea", gold.render, show_details=SHOW_DEBUG_DETAILS)

elif page == "₿ Bitcoin":
    safe_render("Bitcoin", bitcoin.render, show_details=SHOW_DEBUG_DETAILS)

elif page == "🛢 Öljy ja polttoaineet":
    safe_render("Öljy ja polttoaineet", oil.render, show_details=SHOW_DEBUG_DETAILS)

elif page == "🏠 Kiinteistöt ja rakentaminen":
    safe_render("Kiinteistöt ja rakentaminen", realestate.render, show_details=SHOW_DEBUG_DETAILS)

elif page == "🌲 Metsätalous":
    safe_render("Metsätalous", wood.render, show_details=SHOW_DEBUG_DETAILS)

elif page == "☢️ Energia":
    safe_render("Energia", energy.render, show_details=SHOW_DEBUG_DETAILS)















