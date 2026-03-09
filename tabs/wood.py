# tabs/wood.py
import streamlit as st
import pandas as pd
import plotly.express as px

from services.luke_pxweb import fetch_wood_prices, add_week_sort_key


@st.cache_data(show_spinner="Haetaan LUKE:n viikkokantohintoja…")
def load_wood_data() -> pd.DataFrame:
    # Ladataan koko setti kerralla -> suodatus käyttöliittymässä
    return fetch_wood_prices(response_format="json-stat2")


def _guess_cols(df: pd.DataFrame):
    cols = list(df.columns)

    week_candidates = [c for c in cols if str(c).lower() in ("w", "viikko", "week")]
    week_col = week_candidates[0] if week_candidates else None

    area_candidates = [c for c in cols if str(c).lower() in ("mpkh", "alue", "hinta-alue", "hintaalue")]
    area_col = area_candidates[0] if area_candidates else None

    h_candidates = [c for c in cols if str(c).lower() in ("hakt", "hakkuutapa")]
    h_col = h_candidates[0] if h_candidates else None

    p_candidates = [c for c in cols if str(c).lower() in ("ptl", "puutavaralaji")]
    p_col = p_candidates[0] if p_candidates else None

    return week_col, area_col, h_col, p_col


def render():
    st.subheader("🪵 Puun kantohinnat (viikkoseuranta)")
    st.caption("Lähde: Luonnonvarakeskus (Luke) / PXWeb – 0100_metryv (viikkokantohinnat)")

    try:
        df = load_wood_data()
    except Exception as e:
        st.error(f"Datan haku epäonnistui: {e}")
        return

    if df is None or df.empty:
        st.warning("Dataa ei saatu ladattua (tyhjä vastaus).")
        return

    week_col, area_col, h_col, p_col = _guess_cols(df)
    if not all([week_col, area_col, h_col, p_col]):
        st.error(
            "En löytänyt tarvittavia sarakkeita (viikko/alue/hakkuutapa/puutavaralaji). "
            f"Sarakkeet: {list(df.columns)}"
        )
        return

    # Sort-avain viikoille
    df = add_week_sort_key(df, week_col).copy()
    df = df.dropna(subset=["sort_key", "Arvo"]).sort_values("sort_key").reset_index(drop=True)

    # Varmistetaan numerot
    df["sort_key"] = pd.to_numeric(df["sort_key"], errors="coerce")
    df["Arvo"] = pd.to_numeric(df["Arvo"], errors="coerce")
    df = df.dropna(subset=["sort_key", "Arvo"])

    # UI-valinnat
    with st.expander("⚙️ Valinnat", expanded=True):
        c1, c2, c3 = st.columns(3)

        all_areas = sorted(df[area_col].dropna().unique().tolist())
        all_hakt = sorted(df[h_col].dropna().unique().tolist())
        all_ptl = sorted(df[p_col].dropna().unique().tolist())

        with c1:
            chosen_area = st.selectbox("Alue", all_areas, index=0 if all_areas else None)
        with c2:
            chosen_hakt = st.selectbox("Hakkuutapa", all_hakt, index=0 if all_hakt else None)
        with c3:
            chosen_ptl = st.multiselect("Puutavaralajit", all_ptl, default=all_ptl)

        keys = sorted(df["sort_key"].astype(int).unique().tolist())
        if len(keys) < 2:
            st.warning("Viikkoja liian vähän suodatukseen.")
            return

        min_key, max_key = int(keys[0]), int(keys[-1])
        default_start = int(keys[max(0, len(keys) - 260)])  # ~5 vuotta
        key_range = st.slider(
            "Aikaväli (viikot, YYYYWW)",
            min_value=min_key,
            max_value=max_key,
            value=(default_start, max_key),
            step=1
        )

    # Suodatetaan data: yksi alue + yksi hakkuutapa + valitut puutavaralajit + aikaväli
    f = df[
        (df[area_col] == chosen_area)
        & (df[h_col] == chosen_hakt)
        & (df[p_col].isin(chosen_ptl))
        & (df["sort_key"].between(key_range[0], key_range[1]))
    ].copy()

    if f.empty:
        st.info("Valinnoilla ei löytynyt dataa.")
        return

    # ---------- Koonti: viimeisin viikko ----------
    latest_key = int(f["sort_key"].max())
    latest_week = f.loc[f["sort_key"] == latest_key, week_col].iloc[0]
    latest = f[f["sort_key"] == latest_key].copy()

    st.markdown("### 🧾 Koonti (valittu alue + hakkuutapa)")
    st.write(f"**Alue:** {chosen_area}  |  **Hakkuutapa:** {chosen_hakt}  |  **Viikko:** {latest_week}")

    # Bar: viimeisin viikko (puutavaralajit)
    bar = px.bar(
        latest.sort_values("Arvo", ascending=False),
        x=p_col,
        y="Arvo",
        title="Kantohinnat (€/m³) – viimeisin viikko",
        labels={"Arvo": "€/m³", p_col: "Puutavaralaji"},
    )
    st.plotly_chart(bar, use_container_width=True)

    # ---------- Selkeä viivakaavio ----------
    st.markdown("### 📈 Kantohinnan kehitys (€/m³)")
    line = px.line(
        f,
        x="sort_key",
        y="Arvo",
        color=p_col,
        markers=False,
        title=f"Kehitys viikoittain – {chosen_area} / {chosen_hakt}",
        labels={"sort_key": "Viikko (YYYYWW)", "Arvo": "€/m³", p_col: "Puutavaralaji"},
    )

    # Parannetaan luettavuutta (isot fontit + enemmän tilaa legendalle)
    line.update_layout(
        legend_title_text="Puutavaralaji",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    st.plotly_chart(line, use_container_width=True)

    # ---------- Aluevertailu viimeisimmällä viikolla ----------
    st.markdown("### 🗺️ Aluevertailu (viimeisin viikko)")

    # Otetaan koko datasta sama viimeisin viikko (ei rajata alueeseen)
    global_latest_key = int(df["sort_key"].max())
    gl = df[df["sort_key"] == global_latest_key].copy()
    if gl.empty:
        st.info("Aluevertailuun ei löytynyt dataa.")
        return

    global_latest_week = gl[week_col].iloc[0]
    st.write(f"**Viikko:** {global_latest_week}")

    cA, cB = st.columns(2)
    with cA:
        comp_hakt = st.selectbox("Vertailu – hakkuutapa", sorted(gl[h_col].unique().tolist()), key="comp_hakt")
    with cB:
        comp_ptl = st.selectbox("Vertailu – puutavaralaji", sorted(gl[p_col].unique().tolist()), key="comp_ptl")

    comp = gl[(gl[h_col] == comp_hakt) & (gl[p_col] == comp_ptl)].copy()
    comp = comp.dropna(subset=[area_col, "Arvo"]).sort_values("Arvo", ascending=False)

    comp_bar = px.bar(
        comp,
        x=area_col,
        y="Arvo",
        title=f"Aluevertailu (€/m³): {comp_ptl} / {comp_hakt}",
        labels={"Arvo": "€/m³", area_col: "Alue"},
    )
    st.plotly_chart(comp_bar, use_container_width=True)

    with st.expander("🔍 Näytä raakadata"):
        st.dataframe(f, use_container_width=True)

