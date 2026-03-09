# services/ui.py
import streamlit as st

def apply_base_styles():
    st.markdown(
        """
        <style>
          .block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1400px; }

          /* --- TOP NAVBAR --- */
          .topbar {
            position: sticky;
            top: 0;
            z-index: 999;
            background: #0b1220;
            border-bottom: 1px solid rgba(255,255,255,0.08);
            padding: 10px 14px;
            border-radius: 14px;
            margin-bottom: 14px;
          }
          .topbar-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
          }
          .brand {
            display: flex;
            align-items: center;
            gap: 10px;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: #f3f4f6;
            font-size: 18px;
          }
          .navlinks {
            display: flex;
            gap: 14px;
            flex-wrap: wrap;
            align-items: center;
          }
          .navlinks a {
            color: rgba(255,255,255,0.8);
            text-decoration: none;
            font-weight: 600;
            padding: 8px 10px;
            border-radius: 10px;
            border: 1px solid transparent;
          }
          .navlinks a:hover {
            background: rgba(255,255,255,0.06);
            border-color: rgba(255,255,255,0.08);
          }

          /* --- KPI PANEL (right) --- */
          .kpi-panel {
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            background: rgba(255,255,255,0.03);
            padding: 14px;
          }
          .kpi-title {
            font-weight: 800;
            letter-spacing: -0.02em;
            margin-bottom: 10px;
          }
          .kpi-item {
            display:flex;
            align-items:center;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid rgba(255,255,255,0.06);
          }
          .kpi-item:last-child { border-bottom: 0; }
          .kpi-left {
            display:flex;
            flex-direction: column;
            gap: 2px;
          }
          .kpi-name {
            font-weight: 700;
            opacity: 0.95;
          }
          .kpi-sub {
            font-size: 12px;
            opacity: 0.65;
          }
          .kpi-right {
            text-align: right;
            font-weight: 800;
          }
          .kpi-delta {
            font-size: 12px;
            font-weight: 700;
          }
          .pos { color: #22c55e; }
          .neg { color: #ef4444; }
          .muted { opacity: 0.78; font-size: 0.92rem; }

          /* Sidebar border */
          section[data-testid="stSidebar"] {
            border-right: 1px solid rgba(255,255,255,0.08);
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def top_navbar(active: str | None = None):
    # Linkit eivät “vaihda sivua” itsessään (Streamlitissä reititys hoidetaan pythonilla),
    # mutta tämä antaa Binance-fiiliksen ja voi toimia ankkureina.
    st.markdown(
        f"""
        <div class="topbar">
          <div class="topbar-row">
            <div class="brand">📊 Suomen talouden seuranta</div>
            <div class="navlinks">
              <a href="#dashboard">Dashboard</a>
              <a href="#makro">Makrotalous</a>
              <a href="#markkinat">Markkinat</a>
              <a href="#kiinteistot">Kiinteistöt</a>
              <a href="#metsa">Metsä</a>
              <a href="#blogi">Blogi</a>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_panel(items: list[dict]):
    """
    items = [
      {"name":"BTC", "sub":"24h", "value":"$88,351", "delta":"+0.05%"},
      ...
    ]
    """
    rows = ""
    for it in items:
        delta = it.get("delta", "")
        cls = ""
        if isinstance(delta, str):
            if delta.strip().startswith("-"):
                cls = "neg"
            elif delta.strip().startswith("+"):
                cls = "pos"

        rows += f"""
          <div class="kpi-item">
            <div class="kpi-left">
              <div class="kpi-name">{it.get("name","")}</div>
              <div class="kpi-sub">{it.get("sub","")}</div>
            </div>
            <div class="kpi-right">
              <div>{it.get("value","")}</div>
              <div class="kpi-delta {cls}">{delta}</div>
            </div>
          </div>
        """

    st.markdown(
        f"""
        <div class="kpi-panel">
          <div class="kpi-title">Keskeiset tunnusluvut</div>
          {rows}
        </div>
        """,
        unsafe_allow_html=True,
    )

