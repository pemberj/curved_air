"""
curved_air web interface

Main entry point for the Streamlit-based calculator web app.
Run with:  streamlit run src/curved_air/webapp/app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Curved Air — Optical Design Calculators",
    page_icon="🌈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Navigation ──────────────────────────────────────────────────────────────
pages: dict[str, list] = {
    "": [
        st.Page("pages/home.py", title="Home", icon="🏠"),
        st.Page("pages/spectral_format.py", title="Spectral Format", icon="🌈"),
    ],
}

pg = st.navigation(pages)
pg.run()
