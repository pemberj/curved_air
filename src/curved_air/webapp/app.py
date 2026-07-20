#!uv run streamlit run

"""
curved_air web interface

Main entry point for the multi-page Streamlit webapp

Run with `uv run streamlit run src/curved_air/webapp/app.py`
"""

from pathlib import Path

import streamlit as st

APP_PATH: Path = Path(__file__).parent

st.set_page_config(
    page_title="Curved Air: Optical Design Tools",
    page_icon=Path(APP_PATH / "static" / "icon.png"),
    layout="wide",
    initial_sidebar_state="expanded",
)

pages: list = [
    st.Page(
        page=APP_PATH / "pages" / "about.py",
        title="About",
        icon=":material/info:",
    ),
    st.Page(
        page=APP_PATH / "pages" / "echelle_spectrograph.py",
        title="Echelle Spectrograph",
        icon="🌈",
    ),
    st.Page(
        page=APP_PATH / "pages" / "low_resolution_spectrograph.py",
        title="Low Resolution Spectrograph",
        icon="🌈",
    ),
    st.Page(
        page=APP_PATH / "pages" / "atmospheric_dispersion_corrector.py",
        title="Atmospheric Dispersion Corrector",
        icon="🌠",
    ),
    st.Page(
        page=APP_PATH / "pages" / "fibre_loss_converter.py",
        title="Fibre Loss Converter",
        icon="🧵",
    ),
    st.Page(
        page=APP_PATH / "pages" / "glass_explorer.py",
        title="Glass Explorer",
        icon="🔍",
    ),
    st.Page(
        page=APP_PATH / "pages" / "surface_tolerances.py",
        title="Surface Tolerances",
        icon="🎯",
    ),
    st.Page(
        page=APP_PATH / "pages" / "telescope_designer.py",
        title="Telescope Designer",
        icon="🔭",
    ),
    st.Page(
        page=APP_PATH / "pages" / "cte_match.py",
        title="CTE Match",
        icon="🔩",
    ),
    st.Page(
        page=APP_PATH / "pages" / "solid_angle_and_etendue.py",
        title="Solid Angle & Étendue",
        icon="📐",
    ),
    st.Page(
        page=APP_PATH / "pages" / "guiding_simulator.py",
        title="Guiding Simulator",
        icon="✨",
    ),
]

nav = st.navigation(pages={"Curved Air: Optical Design Tools": pages})

nav.run()
