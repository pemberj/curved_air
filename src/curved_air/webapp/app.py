#!uv run streamlit run

"""
curved_air web interface

Main entry point for the multi-page Streamlit webapp

Run with `uv run streamlit run src/curved_air/webapp/app.py`
"""

import streamlit as st

st.set_page_config(
    page_title="Curved Air: Optical Design Tools",
    page_icon="🌈",
    layout="wide",
    initial_sidebar_state="expanded",
)

nav = st.navigation(
    pages=[
        st.Page(
            page="pages/home.py",
            title="Home",
            icon="🏠",
            default=True,
        ),
        st.Page(
            page="pages/echelle_spectrograph.py",
            title="Echelle Spectrograph",
            icon="🌈",
        ),
        st.Page(
            page="pages/low_resolution_spectrograph.py",
            title="Low Resolution Spectrograph",
            icon="🌈",
        ),
        st.Page(
            page="pages/atmospheric_dispersion_corrector.py",
            title="Atmospheric Dispersion Corrector",
            icon="🌠",
        ),
        st.Page(
            page="pages/fibre_loss_converter.py",
            title="Fibre Loss Converter",
            icon="🧵",
        ),
        st.Page(
            page="pages/glass_explorer.py",
            title="Glass Explorer",
            icon="🔍",
        ),
        st.Page(
            page="pages/surface_tolerances.py",
            title="Surface Tolerances",
            icon="🎯",
        ),
        st.Page(
            page="pages/telescope_designer.py",
            title="Telescope Designer",
            icon="🔭",
        ),
        st.Page(
            page="pages/cte_match.py",
            title="CTE Match",
            icon="🔩",
        ),
        st.Page(
            page="pages/solid_angle_and_etendue.py",
            title="Solid Angle & Étendue",
            icon="📐",
        ),
    ]
)

nav.run()
