#!uv run streamlit run

"""
curved_air web interface

Main entry point for the Streamlit-based calculator web app.
Run with:  uv run streamlit run src/curved_air/webapp/app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Curved Air: Optical Design Tools",
    page_icon="🌈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.navigation(
    pages=[
        st.Page(
            page="pages/home.py",
            title="Home",
            icon="🏠",
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
    ]
).run()
