import streamlit as st

st.title("Curved Air: Optical Design Calculators")

st.markdown(
    """
Welcome to the **Curved Air** optical design calculator. Use the sidebar to navigate between
the available tools:

| Calculator | Description |
|---|---|
| **Spectral Format** | Configure a full echelle spectrograph (grating + cross-disperser + detector) and visualise the échellogram on the detector. |
| **Atmospheric Dispersion Corrector** | Design a rotational atmospheric dispersion corrector (ADC) with two different glasses. |

---
*Powered by [curved_air](https://github.com/pemberj/curved_air) · Built with
[Streamlit](https://streamlit.io)*
"""
)
