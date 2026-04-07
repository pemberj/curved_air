#! uv run streamlit run

"""
Glass Explorer

Create a frontend for loading one or more existing catalogues (with the option to upload
your own), viewing the glasses on an Abbe diagram and others, filtering glasses by
various parameters, and saving (downloading) the filtered catalogue as an AGF for use
in Zemax (and elsewhere?).
"""


# TODO: Multi-select glass catalogues from those included in curved_air
# TODO: Upload custom glass catalogue in AGF format (just one is sufficient I guess)
# TODO: Plot glasses on interactive Abbe diagram
# TODO: Plot glasses for differential relative partial dispersion (dRPD) diagra
# TODO: Allow filtering glasses by:
#   - Transmission properties (e.g. cutoff wavelength)
#   - Abbe number
#   - Refractive index
#   - Relative partial dispersion
#   - Cost (if available)
#   - Availability (if available)
#   - Glass 'status' eg. preferred, not recommended, etc. (if available)
# TODO: Save the filtered catalogue as an AGF file, download from the interface

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from astropy import units as u
from matplotlib import pyplot as plt

from curved_air.glass import GLASS_CATALOGUES, Glass, GlassCatalogue
from curved_air.plotting import plot_style, wavelength_to_rgb, wl_to_hex

if TYPE_CHECKING:
    from pathlib import Path

    from curved_air.type_hinting import ArrayQuantity, ScalarQuantity

plt.style.use(plot_style)

st.set_page_config(
    page_title="Curved Air: Optical Design Calculators",
    page_icon="🌈",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.header("Glass Explorer")

with st.expander("About this tool", expanded=False):
    st.markdown(
        """
        This tool allows querying, inspection, and filtering of glass catalogues for the
        purpose of creating custom catalogues for use in Zemax projects or similar.

        All code is written by [Jake Pember](https://github.com/pemberj/)

        Please get in contact if you have any questions, suggestions, or find any bugs!
        """
    )

st.divider()

# Interface
with st.sidebar:
    working_catalogue: None | GlassCatalogue = None

    base_catalogues: list[str] = st.multiselect(
        "Select base catalogue(s)",
        options=list(GLASS_CATALOGUES.keys()),
        help="""Optionally choose one or more glass catalogues from which to start...""",
    )

    agf_paths: list[Path] = [GLASS_CATALOGUES[c] for c in base_catalogues]

    if base_catalogues:
        working_catalogue = GlassCatalogue(agf=agf_paths)

    uploaded: list[st.UploadedFile] = st.file_uploader(
        "Upload AGF file(s)",
        type="agf",
        accept_multiple_files=True,
        max_upload_size=10,  # Megabytes
    )

    if uploaded:
        ...

    with st.expander("Filters", expanded=False):
        st.toggle("Filter by glass name")
        st.toggle("Filter by transmission")
        st.toggle("Filter by glass status")
        st.toggle("Filter by refractive index")
        st.toggle("Filter by ???")

if working_catalogue:
    df: pd.DataFrame = pd.DataFrame({"Glass Name": working_catalogue.glass_list})
else:
    df = None

st.dataframe(df)
