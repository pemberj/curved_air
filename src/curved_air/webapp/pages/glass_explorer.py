#! uv run streamlit run

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
from curved_air.plotting import plot_style, wavelength_to_rgb, wl_to_hex  # noqa: F401

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from curved_air.type_hinting import ArrayQuantity, ScalarQuantity  # noqa: F401

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
        if not working_catalogue:
            st.info(
                """Please select or upload one or more base catalogues to enable
                filtering options."""
            )
            st.stop()
        st.toggle("Filter by transmission", key="filter_transmission")

        if st.session_state.get("filter_transmission", False):
            T_wls: ArrayQuantity = (
                st.slider(
                    "Transmission wavelengths (nm)",
                    min_value=200.0,
                    max_value=2000.0,
                    value=(380.0, 950.0),
                    step=5.0,
                )
                * u.nm
            )

            T_min: float = (
                st.slider(
                    "Minimum transmission (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=90.0,
                    step=0.25,
                )
                / 100
            )

            working_catalogue = working_catalogue.filter(
                min_wl=T_wls[0], max_wl=T_wls[1], min_transmission=T_min
            )

        st.toggle("Filter by glass status", key="filter_status")
        st.toggle("Filter by refractive index", key="filter_n")
        st.toggle("Filter by abbe number", key="filter_V")

        try:
            st.download_button(
                "Download filtered catalogue as AGF",
                data=working_catalogue.to_AGF_string(),
                file_name="filtered_catalogue.agf",
                mime="text/plain",
            )
        except Exception as e:
            st.error(
                """An error occurred while trying to generate the AGF file. Please
                check the filters and try again."""
            )
            st.exception(e)

if working_catalogue:
    df: pd.DataFrame = pd.DataFrame(
        data={
            "Glass Name": working_catalogue.glass_list,
            "n_d": [g.n_d for g in working_catalogue],
            "V_d": [g.V_d for g in working_catalogue],
        },
    )
else:
    df = None


if working_catalogue:
    selected = st.dataframe(
        df,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-cell",
    )

    if selected.selection and selected.selection.cells:
        glass_index = selected.selection.cells[0][0]
        glass_name = df.iloc[glass_index]["Glass Name"]
        selected_glass: Glass = working_catalogue[glass_name]

        st.subheader(f"Glass: {selected_glass.name}")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Refractive Index (n$_\\textup{d}$)", f"{selected_glass.n_d:.6f}")
        with col2:
            st.metric("Abbe Number (V$_\\textup{d}$)", f"{selected_glass.V_d:.2f}")
        with col3:
            st.metric("Glass status", f"{selected_glass.status_description}")

        min_wl, max_wl = st.slider(
            "Plot wavelengths (nm)",
            min_value=200.0,
            max_value=2000.0,
            value=(
                min(st.session_state.get("plot_wls", selected_glass.wl)) * 1000,
                max(st.session_state.get("plot_wls", selected_glass.wl)) * 1000,
            ),
            step=5.0,
            key="min_max_wls",
        )

        plot_wls: NDArray = np.linspace(min_wl, max_wl, 100) * u.nm

        cols = st.columns(2)

        with cols[0]:
            # Refractive index plot
            fig: go.Figure = go.Figure()
            fig.add_trace(
                trace=go.Scatter(
                    x=plot_wls,
                    y=selected_glass.refractive_index(plot_wls),
                    mode="lines",
                )
            )
            fig.update_layout(
                title=f"{glass_name} Refractive Index",
                xaxis_title="Wavelength (nm)",
                yaxis_title="Refractive Index",
                template="plotly_dark",
            )

            st.plotly_chart(fig)

        with cols[1]:
            # Transmission plot
            fig: go.Figure = go.Figure()
            fig.add_trace(
                trace=go.Scatter(
                    x=plot_wls,
                    y=selected_glass.transmission(plot_wls),
                    mode="lines",
                )
            )
            fig.update_layout(
                title=f"{glass_name} Transmission",
                xaxis_title="Wavelength (nm)",
                yaxis_title="Transmission",
                template="plotly_dark",
            )

            st.plotly_chart(fig)

        st.session_state["plot_wls"] = plot_wls.to(u.um).value
