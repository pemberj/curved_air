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

from io import StringIO
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from astropy import units as u
from matplotlib import pyplot as plt
from streamlit.components.v1 import html

from curved_air.glass import GLASS_CATALOGUES, AGFData, Glass, GlassCatalogue
from curved_air.plotting import plot_style, wavelength_to_rgb, wl_to_hex  # noqa: F401

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from curved_air.type_hinting import ArrayQuantity, ScalarQuantity  # noqa: F401

plt.style.use(plot_style)

wavelength_options: dict[str, ScalarQuantity | None] = {
    "G": 380.0 * u.nm,
    "F": 486.1 * u.nm,
    "d": 589.3 * u.nm,
    "C": 656.3 * u.nm,
    "Custom": None,
}

st.set_page_config(
    page_title="Curved Air: Optical Design Tools",
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

        You can start from one or more of the catalogues included in the curved_air
        repo, and/or upload your own AGF files.

        The catalogue can then be filtered by various parameters, and the filtered
        catalogue can be downloaded as an AGF file for use in your optical design
        projects.

        All code is written by Jake Pember: [Github](https://github.com/pemberj/) / [Email](mailto:jacob.pember@ikmail.com)

        Please get in contact if you have any questions, suggestions, or find any bugs!

        You can also [Buy me a pint](https://buymeacoffee.com/pemberj) if you find this tool useful and want to support its development!
        """
    )

    html(
        """
        <script type="text/javascript" src="https://cdnjs.buymeacoffee.com/1.0.0/button.prod.min.js" data-name="bmc-button" data-slug="pemberj" data-color="#FF5F5F" data-emoji="🍺" data-font="Cookie" data-text="Buy me a pint" data-outline-color="#000000" data-font-color="#ffffff"></script>
        """,
        height=70,
    )

st.divider()

# Interface
with st.sidebar:
    working_catalogue = GlassCatalogue()

    base_catalogues: list[str] = st.multiselect(
        "Select base catalogue(s)",
        options=list(GLASS_CATALOGUES.keys()),
        default=["Schott", "Ohara"],
        help="""Optionally choose one or more glass catalogues from which to start...""",
    )

    agf_paths: list[Path] = [GLASS_CATALOGUES[c] for c in base_catalogues]

    if base_catalogues:
        working_catalogue += GlassCatalogue(agf=agf_paths)

    uploaded_files: list[st.UploadedFile] = st.file_uploader(
        "Upload AGF file(s)",
        type="agf",
        accept_multiple_files=True,
        max_upload_size=10,  # Megabytes
    )

    if uploaded_files:
        working_catalogue += GlassCatalogue(
            agf=[
                AGFData(StringIO(f.getvalue().decode("utf-8")).read(), filename=f.name)
                for f in uploaded_files
            ]
        )

    if working_catalogue is not None:
        st.metric(
            "Glasses in working catalogue",
            f"{len(working_catalogue)}",
        )

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
                st.number_input(
                    "Minimum transmission",
                    min_value=0.0,
                    max_value=100.0,
                    value=95.0,
                    step=1.0,
                )
                / 100
            )

            working_catalogue = working_catalogue.filter(
                min_wl=T_wls[0], max_wl=T_wls[1], min_transmission=T_min
            )

            st.divider()

        st.toggle("Filter by glass status", key="filter_status")

        if st.session_state.get("filter_status", False):
            status_options = {g.status_description for g in working_catalogue}
            selected_statuses: list[str] = st.multiselect(
                "Select glass status descriptions to include",
                options=status_options,
                default=list(status_options),
            )

            working_catalogue = working_catalogue.filter(
                status_description=selected_statuses
            )

            st.divider()

        st.toggle("Filter by refractive index", key="filter_n")

        if st.session_state.get("filter_n", False):
            cols = st.columns(2)

            with cols[0]:
                n_wavelength: str | None = st.pills(
                    "Refractive index wavelength",
                    options=wavelength_options,
                    format_func=lambda x: f"{x}",
                    key="n_wavelength_pills",
                )

                if n_wavelength is None:
                    n_wavelength = "d"

                if n_wavelength != "Custom":
                    n_wavelength: ScalarQuantity = wavelength_options[n_wavelength]  # ty:ignore[invalid-assignment]
                    st.session_state["n_wavelength"] = n_wavelength.to(u.nm).value

            with cols[1]:
                if n_wavelength == "Custom":
                    n_wavelength: ScalarQuantity = (
                        st.number_input(
                            "Refractive index wavelength (nm)",
                            min_value=200.0,
                            max_value=2000.0,
                            value=st.session_state.get("n_wavelength", 589.3),
                            step=1.0,
                            key="n_wavelength_input",
                        )
                        * u.nm
                    )

            st.metric(
                "Selected wavelength",
                f"{n_wavelength:.1f}",
            )

            n_range: tuple[float, float] = st.slider(
                "Allowed refractive index range",
                min_value=1.0,
                max_value=2.5,
                value=(1.4, 1.8),
                step=0.01,
            )

            working_catalogue = working_catalogue.filter(
                min_n=n_range[0], max_n=n_range[1], n_wavelength=n_wavelength
            )

            st.divider()

        st.toggle("Filter by Abbe number", key="filter_V")

        if st.session_state.get("filter_V", False):
            V_range: tuple[float, float] = st.slider(
                "Allowed Abbe number range",
                min_value=10.0,
                max_value=100.0,
                value=(10.0, 100.0),
                step=1.0,
            )

            st.text(V_range)

            working_catalogue: GlassCatalogue = working_catalogue.filter(
                min_V_d=V_range[0], max_V_d=V_range[1]
            )

        st.divider()

        assert working_catalogue is not None
        st.metric("Glasses remaining after filtering", f"{len(working_catalogue)}")

        st.divider()

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
            "Catalogue": [g.manufacturer for g in working_catalogue],
            "Glass Name": working_catalogue.glass_list,
            "n_d": [g.n_d for g in working_catalogue],
            "V_d": [g.V_d for g in working_catalogue],
            "Status": [g.status_description for g in working_catalogue],
        },
    )
else:
    df = None


if working_catalogue:
    # st.text(working_catalogue.comment)

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

        with st.container(horizontal_alignment="center"):
            min_wl, max_wl = st.slider(
                "Plot wavelengths (nm)",
                min_value=200.0,
                max_value=2000.0,
                value=(
                    min(st.session_state.get("plot_wls", selected_glass.wl)) * 1000,
                    max(st.session_state.get("plot_wls", selected_glass.wl)) * 1000,
                ),
                step=5.0,
                width=500,
                key="min_max_wls",
            )

        plot_wls: NDArray = np.linspace(min_wl, max_wl, 100) * u.nm
        ref_index: NDArray = selected_glass.refractive_index(plot_wls)
        transmission: NDArray = selected_glass.transmission(plot_wls)

        cols = st.columns(2)

        with cols[0]:
            # Refractive index plot
            fig: go.Figure = go.Figure()
            try:
                fig.add_trace(
                    trace=go.Scatter(
                        x=plot_wls,
                        y=ref_index,
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
            except Exception:
                st.error(
                    """An error occurred while trying to compute the refractive index
                    for the selected glass. This may be due to missing or invalid
                    dispersion coefficients, or an unsupported dispersion formula."""
                )
                # st.exception(e)

        with cols[1]:
            # Transmission plot
            fig: go.Figure = go.Figure()

            try:
                fig.add_trace(
                    trace=go.Scatter(
                        x=plot_wls,
                        y=transmission,
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

            except Exception as e:
                st.error(
                    """An error occurred while trying to compute the transmission
                    for the selected glass. This may be due to missing or invalid
                    transmission data."""
                )
                st.exception(e)

        st.session_state["plot_wls"] = plot_wls.to(u.um).value
