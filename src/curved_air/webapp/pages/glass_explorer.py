#! uv run streamlit run

"""
Glass Explorer

A frontend for loading one or more existing glass catalogues in AGF format (with the
option to upload your own), viewing the glasses on an Abbe diagram and others, filtering
glasses by various parameters, and saving (downloading) the filtered catalogue as an AGF
for use in Zemax (and elsewhere?).
"""

# TODO: Plot differential relative partial dispersion (dRPD) diagram

from __future__ import annotations

from io import StringIO
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from astropy import units as u
from streamlit.components.v1 import html

from curved_air.glass import GLASS_CATALOGUES, AGFData, Glass, GlassCatalogue
from curved_air.plotting import plot_style, wavelength_to_rgb, wl_to_hex  # noqa: F401

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from curved_air.type_hinting import ArrayQuantity, ScalarQuantity  # noqa: F401

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
                AGFData(
                    agf_string=StringIO(f.getvalue().decode("utf-8")).read(),
                    filename=f.name,
                )
                for f in uploaded_files
            ]
        )

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

        else:
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

                working_catalogue: GlassCatalogue = working_catalogue.filter(
                    min_wl=T_wls[0],
                    max_wl=T_wls[1],
                    min_transmission=T_min,
                    debug_level=1,
                )

                st.divider()

            st.toggle("Filter by glass status", key="filter_status")

            if st.session_state.get("filter_status", False):
                status_options: set[str] = {
                    g.status_description for g in working_catalogue
                }
                selected_statuses: list[str] = st.multiselect(
                    "Select glass status descriptions to include",
                    options=status_options,
                    default=list(status_options),
                )

                working_catalogue: GlassCatalogue = working_catalogue.filter(
                    min_wl=None,
                    max_wl=None,
                    status_description=selected_statuses,
                    debug_level=1,
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
                    min_wl=None,
                    max_wl=None,
                    min_n=n_range[0],
                    max_n=n_range[1],
                    n_wavelength=n_wavelength,
                    debug_level=1,
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
                    min_wl=None,
                    max_wl=None,
                    min_V_d=V_range[0],
                    max_V_d=V_range[1],
                    debug_level=1,
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

if not working_catalogue:
    with st.container(horizontal_alignment="center"):
        st.info(
            """Please select or upload one or more base catalogues to view and filter
            the catalogue contents."""
        )
    st.stop()

df: pd.DataFrame = pd.DataFrame(
    data={
        "Catalogue": [g.manufacturer for g in working_catalogue],
        "Glass Name": working_catalogue.glass_list,
        "n_d": [g.n_d for g in working_catalogue],
        "V_d": [g.V_d for g in working_catalogue],
        "Status": [g.status_description for g in working_catalogue],
    },
)

# st.text(working_catalogue.comment)

selected_glass_name: str | None = st.session_state.get("selected_glass_name")

cols = st.columns(2)

with cols[0]:
    st.toggle("Glass table", value=True, key="display_table")
    if st.session_state.get("display_table", False):
        event = st.dataframe(
            df,
            hide_index=True,
            on_select="rerun",
            selection_mode="multi-row",
            height=600,
        )

        selected: list = event.selection.rows if event.selection else []
        selected_glasses: list[Glass] = [
            working_catalogue[f"{df.iloc[i]['Glass Name']}"] for i in selected
        ]
    else:
        selected_glasses: list[Glass] = []

with cols[1]:
    st.toggle("Abbe diagram", value=True, key="display_abbe")

    if st.session_state.get("display_abbe", False):
        # Abbe diagram
        fig = go.Figure()

        unique_catalogues: set[str] = {g.manufacturer for g in working_catalogue}
        # Define default colours for each catalogue, so they don't change depending on what is selected
        # Use a default colour palette cycler
        catalogue_colors: dict[str, str] = {
            catalogue: px.colors.qualitative.Dark24[i]
            for i, catalogue in enumerate(list(GLASS_CATALOGUES.keys()))
        }

        # add selected point(s) if applicable
        for selected_glass in selected_glasses:
            fig.add_trace(
                trace=go.Scatter(
                    x=[selected_glass.V_d],
                    y=[selected_glass.n_d],
                    mode="markers",
                    marker={
                        "size": 16,
                        "symbol": "cross-thin",
                        "line": {"width": 1.5, "color": "white"},
                    },
                    text=[f"{selected_glass.manufacturer} {selected_glass.name}"],
                    hovertemplate="%{text}<br>V_d: %{x:.2f}<br>n_d: %{y:.6f}<extra></extra>",
                    showlegend=False,
                )
            )

        for catalogue in unique_catalogues:
            catalogue_glasses: list[Glass] = [
                g for g in working_catalogue if g.manufacturer == catalogue
            ]
            fig.add_trace(
                trace=go.Scatter(
                    x=[g.V_d for g in catalogue_glasses],
                    y=[g.n_d for g in catalogue_glasses],
                    mode="markers",
                    text=[f"{g.manufacturer} {g.name}" for g in catalogue_glasses],
                    customdata=[g.name for g in catalogue_glasses],
                    hovertemplate="%{text}<br>V_d: %{x:.2f}<br>n_d: %{y:.6f}<extra></extra>",
                    name=catalogue,
                    marker={"color": catalogue_colors[catalogue]},
                )
            )

        fig.update_layout(
            title="Abbe Diagram of Glasses in Working Catalogue",
            xaxis_title="Abbe Number (V_d)",
            yaxis_title="Refractive Index (n_d)",
            template="plotly_dark",
            height=600,
            clickmode="event+select",
        )

        fig.update_xaxes(autorange="reversed")

        abbe_selected = st.plotly_chart(
            fig,
            width="stretch",
            on_select="rerun",
            selection_mode="points",
            key="abbe_diagram",
        )

st.divider()

if not selected_glasses:
    with st.container(horizontal_alignment="center"):
        st.info(
            """You can select a glass from the table above to view its properties and
            dispersion/transmission curves.""",
            width=600,
        )
    st.stop()

cols = st.columns(2, gap="large")

with cols[0], st.container(horizontal_alignment="right"):
    min_wl: float = st.number_input(
        "Minimum wavelength (nm)",
        min_value=200.0,
        max_value=2000.0,
        value=st.session_state.get("min_max_wls", (370.0, 500.0))[0],
        step=5.0,
        key="min_wl_input",
        width=200,
    )

with cols[1], st.container(horizontal_alignment="left"):
    max_wl: float = st.number_input(
        "Maximum wavelength (nm)",
        min_value=200.0,
        max_value=2000.0,
        value=st.session_state.get("min_max_wls", (370.0, 500.0))[1],
        step=5.0,
        key="max_wl_input",
        width=200,
    )

st.session_state["min_max_wls"] = (min_wl, max_wl)

plot_wls: ArrayQuantity = np.linspace(min_wl, max_wl, 100) * u.nm


cols = st.columns(2)

with cols[0]:
    # Refractive index plot
    fig: go.Figure = go.Figure()

    for selected_glass in selected_glasses:
        ref_index: NDArray = selected_glass.refractive_index(plot_wls)
        try:
            fig.add_trace(
                trace=go.Scatter(
                    x=plot_wls,
                    y=ref_index,
                    mode="lines",
                    name=f"{selected_glass.manufacturer} {selected_glass.name}",
                )
            )
        except Exception:
            st.error(
                """An error occurred while trying to compute the refractive index
                for the selected glass. This may be due to missing or invalid
                dispersion coefficients, or an unsupported dispersion formula."""
            )
            # st.exception(e)

    fig.update_layout(
        title="Refractive Indices",
        xaxis_title="Wavelength (nm)",
        yaxis_title="Refractive Index",
        template="plotly_dark",
        height=600,
    )

    st.plotly_chart(fig)

with cols[1]:
    # Transmission plot
    fig: go.Figure = go.Figure()

    for selected_glass in selected_glasses:
        transmission: NDArray = selected_glass.transmission(plot_wls)  # ty:ignore[invalid-assignment]
        try:
            fig.add_trace(
                trace=go.Scatter(
                    x=plot_wls,
                    y=transmission,
                    mode="lines",
                    name=f"{selected_glass.manufacturer} {selected_glass.name}",
                )
            )
        except Exception as e:
            st.error(
                """An error occurred while trying to compute the transmission for
                the selected glass. This may be due to missing or invalid
                transmission data."""
            )
            st.exception(e)

    fig.update_layout(
        title="Transmission",
        xaxis_title="Wavelength (nm)",
        yaxis_title="Transmission",
        template="plotly_dark",
        height=600,
    )

    st.plotly_chart(fig)


st.session_state["plot_wls"] = plot_wls.to(u.um).value
