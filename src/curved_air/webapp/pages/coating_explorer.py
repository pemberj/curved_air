#! uv run streamlit run

"""
curved_air

webapp

coatings

This app provides a frontend to explore and compare optical coatings that are available
from different manufacturers. So far, I include several (not yet all) of the coatings
available on off-the-shelf optics from Edmund Optics and Thorlabs.

TODO:
    - Add more coatings that I have data for (not proprietary stuff, make sure it's
    advertised by the manufacturer)
    - Allow the user to upload CSV files with coating curves
    - Maybe a companion app: thin-film coating designer/optimiser: with a maximum number
    of layers (to limit server utilisation), optimise a coating stack for reflectivity
    at given wavelengths. Export the coating curve in a format that can be uploaded to
    the coating explorer/comparison app.
"""

from __future__ import annotations

import streamlit as st
from astropy import units as u
from plotly import graph_objects as go

from curved_air.coatings import (
    EDMUND_OPTICS_AR_BROAD,
    EDMUND_OPTICS_AR_NARROW,
    EDMUND_OPTICS_HR_BROAD,
    EDMUND_OPTICS_HR_NARROW,
    THORLABS_AR_BROAD,
    THORLABS_AR_NARROW,
    THORLABS_HR_BROAD,
    THORLABS_HR_NARROW,
    Coating,
)
from curved_air.utils import st_quantity_input
from curved_air.webapp import badges, bmac

st.set_page_config(
    page_title="Coating Explorer",
    layout="wide",
)

st.header("Coating Explorer", anchor=False)

with st.expander("About this tool", expanded=False):
    f"""
    Use this tool to visualise and compare various optical coatings. A number of
    standard commercial coatings available from Edmund Optics and Thorlabs are included,
    or you can input your own data.

    All code is written by Jake Pember {badges}

    Please get in contact if you have any questions, suggestions, or find any bugs!

    You can also [Buy me a pint](https://buymeacoffee.com/pemberj) if you find this tool
    useful and want to support its development!
    """

    bmac()

selected_ar_broad_coatings: list[Coating] = []
selected_ar_narrow_coatings: list[Coating] = []
selected_hr_broad_coatings: list[Coating] = []
selected_hr_narrow_coatings: list[Coating] = []

tabs_col, plot_col = st.columns([2, 3])

with tabs_col:
    ar_tab, hr_tab = st.tabs(["AR Coatings", "HR Coatings"], on_change="rerun")

    if ar_tab.open:
        with ar_tab:
            st.subheader("Anti-Reflection (AR) Coatings", anchor=False)

            st.multiselect(
                "Select Broadband AR Coatings to Compare",
                options=EDMUND_OPTICS_AR_BROAD + THORLABS_AR_BROAD,
                format_func=lambda coating: f"{coating.name}",
                wrap=True,
                key="selected_ar_broad_coatings",
            )

            st.multiselect(
                "Select Narrowband AR Coatings to Compare",
                options=EDMUND_OPTICS_AR_NARROW + THORLABS_AR_NARROW,
                format_func=lambda coating: f"{coating.name}",
                wrap=True,
                key="selected_ar_narrow_coatings",
            )

    if hr_tab.open:
        with hr_tab:
            st.subheader("High-Reflectivity (HR) Coatings", anchor=False)

            st.multiselect(
                "Select Broadband HR Coatings to Compare",
                options=EDMUND_OPTICS_HR_BROAD + THORLABS_HR_BROAD,
                format_func=lambda coating: f"{coating.name}",
                wrap=True,
                key="selected_hr_broad_coatings",
            )

            st.multiselect(
                "Select Narrowband HR Coatings to Compare",
                options=EDMUND_OPTICS_HR_NARROW + THORLABS_HR_NARROW,
                format_func=lambda coating: f"{coating.name}",
                wrap=True,
                key="selected_hr_narrow_coatings",
            )

    with st.container(horizontal_alignment="center"):  # noqa: SIM117
        with st.container(width=500):
            st_quantity_input(
                "Minimum Wavelength",
                value=380.0,
                min_value=0.0,
                step=10.0,
                possible_units=["Å", "nm", "µm"],
                default_unit="nm",
                format="%.5g",
                key="min_wavelength",
            )

            st_quantity_input(
                "Maximum Wavelength",
                value=900.0,
                min_value=0.0,
                step=10.0,
                possible_units=["Å", "nm", "µm"],
                default_unit="nm",
                format="%.5g",
                key="max_wavelength",
            )

with plot_col:
    # Make a plotly plot
    fig = go.Figure()

    min_y_in_range: float = 1.0
    max_y_in_range: float = 0.0

    if ar_tab.open:
        selected_coatings: list[Coating] = st.session_state.get(
            "selected_ar_broad_coatings", []
        ) + st.session_state.get("selected_ar_narrow_coatings", [])
    elif hr_tab.open:
        selected_coatings: list[Coating] = st.session_state.get(
            "selected_hr_broad_coatings", []
        ) + st.session_state.get("selected_hr_narrow_coatings", [])

    for coating in selected_coatings:
        range_mask = (coating.wavelengths >= st.session_state["min_wavelength"]) & (
            coating.wavelengths <= st.session_state["max_wavelength"]
        )

        try:
            candidate_max_y: float = max(coating.reflectivity[range_mask])
        except ValueError:
            continue

        try:
            candidate_min_y: float = min(coating.reflectivity[range_mask])
        except ValueError:
            continue

        if candidate_min_y < min_y_in_range:
            min_y_in_range: float = candidate_min_y

        if candidate_max_y > max_y_in_range:
            max_y_in_range: float = candidate_max_y

        fig.add_trace(
            trace=go.Scatter(
                x=coating.wavelengths.to(u.nm),
                y=coating.reflectivity,
                mode="lines",
                hovertemplate=f"{coating.name.split('(')[0]}<br />"
                + "%{y:.4g}<extra></extra>",
                name=f"{coating.name}",
            )
        )

    fig.update_layout(
        title="",
        xaxis_title="Wavelength (nm)",
        xaxis_range=[
            st.session_state["min_wavelength"].to(u.nm).value,
            st.session_state["max_wavelength"].to(u.nm).value,
        ],
        yaxis_title="Reflectivity",
        yaxis_range=[max(min_y_in_range, 0), min(max_y_in_range * 1.1, 1.0)],
        template="plotly_dark",
        height=800,
        hovermode="x",
    )

    fig.update_xaxes(
        showspikes=True,
        spikecolor="white",
        spikesnap="cursor",
        spikemode="across",
    )

    st.plotly_chart(
        fig,
        width="stretch",
        on_select="rerun",
        selection_mode="points",
        key="ar_coating_plot",
    )
