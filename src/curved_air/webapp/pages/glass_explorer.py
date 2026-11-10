#! uv run streamlit run

"""
Glass Explorer

A frontend for loading one or more existing glass catalogues in AGF format (with the
option to upload your own), viewing the glasses on an Abbe diagram and others,
filtering glasses by various parameters, and saving (downloading) the filtered
catalogue as an AGF for use in Zemax (and elsewhere?).

TODO: Can I show a 3D Abbe diagram? Third axis is wavelength, each glass becomes a 3D
line rather than a single point
"""

from __future__ import annotations

from io import StringIO
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from astropy import units as u
from astropy.units import Quantity

from curved_air.glass import GLASS_CATALOGUES, AGFData, Glass, GlassCatalogue
from curved_air.plotting import plot_style, wavelength_to_rgb, wl_to_hex  # noqa: F401
from curved_air.utils import st_quantity_input
from curved_air.webapp import badges, bmac

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from curved_air.type_hinting import ArrayQuantity, ScalarQuantity

wavelength_options: dict[str, ScalarQuantity | None] = {
    "i": 365.01 * u.nm,
    "h": 404.66 * u.nm,
    "g": 435.84 * u.nm,
    "F'": 479.99 * u.nm,
    "F": 486.13 * u.nm,
    "e": 546.07 * u.nm,
    "D": 587.56 * u.nm,
    "d": 589.30 * u.nm,
    "C'": 643.85 * u.nm,
    "C": 656.27 * u.nm,
    "r": 706.52 * u.nm,
    "A'": 768.20 * u.nm,
    "s": 852.11 * u.nm,
    "t": 1013.98 * u.nm,
    "Custom": None,
}


st.header("Glass Explorer", anchor=False)

with st.expander("About this tool", expanded=False):
    f"""
    This tool allows querying, inspection, and filtering of glass catalogues for the
    purpose of creating custom catalogues for use in Zemax projects or similar.

    You can start from one or more of the catalogues included in the curved_air repo,
    and/or upload your own AGF files.

    The catalogue can then be filtered by various parameters, and the filtered catalogue
    can be downloaded as an AGF file for use in your optical design projects.

    All code is written by Jake Pember {badges}

    Please get in contact if you have any questions, suggestions, or find any bugs!

    You can also [Buy me a pint](https://buymeacoffee.com/pemberj) if you find this tool
    useful and want to support its development!
    """

    bmac()

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
        commented = False
        for c in base_catalogues:
            _catalogue = GlassCatalogue(agf=GLASS_CATALOGUES[c])

            if _catalogue.comment:
                commented = True
                st.text(f"{c} comment: {_catalogue.comment}")

            working_catalogue += _catalogue

        if commented:
            st.divider()

    uploaded_files: list[st.UploadedFile] = st.file_uploader(
        "Upload AGF file(s)",
        type="agf",
        accept_multiple_files=True,
        max_upload_size=10,  # Megabytes
    )

    if uploaded_files:
        commented = False

        for f in uploaded_files:
            _catalogue = GlassCatalogue(
                agf=[
                    AGFData(
                        agf_string=StringIO(f.getvalue().decode("utf-8")).read(),
                        filename=f.name,
                    )
                ]
            )

            if _catalogue.comment:
                commented = True
                st.text(f"{f.name} comment: {_catalogue.comment}")

            working_catalogue += _catalogue

        if commented:
            st.divider()

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
                T_min_wl: ScalarQuantity = st_quantity_input(
                    "Min λ",
                    min_value=0.0,
                    value=380.0,
                    step=1.0,
                    possible_units=["nm", "µm"],
                    default_unit="nm",
                    key="T_min_wl",
                ).to(u.nm)

                T_max_wl: ScalarQuantity = st_quantity_input(
                    "Max λ",
                    min_value=0.0,
                    value=950.0,
                    step=1.0,
                    possible_units=["nm", "µm"],
                    default_unit="nm",
                    key="T_max_wl",
                ).to(u.nm)

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
                    min_wl=T_min_wl,
                    max_wl=T_max_wl,
                    min_transmission=T_min,
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
                )

                st.divider()

            st.toggle("Filter by refractive index", key="filter_n")

            if st.session_state.get("filter_n", False):
                st.toggle("Select wavelength)", key="select_n_filter_wavelength")

                if st.session_state.get("select_n_filter_wavelength", False):
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

                    if n_wavelength == "Custom":
                        n_wavelength: ScalarQuantity = st_quantity_input(
                            "Refractive index wavelength",
                            min_value=0.0,
                            value=st.session_state.get("n_wavelength", 589.3),
                            step=1.0,
                            possible_units=["nm", "µm"],
                            default_unit="nm",
                            key="n_wavelength_input",
                        )

                    st.metric(
                        "Selected wavelength",
                        f"{n_wavelength:.1f}",
                    )
                else:
                    n_wavelength = "d"

                min_n = float(min([g.n(n_wavelength) for g in working_catalogue]))
                max_n = float(max([g.n(n_wavelength) for g in working_catalogue]))

                n_range: tuple[float, float] = st.slider(
                    "Allowed refractive index range",
                    min_value=min_n,
                    max_value=max_n,
                    value=(min_n, max_n),
                    step=0.01,
                )

                working_catalogue = working_catalogue.filter(
                    min_wl=None,
                    max_wl=None,
                    min_n=n_range[0],
                    max_n=n_range[1],
                    n_wavelength=n_wavelength,
                )

                st.divider()

            st.toggle("Filter by Abbe number", key="filter_V")

            if st.session_state.get("filter_V", False):
                min_V = float(min([g.V_d for g in working_catalogue]))
                max_V = float(max([g.V_d for g in working_catalogue]))

                V_range: tuple[float, float] = st.slider(
                    "Allowed Abbe number range",
                    min_value=min_V,
                    max_value=max_V,
                    value=(min_V, max_V),
                    step=1.0,
                )

                st.text(V_range)

                working_catalogue: GlassCatalogue = working_catalogue.filter(
                    min_wl=None,
                    max_wl=None,
                    min_V_d=V_range[0],
                    max_V_d=V_range[1],
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

cols = st.columns(2)

with cols[0]:
    st.toggle("Select wavelength", key="select_nV_wavelength")

if st.session_state.get("select_nV_wavelength", False):
    with cols[1]:
        st.toggle(
            "Use differential Abbe number (dV)", key="use_differential_Abbe_number"
        )

if st.session_state.get("select_nV_wavelength", False):
    cols = st.columns(4, gap="xsmall")

    with cols[0]:
        nV_plot_wavelength: str | None = st.pills(
            "Refractive index / Abbe number wavelength",
            options=wavelength_options,
            format_func=lambda x: f"{x}",
            key="nV_plot_wavelength_pills",
        )

        if nV_plot_wavelength is None:
            nV_plot_wavelength = "d"

        if nV_plot_wavelength != "Custom":
            nV_plot_wavelength: ScalarQuantity = wavelength_options[nV_plot_wavelength]  # ty:ignore[invalid-assignment]
            st.session_state["nV_plot_wavelength"] = nV_plot_wavelength.to(u.nm).value

    if nV_plot_wavelength == "Custom":
        # I didn't replace this with a st_quantity_input because it's not using columns
        # 0 and 1
        with cols[1]:
            nV_plot_wavelength_value: float = st.number_input(
                "Custom wavelength",
                min_value=0.1,
                max_value=2_000.0,
                value=st.session_state.get("nV_plot_wavelength", 589.3),
                step=1.0,
                key="nV_plot_wavelength_input",
            )

        with cols[2]:
            nV_plot_wavelength_unit: u.Unit = u.Unit(
                st.segmented_control(
                    "Unit",
                    options=["nm", "um"],
                    default="nm",
                    key="nV_plot_wavelength_unit",
                )
            )

        nV_plot_wavelength: ScalarQuantity = (
            nV_plot_wavelength_value * nV_plot_wavelength_unit
        )

    with cols[3]:
        st.metric(
            "Selected wavelength",
            f"{nV_plot_wavelength:.1f}",
        )

    if not st.session_state.get("use_differential_Abbe_number", False):
        V_string = "V_d"

    if st.session_state.get("nV_plot_wavelength_pills") == "Custom":
        n_string = "n"
        if st.session_state.get("use_differential_Abbe_number", False):
            V_string = "dV"
    elif st.session_state.get("nV_plot_wavelength_pills") is None:
        n_string = "n_d"
        if st.session_state.get("use_differential_Abbe_number", False):
            V_string = "dV_d"
    else:
        n_string = f"n_{st.session_state.get('nV_plot_wavelength_pills', 'd')}"
        if st.session_state.get("use_differential_Abbe_number", False):
            V_string = f"dV_{st.session_state.get('nV_plot_wavelength_pills', 'd')}"

else:
    n_string = "n_d"
    V_string = "V_d"
    nV_plot_wavelength: ScalarQuantity = wavelength_options["d"]

assert isinstance(nV_plot_wavelength, Quantity)

df: pd.DataFrame = pd.DataFrame(
    data={
        "Catalogue": [g.manufacturer for g in working_catalogue],
        "Glass Name": working_catalogue.glass_list,
        n_string: [g.n(nV_plot_wavelength) for g in working_catalogue],
        V_string: [
            g.dV(nV_plot_wavelength)
            if st.session_state.get("use_differential_Abbe_number", False)
            else g.V_d
            for g in working_catalogue
        ],
        "Relative Cost": [
            g.relative_cost if g.relative_cost != -1 else None
            for g in working_catalogue
        ],
        "Status": [g.status_description for g in working_catalogue],
    },
)

cols = st.columns(2)

if "selected_glasses" not in st.session_state:
    st.session_state["selected_glasses"]: list[Glass] = []

st.session_state["selection_state"] = {
    "selection": {
        "rows": [
            df.index.get_loc(df[df["Glass Name"] == g.name].index[0])
            for g in st.session_state["selected_glasses"]
        ],
        "columns": [],
        "cells": [],
    }
}


with cols[0]:
    event = st.dataframe(
        df,
        hide_index=True,
        on_select="rerun",
        selection_mode="multi-row",
        # selection_default=st.session_state.get(
        #     "selection_state",
        #     {"selection": {}},
        # ),
        height=600,
    )

    st.session_state["selected_indices"]: list = (
        event.selection.rows if event.selection else []
    )

    st.session_state["selected_glasses"]: list[Glass] = [
        working_catalogue[f"{df.iloc[i]['Glass Name']}"]
        for i in st.session_state["selected_indices"]
    ]

with cols[1]:
    # Abbe diagram
    fig = go.Figure()

    unique_catalogues: set[str] = {g.manufacturer for g in working_catalogue}
    # Define default colours for each catalogue, so they don't change depending on what is selected
    # Use a default colour palette cycler
    catalogue_colors: dict[str, str] = {
        catalogue: px.colors.qualitative.Dark24[i]
        for i, catalogue in enumerate(
            list(GLASS_CATALOGUES.keys())
            + [m for m in df["Catalogue"].unique() if m not in GLASS_CATALOGUES]
        )  # TODO: add uploaded catalogues to this list and color dict
    }

    for catalogue in unique_catalogues:
        catalogue_glasses: list[Glass] = [
            g for g in working_catalogue if g.manufacturer == catalogue
        ]
        fig.add_trace(
            trace=go.Scatter(
                x=[
                    g.dV(nV_plot_wavelength)
                    if st.session_state.get("use_differential_Abbe_number", False)
                    else g.V_d
                    for g in catalogue_glasses
                ],
                y=[g.n(nV_plot_wavelength) for g in catalogue_glasses],
                mode="markers",
                text=[f"{g.manufacturer} {g.name}" for g in catalogue_glasses],
                customdata=[g.name for g in catalogue_glasses],
                hovertemplate=f"%{{text}}<br>V_d: %{{x:.2f}}<br>{n_string}: %{{y:.6f}}<extra></extra>",
                name=catalogue,
                marker={"color": catalogue_colors[catalogue]},
            )
        )

    # add selected point(s) if applicable
    fig.add_trace(
        trace=go.Scatter(
            x=[
                g.dV(nV_plot_wavelength)
                if st.session_state.get("use_differential_Abbe_number", False)
                else g.V_d
                for g in st.session_state["selected_glasses"]
            ],
            y=[g.n(nV_plot_wavelength) for g in st.session_state["selected_glasses"]],
            mode="markers",
            text=[
                f"{g.manufacturer} {g.name}"
                for g in st.session_state["selected_glasses"]
            ],
            customdata=[g.name for g in st.session_state["selected_glasses"]],
            hovertemplate=f"%{{text}}<br>V_d: %{{x:.2f}}<br>{n_string}: %{{y:.6f}}<extra></extra>",
            marker={
                "size": 16,
                "symbol": "cross-thin",
                "line": {"width": 1.5, "color": "white"},
            },
            name="Selected",
        )
    )

    # Add box showing nearby glass region
    if len(st.session_state["selected_glasses"]) == 1 and st.session_state.get(
        "compare_equivalent_glasses", False
    ):
        delta_n: float = st.session_state.get("delta_n", 0.005)
        delta_V: float = st.session_state.get("delta_V", 0.5)

        single_selected_glass = st.session_state["selected_glasses"][0]

        fig.add_shape(
            type="rect",
            x0=single_selected_glass.V_d - delta_V,
            x1=single_selected_glass.V_d + delta_V,
            y0=single_selected_glass.n(nV_plot_wavelength) - delta_n,
            y1=single_selected_glass.n(nV_plot_wavelength) + delta_n,
            line={"color": "white", "width": 1, "dash": "dot"},
            fillcolor="rgba(255, 255, 255, 0.1)",
            layer="below",
        )

    fig.update_layout(
        title="Abbe Diagram of Glasses in Working Catalogue",
        xaxis_title=f"Abbe Number ({V_string})",
        yaxis_title=f"Refractive Index ({n_string})",
        template="plotly_dark",
        height=600,
        clickmode="event+select",
    )

    fig.update_xaxes(autorange="reversed")

    st.plotly_chart(
        fig,
        width="stretch",
        on_select="rerun",
        selection_mode="points",
        key="abbe_diagram",
    )


st.divider()

if not st.session_state["selected_glasses"]:
    with st.container(horizontal_alignment="center"):
        st.info(
            """You can select one or more glasses from the table above to view their
            properties and dispersion/transmission curves.""",
            width=600,
        )
    st.stop()

single_selected_glass = (
    st.session_state["selected_glasses"][0]
    if len(st.session_state["selected_glasses"]) == 1
    else None
)

if single_selected_glass:
    st.subheader(f"{single_selected_glass.manufacturer} {single_selected_glass.name}")

    cols = st.columns(4)

    with cols[0]:
        st.metric("$n_d$", f"{single_selected_glass.n_d:.4g}")
    with cols[1]:
        st.metric("$V_d$", f"{single_selected_glass.V_d:.2f}")
    with cols[2]:
        st.metric("Status", f"{single_selected_glass.status_description}")

    with cols[3], st.container(width=400, border=True):
        st.toggle(
            "Compare possible equivalent glasses",
            value=True,
            key="compare_equivalent_glasses",
        )

        if st.session_state.get("compare_equivalent_glasses", False):
            subcols = st.columns(2)

            subcols[0].number_input(
                "delta n",
                value=0.005,
                step=0.001,
                min_value=0.0,
                format="%.4g",
                key="delta_n",
            )

            subcols[1].number_input(
                "delta V",
                value=0.5,
                min_value=0.0,
                step=0.1,
                format="%.2g",
                key="delta_V",
            )

            possible_equivalents: list[Glass] = working_catalogue.find_equivalents(
                single_selected_glass,
                delta_n=st.session_state.get("delta_n", 0.005),
                delta_V=st.session_state.get("delta_V", 0.5),
            )

            st.multiselect(
                "Optionally select one or more to compare",
                options=possible_equivalents,
                format_func=lambda g: f"{g.manufacturer} {g.name}",
                key="selected_equivalent_glasses",
            )

        else:
            st.session_state["selected_equivalent_glasses"]: list[Glass] = []

cols = st.columns(2, gap="large")

with cols[0], st.container(horizontal_alignment="right"):  # noqa: SIM117
    with st.container(width=400):
        min_wl: ScalarQuantity = st_quantity_input(
            "Min λ",
            min_value=0.0,
            value=380.0,
            step=1.0,
            possible_units=["nm", "µm"],
            default_unit="nm",
            key="min_wl_input",
        ).to(u.nm)

with cols[1], st.container(width=400, horizontal_alignment="left"):  # noqa: SIM117
    with st.container(width=400):
        max_wl: ScalarQuantity = st_quantity_input(
            "Max λ",
            min_value=0.0,
            value=950.0,
            step=1.0,
            possible_units=["nm", "µm"],
            default_unit="nm",
            key="max_wl_input",
        ).to(u.nm)

st.session_state["min_max_wls"] = (min_wl, max_wl)

plot_wls: ArrayQuantity = np.linspace(min_wl.value, max_wl.value, 100) * u.nm


cols = st.columns(2)

with cols[0]:
    # Refractive index plot
    fig: go.Figure = go.Figure()

    for g in st.session_state["selected_glasses"] + st.session_state.get(
        "selected_equivalent_glasses", []
    ):
        ref_index: NDArray = g.refractive_index(plot_wls)
        try:
            fig.add_trace(
                trace=go.Scatter(
                    x=plot_wls,
                    y=ref_index,
                    mode="lines",
                    name=f"{g.manufacturer} {g.name}",
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
        title="Refractive Index",
        xaxis_title="Wavelength (nm)",
        yaxis_title="Refractive Index",
        template="plotly_dark",
        height=600,
    )

    st.plotly_chart(fig)

with cols[1]:
    # Transmission plot
    fig: go.Figure = go.Figure()

    for g in st.session_state["selected_glasses"] + st.session_state.get(
        "selected_equivalent_glasses", []
    ):
        transmission: NDArray = g.transmission(plot_wls)
        try:
            fig.add_trace(
                trace=go.Scatter(
                    x=plot_wls,
                    y=transmission,
                    mode="lines",
                    name=f"{g.manufacturer} {g.name}",
                )
            )
        except Exception as e:
            st.error(
                """An error occurred while trying to compute the transmission for the
                selected glass. This may be due to missing or invalid transmission
                data."""
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


cols = st.columns(2)

with cols[0]:
    d_lambda: ScalarQuantity = (
        st.number_input(
            "Wavelength step for dV/dRPD calculation (nm)",
            min_value=0.1,
            max_value=100.0,
            value=10.0,
            step=0.1,
            key="d_lambda",
            help="""The wavelength step (in nm) used for calculating the numerical
            derivatives for dV and dRPD calculations. A smaller step will give a more
            accurate result but may be noisier, while a larger step will be smoother but
            perhaps less accurate.""",
        )
        * u.nm
    )

    differential_plot_wls: NDArray = (
        np.arange(
            min(plot_wls.to(u.nm).value),
            max(plot_wls.to(u.nm).value),
            d_lambda.to(u.nm).value,
        )
        * u.nm
    )

    fig: go.Figure = go.Figure()

    for g in st.session_state["selected_glasses"] + st.session_state.get(
        "selected_equivalent_glasses", []
    ):
        try:
            fig.add_trace(
                trace=go.Scatter(
                    x=plot_wls,
                    y=g.dV(differential_plot_wls),
                    mode="lines",
                    name=f"{g.manufacturer} {g.name}",
                )
            )
        except Exception as e:
            st.exception(e)

    fig.update_layout(
        title="Differential Abbe Number (dV)",
        xaxis_title="Wavelength (nm)",
        yaxis_title="dV",
        template="plotly_dark",
        height=600,
    )

    st.plotly_chart(fig)

with cols[1]:
    reference_glass: Glass = st.selectbox(
        "Reference glass for dRPD calculation",
        options=(
            st.session_state["selected_glasses"]
            + st.session_state.get("selected_equivalent_glasses", [])
        ),
        index=0,
        format_func=lambda g: f"{g.manufacturer} {g.name}",
        key="dRPD_reference_glass",
    )

    fig: go.Figure = go.Figure()

    dRPD_ref: NDArray = reference_glass.dRPD(differential_plot_wls)

    for g in st.session_state["selected_glasses"] + st.session_state.get(
        "selected_equivalent_glasses", []
    ):
        try:
            fig.add_trace(
                trace=go.Scatter(
                    x=plot_wls,
                    y=g.dRPD(differential_plot_wls) - dRPD_ref,
                    mode="lines",
                    name=f"{g.manufacturer} {g.name}",
                )
            )
        except Exception as e:
            st.exception(e)

    fig.update_layout(
        title="Differential Relative Partial Dispersion (dRPD)",
        xaxis_title="Wavelength (nm)",
        yaxis_title="dRPD - dRPD_ref",
        template="plotly_dark",
        height=600,
    )

    st.plotly_chart(fig)

st.html("""For more information on the dV and dRPD parameters, how they are calculated,
and how to use them, see this article by Damien Jones of Prime Optics:
<a href=https://www.primeoptics.com.au/Reference/dRPD_description.pdf>https://www.primeoptics.com.au/Reference/dRPD_description.pdf</a>""")


st.session_state["plot_wls"] = plot_wls.to(u.um).value
