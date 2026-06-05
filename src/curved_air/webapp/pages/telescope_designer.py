#! uv run streamlit run

from __future__ import annotations

from typing import TYPE_CHECKING

# import matplotlib.pyplot as plt
# import numpy as np
import pandas as pd
import streamlit as st

from curved_air.telescopes import (
    CONIC_NOTES,
    DESIGNS,
    Mirror,
    TelescopeDesign,
)
from curved_air.utils import st_quantity_input

if TYPE_CHECKING:
    from curved_air.type_hinting import ScalarQuantity


def render_table(title: str, rows: list[tuple[str, str]]) -> None:
    st.markdown(f"**{title}**")
    st.table(
        pd.DataFrame([{"Parameter": label, "Value": value} for label, value in rows]),
        hide_index=True,
        hide_header=True,
    )


st.set_page_config(
    page_title="Curved Air: Optical Design Tools",
    page_icon="🌈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.header("Telescope Designer", anchor=False)

with st.expander("About this tool", expanded=False):
    st.markdown(
        """
        This tool calculates a two-mirror telescope prescription from the primary diameter, system focal length, primary mirror f/ratio, and back focal distance.

        It reports the approximate mirror radii of curvature, conic constants, mirror separation, and central obstruction for Newtonian, Cassegrain, Ritchey-Chrétien, Dall-Kirkham, and Gregorian designs.

        All code is written by Jake Pember: [Github](https://github.com/pemberj/) / [Email](mailto:jacob.pember@ikmail.com)

        Please get in contact if you have any questions, suggestions, or find any bugs!

        You can also [Buy me a pint](https://buymeacoffee.com/pemberj) if you find this tool useful and want to support its development!
        """
    )

    st.iframe(
        """
        <script type="text/javascript" src="https://cdnjs.buymeacoffee.com/1.0.0/button.prod.min.js" data-name="bmc-button" data-slug="pemberj" data-color="#FF5F5F" data-emoji="🍺" data-font="Cookie" data-text="Buy me a pint" data-outline-color="#000000" data-font-color="#ffffff"></script>
        """,
    )

cols = st.columns(2)

with cols[0]:
    design: str = st.selectbox(
        "Telescope design",
        DESIGNS,
        index=DESIGNS.index("Ritchey-Chrétien"),
        key="design",
    )

    st.info(f"{design}: {CONIC_NOTES[design]}")

    diameter: ScalarQuantity = st_quantity_input(
        "Primary mirror diameter (D)",
        min_value=0.0,
        value=2.2,
        step=0.1,
        possible_units=["mm", "cm", "m"],
        default_unit="m",
        key="diameter",
    )

    if design != "Newtonian":
        system_focal_length: ScalarQuantity = st_quantity_input(
            "System focal length (F)",
            min_value=1.0,
            value=17.6,
            step=0.1,
            possible_units=["mm", "cm", "m"],
            default_unit="m",
            disabled=design == "Newtonian",
            key="system_focal_length",
        )

    mode: str | None = st.segmented_control(
        "Specify the primary mirror by",
        ["Focal Length", "f/#"],
        default="f/#",
        key="mode",
    )

    if mode is None:
        st.error("Please select a specification mode.")
        st.stop()

    elif mode == "f/#":
        subcols = st.columns([1, 1.5])
        with subcols[0]:
            primary_f_number: float = st.number_input(
                "Primary mirror f/#",
                min_value=0.0,
                value=8.0 if design == "Newtonian" else 3.0,
                step=0.1,
                width=200,
                key="f_number",
            )

        primary_focal_length: ScalarQuantity = primary_f_number * diameter

        if design == "Newtonian":
            system_focal_length: ScalarQuantity = primary_focal_length

        with subcols[1]:
            st.metric(
                "Primary mirror focal length",
                f"{primary_focal_length:.6g}",
            )

    elif mode == "Focal Length":
        primary_focal_length: ScalarQuantity = st_quantity_input(
            "Primary mirror focal length",
            min_value=1.0,
            value=17.6 if design == "Newtonian" else 8.0,
            step=1.0,
            possible_units=["mm", "cm", "m"],
            default_unit="m",
            key="primary_focal_length",
        )

        if design == "Newtonian":
            system_focal_length: ScalarQuantity = primary_focal_length

        f_number: float = (primary_focal_length / diameter).si.value

    back_focal_distance: ScalarQuantity = st_quantity_input(
        "Back focal distance (BFD)",
        min_value=0.0,
        value=300.0,
        step=1.0,
        possible_units=["mm", "cm", "m"],
        default_unit="mm",
        help="Distance from the secondary vertex to the final focus.",
    )

    maximum_field: ScalarQuantity = st_quantity_input(
        "Maximum field",
        min_value=0.0,
        value=1.0,
        step=0.1,
        possible_units=["arcsec", "arcmin", "mrad", "deg"],
        default_unit="deg",
        key="maximum_field",
    )

    m2_diameter = None
    separation = None

    # if design == "Dall-Kirkham":
    #     # Need an additional parameter to uniquely determine the prescription for Dall-Kirkham, since the secondary is spherical and doesn't have a conic constant to solve for.
    #     st.info(
    #         """For a Dall-Kirkham design, you must specify one additional parameter to
    #         uniquely determine the prescription."""
    #     )

    #     dall_kirkham_parameter: str | None = st.segmented_control(
    #         "Specify the secondary mirror by",
    #         ["M2 Diameter", "M1-M2 Separation"],
    #         default="M2 Diameter",
    #         key="dall_kirkham_parameter",
    #     )

    #     if dall_kirkham_parameter is None:
    #         st.error("Please select a Dall-Kirkham specification mode.")
    #         st.stop()
    #     elif dall_kirkham_parameter == "M2 Diameter":
    #         m2_diameter: ScalarQuantity = st_quantity_input(
    #             "Secondary mirror diameter (D2)",
    #             min_value=0.0,
    #             value=0.3 * diameter.to(u.mm).value,
    #             step=1.0,
    #             possible_units=["mm", "cm", "m"],
    #             default_unit="mm",
    #             key="m2_diameter",
    #         )
    #         separation = None
    #     elif dall_kirkham_parameter == "M1-M2 Separation":
    #         separation: ScalarQuantity = st_quantity_input(
    #             "Primary-secondary separation (d)",
    #             min_value=0.0,
    #             value=0.5 * primary_focal_length.to(u.m).value,
    #             step=1.0,
    #             possible_units=["mm", "cm", "m"],
    #             default_unit="m",
    #             key="separation",
    #         )
    #         m2_diameter = None

if design != "Newtonian" and system_focal_length < primary_focal_length:
    st.error(
        f"System focal length ({system_focal_length:.1f}) must be greater than the primary focal length ({primary_focal_length:.1f})."
    )
    st.stop()

if design != "gregorian" and back_focal_distance >= primary_focal_length:
    st.error(
        "Back focal distance must be less than the primary focal length for non-Gregorian designs."
    )
    st.stop()

telescope: TelescopeDesign = TelescopeDesign(
    name=f"{design} D={diameter:.1f} F={system_focal_length:.1f}",
    design=design,
    M1=Mirror(diameter=diameter, radius=2 * primary_focal_length),
    M2=Mirror(diameter=m2_diameter),
    system_focal_length=system_focal_length,
    back_focal_distance=back_focal_distance,
    maximum_field=maximum_field,
    primary_secondary_separation=separation,
)

with cols[1]:
    metric_cols = st.columns(2)

    with metric_cols[0]:
        st.metric(
            "System focal length", f"{telescope.system_focal_length:.6g}", border=True
        )

    with metric_cols[1]:
        st.metric("System f/number", f"{telescope.system_f_number:,.5g}", border=True)

    with metric_cols[0]:
        st.metric(
            "Primary-secondary separation",
            f"{telescope.primary_secondary_separation:,.5g}",
            border=True,
        )

    with metric_cols[1]:
        st.metric(
            "Central obstruction", f"{telescope.central_obstruction:.3g}", border=True
        )

    with metric_cols[0]:
        st.metric("Primary radius R1", f"{telescope.M1.radius:.6g}", border=True)

    with metric_cols[1]:
        st.metric("Primary conic K1", f"{telescope.M1.conic:,.5g}", border=True)

    with metric_cols[0]:
        st.metric("Secondary radius R2", f"{telescope.M2.radius:,.5g}", border=True)

    with metric_cols[1]:
        st.metric("Secondary conic K2", f"{telescope.M2.conic:,.5g}", border=True)

st.divider()

table_cols = st.columns(3)
with table_cols[0]:
    render_table(
        "System Parameters",
        [
            ("System focal length", f"{telescope.system_focal_length:,.5g}"),
            ("System f/#", f"{telescope.system_f_number:,.5g}"),
            (
                "Primary-secondary separation d",
                f"{telescope.primary_secondary_separation:,.5g}",
            ),
            ("Central obstruction fraction", f"{telescope.central_obstruction:,.5g}"),
            ("Back focal distance", f"{telescope.back_focal_distance:,.5g}"),
        ],
    )

with table_cols[1]:
    render_table(
        "Primary Mirror",
        [
            ("Diameter D1", f"{telescope.M1.diameter:,.5g}"),
            ("Radius of curvature R1", f"{telescope.M1.radius:,.5g}"),
            ("Conic constant K1", f"{telescope.M1.conic:,.5g}"),
            ("Focal length f1", f"{telescope.M1.focal_length:,.5g}"),
            ("Primary f/number", f"{telescope.M1.f_number:,.5g}"),
        ],
    )
with table_cols[2]:
    render_table(
        "Secondary Mirror",
        [
            ("Diameter D2", f"{telescope.M2.diameter:,.5g}"),
            ("Radius of curvature R2", f"{telescope.M2.radius:,.5g}"),
            ("Conic constant K2", f"{telescope.M2.conic:,.5g}"),
            ("Focal length f2", f"{telescope.M2.focal_length:,.5g}"),
            ("Extension factor", f"{telescope.extension_factor:,.5g}"),
        ],
    )

performance_cols = st.columns(2)
with performance_cols[0]:
    render_table(
        "Optical Aberrations",
        [
            ("Coma coefficient", f"{telescope.coma_coefficient:,.5g}"),
            ("Coma", f"{telescope.coma:,.5g}"),
            ("Astigmatism coefficient", f"{telescope.astigmatism_coefficient:,.5g}"),
            ("Astigmatism", f"{telescope.astigmatism:,.5g}"),
            ("Petzval curvature", f"{telescope.petzval_curvature:,.5g}"),
        ],
    )
with performance_cols[1]:
    render_table(
        "Focal plane",
        [
            ("Plate scale", f"{telescope.plate_scale:,.5g}"),
            ("Image height at max field", f"{telescope.image_height:,.5g}"),
            # ("Angular field of view", f"{telescope.angular_field_of_view:,.5g}"),
            ("Angular resolution", f"{telescope.angular_resolution:,.5g}"),
            ("Field curvature", f"{telescope.field_curvature:,.5g}"),
            ("Radius of field curvature", f"{telescope.field_curvature_radius:,.5g}"),
        ],
    )
