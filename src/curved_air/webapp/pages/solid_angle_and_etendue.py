#! uv run streamlit run

import numpy as np
import pandas as pd
import streamlit as st
from astropy import units as u
from astropy.units import Quantity

from curved_air.type_hinting import ScalarQuantity
from curved_air.utils import st_quantity_input


def compute_solid_angle(half_cone_angle: ScalarQuantity) -> ScalarQuantity:
    return 2 * np.pi * (1 - np.cos(half_cone_angle)) * u.sr


st.header("Solid Angle & Étendue Calculator", anchor=False)

with st.expander("About this tool", expanded=False):
    """
    Use this tool to compute the solid angle subtended by a beam described by one of:
    full cone angle, half cone angle, focal ratio (f/#) or numerical aperture (NA).

    You can also optionally compute the étendue with an additional input of the
    emitting area.

    To compare multiple computations, you can add results to the table, where the data
    can be sorted and downloaded as a CSV file.

    All code is written by Jake Pember :gray-badge[:material/language: [Web](https://www.jakepember.com)] :gray-badge[:material/email: [Email](mailto:jacob.pember@ikmail.com)] :gray-badge[:material/code: [Github](https://www.github.com/pemberj)]

    Please get in contact if you have any questions, suggestions, or find any bugs!

    You can also [Buy me a pint](https://buymeacoffee.com/pemberj) if you find this tool
    useful and want to support its development!
    """

    st.iframe(
        """
        <script type="text/javascript" src="https://cdnjs.buymeacoffee.com/1.0.0/button.prod.min.js" data-name="bmc-button" data-slug="pemberj" data-color="#E5660C" data-emoji="🍺" data-font="Cookie" data-text="Buy me a pint" data-outline-color="#000000" data-font-color="#ffffff"></script>
        """,
    )

cols = st.columns(2)

with cols[0]:
    st.subheader("Solid angle input", anchor=False)

    full_cone_angle = None
    half_cone_angle = None
    numerical_aperture = None
    f_number = None

    mode: str | None = st.segmented_control(
        "Choose input type",
        [
            "Full cone angle",
            "Half cone angle",
            "Focal ratio (f/#)",
            "Numerical aperture (NA)",
        ],
        default="Half cone angle",
    )
    if mode is None:
        st.warning("Please select an input type for the solid angle.")
        st.stop()

    if mode == "Full cone angle":
        full_cone_angle: ScalarQuantity = st_quantity_input(
            "Full cone angle",
            min_value=0.0,
            max_value=180.0,
            value=14.362,
            step=0.1,
            format="%.6g",
            possible_units=["arcsec", "arcmin", "deg", "mrad", "rad"],
            default_unit="deg",
            key="full_cone_angle",
        )
        half_cone_angle: ScalarQuantity = full_cone_angle / 2

    elif mode == "Half cone angle":
        half_cone_angle: ScalarQuantity = st_quantity_input(
            "Half cone angle",
            min_value=0.0,
            max_value=90.0,
            value=7.181,
            step=0.1,
            format="%.6g",
            possible_units=["arcsec", "arcmin", "deg", "mrad", "rad"],
            default_unit="deg",
            key="half_cone_angle",
        )

    elif mode == "Focal ratio (f/#)":
        f_number: float = st.number_input(
            "Focal ratio f/#",
            min_value=0.5,
            value=4.0,
            step=0.1,
            format="%.4g",
            width=200,
            help="Calculation uses paraxial relation in air: NA~1/(2 f/#).",
            key="f_number",
        )

        # For paraxial rays in air: NA~1/(2 f/#)
        numerical_aperture: float = 1 / (2 * f_number)
        half_cone_angle: ScalarQuantity = (np.arcsin(numerical_aperture) * u.rad).to(
            u.deg
        )

    else:
        numerical_aperture: float = st.number_input(
            "Numerical aperture NA",
            min_value=0.0,
            max_value=1.0,
            value=0.125,
            step=0.01,
            format="%.6g",
            width=200,
            key="numerical_aperture",
        )
        half_cone_angle: ScalarQuantity = (np.arcsin(numerical_aperture) * u.rad).to(
            u.deg
        )

    assert isinstance(half_cone_angle, Quantity)

    if half_cone_angle >= 90 * u.deg:
        st.error(
            "The values you have entered correspond to a half cone angle greater than 90 degrees."
        )

    solid_angle: ScalarQuantity = compute_solid_angle(half_cone_angle=half_cone_angle)

    if full_cone_angle is None:
        full_cone_angle: ScalarQuantity = 2 * half_cone_angle

    if numerical_aperture is None:
        numerical_aperture: float = np.sin(half_cone_angle)  # ty:ignore[invalid-assignment]

    if f_number is None:
        f_number: float = 1 / (2 * numerical_aperture)

    subcols = st.columns(3)

    with subcols[0]:
        st.metric("Half cone angle", f"{half_cone_angle:.4g}")
    with subcols[1]:
        st.metric("Full cone angle", f"{full_cone_angle:.4g}")
    with subcols[2]:
        st.metric("Solid angle $\\Omega$", f"{solid_angle.to(u.sr):.4g}")

    with subcols[0]:
        st.metric("Numerical aperture NA", f"{numerical_aperture:.4g}")
    with subcols[1]:
        st.metric("Focal ratio f/# (in air)", f"{f_number:,.5g}")

    st.subheader("Étendue", anchor=False)

    area_mode: str = st.radio(
        "Area input", ["None", "Direct area", "Circle diameter"], horizontal=True
    )

    area = None
    if area_mode == "Direct area":
        area: ScalarQuantity = st_quantity_input(
            "Area A",
            min_value=0.0,
            value=10.0,
            step=0.01,
            # format="%.8f",
            possible_units=["m²", "cm²", "mm²"],
            default_unit="cm²",
            key="area",
        )
    elif area_mode == "Circle diameter":
        diameter: ScalarQuantity = st_quantity_input(
            "Circle diameter D",
            min_value=0.0,
            value=10.0,
            step=0.01,
            possible_units=["µm", "mm", "cm", "m"],
            default_unit="mm",
            key="diameter",
        )
        area: ScalarQuantity = np.pi * (diameter / 2) ** 2

    if area is not None:
        etendue: ScalarQuantity = area * solid_angle.to(u.sr)

        subcols = st.columns(2)

        with subcols[0]:
            st.metric("Area A", f"{area.value:.4g} {area.unit:latex_inline}")
        with subcols[1]:
            st.metric(
                "Etendue G = A·Ω",
                f"{etendue.to(area.unit * u.sr).value:.4g} {area.unit * u.sr:latex_inline}",
            )

    line: dict[str, str] = {
        "Half cone angle": f"{half_cone_angle.to(u.deg):.5g}",
        "Full cone angle": f"{full_cone_angle.to(u.deg):.5g}",
        "NA": f"{numerical_aperture:.5g}",
        "f/#": f"{f_number:,.5g}",
        "Solid angle": f"{solid_angle:.5g}",
        "Area": f"{area:.5g}" if area is not None else "--",
        "Étendue": f"{etendue:.5g}" if area is not None else "--",
    }

    if "table" not in st.session_state:
        st.session_state["table"] = []

    with st.container(horizontal_alignment="center"):
        st.button(
            "Add to table",
            on_click=lambda: st.session_state["table"].append(line),
            type="primary",
        )

    df = pd.DataFrame(data=st.session_state["table"])

with cols[1]:
    st.dataframe(df, hide_index=True, width="stretch", height=650)

    subcols = st.columns(2)

    with subcols[0], st.container(horizontal_alignment="center"):
        st.button(
            "Clear table",
            on_click=lambda: st.session_state.pop("table", None),
            type="secondary",
            disabled=df.empty,
            help="Clear all surface data. This cannot be undone.",
        )

    with subcols[1], st.container(horizontal_alignment="center"):
        st.download_button(
            "Download table as CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="surface_tolerances.csv",
            disabled=df.empty,
            mime="text/csv",
        )
