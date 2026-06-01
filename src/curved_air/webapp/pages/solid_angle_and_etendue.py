#! uv run streamlit run

import numpy as np
import pandas as pd
import streamlit as st
from astropy import units as u

from curved_air.type_hinting import ScalarQuantity
from curved_air.utils import st_quantity_input


def compute_solid_angle(half_cone_angle: ScalarQuantity) -> ScalarQuantity:
    return 2 * np.pi * (1 - np.cos(half_cone_angle)) * u.sr


st.set_page_config(
    page_title="Curved Air: Optical Design Tools",
    page_icon="🌈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.header("Solid Angle & Étendue Calculator")

with st.expander("About this tool", expanded=False):
    st.markdown(
        """
        Use this tool to compute the solid angle subtended by a beam described by one of: full cone angle, half cone angle, focal ratio (f/#) or numerical aperture (NA)."

        You can also optionally compute the étendue with an additional input of the emitting area.

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
    st.subheader("Solid angle input")
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

    half_cone_angle = None
    note = ""

    if mode == "Full cone angle":
        full_cone_angle: ScalarQuantity = st_quantity_input(
            "Full cone angle",
            min_value=0.0,
            max_value=180.0,
            value=30.0,
            step=0.1,
            possible_units=["arcsec", "arcmin", "deg", "mrad", "rad"],
            default_unit="deg",
            key="full_cone_angle",
        )
        half_cone_angle: ScalarQuantity = 0.5 * full_cone_angle

    elif mode == "Half cone angle":
        half_cone_angle: ScalarQuantity = st_quantity_input(
            "Half cone angle",
            min_value=0.0,
            max_value=180.0,
            value=30.0,
            step=0.1,
            possible_units=["arcsec", "arcmin", "deg", "mrad", "rad"],
            default_unit="deg",
            key="half_cone_angle",
        )

    elif mode == "Focal ratio (f/#)":
        f_number: float = st.number_input(
            "Focal ratio f/#",
            min_value=0.5,
            value=2.0,
            step=0.1,
            width=200,
            help="Calculation uses paraxial relation in air: NA~1/(2 f/#).",
        )
        # For paraxial rays in air: NA~1/(2 f/#)
        numerical_aperture: float = 1 / (2 * f_number)
        half_cone_angle: ScalarQuantity = np.arcsin(numerical_aperture) * u.rad

    else:
        numerical_aperture: float = st.number_input(
            "Numerical aperture NA",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.01,
            width=200,
        )
        half_cone_angle = np.arcsin(numerical_aperture) * u.rad

    solid_angle: ScalarQuantity = compute_solid_angle(half_cone_angle)
    full_cone_angle_out: ScalarQuantity = 2 * half_cone_angle
    na_out: ScalarQuantity = np.sin(half_cone_angle)  # ty:ignore[invalid-assignment]

    if note:
        st.caption(note)

    subcols = st.columns(3)

    with subcols[0]:
        st.metric("Half cone angle", f"{half_cone_angle.to(u.deg):.4g}")
    with subcols[1]:
        st.metric("Full cone angle", f"{full_cone_angle_out.to(u.deg):.4g}")
    with subcols[2]:
        st.metric("Solid angle $\\Omega$", f"{solid_angle.to(u.sr):.4g}")

    with subcols[0]:
        st.metric("Numerical aperture NA", f"{na_out:.4g}")
    with subcols[1]:
        st.metric("Focal ratio f/# (in air)", f"{0.5 / na_out:.4g}")

    st.subheader("Étendue")
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
        etendue: ScalarQuantity = area * solid_angle
        subcols = st.columns(2)

        with subcols[0]:
            st.metric("Area A", f"{area.value:.4g} {area.unit:latex_inline}")
        with subcols[1]:
            st.metric(
                "Etendue G = A·Ω",
                f"{etendue.to(u.m**2 * u.sr).value:.4g} {u.m**2 * u.sr:latex_inline}",
            )

    line: dict[str, str] = {
        "Half cone angle": f"{half_cone_angle:.5g}",
        "Full cone angle": f"{full_cone_angle_out:.5g}",
        "NA": f"{na_out:.5g}",
        "f/#": f"{0.5 / na_out:.5g}",
        "Solid angle": f"{solid_angle:.5g}",
        "Area": f"{area:.5g}" if area is not None else "--",
        "Étendue": f"{etendue.to(u.m**2 * u.sr):.5g}" if area is not None else "--",
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
    if not df.empty:
        st.dataframe(df, hide_index=True, width="stretch", height=600)

        subcols = st.columns(2)

        with subcols[0], st.container(horizontal_alignment="center"):
            st.button(
                "Clear table",
                on_click=lambda: st.session_state.pop("table", None),
                type="secondary",
                help="Clear all surface data. This cannot be undone.",
            )

        with subcols[1], st.container(horizontal_alignment="center"):
            st.download_button(
                "Download table as CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="surface_tolerances.csv",
                mime="text/csv",
            )
