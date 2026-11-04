#! uv run streamlit run

# This calculator allows for calculating and generating a report on the performance of a
# rotational atmospheric dispersion corrector (ADC) design with two glass types.

from __future__ import annotations

from itertools import cycle
from typing import TYPE_CHECKING, Literal

import numpy as np
import streamlit as st
from astropy import units as u
from matplotlib import pyplot as plt

from curved_air.glass import GLASS_CATALOGUES, Glass, GlassCatalogue
from curved_air.instruments.ADC import ADC, PrismStack
from curved_air.optics.dispersers import Prism
from curved_air.other import Air
from curved_air.plotting import plot_style
from curved_air.utils import st_glass_chooser, st_quantity_input
from curved_air.webapp import badges, bmac

if TYPE_CHECKING:
    from collections.abc import Iterable

    from curved_air.type_hinting import ScalarQuantity

plt.style.use(plot_style)

MAX_ELEMENTS: int = 5


def _edit_prism(index: int) -> None:
    """
    Widget to edit the glass of the prism at the given index, lazy loading the glass
    catalogue chooser (can be a lot to load) in a popover dialog
    """

    assert "prisms" in st.session_state, "Prisms not initialized in session state."

    assert 0 <= index < len(st.session_state["prisms"]), "Prism index out of range."

    with st.container(border=True):
        cols = st.columns([1, 1], gap="small", vertical_alignment="center")

        with cols[0]:
            st.write(f"Prism {index + 1}")

        with cols[1], st.container(horizontal_alignment="right"):
            if st.button(
                "Remove",
                type="tertiary",
                key=f"remove_prism_{i + 1}",
            ):
                st.session_state["prisms"].pop(i)
                st.rerun()

        p: Prism = st.session_state["prisms"][index]

        glass: Glass = st_glass_chooser(
            current_glass=p.glass,
            catalogues=GLASS_CATALOGUES,
            key=f"prism_{index + 1}_glass",
        )

        st.session_state["prisms"][index].glass = glass


st.header("Atmospheric Dispersion Corrector", anchor=False)

with st.expander("About this tool", expanded=False):
    f"""
    This tool allows for the basic design of a rotational atmospheric dispersion
    corrector (ADC) with two different glasses. Given the glass choice and other input
    parameters, an ADC system is optimised and a report is generated on its expected
    performance.

    All code is written by Jake Pember {badges}

    Please get in contact if you have any questions, suggestions, or find any bugs!

    You can also [Buy me a pint](https://buymeacoffee.com/pemberj) if you find this tool
    useful and want to support its development!
    """

    bmac()

# Interface
with st.sidebar:
    with st.expander(label="Air model", expanded=False):
        model_name: Literal["ciddor", "edlen", "zemax"] = st.selectbox(
            "Air model",
            ["ciddor", "edlen", "zemax"],
            index=2,
            format_func=lambda x: x.capitalize(),
            help="""The air model to use for calculating the refractive index of air.
            - `ciddor`: Ciddor (1996) air model
            - `edlen`: Edlen (1966) air model
            - `zemax`: Zemax air model (default)""",
        )

        temperature: ScalarQuantity | None = st_quantity_input(
            "Temperature",
            min_value=-100.0,
            max_value=100.0,
            value=15.0,
            step=1.0,
            possible_units=["°C", "K"],
            default_unit="°C",
            format="%.2g",
            key="air_model_temperature",
        )

        pressure: ScalarQuantity | None = st_quantity_input(
            "Pressure",
            min_value=0.0,
            # max_value=2000.0,
            value=1013.25,
            step=10.0,
            possible_units=["hPa", "Pa", "mbar", "bar", "torr"],
            default_unit="hPa",
            format="%.2g",
            key="air_model_pressure",
        )

        humidity: float | None = st.number_input(
            "Relative humidity",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
            help="""Relative humidity as a fraction (0.0 - 1.0).
            Note: The Zemax air model does not support humidity.""",
            disabled=model_name == "zemax",
            format="%.2g",
            key="air_model_humidity",
        )

        carbon_dioxide: float | None = st.number_input(
            "CO₂ concentration",
            min_value=0.0,
            max_value=10_000.0,
            value=425.0,
            step=10.0,
            help="""CO₂ concentration in ppm (parts per million).
            Note: The Zemax and Edlen air models do not support CO₂ concentration.""",
            disabled=model_name != "ciddor",
            format="%.2g",
            key="air_model_co2",
        )

        air_model = Air(
            air_model=model_name,
            temperature=temperature,
            pressure=pressure,
            humidity=humidity if model_name != "zemax" else None,
            carbon_dioxide=carbon_dioxide if model_name in "ciddor" else None,
        )

    with st.expander(label="Wavelength range", expanded=False):
        min_wl: ScalarQuantity = st_quantity_input(
            "Min λ",
            min_value=0.0,
            value=380.0,
            step=5.0,
            possible_units=["nm", "µm"],
            default_unit="nm",
            format="%.2g",
            key="adc_min_wl",
        )

        max_wl: ScalarQuantity = st_quantity_input(
            "Max λ",
            min_value=0.0,
            value=950.0,
            step=5.0,
            possible_units=["nm", "µm"],
            default_unit="nm",
            format="%.2g",
            key="adc_max_wl",
        )

    with st.expander(label="Prism elements", expanded=False):
        if "prisms" not in st.session_state:
            st.session_state["prisms"]: list[Prism] = [
                Prism(glass=GlassCatalogue("Schott")["K10"]),
                Prism(glass=GlassCatalogue("Ohara")["S-FPL51"]),
            ]

        for i, _p in enumerate(st.session_state["prisms"]):
            _edit_prism(i)

        if len(st.session_state["prisms"]) < MAX_ELEMENTS:  # noqa: SIM102
            if st.button("Add prism element"):
                st.session_state["prisms"].append(
                    Prism(glass=GlassCatalogue("Schott")["K10"])
                )
                st.rerun()

        else:
            st.write(
                "This app only supports compound prisms with a maximum of 5 elements."
            )

    with st.expander(label="Beam sizes", expanded=False):
        D_telescope: ScalarQuantity = st_quantity_input(
            "Telescope diameter",
            min_value=1.0,
            max_value=1000.0,
            value=800.0,
            step=100.0,
            possible_units=["mm", "cm", "m"],
            default_unit="mm",
            format="%.2g",
            key="D_telescope",
        )

        D_ADC: ScalarQuantity = st_quantity_input(
            "ADC diameter",
            min_value=1.0,
            max_value=1000.0,
            value=25.0,
            step=5.0,
            possible_units=["mm", "cm", "m"],
            default_unit="mm",
            format="%.2g",
            key="D_ADC",
        )

    with st.expander(label="Zenith angles", expanded=False):
        z_max: ScalarQuantity = (
            st.number_input(
                "Maximum zenith angle (degrees)",
                min_value=0.0,
                max_value=90.0,
                value=60.0,
                step=1.0,
                format="%.2g",
                key="z_max",
            )
            * u.degree
        )

    with st.expander(label="Constraints", expanded=False):
        optimisation_criterion: str = st.segmented_control(
            "ADC optimisation criterion",
            ["PTV", "RMS", "RSS", "Max. Abs."],
            default="Max. Abs.",
        )  # ty:ignore[invalid-assignment]
        if optimisation_criterion == "Max. Abs.":
            optimisation_criterion = "max_abs"

        # allow_compound_angle: bool = st.toggle(
        #     "Allow compound angle",
        #     value=True,
        #     help="If checked, the ADC can be optimised with a compound angle.",
        # )

    st.space(10)

    with st.container(horizontal_alignment="center"):
        st.button("Recompute", type="primary")


# Create and optimise ADC
try:
    adc = ADC(
        prism_stack=PrismStack(st.session_state["prisms"]),
        air_model=air_model,
        min_wl=min_wl,
        max_wl=max_wl.to(u.nm),
        D_telescope=D_telescope,
        D_ADC=D_ADC,
        z_max=z_max,
        criterion=optimisation_criterion,
    )
    print(adc)
except Exception as e:
    st.exception(e)
    st.stop()

with st.container(horizontal_alignment="center"):  # noqa: SIM117
    with st.spinner("Optimising ADC..."):
        adc.optimise()

# Summary
cols: Iterable = cycle(
    st.columns(
        4,
        gap="small",
        vertical_alignment="top",
        width="stretch",
    )
)

with next(cols):
    st.metric(
        "Airmass",
        f"{1 / np.cos(z_max.to(u.rad).value):.3g}",
        border=True,
        help="""The airmass at the maximum zenith angle. This is calculated using the
        standard airmass formula (1/cos(z_max)).""",
    )

with next(cols):
    st.metric(
        "PTV",
        f"{adc.residuals_PTV:.3g}",
        border=True,
        help="""The PTV (Peak-to-Valley) residuals of the ADC. This is the maximum
        deviation from the ideal dispersion correction across all wavelengths in the
        spectrum **projected on sky**.""",
    )

with next(cols):
    st.metric(
        "RMS",
        f"{adc.residuals_RMS:.3g}",
        border=True,
        help="""The RMS (Root Mean Square) residuals of the ADC. This is the root-mean-
        square deviation from the ideal dispersion correction across all wavelengths in
        the spectrum **projected on sky**.""",
    )

with next(cols):
    st.metric(
        "Zero-deviation wavelength",
        f"{adc.wl_0:.2f}",
        border=True,
        help="""The zero-deviation wavelength of the ADC. This is the wavelength at
        which the ADC does not deviate an incoming beam. """,
    )

for i, p in enumerate(adc.prism_stack.prism_list, start=1):
    with next(cols):
        st.metric(
            f"Prism {i} apex angle",
            f"{abs(p.apex_angle):.3f}",
            border=True,
            help=f"""The apex angle of prism {i} in the ADC. This
            is the angle at which the prism is rotated to achieve the optimal dispersion
            correction.""",
        )

with st.container(horizontal_alignment="center"), st.spinner("Plotting..."):
    # st.pyplot(adc.plot(show=False), width=1000)
    figure: plt.Figure = adc.plot(show=False, layout=False)
    st.pyplot(figure, width=1000)
