"""
This calculator allows for calculating and generating a report on the performance of a
rotational atmospheric dispersion corrector (ADC) design with two glass types.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import streamlit as st
from astropy import units as u
from matplotlib import pyplot as plt

from curved_air.glass import GLASS_CATALOGUES, Glass, GlassCatalogue
from curved_air.instruments.ADC import ADC, PrismStack
from curved_air.plotting import plot_style

if TYPE_CHECKING:
    from collections.abc import Iterable

plt.style.use(plot_style)


st.header("Atmospheric Dispersion Corrector")

with st.expander("About this calculator", expanded=False):
    st.markdown(
        """
        This calculator allows for the basic design of a rotational atmospheric
        dispersion corrector (ADC) with two different glasses. Given the glass choice
        and other input parameters, an ADC system is optimised and a report is generated
        on its performance.

        All code is written by [Jake Pember](https://github.com/pemberj/)

        Please get in contact if you have any questions, suggestions, or find any bugs!
        """
    )

st.divider()

# Interface
with st.sidebar:
    with st.expander(label="Wavelength range", expanded=True):
        min_wl: float = (
            st.number_input(
                "Min λ (nm)",
                min_value=100.0,
                max_value=5000.0,
                value=380.0,
                step=5.0,
                key="min_wl",
            )
            * u.nm
        )

        max_wl: float = (
            st.number_input(
                "Max λ (nm)",
                min_value=100.0,
                max_value=5000.0,
                value=950.0,
                step=5.0,
                key="max_wl",
            )
            * u.nm
        )

    with st.expander(label="Glass selection", expanded=True):
        catalogue1: str = st.selectbox(
            "Catalogue for glass 1",
            GLASS_CATALOGUES.keys(),
            index=list(GLASS_CATALOGUES.keys()).index("Schott")
            if "Schott" in GLASS_CATALOGUES
            else 0,
        )

        try:
            glasses1 = GlassCatalogue(catalogue1)
            glass_names1: list[str] = sorted(glasses1.keys())
        except Exception:
            st.error(f"Could not load {catalogue1} glass catalogue.")

        glass1: str = st.selectbox(
            "Glass 1",
            glass_names1,
            index=glass_names1.index("K10") if "K10" in glass_names1 else 0,
        )

        glass1: Glass = glasses1[glass1]

        catalogue2: str = st.selectbox(
            "Catalogue for glass 2",
            GLASS_CATALOGUES.keys(),
            index=list(GLASS_CATALOGUES.keys()).index("Ohara")
            if "Ohara" in GLASS_CATALOGUES
            else 0,
        )

        try:
            glasses2 = GlassCatalogue(catalogue2)
            glass_names2: list[str] = sorted(glasses2.keys())
        except Exception:
            st.error(f"Could not load {catalogue2} glass catalogue.")

        glass2: str = st.selectbox(
            "Glass 2",
            glass_names2,
            index=glass_names2.index("S-FPL51") if "S-FPL51" in glass_names2 else 0,
        )

        glass2: Glass = glasses2[glass2]

        # st.toggle(
        #     "Plot glass refractive indices",
        #     value=False,
        # )

        # if st.session_state["ref_index_plot_toggle"]:
        #     wls: ArrayQuantity = (
        #         np.linspace(min_wl.to(u.nm).value, max_wl.to(u.nm).value, 200) * u.nm
        #     )

        #     fig: go.Figure = go.Figure()  # go figure!

        #     for g in [glass1, glass2]:
        #         fig.add_trace(
        #             trace=go.Scatter(
        #                 x=wls,
        #                 y=g.refractive_index(wls),
        #                 mode="lines",
        #             )
        #         )
        #     fig.update_layout(
        #         xaxis_title="Wavelength (nm)",
        #         yaxis_title="Refractive Index",
        #         template="plotly_dark",
        #     )

        #     st.plotly_chart(fig)

    with st.expander(label="Beam sizes", expanded=True):
        cols = st.columns(2)
        with cols[0]:
            D_telescope: float = st.number_input(
                "Telescope diameter",
                min_value=1.0,
                max_value=1000.0,
                value=800.0,
                step=100.0,
            )

        with cols[1]:
            D_telescope_unit: u.Unit = u.Unit(
                st.segmented_control(
                    "Unit",
                    options=["mm", "cm", "m"],
                    default="mm",
                    key="D_telescope_unit",
                )
            )

        D_telescope *= D_telescope_unit

        cols = st.columns(2)
        with cols[0]:
            D_ADC: float = st.number_input(
                "ADC diameter",
                min_value=1.0,
                max_value=1000.0,
                value=25.0,
                step=5.0,
            )

        with cols[1]:
            D_ADC_unit: u.Unit = u.Unit(
                st.segmented_control(
                    "Unit",
                    options=["mm", "cm", "m"],
                    default="mm",
                    key="D_ADC_unit",
                )
            )

        D_ADC *= D_ADC_unit

    with st.expander(label="Zenith angles", expanded=True):
        z_max: float = (
            st.number_input(
                "Maximum zenith angle (degrees)",
                min_value=0.0,
                max_value=90.0,
                value=60.0,
                step=1.0,
                key="z_max",
            )
            * u.degree
        )

    with st.expander(label="Contraints", expanded=True):
        optimisation_criterion: str = st.segmented_control(
            "ADC optimisation criterion",
            ["PTV", "RMS", "RSS", "Max. Abs."],
            default="PTV",
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
adc = ADC(
    prism_stack=PrismStack([glass1, glass2]),
    min_wl=min_wl,
    max_wl=max_wl,
    D_telescope=D_telescope,
    D_ADC=D_ADC,
    z_max=z_max,
    criterion=optimisation_criterion,
)

with st.spinner("Optimising ADC..."):
    adc.optimise()

# Summary
cols: Iterable = iter(
    st.columns(
        6,
        gap="xxsmall",
        vertical_alignment="top",
        border=True,
        width="stretch",
    )
)

with next(cols):
    st.metric(
        "Airmass",
        f"{1 / np.cos(z_max.to(u.rad).value):.3f}",
        help="""The airmass at the maximum zenith angle. This is calculated using the
        standard airmass formula (1/cos(z_max)).""",
    )

with next(cols):
    st.metric(
        "PTV",
        f"{adc.residuals_PTV:.3f}",
        help="""The PTV (Peak-to-Valley) residuals of the ADC. This is the maximum
        deviation from the ideal dispersion correction across all wavelengths in the
        spectrum **projected on sky**.""",
    )

with next(cols):
    st.metric(
        "RMS Residuals",
        f"{adc.residuals_RMS:.3f}",
        help="""The RMS (Root Mean Square) residuals of the ADC. This is the root-mean-
        square deviation from the ideal dispersion correction across all wavelengths in
        the spectrum **projected on sky**.""",
    )

with next(cols):
    st.metric(
        "Apex angle $\\alpha$",
        f"{adc.first_apex_angle:.2f}",
        help="""The required apex angle of the ADC.""",
    )

with next(cols):
    st.metric(
        "Compound angle $\\beta$",
        f"{adc.beta:.2f}",
        help="""The compound angle of the ADC. This is the angle at which the ADC is
        rotated relative to the telescope axis, and is used to correct for atmospheric
        dispersion in the spectrograph. It is calculated based on the required apex
        angle and the spectral resolution of the spectrograph.""",
    )

with next(cols):
    st.metric(
        "Zero-deviation wavelength",
        f"{adc.wl_0:.2f}",
        help="""The zero-deviation wavelength of the ADC. This is the wavelength at
        which the ADC does not deviate an incoming beam. """,
    )

with st.container(horizontal_alignment="center"), st.spinner("Plotting..."):
    st.pyplot(adc.plot(show=False), width=1000)
