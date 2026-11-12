#! uv run streamlit run

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from astropy import units as u

from curved_air.star_counts import (
    SKY_REGION_MODELS,
    SkyRegionModel,
    base_cumulative_density,
    limiting_magnitude_for_area,
)
from curved_air.utils import st_quantity_input
from curved_air.webapp import badges, bmac

if TYPE_CHECKING:
    from numpy.typing import NDArray  # noqa: F401

    from curved_air.type_hinting import ScalarQuantity

st.set_page_config(
    page_title="Guide Star Field of View",
    layout="wide",
)

st.header("Guide Star Field of View", anchor=False)

with st.expander("About this tool", expanded=False):
    f"""
    This tool estimates the field of view required so that a telescope pointing is
    expected to contain at least one star brighter than a chosen magnitude limit, or
    vice-versa to estimate the brightest magnitude limit for which at least one star is
    expected within a field of view of a particular angular size.

    The model is not based on specific star counting, but rather on a smooth multi-part
    exponential fit to the cumulative stellar surface density as a function of
    magnitude and galactic latitude. G is the native model band. V is approximated as a
    modestly sparser distribution.

    The tool is intended for high-level trade studies in order to define the optical
    requirements of instruments and guide systems.

    All code is written by Jake Pember {badges}

    Please get in contact if you have any questions, suggestions, or find any bugs!

    You can also [Buy me a pint](https://buymeacoffee.com/pemberj) if you find this tool
    useful and want to support its development!
    """

    bmac()

cols = st.columns([1, 3])

with cols[0]:
    mode: str | None = st.segmented_control(
        "Calculation mode",
        options=["Magnitude -> FoV", "FoV -> Magnitude"],
        default="Magnitude -> FoV",
        help="""Select whether you want to calculate the required field of view for a
        given magnitude limit, or the limiting magnitude for a given field of view.""",
    )

    band: Literal["G", "V"] = st.pills(
        "Photometric band",
        options=["G", "V"],
        selection_mode="single",
        default="G",
        required=True,
        help="G is the native model band. V is treated as slightly sparser than G.",
    )

    selected_model: SkyRegionModel = st.selectbox(
        "Sky coverage target",
        options=SKY_REGION_MODELS,
        format_func=lambda model: (
            f"{model.label} (~{model.equivalent_coverage * 100:.0f}% coverage)"
            if model.equivalent_coverage is not None
            else model.label
        ),
        index=3,
    )

    if selected_model.equivalent_coverage is not None:
        st.metric(
            "Equivalent coverage",
            f"{selected_model.equivalent_coverage * 100:.0f}%",
            help="""Maps to a conservative Galactic-latitude regime for that coverage
            level.""",
        )
    else:
        st.caption(
            """Using the all-sky average density regime. This is a rough estimate for
            the entire sky."""
        )

    region: str = selected_model.label

    st.divider()

    if mode == "Magnitude -> FoV":
        target_magnitude: float = st.number_input(
            f"Target {band}-band magnitude",
            min_value=-2.0,
            max_value=18.0,
            value=12.0,
            step=0.1,
            width=200,
            format="%.3f",
        )
        input_area = None
    else:
        diameter_mode: str = st.radio(
            "FoV input type",
            options=["Circular diameter", "Area"],
            horizontal=True,
        )

        if diameter_mode == "Circular diameter":
            input_diameter: ScalarQuantity = st_quantity_input(
                "Field diameter",
                min_value=0.0,
                value=10.0,
                step=0.5,
                possible_units=["arcsec", "arcmin", "deg"],
                default_unit="arcmin",
                key="guide_star_fov_diameter",
            )
            input_area: ScalarQuantity = (np.pi * (input_diameter / 2) ** 2).to(
                u.deg**2
            )
        else:
            input_area: ScalarQuantity = st_quantity_input(
                "Field area",
                min_value=0.0,
                value=0.02,
                step=0.005,
                possible_units=["arcmin^2", "deg^2"],
                default_unit="deg^2",
                key="guide_star_fov_area",
            )
        target_magnitude = 0.0

    st.info("""Criterion: expected star count in the field is at least 1. This is a
        planning estimate, not a strict catalogue-backed guarantee for every pointing.""")


if mode is None:
    st.warning("Please select a calculation mode.")
    st.stop()


def region_results_dataframe(
    magnitude: float | None,
    area: ScalarQuantity | None,
    band_name: Literal["G", "V"] = "G",
) -> pd.DataFrame:
    rows: list[dict[str, str | float]] = []
    for model in SKY_REGION_MODELS:
        if magnitude is not None:
            req_area: ScalarQuantity = (
                1
                / (
                    base_cumulative_density(magnitude=magnitude, band=band_name)
                    * model.density_scale
                )
            ) * u.deg**2

            diameter: ScalarQuantity = (2 * np.sqrt(req_area / np.pi)).to(u.arcmin)

            density: float = (
                base_cumulative_density(magnitude=magnitude, band=band)
                * model.density_scale
            )

            rows.append(
                {
                    "Sky regime": model.label,
                    "Stars / deg^2": density,
                    "Required area (deg^2)": req_area.to(u.deg**2).value,
                    "Equivalent diameter (arcmin)": diameter.to(u.arcmin).value,
                }
            )

        else:
            assert area is not None

            limiting_mag: float = limiting_magnitude_for_area(
                area=area,
                band=band_name,
                region=model,
            )

            density: float = (
                base_cumulative_density(magnitude=target_magnitude, band=band)
                * model.density_scale
            )

            rows.append(
                {
                    "Sky regime": model.label,
                    f"Limiting {band_name} mag": limiting_mag,
                    "Stars / deg^2": density,
                    "Expected stars in FoV": density * area.to(u.deg**2).value,
                }
            )
    return pd.DataFrame(rows)


with cols[1]:
    if mode == "Magnitude -> FoV":
        required_area: ScalarQuantity = (
            1
            / (
                base_cumulative_density(magnitude=target_magnitude, band=band)
                * selected_model.density_scale
            )
        ) * u.deg**2

        required_diameter: ScalarQuantity = (2 * np.sqrt(required_area / np.pi)).to(
            u.arcmin
        )
        density: float = (
            base_cumulative_density(magnitude=target_magnitude, band=band)
            * selected_model.density_scale
        )

        subcols = st.columns(3)
        with subcols[0]:
            st.metric(
                "Surface density",
                f"{density:.3g} stars / deg$^2$",
                help="""The expected number of stars per square degree that are at least
                as bright as the target magnitude.""",
            )
        with subcols[1]:
            st.metric(
                "Required area",
                f"{required_area.to(u.deg**2).value:.4e} deg$^2$",
                help="""The required field of view area in which at least one star
                brighter than the target magnitude is expected to always be present.""",
            )
        with subcols[2]:
            st.metric(
                "Equivalent circular diameter",
                f"{required_diameter.to(u.arcmin):.4g}",
                help="""The equivalent circular diameter of the required field of view
                area. This is the circular diameter within which there is expected to be
                at least one star brighter than the target magnitude present.""",
            )

        results_df: pd.DataFrame = region_results_dataframe(
            magnitude=target_magnitude,
            area=None,
            band_name=band,
        )

        magnitudes: NDArray = np.linspace(
            max(-2.0, target_magnitude - 3),
            min(18.0, target_magnitude + 3),
            150,
        )

        fig = go.Figure()

        for model in SKY_REGION_MODELS:
            diameters_arcmin: list[ScalarQuantity] = [
                (
                    2
                    * np.sqrt(
                        (
                            1
                            / (
                                base_cumulative_density(magnitude=mag, band=band)
                                * model.density_scale
                            )
                        )
                        * u.deg**2
                        / np.pi
                    )
                )
                .to(u.arcmin)
                .value
                for mag in magnitudes
            ]

            fig.add_trace(
                go.Scatter(
                    x=magnitudes,
                    y=diameters_arcmin,
                    mode="lines",
                    name=model.label,
                )
            )

        fig.add_trace(
            go.Scatter(
                x=[target_magnitude],
                y=[required_diameter.to(u.arcmin).value],
                mode="markers",
                name=f"Magnitude {target_magnitude:.3g} (), {required_diameter.to(u.arcmin):.4g}",
                marker={
                    "size": 20,
                    "symbol": "x",
                    "color": "white",
                    "line": {"width": 2, "color": "black"},
                },
            )
        )

        fig.update_layout(
            title="Required circular FoV diameter as a function of magnitude",
            xaxis_title=f"Magnitude limit ({band})",
            yaxis_title="Equivalent diameter (arcmin)",
            legend_title="Sky regime",
        )

    else:
        assert input_area is not None

        limiting_mag: float = limiting_magnitude_for_area(
            area=input_area,
            band=band,
            region=selected_model,
        )

        density: float = (
            base_cumulative_density(magnitude=target_magnitude, band=band)
            * selected_model.density_scale
        )

        input_diameter_equivalent: ScalarQuantity = (
            2 * np.sqrt(input_area / np.pi)
        ).to(u.arcmin)

        subcols = st.columns(3)

        with subcols[0]:
            st.metric(
                f"Limiting {band} magnitude",
                f"{limiting_mag:.3f}",
                help="""The brightest magnitude limit for which at least one star is
                expected to always appear in the field of view.""",
            )

        with subcols[1]:
            st.metric(
                "FoV area",
                f"{input_area.to(u.deg**2):.4g}",
                help="""The field of view area within which at least one star brighter
                than the limiting magnitude is expected to always be present.""",
            )

        with subcols[2]:
            st.metric(
                "Equivalent circular diameter",
                f"{input_diameter_equivalent.to(u.arcmin):.4g}",
                help="""The equivalent circular diameter of the field of view area. This
                is the circular diameter within which there is expected to be at least
                one star brighter than the limiting magnitude present.""",
            )

        st.caption(
            f"""Expected stars in selected regime:
            {density * input_area.to(u.deg**2).value:.3g}"""
        )

        results_df: pd.DataFrame = region_results_dataframe(
            magnitude=None,
            area=input_area,
            band_name=band,
        )

        magnitudes: NDArray = np.linspace(
            max(-2.0, limiting_mag - 3),
            min(18.0, limiting_mag + 3),
            150,
        )

        fig = go.Figure()
        for model in SKY_REGION_MODELS:
            expected_counts = [
                base_cumulative_density(magnitude=mag, band=band)
                * model.density_scale
                * input_area.to(u.deg**2).value
                for mag in magnitudes
            ]
            fig.add_trace(
                go.Scatter(
                    x=magnitudes,
                    y=expected_counts,
                    mode="lines",
                    name=model.label,
                )
            )

        # Draw horizontal line at y=1 to indicate the threshold for expected stars
        fig.add_hline(y=1.0, line_dash="dash", line_color="white")

        fig.add_vline(
            x=limiting_mag,
            line_dash="dash",
            line_color="white",
            annotation_text=f"Limiting mag: {limiting_mag:.3f}",
            annotation_position="top right",
            annotation_textangle=-90,
        )

        fig.update_layout(
            title="Expected stars in FoV vs magnitude limit",
            xaxis_title=f"Magnitude limit ({band})",
            yaxis_title="Expected stars in field",
            legend_title="Sky regime",
        )

    st.plotly_chart(fig, width="stretch")

    st.divider()

    st.subheader("Sky-regime comparison", anchor=False)

    st.dataframe(results_df, hide_index=True, width="stretch")
