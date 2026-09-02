#! uv run streamlit run

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from astropy import units as u

from curved_air.star_counts import (
    DEFAULT_SKY_COVERAGE_TO_REGION,
    SKY_REGION_MODELS,
    coverage_region,
    cumulative_density,
    equivalent_circular_diameter,
    limiting_magnitude_for_area,
    required_field_area,
)
from curved_air.utils import st_quantity_input

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from curved_air.type_hinting import ScalarQuantity

st.header("Guide Star Field of View", anchor=False)

with st.expander("About this tool", expanded=False):
    """
    This tool estimates the field of view required so that a telescope pointing is
    expected to contain at least one star brighter than a chosen magnitude limit.

    The reverse calculation is also available: for a chosen field of view, the tool
    estimates the brightest magnitude limit for which at least one star is expected.

    The current implementation uses a lightweight analytic cumulative star-count model
    with different density scalings for Galactic-latitude regimes. It is intended for
    rapid guider and acquisition trade studies, not as a strict all-sky catalogue proof.

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


cols = st.columns([1, 2, 2])

with cols[0]:
    mode: str | None = st.segmented_control(
        "Calculation mode",
        options=["Magnitude -> FoV", "FoV -> Magnitude"],
        default="Magnitude -> FoV",
    )

    band: Literal["G", "V"] = st.selectbox(
        "Photometric band",
        options=["G", "V"],
        index=0,
        help="G is the native model band. V is treated as slightly sparser than G.",
    )

    coverage_percent: int = st.select_slider(
        "Sky coverage target",
        options=list(DEFAULT_SKY_COVERAGE_TO_REGION.keys()),
        value=95,
        help="Maps to a conservative Galactic-latitude regime for that coverage level.",
    )

    region: str = coverage_region(coverage_percent)
    st.caption(f"Using density regime: {region}")

    st.divider()

    if mode == "Magnitude -> FoV":
        target_magnitude: float = st.number_input(
            f"Target {band}-band magnitude",
            min_value=-2.0,
            max_value=18.0,
            value=12.0,
            step=0.1,
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
                possible_units=["arcmin²", "deg²"],
                default_unit="deg²",
                key="guide_star_fov_area",
            )
        target_magnitude = 0.0

    st.info(
        "Criterion: expected star count in the field is at least 1. This is a planning"
        " estimate, not a strict catalogue-backed guarantee for every pointing."
    )


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
            req_area: ScalarQuantity = required_field_area(
                magnitude=magnitude,
                band=band_name,
                region=model.label,
            )
            diameter: ScalarQuantity = equivalent_circular_diameter(req_area)
            density: float = cumulative_density(
                magnitude, band=band_name, region=model.label
            )
            rows.append(
                {
                    "Sky regime": model.label,
                    "Stars / deg²": density,
                    "Required area (deg²)": req_area.to(u.deg**2).value,
                    "Equivalent diameter (arcmin)": diameter.to(u.arcmin).value,
                }
            )
        else:
            assert area is not None
            limiting_mag: float = limiting_magnitude_for_area(
                area=area,
                band=band_name,
                region=model.label,
            )
            density: float = cumulative_density(
                magnitude=limiting_mag,
                band=band_name,
                region=model.label,
            )
            rows.append(
                {
                    "Sky regime": model.label,
                    f"Limiting {band_name} mag": limiting_mag,
                    "Stars / deg²": density,
                    "Expected stars in FoV": density * area.to(u.deg**2).value,
                }
            )
    return pd.DataFrame(rows)


with cols[1]:
    if mode == "Magnitude -> FoV":
        required_area: ScalarQuantity = required_field_area(
            magnitude=target_magnitude,
            band=band,
            region=region,
        )
        required_diameter: ScalarQuantity = equivalent_circular_diameter(required_area)
        density: float = cumulative_density(target_magnitude, band=band, region=region)

        subcols = st.columns(3)
        with subcols[0]:
            st.metric("Surface density", f"{density:.3g} stars / deg²")
        with subcols[1]:
            st.metric("Required area", f"{required_area.to(u.deg**2):.4g}")
        with subcols[2]:
            st.metric(
                "Equivalent circular diameter",
                f"{required_diameter.to(u.arcmin):.4g}",
            )

        results_df = region_results_dataframe(
            magnitude=target_magnitude,
            area=None,
            band_name=band,
        )

        magnitudes = np.linspace(
            max(-2.0, target_magnitude - 5), min(18.0, target_magnitude + 5), 150
        )
        fig = go.Figure()
        for model in SKY_REGION_MODELS:
            diameters_arcmin: list[ScalarQuantity] = [
                equivalent_circular_diameter(
                    area=required_field_area(
                        magnitude=mag, band=band, region=model.label
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

        fig.update_layout(
            title="Required circular FoV diameter vs magnitude",
            xaxis_title=f"Magnitude limit ({band})",
            yaxis_title="Equivalent diameter (arcmin)",
            legend_title="Sky regime",
        )
    else:
        assert input_area is not None
        limiting_mag: float = limiting_magnitude_for_area(
            area=input_area,
            band=band,
            region=region,
        )
        density: float = cumulative_density(
            magnitude=limiting_mag, band=band, region=region
        )
        input_diameter_equivalent: ScalarQuantity = equivalent_circular_diameter(
            area=input_area
        )

        subcols = st.columns(3)
        with subcols[0]:
            st.metric(f"Limiting {band} magnitude", f"{limiting_mag:.3f}")
        with subcols[1]:
            st.metric("FoV area", f"{input_area.to(u.deg**2):.4g}")
        with subcols[2]:
            st.metric(
                "Equivalent circular diameter",
                f"{input_diameter_equivalent.to(u.arcmin):.4g}",
            )

        st.caption(
            f"Expected stars in selected regime: {density * input_area.to(u.deg**2).value:.3g}"
        )

        results_df = region_results_dataframe(
            magnitude=None,
            area=input_area,
            band_name=band,
        )

        magnitudes: NDArray = np.linspace(-2.0, 18.0, 150)
        fig = go.Figure()
        for model in SKY_REGION_MODELS:
            expected_counts = [
                cumulative_density(mag, band=band, region=model.label)
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

        fig.add_hline(y=1.0, line_dash="dash", line_color="black")
        fig.update_layout(
            title="Expected stars in FoV vs magnitude limit",
            xaxis_title=f"Magnitude limit ({band})",
            yaxis_title="Expected stars in field",
            legend_title="Sky regime",
        )

    st.plotly_chart(fig, width="stretch")


with cols[2]:
    st.subheader("Sky-regime comparison", anchor=False)
    st.dataframe(results_df, hide_index=True, width="stretch")

    st.subheader("Model notes", anchor=False)
    """
    - The model uses cumulative stellar surface density and the criterion $N = \\Sigma A \ge 1$.
    - The sky-coverage selector maps to a conservative Galactic-latitude regime rather than a strict all-sky proof.
    - `G` is the native model band. `V` is approximated as a modestly sparser distribution.
    - This is most suitable for rapid acquisition or guide-camera trade studies.
    """
