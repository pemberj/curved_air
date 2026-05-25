#! uv run streamlit run

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

from curved_air.type_hinting import ScalarQuantity
from curved_air.utils import st_quantity_input

DESIGNS: list[str] = [
    "Newtonian",
    "Classic Cassegrain",
    "Ritchey-Chrétien",
    "Dall-Kirkham",
    "Gregorian",
]

CONIC_NOTES: dict[str, str] = {
    "Newtonian": "Parabolic primary (K1 = -1), flat secondary.",
    "Classic Cassegrain": "Parabolic primary (K1 = -1), hyperbolic convex secondary.",
    "Ritchey-Chrétien": "Both mirrors are hyperboloids; corrects for coma (aplanatic).",
    "Dall-Kirkham": "Elliptical primary, spherical secondary (K2 = 0). Not aplanatic.",
    "Gregorian": "Parabolic primary (K1 = -1), elliptical concave secondary.",
}


@dataclass
class Mirror:
    diameter: ScalarQuantity | None = None
    radius: ScalarQuantity | None = None
    conic: float | None = None

    @property
    def focal_length(self) -> ScalarQuantity | None:
        if self.radius is not None:
            return self.radius / 2
        return None

    @property
    def f_number(self) -> float | None:
        if self.focal_length is not None:
            return (self.focal_length / self.diameter).si.value
        return None


@dataclass
class TelescopeDesign:
    name: str
    design: str
    M1: Mirror
    M2: Mirror
    system_focal_length: ScalarQuantity
    back_focal_distance: ScalarQuantity
    separation: ScalarQuantity | None = None
    magnification: float | None = None

    @property
    def system_f_number(self) -> float:
        return (self.system_focal_length / self.M1.diameter).si.value

    @property
    def central_obstruction(self) -> float | None:
        if self.M2.diameter is not None:
            return 1 - (self.M2.diameter / self.M1.diameter).si.value
        return None

    def __post_init__(self) -> None:
        if self.design not in DESIGNS:
            raise ValueError(
                f"Invalid design: {self.design}. Must be one of {DESIGNS}."
            )

        assert isinstance(self.M1.diameter, ScalarQuantity), (
            "Primary diameter must be specified."
        )
        assert isinstance(self.M1.radius, ScalarQuantity), (
            "Primary radius must be specified."
        )

        assert self.system_focal_length >= self.M1.focal_length, (
            "System focal length must be greater than primary focal length."
        )

        if self.M1.focal_length is not None:
            self.M1.radius: ScalarQuantity = 2 * self.M1.focal_length

        self.magnification: float = (
            self.system_focal_length / self.M1.focal_length
        ).si.value

        self.compute_prescription()

    def compute_prescription(self) -> TelescopeDesign:
        """
        Compute a two-mirror telescope prescription
        """

        assert self.magnification is not None

        if self.design == "Newtonian":
            self.M1.conic = -1
            self.M2.radius: float = np.inf
            self.M2.conic = 0
            self.separation: ScalarQuantity = (
                self.M1.focal_length - 1 * self.back_focal_distance
            )
            self.M2.diameter: ScalarQuantity = (
                self.M1.diameter * self.separation / self.M1.focal_length
            )
            return self

        # First, compute the mirror separation and secondary focal length
        if self.design == "Gregorian":
            self.separation: ScalarQuantity = (
                self.system_focal_length + self.back_focal_distance
            ) / (self.magnification - 1)

            self.M2.radius: ScalarQuantity = 2 * (
                self.magnification * self.separation / (self.magnification + 1)
            )

        else:
            self.separation: ScalarQuantity = (
                self.system_focal_length - self.back_focal_distance
            ) / (self.magnification + 1)

            self.M2.radius: ScalarQuantity = 2 * (
                self.magnification * self.separation / (self.magnification - 1)
            )

        self.M2.diameter: ScalarQuantity = (
            self.M1.diameter * self.separation / self.M1.focal_length
        )

        # Then compute the conic constants based on the design type
        if self.design == "Classic Cassegrain":
            self.M1.conic = -1
            self.eccentricity: float = (self.magnification + 1) / (
                self.magnification - 1
            )
            self.M2.conic: float = -(self.eccentricity**2)

        elif self.design == "Ritchey-Chrétien":
            self.back_focus_ratio: float = (
                self.back_focal_distance / self.M1.focal_length
            ).si.value
            self.M1.conic: float = (
                -1 - (2 / self.magnification**3) * self.back_focus_ratio
            )
            self.M2.conic: float = -(
                ((self.magnification + 1) / (self.magnification - 1)) ** 2
            ) - (2.0 * self.magnification * (self.magnification + 1)) / (
                (self.magnification - 1) ** 3 * self.back_focus_ratio
            )

        elif self.design == "Dall-Kirkham":
            self.M1.conic: float = -(
                ((self.magnification - 1) / (self.magnification + 1)) ** 2
            )
            self.M2.conic = 0

        elif self.design == "Gregorian":
            self.M1.conic = -1
            self.eccentricity: float = (self.magnification + 1) / (
                self.magnification - 1
            )
            self.M2.conic: float = -(self.eccentricity**2)
        else:
            self.M1.conic: float = np.nan
            self.M2.conic: float = np.nan

        return self


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

st.header("Telescope Designer")

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

st.divider()

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

# try:
telescope: TelescopeDesign = TelescopeDesign(
    name=f"{design} D={diameter:.1f} F={system_focal_length:.1f}",
    design=design,
    M1=Mirror(diameter=diameter, radius=2 * primary_focal_length),
    M2=Mirror(),
    system_focal_length=system_focal_length,
    back_focal_distance=back_focal_distance,
)

# except Exception as exc:
#     st.error(f"Calculation error: {exc}")
#     st.stop()

with cols[1]:
    metric_cols = st.columns(2)

    with metric_cols[0]:
        st.metric(
            "System focal length", f"{telescope.system_focal_length:.6g}", border=True
        )

    with metric_cols[1]:
        st.metric("System f/number", f"{telescope.system_f_number:.4g}", border=True)

    with metric_cols[0]:
        st.metric(
            "Primary-secondary separation", f"{telescope.separation:.4g}", border=True
        )

    with metric_cols[1]:
        st.metric(
            "Central obstruction", f"{telescope.central_obstruction:.3g}", border=True
        )

    with metric_cols[0]:
        st.metric("Primary radius R1", f"{telescope.M1.radius:.6g}", border=True)

    with metric_cols[1]:
        st.metric("Primary conic K1", f"{telescope.M1.conic:.4g}", border=True)

    with metric_cols[0]:
        st.metric("Secondary radius R2", f"{telescope.M2.radius:.4g}", border=True)

    with metric_cols[1]:
        st.metric("Secondary conic K2", f"{telescope.M2.conic:.4g}", border=True)

st.divider()

table_cols = st.columns(3)
with table_cols[0]:
    render_table(
        "System Parameters",
        [
            ("System focal length", f"{telescope.system_focal_length:.5g}"),
            ("System f/#", f"{telescope.system_f_number:.4g}"),
            ("Primary-secondary separation d", f"{telescope.separation:.4g}"),
            ("Central obstruction fraction", f"{telescope.central_obstruction:.4g}"),
            ("Back focal distance", f"{telescope.back_focal_distance:.4g}"),
        ],
    )

with table_cols[1]:
    render_table(
        "Primary Mirror",
        [
            ("Diameter D1", f"{telescope.M1.diameter:.4g}"),
            ("Radius of curvature R1", f"{telescope.M1.radius:.4g}"),
            ("Conic constant K1", f"{telescope.M1.conic:.4g}"),
            ("Focal length f1", f"{telescope.M1.focal_length:.4g}"),
            ("Primary f/number", f"{telescope.M1.f_number:.4g}"),
        ],
    )
with table_cols[2]:
    render_table(
        "Secondary Mirror",
        [
            ("Diameter D2", f"{telescope.M2.diameter:.4g}"),
            ("Radius of curvature R2", f"{telescope.M2.radius:.4g}"),
            ("Conic constant K2", f"{telescope.M2.conic:.4g}"),
            ("Focal length f2", f"{telescope.M2.focal_length:.4g}"),
        ],
    )
