#! uv run streamlit run

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from astropy import units as u
from astropy.constants import G, M_sun  # ty:ignore[unresolved-import]

from curved_air.utils import st_quantity_input

if TYPE_CHECKING:
    from curved_air.type_hinting import ArrayQuantity, ScalarQuantity


NUM_TIMESTEPS = 500
DEFAULT_TIME_RANGE = (0.0, 16.0)  # years


@dataclass
class Orbit:
    period: ScalarQuantity = 1 * u.yr
    inclination: ScalarQuantity = 90 * u.deg
    eccentricity: ScalarQuantity | float = 0
    phase: ScalarQuantity = random.uniform(0, 2 * np.pi) * u.rad  # noqa: S311

    def __post_init__(self) -> None:
        if not isinstance(self.eccentricity, u.Quantity):
            self.eccentricity = self.eccentricity * u.dimensionless_unscaled

    @property
    def semimajor_axis(self) -> ScalarQuantity:
        """Calculate the semi-major axis of the orbit using Kepler's third law."""
        return ((self.period**2 * G * M_sun / (4 * np.pi**2)) ** (1 / 3)).to(u.au)


@dataclass
class Planet:
    name: str | None = None
    mass: ScalarQuantity = 1 * u.M_earth
    orbit: Orbit = field(default_factory=Orbit)

    @property
    def period(self) -> ScalarQuantity:
        return self.orbit.period

    @property
    def inclination(self) -> ScalarQuantity:
        return self.orbit.inclination

    @property
    def i(self) -> ScalarQuantity:
        return self.orbit.inclination

    @property
    def eccentricity(self) -> ScalarQuantity | float:
        return self.orbit.eccentricity

    @property
    def semimajor_axis(self) -> ScalarQuantity:
        return self.orbit.semimajor_axis


solar_system_planets: list[Planet] = [
    Planet(
        name="Mercury",
        mass=0.055 * u.M_earth,
        orbit=Orbit(
            period=0.241 * u.yr,
            inclination=83 * u.deg,
            eccentricity=0.206,
            phase=0 * u.rad,
        ),
    ),
    Planet(
        name="Venus",
        mass=0.815 * u.M_earth,
        orbit=Orbit(
            period=0.615 * u.yr,
            inclination=86.6 * u.deg,
            eccentricity=0.007,
            phase=0 * u.rad,
        ),
    ),
    Planet(
        name="Earth",
        mass=1 * u.M_earth,
        orbit=Orbit(
            period=1 * u.yr,
            inclination=90 * u.deg,
            eccentricity=0.017,
            phase=0 * u.rad,
        ),
    ),
    Planet(
        name="Mars",
        mass=0.107 * u.M_earth,
        orbit=Orbit(
            period=1.88 * u.yr,
            inclination=88.15 * u.deg,
            eccentricity=0.093,
            phase=0 * u.rad,
        ),
    ),
    Planet(
        name="Jupiter",
        mass=1 * u.M_jup,
        orbit=Orbit(
            period=11.86 * u.yr,
            inclination=88.7 * u.deg,
            eccentricity=0.049,
            phase=0 * u.rad,
        ),
    ),
    Planet(
        name="Saturn",
        mass=95.2 * u.M_earth,
        orbit=Orbit(
            period=29.46 * u.yr,
            inclination=87.51 * u.deg,
            eccentricity=0.056,
            phase=0 * u.rad,
        ),
    ),
    Planet(
        name="Uranus",
        mass=14.5 * u.M_earth,
        orbit=Orbit(
            period=84.01 * u.yr,
            inclination=89.23 * u.deg,
            eccentricity=0.046,
            phase=0 * u.rad,
        ),
    ),
    Planet(
        name="Neptune",
        mass=17.1 * u.M_earth,
        orbit=Orbit(
            period=164.8 * u.yr,
            inclination=88.23 * u.deg,
            eccentricity=0.010,
            phase=0 * u.rad,
        ),
    ),
]


@dataclass
class Star:
    mass: ScalarQuantity = 1 * u.M_sun
    radius: ScalarQuantity = 1 * u.R_sun


@dataclass
class System:
    star: Star = field(default_factory=Star)
    planets: list[Planet] = field(default_factory=list)

    def radial_velocity(self, time: ArrayQuantity) -> ArrayQuantity:
        """Calculate the radial velocity of the system at a given time."""

        rv = np.zeros_like(time.value) * u.m / u.s

        for planet in self.planets:
            rv += self.radial_velocity_planet(planet, time)

        return rv

    def radial_velocity_planet(
        self,
        planet: Planet,
        time: ArrayQuantity,
    ) -> ArrayQuantity:
        """Calculate the radial velocity of a planet at a given time."""

        mean_anomaly = (
            2 * np.pi * (time.to(u.yr) / planet.period.to(u.yr)).si.value
        ) + planet.orbit.phase.to(u.rad).value

        # Calculate the eccentric anomaly using Newton's method.
        eccentric_anomaly = (
            mean_anomaly + planet.eccentricity * np.sin(mean_anomaly)
        ).si.value

        for _ in range(10):
            eccentric_anomaly = (
                mean_anomaly + planet.eccentricity * np.sin(eccentric_anomaly)
            ).si.value

        # Calculate the true anomaly.
        theta = (
            (
                2
                * np.arctan2(
                    np.sqrt(1 + planet.eccentricity) * np.sin(eccentric_anomaly / 2),
                    np.sqrt(1 - planet.eccentricity) * np.cos(eccentric_anomaly / 2),
                )
            )
            .to(u.rad)
            .value
        )

        K = (
            np.sqrt(G / (1 - planet.eccentricity**2))
            * planet.mass
            * np.sin(planet.inclination)
            * (1 / np.sqrt(self.star.mass + planet.mass))
            * (1 / np.sqrt(planet.semimajor_axis))
        ).to(u.m / u.s)

        # Calculate the radial velocity.
        return K * np.sin(planet.i.to(u.rad).value) * np.cos(theta + np.pi / 2)


def initialise_planets() -> None:
    if "planets" not in st.session_state:
        st.session_state.planets: list[Planet] = solar_system_planets.copy()

    if "selected_planet_index" not in st.session_state:
        st.session_state.selected_planet_index = 0

    if st.session_state.selected_planet_index >= len(st.session_state.planets):
        st.session_state.selected_planet_index = len(st.session_state.planets) - 1


def add_planet() -> None:
    st.session_state.planets.append(Planet())
    st.session_state.selected_planet_index = len(st.session_state.planets) - 1


def delete_selected_planet() -> None:
    if len(st.session_state.planets) == 0:
        return

    delete_planet(st.session_state.selected_planet_index)


def delete_planet(index: int) -> None:
    if not 0 <= index < len(st.session_state.planets):
        return

    del st.session_state.planets[index]

    if not st.session_state.planets:
        st.session_state.planets = [Planet()]
        st.session_state.selected_planet_index = 0
        return

    if index < st.session_state.selected_planet_index:
        st.session_state.selected_planet_index -= 1
    elif index == st.session_state.selected_planet_index:
        st.session_state.selected_planet_index = min(
            index, len(st.session_state.planets) - 1
        )


def clear_all_planets() -> None:
    st.session_state.planets: list[Planet] = []
    st.session_state.selected_planet_index = 0


def select_planet(index: int) -> None:
    if 0 <= index < len(st.session_state.planets):
        st.session_state.selected_planet_index = index


def planet_summary(planet: Planet) -> str:
    return f"""{planet.mass.to(u.M_earth).value:.3g} M_e | P={planet.period.to(u.yr):.3g} | a={planet.semimajor_axis.to(u.au):.3g} | e={planet.eccentricity.value:.3g}"""


st.header("Keplerian Signals", anchor=False)

with st.expander("About this tool", expanded=False):
    """
    You can use this tool to visualise the radial velocity signals from planets on
    Keplerian orbits.

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


initialise_planets()

planets, config, plot = st.columns([1, 1, 2.5])

with planets:
    st.subheader("Planets")

    for index, planet in enumerate(st.session_state.planets):
        with st.container(border=True):  # noqa: SIM117
            row = st.columns([4, 1, 1])
            with row[0]:
                f"{planet.name}" if planet.name else f"Planet {index + 1}"
                st.caption(planet_summary(planet))
            if len(st.session_state.planets) > 1:
                with row[1]:
                    if index != st.session_state.selected_planet_index:
                        button_label = "Select"
                        st.button(
                            "Select",
                            key=f"select_planet_{index}",
                            type="primary",
                            on_click=select_planet,
                            args=(index,),
                        )
                    else:
                        st.button(
                            "Editing",
                            key=f"select_planet_{index}",
                            type="secondary",
                            disabled=True,
                        )
                with row[2]:
                    st.button(
                        "Delete",
                        key=f"delete_planet_{index}",
                        type="tertiary",
                        on_click=delete_planet,
                        args=(index,),
                    )

    with st.container(width=500):
        cols = st.columns([1, 1])

        with cols[0], st.container(horizontal_alignment="right"):
            st.button(
                "Add Planet",
                on_click=add_planet,
                type="primary",
            )

        with cols[1], st.container(horizontal_alignment="left"):
            st.button(
                "Clear All Planets",
                on_click=clear_all_planets,
                type="tertiary",
            )

with config:
    if not st.session_state.planets:
        st.info("Add a planet to begin.")
    else:
        planet: Planet = st.session_state.planets[
            st.session_state.selected_planet_index
        ]
        st.subheader("Edit planet properties")

        with st.container(border=True):
            planet.name = st.text_input(
                "Name",
                value=planet.name
                or f"Planet {st.session_state.selected_planet_index + 1}",
                key=f"planet_{st.session_state.selected_planet_index}_name",
            )

            planet.mass: ScalarQuantity = st_quantity_input(
                "Mass",
                possible_units=["kg", "M_earth", "M_jup", "M_sun"],
                value=planet.mass.to(u.M_earth).value,
                min_value=0.01,
                step=0.1,
                default_unit="M_earth",
                format="%.3g",
                key=f"planet_{st.session_state.selected_planet_index}_mass",
            )

            planet.orbit.period: ScalarQuantity = st_quantity_input(
                "Period",
                possible_units=["h", "d", "yr"],
                value=planet.orbit.period.to(u.yr).value,
                min_value=0.1,
                step=0.1,
                default_unit="yr",
                format="%.3g",
                key=f"planet_{st.session_state.selected_planet_index}_period",
            )

            planet.orbit.inclination: ScalarQuantity = st_quantity_input(
                "Inclination",
                possible_units=["deg", "rad"],
                value=planet.orbit.inclination.to(u.deg).value,
                default_unit="deg",
                format="%.3g",
                key=f"planet_{st.session_state.selected_planet_index}_inclination",
            )

            planet.orbit.eccentricity: float = (
                st.number_input(
                    "Eccentricity",
                    min_value=0.0,
                    max_value=0.99,
                    value=float(planet.orbit.eccentricity.value),
                    step=0.01,
                    format="%.3g",
                    width=200,
                    key=f"planet_{st.session_state.selected_planet_index}_eccentricity",
                )
                * u.dimensionless_unscaled
            )

            planet.orbit.phase: ScalarQuantity = st_quantity_input(
                "Phase (Randomly initialised)",
                possible_units=["deg", "rad"],
                value=planet.orbit.phase.to(u.deg).value,
                default_unit="deg",
                format="%.3g",
                key=f"planet_{st.session_state.selected_planet_index}_phase",
            )

with plot:
    st.subheader("Radial velocity")

    time: ArrayQuantity = (
        np.linspace(
            st.session_state.get("time_range", DEFAULT_TIME_RANGE)[0],
            st.session_state.get("time_range", DEFAULT_TIME_RANGE)[1],
            NUM_TIMESTEPS,
        )
        * u.yr
    )

    if st.session_state.get("simulate_observations", False):
        # Simulate observations with random samples and noise
        num_observations = 50
        observation_times = (
            np.sort(
                np.random.uniform(
                    st.session_state.get("time_range", DEFAULT_TIME_RANGE)[0],
                    st.session_state.get("time_range", DEFAULT_TIME_RANGE)[1],
                    num_observations,
                )
            )
            * u.yr
        )

    system = System(planets=st.session_state.planets)

    if st.session_state.get("simulate_observations", False):
        # Add noise to the radial velocity signal
        noise_std = st.session_state.get("measurement_precision", 1.0) * u.m / u.s
        noise = (
            np.random.normal(
                0,
                noise_std.to(u.m / u.s).value,
                size=observation_times.shape,
            )
            * u.m
            / u.s
        )

    fig = go.Figure()

    if st.session_state.get("show_individual_planet_signals", True):
        for index, planet in enumerate(st.session_state.planets):
            fig.add_trace(
                go.Scatter(
                    x=time.to(u.yr).value,
                    y=system.radial_velocity_planet(planet, time).to(u.m / u.s).value,
                    mode="lines",
                    opacity=0.5,
                    name=f"{planet.name}" if planet.name else f"Planet {index + 1}",
                )
            )

    fig.add_trace(
        go.Scatter(
            x=time.to(u.yr).value,
            y=system.radial_velocity(time).to(u.m / u.s).value,
            mode="lines",
            name="System radial velocity",
        )
    )

    if st.session_state.get("simulate_observations", False):
        fig.add_trace(
            go.Scatter(
                x=observation_times.to(u.yr).value,
                y=(system.radial_velocity(observation_times) + noise)
                .to(u.m / u.s)
                .value,
                error_y={
                    "type": "data",
                    "array": [st.session_state.get("measurement_precision", 1.0)]
                    * len(observation_times),
                    "color": "gray",
                    "visible": True,
                },
                mode="markers",
                marker={"color": "gray", "size": 6},
                name="Simulated observations",
            )
        )

    fig.update_layout(
        xaxis_title="Time (years)",
        yaxis_title="Radial velocity (m/s)",
        template="plotly_white",
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
    )
    st.plotly_chart(fig, height=600, width="stretch")

    with st.container(horizontal_alignment="center"):  # noqa: SIM117
        with st.container(width=650):
            st.slider(
                "Time range (years)",
                min_value=0.0,
                max_value=30.0,
                value=DEFAULT_TIME_RANGE,
                step=0.1,
                format="%.3g",
                key="time_range",
            )

            cols = st.columns([1, 1, 2])

            with cols[0]:
                st.toggle(
                    "Show individual planet signals",
                    value=True,
                    key="show_individual_planet_signals",
                )

            with cols[1]:
                st.toggle(
                    "Simulate observations",
                    help="""If checked, the system radial velocity signal will be sampled at
                    random times and noise will be added to simulate real observations.""",
                    key="simulate_observations",
                )

            with cols[2]:
                if st.session_state.get("simulate_observations", False):
                    st.number_input(
                        "Measurement Precision (m/s)",
                        min_value=0.01,
                        value=1.0,
                        step=0.01,
                        format="%.2g",
                        width=200,
                        key="measurement_precision",
                    )
