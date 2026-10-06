#! uv run streamlit run

# This calculator allows for calculating and generating a report on the performance of a
# rotational atmospheric dispersion corrector (ADC) design with two glass types.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd
import streamlit as st
from astropy import units as u
from matplotlib import pyplot as plt

from curved_air.glass import GLASS_CATALOGUES, Glass, GlassCatalogue
from curved_air.other import material_CTEs
from curved_air.plotting import plot_style
from curved_air.webapp import badges, bmac

if TYPE_CHECKING:
    from curved_air.type_hinting import ScalarQuantity

plt.style.use(plot_style)


st.header("CTE Match", anchor=False)

with st.expander("About this tool", expanded=False):
    f"""
    Use this tool to find materials with a coefficient of thermal expansion (CTE) that
    matches that of a given optical glass. This is useful for selecting materials for
    optomechanical components such as lens mounts, where a good CTE match can help to
    reduce stress on the optics and improve the overall performance of sensitive systems.

    All code is written by Jake Pember {badges}

    Please get in contact if you have any questions, suggestions, or find any bugs!

    You can also [Buy me a pint](https://buymeacoffee.com/pemberj) if you find this tool
    useful and want to support its development!
    """

    bmac()


@dataclass
class Material:
    """
    Typically CTE is only specified as a single value for materials, however to match
    the way glasses are specified in many glass catalogues this class allows for storing
    several CTE values corresponding to different temperature ranges.
    """

    name: str
    cte_by_range: dict[str, ScalarQuantity]

    @classmethod
    def single_cte(cls, name: str, cte: ScalarQuantity) -> Material:
        """
        Create a Material with a single CTE value for all temperature ranges.
        """
        return cls(
            name=name,
            cte_by_range={
                "-30°C to +70°C": cte,
                "+20°C to +300°C": cte,
            },
        )

    def cte(self, temperature_range: str) -> ScalarQuantity:
        return self.cte_by_range.get(temperature_range, 0 / u.K)


materials: dict[str, Material] = {
    name: Material.single_cte(name, cte) for name, cte in material_CTEs.items()
}

# Example definition of CTE data with multiple values:
# materials["Aluminium 6061"] = Material(
#     name="Aluminium 6061",
#     cte_by_range={
#         "-30°C to +70°C": 23.6e-6 / u.K,
#         "+20°C to +300°C": 24.0e-6 / u.K,
#     },
# )

cols = st.columns([1, 2])

with cols[0]:
    st.subheader("Select Glass", anchor=False)

    catalogue: str = st.selectbox(
        "Glass Catalogue",
        GLASS_CATALOGUES.keys(),
        index=list(GLASS_CATALOGUES.keys()).index("Schott")
        if "Schott" in GLASS_CATALOGUES
        else 0,
        width=300,
    )

    glasses = GlassCatalogue(catalogue)
    glass_names: list[str] = sorted(glasses.keys())

    try:
        default_index: int = glass_names.index("N-BK7")
    except ValueError:
        default_index = 0

    glass_str: str = st.selectbox("Glass", glass_names, index=default_index, width=300)
    glass: Glass = glasses[glass_str]

    temperature_range: str | None = st.segmented_control(
        "Temperature Range for CTE",
        options=["-30°C to +70°C", "+20°C to +300°C"],
        default="-30°C to +70°C",
        help="""The temperature range over which the CTE is calculated. Different
        materials may have different CTE values depending on the temperature range, so
        it's important to select the appropriate range for your application.""",
    )

    try:
        assert temperature_range is not None
    except AssertionError:
        st.warning("Please select a temperature range for the CTE.")
        st.stop()
    assert temperature_range is not None

    glass_cte: ScalarQuantity = (
        glass.cte1 if temperature_range == "-30°C to +70°C" else glass.cte2
    ) * (1e-6 / u.K)

    glass_cte_value_formatted: str = (
        "$"
        + f"{glass_cte.value:.2e}".split("e")[0]
        + "×10^"  # noqa: RUF001
        + "{"
        + f"{int(f'{glass_cte.value:.2e}'.split('e')[1])}"
        + "}"
        + "$"
    )

    st.metric(
        "Glass CTE",
        f"{glass_cte_value_formatted} per K",
        help="""The coefficient of thermal expansion (CTE) of the selected glass, in
        units of 1/K. This value is used to compare against the CTE of the metals to
        find good matches.""",
    )

    if glass_cte.value == 0:
        st.warning(
            """The CTE value of the selected glass in this range is zero, which likely
            means no data is available."""
        )

    st.toggle("Override CTE value", key="override_cte")

    if st.session_state["override_cte"]:
        custom_cte: float = st.number_input(
            "Custom CTE (1e-6/K)",
            min_value=0.0,
            value=float(glass_cte.value * 1e6),
            step=0.01,
            width=300,
            help="""Override the CTE value of the glass with a custom value. This can be
            useful if you have specific data for the CTE of the glass at a certain
            temperature range that differs from the default values provided.""",
        )
        glass_cte: ScalarQuantity = custom_cte * 1e-6 / u.K


with cols[1]:
    # List matches
    rows: list[dict] = []

    for material in materials.values():
        material_cte = material.cte(temperature_range)
        cte_diff = abs(material_cte - glass_cte)
        rows.append(
            {
                "Material": material.name,
                "CTE (1e-6/K)": float(material_cte.value) * 1e6,
                "Difference (1e-6/K)": float(cte_diff.value) * 1e6,
                "Δx across 100mm for ΔT=1K": f"{(cte_diff * 1 * u.K * 100 * u.mm).to(u.um):.2f}",
            }
        )

    df: pd.DataFrame = (
        pd.DataFrame(rows).sort_values("Difference (1e-6/K)").reset_index(drop=True)
    )

    def highlight_difference(row: pd.Series) -> list[str]:
        difference: float = float(row["Difference (1e-6/K)"])
        if difference <= 2:  # noqa: PLR2004
            return ["background-color: #00ff001a"] * len(row)
        # if 2 < difference < 7:  # noqa: PLR2004
        # return ["background-color: #ffff001a"] * len(row)
        if difference >= 7:  # noqa: PLR2004
            return ["background-color: #ff00001a"] * len(row)
        return [""] * len(row)

    styled_df = df.style.apply(highlight_difference, axis=1).format(
        {
            "CTE (1e-6/K)": lambda x: f"{x:.4g}",
            "Difference (1e-6/K)": lambda x: f"{x:.4g}",
        }
    )

    st.subheader("Best Matches", anchor=False)
    st.dataframe(styled_df, width="stretch", height=500, hide_index=True)
