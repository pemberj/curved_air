"""
curved_air

coatings

This module contains the `Coating` class, which represents an optical coating with its
properties and methods, as well as functions to parse coating reflectivity data from
Excel spreadsheet files that are available from optics suppliers such as Thorlas and
Edmund Optics.

TODO: Handle HR coatings. In Thorlabs data files the S- and P-polarised reflectivity
curves are separate, and any different incidence angles are also separated. How to
handle this?

TODO: Grab more Edmund Optics coating files. I have none for HR coatings yet
"""

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from numpy.typing import ArrayLike

from curved_air.plotting import plot_style
from curved_air.type_hinting import ArrayQuantity, ScalarQuantity

plt.style.use(plot_style)

COATING_DATA: Path = Path(__file__).parent / "data" / "coatings"


@dataclass
class Coating:
    name: str
    wavelengths: ArrayQuantity | None = None
    reflectivity: float | ArrayLike | None = None
    transmission: float | ArrayLike | None = None
    valid_wavelength_range: tuple[ScalarQuantity, ScalarQuantity] = field(
        default=(380 * u.nm, 950 * u.nm)
    )
    reference: str | None = None

    def min_R(self) -> float:
        """The minimum reflectivity of the coating."""
        if self.reflectivity is None:
            raise ValueError("Reflectivity data is not available for this coating.")
        return np.min(self.reflectivity)

    def max_R(self) -> float:
        """The maximum reflectivity of the coating."""
        if self.reflectivity is None:
            raise ValueError("Reflectivity data is not available for this coating.")
        return np.max(self.reflectivity)

    def min_T(self) -> float:
        """The minimum transmission of the coating."""
        if self.transmission is None:
            raise ValueError("Transmission data is not available for this coating.")
        return np.min(self.transmission)

    def max_T(self) -> float:
        """The maximum transmission of the coating."""
        if self.transmission is None:
            raise ValueError("Transmission data is not available for this coating.")
        return np.max(self.transmission)

    def plot(self, ax: plt.Axes | None = None) -> plt.Figure | None:
        """Plot the reflectivity and transmission of the coating."""

        if ax is None:
            fig: plt.Figure = plt.figure(figsize=(8, 5))
            ax: plt.Axes = fig.gca()
            return_fig: bool = True
        else:
            return_fig: bool = False

        if self.reflectivity is not None:
            ax.plot(
                self.wavelengths.to(u.nm),
                self.reflectivity,
                label=f"{self.name}",
            )
        if self.transmission is not None:
            ax.plot(
                self.wavelengths.to(u.nm),
                self.transmission,
                label=f"{self.name}",
            )
        ax.set_xlabel("Wavelength (nm)")
        ylabel = ""
        if self.reflectivity is not None:
            ylabel += "Reflectivity"
        if self.transmission is not None:
            if ylabel:
                ylabel += " / "
            ylabel += "Transmission"
        ax.set_ylabel(ylabel)

        ax.legend()

        if return_fig:
            return fig

        return None


@dataclass
class ARCoating(Coating):
    """A class representing an anti-reflection coating."""


@dataclass
class HRCoating(Coating):
    """A class representing a high-reflection coating."""


def parse_edmund_optics_ar_coating_xlsx(file: Path) -> Coating:

    df: pd.DataFrame = pd.read_excel(file, sheet_name="Chart Data")
    # get the first row of values
    # header = df.iloc[0]
    line1 = df.iloc[1]

    coating_name = f"Edmund Optics {line1['CurveName'].split('Coating Curve')[0]}"

    wavelength_data: str = df["XData"]
    wavelength_unit = u.Unit(line1["XColumn"].split("(")[-1].strip(")"))

    reflectivity_data: str = df["YData"]
    reflectivity_unit: str = (
        "percent" if "%" in line1["YColumn"].lower() else "fraction"
    )

    return ARCoating(
        name=coating_name,
        wavelengths=wavelength_data.values * wavelength_unit,
        reflectivity=reflectivity_data.values
        / (100.0 if reflectivity_unit == "percent" else 1.0),
        valid_wavelength_range=(
            np.min(wavelength_data.values) * wavelength_unit,
            np.max(wavelength_data.values) * wavelength_unit,
        ),
        reference="Edmund Optics",
    )


def parse_thorlabs_ar_coating_xlsx(file: Path) -> Coating:

    _df = pd.read_excel(file, skiprows=7, nrows=1)
    coating_name = f"Thorlabs {_df.columns[0].replace(' - ', '-').replace('µ', 'u').split('AR Coating')[0].split('Broadband')[0].split(' V ')[0]}"

    try:
        reflectance_df: pd.DataFrame = pd.read_excel(
            file, sheet_name="Reflectance", skiprows=1
        )

        try:
            wavelength_data: str = reflectance_df["Wavelength (nm)"]
            wavelength_unit: u.Unit = u.nm
        except KeyError:
            wavelength_data: str = reflectance_df["Wavelength (µm)"]
            wavelength_unit: u.Unit = u.um

        try:
            reflectivity_data: str = reflectance_df["Reflectance (%)"]
        except KeyError:
            reflectivity_data: str = reflectance_df["% Reflectance"]
        reflectivity_unit: str = "percent"
    except Exception as e:
        print(f"Error reading Thorlabs coating data from {file}: {e}")
        return Coating(name=coating_name)

    return ARCoating(
        name=coating_name,
        wavelengths=wavelength_data.values * wavelength_unit,
        reflectivity=reflectivity_data.values
        / (100.0 if reflectivity_unit == "percent" else 1.0),
        valid_wavelength_range=(
            np.min(wavelength_data.values) * wavelength_unit,
            np.max(wavelength_data.values) * wavelength_unit,
        ),
        reference="https://www.thorlabs.com/optical-coatings",
    )


EDMUND_OPTICS_AR: list[Coating] = [
    parse_edmund_optics_ar_coating_xlsx(file)
    for file in list(COATING_DATA.glob("Edmund Optics/AR/EO_*.xlsx"))
]

THORLABS_AR: list[Coating] = [
    parse_thorlabs_ar_coating_xlsx(file)
    for file in list(COATING_DATA.glob("Thorlabs/AR/THORLABS_*.xlsx"))
]


def test() -> None:

    fig = plt.figure()
    ax = fig.gca()

    for coating in EDMUND_OPTICS_AR:
        coating.plot(ax=ax)

    for coating in THORLABS_AR:
        coating.plot(ax=ax)

    plt.show()


if __name__ == "__main__":
    test()
