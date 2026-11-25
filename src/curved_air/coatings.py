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
from typing import overload

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from astropy import units as u
from numpy.typing import ArrayLike, NDArray

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


@dataclass
class ComplexIndexMaterial:
    """
    A class representing a metal coating material with a complex-valued refractive index
    """

    name: str
    wavelengths: ArrayQuantity | None = None
    refractive_index_n: float | NDArray[np.floating] | None = None
    refractive_index_k: float | NDArray[np.floating] | None = None
    valid_wavelength_range: tuple[ScalarQuantity, ScalarQuantity] = field(
        default=(380 * u.nm, 950 * u.nm)
    )
    reference: str | None = None
    comments: str | None = None

    def parse_file(self, file: Path) -> None:
        """
        Parse a data file containing the complex refractive index of the material.

        YAML files from refractiveindex.info (`Full database record' files) can be
        loaded and parsed with this function. Tabulated data is supported, either as a
        single `tabulated nk' block or as separate `tabulated n' and `tabulated k'
        blocks (k is interpolated onto the n wavelength grid if they differ).

        Wavelengths in these files are assumed to be in microns.
        """

        file = Path(file)
        if file.suffix.lower() not in (".yml", ".yaml"):
            raise ValueError(f"Unsupported file type: {file.suffix}")

        with file.open("r", encoding="utf-8") as f:
            record: dict = yaml.safe_load(f)

        n_data: np.ndarray | None = None
        k_data: np.ndarray | None = None

        for block in record.get("DATA", []):
            data_type: str = block["type"].strip()
            if not data_type.startswith("tabulated"):
                raise NotImplementedError(
                    f"{file.name}: data type '{data_type}' is not supported "
                    "(only tabulated data can be parsed)"
                )
            table: NDArray[np.floating] = np.loadtxt(
                block["data"].strip().splitlines(), ndmin=2
            )

            if data_type == "tabulated nk":
                n_data: NDArray[np.floating] = table[:, [0, 1]]
                k_data: NDArray[np.floating] = table[:, [0, 2]]

            elif data_type == "tabulated n":
                n_data: NDArray[np.floating] = table[:, [0, 1]]

            elif data_type == "tabulated k":
                k_data: NDArray[np.floating] = table[:, [0, 1]]

            else:
                raise NotImplementedError(
                    f"{file.name}: unknown data type '{data_type}'"
                )

        if n_data is None:
            raise ValueError(f"{file.name}: no refractive index (n) data found!")

        wavelengths_um: NDArray[np.floating] = n_data[:, 0]

        if k_data is None:
            k: NDArray[np.floating] = np.zeros_like(wavelengths_um)

        elif np.array_equal(k_data[:, 0], wavelengths_um):
            k: NDArray[np.floating] = k_data[:, 1]

        else:
            k: NDArray[np.floating] = np.interp(
                wavelengths_um, k_data[:, 0], k_data[:, 1]
            )

        self.wavelengths: ArrayQuantity = wavelengths_um * u.um
        self.refractive_index_n: NDArray[np.floating] = n_data[:, 1]
        self.refractive_index_k: NDArray[np.floating] = k
        self.valid_wavelength_range: tuple[ScalarQuantity, ScalarQuantity] = (
            np.min(wavelengths_um) * u.um,
            np.max(wavelengths_um) * u.um,
        )

        if "REFERENCES" in record:
            self.reference: str = record["REFERENCES"].strip()

        if "COMMENTS" in record or "NOTES" in record:
            self.comments: str = "\n".join(
                record[key].strip() for key in ("COMMENTS", "NOTES") if key in record
            )

    @overload
    def refractive_index(self, wl: ScalarQuantity) -> float: ...

    @overload
    def refractive_index(self, wl: ArrayQuantity) -> NDArray: ...

    @overload
    def refractive_index(self, wl: None = None) -> NDArray: ...

    def refractive_index(
        self,
        wl: ScalarQuantity | ArrayQuantity | None = None,
    ) -> float | NDArray:
        """
        The complex refractive index n + ik at the given wavelength
        """

        if self.refractive_index_n is None:
            raise ValueError("Refractive index data is not available.")

        n: float | NDArray[np.floating] = self.refractive_index_n

        k: float | NDArray[np.floating] = (
            0.0 if self.refractive_index_k is None else self.refractive_index_k
        )

        if wl is None:
            return np.asarray(self.refractive_index_n) + 1j * np.asarray(k)

        wl: float | NDArray[np.floating] = wl.to(u.um).value

        all_wls: NDArray[np.floating] = self.wavelengths.to(u.um).value

        n_interp: NDArray[np.floating] = np.interp(wl, all_wls, n)

        k_interp: NDArray[np.floating] = np.interp(wl, all_wls, k)

        return n_interp + 1j * k_interp


EDMUND_OPTICS_AR_BROAD: list[Coating] = [
    parse_edmund_optics_ar_coating_xlsx(file)
    for file in sorted(COATING_DATA.glob("Edmund Optics/AR_broad/EO_*.xlsx"))
]

EDMUND_OPTICS_AR_NARROW: list[Coating] = [
    parse_edmund_optics_ar_coating_xlsx(file)
    for file in sorted(COATING_DATA.glob("Edmund Optics/AR_narrow/EO_*.xlsx"))
]

EDMUND_OPTICS_HR_BROAD: list[Coating] = [
    # parse_edmund_optics_hr_coating_xlsx(file)
    # for file in sorted(COATING_DATA.glob("Edmund Optics/HR_broad/EO_*.xlsx"))
]

EDMUND_OPTICS_HR_NARROW: list[Coating] = [
    # parse_edmund_optics_hr_coating_xlsx(file)
    # for file in sorted(COATING_DATA.glob("Edmund Optics/HR_narrow/EO_*.xlsx"))
]

THORLABS_AR_BROAD: list[Coating] = [
    parse_thorlabs_ar_coating_xlsx(file)
    for file in sorted(COATING_DATA.glob("Thorlabs/AR_broad/THORLABS_*.xlsx"))
]

THORLABS_AR_NARROW: list[Coating] = [
    parse_thorlabs_ar_coating_xlsx(file)
    for file in sorted(COATING_DATA.glob("Thorlabs/AR_narrow/THORLABS_*.xlsx"))
]

THORLABS_HR_BROAD: list[Coating] = [
    # parse_thorlabs_hr_coating_xlsx(file)
    # for file in sorted(COATING_DATA.glob("Thorlabs/HR_broad/THORLABS_*.xlsx"))
]

THORLABS_HR_NARROW: list[Coating] = [
    # parse_thorlabs_hr_coating_xlsx(file)
    # for file in sorted(COATING_DATA.glob("Thorlabs/HR_narrow/THORLABS_*.xlsx"))
]


def test_commercial_coatings() -> None:

    fig = plt.figure()
    ax = fig.gca()

    for coating in EDMUND_OPTICS_AR_BROAD:
        coating.plot(ax=ax)

    for coating in EDMUND_OPTICS_AR_NARROW:
        coating.plot(ax=ax)

    for coating in THORLABS_AR_BROAD:
        coating.plot(ax=ax)

    for coating in THORLABS_AR_NARROW:
        coating.plot(ax=ax)

    plt.show()


def test_complex_index() -> None:

    fig: plt.Figure = plt.figure()
    ax: plt.Axes = fig.gca()

    wls: ArrayQuantity = np.linspace(370, 900, 100) * u.nm

    for file in sorted((COATING_DATA / "metallic_complex").glob("*.yml")):
        material = ComplexIndexMaterial(name=file.stem)
        material.parse_file(file)

        n: NDArray[np.floating] = np.real(material.refractive_index(wls))
        k: NDArray[np.floating] = np.imag(material.refractive_index(wls))

        ax.plot(wls, n, label=f"{material.name} (n)")
        ax.plot(wls, k, label=f"{material.name} (k)")

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Refractive Index")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    # test_commercial_coatings()

    test_complex_index()
