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

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, overload

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from astropy import units as u

from curved_air.plotting import plot_style

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

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

    def parse_file(self, file: Path) -> ComplexIndexMaterial:
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

        return self

    @overload
    def n(self, wl: ScalarQuantity) -> float: ...

    @overload
    def n(self, wl: ArrayQuantity) -> NDArray[np.floating]: ...

    @overload
    def n(self, wl: None = None) -> NDArray[np.floating]: ...

    def n(
        self, wl: ScalarQuantity | ArrayQuantity | None = None
    ) -> float | NDArray[np.floating] | None:
        """
        The real part of the refractive index n at the given wavelength
        """

        if self.refractive_index_n is None:
            raise ValueError("Refractive index data is not available.")

        if wl is None:
            return self.refractive_index_n

        if isinstance(self.refractive_index_n, float):
            return self.refractive_index_n * np.ones_like(wl.value, dtype=float)

        wl: NDArray[np.floating] | float = wl.to(u.um).value

        all_wls: NDArray[np.floating] = self.wavelengths.to(u.um).value

        return np.interp(wl, all_wls, self.refractive_index_n)

    @overload
    def k(self, wl: ScalarQuantity) -> float: ...

    @overload
    def k(self, wl: ArrayQuantity) -> NDArray[np.floating]: ...

    @overload
    def k(self, wl: None = None) -> NDArray[np.floating]: ...

    def k(
        self, wl: ScalarQuantity | ArrayQuantity | None = None
    ) -> float | NDArray[np.floating]:
        """
        The imaginary part of the refractive index k at the given wavelength
        """

        if self.refractive_index_k is None:
            return 0.0

        if wl is None:
            return self.refractive_index_k

        if isinstance(self.refractive_index_k, float):
            return self.refractive_index_k * np.ones_like(wl.value, dtype=float)

        wl: NDArray[np.floating] | float = wl.to(u.um).value

        all_wls: NDArray[np.floating] = self.wavelengths.to(u.um).value

        return np.interp(wl, all_wls, self.refractive_index_k)

    @overload
    def refractive_index(self, wl: ScalarQuantity) -> complex: ...

    @overload
    def refractive_index(self, wl: ArrayQuantity) -> NDArray[np.complexfloating]: ...

    @overload
    def refractive_index(self, wl: None = None) -> NDArray[np.complexfloating]: ...

    def refractive_index(
        self,
        wl: ScalarQuantity | ArrayQuantity | None = None,
    ) -> complex | NDArray[np.floating | np.complexfloating]:
        """
        The complex refractive index n - ik at the given wavelength
        """

        if self.refractive_index_n is None:
            raise ValueError("Refractive index data is not available.")

        if wl is None:
            return self.refractive_index_n - 1j * (
                self.refractive_index_k if self.refractive_index_k is not None else 0.0
            )

        return self.n(wl) - 1j * self.k(wl)


McPeak_Al: ComplexIndexMaterial = ComplexIndexMaterial(
    name="Aluminium [McPeak+2025]"
).parse_file(COATING_DATA / "metallic_complex" / "McPeak_Al.yml")

McPeak_Ag: ComplexIndexMaterial = ComplexIndexMaterial(
    name="Silver [McPeak+2015]"
).parse_file(COATING_DATA / "metallic_complex" / "McPeak_Ag.yml")

McPeak_Au: ComplexIndexMaterial = ComplexIndexMaterial(
    name="Gold [McPeak+2015]"
).parse_file(COATING_DATA / "metallic_complex" / "McPeak_Au.yml")

Sytchkova_Cr: ComplexIndexMaterial = ComplexIndexMaterial(
    name="Chromium [Sytchkova+2021]"
).parse_file(COATING_DATA / "metallic_complex" / "Sytchkova_Cr.yml")


@dataclass
class Layer:
    """
    A class representing a single layer in a thin-film stack, with a complex refractive
    index and a thickness.
    """

    material: ComplexIndexMaterial
    thickness: ScalarQuantity


@dataclass
class ReflectanceResult:
    """All arrays have shape (n_angles, n_wavelengths)."""

    wls: ScalarQuantity | ArrayQuantity
    aoi: ScalarQuantity | ArrayQuantity
    rs: NDArray[np.complexfloating]  # complex amplitude coefficients
    rp: NDArray[np.complexfloating]
    Rs: NDArray[np.floating]  # intensity reflectances
    Rp: NDArray[np.floating]
    R: NDArray[np.floating]  # unpolarised = (Rs + Rp) / 2
    retardance_deg: NDArray[np.floating]  # arg(rp) - arg(rs), wrapped to (-180, 180]


def _admittance_index(
    material: ComplexIndexMaterial,
    wl: ArrayQuantity,
) -> NDArray[np.complexfloating]:
    """
    Complex index of `material` at `wl` in the thin-film (Macleod) convention N = n - ik.

    Note the sign: ComplexIndexMaterial.refractive_index() returns n + ik, which is
    the opposite convention to the one the characteristic-matrix maths below uses.
    """
    n = np.broadcast_to(np.asarray(material.n(wl), dtype=float), wl.shape)
    k = np.broadcast_to(np.abs(np.asarray(material.k(wl), dtype=float)), wl.shape)
    return n - 1j * k


def thin_film_reflectance(
    wls: ScalarQuantity | ArrayQuantity,
    aoi: ScalarQuantity | ArrayQuantity,
    incident_material: ComplexIndexMaterial,
    substrate_material: ComplexIndexMaterial,
    layers: list[Layer] | Layer | None = None,
) -> ReflectanceResult:
    """
    Calculate the reflectance of a thin-film stack on a semi-infinite (opaque) substrate

    Typical use-cases
    ---------------
    Front-surface protected Al:  incident=Air, layers=[Layer(silica, 100 nm)], substrate=McPeak_Al
    Back-surface Ag on silica:   incident=silica, layers=[Layer(nicr, 2 nm)], substrate=McPeak_Ag
    (for a back-surface mirror the paint/overcoat behind the metal is irrelevant
    as long as the metal is opaque, i.e. >~100 nm.)

    Keep opaque metal as the substrate rather than as a thick layer: a thick
    absorbing layer gives exponentially large matrix elements.
    """

    if layers is None:
        layers: list[Layer] = []
    elif isinstance(layers, Layer):
        layers: list[Layer] = [layers]

    shape: tuple[int, int] = (aoi.size, wls.size)

    n0: NDArray[np.floating] = np.broadcast_to(
        np.asarray(incident_material.n(wls)), wls.shape
    )
    k0: NDArray[np.floating] = np.broadcast_to(
        np.abs(np.asarray(incident_material.k(wls))), wls.shape
    )

    if np.any(k0) > 1e-9:  # noqa: PLR2004
        raise ValueError("The incident medium must be non-absorbing (i.e. k = 0)")

    # n0: NDArray[np.complexfloating] = n - 1j * k
    # n0: NDArray[np.floating] = n0.real[None, :]  # (1, W)

    wl2 = wls[None, :]

    beta = n0 * np.sin(aoi)  # conserved n sin(theta), (A, W)
    cos0 = np.cos(aoi)
    eta0_s = n0 * cos0
    eta0_p = n0 / cos0

    def tilted(
        n: complex | NDArray[np.complexfloating],
    ) -> tuple[
        NDArray[np.complexfloating],
        NDArray[np.complexfloating],
        NDArray[np.complexfloating],
    ]:
        """Return N cos(theta), eta_s, eta_p for a medium of complex index N."""

        # Why do this with beta instead of theta directly? Because the angle in the
        # medium is complex for absorbing layers,and the complex square root has a
        # branch cut. Using beta avoids this problem.
        n_cos_theta: NDArray[np.complexfloating] = np.sqrt(n**2 - beta**2 + 0j)

        # n - ik convention: absorbing/evanescent fields need Im(N cos) <= 0
        n_cos_theta: NDArray[np.complexfloating] = np.where(
            n_cos_theta.imag > 0, -n_cos_theta, n_cos_theta
        )

        return n_cos_theta, n_cos_theta, n**2 / n_cos_theta

    one: NDArray[np.floating] = np.ones(shape, dtype=complex)
    zero: NDArray[np.floating] = np.zeros(shape, dtype=complex)

    # Initialise identity matrices for S- and P-polarisation, to be multiplied by each
    # layer's matrix
    Ms: list[NDArray[np.floating]] = [one.copy(), zero.copy(), zero.copy(), one.copy()]
    Mp: list[NDArray[np.floating]] = [one.copy(), zero.copy(), zero.copy(), one.copy()]

    for layer in layers:
        n: complex | NDArray[np.complexfloating] = layer.material.refractive_index(wls)

        n_c, eta_s, eta_p = tilted(n)

        delta = 2 * np.pi * n_c * layer.thickness.to_value(u.nm) / wl2

        c, s = np.cos(delta), np.sin(delta)

        for M, eta in ((Ms, eta_s), (Mp, eta_p)):
            l11, l12, l21, l22 = c, 1j * s / eta, 1j * eta * s, c

            a, b, cc, dd = M

            M[:] = [
                a * l11 + b * l21,
                a * l12 + b * l22,
                cc * l11 + dd * l21,
                cc * l12 + dd * l22,
            ]

    Nsub: complex | NDArray[np.complexfloating] = substrate_material.refractive_index(
        wls
    )
    _, eta_sub_s, eta_sub_p = tilted(Nsub)

    def amp(
        M: list[NDArray[np.floating]],
        eta0: float,
        eta_sub: NDArray[np.complexfloating],
    ) -> NDArray[np.complexfloating]:
        """
        Matrix multiplication to produce the complex amplitude reflection coefficient
        r = (eta0 B - C) / (eta0 B + C) for either S- or P-polarisation.
        """

        B: NDArray[np.complexfloating] = M[0] + M[1] * eta_sub
        C: NDArray[np.complexfloating] = M[2] + M[3] * eta_sub
        return (eta0 * B - C) / (eta0 * B + C)

    rs: NDArray[np.complexfloating] = amp(Ms, eta0_s, eta_sub_s)
    rp: NDArray[np.complexfloating] = amp(Mp, eta0_p, eta_sub_p)
    Rs, Rp = np.abs(rs) ** 2, np.abs(rp) ** 2
    ret = np.angle(rp * np.conj(rs), deg=True)

    return ReflectanceResult(wls, aoi, rs, rp, Rs, Rp, 0.5 * (Rs + Rp), ret)


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


def test_thin_film_reflectance() -> None:

    wls: ArrayQuantity = np.linspace(350, 950, 100) * u.nm

    Air = ComplexIndexMaterial(
        name="Air", wavelengths=wls, refractive_index_n=1.0, refractive_index_k=0.0
    )

    Glass = ComplexIndexMaterial(
        name="Glass", wavelengths=wls, refractive_index_n=1.5, refractive_index_k=0.0
    )

    fig: plt.Figure = plt.figure()
    ax: plt.Axes = fig.gca()

    for material in (McPeak_Al, McPeak_Ag, McPeak_Au, Sytchkova_Cr):
        air_incidence_result: ReflectanceResult = thin_film_reflectance(
            wls=wls,
            aoi=45 * u.deg,
            incident_material=Air,
            substrate_material=material,
        )

        glass_incidence_result: ReflectanceResult = thin_film_reflectance(
            wls=wls,
            aoi=45 * u.deg,
            incident_material=Glass,
            substrate_material=material,
        )

        air_incidence = ax.plot(
            wls,
            air_incidence_result.R[0],
            label=f"Unpolarised, {material.name} incident from Air",
        )
        _glass_incidence = ax.plot(
            wls,
            glass_incidence_result.R[0],
            ls="--",
            color=air_incidence[0].get_color(),
            label=f"Unpolarised, {material.name} incident from Glass",
        )

    ax.set_title("Reflectance of unpolarised light from metallic coatings at 45° AOI")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Reflectance")
    ax.legend()

    ax.set_xlim(350, 950)
    ax.set_ylim(0.5, 1.15)

    ax.axhline(1.0, color="k", ls="--", alpha=0.5)
    ax.axvline(x=370, color="k", ls="--", alpha=0.5)
    ax.axvline(x=900, color="k", ls="--", alpha=0.5)

    plt.show()


if __name__ == "__main__":
    # test_commercial_coatings()

    # test_complex_index()

    test_thin_film_reflectance()
