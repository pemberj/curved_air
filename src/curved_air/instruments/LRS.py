"""
A description of a low-resolution spectrograph
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, overload

import numpy as np
from astropy import units as u
from astropy.units import Quantity
from matplotlib import pyplot as plt

from curved_air.detectors import Detector
from curved_air.glass import GlassCatalogue  # noqa: F401
from curved_air.optics.dispersers import Disperser, Prism, SimpleGrating  # noqa: F401
from curved_air.plotting import wavelength_to_rgb

if TYPE_CHECKING:
    from matplotlib import pyplot as plt
    from numpy.typing import NDArray

    from curved_air.type_hinting import ArrayQuantity, ScalarQuantity


@dataclass
class Coordinate:
    """
    Represents a position in either the slit plane or the focal plane.

    x is always the dispersion axis and y is always the slit axis.
    """

    x: ScalarQuantity
    y: ScalarQuantity

    def __post_init__(self) -> None:
        assert isinstance(self.x, Quantity)
        assert self.x.isscalar

        assert isinstance(self.y, Quantity)
        assert self.y.isscalar


@dataclass
class LowResolutionSpectrograph:
    name: str = ""
    disperser: Disperser = field(default_factory=Prism)
    detector: Detector = field(default_factory=Detector)

    min_wl: ScalarQuantity = 400 * u.nm
    max_wl: ScalarQuantity = 800 * u.nm
    beam_diameter: ScalarQuantity = 100 * u.mm
    collimator_focal_length: ScalarQuantity | None = 800 * u.mm
    slit_width: ScalarQuantity = 50 * u.um
    slit_length: ScalarQuantity = 2 * u.mm
    slit_native_focal_ratio: float = 4
    camera_focal_length: ScalarQuantity = 600 * u.mm

    def __post_init__(self) -> None:
        if self.disperser.min_wl != self.min_wl:
            self.disperser.min_wl: ScalarQuantity = self.min_wl
        if self.disperser.max_wl != self.max_wl:
            self.disperser.max_wl: ScalarQuantity = self.max_wl

    @cached_property
    def mid_wl(self) -> ScalarQuantity:
        return (self.min_wl + self.max_wl) / 2

    def field_angle(
        self,
        slit_position: Coordinate,
    ) -> tuple[ScalarQuantity, ScalarQuantity]:
        """
        Compute the field angle (in the collimated beam / at the disperser) for a given
        position in the slit plane. This is a simple calculation based on the focal
        length (plate scale) of the collimator.
        """

        return (
            (slit_position.x * self.collimator_plate_scale).to(u.deg),
            (slit_position.y * self.collimator_plate_scale).to(u.deg),
        )

    def field_angles(self, num: int = 30) -> ArrayQuantity:
        """
        Returns an array of field angles (in the collimated beam) running along the
        length of the slit. Here it is assumed that the width of the slit is small
        enough that the change in field angle across the width of the slit is
        negligible.
        """

        return [
            self.field_angle(Coordinate(0 * u.mm, y)).y.to(u.deg).value
            for y in np.linspace(-self.slit_length / 2, self.slit_length / 2, num=num)
        ] * u.deg

    @cached_property
    def collimator_plate_scale(self) -> ScalarQuantity:
        return (1 * u.radian / self.collimator_focal_length).to(u.deg / u.mm)

    @cached_property
    def collimator_focal_ratio(self) -> float:
        if self.collimator_focal_length is None or self.beam_diameter is None:
            raise ValueError(
                "Collimator focal length and beam diameter must be defined."
            )
        return self.collimator_focal_length / self.beam_diameter

    @cached_property
    def camera_plate_scale(self) -> ScalarQuantity:
        return (1 * u.radian / self.camera_focal_length).to(u.deg / u.mm)

    @cached_property
    def camera_focal_ratio(self) -> float:
        if self.camera_focal_length is None or self.beam_diameter is None:
            raise ValueError("Camera focal length and beam diameter must be defined.")
        return self.camera_focal_length / self.beam_diameter

    @cached_property
    def slit_magnification(self) -> float:
        return float((self.camera_focal_ratio / self.slit_native_focal_ratio).si.value)

    @cached_property
    def slit_sampling(self) -> float:
        slit_image_width = self.slit_width * self.slit_magnification
        return float((slit_image_width / self.detector.pixel_size).si.value)

    def image_plane_position(
        self,
        wavelength: ScalarQuantity,
        slit_position: Coordinate | None = None,
    ) -> Coordinate:
        """
        Trace a single ray from the slit through the spectrograph to the image plane and
        return the position of the ray in the image plane.

        To produce an image of the slit, this method can be called for multiple field
        angles across the length of the slit.
        """

        if slit_position is None:
            slit_position = Coordinate(x=0 * u.mm, y=0 * u.mm)

        assert isinstance(wavelength, Quantity)
        assert wavelength.isscalar

        dispersion_axis_angle: ScalarQuantity = self.disperser.dispersion_angle(
            wl=wavelength,
            field_angle=self.field_angle(slit_position)[1],
        )

        field_angle = self.field_angle(slit_position)
        dispersion_axis_field_angle: ScalarQuantity = field_angle[0]
        slit_axis_angle: ScalarQuantity = field_angle[1]

        dispersion_axis_position: ScalarQuantity = (
            (dispersion_axis_angle + dispersion_axis_field_angle)
            / self.camera_plate_scale
        ).to(u.mm)
        slit_axis_position: ScalarQuantity = (
            slit_axis_angle / self.camera_plate_scale
        ).to(u.mm)

        return Coordinate(dispersion_axis_position, slit_axis_position)

    @cached_property
    def slit_vertices(self) -> list[Coordinate]:
        """
        The coordinates of the vertices of the slit in the slit plane. These can be
        used to trace through the system to produce an image of the slit in the image
        plane.
        """

        slit_half_width: ScalarQuantity = self.slit_width / 2
        slit_half_length: ScalarQuantity = self.slit_length / 2

        print(slit_half_length, slit_half_width)

        return [
            Coordinate(-slit_half_width, -slit_half_length),
            Coordinate(slit_half_width, -slit_half_length),
            Coordinate(slit_half_width, slit_half_length),
            Coordinate(-slit_half_width, slit_half_length),
        ]

    def slit_image_vertices(
        self,
        wavelength: ScalarQuantity,
    ) -> list[Coordinate]:
        """
        Produce an image of the slit at the given wavelength in the images plane. This
        takes into account the total system magnification, dispersion, and effects of
        field angle variation across the length of the slit.
        """

        return [
            self.image_plane_position(wavelength=wavelength, slit_position=sp)
            for sp in self.slit_vertices
        ]

    @property
    def slit_image_width(self) -> ScalarQuantity:
        """Width of the slit image in the image plane"""
        return self.slit_width * self.slit_magnification

    @property
    def slit_image_length(self) -> ScalarQuantity:
        """Width of the slit image in the image plane"""
        return self.slit_length * self.slit_magnification

    @overload
    def resolution_element(self, wl: ScalarQuantity) -> ScalarQuantity: ...

    @overload
    def resolution_element(self, wl: ArrayQuantity) -> ArrayQuantity: ...

    def resolution_element(self, wl: Quantity) -> Quantity:
        angular_dispersion = self.disperser.angular_dispersion(wl=wl)
        spatial_dispersion = angular_dispersion / self.camera_plate_scale

        return (1 / (spatial_dispersion / self.slit_image_width)).to(wl.unit)

    @overload
    def spectral_resolution(self, wl: ScalarQuantity) -> float: ...

    @overload
    def spectral_resolution(self, wl: ArrayQuantity | None = None) -> list[float]: ...

    def spectral_resolution(
        self,
        wl: ScalarQuantity | ArrayQuantity | None = None,
    ) -> float | list[float]:
        if wl is None:
            print(
                "Warning: spectral resolution is wavelength dependent. "
                + f"Returning spectral resolution at {self.mid_wl}, "
                + "the midpoint of the wavelength range."
            )
            return self.spectral_resolution(wl=self.mid_wl)

        if wl.isscalar:
            return float((wl / self.resolution_element(wl=wl)).si.value)

        return [float((iwl / self.resolution_element(wl=iwl)).si.value) for iwl in wl]

    def plot_detector_edge(self, ax: plt.Axes, name: bool = True) -> None:
        self.detector.plot_edge(ax=ax, name=name)

    def plot_slit_image(self, ax: plt.Axes, wavelength: ScalarQuantity) -> None:
        slit_image_vertices: list[Coordinate] = self.slit_image_vertices(
            wavelength=wavelength
        )

        slit_image_x: list[float] = [float(v.x.value) for v in slit_image_vertices]
        slit_image_y: list[float] = [float(v.y.value) for v in slit_image_vertices]

        ax.plot(
            [*slit_image_x, slit_image_x[0]],
            [*slit_image_y, slit_image_y[0]],
            color=wavelength_to_rgb(wavelength),
        )

    def plot_spectrum(self, ax: plt.Axes, num_wavelengths: int = 30) -> None:
        wavelengths: ArrayQuantity = self.equidistant_wavelengths(num=num_wavelengths)

        for wl in wavelengths:
            self.plot_slit_image(ax=ax, wavelength=wl)

        ax.set_xlabel("Dispersion axis (mm)")
        ax.set_ylabel("Slit axis (mm)")

    def equidistant_wavelengths(self, num: int = 30) -> ArrayQuantity:
        """
        Returns an array of wavelengths that are equidistant in the image plane. This is
        useful for plotting the spectrum with evenly spaced colors.
        """
        minx: float = self.image_plane_position(wavelength=self.min_wl).x.value
        maxx: float = self.image_plane_position(wavelength=self.max_wl).x.value

        wavelengths = np.linspace(
            self.min_wl.to(u.nm).value,
            self.max_wl.to(u.nm).value,
            num=max(200, 10 * num),
        )

        x_values = np.array(
            [
                self.image_plane_position(wavelength=wl * u.nm).x.value
                for wl in wavelengths
            ]
        )

        xs: NDArray = np.linspace(minx, maxx, num=num)

        # Interpolation function requires sorted data
        sort_idx = np.argsort(x_values)

        x_sorted = x_values[sort_idx]
        wl_sorted = wavelengths[sort_idx]

        x_unique, unique_idx = np.unique(x_sorted, return_index=True)
        wl_unique = wl_sorted[unique_idx]

        equidistant_wavelengths: np.ndarray = np.interp(xs, x_unique, wl_unique)

        return equidistant_wavelengths * u.nm

    def print_parameters(self) -> LowResolutionSpectrograph:
        print(f"{'Collimated beam diameter:':<30}", end="")
        print(f"{self.beam_diameter:.3g}")

        print(f"{'Slit magnification factor:':<30}", end="")
        print(f"{self.slit_magnification:.3g}x")

        print(f"{'Slit length at detector:':<30}", end="")
        print(
            f"{(self.slit_length * self.slit_magnification / self.detector.pixel_size).si:.3g} pixels"  # noqa: E501
        )

        print(f"{'Slit width (object plane):':<30}", end="")
        print(f"{self.slit_width:.3g}")

        print(f"{'Slit length (object plane):':<30}", end="")
        print(f"{self.slit_length:.3g}")

        print(f"{'Slit width (image plane):':<30}", end="")
        print(f"{(self.slit_width * self.slit_magnification):.3g}")

        print(f"{'Sampling of slit image:':<30}", end="")
        print(f"{self.slit_sampling:.3g} pixels")

        print(f"{'Spectral resolution:':<30}", end="")
        print(
            f"{self.spectral_resolution(wl=self.min_wl):,.0f} @ {self.min_wl:.0f}",
            end=", ",
        )
        print(
            f"{self.spectral_resolution(wl=self.mid_wl):,.0f} @ {self.mid_wl:.0f}",
            end=", ",
        )
        print(f"{self.spectral_resolution(wl=self.max_wl):,.0f} @ {self.max_wl:.0f}")

        return self


def test() -> None:
    lrs = LowResolutionSpectrograph(
        name="Test LRS",
        disperser=Prism(glass=GlassCatalogue("Schott")["F2"], apex_angle=60 * u.deg),
        # disperser=SimpleGrating(groove_frequency=300 / u.mm),
        detector=Detector(pixel_size=15 * u.um, nx=2048, ny=2048),
        min_wl=390 * u.nm,
        max_wl=950 * u.nm,
        beam_diameter=100 * u.mm,
        collimator_focal_length=800 * u.mm,
        slit_width=50 * u.um,
        slit_length=2 * u.mm,
        slit_native_focal_ratio=4,
        camera_focal_length=600 * u.mm,
    )

    lrs.print_parameters()


if __name__ == "__main__":
    test()
