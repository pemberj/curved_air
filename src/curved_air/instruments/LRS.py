"""
A description of a low-resolution spectrograph
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, overload

import numpy as np
from astropy import units as u
from astropy.units import Quantity

from curved_air.detectors import Detector
from curved_air.glass import GlassCatalogue
from curved_air.optics.dispersers import Disperser, Prism, SimpleGrating

if TYPE_CHECKING:
    from matplotlib import pyplot as plt

    from curved_air.type_hinting import ArrayQuantity, ScalarQuantity


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
        if self.disperser.min_wl is None:
            self.disperser.min_wl: ScalarQuantity = self.min_wl
        if self.disperser.max_wl is None:
            self.disperser.max_wl: ScalarQuantity = self.max_wl

    def field_angles(self, num: int = 30) -> ArrayQuantity:
        """
        Returns the field angles at the disperser corresponding to the edges of the
        slit.
        """

        slit_half_length: ScalarQuantity = self.slit_length / 2

        field_angle_y: float = (
            (
                slit_half_length.to(u.mm).value
                * self.collimator_plate_scale.to(u.deg / u.mm).value
            )
            .to(u.deg)
            .value
        )

        return np.linspace(-field_angle_y, field_angle_y, num=num) * u.deg

    @property
    def collimator_focal_ratio(self) -> float:
        if self.collimator_focal_length is None or self.beam_diameter is None:
            raise ValueError(
                "Collimator focal length and beam diameter must be defined."
            )
        return self.collimator_focal_length / self.beam_diameter

    @property
    def camera_focal_ratio(self) -> float:
        if self.camera_focal_length is None or self.beam_diameter is None:
            raise ValueError("Camera focal length and beam diameter must be defined.")
        return self.camera_focal_length / self.beam_diameter

    @property
    def slit_magnification(self) -> float:
        return float((self.camera_focal_ratio / self.slit_native_focal_ratio).si.value)

    @property
    def slit_sampling(self, slit_width: ScalarQuantity | None = None) -> float:
        if slit_width is None:
            slit_width = self.slit_width

        slit_image_width = slit_width * self.slit_magnification
        return float((slit_image_width / self.detector.pixel_size).si.value)

    @property
    def collimator_plate_scale(self) -> ScalarQuantity:
        return (1 * u.radian / self.collimator_focal_length).to(u.deg / u.mm)

    @property
    def camera_plate_scale(self) -> ScalarQuantity:
        return (1 * u.radian / self.camera_focal_length).to(u.deg / u.mm)

    @overload
    def image_plane_position(
        self,
        wavelength: ScalarQuantity,
        field_angle: ScalarQuantity = 0 * u.deg,
    ) -> tuple[ScalarQuantity, ScalarQuantity]: ...

    @overload
    def image_plane_position(
        self,
        wavelength: ScalarQuantity,
        field_angle: ArrayQuantity | list[ScalarQuantity],
    ) -> tuple[ScalarQuantity, ScalarQuantity]: ...

    @overload
    def image_plane_position(
        self,
        wavelength: ArrayQuantity | list[ScalarQuantity] | tuple[ScalarQuantity],
        field_angle: ScalarQuantity = 0 * u.deg,
    ) -> ArrayQuantity: ...

    def image_plane_position(
        self,
        wavelength: ScalarQuantity
        | ArrayQuantity
        | list[ScalarQuantity]
        | tuple[ScalarQuantity],
        field_angle: ScalarQuantity = 0 * u.deg,
    ) -> ArrayQuantity:
        """ """

        if isinstance(wavelength, list | tuple) or (
            isinstance(wavelength, Quantity) and not wavelength.isscalar
        ):
            return [
                self.image_plane_position(wl, field_angle=field_angle).value
                for wl in wavelength
            ] * u.mm

        if isinstance(wavelength, list | tuple) or (
            isinstance(wavelength, Quantity) and not wavelength.isscalar
        ):
            assert isinstance(wavelength, Quantity) & wavelength.isscalar
            return [
                self.image_plane_position(wavelength, field_angle=ifa).value
                for ifa in field_angle
            ] * u.mm

        assert isinstance(wavelength, Quantity)
        assert wavelength.isscalar

        grating_angle: ScalarQuantity = self.disperser.diffraction_angle(wl=wavelength)

        cross_disperser_angle: ScalarQuantity = self.cross_disperser.dispersion_angle(
            wl=wavelength,
            field_angle=grating_angle,
        )

        dispersion_direction_position: ScalarQuantity = (
            grating_angle / self.camera_plate_scale
        )
        cross_dispersion_direction_position: ScalarQuantity = (
            cross_disperser_angle / self.camera_plate_scale
        )

        return (
            cross_dispersion_direction_position.value,
            dispersion_direction_position.value,
        ) * u.mm

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
            # Default behaviour is to return the spectral resolution at a range of
            # wavlenegths across the spectrograph's bandpass
            wl = (
                np.linspace(
                    self.min_wl.to(u.nm).value, self.max_wl.to(u.nm).value, num=30
                )
                * u.nm
            )

        if wl.isscalar:
            return float((wl / self.resolution_element(wl=wl)).si.value)

        return [float((iwl / self.resolution_element(wl=iwl)).si.value) for iwl in wl]

    def plot_detector_edge(self, ax: plt.Axes, name: bool = True) -> None:
        self.detector.plot_edge(ax=ax, name=name)

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
        print(f"{self.spectral_resolution(wl=0.55 * u.um):,.0f}")

        return self


def test() -> None:
    lrs = LowResolutionSpectrograph(
        name="Test LRS",
        # disperser=Prism(glass=GlassCatalogue("Schott")["F2"], apex_angle=60 * u.deg),
        disperser=SimpleGrating(groove_frequency=300 / u.mm),
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

    # print(lrs.disperser.dispersion_angle(wl=0.55 * u.um))

    lrs.print_parameters()


if __name__ == "__main__":
    test()
