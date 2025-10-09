from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
from astropy import units as u
from astropy.units import Quantity

if TYPE_CHECKING:
    from matplotlib import pyplot as plt

from curved_air.optics.dispersers import EchelleGrating, Prism, Grism
from curved_air.detectors import Detector


@dataclass
class EchelleSpectrograph:
    name: str = ""
    grating: EchelleGrating = field(default_factory=EchelleGrating)
    cross_disperser: Prism | Grism = field(default_factory=Prism)
    detector: Detector = field(default_factory=Detector)

    beam_diameter: Quantity | None = None
    min_wl: Quantity = 390 * u.nm
    max_wl: Quantity = 950 * u.nm
    collimator_focal_length: Quantity = 1125 * u.mm
    slit_width: Quantity = 40 * u.um
    slit_length: Quantity = 0.5 * u.mm
    slit_native_focal_ratio: float = 3.85
    camera_focal_length: Quantity = 730 * u.mm

    def __post_init__(self) -> None:
        if self.beam_diameter is None:
            self.beam_diameter = self.grating.beam_diameter

        if self.grating.min_wl is None:
            self.grating.min_wl = self.min_wl
        if self.grating.max_wl is None:
            self.grating.max_wl = self.max_wl

        if self.cross_disperser.min_wl is None:
            self.cross_disperser.min_wl = self.min_wl
        if self.cross_disperser.max_wl is None:
            self.cross_disperser.max_wl = self.max_wl

    def plot_detector_edge(self, ax: plt.Axes, name: bool = True) -> None:
        self.detector.plot_edge(ax=ax, name=name)

    @property
    def collimator_focal_ratio(self) -> float:
        return self.collimator_focal_length / self.beam_diameter

    @property
    def camera_focal_ratio(self) -> float:
        return self.camera_focal_length / self.beam_diameter

    @property
    def slit_magnification(self) -> float:
        return float((self.camera_focal_ratio / self.slit_native_focal_ratio).si.value)

    @property
    def slit_sampling(self, slit_width: Quantity | None = None) -> float:
        if slit_width is None:
            slit_width = self.slit_width

        slit_image_width = slit_width * self.slit_magnification
        return float((slit_image_width / self.detector.pixel_size).si.value)

    @property
    def spectral_orders(self) -> list[int]:
        """ """

        return self.grating.diffraction_orders

    def order_wavelengths(
        self,
        m: int,
        num: int = 100,
        FSR_fraction: float = 1,
    ) -> Quantity:
        """ """

        min_m_wl, max_m_wl = self.grating.order_FSR_bounds(
            m=m,
            FSR_fraction=FSR_fraction,
        )

        return np.linspace(min_m_wl.to(u.um).value, max_m_wl.to(u.um).value, num) * u.um

    def full_wavelengths(self) -> dict[int, list[Quantity]]:
        """ """

        return {m: self.order_wavelengths(m) for m in self.spectral_orders}

    @property
    def collimator_plate_scale(self) -> Quantity:
        return (1 * u.radian / self.collimator_focal_length).to(u.deg / u.mm)

    @property
    def camera_plate_scale(self) -> Quantity:
        return (1 * u.radian / self.camera_focal_length).to(u.deg / u.mm)

    def image_plane_position(
        self,
        m: int,
        wavelength: Quantity | list[Quantity],
    ) -> tuple[Quantity, Quantity]:
        """ """

        if isinstance(wavelength, list | tuple):
            return [self.image_plane_position(m, wl) for wl in wavelength]

        grating_angle: Quantity = self.grating.diffraction_angle(m=m, wl=wavelength)

        cross_disperser_angle: Quantity = self.cross_disperser.dispersion_angle(
            wl=wavelength,
            field_angle=grating_angle,
        )

        return (
            cross_disperser_angle / self.camera_plate_scale,
            grating_angle / self.camera_plate_scale,
        )

    @cached_property
    def reddest_order_image_plane_separation(self) -> Quantity:
        """ """

        o1, o2, *_ = sorted(self.spectral_orders)

        wl1: Quantity = self.grating.blaze_wavelength(m=o1)
        wl2: Quantity = self.grating.blaze_wavelength(m=o2)

        x1: Quantity = self.image_plane_position(m=o1, wavelength=wl1)
        x2: Quantity = self.image_plane_position(m=o2, wavelength=wl2)

        return abs(x1[0] - x2[0])

    @cached_property
    def bluest_order_image_plane_separation(self) -> Quantity:
        """ """

        *_, o1, o2 = sorted(self.spectral_orders)

        wl1: Quantity = self.grating.blaze_wavelength(m=o1)
        wl2: Quantity = self.grating.blaze_wavelength(m=o2)

        x1: Quantity = self.image_plane_position(m=o1, wavelength=wl1)
        x2: Quantity = self.image_plane_position(m=o2, wavelength=wl2)

        return abs(x1[0] - x2[0])

    @property
    def slit_image_width(self) -> Quantity:
        """Width of the slit image in the image plane"""
        return self.slit_width * self.slit_magnification

    @property
    def slit_image_length(self) -> Quantity:
        """Width of the slit image in the image plane"""
        return self.slit_length * self.slit_magnification

    def resolution_element(self, wl: Quantity, m: int | None = None) -> Quantity:
        if m is None:
            # Find the diffraction order in which this wavelength is within the FSR
            m = self.grating.order_number(wl=wl, quantised=True)

        angular_dispersion = self.grating.angular_dispersion(m=m, wl=wl)
        spatial_dispersion = angular_dispersion / self.camera_plate_scale

        return (1 / (spatial_dispersion / self.slit_image_width)).to(wl.unit)

    def spectral_resolution(self, wl: Quantity, m: int | None = None) -> float:
        return int(wl / self.resolution_element(wl=wl, m=m))

    def print_parameters(self) -> EchelleSpectrograph:
        print(f"{'Collimated beam diameter:':<30}", end="")
        print(f"{self.beam_diameter:.3g}")

        print(f"{'Slit magnification factor:':<30}", end="")
        print(f"{self.slit_magnification:.3g}x")

        print(f"{'Slit length at detector:':<30}", end="")
        print(
            f"{(self.slit_length * self.slit_magnification / self.detector.pixel_size).si:.3g} pixels"  # noqa: E501
        )

        print(f"{'Bluest orders separation:':<30}", end="")
        print(
            f"{(self.bluest_order_image_plane_separation / self.detector.pixel_size).si:.3g} pixels"  # noqa: E501
        )

        print(f"{'Reddest orders separation:':<30}", end="")
        print(
            f"{(self.reddest_order_image_plane_separation / self.detector.pixel_size).si:.3g} pixels"  # noqa: E501
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


@dataclass
class DoublePassEchelleSpectrograph(EchelleSpectrograph):
    """
    Description of a double-pass echelle spectrograph

    Extends the EchelleSpectrograph class to include a second pass through the
    cross-disperser. It needs to redefine the Grating after the first pass for each
    wavelength.

    WIP
    """

    def image_plane_position(
        self,
        m: int,
        wavelength: Quantity,
    ) -> tuple[Quantity, Quantity]:
        """ """

        # TODO: how to check if a Quantity is actually a list or array underneath?
        if (isinstance(wavelength, Quantity) and wavelength.size > 1) or isinstance(
            wavelength.value, list | tuple
        ):
            return [
                self.image_plane_position(m, wl).to(u.mm).value for wl in wavelength
            ] * u.mm

        prism_first_pass_angle = self.cross_disperser.dispersion_angle(
            wl=wavelength,
            field_angle=0 * u.deg,
        )

        _grating: EchelleGrating = EchelleGrating(
            groove_frequency=self.grating.groove_frequency,
            blaze_angle=self.grating.blaze_angle,
            gamma_angle=self.grating.gamma_angle + prism_first_pass_angle,
            min_wl=self.grating.min_wl,
            max_wl=self.grating.max_wl,
        )

        grating_angle = _grating.diffraction_angle(m=m, wl=wavelength)

        _prism_second_pass: Prism = Prism(
            glass=self.cross_disperser.glass,
            apex_angle=self.cross_disperser.apex_angle,
            min_wl=self.cross_disperser.min_wl,
            max_wl=self.cross_disperser.max_wl,
            incidence_angle=self.cross_disperser.incidence_angle
            - 2 * self.grating.gamma_angle
            - 2 * prism_first_pass_angle,
        )

        prism_second_pass_angle = 2 * _prism_second_pass.dispersion_angle(
            wl=wavelength,
            field_angle=grating_angle,
        )

        return (
            (prism_second_pass_angle / self.camera_plate_scale).to(u.mm).value,
            (grating_angle / self.camera_plate_scale).to(u.mm).value,
        ) * u.mm
