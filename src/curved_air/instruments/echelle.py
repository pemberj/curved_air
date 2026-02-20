"""
A description of an echelle spectrograph

The included class EchelleSpectrograph can be used for various types of analysis, plot
generation, etc. for echelle spectrographs.

Also included is a DoublePassEchelleSpectrograph class for describing an echelle
spectrograph where the light passes twice through the cross-disperser. For now the
cross-disperser of this type of instrument must be a Prism.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, overload

import numpy as np
from astropy import units as u
from astropy.units import Quantity

from curved_air.detectors import Detector
from curved_air.optics.dispersers import (
    CrossDisperser,
    EchelleGrating,
    Prism,
)

if TYPE_CHECKING:
    from matplotlib import pyplot as plt

    from curved_air.type_hinting import ArrayQuantity, ScalarQuantity


@dataclass
class EchelleSpectrograph:
    name: str = ""
    grating: EchelleGrating = field(default_factory=EchelleGrating)
    cross_disperser: CrossDisperser = field(default_factory=Prism)
    detector: Detector = field(default_factory=Detector)

    beam_diameter: ScalarQuantity | None = None
    min_wl: ScalarQuantity | None = 390 * u.nm
    max_wl: ScalarQuantity | None = 950 * u.nm
    collimator_focal_length: ScalarQuantity | None = 1125 * u.mm
    slit_width: ScalarQuantity = 40 * u.um
    slit_length: ScalarQuantity = 0.5 * u.mm
    slit_native_focal_ratio: float = 3.85
    camera_focal_length: ScalarQuantity = 730 * u.mm

    def __post_init__(self) -> None:
        if self.beam_diameter is None:
            self.beam_diameter: ScalarQuantity = self.grating.beam_diameter

        if self.grating.min_wl is None:
            self.grating.min_wl: ScalarQuantity = self.min_wl
        if self.grating.max_wl is None:
            self.grating.max_wl: ScalarQuantity = self.max_wl

        if self.cross_disperser.min_wl is None:
            self.cross_disperser.min_wl: ScalarQuantity = self.min_wl
        if self.cross_disperser.max_wl is None:
            self.cross_disperser.max_wl: ScalarQuantity = self.max_wl

    def plot_detector_edge(self, ax: plt.Axes, name: bool = True) -> None:
        self.detector.plot_edge(ax=ax, name=name)

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
    def spectral_orders(self) -> list[int]:
        """ """

        return self.grating.diffraction_orders

    def order_wavelengths(
        self,
        m: int,
        num: int = 100,
        FSR_fraction: float = 1,
    ) -> ArrayQuantity:
        """ """

        min_m_wl, max_m_wl = self.grating.order_angular_FSR_bounds(
            m=m,
            FSR_fraction=FSR_fraction,
        )

        return np.linspace(min_m_wl.to(u.um).value, max_m_wl.to(u.um).value, num) * u.um

    def full_wavelengths(self) -> dict[int, ArrayQuantity]:
        """ """

        return {m: self.order_wavelengths(m) for m in self.spectral_orders}

    @property
    def collimator_plate_scale(self) -> ScalarQuantity:
        return (1 * u.radian / self.collimator_focal_length).to(u.deg / u.mm)

    @property
    def camera_plate_scale(self) -> ScalarQuantity:
        return (1 * u.radian / self.camera_focal_length).to(u.deg / u.mm)

    @overload
    def image_plane_position(
        self,
        m: int,
        wavelength: ScalarQuantity,
    ) -> tuple[ScalarQuantity, ScalarQuantity]: ...

    @overload
    def image_plane_position(
        self,
        m: int,
        wavelength: ArrayQuantity | list[ScalarQuantity] | tuple[ScalarQuantity],
    ) -> ArrayQuantity: ...

    def image_plane_position(
        self,
        m: int,
        wavelength: ScalarQuantity
        | ArrayQuantity
        | list[ScalarQuantity]
        | tuple[ScalarQuantity],
    ) -> ArrayQuantity:
        """ """

        if isinstance(wavelength, list | tuple) or (
            isinstance(wavelength, Quantity) and not wavelength.isscalar
        ):
            return [self.image_plane_position(m, wl).value for wl in wavelength] * u.mm

        assert isinstance(wavelength, Quantity)
        assert wavelength.isscalar

        grating_angle: ScalarQuantity = self.grating.diffraction_angle(
            m=m, wl=wavelength
        )

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

    @cached_property
    def reddest_order_image_plane_separation(self) -> ScalarQuantity:
        """ """

        o1, o2, *_ = sorted(self.spectral_orders)

        wl1: ScalarQuantity = self.grating.blaze_wavelength(m=o1)
        wl2: ScalarQuantity = self.grating.blaze_wavelength(m=o2)

        x1: tuple[ScalarQuantity, ScalarQuantity] = self.image_plane_position(
            m=o1, wavelength=wl1
        )
        x2: tuple[ScalarQuantity, ScalarQuantity] = self.image_plane_position(
            m=o2, wavelength=wl2
        )

        return abs(x1[0] - x2[0])

    @cached_property
    def bluest_order_image_plane_separation(self) -> ScalarQuantity:
        """ """

        *_, o1, o2 = sorted(self.spectral_orders)

        wl1: ScalarQuantity = self.grating.blaze_wavelength(m=o1)
        wl2: ScalarQuantity = self.grating.blaze_wavelength(m=o2)

        x1: tuple[ScalarQuantity, ScalarQuantity] = self.image_plane_position(
            m=o1, wavelength=wl1
        )
        x2: tuple[ScalarQuantity, ScalarQuantity] = self.image_plane_position(
            m=o2, wavelength=wl2
        )

        return abs(x1[0] - x2[0])

    @property
    def slit_image_width(self) -> ScalarQuantity:
        """Width of the slit image in the image plane"""
        return self.slit_width * self.slit_magnification

    @property
    def slit_image_length(self) -> ScalarQuantity:
        """Width of the slit image in the image plane"""
        return self.slit_length * self.slit_magnification

    @overload
    def resolution_element(
        self, wl: ScalarQuantity, m: int | None
    ) -> ScalarQuantity: ...

    @overload
    def resolution_element(self, wl: ArrayQuantity, m: int | None) -> ArrayQuantity: ...

    def resolution_element(self, wl: Quantity, m: int | None = None) -> Quantity:
        if m is None:
            # Find the diffraction order in which this wavelength is within the FSR
            m = self.grating.order_number(wl=wl, quantised=True)

        angular_dispersion = self.grating.angular_dispersion(m=m, wl=wl)
        spatial_dispersion = angular_dispersion / self.camera_plate_scale

        return (1 / (spatial_dispersion / self.slit_image_width)).to(wl.unit)

    @overload
    def spectral_resolution(
        self, wl: ScalarQuantity, m: int | None = None
    ) -> float: ...

    @overload
    def spectral_resolution(
        self, wl: ArrayQuantity, m: int | None = None
    ) -> list[float]: ...

    def spectral_resolution(
        self,
        wl: Quantity,
        m: int | None = None,
    ) -> float | list[float]:
        if wl.isscalar:
            return float((wl / self.resolution_element(wl=wl, m=m)).si.value)

        return [(iwl / self.resolution_element(wl=iwl, m=m)).si.value for iwl in wl]

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

    @overload
    def image_plane_position(
        self,
        m: int,
        wavelength: ScalarQuantity,
    ) -> tuple[ScalarQuantity, ScalarQuantity]: ...

    @overload
    def image_plane_position(
        self,
        m: int,
        wavelength: ArrayQuantity | list[ScalarQuantity] | tuple[ScalarQuantity],
    ) -> ArrayQuantity: ...

    def image_plane_position(
        self,
        m: int,
        wavelength: ScalarQuantity
        | ArrayQuantity
        | list[ScalarQuantity]
        | tuple[ScalarQuantity],
    ) -> tuple[ScalarQuantity, ScalarQuantity] | ArrayQuantity:
        """ """

        # For now, this only works with a Prism cross-disperser
        assert isinstance(self.cross_disperser, Prism)

        # TODO: how best to check if a Quantity is actually a list or array underneath?
        if (
            not isinstance(wavelength, Quantity)
            and isinstance(wavelength, list | tuple)
        ) or (isinstance(wavelength, Quantity) and not wavelength.isscalar):
            assert all(isinstance(iwl, Quantity) for iwl in wavelength)
            return [
                self.image_plane_position(m, wl).to(u.mm).value for wl in wavelength
            ] * u.mm

        # Otherwise, wavelength is a scalar Quantity
        assert isinstance(wavelength, Quantity)
        assert wavelength.isscalar

        prism_first_pass_angle: Quantity = self.cross_disperser.dispersion_angle(
            wl=wavelength,
            field_angle=0 * u.deg,
        )

        _grating: EchelleGrating = EchelleGrating(
            groove_frequency=self.grating.groove_frequency,
            blaze_angle=self.grating.blaze_angle,
            gamma_angle=(self.grating.gamma_angle + prism_first_pass_angle).to(u.deg),
            min_wl=self.grating.min_wl,
            max_wl=self.grating.max_wl,
        )

        grating_angle: Quantity = _grating.diffraction_angle(m=m, wl=wavelength)

        _prism_second_pass: Prism = Prism(
            glass=self.cross_disperser.glass,
            apex_angle=self.cross_disperser.apex_angle,
            min_wl=self.cross_disperser.min_wl,
            max_wl=self.cross_disperser.max_wl,
            incidence_angle=self.cross_disperser.incidence_angle
            - 2 * self.grating.gamma_angle
            - 2 * prism_first_pass_angle,
        )

        prism_second_pass_angle: Quantity = 2 * _prism_second_pass.dispersion_angle(
            wl=wavelength,
            field_angle=grating_angle,
        )

        return (
            (prism_second_pass_angle / self.camera_plate_scale).to(u.mm).value,
            (grating_angle / self.camera_plate_scale).to(u.mm).value,
        ) * u.mm
