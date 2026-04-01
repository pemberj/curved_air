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
    TwoPrisms,
)

if TYPE_CHECKING:
    from matplotlib import pyplot as plt

    from curved_air.glass import Glass
    from curved_air.type_hinting import ArrayQuantity, ScalarQuantity


@dataclass
class EchelleSpectrograph:
    name: str = ""
    grating: EchelleGrating = field(default_factory=EchelleGrating)
    cross_disperser: CrossDisperser = field(default_factory=Prism)
    detector: Detector = field(default_factory=Detector)

    beam_diameter: ScalarQuantity | None = None
    min_wl: ScalarQuantity = 390 * u.nm
    max_wl: ScalarQuantity = 950 * u.nm
    collimator_focal_length: ScalarQuantity | None = 1125 * u.mm
    slit_width: ScalarQuantity = 40 * u.um
    slit_length: ScalarQuantity = 0.5 * u.mm
    slit_native_focal_ratio: float = 3.85
    camera_focal_length: ScalarQuantity = 730 * u.mm

    def __post_init__(self) -> None:
        if self.beam_diameter is None:
            self.beam_diameter: ScalarQuantity = self.grating.beam_diameter

        self.grating.min_wl: ScalarQuantity = self.min_wl
        self.grating.max_wl: ScalarQuantity = self.max_wl

        self.cross_disperser.min_wl: ScalarQuantity = self.min_wl
        self.cross_disperser.max_wl: ScalarQuantity = self.max_wl

        if isinstance(self.cross_disperser, Prism):
            if self.min_wl < self.cross_disperser.glass.wlmin * u.um:
                raise ValueError(
                    f"Minimum wavelength {self.min_wl} is below the glass transmission "
                    + f"limit of {self.cross_disperser.glass.wlmin * u.um}."
                )
            if self.max_wl > self.cross_disperser.glass.wlmax * u.um:
                raise ValueError(
                    f"Maximum wavelength {self.max_wl} is above the glass transmission "
                    + f"limit of {self.cross_disperser.glass.wlmax * u.um}."
                )

            # Update incidence angle based on the wavelength range
            self.cross_disperser.incidence_angle: ScalarQuantity = (
                self.cross_disperser.find_incidence_angle()
            )

        if isinstance(self.cross_disperser, TwoPrisms):
            self.cross_disperser.prism1.min_wl: ScalarQuantity = self.min_wl
            self.cross_disperser.prism1.max_wl: ScalarQuantity = self.max_wl

            p1glass: Glass = self.cross_disperser.prism1.glass
            p2glass: Glass = self.cross_disperser.prism2.glass
            if self.min_wl < p1glass.wlmin * u.um or self.min_wl < p2glass.wlmin * u.um:
                raise ValueError(
                    f"Minimum wavelength {self.min_wl} is below one or more of the glass"
                    f"transmission limits of {p1glass.wlmin * u.um} and {p2glass.wlmin * u.um}."
                )
            if self.max_wl > p1glass.wlmax * u.um or self.max_wl > p2glass.wlmax * u.um:
                raise ValueError(
                    f"Maximum wavelength {self.max_wl} is above one or more of the glass"
                    f"transmission limits of {p1glass.wlmax * u.um} and {p2glass.wlmax * u.um}."
                )

            # Update incidence angles based on the wavelength range
            self.cross_disperser.prism1.incidence_angle: ScalarQuantity = (
                self.cross_disperser.prism1.find_incidence_angle()
            )

            self.cross_disperser.prism2.incidence_angle: ScalarQuantity = (
                self.cross_disperser.prism2.find_incidence_angle()
            )

            self.cross_disperser.incidence_angle: ScalarQuantity = (
                self.cross_disperser.prism1.incidence_angle
            )

    @property
    def mid_wavelength(self) -> ScalarQuantity:
        return (self.max_wl + self.min_wl) / 2

    @property
    def mid_wavelength_rounded(self) -> ScalarQuantity:
        """
        The mid wavelength rounded to the nearest 50nm, for use in plots and displays
        where a clean number is preferable.
        """
        mid_wavelength: ScalarQuantity = (
            round(self.mid_wavelength.to(u.nm).value / 50) * 50 * u.nm
        )
        return mid_wavelength

    @property
    def mid_blaze_wavelength(self) -> ScalarQuantity:
        """
        First, find the mid wavelength of the spectrograph's bandpass. Then find the
        diffraction order in which that wavelength appears within the FSR. Finally,
        find and return the blaze wavelength of that order.
        """
        mid_order: int = self.grating.order_number(
            wl=self.mid_wavelength, quantised=True
        )
        return self.grating.blaze_wavelength(m=mid_order)

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

        return float((self.slit_image_width() / self.detector.pixel_size).si.value)

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

    def slit_image_width(
        self, m: int | None = None, wl: ScalarQuantity | None = None
    ) -> ScalarQuantity:
        """Width of the slit image in the image plane"""

        if wl is not None:
            if m is None:
                m: int = self.grating.order_number(wl=wl, quantised=True)
            return (
                self.slit_width
                * self.slit_magnification
                * self.anamorphic_magnification(wl=wl, m=m)
            )

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
        """ """

        if m is None:
            # Find the diffraction order in which this wavelength is within the FSR
            m: int = self.grating.order_number(wl=wl, quantised=True)

        angular_dispersion: ScalarQuantity = self.grating.angular_dispersion(m=m, wl=wl)
        spatial_dispersion: ScalarQuantity = (
            angular_dispersion / self.camera_plate_scale
        )

        return (1 / (spatial_dispersion / self.slit_image_width(m=m, wl=wl))).to(
            wl.unit
        )

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
        wl: ScalarQuantity | ArrayQuantity,
        m: int | None = None,
    ) -> float | list[float]:
        """
        See also docstrings for resolution_element and slit_image_width.

        Currently this function will only return the spectral resolution at the blaze
        wavelength of the relevant order
        """

        if wl.isscalar:
            if m is None:
                m: int = self.grating.order_number(wl=wl, quantised=True)

            wl_blaze: ScalarQuantity = self.grating.blaze_wavelength(m=m)
            return float((wl / self.resolution_element(wl=wl_blaze, m=m)).si.value)

        if m is None:
            m: list[int] = [
                self.grating.order_number(wl=iwl, quantised=True) for iwl in wl
            ]
        assert isinstance(m, list)
        assert len(m) == len(wl)

        wls_blaze: list[ScalarQuantity] = [
            self.grating.blaze_wavelength(m=im) for im in m
        ]

        return [
            (iwl / self.resolution_element(wl=iwl, m=im)).si.value
            for iwl, im in zip(wls_blaze, m, strict=True)
        ]

        # return [(iwl / self.resolution_element(wl=iwl, m=m)).si.value for iwl in wl]

    @overload
    def anamorphic_magnification(
        self,
        wl: None = None,
        m: int | None = None,
        FSR_fraction: float = 1.0,
    ) -> tuple[float, float]: ...

    @overload
    def anamorphic_magnification(
        self,
        wl: ScalarQuantity,
        m: int | None = None,
        FSR_fraction: float = 1.0,
    ) -> tuple[float, float]: ...

    def anamorphic_magnification(
        self,
        wl: ScalarQuantity | None = None,
        m: int | None = None,
        FSR_fraction: float = 1.0,
    ) -> float | tuple[float, float]:
        """
        Gives the effective magnification factor at each end of a given spectral order.
        If no order is specified, the order corresponding to the maximum wavelength is
        used.

        Anamorphic magnification arises from a difference in projected grating (pupil)
        size as a function of diffraction angle. For longer (larger opening angle)
        orders, this effect is larger.
        """
        if wl is not None:
            if m is None:
                m: int = self.grating.order_number(wl=wl, quantised=True)
            return self.grating.anamorphic_magnification(m=m, wl=wl)

        # If no wavelength is specified, return the anamorphic magnification at each end
        # of the given or the maximum order. This is useful for calculating the range of
        # anamorphic magnification or the ratio from one end of the order to the other.

        if m is None:
            m: int = self.grating.order_number(self.max_wl, quantised=True)

        wls: ArrayQuantity = self.grating.order_FSR_bounds(
            m=m, FSR_fraction=FSR_fraction
        )

        return (
            self.grating.anamorphic_magnification(m=m, wl=wls[0]),
            self.grating.anamorphic_magnification(m=m, wl=wls[1]),
        )

    def print_parameters(self, column_width: int = 35) -> EchelleSpectrograph:

        w: int = column_width

        print(f"{'Collimated beam diameter:':<{w}}", end="")
        print(f"{self.beam_diameter:.3g}")

        print(f"{'Slit magnification factor:':<{w}}", end="")
        print(f"{self.slit_magnification:.3g}x")

        print(f"{'Slit length at detector:':<{w}}", end="")
        print(
            f"{(self.slit_length * self.slit_magnification / self.detector.pixel_size).si:.3g} pixels"  # noqa: E501
        )

        print(f"{'Bluest orders separation:':<{w}}", end="")
        print(
            f"{(self.bluest_order_image_plane_separation / self.detector.pixel_size).si:.3g} pixels"  # noqa: E501
        )

        print(f"{'Reddest orders separation:':<{w}}", end="")
        print(
            f"{(self.reddest_order_image_plane_separation / self.detector.pixel_size).si:.3g} pixels"  # noqa: E501
        )

        print(f"{'Slit width (object plane):':<{w}}", end="")
        print(f"{self.slit_width:.3g}")

        print(f"{'Slit length (object plane):':<{w}}", end="")
        print(f"{self.slit_length:.3g}")

        print(f"{'Slit width (image plane):':<{w}}", end="")
        print(f"{(self.slit_width * self.slit_magnification):.3g}")

        print(f"{'Sampling of slit image:':<{w}}", end="")
        print(f"{self.slit_sampling:.3g} pixels")

        print(
            f"{f'Spectral resolution @ {self.mid_wavelength_rounded:.0f}:':<{w}}",
            end="",
        )
        print(f"{self.spectral_resolution(wl=self.mid_blaze_wavelength):,.0f}")

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
