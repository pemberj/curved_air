"""
curved_air

optics

Definitions of several optical components, including Prisms, compound prisms (here
PrismStacks), ADCs (a PrismStack to be optimised to match the dispersion of air),
Gratings, and Etalons.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any, overload

import numpy as np
from astropy import units as u
from astropy.units import Quantity

from curved_air.glass import Glass
from curved_air.other import Air

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from curved_air.type_hinting import ArrayQuantity, ScalarQuantity


@dataclass
class Disperser:
    name: str = ""


@dataclass
class EchelleGrating(Disperser):
    min_wl: ScalarQuantity = 0.38 * u.um
    max_wl: ScalarQuantity = 0.95 * u.um

    catalogue_number: str = ""
    groove_frequency: ScalarQuantity = 31.6 / u.mm
    blaze_angle: ScalarQuantity = 76 * u.deg
    gamma_angle: ScalarQuantity = 0 * u.deg
    ruled_width: ScalarQuantity = 400 * u.mm
    groove_length: ScalarQuantity | None = None

    @cached_property
    def R(self) -> float:
        return round(np.tan(self.blaze_angle).value, 1)

    @cached_property
    def m_times_lambda(self) -> Quantity:
        return (2 * np.cos(self.gamma) * self.sigma * np.sin(self.blaze_angle)).to(u.um)

    @property
    def gamma(self) -> ScalarQuantity:
        return self.gamma_angle

    @property
    def theta_B(self) -> ScalarQuantity:
        return self.blaze_angle

    @property
    def groove_period(self) -> ScalarQuantity:
        return 1 / self.groove_frequency

    @cached_property
    def sigma(self) -> ScalarQuantity:
        return self.groove_period.to(u.um)

    @cached_property
    def gpmm(self) -> ScalarQuantity:
        return self.groove_frequency.to(1 / u.mm)

    @cached_property
    def max_order(self) -> int:
        return int((self.m_times_lambda / self.min_wl) - 0.5)

    @cached_property
    def min_order(self) -> int:
        return int((self.m_times_lambda / self.max_wl) + 0.5)

    @cached_property
    def diffraction_orders(self) -> list[int]:
        assert isinstance(self.min_wl, Quantity)
        assert self.min_wl.isscalar

        assert isinstance(self.max_wl, Quantity)
        assert self.max_wl.isscalar

        min_m: int = self.order_number(self.max_wl, quantised=True)
        max_m: int = self.order_number(self.min_wl, quantised=True)

        return list(range(min_m, max_m + 1))

    @cached_property
    def beam_diameter(self) -> ScalarQuantity:
        return self.ruled_width / self.R

    def order_containing_500nm(self, quantised: bool = True) -> int | float:
        return self.order_number(wl=500 * u.nm, quantised=quantised)

    @overload
    def blaze_wavelength(self, m: int) -> ScalarQuantity: ...

    @overload
    def blaze_wavelength(
        self, m: list[int] | tuple[int] | np.ndarray
    ) -> ArrayQuantity: ...

    def blaze_wavelength(
        self, m: int | list[int] | tuple[int] | np.ndarray
    ) -> Quantity:
        if isinstance(m, list | tuple | np.ndarray):
            return [self.blaze_wavelength(im).to(u.um).value for im in m] * u.um

        return (self.m_times_lambda / m).to(u.um)

    @overload
    def order_number(
        self,
        wl: float | ScalarQuantity,
        quantised: bool = True,
    ) -> int: ...

    @overload
    def order_number(
        self,
        wl: float | ScalarQuantity,
        quantised: bool = False,
    ) -> float: ...

    @overload
    def order_number(
        self,
        wl: tuple[float] | list[float] | ArrayQuantity,
        quantised: bool = True,
    ) -> list[int]: ...

    @overload
    def order_number(
        self,
        wl: tuple[float] | list[float] | ArrayQuantity,
        quantised: bool = False,
    ) -> list[float]: ...

    def order_number(
        self,
        wl: float | tuple[float] | list[float] | Quantity = 500 * u.nm,
        quantised: bool = False,
    ) -> int | list[int] | float | list[float]:
        """ """
        if isinstance(wl, tuple | list) or (
            isinstance(wl, Quantity) and not wl.isscalar
        ):
            return [self.order_number(iwl) for iwl in wl]

        if isinstance(wl, int | float):
            print(f"Input was unitless value {wl}. Defaulting to {wl * u.um}")
            wl *= u.um

        if quantised:
            return int(round(float((self.m_times_lambda / wl).si), 0))

        return float((self.m_times_lambda / wl).si)

    @overload
    def order_FSR(self, m: int, FSR_fraction: float = 1.0) -> ScalarQuantity: ...

    @overload
    def order_FSR(
        self, m: list[int] | tuple[int] | np.ndarray, FSR_fraction: float = 1.0
    ) -> ArrayQuantity: ...

    def order_FSR(
        self,
        m: int | list[int] | tuple[int] | np.ndarray,
        FSR_fraction: float = 1.0,
    ) -> ScalarQuantity | ArrayQuantity:
        """
        The free spectral range (FSR) of a given diffraction order is the spectral
        width (in units of wavelength) of wavelengths that appear closest to THIS
        order's blaze peak. Any further and the wavelength in question appears closer
        to another order's blaze peak.
        """
        if isinstance(m, list | tuple | np.ndarray):
            ms: list[int] = [int(im) for im in m]
            return [self.order_FSR(m).to(u.nm).value for m in ms] * u.nm

        assert isinstance(m, int), f"Expected int, got {m}, {type(m)}"

        m_wavelength: ScalarQuantity = self.blaze_wavelength(m=m)
        m_plus_one_wavelength: ScalarQuantity = self.blaze_wavelength(m=m + 1)

        return FSR_fraction * (m_wavelength - m_plus_one_wavelength).to(u.nm)

    def order_FSR_bounds(self, m: int, FSR_fraction: float = 1.0) -> ArrayQuantity:
        """
        Return the wavelength bounds that define the given order's free spectral range.
        These bounds can be used to generate a grid of wavelengths that trace out this
        order.
        """
        FSR: ScalarQuantity = (
            self.order_FSR(m=m, FSR_fraction=FSR_fraction).to(u.nm).value
        )
        blaze_wavelength: ScalarQuantity = self.blaze_wavelength(m=m).to(u.nm).value

        return (blaze_wavelength - FSR / 2, blaze_wavelength + FSR / 2) * u.nm

    def order_angular_FSR_bounds(
        self, m: int, FSR_fraction: float = 1.0
    ) -> ArrayQuantity:
        """
        Instead of returning the order bounds based on the FSR in wavelength units,
        return the FSR based on an order's opening angle. This should make the orders
        symmetric about the blaze peak -- much nicer!
        """

        return [
            self.wavelength_at_angle(m=m, angle=self.blaze_angle + a).to(u.um).value
            for a in self.order_opening_angle(
                m=m,
                FSR_fraction=FSR_fraction,
                return_bounds=True,
            )
        ] * u.um

    @overload
    def order_opening_angle(
        self, m: int, FSR_fraction: float = 1.0, return_bounds: bool = False
    ) -> Quantity: ...

    @overload
    def order_opening_angle(
        self, m: list[int], FSR_fraction: float = 1.0, return_bounds: bool = False
    ) -> tuple[Quantity]: ...

    def order_opening_angle(
        self,
        m: int | list[int],
        FSR_fraction: float = 1.0,
        return_bounds: bool = False,
    ) -> Quantity | list[Quantity]:
        if isinstance(m, list | tuple):
            return [self.order_opening_angle(im) for im in m]

        if return_bounds:
            return (
                (-FSR_fraction * np.tan(self.blaze_angle) * u.radian / m).to(u.deg),
                (FSR_fraction * np.tan(self.blaze_angle) * u.radian / m).to(u.deg),
            )

        # Otherwise, just return the opening angle
        return (FSR_fraction * 2 * np.tan(self.blaze_angle) * u.rad / m).to(u.deg)

    @overload
    def diffraction_angle(self, m: int, wl: ScalarQuantity) -> ScalarQuantity: ...

    @overload
    def diffraction_angle(self, m: int, wl: list | ArrayQuantity) -> ArrayQuantity: ...

    def diffraction_angle(
        self,
        m: int,
        wl: Quantity | list[Quantity] = 550 * u.nm,
    ) -> Quantity:
        if isinstance(wl, Quantity) and wl.size > 1:
            return [
                self.diffraction_angle(m, iwl).to(u.degree).value for iwl in wl
            ] * u.degree

        if isinstance(wl, list | tuple):
            assert all(isinstance(iwl, Quantity) for iwl in wl)
            return [
                self.diffraction_angle(m=m, wl=iwl).to(u.degree).value for iwl in wl
            ] * u.degree

        if isinstance(wl, float):
            print(f"No wavelength units passed ({wl}). Assuming nanometers")
            wl *= u.nm

        arcsin_argument: float = (
            (m * wl).to(u.um).value / (np.cos(self.gamma) * self.sigma.to(u.um).value)
        ) - np.sin(self.blaze_angle)
        return (np.arcsin(arcsin_argument) - self.blaze_angle).to(u.deg)

    def wavelength_at_angle(self, m: int, angle: ScalarQuantity) -> ScalarQuantity:
        # angle += self.blaze_angle

        return (
            np.cos(self.gamma) * self.sigma * (np.sin(self.blaze_angle) + np.sin(angle))
        ) / m

    def order_FSR_bounds_from_angle(
        self,
        m: int,
    ) -> tuple[ScalarQuantity, ScalarQuantity]:
        angle: ScalarQuantity = self.order_opening_angle(m=m) / 2

        return (
            self.wavelength_at_angle(m=m, angle=self.blaze_angle - angle),
            self.wavelength_at_angle(m=m, angle=self.blaze_angle + angle),
        )

    def aspect_ratio(
        self,
        min_wl: ScalarQuantity | None = None,
        max_wl: ScalarQuantity | None = None,
    ) -> float:
        if not min_wl:
            min_wl: ScalarQuantity = self.min_wl
        if not max_wl:
            max_wl: ScalarQuantity = self.max_wl

        blue_angle: ScalarQuantity = self.order_opening_angle(
            m=self.order_number(min_wl)
        ).to(u.deg)
        red_angle: ScalarQuantity = self.order_opening_angle(
            m=self.order_number(max_wl)
        ).to(u.deg)

        # print(f"{blue_angle=}")
        # print(f"{red_angle=}")

        return red_angle / blue_angle

    def angular_dispersion(self, m: int, wl: None | Quantity = None) -> Quantity:
        """
        Returns the angular dispersion of the grating at a given order and wavelength

        If no wavelength is passed, the dispersion is calculated at the blaze wavelength

        Args:
            m (int): _description_
            wl (Quantity, optional): _description_. Defaults to 550*u.nm.

        Returns:
            Quantity: _description_
        """

        if wl is None:
            wl: ScalarQuantity = self.blaze_wavelength(m=m)

        delta_wl: ScalarQuantity = self.order_FSR(m=m) / 100

        delta_theta: ScalarQuantity = (
            self.diffraction_angle(m=m, wl=wl + delta_wl).value
            - self.diffraction_angle(m=m, wl=wl).value
        ) * u.deg

        return delta_theta / delta_wl

    def anamorphic_magnification(self, m: int, wl: Quantity) -> float:
        return np.cos(self.blaze_angle) / np.cos(
            self.blaze_angle - self.diffraction_angle(m=m, wl=wl)
        )

    def relative_anamorphic_magnification(
        self,
        m: int | None = None,
        FSR_fraction: float = 1.0,
    ) -> float:
        """
        Returns the relative anamorphic magnification of the grating for wavelengths at
        the ends of a given diffraction order.
        """

        if m is None:
            m: int = self.order_number(self.max_wl, quantised=True)

        wls: ArrayQuantity = self.order_FSR_bounds(m=m, FSR_fraction=FSR_fraction)

        return self.anamorphic_magnification(
            m=m, wl=wls[0]
        ) / self.anamorphic_magnification(m=m, wl=wls[1])

    def relative_intensity(
        self,
        # m: int,
        wl: Quantity,
    ) -> float | NDArray[np.floating]:
        """
        Calculate the relative intensity of a particular wavelength or wavelengths in a
        given diffraction order.

        TODO: fix this up!

        At the moment this finds the diffraction order in which the wavelength is
        closest to the blaze peak. Maybe I should put the `m` parameter back in,
        allowing the user to specify the order they want the efficiency for?
        """

        # N = 400 * u.mm / self.groove_period  # Number of grooves illuminated

        frac_m: float | np.ndarray = np.array(self.order_number(wl=wl, quantised=False))

        # beta = self.diffraction_angle(m=m, wl=wl)

        # p: Quantity["angle"] = (  # noqa: F821, UP037
        #     (np.pi * self.groove_period / wl) * (np.sin(self.theta_B) + np.sin(beta))
        # ).si * u.radian

        # H: Quantity["angle"] = (  # noqa: F821, UP037
        #     (np.pi * self.groove_period * np.cos(self.theta_B) / wl)
        #     * np.sin(self.theta_B - beta)
        # ).si * u.radian

        # return (np.sin(p) / p) ** 2  # * (np.sin(N * p) ** 2 / np.sin(p) ** 2)

        return 1 - np.sinc(frac_m * u.radian) ** 2

    @property
    def reddest_order_separation(self) -> ScalarQuantity:
        return self.order_FSR(
            m=self.order_number(wl=self.max_wl, quantised=True), FSR_fraction=1.0
        )

    def __str__(self) -> str:
        return (
            "Grating: "
            f"{self.catalogue_number}, "
            f"R{self.R}, {self.groove_frequency.to(1 / u.mm).value:.3g} grooves/mm"
        )

    def __repr__(self) -> str:
        return (
            "Grating("
            f"catalogue_number = {self.catalogue_number}, "
            f"groove_frequency = {self.groove_frequency:.3f}, "
            f"blaze_angle = {self.blaze_angle:.3f}, "
            f"gamma_angle = {self.gamma_angle:.3f}, "
            f"min_wl = {self.min_wl:.3f}, "
            f"max_wl = {self.max_wl:.3f})"
        )


@dataclass
class CrossDisperser(Disperser):
    incidence_angle: ScalarQuantity | None = None
    min_wl: ScalarQuantity | None = None
    max_wl: ScalarQuantity | None = None


@dataclass
class Prism(CrossDisperser):
    glass: Glass = field(default_factory=Glass)
    apex_angle: ScalarQuantity = 60 * u.deg

    def __post_init__(self) -> None:
        assert self.glass.name != "_default_"

        if self.incidence_angle is None:
            if self.min_wl is not None and self.max_wl is not None:
                self.incidence_angle: ScalarQuantity = self.find_incidence_angle()
            else:
                self.incidence_angle: ScalarQuantity = (
                    self.minimum_deviation_incidence_angle()
                )

        if self.min_wl is None:
            self.min_wl: ScalarQuantity = self.glass.wlmin * u.um

        if self.max_wl is None:
            self.max_wl: ScalarQuantity = self.glass.wlmax * u.um

    @overload
    def refractive_index(self, wavelength: None) -> Callable: ...

    @overload
    def refractive_index(self) -> Callable: ...

    @overload
    def refractive_index(self, wavelength: ScalarQuantity) -> float: ...

    @overload
    def refractive_index(self, wavelength: ArrayQuantity) -> np.ndarray: ...

    def refractive_index(
        self, wavelength: Quantity | None = None
    ) -> Callable | float | np.ndarray:
        if wavelength is None:
            return self.glass.refractive_index

        return self.glass.refractive_index(wavelength)

    @property
    def n(self) -> Callable:
        """Alias of refractive_index"""

        return self.refractive_index

    @cached_property
    def mid_wavelength(self) -> ScalarQuantity:
        """
        Calculate the mid-wavelength of the prism.
        """

        assert isinstance(self.min_wl, Quantity) and self.min_wl.isscalar
        assert isinstance(self.max_wl, Quantity) and self.max_wl.isscalar

        mid_refractive_index: float = (
            self.glass.refractive_index(self.min_wl)
            + self.glass.refractive_index(self.max_wl)
        ) / 2

        wls: NDArray[np.floating] = np.linspace(
            self.min_wl.to(u.um).value,
            self.max_wl.to(u.um).value,
            500,
        )

        return wls[np.argmin(abs(self.glass.n(wls) - mid_refractive_index))] * u.um

    @overload
    def deviation_angle(
        self,
        wl: ScalarQuantity,
        incidence_angle: ScalarQuantity | None = None,
        field_angle: ScalarQuantity = 0 * u.deg,
    ) -> ScalarQuantity: ...

    @overload
    def deviation_angle(
        self,
        wl: ArrayQuantity,
        incidence_angle: ScalarQuantity | None = None,
        field_angle: ScalarQuantity = 0 * u.deg,
    ) -> ArrayQuantity: ...

    @overload
    def deviation_angle(
        self,
        wl: ScalarQuantity,
        incidence_angle: ArrayQuantity,
        field_angle: ScalarQuantity = 0 * u.deg,
    ) -> ArrayQuantity: ...

    @overload
    def deviation_angle(
        self,
        wl: ScalarQuantity,
        incidence_angle: ScalarQuantity | None = None,
        field_angle: ArrayQuantity = 0 * u.deg,
    ) -> ArrayQuantity: ...

    @overload
    def deviation_angle(
        self,
        wl: ArrayQuantity,
        incidence_angle: ArrayQuantity,
        field_angle: ArrayQuantity,
    ) -> ArrayQuantity: ...

    def deviation_angle(
        self,
        wl: ScalarQuantity = 0.6328 * u.um,
        incidence_angle: ScalarQuantity | None = None,
        field_angle: ScalarQuantity = 0 * u.deg,
    ) -> Quantity:
        if incidence_angle is None:
            assert isinstance(self.incidence_angle, Quantity)
            incidence_angle: ScalarQuantity = self.incidence_angle

        assert isinstance(wl, Quantity)
        assert isinstance(field_angle, Quantity)

        if not wl.isscalar:
            wl: ArrayQuantity = wl.value.flatten() * wl.unit
            wl_size: int = len(wl)
        else:
            wl_size: int = 1

        if not incidence_angle.isscalar:
            incidence_angle: ArrayQuantity = (
                incidence_angle.value.flatten() * incidence_angle.unit
            )
            incidence_angle_size: int = len(incidence_angle)
        else:
            incidence_angle_size: int = 1

        if not field_angle.isscalar:
            field_angle: ArrayQuantity = field_angle.value.flatten() * field_angle.unit
            field_angle_size: int = len(field_angle)
        else:
            field_angle_size: int = 1

        if any([wl_size > 1, incidence_angle_size > 1, field_angle_size > 1]):
            # Ensure all inputs are arrays of the same length, OR scalars
            if wl_size > 1:
                assert all(
                    [
                        (incidence_angle_size == wl_size or incidence_angle_size == 1),
                        (field_angle_size == wl_size or field_angle_size == 1),
                    ]
                )
            elif incidence_angle_size > 1:
                assert all(
                    [
                        (wl_size == incidence_angle_size or wl_size == 1),
                        (
                            field_angle_size == incidence_angle_size
                            or field_angle_size == 1
                        ),
                    ]
                )
            elif field_angle_size > 1:
                assert all(
                    [
                        (wl_size == field_angle_size or wl_size == 1),
                        (
                            incidence_angle_size == field_angle_size
                            or incidence_angle_size == 1
                        ),
                    ]
                )

            # Then broadcast scalars to arrays in the full size
            if wl_size == 1:
                wl: ArrayQuantity = (
                    np.full(max(incidence_angle_size, field_angle_size), wl.value)
                    * wl.unit
                )

            if incidence_angle_size == 1:
                incidence_angle: ArrayQuantity = (
                    np.full(max(wl_size, field_angle_size), incidence_angle.value)
                    * incidence_angle.unit
                )

            if field_angle_size == 1:
                field_angle: ArrayQuantity = (
                    np.full(max(wl_size, incidence_angle_size), field_angle.value)
                    * field_angle.unit
                )

            return [
                self.deviation_angle(wl=w, incidence_angle=i, field_angle=f).value
                for w, i, f in zip(wl, incidence_angle, field_angle, strict=False)
            ] * u.deg

        # Otherwise, all inputs are scalars
        i: ScalarQuantity = incidence_angle  # shorthand
        if field_angle.value != 0:
            effective_apex_angle: ScalarQuantity = self.effective_apex_angle(
                field_angle=field_angle
            )

            _self = Prism(
                glass=self.glass,
                apex_angle=effective_apex_angle,
                incidence_angle=self.incidence_angle,
            )

            return (
                i
                - _self.apex_angle
                + np.arcsin(
                    _self.glass.n(wl)
                    * np.sin(
                        _self.apex_angle - np.arcsin(np.sin(i) / _self.glass.n(wl))
                    )
                )
            )

        return (
            i
            - self.apex_angle
            + np.arcsin(
                self.glass.n(wl)
                * np.sin(self.apex_angle - np.arcsin(np.sin(i) / self.glass.n(wl)))
            )
        )

    @overload
    def exitance_angle(
        self, wl: ScalarQuantity, incidence_angle: ScalarQuantity | None = None
    ) -> ScalarQuantity: ...

    @overload
    def exitance_angle(
        self, wl: ArrayQuantity, incidence_angle: ScalarQuantity | None = None
    ) -> ArrayQuantity: ...

    @overload
    def exitance_angle(
        self, wl: ScalarQuantity, incidence_angle: ArrayQuantity
    ) -> ArrayQuantity: ...

    def exitance_angle(
        self,
        wl: Quantity = 0.6328 * u.um,
        incidence_angle: Quantity | None = None,
    ) -> Quantity:
        if incidence_angle is None:
            assert isinstance(self.incidence_angle, Quantity)
            assert self.incidence_angle.isscalar
            incidence_angle: ScalarQuantity = self.incidence_angle

        return (
            self.deviation_angle(wl=wl, incidence_angle=incidence_angle)
            + self.apex_angle
            - incidence_angle
        )

    def find_incidence_angle(
        self,
        min_wl: ScalarQuantity | None = None,
        max_wl: ScalarQuantity | None = None,
    ) -> ScalarQuantity:
        """
        Find the incidence angle that provides a symmetric deviation for the extreme
        wavelengths that the prism is designed for.
        """

        if min_wl is None:
            assert isinstance(self.min_wl, Quantity)
            assert self.min_wl.isscalar
            min_wl: ScalarQuantity = self.min_wl

        if max_wl is None:
            assert isinstance(self.max_wl, Quantity)
            assert self.max_wl.isscalar
            max_wl: ScalarQuantity = self.max_wl

        min_wl_incidence: ScalarQuantity = self.minimum_deviation_incidence_angle(
            wl=min_wl
        )
        max_wl_incidence: ScalarQuantity = self.minimum_deviation_incidence_angle(
            wl=max_wl
        )

        return ((min_wl_incidence + max_wl_incidence) / 2).to(u.deg)

    def minimum_deviation_angle(
        self,
        wl: ScalarQuantity = 0.6328 * u.um,
    ) -> ScalarQuantity:
        minimum_deviation_angle: Quantity = (
            2 * np.arcsin(self.glass.n(wl) * np.sin(self.apex_angle / 2))
            - self.apex_angle
        )

        return minimum_deviation_angle.to(u.deg)

    @overload
    def minimum_deviation_incidence_angle(
        self,
        wl: ScalarQuantity = 632.8 * u.nm,
    ) -> ScalarQuantity: ...

    @overload
    def minimum_deviation_incidence_angle(
        self,
        wl: ArrayQuantity,
    ) -> ArrayQuantity: ...

    def minimum_deviation_incidence_angle(
        self,
        wl: Quantity = 632.8 * u.nm,
    ) -> Quantity:
        """
        Calculate the incidence angle that provides the minimum deviation condition for
        the prism at a given wavelength.
        """

        return (self.apex_angle + self.minimum_deviation_angle(wl)) / 2

    def effective_apex_angle(
        self,
        field_angle: ScalarQuantity = 0 * u.deg,
    ) -> ScalarQuantity:
        """
        I don't know why this works. The 74° is empirically determined, but matches very
        well up to a field angle of ~10°.
        """
        return self.apex_angle + 74 * u.deg * (1 - np.cos(field_angle))

    @overload
    def dispersion_angle(
        self, wl: ScalarQuantity, field_angle: ScalarQuantity = 0 * u.deg
    ) -> ScalarQuantity: ...

    @overload
    def dispersion_angle(
        self, wl: ArrayQuantity, field_angle: ScalarQuantity = 0 * u.deg
    ) -> ArrayQuantity: ...

    @overload
    def dispersion_angle(
        self, wl: ScalarQuantity, field_angle: ArrayQuantity
    ) -> ArrayQuantity: ...

    def dispersion_angle(
        self,
        wl: ScalarQuantity | ArrayQuantity,
        field_angle: Quantity | ArrayQuantity = 0 * u.deg,
    ) -> Quantity:
        """
        TODO: Allow for EITHER multiple wavelengths or multiple field angles as per overloads
        """

        if not wl.isscalar:
            wl_size: int = len(wl)
        else:
            wl_size: int = 1

        if not field_angle.isscalar:
            field_angle_size: int = len(field_angle)
        else:
            field_angle_size: int = 1

        if any([wl_size > 1, field_angle_size > 1]):
            # Ensure all inputs are arrays of the same length, OR scalars
            if wl_size > 1:
                assert field_angle_size == wl_size or field_angle_size == 1, (
                    f"{field_angle = }, {field_angle_size = }, {wl = }, {wl_size = }"
                )
            elif field_angle_size > 1:
                assert wl_size == field_angle_size or wl_size == 1

            # Then broadcast scalars to arrays in the full size
            if wl_size == 1:
                wl: ArrayQuantity = np.full(field_angle_size, wl.value) * wl.unit

            if field_angle_size == 1:
                field_angle: ArrayQuantity = (
                    np.full(wl_size, field_angle.value) * field_angle.unit
                )

            return [
                self.dispersion_angle(wl=w, field_angle=f).value
                for w, f in zip(wl, field_angle, strict=False)
            ] * u.deg

        # Otherwise, all inputs are scalars
        zero_angle: ScalarQuantity = self.deviation_angle(wl=self.mid_wavelength)

        if field_angle.value != 0:
            effective_apex_angle: Quantity = self.effective_apex_angle(
                field_angle=field_angle
            )

            _self = Prism(
                glass=self.glass,
                apex_angle=effective_apex_angle,
                incidence_angle=self.incidence_angle,
            )

            return (_self.deviation_angle(wl=wl) - zero_angle).to(u.deg)

        return (self.deviation_angle(wl=wl) - zero_angle).to(u.deg)

    def angular_spread(
        self,
        min_wl: ScalarQuantity | None = None,
        max_wl: ScalarQuantity | None = None,
    ) -> Quantity:
        """ """

        if min_wl is None:
            assert isinstance(self.min_wl, Quantity)
            assert self.min_wl.isscalar
            min_wl: ScalarQuantity = self.min_wl

        if max_wl is None:
            assert isinstance(self.max_wl, Quantity)
            assert self.max_wl.isscalar
            max_wl: ScalarQuantity = self.max_wl

        min_angle: ScalarQuantity = (-self.deviation_angle(wl=min_wl)).to(u.deg)
        max_angle: ScalarQuantity = (-self.deviation_angle(wl=max_wl)).to(u.deg)
        return max_angle - min_angle

    def local_angular_dispersion(self, wl: Quantity) -> Quantity:
        delta_wavelength: ScalarQuantity = 1 * u.nm

        wl1: ScalarQuantity = wl - delta_wavelength / 2
        wl2: ScalarQuantity = wl + delta_wavelength / 2

        return self.angular_spread(min_wl=wl1, max_wl=wl2) / delta_wavelength

    def trace(
        self,
        wls: Quantity | None = None,
        incidence_angle: Quantity | None = None,
        debug_level: float = 0,
    ) -> ArrayQuantity:
        """
        Trace a set of rays through the Prism, solving Snell's Law at each
        interface
        """

        if wls is None:
            wls: ArrayQuantity = [0.4861, 0.5893, 0.6563] * u.um

        assert isinstance(wls, Quantity)

        if incidence_angle is None:
            assert isinstance(self.incidence_angle, Quantity)
            assert self.incidence_angle.isscalar
            incidence_angle: ScalarQuantity = self.incidence_angle

        assert isinstance(incidence_angle, Quantity)
        assert incidence_angle.isscalar

        try:
            rays: ArrayQuantity = incidence_angle * np.ones(len(wls))
        except TypeError:  # wls is a single value
            rays: ScalarQuantity = incidence_angle

        previous_angle: ScalarQuantity = 0 * u.degree

        rays: Quantity = _refract(
            n_1=Air().refractive_index(wls),
            n_2=self.refractive_index,
            theta_in=rays - previous_angle,
            wavelength=wls,
        )

        if debug_level > 0:
            print(f"Refracting through a {self.glass} prism of {self.apex_angle}")

        # Refract through the last surface into the exit medium
        rays: Quantity = _refract(
            n_1=self.refractive_index(wls),
            n_2=Air().refractive_index(wls),
            theta_in=(rays - self.apex_angle).to(u.deg),
            wavelength=wls,
        )

        return rays.to(u.degree)

    def __repr__(self) -> str:
        return (
            f"Prism(name={self.name}, "
            f"glass={self.glass.name}, "
            f"apex_angle={self.apex_angle:.0f}, "
            f"min_wl={self.min_wl:.0f}, "
            f"max_wl={self.max_wl:.0f}, "
            f"incidence_angle={self.incidence_angle:.3f})"
        )


@dataclass
class TwoPrisms(CrossDisperser):
    prism1: Prism = field(default_factory=Prism)
    prism2: Prism | None = None

    def __post_init__(self) -> None:
        if self.prism2 is None:
            self.prism2: Prism = self.prism1

        self.prism1.min_wl: Quantity | None = self.min_wl
        self.prism2.min_wl: Quantity | None = self.min_wl

        if self.incidence_angle is None:
            self.incidence_angle: ScalarQuantity = self.prism1.incidence_angle

        if self.min_wl is None:
            assert isinstance(self.prism1.min_wl, Quantity)
            assert isinstance(self.prism2.min_wl, Quantity)
            self.min_wl: ScalarQuantity = min(self.prism1.min_wl, self.prism2.min_wl)

        if self.max_wl is None:
            assert isinstance(self.prism1.max_wl, Quantity)
            assert isinstance(self.prism2.max_wl, Quantity)
            self.max_wl: ScalarQuantity = max(self.prism1.max_wl, self.prism2.max_wl)

    def dispersion_angle(
        self,
        wl: Quantity = 550 * u.nm,
        field_angle: ScalarQuantity = 0 * u.deg,
    ) -> Quantity:
        """
        Calculate the total dispersion angle from both prisms.
        """

        first_prism_angle: Quantity = self.prism1.dispersion_angle(
            wl=wl, field_angle=field_angle
        )

        assert isinstance(self.prism2.incidence_angle, Quantity)
        p2 = Prism(
            glass=self.prism2.glass,
            apex_angle=self.prism2.apex_angle,
            incidence_angle=(
                first_prism_angle.to(u.deg).value
                + self.prism2.incidence_angle.to(u.deg).value
            )
            * u.deg,
        )

        second_prism_angle: Quantity = p2.dispersion_angle(
            wl=wl, field_angle=field_angle
        )

        return second_prism_angle

    def deviation_angle(
        self,
        wl: Quantity = 550 * u.nm,
        field_angle: ScalarQuantity = 0 * u.deg,
    ) -> Quantity:
        """
        Calculate the total deviation angle from both prisms.
        """

        first_prism_angle: Quantity = self.prism1.deviation_angle(
            wl=wl,
            field_angle=field_angle,
        )

        assert isinstance(self.prism2.incidence_angle, Quantity)
        p2 = Prism(
            glass=self.prism2.glass,
            apex_angle=self.prism2.apex_angle,
            incidence_angle=(
                first_prism_angle.to(u.deg).value
                + self.prism2.incidence_angle.to(u.deg).value
            )
            * u.deg,
        )

        second_prism_angle: Quantity = p2.deviation_angle(
            wl=wl,
            field_angle=field_angle,
        )

        return second_prism_angle


@dataclass
class Grism(CrossDisperser):
    apex_angle: ScalarQuantity = 0 * u.deg
    glass: Glass = field(default_factory=Glass)
    beam_diameter: ScalarQuantity = 200 * u.mm
    line_spacing: ScalarQuantity = 110 / u.mm
    diffraction_order: int = 1

    def __post_init__(self) -> None:
        if self.incidence_angle is None:
            self.incidence_angle = 0 * u.deg

    @property
    def groove_period(self) -> ScalarQuantity:
        return 1 / self.line_spacing

    def refraction_angle(
        self,
        wl: Quantity = 550 * u.nm,
        incidence_angle: Quantity | None = None,
        field_angle: Quantity = 0 * u.deg,
    ) -> Quantity:
        """
        TODO: Allow for array inputs?
        """
        if incidence_angle is None:
            assert isinstance(self.incidence_angle, Quantity)
            assert self.incidence_angle.isscalar
            incidence_angle: ScalarQuantity = self.incidence_angle

        assert isinstance(incidence_angle, Quantity)

        # Just shorthand
        i: Quantity = incidence_angle

        if hasattr(wl.value, "__len__") and hasattr(field_angle.value, "__len__"):
            return [
                self.dispersion_angle(wl=w, field_angle=f).value
                for w, f in zip(wl, field_angle, strict=False)
            ] * u.deg

        if field_angle.value != 0:
            effective_apex_angle = self.effective_apex_angle(field_angle=field_angle)

            _self: Grism = Grism(
                glass=self.glass,
                apex_angle=effective_apex_angle,
                incidence_angle=self.incidence_angle,
            )

            return (
                i
                - _self.apex_angle
                + np.arcsin(
                    _self.glass.n(wl)
                    * np.sin(
                        _self.apex_angle - np.arcsin(np.sin(i) / _self.glass.n(wl))
                    )
                )
            )

        return (
            i
            - self.apex_angle
            + np.arcsin(
                self.glass.n(wl)
                * np.sin(self.apex_angle - np.arcsin(np.sin(i) / self.glass.n(wl)))
            )
        )

    @overload
    def diffraction_angle(self, wl: ScalarQuantity | float) -> ScalarQuantity: ...

    @overload
    def diffraction_angle(self, wl: ArrayQuantity) -> ArrayQuantity: ...

    def diffraction_angle(
        self,
        wl: Quantity | float = 550 * u.nm,
    ) -> Quantity:
        """
        Calculate the diffraction angle for a given order and wavelength.
        """
        if isinstance(wl, float):
            print(f"No wavelength units passed ({wl}). Assuming nanometers")
            wl *= u.nm

        assert isinstance(wl, Quantity)

        return (
            np.arcsin(
                wl.to(u.um).value
                / (self.groove_period.to(u.um).value * np.cos(self.apex_angle))
            )
            - self.apex_angle
        ).to(u.deg)

    @overload
    def dispersion_angle(
        self, wl: ScalarQuantity | float, field_angle: ScalarQuantity
    ) -> ScalarQuantity: ...

    @overload
    def dispersion_angle(
        self, wl: ArrayQuantity, field_angle: ScalarQuantity
    ) -> ArrayQuantity: ...

    def dispersion_angle(
        self,
        wl: Quantity = 550 * u.nm,
        field_angle: ScalarQuantity = 0 * u.deg,
    ) -> Quantity:
        """
        Calculate the dispersion angle for a given wavelength and field angle.
        """
        if isinstance(wl, float):
            print(f"No wavelength units passed ({wl}). Assuming nanometers")
            wl *= u.nm

        return (self.diffraction_angle(wl=wl) + field_angle - self.incidence_angle).to(
            u.deg
        )


@dataclass
class SimpleGrating(CrossDisperser):
    """ """

    groove_frequency: ScalarQuantity = 150 / u.mm
    m: int = 1
    incidence_angle: ScalarQuantity = 0 * u.deg

    @property
    def groove_period(self) -> ScalarQuantity:
        return 1 / self.groove_frequency

    @overload
    def diffraction_angle(self, wl: ScalarQuantity | float) -> ScalarQuantity: ...

    @overload
    def diffraction_angle(self, wl: ArrayQuantity) -> ArrayQuantity: ...

    def diffraction_angle(
        self,
        wl: Quantity | float = 550 * u.nm,
    ) -> ScalarQuantity | ArrayQuantity:
        if isinstance(wl, float):
            print(f"No wavelength units passed ({wl}). Assuming nanometers")
            wl *= u.nm

        return np.arcsin((self.m * wl / self.groove_period).si).to(u.deg)

    @property
    def central_angle(self) -> ScalarQuantity:
        assert isinstance(self.min_wl, Quantity) and self.min_wl.isscalar
        beta_min_wl: ScalarQuantity = self.diffraction_angle(wl=self.min_wl)

        assert isinstance(self.max_wl, Quantity) and self.max_wl.isscalar
        beta_max_wl: ScalarQuantity = self.diffraction_angle(wl=self.max_wl)

        return (beta_min_wl + beta_max_wl) / 2

    @overload
    def dispersion_angle(
        self, wl: ScalarQuantity, field_angle: ScalarQuantity = 0 * u.deg
    ) -> ScalarQuantity: ...

    @overload
    def dispersion_angle(
        self, wl: ArrayQuantity, field_angle: ScalarQuantity = 0 * u.deg
    ) -> ArrayQuantity: ...

    def dispersion_angle(
        self,
        wl: Quantity,
        field_angle: ScalarQuantity = 0 * u.deg,
    ) -> Quantity:
        # No treatment of field angle here

        if field_angle.value != 0:
            # Field angle not supported in SimpleGrating!
            field_angle: ScalarQuantity = 0 * u.deg

        if isinstance(wl, list | tuple):
            return [self.dispersion_angle(iwl).to(u.deg).value for iwl in wl] * u.deg

        if isinstance(wl, float):
            print(f"No wavelength units passed ({wl}). Assuming nanometers")
            wl *= u.nm

        return (self.diffraction_angle(wl=wl) - self.central_angle).to(u.deg)


@overload
def _refract(
    n_1: float | np.ndarrary | Callable,
    n_2: float | np.ndarrary | Callable,
    theta_in: ScalarQuantity,
    wavelength: ScalarQuantity,
) -> ScalarQuantity: ...


@overload
def _refract(
    n_1: float | np.ndarrary | Callable,
    n_2: float | np.ndarrary | Callable,
    theta_in: ScalarQuantity,
    wavelength: ArrayQuantity,
) -> ArrayQuantity: ...


@overload
def _refract(
    n_1: float | np.ndarrary | Callable,
    n_2: float | np.ndarrary | Callable,
    theta_in: ArrayQuantity,
    wavelength: ScalarQuantity,
) -> ArrayQuantity: ...


def _refract(
    n_1: float | np.ndarrary | Callable,
    n_2: float | np.ndarrary | Callable,
    theta_in: ScalarQuantity | ArrayQuantity = 0 * u.deg,
    wavelength: ScalarQuantity | ArrayQuantity | None = None,
    # debug_level: float = 0.0,
) -> ScalarQuantity | ArrayQuantity:
    """
    TODO: Handle array inputs for either theta_in or wavelength

    TODO: implement debug levels and logging
    """

    # Compute the refracted angle of a ray or rays at a single material
    # interface
    # This can handle EITHER an array of input angles at a single wavelength
    # OR a single input angle at a range of wavelengths

    if wavelength is None:
        _n_1: Any = n_1
        _n_2: Any = n_2

    else:
        if callable(n_1):
            try:
                _n_1: float = n_1(wavelength)
            except TypeError:
                _n_1: Any = n_1

        elif isinstance(n_1, float | np.ndarray):
            _n_1: Any = n_1

        if callable(n_2):
            try:
                _n_2: float = n_2(wavelength)
            except TypeError:
                _n_2: Any = n_2

        elif isinstance(n_2, float | np.ndarray):
            _n_2: Any = n_2

    # By this point, _n_1 and _n_2 should be either floats, ints, or
    # lists/arrays of such
    assert not callable(_n_1)
    assert not callable(_n_2)

    theta_out: Quantity = np.arcsin((_n_1 / _n_2) * np.sin(theta_in))

    return theta_out.to(u.degree)


def main() -> None:
    from curved_air.glass import GlassCatalogue  # noqa: PLC0415

    F2: Glass = GlassCatalogue("Schott")["F2"]

    TwoES_XD: CrossDisperser = TwoPrisms(
        prism1=Prism(glass=F2, apex_angle=48.7 * u.deg),
        # prism2=Prism(glass=F2, apex_angle=48.7 * u.deg),
        min_wl=0.375 * u.um,
        max_wl=0.9 * u.um,
    )

    print(TwoES_XD.dispersion_angle(wl=0.55 * u.um, field_angle=0 * u.deg))


if __name__ == "__main__":
    main()
