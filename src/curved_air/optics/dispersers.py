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

import numpy as np
from astropy import units as u
from astropy.units import Quantity
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from numpy.typing import ArrayLike

from curved_air.other import Air
from curved_air.glass import Glass


@dataclass
class Disperser:
    name: str = ""
    min_wl: Quantity | None = None
    max_wl: Quantity | None = None


@dataclass
class EchelleGrating(Disperser):
    catalogue_number: str = ""
    groove_frequency: Quantity = 31.6 / u.mm
    blaze_angle: Quantity = 76 * u.deg
    gamma_angle: Quantity = 0 * u.deg
    ruled_width: Quantity = 400 * u.mm
    groove_length: Quantity | None = None

    @cached_property
    def R(self) -> float:
        return round(np.tan(self.blaze_angle).value, 1)

    @cached_property
    def m_times_lambda(self) -> Quantity:
        return (2 * np.cos(self.gamma) * self.sigma * np.sin(self.blaze_angle)).to(u.um)

    @property
    def gamma(self) -> Quantity:
        return self.gamma_angle

    @property
    def theta_B(self) -> Quantity:
        return self.blaze_angle

    @property
    def groove_period(self) -> Quantity:
        return 1 / self.groove_frequency

    @cached_property
    def sigma(self) -> Quantity:
        return self.groove_period.to(u.um)

    @cached_property
    def gpmm(self) -> Quantity:
        return self.groove_frequency.to(1 / u.mm)

    @cached_property
    def max_order(self) -> int:
        return int((self.m_times_lambda / self.min_wl) - 0.5)

    @cached_property
    def min_order(self) -> int:
        return int((self.m_times_lambda / self.max_wl) + 0.5)

    @cached_property
    def diffraction_orders(self) -> list[int]:
        min_m = self.order_number(self.max_wl, quantised=True)
        max_m = self.order_number(self.min_wl, quantised=True)

        return list(range(min_m, max_m + 1))

    @cached_property
    def beam_diameter(self) -> Quantity:
        return self.ruled_width / self.R

    def order_containing_500nm(self, quantised: bool = True) -> int:
        return self.order_number(wl=500 * u.nm, quantised=quantised)

    def blaze_wavelength(self, m: int | list[int] | ArrayLike) -> Quantity:
        if isinstance(m, list | tuple):
            return [self.blaze_wavelength(im) for im in m]  # ty: ignore

        return (self.m_times_lambda / m).to(u.um)

    def order_number(
        self,
        wl: float | Quantity = 500 * u.nm,
        quantised: bool = False,
    ) -> int | list[int] | float | list[float]:
        if isinstance(wl, tuple) or (isinstance(wl, Quantity) and wl.size > 1):
            return [self.order_number(iwl) for iwl in wl]

        if isinstance(wl, int | float):
            print(f"Input was unitless value {wl}. Defaulting to {wl * u.um}")
            wl *= u.um

        if quantised:
            return int(round(float((self.m_times_lambda / wl).si), 0))

        return float((self.m_times_lambda / wl).si)

    def order_FSR(self, m: int | list[int], FSR_fraction: float = 1.0) -> Quantity:
        if isinstance(m, list | tuple):
            return [self.order_FSR(im) for im in m]  # ty: ignore

        assert isinstance(m, int)

        m_wavelength = self.blaze_wavelength(m=m)
        m_plus_one_wavelength = self.blaze_wavelength(m=m + 1)

        return FSR_fraction * (m_wavelength - m_plus_one_wavelength).to(u.nm)

    def order_FSR_bounds(self, m: int, FSR_fraction: float = 1.0) -> Quantity:
        FSR = self.order_FSR(m=m, FSR_fraction=FSR_fraction).to(u.nm).value
        blaze_wavelength = self.blaze_wavelength(m=m).to(u.nm).value

        return (blaze_wavelength - FSR / 2, blaze_wavelength + FSR / 2) * u.nm

    def order_opening_angle(
        self,
        m: int | list[int],
        FSR_fraction: float = 1.0,
        return_bounds: bool = False,
    ) -> Quantity:
        if isinstance(m, list | tuple):
            return [self.order_opening_angle(im) for im in m]  # ty: ignore

        FSR_bounds = self.order_FSR_bounds(m=m, FSR_fraction=FSR_fraction)

        # angles = (self.diffraction_angle(m, wl = FSR_bounds[0]),
        #           self.diffraction_angle(m, wl = FSR_bounds[1]))

        angles = self.diffraction_angle(m=m, wl=FSR_bounds)

        if return_bounds:
            return angles

        return max(angles) - min(angles)

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
            return [
                self.diffraction_angle(iwl).to(u.degree).value for iwl in wl
            ] * u.degree

        if isinstance(wl, float):
            print(f"No wavelength units passed ({wl}). Assuming nanometers")
            wl *= u.nm

        arcsin_argument = (
            (m * wl).to(u.um).value
            / (np.cos(self.gamma) * self.sigma.to(u.um).value)  # ty:ignore
        ) - np.sin(self.blaze_angle)
        return (np.arcsin(arcsin_argument) - self.blaze_angle).to(u.deg)

    def wavelength_at_angle(self, m: int, angle: Quantity) -> Quantity:
        # angle += self.blaze_angle

        return (
            np.cos(self.gamma) * self.sigma * (np.sin(self.blaze_angle) + np.sin(angle))
        ) / m

    def order_FSR_bounds_from_angle(self, m: int) -> tuple[Quantity, Quantity]:
        angle: Quantity = self.order_opening_angle(m=m) / 2

        return (
            self.wavelength_at_angle(m=m, angle=self.blaze_angle - angle),
            self.wavelength_at_angle(m=m, angle=self.blaze_angle + angle),
        )

    def aspect_ratio(
        self,
        min_wl: Quantity | None = None,
        max_wl: Quantity | None = None,
    ) -> float:
        if not min_wl:
            min_wl = self.min_wl
        if not max_wl:
            max_wl = self.max_wl

        blue_angle = self.order_opening_angle(m=self.order_number(min_wl)).to(u.deg)
        red_angle = self.order_opening_angle(m=self.order_number(max_wl)).to(u.deg)

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
            wl = self.blaze_wavelength(m=m)

        delta_wl = self.order_FSR(m=m) / 100
        delta_theta = self.diffraction_angle(
            m=m, wl=wl + delta_wl
        ) - self.diffraction_angle(m=m, wl=wl)

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
            m = self.order_number(self.max_wl, quantised=True)

        wls = self.order_FSR_bounds(m=m, FSR_fraction=FSR_fraction)

        return self.anamorphic_magnification(
            m=m, wl=wls[0]
        ) / self.anamorphic_magnification(m=m, wl=wls[1])

    def relative_intensity(
        self,
        # m: int,
        wl: Quantity | ArrayLike,
    ) -> float | ArrayLike:
        """
        Calculate the relative intensity of a particular wavelength or wavelengths in a
        given diffraction order.

        TODO: fix this up!
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
    def reddest_order_separation(self) -> Quantity:
        return self.order_FSR(m=self.order_number(self.max_wl), FSR_fraction=1.0)

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
    incidence_angle: Quantity | None = None


@dataclass
class Prism(CrossDisperser):
    glass: Glass = field(default_factory=Glass)
    apex_angle: Quantity = 60 * u.deg

    def __post_init__(self) -> None:
        assert self.glass.name != "_default_"

        if self.incidence_angle is None:
            if self.min_wl is not None and self.max_wl is not None:
                self.incidence_angle = self.find_incidence_angle()
            else:
                self.incidence_angle = self.minimum_deviation_incidence_angle()

        if self.min_wl is None:
            self.min_wl = self.glass.wlmin

        if self.max_wl is None:
            self.max_wl = self.glass.wlmax

    @property
    def refractive_index(self) -> Callable:
        return self.glass.refractive_index

    @property
    def n(self) -> Callable:
        """Alias of refractive_index"""

        return self.refractive_index

    @cached_property
    def mid_wavelength(self) -> Quantity:
        """
        Calculate the mid-wavelength of the prism.
        """

        mid_refractive_index = (
            self.glass.refractive_index(self.min_wl)
            + self.glass.refractive_index(self.max_wl)
        ) / 2

        wls = np.linspace(
            self.min_wl.to(u.um).value,
            self.max_wl.to(u.um).value,
            500,
        )

        return wls[np.argmin(abs(self.glass.n(wls) - mid_refractive_index))] * u.um

    def deviation_angle(
        self,
        wl: ArrayLike = 0.6328 * u.um,
        incidence_angle: Quantity | None = None,
        field_angle: Quantity = 0 * u.deg,
    ) -> Quantity:
        if incidence_angle is None:
            incidence_angle = self.incidence_angle

        i = incidence_angle

        if hasattr(wl.value, "__len__") and hasattr(field_angle.value, "__len__"):
            return [
                self.dispersion_angle(wl=w, field_angle=f).value
                for w, f in zip(wl, field_angle, strict=False)
            ] * u.deg

        if field_angle.value != 0:
            effective_apex_angle = self.effective_apex_angle(field_angle=field_angle)

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

    def exitance_angle(
        self,
        wl: ArrayLike = 0.6328 * u.um,
        incidence_angle: Quantity | None = None,
    ) -> Quantity:
        if incidence_angle is None:
            incidence_angle = self.incidence_angle

        return (
            self.deviation_angle(wl=wl, incidence_angle=incidence_angle)
            + self.apex_angle
            - incidence_angle
        )

    def find_incidence_angle(
        self,
        min_wl: Quantity | None = None,
        max_wl: Quantity | None = None,
    ) -> Quantity:
        """
        Find the incidence angle that provides a symmetric deviation for the extreme
        wavelengths that the prism is designed for.
        """

        if min_wl is None:
            min_wl = self.min_wl
        if max_wl is None:
            max_wl = self.max_wl

        min_wl_incidence = self.minimum_deviation_incidence_angle(wl=min_wl)
        max_wl_incidence = self.minimum_deviation_incidence_angle(wl=max_wl)

        return ((min_wl_incidence + max_wl_incidence) / 2).to(u.deg)

    def minimum_deviation_angle(self, wl: ArrayLike = 0.6328 * u.um) -> Quantity:
        minimum_deviation_angle = (
            2 * np.arcsin(self.glass.n(wl) * np.sin(self.apex_angle / 2))
            - self.apex_angle
        )

        return minimum_deviation_angle.to(u.deg)

    def minimum_deviation_incidence_angle(
        self, wl: ArrayLike = 632.8 * u.nm
    ) -> Quantity:
        """
        Calculate the incidence angle that provides the minimum deviation condition for
        the prism at a given wavelength.
        """

        return (self.apex_angle + self.minimum_deviation_angle(wl)) / 2

    def effective_apex_angle(self, field_angle: Quantity = 0 * u.deg) -> Quantity:
        """
        I don't know why this works. The 74° is empirically determined, but matches very
        well up to a field angle of ~10°.
        """
        return self.apex_angle + 74 * u.deg * (1 - np.cos(field_angle))

    def dispersion_angle(
        self,
        wl: Quantity | list[Quantity],
        field_angle: Quantity = 0 * u.deg,
    ) -> Quantity | list[Quantity]:
        """ """

        if hasattr(wl.value, "__len__") and hasattr(
            field_angle.value, "__len__"
        ):  # ty: ignore
            return [
                self.dispersion_angle(wl=w, field_angle=f).value
                for w, f in zip(wl, field_angle, strict=False)
            ] * u.deg

        zero_angle = self.deviation_angle(wl=self.mid_wavelength)

        if field_angle.value != 0:
            effective_apex_angle = self.effective_apex_angle(field_angle=field_angle)

            _self = Prism(
                glass=self.glass,
                apex_angle=effective_apex_angle,
                incidence_angle=self.incidence_angle,
            )

            return _self.deviation_angle(wl=wl) - zero_angle

        return self.deviation_angle(wl=wl) - zero_angle

    def angular_spread(
        self,
        min_wl: Quantity | None = None,
        max_wl: Quantity | None = None,
    ) -> Quantity:
        """ """

        if min_wl is None:
            min_wl = self.min_wl
        if max_wl is None:
            max_wl = self.max_wl

        min_angle = -self.deviation_angle(wl=min_wl)
        max_angle = -self.deviation_angle(wl=max_wl)
        return max_angle - min_angle

    def local_angular_dispersion(self, wl: Quantity) -> Quantity:
        delta_wavelength = 1 * u.nm

        wl1 = wl - delta_wavelength / 2
        wl2 = wl + delta_wavelength / 2

        return self.angular_spread(min_wl=wl1, max_wl=wl2) / delta_wavelength

    def trace(
        self,
        wls: ArrayLike = None,
        incidence_angle: Quantity | None = None,
        debug_level: float = 0,
    ) -> ArrayLike:
        """
        Trace a set of rays through the Prism, solving Snell's Law at each
        interface
        """

        if wls is None:
            wls = [0.4861, 0.5893, 0.6563] * u.um

        if incidence_angle is None:
            incidence_angle = self.incidence_angle

        try:
            rays = incidence_angle * np.ones(len(wls))
        except TypeError:  # wls is a single value
            rays = incidence_angle

        previous_angle = 0 * u.degree

        rays = _refract(
            n_1=Air().refractive_index(wls),
            n_2=self.refractive_index,
            theta_in=rays - previous_angle,
            wavelength=wls,
        )

        if debug_level > 0:
            print(f"Refracting through a {self.glass} prism of {self.apex_angle}")

        # Refract through the last surface into the exit medium
        rays = _refract(
            n_1=self.refractive_index(wls),
            n_2=Air().refractive_index(wls),
            theta_in=rays - self.apex_angle,
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
class Grism(CrossDisperser):
    apex_angle: Quantity = 0 * u.deg
    glass: Glass = field(default_factory=Glass)
    beam_diameter: Quantity = 200 * u.mm
    line_spacing: Quantity = 110 / u.mm
    diffraction_order: int = 1

    def __post_init__(self) -> None:
        if self.incidence_angle is None:
            self.incidence_angle = 0 * u.deg

    @property
    def groove_period(self) -> Quantity:
        return 1 / self.line_spacing

    def refraction_angle(
        self,
        wl: Quantity = 550 * u.nm,
        incidence_angle: Quantity | None = None,
        field_angle: Quantity = 0 * u.deg,
    ) -> Quantity:
        """ """
        if incidence_angle is None:
            incidence_angle: Quantity = self.incidence_angle

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

    def diffraction_angle(
        self,
        wl: Quantity = 550 * u.nm,
    ) -> Quantity:
        """
        Calculate the diffraction angle for a given order and wavelength.
        """
        if isinstance(wl, float):
            print(f"No wavelength units passed ({wl}). Assuming nanometers")
            wl *= u.nm

        return (
            np.arcsin(
                wl.to(u.um).value
                / (self.groove_period.to(u.um).value * np.cos(self.apex_angle))
            )
            - self.apex_angle
        ).to(u.deg)

    def dispersion_angle(
        self,
        wl: Quantity = 550 * u.nm,
        field_angle: Quantity = 0 * u.deg,
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

    groove_frequency: Quantity = 150 / u.mm
    m: int = 1
    incidence_angle: Quantity = 0 * u.deg

    @property
    def groove_period(self) -> Quantity:
        return 1 / self.groove_frequency

    def diffraction_angle(
        self,
        wl: Quantity = 550 * u.nm,
    ) -> Quantity:
        if isinstance(wl, float):
            print(f"No wavelength units passed ({wl}). Assuming nanometers")
            wl *= u.nm

        return np.arcsin((self.m * wl / self.groove_period).si).to(u.deg)

    @property
    def central_angle(self) -> Quantity:
        beta_min_wl = self.diffraction_angle(wl=self.min_wl)
        beta_max_wl = self.diffraction_angle(wl=self.max_wl)

        return (beta_min_wl + beta_max_wl) / 2

    def dispersion_angle(
        self,
        wl: Quantity,
        field_angle: Quantity = 0 * u.deg,  # noqa: ARG002
    ) -> Quantity | list[Quantity]:
        # No treatment of field angle here

        if isinstance(wl, list | tuple):
            return [self.dispersion_angle(iwl) for iwl in wl]

        if isinstance(wl, float):
            print(f"No wavelength units passed ({wl}). Assuming nanometers")
            wl *= u.nm

        return (self.diffraction_angle(wl=wl) - self.central_angle).to(u.deg)


def _refract(
    n_1: float | Callable,
    n_2: float | Callable,
    theta_in: Quantity = 0 * u.deg,
    wavelength: Quantity | None = None,
    debug_level: float = 0.0,  # TODO: use this!  # noqa: ARG001
) -> Quantity:
    # Compute the refracted angle of a ray or rays at a single material
    # interface
    # This can handle EITHER an array of input angles at a single wavelength
    # OR a single input angle at a range of wavelengths

    if wavelength is None:
        _n_1 = n_1
        _n_2 = n_2

    else:
        if callable(n_1):
            try:
                _n_1 = n_1(wavelength)
            except TypeError:
                _n_1 = n_1
        if callable(n_2):
            try:
                _n_2 = n_2(wavelength)
            except TypeError:
                _n_2 = n_2

    # By this point, _n_1 and _n_2 should be either floats, ints, or
    # lists/arrays of such
    assert not callable(_n_1)
    assert not callable(_n_2)

    theta_out = np.arcsin((_n_1 / _n_2) * np.sin(theta_in))

    return theta_out.to(u.degree)


def main() -> None: ...


if __name__ == "__main__":
    main()
