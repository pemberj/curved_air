"""
curved_air

beams

This module contains definitions of a few types of analyticall-described optical beams.

These include Gaussian beams, plane waves, and spherical waves. This code was originally
written for the EtalonSim project, but has been adapted to be more general and reusable
here.
"""

from dataclasses import dataclass, field
from typing import overload

import astropy.units as u
import numpy as np
from astropy.units import Quantity
from numpy.typing import ArrayLike
from scipy.integrate import quad

from curved_air.type_hinting import ArrayQuantity, ScalarQuantity


@dataclass
class Beam:
    """
    Basic properties of a simple monochromatic light beam.

    No propagation is described here, for that see the subclasses.
    """

    wavelength: ScalarQuantity = 633 * u.nm
    E_0: float = 1
    n: float = 1

    debug_level: int = 0

    @property
    def k_scalar(self) -> ScalarQuantity:
        return 2 * np.pi * self.n / self.wavelength

    @property
    def k(self) -> ScalarQuantity:
        return self.k_scalar

    def __str__(self) -> str:
        return "Beam:\n" + "".join(
            [f"{p:>12}: {val:<6}\n" for p, val in self.__dict__.items()]
        )


# EtalonSim / GaussianBeam.py
@dataclass
class GaussianBeam(Beam):
    """
    An implementation of a Gaussian beam as per wikipedia:
    https://en.wikipedia.org/wiki/Gaussian_beam
    """

    w_0: ScalarQuantity = 10 * u.um

    printed: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.debug_level > 0:
            for p, val in self.__dict__.items():
                print(f"{p}: {val}")

    @property
    def z_R(self) -> ScalarQuantity:
        """
        z_R, Rayleigh range, sometimes z_0
        (not to be confused with a distance to the beam waist)
        """

        return ((np.pi * self.n * self.w_0**2) / self.wavelength).si

    @overload
    def w_squared(self, z: ScalarQuantity) -> ScalarQuantity: ...

    @overload
    def w_squared(self, z: ArrayQuantity) -> ArrayQuantity: ...

    def w_squared(self, z: Quantity = 0 * u.mm) -> Quantity:
        """
        Square of the beam width
        """

        _z: Quantity = (z / self.z_R).si

        if self.debug_level > 0 and not self.printed.get("w(z)^2"):
            self.printed["w(z)^2"] = True
            print("z / self.z_0:")
            print(_z)
            print("w(z)^2:")
            print((self.w_0**2 * (1 + (_z) ** 2)).si)

        return (self.w_0**2 * (1 + (_z) ** 2)).si

    @overload
    def w(self, z: ScalarQuantity) -> ScalarQuantity: ...

    @overload
    def w(self, z: ArrayQuantity) -> ArrayQuantity: ...

    def w(self, z: Quantity = 0 * u.mm) -> Quantity:
        """
        w, beam width
        Defined as the radius at which the field amplitudes fall to 1/e of their
        axial values (i.e., where the intensity values fall to 1/e2 of their
        axial values)
        """

        if self.debug_level > 0 and not self.printed.get("w(z)"):
            self.printed["w(z)"] = True
            print("w(z):")
            print(np.sqrt(self.w_squared(z)).si)

        return np.sqrt(self.w_squared(z)).si

    @overload
    def R(self, z: ScalarQuantity) -> ScalarQuantity: ...

    @overload
    def R(self, z: ArrayQuantity) -> ArrayQuantity: ...

    def R(self, z: Quantity = 0 * u.mm) -> Quantity:
        """
        R, radius of curvature of the wavefront
        """

        _z: Quantity = (self.z_R / z).si

        if self.debug_level > 0 and not self.printed.get("R(z)"):
            self.printed["R(z)"] = True
            print("z_0 / z:")
            print(_z)
            print("R(z):")
            print((z * (1 + (_z) ** 2)).si)

        R: Quantity = (z * (1 + (_z) ** 2)).si
        # Account for infinite radius of curvature at beam waist
        R[np.isnan(R)] = np.inf * u.m

        return R

    def amplitude(
        self,
        r: ScalarQuantity | None = None,
        x: ScalarQuantity = 0 * u.mm,
        y: ScalarQuantity = 0 * u.mm,
        z: ScalarQuantity = 0 * u.mm,
    ) -> ScalarQuantity:
        """
        The amplitude of the electric field at the given coordinates
        """
        r: Quantity = np.sqrt(x**2 + y**2).si if r is None else r.si

        return (
            self.E_0
            * (self.w_0 / self.w(z)).si
            * np.exp((-(r**2) / (self.w_squared(z))).si)
        )

    @overload
    def Gouy_phase(self, z: ScalarQuantity) -> ScalarQuantity: ...

    @overload
    def Gouy_phase(self, z: ArrayQuantity) -> ArrayQuantity: ...

    def Gouy_phase(self, z: Quantity = 0 * u.mm) -> Quantity:
        """
        The Gouy phase at a distance z from the beam waist
        """

        if self.debug_level > 0 and not self.printed.get("Gouy"):
            self.printed["Gouy"] = True
            print("Gouy phase, arctan(z/z_0):")
            print(np.arctan((z / self.z_R).si))

        return np.arctan((z / self.z_R).si) / u.radian

    @overload
    def longitudinal_phase(self, z: ScalarQuantity) -> ScalarQuantity: ...

    @overload
    def longitudinal_phase(self, z: ArrayQuantity) -> ArrayQuantity: ...

    def longitudinal_phase(self, z: Quantity = 0 * u.mm) -> Quantity:
        """
        The longitudinal phase of the E field at a distance z from the beam waist
        """

        return (self.k * z).si

    @overload
    def radial_phase(
        self,
        r: ScalarQuantity | None,
        x: ScalarQuantity,
        y: ScalarQuantity,
        z: ScalarQuantity,
    ) -> ScalarQuantity: ...

    @overload
    def radial_phase(
        self,
        x: ScalarQuantity,
        y: ScalarQuantity,
        z: ScalarQuantity,
    ) -> ScalarQuantity: ...

    @overload
    def radial_phase(
        self,
        z: ScalarQuantity,
    ) -> ScalarQuantity: ...

    @overload
    def radial_phase(
        self,
        r: ScalarQuantity | None,
        x: ScalarQuantity,
        y: ScalarQuantity,
        z: ArrayQuantity,
    ) -> ArrayQuantity: ...

    @overload
    def radial_phase(
        self,
        x: ScalarQuantity,
        y: ScalarQuantity,
        z: ArrayQuantity,
    ) -> ArrayQuantity: ...

    @overload
    def radial_phase(
        self,
        z: ArrayQuantity,
    ) -> ArrayQuantity: ...

    @overload
    def radial_phase(
        self,
        r: ScalarQuantity,
        z: ScalarQuantity | ArrayQuantity,
    ) -> ArrayQuantity: ...

    def radial_phase(
        self,
        r: ScalarQuantity | None = None,
        x: ScalarQuantity = 0 * u.mm,
        y: ScalarQuantity = 0 * u.mm,
        z: ScalarQuantity | ArrayQuantity = 0 * u.mm,
    ) -> ScalarQuantity | ArrayQuantity:
        """
        The radial phase of the E field at the given coordinates
        """
        if r is not None and (x is not None or y is not None):
            print("Either r or (x, y) should be specified. Using r value.")

        r: ScalarQuantity = np.sqrt(x**2 + y**2) if r is None else r.si

        return (self.k.si * (r**2).si) / (2 * self.R(z).si)

    def E_field(
        self,
        x: ScalarQuantity = 0 * u.mm,
        y: ScalarQuantity = 0 * u.mm,
        z: ScalarQuantity = 0 * u.mm,
    ) -> ScalarQuantity:
        """
        Returns the complex E field of the wave, encoding amplitude and phase,
        at the given coordinates
        """

        return self.amplitude(x, y, z) * np.exp(
            -1j
            * (
                self.longitudinal_phase(z)
                + self.radial_phase(x=x, y=y, z=z)
                - self.Gouy_phase(z)
            )
        )

    def E_field_partial(
        self,
        r: ScalarQuantity | None = None,
        x: ScalarQuantity = 0 * u.mm,
        y: ScalarQuantity = 0 * u.mm,
        z: ScalarQuantity = 0 * u.mm,
        include: str = "amplitude longitudinal",
    ) -> ScalarQuantity:
        """
        A function to construct partial Gaussian beams, for instance including
        only Gouy phase or only radial phase, for inspection of the
        contributions of each phase component, while taking in to account the
        amplitude variations of these contributions.

        Args:
            r, radius from the optical axis in length units. Defaults to None.
            x ([type], optional): [description]. Defaults to 0*u.mm.
            y ([type], optional): [description]. Defaults to 0*u.mm.
            z ([type], optional): [description]. Defaults to 0*u.mm.
            include (str, optional): [description].
                Defaults to "amplitude longitudinal".
        """

        r: Quantity = np.sqrt(x**2 + y**2) if r is None else r.si
        E: float = 1.0

        if "amplitude" in include.lower():
            # print("Computing amplitude")
            E: Quantity = E * self.amplitude(r=r, z=z)

        if "longitudinal" in include.lower():
            # print("Computing longitudinal phase")
            E: Quantity = E * np.exp(-1j * self.longitudinal_phase(z=z))

        if "radial" in include.lower():
            # print("Computing radial phase")
            E: Quantity = E * np.exp(-1j * self.radial_phase(r=r, z=z))

        if "gouy" in include.lower():
            # print("Computing Gouy phase")
            E: Quantity = E * np.exp(+1j * self.Gouy_phase(z=z))

        assert isinstance(E, Quantity), "Something went wrong! {include = }"

        return E

    def E_gouy_integrated(
        self,
        r: float = np.inf,
        z: ScalarQuantity = 0 * u.mm,
    ) -> ScalarQuantity:
        """
        A function that returns the E-field incorporating only the Gouy phase component
        integrated from r=0 to the given r in the plane a given distance z from the beam
        waist.
        """

        ret: ScalarQuantity = (
            self.E_0
            * self.w_0
            * self.w(z=z)
            * np.pi
            * np.exp(1j * self.Gouy_phase(z=z))
        )

        if np.isinf(r):
            return ret

        exponent: float = -((r**2) / (self.w(z) ** 2)).si.value
        ret *= 1 - np.exp(exponent)
        return ret

    def E_radial_integrated(
        self, r: float = np.inf, z: ScalarQuantity = 0 * u.mm
    ) -> ScalarQuantity:
        """
        A function that returns the E-field incorporating only the radial phase
        component integrated from r=0 to the given radius r in the plane a given
        distance z from the beam waist.
        """

        ret: ScalarQuantity = (
            self.E_0
            * (self.w_0 / self.w(z=z)).si
            * np.pi
            / ((1 / (self.w(z=z) ** 2)) + ((1j * self.k) / (2 * self.R(z=z))))
        )

        if np.isinf(r):
            # print(f"{self.E_0 = }")
            # print(f"{self.w_0 = }")
            # print(f"{self.w(z=z) = }")
            # print(f"{self.k = }")
            # print(f"{self.R(z=z) = }")
            return ret

        exponent1: ScalarQuantity = -(r**2) * (1 / self.w(z=z) ** 2).si
        assert exponent1.unit == u.dimensionless_unscaled
        exponent2: ScalarQuantity = -(r**2) * ((1j * self.k) / (2 * self.R(z=z))).si
        assert exponent2.unit == u.dimensionless_unscaled
        exponent: float = (exponent1 + exponent2).value
        ret *= 1 - np.exp(exponent)

        return ret

    def summed_E_field(
        self,
        z: ScalarQuantity = 0 * u.mm,
        exclude: str = "none",
    ) -> ScalarQuantity:
        """
        For a beam symmetric about the optical axis. Integrating over r from 0
        to infinity gives the summed (inegrated) E field across a plane at z.
        Used for calculating the interference of full waves in devices like a
        Fabry-Perot etalon.
        """

        A: ScalarQuantity = (u.m**2 / self.w(z) ** 2) + 1j * (
            (self.k * u.m**2) / (2 * self.R(z))
        )
        integral_over_r: ScalarQuantity = np.pi / A

        # baseline result, including only the amplitude and longitudinal phase,
        # ie. a plane wave beam
        ret: ScalarQuantity = (
            self.E_0 * (self.w_0 / self.w(z)) * np.exp(-1j * self.longitudinal_phase(z))
        )

        try:
            if "gouy" not in exclude.lower():
                if self.debug_level > 0:
                    print("Including Gouy phase")
                ret *= np.exp(1j * self.Gouy_phase(z))

            else:
                print("Returning results without Gouy phase...")

            if "radial" not in exclude.lower():
                if self.debug_level > 0:
                    print(
                        "Including radial effects "
                        + "(wavefront curvature, intensity variations)"
                    )
                ret *= integral_over_r

            else:
                print("Returning results with no radial effects...")

            return ret.si

        except Exception as e:
            print(e)
            print("Assuming full Gaussian beam...")

            return (
                self.E_0
                * (self.w_0 / self.w(z))
                * np.exp(-1j * self.longitudinal_phase(z))
                * np.exp(1j * self.Gouy_phase(z))
                * integral_over_r
            ).si


# EtalonSim / PlanarBeam.py
@dataclass
class PlanarBeam(Beam):
    """
    An implementation of a very simple plane electromagnetic wave
    """

    plane_normal: tuple[float, ...] = (0, 0, 1)

    def __post_init__(self) -> None:
        if self.debug_level > 0:
            print(self.__dict__)

    @property
    def k_scalar(self) -> float:
        return (2 * np.pi * self.n / self.wavelength) * np.linalg.norm(
            self.plane_normal
        )

    @property
    def k_vector(self) -> tuple[float, ...]:
        return tuple(self.k_scalar * i for i in self.plane_normal)

    def E_field(
        self,
        x: ScalarQuantity = 0 * u.mm,
        y: ScalarQuantity = 0 * u.mm,
        z: ScalarQuantity = 0 * u.mm,
    ) -> ScalarQuantity:
        """
        https://en.wikipedia.org/wiki/Electromagnetic_wave_equation
        """

        x: float = x.to(u.mm).value
        y: float = y.to(u.mm).value
        z: float = z.to(u.mm).value

        # Position vector r
        r_unitless: np.ndarray = np.array([x, y, z])
        # r = r_unitless * u.mm
        k_vector_unitless: np.ndarray = self.k_vector.to(1 / u.mm).value

        return self.E_0 * np.exp(-1j * np.dot(k_vector_unitless, r_unitless))


# EtalonSim / SphericalBeam2.py
@dataclass
class SphericalBeam(Beam):
    """
    Light emitting from a point source in a spherical wavefront
    """

    f_number: int | float = 10
    plane_normal: tuple[float, ...] = (0, 0, 1)

    k_vector: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        self.plane_normal: tuple[float, ...] = tuple(
            np.array(self.plane_normal) / np.linalg.norm(self.plane_normal)
        )

        self.k_vector: tuple[float, ...] = tuple(
            self.k_scalar * i for i in self.plane_normal
        )

    def beam_diameter(self, z: ScalarQuantity) -> ScalarQuantity:
        return z / self.f_number

    def E_field(
        self,
        x: ArrayLike | ScalarQuantity = 0 * u.mm,
        y: ArrayLike | ScalarQuantity = 0 * u.mm,
        z: ArrayLike | ScalarQuantity = 0 * u.mm,
    ) -> ScalarQuantity:
        """
        Calculate the electric field at a point in space
        """

        if isinstance(x, Quantity):
            x: float = x.to(u.mm).value
        if isinstance(y, Quantity):
            y: float = y.to(u.mm).value
        if isinstance(z, Quantity):
            z: float = z.to(u.mm).value

        x_shape: tuple[int, ...] = np.shape(x)
        # y_shape = np.shape(y)  # TODO: don't assume x, y, and z are the same shape
        # z_shape = np.shape(z)

        x: np.ndarray = np.array(x).flatten()
        y: np.ndarray = np.array(y).flatten()

        if isinstance(z, float):
            z: np.ndarray = z * np.ones_like(x)

        print(f"{np.shape(x) = }")
        print(f"{np.shape(y) = }")
        print(f"{np.shape(z) = }")

        r: ArrayQuantity = np.array([x, y, z]) * u.mm

        # r_norm = np.linalg.norm(r, axis=0)

        # area = np.pi * (r_norm / (2 * self.f_number))**2
        area: ScalarQuantity = 1 * u.mm**2

        E: ScalarQuantity = self.E_0 / area.value
        phase: ScalarQuantity = np.exp(1j * (np.dot(self.k_vector, r).si))  # type: ignore

        return np.array(E.value * phase).reshape(x_shape) * E.unit

    @overload
    def summed_E_field(
        self,
        z: ScalarQuantity,
        max_radius: ScalarQuantity | None,
    ) -> Quantity: ...

    @overload
    def summed_E_field(
        self,
        z: ArrayQuantity,
        max_radius: ScalarQuantity | None,
    ) -> ArrayQuantity: ...

    def summed_E_field(
        self,
        z: ScalarQuantity | ArrayQuantity = 0 * u.mm,
        max_radius: ScalarQuantity | None = None,
        force_radius: bool = False,
    ) -> ScalarQuantity | ArrayQuantity | float:  # TODO fix
        """
        Using the analytic expression for a spherical beam, this function
        calculates the integral of the E field over a circular aperture
        perpendicular to the beam axis at a distance z from the source.
        """

        if isinstance(z, Quantity):
            z: float = z.to(u.mm).value

        if not z.isscalar:
            return [
                self.summed_E_field(zi, max_radius).value for zi in z
            ] * self.E_0.unit

        if z == 0:
            return self.E_0

        if force_radius:
            if isinstance(max_radius, Quantity):
                radius: Quantity = max_radius
            elif max_radius is None:
                radius: Quantity = self.beam_diameter(z * u.mm) / 2
        elif isinstance(max_radius, Quantity):
            radius: Quantity = min(max_radius, self.beam_diameter(z * u.mm) / 2)

        k: float = self.k_scalar.to(1 / u.mm).value

        # Define the integrand
        def integrand(r: float) -> float:
            sqrtr2x2 = np.sqrt(r**2 + z**2)

            return (1 / sqrtr2x2) * np.exp(-1j * k * sqrtr2x2) * r

        I, _error = quad(integrand, 0, radius.to(u.mm).value, limit=100)  # noqa: E741

        return 2 * np.pi * self.E_0 * I

    def integrated_E_field(
        self,
        z: Quantity = 0 * u.mm,
        radius: Quantity | float = 10 * u.mm,
    ) -> Quantity:
        """Use the analytic expression for a slice through the beam at a distance z"""

        if isinstance(z, list):
            return [
                self.integrated_E_field(z=zi, radius=radius).value for zi in z
            ] * self.E_0.unit

        if isinstance(z, Quantity):
            z: float | np.ndarray = z.to(u.mm).value

        if isinstance(radius, Quantity):
            radius: float = radius.to(u.mm).value

        beam_edge: float | np.ndarray = z / self.f_number

        R: float | np.ndarray = np.minimum(radius, beam_edge)

        k: float = self.k_scalar.to(1 / u.mm).value

        constant: Quantity = (2 * np.pi * self.E_0) / (1j * k)
        exponent_1: float = 1j * (k * np.sqrt(R**2 + z**2))
        exponent_2: float = 1j * (k * z)

        return constant * (np.exp(exponent_1) * np.exp(-exponent_2))


def main() -> None: ...


if __name__ == "__main__":
    main()
