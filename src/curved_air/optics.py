"""
curved_air

optics

Definitions of several optical components, including Prisms, compound prisms (here
PrismStacks), ADCs (a PrismStack to be optimised to match the dispersion of air),
Gratings, and Etalons.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from astropy import units as u
from astropy.units import Quantity
from matplotlib import pyplot as plt
from numpy.typing import ArrayLike
from scipy.optimize import minimize

from curved_air.glass import Glass
from curved_air.other import Air, Model


@dataclass
class Prism:
    glass: Glass
    name: str = ""
    apex_angle: Quantity = 60 * u.deg
    min_wl: Quantity | None = None
    max_wl: Quantity | None = None
    incidence_angle: Quantity | None = None

    def __post_init__(self) -> None:
        assert self.glass is not None

        if self.incidence_angle is None:
            if self.min_wl is not None and self.max_wl is not None:
                self.incidence_angle = self.find_incidence_angle(
                    min_wl=self.min_wl, max_wl=self.max_wl
                )
            else:
                self.incidence_angle = self.minimum_deviation_incidence_angle()

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
        incidence_angle: Quantity = None,
    ) -> Quantity:
        if incidence_angle is None:
            incidence_angle = self.incidence_angle

        i = incidence_angle

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
        incidence_angle: Quantity = None,
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
        min_wl: Quantity = 385 * u.nm,
        max_wl: Quantity = 950 * u.nm,
    ) -> Quantity:
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

        if hasattr(wl.value, "__len__") and hasattr(field_angle.value, "__len__"):
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
        incidence_angle: Quantity = None,
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
class PrismStack:
    """
    TODO:
    - [ ] Add method effective_refractive_index(), to return the effective
    refractive index of the PrismStack. How to define this? Take the angle
    between front and back surfaces? But what if that angle is zero? Refractive
    index is effectively infinite... Need to have an "effective apex angle" in
    there as well somewhere... Take the average of absolute apex angles of each
    of the components? Need to think about the best way to define this so that
    it generalises well, but this method will be SO USEFUL in optimising -> you
    can then just match this (or a scaled/offset version) to any desired
    refractive index vector, and drive the difference to be small.
    """

    prism_list: list[Prism] = field(default_factory=list)
    entrance_medium: Model = field(default_factory=Air)
    exit_medium: Model = field(default_factory=Air)

    def add_prism(self, prism: Prism) -> None:
        self.prism_list.append(prism)

    def trace(
        self,
        wls: ArrayLike = None,
        incidence_angle: Quantity = 0 * u.deg,
        debug_level: float = 0,
    ) -> ArrayLike:
        """
        Trace a set of rays through the PrismStack, solving Snell's Law at each
        interface
        """

        if wls is None:
            wls = [0.4861, 0.5893, 0.6563] * u.um

        try:
            rays = incidence_angle * np.ones(len(wls))
        except TypeError:  # wls is a single value
            rays = incidence_angle

        previous_material = self.entrance_medium
        previous_angle = 0 * u.degree

        for _n, prism in enumerate(self.prism_list):
            rays = _refract(
                n_1=previous_material.refractive_index(wls),
                n_2=prism.refractive_index,
                theta_in=rays - previous_angle,
                wavelength=wls,
            )

            if debug_level > 0:
                print(f"Refracting through a {prism.glass} prism of {prism.apex_angle}")

            previous_material = prism.glass
            previous_angle = prism.apex_angle

            # print(rays.to("degree"))

        # Refract through the last surface into the exit medium
        rays = _refract(
            n_1=previous_material.refractive_index(wls),
            n_2=self.exit_medium.refractive_index(wls),
            theta_in=rays - prism.apex_angle,
            wavelength=wls,
        )

        return rays.to(u.degree)

    def zero_deviation_wavelengths(
        self,
        min_wl: Quantity = 0.39 * u.um,
        max_wl: Quantity = 1.50 * u.um,
        num_wls: float = 100,
        incidence_angle: Quantity = 0 * u.deg,
    ) -> ArrayLike:
        """
        This method traces the dispersion of light through a compound prism and
        returns the wavelengths at which the prism does not deviate the light.
        These zero-deviation wavelengths are interesting in design for
        astronomical ADCs, and may be a fast way to check that a design is
        working the way it should.
        """

        if isinstance(min_wl, Quantity) and isinstance(max_wl, Quantity):
            wls = (
                np.linspace(min_wl.to("um").value, max_wl.to("um").value, num_wls)
                * u.um
            )
        elif isinstance(min_wl, int | float) and isinstance(max_wl, int | float):
            wls = np.linspace(min_wl, max_wl, num_wls) * u.um
        else:
            raise TypeError(
                "min_wl and max_wl parameters must be int, float, "
                + "or astropy.units.quantity.Quantity with appropriate units."
            )

        out_angles = self.trace(wls=wls, incidence_angle=incidence_angle)
        zwls = wls[
            np.where(
                np.array(out_angles.value)[:-1] / np.array(out_angles.value)[1:] < 0
            )
        ]

        if len(zwls) >= 2:  # noqa: PLR2004
            return zwls
        if len(zwls) == 1:
            return zwls[0]
        return None


@dataclass
class ADC:
    prism_stack: PrismStack
    air_model: Model = field(default_factory=Air)
    min_wl: Quantity = 390 * u.nm
    max_wl: Quantity = 950 * u.nm
    D_telescope: Quantity = 0.8 * u.m
    D_ADC: Quantity = 15 * u.mm
    z_max: Quantity = 65 * u.degree
    criterion: str = "PTV"  # "RMS", "RSS", "PTV", "max_abs"

    alpha: Quantity | None = None
    beta: Quantity | None = None
    wl_0: Quantity | None = None

    residuals_function: Callable | None = None

    @property
    def g1(self) -> Glass:
        return self.prism_stack.prism_list[0]

    @property
    def g2(self) -> Glass:
        return self.prism_stack.prism_list[1]

    @property
    def epsilon(self) -> Quantity:
        if self.wl_0 is None:
            return None
        return (self.g1.refractive_index(self.wl_0) - 1) / (
            self.g2.refractive_index(self.wl_0) - 1
        )

    @property
    def alpha2(self) -> Quantity:
        return self.alpha + self.beta

    @property
    def beam_comp(self) -> Quantity:
        return self.D_telescope / self.D_ADC

    @property
    def magnification_at_ADC(self) -> Quantity:
        return self.beam_comp

    def wls(self, n: int = 100) -> Iterable:
        return (
            np.linspace(self.min_wl.to(u.um).value, self.max_wl.to(u.um).value, n)
            * u.um
        )

    @property
    def residuals_RMS(self) -> Quantity:
        RMS = np.sqrt(np.mean(self.residuals_function(wls=self.wls(100)) ** 2))

        return RMS.to(u.mas)

    @property
    def residuals_RSS(self) -> Quantity:
        RSS = np.sqrt(np.sum(self.residuals_function(wls=self.wls(100)) ** 2))

        return RSS.to(u.mas)

    @property
    def residuals_PTV(self) -> Quantity:
        PTV = abs(
            max(self.residuals_function(wls=self.wls(100)))
            - min(self.residuals_function(wls=self.wls(100)))
        )

        return PTV.to(u.mas)

    def optimise(
        self,
        plot: bool = False,
        num_samples: int = 20,
    ) -> dict:
        """
        This function finds the best main apex angle and zero-deviation
        wavelength for the ADC's dispersion to match the dispersion of air.

        TODO:

        - [ ] Instead of computing residuals based on the difference between
        refractive indices, actually trace rays through the ADC

        - [ ] Allow to match different target dispersion???
        """

        if len(self.prism_stack.prism_list) != 2:  # noqa: PLR2004
            raise NotImplementedError(
                "This method is only implemented for "
                + "two-component compound prisms."
            )

        g1 = self.prism_stack.prism_list[0]
        g2 = self.prism_stack.prism_list[1]

        wls = self.wls(n=num_samples)

        # Find a linear combination of refractive index curves that best match
        # the refractive index of air, in which the apex angles of the two prism
        def residuals(
            x: tuple[float, float] = (1, 0.6328),  # alpha, wl_0 - no units
            wls: ArrayLike = wls,
            return_residuals: bool = False,
            verbose: bool = False,
        ) -> Quantity:
            # Fully parameterised by alpha and wl_0
            alpha, wl_0 = x

            alpha = alpha * u.degree
            wl_0 = wl_0 * u.um

            if verbose:
                print(f"Trying: {alpha = }, {wl_0 = }")

            epsilon = (g1.refractive_index(wl_0) - 1) / (g2.refractive_index(wl_0) - 1)

            delta_air = (  # vvvvvvvvvv   Should this be tan(z_max)?
                self.beam_comp
                * self.z_max
                * (
                    self.air_model.refractive_index(wls)
                    - self.air_model.refractive_index(wl_0)
                )
            ).si.to(u.degree)

            delta_ADC = (
                g1.refractive_index(wls)
                - epsilon * g2.refractive_index(wls)
                + epsilon
                - 1
            ) * alpha

            residuals = (delta_ADC - delta_air) * u.radian
            # Put it back in equivalent on-sky angle
            residuals /= self.z_max * self.beam_comp

            if return_residuals:
                return residuals.to(u.mas)

            if self.criterion == "PTV":
                return abs(max(residuals.to(u.mas)) - min(residuals.to(u.mas)))
            if self.criterion == "RMS":
                return np.sqrt(np.mean(residuals.to(u.mas) ** 2))
            if self.criterion == "RSS":
                return np.sqrt(np.sum(np.square(residuals.to(u.mas).value)))
            if self.criterion == "max_abs":
                return max(np.abs(residuals.to(u.mas)))

            return np.nan

        try:
            result = minimize(
                fun=residuals,
                x0=[1.0, np.median(wls.to(u.um).value)],
                bounds=[  # min  # max
                    (-np.inf, np.inf),
                    (0.4, 1.5),  # um
                ],
            )

            alpha_fit, wl_0_fit = result.x

        except Exception as e:
            print(e)
            print("Unable to converge on solution.")
            return None

        # residuals_PTV = residuals(x = result.x)
        # residuals_curve = residuals(x = result.x, return_residuals=True)

        def residuals_function(wls: ArrayLike = wls) -> Quantity:
            return residuals(
                x=(self.alpha.to(u.degree).value, self.wl_0.to(u.um).value),
                wls=wls,
                return_residuals=True,
            )

        self.residuals_function = residuals_function

        # Solve for individual prism apex angle from fitted constant
        self.first_apex_angle = alpha_fit * u.degree / 2
        epsilon_fit = (g1.refractive_index(wl_0_fit) - 1) / (
            g2.refractive_index(wl_0_fit) - 1
        )
        self.second_apex_angle = epsilon_fit * self.first_apex_angle

        self.alpha = alpha_fit * u.degree
        self.beta = self.second_apex_angle - self.first_apex_angle
        self.wl_0 = wl_0_fit * u.um

        results = {
            "g1": g1,
            "g2": g2,
            "D_telescope": self.D_telescope,
            "D_ADC": self.D_ADC,
            "beam_comp": self.beam_comp,
            "z_max": self.z_max,
            "min_wl": self.min_wl,
            "max_wl": self.max_wl,
            "first_apex_angle": self.first_apex_angle,
            "second_apex_angle": self.second_apex_angle,
            "alpha_fit": self.alpha,
            "wl_0_fit": self.wl_0,
            "epsilon_fit": self.epsilon,
            "residuals_rms": self.residuals_RMS,
            "residuals_rss": self.residuals_RSS,
            "residuals_ptv": self.residuals_PTV,
        }

        if plot:
            self.plot()

        return results

    def plot(
        self,
        num_samples: int = 100,
        save_filename: str | None = None,
        layout: bool = True,
        show: bool = True,
    ) -> None:
        wls = self.wls(n=num_samples)

        from matplotlib import pyplot as plt

        plt.rcParams["axes.labelsize"] = 14
        plt.rcParams["legend.fontsize"] = 12

        min_thickness = 10 * u.mm
        gap = 6 * u.mm
        D_ADC = 25.4 * u.mm

        num_subplots = 4 if layout else 3

        fig, axs = plt.subplots(
            num_subplots,
            1,
            figsize=(8, 2.5 * num_subplots),
            tight_layout=True,
        )

        # fig.suptitle(f"Optimised ADC using {self.g1.name} and {self.g2.name}", size=18)

        # Plot refractive indices of the two glasses
        ax = axs[0]

        ax.plot(wls, self.g1.n(wls), label=f"{self.g1.name}")
        ax.plot(wls, self.g2.n(wls), label=f"{self.g2.name}")

        ax.set_xticklabels([])
        # ax.ticklabel_format(axis="y", useOffset=1.00000000001)
        ax.set_ylabel("Refractive index $n$")
        ax.legend(frameon=True)

        # Plot scaled difference against the normalised refractivity of air
        ax = axs[1]

        ADC_index = self.alpha * (
            self.g1.n(wls) - self.epsilon * self.g2.n(wls) + self.epsilon - 1
        )

        ADC_index = ADC_index.to(u.degree)

        ax.plot(
            wls,
            ADC_index,
            "k--",
            label="2 $\\times$ ("
            + f"$n_1 - {self.epsilon:.5g}n_2 {self.epsilon - 1:+.3g}$) "
            + f"{self.alpha / 2:.3g}",
        )

        d_air = (
            self.beam_comp
            * self.z_max
            * (
                self.air_model.refractive_index(wls)
                - self.air_model.refractive_index(self.wl_0)
            )
        )

        d_air = d_air.to(u.degree)

        ax.plot(
            wls,
            d_air,
            "r-",
            label="$\\frac{D_{tel}}{D_{ADC}}$ ($n_{air} - n_{air}$("
            + f"{(self.wl_0.to(u.nm)):.0f})) $\\times$ "
            + f"tan({self.z_max:.0f})",
        )

        ax.axhline(y=0, color="k", ls="--", lw=1)
        ax.axvline(x=self.wl_0.to(u.um).value, color="k", ls="--", lw=1)
        ax.set_xticklabels([])
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.set_ylabel("Refractivity $\\approx n-1$")
        ax.legend(frameon=True)

        # residuals_curve = (ADC_index - d_air).to(u.mas)
        residuals_curve = self.residuals_function(wls=wls)

        # Plot angular residuals at maximum zenith angle
        ax = axs[2]
        ax.plot(
            wls,
            residuals_curve,
            label="Residuals"
            + "\nRMS: "
            + f"{self.residuals_RMS:.2f}"
            + "\nPTV: "
            + f"{self.residuals_PTV:.2f}",
        )

        ax.axhline(y=0, color="k", ls="--", lw=1)
        ax.axvline(x=self.wl_0.to(u.um).value, color="k", ls="--", lw=1)
        ax.set_xlabel("Wavelength (microns)")
        ax.set_ylabel("Angular dispersion\n" + "residuals at $z_{max}$ (mas)")
        ax.legend(frameon=True)

        if layout:
            # Plot cross-sectional layout of physical compound prisms
            ax = axs[3]

            ax.axhline(y=0, c="k", ls="--", lw=1)

            sign = self.first_apex_angle / abs(self.first_apex_angle)
            R_ADC = D_ADC / 2

            x1 = D_ADC * np.sin(np.radians(self.first_apex_angle))
            x2 = D_ADC * np.sin(np.radians(self.second_apex_angle))
            d = (min_thickness.value - x1.value) / 2 if x1 < min_thickness else 0

            # Plot first Amici prism
            ax.plot([0, 0], [-R_ADC.value, R_ADC.value], color="k", lw=1.3)
            ax.plot(
                [d, d + x1.value],
                [sign * R_ADC.value, -sign * R_ADC.value],
                color="k",
                lw=1.3,
            )
            ax.plot(
                [d + x1.value + d, d + d + x2.value],
                [-sign * R_ADC.value, sign * R_ADC.value],
                color="k",
                lw=1.3,
            )
            ax.plot(
                [0, d + d + x2.value],
                [sign * R_ADC.value, sign * R_ADC.value],
                color="k",
                lw=1.3,
            )
            ax.plot(
                [0, x1.value + 2 * d],
                [-sign * R_ADC.value, -sign * R_ADC.value],
                color="k",
                lw=1.3,
            )

            ax.annotate(
                text=f"{D_ADC:.3g}",
                xy=(-gap.value, R_ADC.value),
                xytext=(-gap.value, 0),
                arrowprops={"arrowstyle": "->"},
                ha="center",
                va="center",
                bbox={"boxstyle": "round", "fc": "white", "ec": "black", "pad": 0.2},
            )
            ax.annotate(
                text=f"{D_ADC:.3g}",
                xy=(-gap.value, -R_ADC.value),
                xytext=(-gap.value, 0),
                arrowprops={"arrowstyle": "->"},
                ha="center",
                va="center",
                bbox={"boxstyle": "round", "fc": "white", "ec": "black", "pad": 0.2},
            )

            # Plot second Amici prism
            xoffset = (max(x1, x2) + 2 * d * u.mm + gap).value
            ax.plot(
                [xoffset + 0, xoffset + 0],
                [-R_ADC.value, R_ADC.value],
                color="k",
                lw=1.3,
            )
            ax.plot(
                [xoffset + 0 + d, xoffset + x1.value + d],
                [-sign * R_ADC.value, sign * R_ADC.value],
                color="k",
                lw=1.3,
            )
            ax.plot(
                [xoffset + d + x1.value + d, xoffset + d + d + x2.value],
                [sign * R_ADC.value, -sign * R_ADC.value],
                color="k",
                lw=1.3,
            )
            ax.plot(
                [xoffset + 0, xoffset + x1.value + 2 * d],
                [sign * R_ADC.value, sign * R_ADC.value],
                color="k",
                lw=1.3,
            )
            ax.plot(
                [xoffset + 0, xoffset + x2.value + 2 * d],
                [-sign * R_ADC.value, -sign * R_ADC.value],
                color="k",
                lw=1.3,
            )

            # Plot annotations of distances
            xtotal = xoffset + (max(x1, x2) + 2 * d * u.mm).value
            ax.annotate(
                text=f"{xtotal:.3g}",
                xy=(0, -R_ADC.value / 2),
                xytext=(xtotal / 2, -R_ADC.value / 2),
                arrowprops={"arrowstyle": "->"},
                ha="center",
                va="center",
                bbox={"boxstyle": "round", "fc": "white", "ec": "black", "pad": 0.2},
            )

            ax.annotate(
                text=f"{xtotal:.3g}",
                xy=(xtotal, -R_ADC.value / 2),
                xytext=(xtotal / 2, -R_ADC.value / 2),
                arrowprops={"arrowstyle": "->"},
                ha="center",
                va="center",
                bbox={"boxstyle": "round", "fc": "white", "ec": "black", "pad": 0.2},
            )

            # For the legend only
            ax.plot(
                0,
                0,
                lw=0,
                label="Apex angles:\n"
                + f"$\\alpha_1 = ${self.first_apex_angle:.3f}\n"
                + f"$\\alpha_2 = ${self.second_apex_angle:.3f}",
            )

            # ax.plot(0, 0, label=f"Minimum compound\nprism thickness: "\
            # +"{abs(x1) + abs(max(abs(x1-x2), abs(x2-x1))):.3f}", lw=0)

            ax.axis("equal")
            ax.set_xlabel("Optical axis distance (mm)")
            ax.set_ylabel("Lateral axis\ndistance (mm)")

            ax.legend()

        if save_filename is not None:
            plt.savefig(save_filename)

            if show:
                plt.show()

        else:
            # Show the plot if no filename is given, even if show is False?
            plt.show()

    def __str__(self) -> str:
        return f"""
            Compound prism rotation al ADC with {self.g1} and {self.g2} glasses.
            Apex angles: {self.first_apex_angle:.3f} / \
            {self.second_apex_angle:.3f} (compound angle {self.beta:.3f}).
            Provides zero beam deviation at {self.wl_0:.3f}.
            """

    def __repr__(self) -> str:
        return (
            f"ADC(prism_stack = {self.prism_stack}, "
            + f"air_model = {self.air_model}, "
            + f"min_wl = {self.min_wl:.3f}, max_wl = {self.max_wl:.3f}, "
            + f"D_telescope = {self.D_telescope:.3f}, "
            + f"D_ADC = {self.D_ADC:.3f}, "
            + f"z_max = {self.z_max:.3f}, "
            + f"alpha = {self.alpha:.3f}, "
            + ")"
        )


def _refract(
    n_1: float | Callable,
    n_2: float | Callable,
    theta_in: Quantity = 0 * u.deg,
    wavelength: Quantity = None,
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
        try:
            _n_1 = n_1(wavelength)
        except TypeError:
            _n_1 = n_1
        try:
            _n_2 = n_2(wavelength)
        except TypeError:
            _n_2 = n_2

    # By this point, _n_1 and _n_2 should be either floats, ints, or
    # lists/arrays of such
    assert not callable(_n_1)
    assert not callable(_n_1)

    theta_out = np.arcsin((_n_1 / _n_2) * np.sin(theta_in))

    return theta_out.to(u.degree)


@dataclass
class Grating:
    name: str = ""
    catalogue_number: str = ""
    groove_frequency: Quantity = 31.6 / u.mm
    blaze_angle: Quantity = 76 * u.deg
    gamma_angle: Quantity = 0 * u.deg

    groove_length: Quantity | None = None
    ruled_width: Quantity | None = None

    min_wl: Quantity | None = None
    max_wl: Quantity | None = None

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
            return [self.blaze_wavelength(im) for im in m]

        return (self.m_times_lambda / m).to(u.um)

    def order_number(
        self,
        wl: float | Quantity = 500 * u.nm,
        quantised: bool = False,
    ) -> int | list[int] | float | list[float]:
        if isinstance(wl, tuple) or (isinstance(wl, Quantity) and wl.size > 1):
            print(f"Iterating over wavelengths {wl}")
            return [self.order_number(iwl) for iwl in wl]

        if isinstance(wl, int | float):
            print(f"Input was unitless value {wl}. Defaulting to {wl * u.um}")
            wl *= u.um

        if quantised:
            return int(round(float((self.m_times_lambda / wl).si), 0))

        return float((self.m_times_lambda / wl).si)

    def order_FSR(self, m: int | list[int], FSR_fraction: float = 1.0) -> Quantity:
        if isinstance(m, list | tuple):
            return [self.order_FSR(im) for im in m]

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
            return [self.order_opening_angle(im) for im in m]

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
        wl: Quantity = 550 * u.nm,
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
            (m * wl).to(u.um).value / (np.cos(self.gamma) * self.sigma.to(u.um).value)
        ) - np.sin(self.blaze_angle)
        return (np.arcsin(arcsin_argument) - self.blaze_angle).to(u.deg)

    def wavelength_at_angle(self, m: int, angle: Quantity) -> Quantity:
        # angle += self.blaze_angle

        return (
            np.cos(self.gamma) * self.sigma * (np.sin(self.blaze_angle) + np.sin(angle))
        ) / m

    def order_FSR_bounds_from_angle(self, m: int) -> tuple[Quantity]:
        angle = self.order_opening_angle(m=m) / 2

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

    @property
    def reddest_order_separation(self) -> Quantity:
        return self.order_FSR(m=self.order_number(self.max_wl), FSR_fraction=1.0)

    def __str__(self) -> str:
        return f"Grating: {self.catalogue_number}, R{self.R}, {self.groove_frequency.to(1 / u.mm).value:.3g} grooves/mm"

    def __repr__(self) -> str:
        return f"Grating(catalogue_number = {self.catalogue_number}, groove_frequency = {self.groove_frequency:.3f}, blaze_angle = {self.blaze_angle:.3f}, gamma_angle = {self.gamma_angle:.3f}, min_wl = {self.min_wl:.3f}, max_wl = {self.max_wl:.3f})"


@dataclass
class EchelleSpectrograph:
    name: str = ""
    grating: Grating = field(default_factory=Grating)
    cross_disperser: Prism = field(default_factory=Prism)
    beam_diameter: Quantity | None = None
    min_wl: Quantity = 390 * u.nm
    max_wl: Quantity = 950 * u.nm
    collimator_focal_length: Quantity = 1125 * u.mm
    slit_width: Quantity = 40 * u.um
    slit_length: Quantity = 0.5 * u.mm
    slit_native_focal_ratio: float = 3.85
    camera_focal_length: Quantity = 730 * u.mm
    detector_pixel_size: Quantity = 9 * u.um
    detector_nx: int = 10560
    detector_ny: int = 10560

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

    @property
    def detector_size(self) -> tuple[Quantity, Quantity]:
        return (
            self.detector_pixel_size * self.detector_nx,
            self.detector_pixel_size * self.detector_ny,
        )

    def plot_detector_edge(self, ax: plt.axes) -> None:
        halfx = self.detector_size[0].to(u.mm).value / 2
        halfy = self.detector_size[1].to(u.mm).value / 2

        ax.plot(
            [-halfx, -halfx, halfx, halfx, -halfx],
            [-halfy, halfy, halfy, -halfy, -halfy],
            ls="--",
            lw=4,
            color="k",
            alpha=0.25,
        )

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
    def slit_sampling(self, slit_width: Quantity | None = None) -> Quantity:
        if slit_width is None:
            slit_width = self.slit_width

        slit_image_width = slit_width * self.slit_magnification
        return float((slit_image_width / self.detector_pixel_size).si.value)

    @property
    def spectral_orders(self) -> list[int]:
        """ """

        return self.grating.diffraction_orders

    def order_wavelengths(
        self,
        m: int,
        num: int = 100,
        FSR_fraction: float = 1,
    ) -> list[Quantity]:
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

        grating_angle = self.grating.diffraction_angle(m=m, wl=wavelength)

        prism_angle = self.cross_disperser.dispersion_angle(
            wl=wavelength,
            field_angle=grating_angle,
        )

        return (
            prism_angle / self.camera_plate_scale,
            grating_angle / self.camera_plate_scale,
        )

    @cached_property
    def reddest_order_image_plane_separation(self) -> Quantity:
        """ """

        o1, o2, *_ = sorted(self.spectral_orders)

        wl1 = self.grating.blaze_wavelength(m=o1)
        wl2 = self.grating.blaze_wavelength(m=o2)

        x1 = self.image_plane_position(
            m=o1,
            wavelength=wl1,
        )
        x2 = self.image_plane_position(
            m=o2,
            wavelength=wl2,
        )

        return abs(x1[0] - x2[0])


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
        wavelength: Quantity | list[Quantity],
    ) -> tuple[Quantity, Quantity]:
        """ """

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

        _grating: Grating = Grating(
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
            + prism_first_pass_angle,
        )

        prism_second_pass_angle = 2 * _prism_second_pass.dispersion_angle(
            wl=wavelength,
            field_angle=grating_angle,
        )

        return (
            (prism_second_pass_angle / self.camera_plate_scale).to(u.mm).value,
            (grating_angle / self.camera_plate_scale).to(u.mm).value,
        ) * u.mm


@dataclass
class Etalon:
    """Description of plane wave illumination of an etalon a la Wikipedia"""

    n: float = 1
    thickness: Quantity = 6 * u.mm
    theta: Quantity = 0 * u.degree
    R_1: float = 0.95

    R_2: float | None = None

    def __post_init__(self) -> None:
        if self.R_2 is None:
            self.R_2 = self.R_1

        self.calculate_properties()

    @cached_property
    def R(self) -> float:
        return np.sqrt(self.R_1 * self.R_2)

    @cached_property
    def T(self) -> float:
        return 1 - self.R

    @cached_property
    def F_coef(self) -> float:
        return (4 * self.R) / ((1 - self.R) ** 2)

    @cached_property
    def finesse(self) -> float:
        return np.pi / (2 * np.arcsin(1 / np.sqrt(self.F_coef)))

    @cached_property
    def R_max(self) -> float:
        """
        Maximum reflectivity, R_max
        """
        return (4 * self.R) / ((1 + self.R) ** 2)

    def delta_lambda(self, wavelength: Quantity = 780 * u.nm) -> Quantity:
        """
        Calculate the free spectral range of the etalon, defaults to wavelength = 780nm
        """
        n_g = 1
        return (
            (wavelength**2)
            / (2 * n_g * self.thickness * np.cos(self.theta) + wavelength)
        ).to("pm")

    def delta(self, wavelength: Quantity) -> float:
        return (
            (2 * np.pi / wavelength) * 2 * self.n * self.thickness * np.cos(self.theta)
        ).si.value

    def Transmittance(self, wavelength: Quantity) -> float:
        """
        Transmittance function, sometimes T_e
        Transmitted beam intensity as a function of wavelength
        """
        return 1 / (1 + self.F_coef * (np.sin(self.delta(wavelength) / 2)) ** 2)

    def intensity(self, wavelength: Quantity) -> float:
        """Alias for Transmittance"""
        return self.Transmittance(wavelength)

    def amplitude(self, m: int, wavelength: Quantity) -> float:
        return self.T * (self.R**m) * np.exp(1j * m * self.delta(wavelength))

    def summed_amplitudes(self, wavelength: Quantity) -> float:
        """
        The geometric infinite sum of ampllitudes over m from 0 to infinity
        """
        return self.T / (1 - self.R * np.exp(1j * self.delta(wavelength)))

    def calculate_properties(self) -> None:
        """
        This function just calls each of the propert: Quantityy values in order to cache
        the calculated values
        """
        self.R  # noqa: B018
        self.T  # noqa: B018
        self.F_coef  # noqa: B018
        self.finesse  # noqa: B018
        self.R_max  # noqa: B018

        return

    def __str__(self) -> str:
        vals = "".join([f"{p:>12}: {val:<6.3g}\n" for p, val in self.__dict__.items()])

        return "Etalon:\n" + vals

    def __repr__(self) -> str:
        return (
            f"Etalon(n={self.n}, l={self.thickness}, θ={self.theta}, "
            + f"R_1={self.R_1}, R_2={self.R_2})"
        )


def main() -> None: ...


if __name__ == "__main__":
    main()
