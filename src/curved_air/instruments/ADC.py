from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from astropy import units as u
from astropy.units import Quantity
from scipy.optimize import minimize
from matplotlib import pyplot as plt

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from numpy.typing import ArrayLike

    from curved_air.glass import Glass
    from curved_air.optics.dispersers import Prism

from curved_air.other import Air, Model
from curved_air.optics.dispersers import _refract


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
        num_wls: int = 100,
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
            wls: Quantity = (
                np.linspace(min_wl.to("um").value, max_wl.to("um").value, num_wls)
                * u.um
            )
        elif isinstance(min_wl, int | float) and isinstance(max_wl, int | float):
            wls: Quantity = np.linspace(min_wl, max_wl, num_wls) * u.um
        else:
            raise TypeError(
                "min_wl and max_wl parameters must be int, float, "
                + "or astropy.units.quantity.Quantity with appropriate units."
            )

        out_angles: Quantity = self.trace(wls=wls, incidence_angle=incidence_angle)
        zwls: Quantity = wls[
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
    def epsilon(self) -> Quantity | None:
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
    ) -> dict | None:
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
        ) -> Quantity | float:
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

        def residuals_function(wls: ArrayLike = wls) -> Quantity | float:
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

        plt.rcParams["axes.labelsize"] = 14
        plt.rcParams["legend.fontsize"] = 12

        min_thickness = 10 * u.mm
        gap = 6 * u.mm
        D_ADC = 25.4 * u.mm

        num_subplots = 4 if layout else 3

        _fig, axs = plt.subplots(
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
