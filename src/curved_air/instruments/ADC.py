from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np
from astropy import units as u
from astropy.units import Quantity
from matplotlib import pyplot as plt
from scipy.optimize import minimize

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from curved_air.glass import Glass
    from curved_air.type_hinting import ArrayQuantity, ScalarQuantity

from curved_air.optics.dispersers import Prism, _refract
from curved_air.other import Air, Model


@dataclass
class PrismStack:
    """ """

    prism_list: list[Prism] = field(default_factory=list)
    entrance_medium: Model = field(default_factory=Air)
    exit_medium: Model = field(default_factory=Air)

    def add_prism(self, prism: Prism) -> None:
        self.prism_list.append(prism)

    @overload
    def trace(
        self,
        wls: ScalarQuantity,
        incidence_angle: ScalarQuantity,
    ) -> ScalarQuantity: ...

    @overload
    def trace(
        self,
        wls: ScalarQuantity,
    ) -> ScalarQuantity: ...

    @overload
    def trace(
        self,
        wls: ArrayQuantity | None,
        incidence_angle: ScalarQuantity,
    ) -> ArrayQuantity: ...

    @overload
    def trace(
        self,
        wls: ArrayQuantity | None,
    ) -> ArrayQuantity: ...

    def trace(
        self,
        wls: ScalarQuantity | ArrayQuantity | None = None,
        incidence_angle: ScalarQuantity = 0 * u.deg,
        debug_level: float = 0,
    ) -> ScalarQuantity | ArrayQuantity:
        """
        Trace a set of rays through the PrismStack, solving Snell's Law at each
        interface
        """

        if wls is None:
            wls: ArrayQuantity = [0.4861, 0.5893, 0.6563] * u.um

        if wls.isscalar:
            rays: ScalarQuantity = incidence_angle
        else:
            rays: ArrayQuantity = incidence_angle * np.ones(len(wls))

        previous_material: Model = self.entrance_medium
        previous_angle: ScalarQuantity = 0 * u.degree

        assert isinstance(rays, Quantity)

        for _n, prism in enumerate(self.prism_list):
            rays: Quantity = _refract(
                n_1=previous_material.refractive_index(wls),
                n_2=prism.refractive_index,
                theta_in=rays - previous_angle,
                wavelength=wls,
            )

            if rays.isscalar:
                rays = [rays.value] * rays.unit

            if debug_level > 0:
                print(f"Refracting through a {prism.glass} prism of {prism.apex_angle}")

            previous_material: Glass = prism.glass
            previous_angle: Quantity = [
                r.to(u.deg).value + prism.apex_angle.to(u.deg).value for r in rays
            ] * u.deg

            # print(rays.to("degree"))

        # Refract through the last surface into the exit medium
        rays: Quantity = _refract(
            n_1=previous_material.refractive_index(wls),
            n_2=self.exit_medium.refractive_index(wls),
            theta_in=(rays - prism.apex_angle).to(u.deg),
            wavelength=wls,
        )

        return rays.to(u.degree)

    def zero_deviation_wavelengths(
        self,
        min_wl: ScalarQuantity | float = 0.39 * u.um,
        max_wl: ScalarQuantity | float = 1.50 * u.um,
        num_wls: int = 100,
        incidence_angle: ScalarQuantity = 0 * u.deg,
    ) -> Quantity:
        """
        This method traces the dispersion of light through a compound prism and returns
        the wavelengths at which the prism does not deviate the light. These
        zero-deviation wavelengths are interesting in design for astronomical ADCs, and
        may be a fast way to check that a design is working the way it should.
        """

        if isinstance(min_wl, Quantity) and isinstance(max_wl, Quantity):
            wls: ArrayQuantity = (
                np.linspace(min_wl.to("um").value, max_wl.to("um").value, num_wls)
                * u.um
            )
        elif isinstance(min_wl, int | float) and isinstance(max_wl, int | float):
            print(
                "No units provided for minimum and maximum wavelengths. Assuming microns."
            )
            wls: ArrayQuantity = np.linspace(min_wl, max_wl, num_wls) * u.um
        else:
            raise TypeError(
                "min_wl and max_wl parameters must be int, float, "
                + "or astropy.units.quantity.Quantity with appropriate units."
            )

        out_angles: ArrayQuantity = self.trace(wls=wls, incidence_angle=incidence_angle)
        # Find the wavelengths where the output angle changes sign, indicating a
        # zero-deviation point in between
        zwls: ArrayQuantity = wls[
            np.where(
                np.array(out_angles.value)[:-1] / np.array(out_angles.value)[1:] < 0
            )
        ]

        if len(zwls) >= 2:  # noqa: PLR2004
            return zwls
        if len(zwls) == 1:
            return zwls[0]

        raise ValueError("No zero-deviation wavelengths found in the given range.")

    def compute_last_apex_angle(self, wl_0: ScalarQuantity) -> ScalarQuantity:
        """Return the final prism apex angle required for zero deviation at wl_0.

        This uses the apex angles already stored on the Prism objects in the stack,
        and solves for the last prism's angle so that
            sum_i (n_i(wl_0) - 1) * alpha_i = 0.
        """
        if len(self.prism_list) < 2:  # noqa: PLR2004
            raise ValueError(
                "At least two prisms are required to solve for a final angle."
            )

        if not isinstance(wl_0, Quantity):
            raise TypeError("wl_0 must be an astropy Quantity with a wavelength unit.")

        refractivity_sum: ScalarQuantity = 0 * u.deg
        for prism in self.prism_list[:-1]:
            refractivity_sum += (
                prism.glass.refractive_index(wl_0) - 1
            ) * prism.apex_angle.to(u.deg)

        last_refractivity: float = self.prism_list[-1].glass.refractive_index(wl_0) - 1
        if np.isclose(last_refractivity, 0.0):
            raise ValueError(
                "The final prism has zero refractivity at wl_0; no finite apex angle can "
                "satisfy the zero-deviation condition."
            )

        self.prism_list[-1].apex_angle = (-refractivity_sum / last_refractivity).to(
            u.deg
        )

        return self.prism_list[-1].apex_angle

    def match_target_dispersion(
        self,
        min_wl: ScalarQuantity | float = 0.38 * u.um,
        max_wl: ScalarQuantity | float = 0.95 * u.um,
        target_dispersion: Model | type[Model] = Air,
    ) -> PrismStack:
        """Fit prism apex angles to a target dispersion curve.

        The optimisation is intentionally simple: it compares the effective dispersion
        from the `trace` method against the target model on a wavelength grid, removes
        the zero-point offset at the middle of the band, and normalises both curves
        before minimising their RMS difference.
        """

        if not self.prism_list:
            raise ValueError("Cannot optimise an empty PrismStack.")

        if isinstance(min_wl, Quantity) and isinstance(max_wl, Quantity):
            wls: ArrayQuantity = (
                np.linspace(min_wl.to(u.um).value, max_wl.to(u.um).value, 100) * u.um
            )
        elif isinstance(min_wl, int | float) and isinstance(max_wl, int | float):
            wls = np.linspace(min_wl, max_wl, 100) * u.um
        else:
            raise TypeError(
                "min_wl and max_wl parameters must be int, float, or astropy.units.quantity.Quantity with appropriate units."
            )

        if isinstance(target_dispersion, Model):
            target_model: Model = target_dispersion
        else:
            try:
                target_model: Model = target_dispersion()
            except Exception as e:
                raise TypeError(
                    "target_dispersion must be an instance of Model or a subclass of Model."
                ) from e

        initial_angles_deg: list[float] = [
            prism.apex_angle.to(u.deg).value for prism in self.prism_list
        ]
        reference_index: int = len(wls) // 2
        finite_penalty: float = 1e9
        angle_limit_deg: float = 60

        def objective(angle_values_deg: list[float] | NDArray[np.float64]) -> float:
            original_angles_deg: list[float] = [
                prism.apex_angle.to(u.deg).value for prism in self.prism_list
            ]
            trial_angles_deg = np.asarray(angle_values_deg, dtype=float)

            try:
                if not np.all(np.isfinite(trial_angles_deg)):
                    return finite_penalty

                if np.any(np.abs(trial_angles_deg) >= angle_limit_deg):
                    return finite_penalty

                for prism, angle_value in zip(
                    self.prism_list, trial_angles_deg, strict=True
                ):
                    prism.apex_angle = angle_value * u.deg

                stack_curve: NDArray[np.float64] = np.asarray(
                    self.trace(wls=wls).to(u.deg).value,
                    dtype=float,
                )
                target_curve: NDArray[np.float64] = np.asarray(
                    target_model.refractive_index(wls),
                    dtype=float,
                )

                if not np.all(np.isfinite(stack_curve)) or not np.all(
                    np.isfinite(target_curve)
                ):
                    return finite_penalty

                stack_reference = float(stack_curve[reference_index])
                target_reference = float(target_curve[reference_index])

                stack_curve = stack_curve - stack_reference
                target_curve = target_curve - target_reference

                residuals = stack_curve - target_curve

                if not np.all(np.isfinite(residuals)):
                    return finite_penalty

                return float(np.sqrt(np.mean(np.square(residuals))))
            except Exception:
                return finite_penalty
            finally:
                # Reset the prism angles to their original values
                for prism, angle in zip(
                    self.prism_list, original_angles_deg, strict=True
                ):
                    prism.apex_angle = angle * u.deg

        result = minimize(
            objective,
            x0=initial_angles_deg,
            method="Powell",
            bounds=[(-angle_limit_deg, angle_limit_deg)] * len(self.prism_list),
            options={"maxiter": 200, "xtol": 1e-4, "ftol": 1e-6},
        )

        if not result.success:
            raise ValueError(f"Unable to optimise prism angles: {result.message}")

        for prism, angle_value in zip(self.prism_list, result.x, strict=True):
            prism.apex_angle = angle_value * u.deg

        return self

    @property
    def summary(self) -> str:
        summary_lines: list[str] = [
            f"PrismStack with {len(self.prism_list)} prisms:",
            f"  Entrance medium: {self.entrance_medium}",
        ]

        for i, prism in enumerate(self.prism_list, start=1):
            summary_lines.append(
                f"  Prism {i}: Glass = {prism.glass}, Apex angle = {prism.apex_angle:.3f}, n_d = {prism.glass.n(0.5893 * u.um):.6f}"
            )

        summary_lines.append(f"  Exit medium: {self.exit_medium}")

        return "\n".join(summary_lines)


@dataclass
class ADC:
    prism_stack: PrismStack
    air_model: Model = field(default_factory=Air)
    min_wl: ScalarQuantity = 390 * u.nm
    max_wl: ScalarQuantity = 950 * u.nm
    D_telescope: ScalarQuantity = 0.8 * u.m
    D_ADC: ScalarQuantity = 15 * u.mm
    z_max: ScalarQuantity = 65 * u.degree
    criterion: str = "PTV"  # "RMS", "RSS", "PTV", "max_abs"

    alphas: ScalarQuantity | None = None
    wl_0: ScalarQuantity | None = None

    residuals_function: Callable | None = None

    debug_level: int = 0

    def __post_init__(self) -> None:
        for p in self.prism_stack.prism_list:
            if not isinstance(p, Prism):
                raise TypeError(
                    f"All elements of prism_stack must be instances of Prism, but got {type(p)}"
                )
            if p.min_wl is None:
                p.min_wl: ScalarQuantity = self.min_wl
            if p.max_wl is None:
                p.max_wl: ScalarQuantity = self.max_wl

    @property
    def g1(self) -> Prism | Glass:
        return self.prism_stack.prism_list[0]

    @property
    def g2(self) -> Prism | Glass:
        return self.prism_stack.prism_list[1]

    @property
    def epsilon(self) -> float | None:
        if self.wl_0 is None:
            return None
        return (self.g1.refractive_index(self.wl_0) - 1) / (
            self.g2.refractive_index(self.wl_0) - 1
        )

    @property
    def alpha2(self) -> ScalarQuantity:
        assert isinstance(self.alpha, Quantity)
        assert isinstance(self.beta, Quantity)

        return (self.alpha + self.beta).to(u.deg)

    @property
    def beam_comp(self) -> float:
        return (self.D_telescope / self.D_ADC).si

    @property
    def magnification_at_ADC(self) -> float:
        return self.beam_comp

    def wls(self, n: int = 100) -> ArrayQuantity:
        return (
            np.linspace(self.min_wl.to(u.um).value, self.max_wl.to(u.um).value, n)
            * u.um
        )

    @property
    def residuals_RMS(self) -> ScalarQuantity:
        assert self.residuals_function is not None
        RMS: float = np.sqrt(np.mean(self.residuals_function(wls=self.wls(100)) ** 2))

        return RMS.to(u.mas)

    @property
    def residuals_RSS(self) -> ScalarQuantity:
        assert self.residuals_function is not None
        RSS: float = np.sqrt(np.sum(self.residuals_function(wls=self.wls(100)) ** 2))

        return RSS.to(u.mas)

    @property
    def residuals_PTV(self) -> ScalarQuantity:
        assert self.residuals_function is not None
        PTV: float = abs(
            max(self.residuals_function(wls=self.wls(100)))
            - min(self.residuals_function(wls=self.wls(100)))
        )

        return PTV.to(u.mas)

    @property
    def residuals_max_abs(self) -> ScalarQuantity:
        assert self.residuals_function is not None
        max_abs: float = max(
            np.abs(self.residuals_function(wls=self.wls(100)).to(u.mas).value)
        )

        return max_abs * u.mas

    @property
    def apex_angles(self) -> list[ScalarQuantity] | None:
        return [p.apex_angle for p in self.prism_stack.prism_list]

    def optimise(
        self,
        plot: bool = False,
        num_samples: int = 20,
        # allow_compound_angle: bool = True
    ) -> dict | None:
        """
        This function finds the best main apex angle and zero-deviation
        wavelength for the ADC's dispersion to match the dispersion of air.

        TODO:

        - [x] Instead of computing residuals based on the difference between
        refractive indices, actually trace rays through the ADC

        - [x] Allow to match different target dispersion??? - Now in PrismStack.match_target_dispersion()

        - [ ] Allow optimising without compound angle?
        """

        # if len(self.prism_stack.prism_list) != 2:  # noqa: PLR2004
        #     raise NotImplementedError(
        #         "This method is only implemented for two-component compound prisms."
        #     )

        g1: Prism = self.prism_stack.prism_list[0]
        g2: Prism = self.prism_stack.prism_list[1]

        wls: ArrayQuantity = self.wls(n=num_samples)

        # Find a linear combination of refractive index curves that best match
        # the refractive index of air, in which the apex angles of the two prism
        def residuals(
            x: tuple[float, ...] = (1, 0.6328),  # alpha, wl_0 - no units!
            wls: ArrayQuantity = wls,
            return_residuals: bool = False,
            verbose: bool = self.debug_level > 0,
        ) -> Quantity | float:

            # Save olf apex angles?
            old_apex_angles: list[ScalarQuantity] = [
                prism.apex_angle for prism in self.prism_stack.prism_list
            ]

            # Fully parameterised by alpha and wl_0
            *alphas, wl_0 = x

            if any(np.isnan(a) for a in alphas) or np.isnan(wl_0):
                return np.inf

            assert len(alphas) == len(self.prism_stack.prism_list) - 1

            alphas: list[ScalarQuantity] = [a * u.degree for a in alphas]

            wl_0: ScalarQuantity = wl_0 * u.um

            if verbose:
                print(f"Trying: {alphas = }, {wl_0 = }")

            for prism, angle in zip(
                self.prism_stack.prism_list[:-1], alphas, strict=True
            ):
                prism.apex_angle: ScalarQuantity = angle

            last_apex_angle: ScalarQuantity = self.prism_stack.compute_last_apex_angle(
                wl_0=wl_0
            )

            if last_apex_angle > 60 * u.deg or last_apex_angle < -60 * u.deg:
                if verbose:
                    print(
                        f"Last apex angle {last_apex_angle:.3f} is out of bounds, skipping."
                    )
                return np.inf

            delta_air: ArrayQuantity = (
                self.beam_comp
                * (np.tan(self.z_max) * u.radian)
                * (
                    self.air_model.refractive_index(wls)
                    - self.air_model.refractive_index(wl_0)
                )
            ).si.to(u.degree)

            delta_ADC: ArrayQuantity = self.prism_stack.trace(
                wls=wls
            ) - 1 * self.prism_stack.trace(wls=wl_0)

            residuals: ArrayQuantity = delta_ADC - delta_air
            # Put it back in equivalent on-sky angle
            residuals /= np.tan(self.z_max) * self.beam_comp

            # Reset prism angles?
            for prism, angle in zip(
                self.prism_stack.prism_list, old_apex_angles, strict=True
            ):
                prism.apex_angle: ScalarQuantity = angle

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
            x0: list[float] = [1.0] * (len(self.prism_stack.prism_list) - 1) + [
                np.median(wls.to(u.um).value)
            ]

            bounds: list[tuple[float, float]] = [(-np.inf, np.inf)] * (
                len(self.prism_stack.prism_list) - 1
            ) + [(0.4, 1.5)]

            result = minimize(fun=residuals, x0=x0, bounds=bounds)

            *alphas_fit, wl_0_fit = result.x

        except Exception as e:
            raise e

        self.alphas: list[ScalarQuantity] = [a * u.degree for a in alphas_fit]
        self.wl_0: ScalarQuantity = wl_0_fit * u.um

        # residuals_PTV = residuals(x = result.x)
        # residuals_curve = residuals(x = result.x, return_residuals=True)

        def residuals_function(wls: ArrayQuantity = wls) -> Quantity | float:

            assert isinstance(self.alphas, list)

            x = tuple(
                [a.to(u.degree).value for a in self.alphas] + [self.wl_0.to(u.um).value]
            )

            return residuals(x=x, wls=wls, return_residuals=True)

        self.residuals_function: Callable = residuals_function

        # Solve for individual prism apex angle from fitted constant

        for prism, angle in zip(
            self.prism_stack.prism_list[:-1], alphas_fit, strict=True
        ):
            prism.apex_angle: ScalarQuantity = angle * u.degree
        self.prism_stack.compute_last_apex_angle(wl_0=wl_0_fit * u.um)

        self.wl_0: ScalarQuantity = wl_0_fit * u.um

        results: dict[str, Any] = {
            "g1": g1,
            "g2": g2,
            "D_telescope": self.D_telescope,
            "D_ADC": self.D_ADC,
            "beam_comp": self.beam_comp,
            "z_max": self.z_max,
            "min_wl": self.min_wl,
            "max_wl": self.max_wl,
            "wl_0_fit": self.wl_0,
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
    ) -> plt.Figure:
        """
        Plot the optical properties, performance, and layout of the ADC
        """

        assert isinstance(self.alphas, list)
        assert all(isinstance(alpha, Quantity) for alpha in self.alphas)
        assert isinstance(self.wl_0, Quantity)

        if layout and len(self.prism_stack.prism_list) > 2:  # noqa: PLR2004
            layout = False

        wls: ArrayQuantity = self.wls(n=num_samples)

        plt.rcParams["axes.labelsize"] = 14
        plt.rcParams["legend.fontsize"] = 12

        min_thickness: ArrayQuantity = 7.5 * u.mm  # TODO: don't hardcode
        gap: ArrayQuantity = 5 * u.mm  # TODO: don't hardcode
        D_ADC: ArrayQuantity = self.D_ADC

        num_subplots: Literal[4, 3] = 4 if layout else 3

        fig, axs = plt.subplots(
            num_subplots,
            1,
            figsize=(8, 2.5 * num_subplots),
            tight_layout=True,
        )

        # fig.suptitle(f"Optimised ADC using {self.g1.name} and {self.g2.name}", size=18)

        # Plot refractive indices of the two glasses
        ax: plt.Axes = axs[0]

        for p in self.prism_stack.prism_list:
            ax.plot(wls, p.glass.n(wls), label=f"{p.glass.name}")

        ax.set_xticklabels([])
        # ax.ticklabel_format(axis="y", useOffset=1.00000000001)
        ax.set_ylabel("Refractive index $n$")
        ax.legend(frameon=True)

        # Plot scaled difference against the normalised refractivity of air
        ax: plt.Axes = axs[1]

        traced_angles = self.prism_stack.trace(wls=wls) - self.prism_stack.trace(
            wls=self.wl_0
        )

        ax.plot(
            wls,
            traced_angles.to(u.deg),
            "k--",
            label="Traced ray angles through ADC",
        )

        ax.plot(
            wls,
            self.beam_comp
            * (np.tan(self.z_max) * u.radian).to(u.deg)
            * (
                self.air_model.refractive_index(wls)
                - self.air_model.refractive_index(self.wl_0)
            ),
            "r-",
            alpha=0.75,
            label="$\\frac{D_{tel}}{D_{ADC}}$ ($n_{air}(\\lambda) - n_{air}$("
            + f"{(self.wl_0.to(u.nm)):.0f})) $\\times$ "
            + f"tan({self.z_max:.0f})",
        )

        ax.axhline(y=0, color="k", ls="--", lw=1)
        ax.axvline(x=self.wl_0.to(u.um).value, color="k", ls="--", lw=1)
        ax.set_xticklabels([])
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.set_ylabel("Refractivity $\\approx n-1$")
        ax.legend(frameon=True)

        if callable(self.residuals_function):
            # residuals_curve = (ADC_index - d_air).to(u.mas)
            residuals_curve = self.residuals_function(wls=wls)

            # Plot angular residuals at maximum zenith angle
            ax = axs[2]
            ax.plot(
                wls,
                residuals_curve.to(u.mas),
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
            ax: plt.Axes = axs[3]

            ax.axhline(y=0, c="k", ls="--", lw=1)

            sign: Literal[1, -1] = self.first_apex_angle / abs(self.first_apex_angle)
            R_ADC: ScalarQuantity = D_ADC / 2

            x1: ScalarQuantity = D_ADC * np.sin(np.radians(self.first_apex_angle))
            x2: ScalarQuantity = D_ADC * np.sin(np.radians(self.second_apex_angle))
            d: float = (min_thickness.value - x1.value) / 2 if x1 < min_thickness else 0

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
            xtotal: float = xoffset + (max(x1, x2) + 2 * d * u.mm).value
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

        return fig

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


def test() -> None:
    from curved_air.glass import GlassCatalogue  # noqa: PLC0415
    from curved_air.optics.dispersers import Prism  # noqa: PLC0415
    from curved_air.other import Constant, Linear  # noqa: PLC0415, F401

    gc = GlassCatalogue(["Schott", "Ohara"])

    stack: PrismStack = PrismStack(
        prism_list=[
            Prism(glass=gc["K10"], apex_angle=2 * u.deg),
            Prism(glass=gc["S-FPL51"], apex_angle=-2 * u.deg),
            Prism(glass=gc["F2"], apex_angle=2 * u.deg),
            Prism(glass=gc["N-BK7"], apex_angle=-2 * u.deg),
        ]
    )
    # target_dispersion = Linear(m=0.1 / u.um, b=0)
    target_dispersion = Linear(m=-1 / u.um, b=0)
    # target_dispersion = Constant(c=1)
    # target_dispersion = Air()

    # print(target_dispersion.refractive_index(wl=np.linspace(0.38, 0.95, 100) * u.um))

    stack.match_target_dispersion(
        min_wl=380 * u.nm,
        max_wl=950 * u.nm,
        target_dispersion=target_dispersion,
    )

    print(stack.summary)

    _fig, axs = plt.subplots(
        2, 1, figsize=(8, 6), tight_layout=True, sharex=True, height_ratios=[2, 1]
    )

    ax = axs[0]

    wls = np.linspace(0.38, 0.95, 100) * u.um

    traced_angles = (
        stack.trace(wls, incidence_angle=0 * u.deg)
        - stack.trace(wls, incidence_angle=0 * u.deg)[len(wls) // 2]
    )

    ax.plot(
        wls,
        traced_angles,  # / np.max(np.abs(traced_angles)),
        label="Output angle (zeroed)",
    )

    target_angles = (
        target_dispersion.refractive_index(wls)
        - target_dispersion.refractive_index(wls)[len(wls) // 2]
    )

    # plot the target dispersion curve for comparison
    ax.plot(
        wls,
        target_angles,  # / np.max(np.abs(target_angles)),
        label="Target dispersion (zeroed)",
    )

    ax.legend()

    ax = axs[1]
    ax.plot(
        wls,
        traced_angles.to(u.deg).value - target_angles,
        label="Residuals (traced - target)",
    )

    ax.legend()

    plt.show()

    # adc: ADC = ADC(prism_stack=stack)

    # results: dict | None = adc.optimise(plot=True)

    # print(results)


if __name__ == "__main__":
    test()
