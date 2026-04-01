from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt
from scipy.optimize import minimize

if TYPE_CHECKING:
    from collections.abc import Callable

    from astropy.units import Quantity

    from curved_air.glass import Glass
    from curved_air.instruments.ADC import PrismStack
    from curved_air.optics.dispersers import Prism
    from curved_air.type_hinting import ArrayQuantity, ScalarQuantity

from curved_air.other import Air, Model


@dataclass
class ADC3:
    prism_stack: PrismStack
    air_model: Model = field(default_factory=Air)
    min_wl: ScalarQuantity = 390 * u.nm
    max_wl: ScalarQuantity = 950 * u.nm
    D_telescope: ScalarQuantity = 0.8 * u.m
    D_ADC: ScalarQuantity = 15 * u.mm
    z_max: ScalarQuantity = 65 * u.degree
    criterion: str = "PTV"  # "RMS", "RSS", "PTV", "max_abs"

    wl_0: ScalarQuantity | None = None
    apex_angles: list[ScalarQuantity] | None = None
    incidence_angle: ScalarQuantity | None = None

    residuals_function: Callable | None = None

    def __post_init__(self) -> None:
        if len(self.prism_stack.prism_list) != 3:  # noqa: PLR2004
            raise NotImplementedError(
                "This class is only implemented for three-component compound prisms only."
            )

    @property
    def g1(self) -> Prism | Glass:
        return self.prism_stack.prism_list[0]

    @property
    def g2(self) -> Prism | Glass:
        return self.prism_stack.prism_list[1]

    @property
    def g3(self) -> Prism | Glass:
        return self.prism_stack.prism_list[2]

    @property
    def beam_comp(self) -> float:
        return (self.D_telescope / self.D_ADC).si

    def wls(self, n: int = 100) -> ArrayQuantity:
        return (
            np.linspace(
                self.min_wl.to(u.um).value,
                self.max_wl.to(u.um).value,
                n,
            )
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

    def optimise(
        self,
        plot: bool = False,
        num_samples: int = 20,
    ) -> dict | None:
        """
        Optimize the apex angles of a three-element compound prism to match the dispersion of air.

        This method optimizes 5 parameters:
        - wl_0: zero-deviation wavelength
        - alpha1, alpha2, alpha3: apex angles of the three prisms
        - incidence_angle: angle of incidence on the first prism

        The optimization minimizes the residual between the ADC's dispersion curve and air's
        dispersion curve across the wavelength range.
        """

        wls: ArrayQuantity = self.wls(n=num_samples)
        wl_min = self.min_wl.to(u.um).value
        wl_max = self.max_wl.to(u.um).value
        wl_med = np.median(wls.to(u.um).value)

        def residuals(
            x: list[float],
            wls: ArrayQuantity = wls,
            return_residuals: bool = False,
            verbose: bool = False,
        ) -> float | Quantity:
            """
            Compute residuals between air dispersion and ADC dispersion.

            Parameters:
            x = [wl_0, alpha1, alpha2, alpha3, incidence_angle]
            """
            wl_0, alpha1, alpha2, alpha3, incidence_angle = x

            wl_0: ScalarQuantity = wl_0 * u.um
            incidence_angle: ScalarQuantity = incidence_angle * u.deg

            if verbose:
                print(
                    f"Trying: {wl_0 = }, alpha = [{alpha1:.3f}, {alpha2:.3f}, {alpha3:.3f}] deg, {incidence_angle = }"
                )

            # Update prism apex angles temporarily for ray tracing
            old_angles = [
                self.g1.apex_angle.copy(),
                self.g2.apex_angle.copy(),
                self.g3.apex_angle.copy(),
            ]

            try:
                self.g1.apex_angle: Quantity = alpha1 * u.deg
                self.g2.apex_angle: Quantity = alpha2 * u.deg
                self.g3.apex_angle: Quantity = alpha3 * u.deg

                # Compute the effective deviation angle from the prism stack
                delta_ADC = self.prism_stack.trace(
                    wls=wls,
                    incidence_angle=incidence_angle,
                ).to(u.degree)

                # Compute the air dispersion relative to the zero-deviation wavelength
                delta_air = (
                    self.beam_comp
                    * self.z_max
                    * (
                        self.air_model.refractive_index(wls)
                        - self.air_model.refractive_index(wl_0)
                    )
                ).si.to(u.degree)

                # Compute residuals
                residuals_vec = (delta_ADC - delta_air).to(u.degree)
                # Convert back to on-sky equivalent angle
                residuals_vec = residuals_vec / (self.z_max * self.beam_comp)

                if return_residuals:
                    return residuals_vec.to(u.mas)

                # Compute optimization criterion
                if self.criterion == "PTV":
                    return abs(
                        max(residuals_vec.to(u.mas)) - min(residuals_vec.to(u.mas))
                    ).value
                if self.criterion == "RMS":
                    return np.sqrt(np.mean(residuals_vec.to(u.mas).value ** 2))
                if self.criterion == "RSS":
                    return np.sqrt(np.sum(residuals_vec.to(u.mas).value ** 2))
                if self.criterion == "max_abs":
                    return max(np.abs(residuals_vec.to(u.mas).value))

                return np.nan

            finally:
                # Restore original apex angles
                self.g1.apex_angle = old_angles[0]
                self.g2.apex_angle = old_angles[1]
                self.g3.apex_angle = old_angles[2]

        # Initial guess: use small positive and alternating negative angles,
        # zero deviation at mid wavelength, normal incidence
        x0 = [
            wl_med,
            10.0,  # alpha1
            -6.0,  # alpha2 (negative to provide opposite dispersion)
            3.0,  # alpha3 (positive to fine-tune)
            0.0,  # incidence angle
        ]

        bounds = [
            (wl_min, wl_max),  # wl_0
            (-20.0, 20.0),  # alpha1
            (-20.0, 20.0),  # alpha2
            (-20.0, 20.0),  # alpha3
            (-10.0, 10.0),  # incidence angle
        ]

        # try:
        result = minimize(
            fun=residuals,
            x0=x0,
            bounds=bounds,
            method="L-BFGS-B",
        )

        if not result.success:
            print(f"Warning: Optimization did not converge. Message: {result.message}")

        wl_0_fit, alpha1_fit, alpha2_fit, alpha3_fit, incidence_fit = result.x

        # except Exception as e:
        #     print(f"Error during optimization: {e}")
        #     print("Unable to converge on solution.")
        #     return None

        # Store results
        self.wl_0: ScalarQuantity = wl_0_fit * u.um
        self.apex_angles: list[ScalarQuantity] = [
            alpha1_fit * u.deg,
            alpha2_fit * u.deg,
            alpha3_fit * u.deg,
        ]
        self.incidence_angle: ScalarQuantity = incidence_fit * u.deg

        # Create residuals function for later evaluation
        def residuals_function(wls: ArrayQuantity) -> Quantity:
            return residuals(
                x=[
                    self.wl_0.to(u.um).value,
                    self.apex_angles[0].to(u.degree).value,
                    self.apex_angles[1].to(u.degree).value,
                    self.apex_angles[2].to(u.degree).value,
                    self.incidence_angle.to(u.degree).value,
                ],
                wls=wls,
                return_residuals=True,
            )

        self.residuals_function: Callable = residuals_function

        results: dict[str, Any] = {
            "g1": self.g1,
            "g2": self.g2,
            "g3": self.g3,
            "D_telescope": self.D_telescope,
            "D_ADC": self.D_ADC,
            "beam_comp": self.beam_comp,
            "z_max": self.z_max,
            "min_wl": self.min_wl,
            "max_wl": self.max_wl,
            "apex_angles": self.apex_angles,
            "incidence_angle": self.incidence_angle,
            "wl_0": self.wl_0,
            "residuals_rms": self.residuals_RMS,
            "residuals_rss": self.residuals_RSS,
            "residuals_ptv": self.residuals_PTV,
            "optimization_success": result.success,
            "optimization_message": result.message,
        }

        if plot:
            self.plot()

        return results

    def plot(
        self,
        num_samples: int = 100,
        save_filename: str | None = None,
        show: bool = True,
    ) -> plt.Figure:
        """
        Plot the three-element ADC optimization results.

        Shows:
        1. Refractive indices of the three glasses
        2. Effective ADC dispersion vs. air dispersion
        3. Residuals
        """
        assert self.residuals_function is not None
        assert self.apex_angles is not None
        assert self.wl_0 is not None

        wls: ArrayQuantity = self.wls(n=num_samples)

        fig, axs = plt.subplots(
            3,
            1,
            figsize=(10, 10),
            tight_layout=True,
        )

        # Plot 1: Refractive indices
        ax = axs[0]
        ax.plot(wls, self.g1.n(wls), label=f"{self.g1.name}", linewidth=2)
        ax.plot(wls, self.g2.n(wls), label=f"{self.g2.name}", linewidth=2)
        ax.plot(wls, self.g3.n(wls), label=f"{self.g3.name}", linewidth=2)
        ax.set_ylabel("Refractive index $n$")
        ax.legend(frameon=True)
        ax.grid(True, alpha=0.3)
        ax.set_xticklabels([])

        # Plot 2: Effective dispersion
        ax = axs[1]

        # Update prism angles and trace through the stack
        old_angles = [
            self.g1.apex_angle.copy(),
            self.g2.apex_angle.copy(),
            self.g3.apex_angle.copy(),
        ]

        try:
            self.g1.apex_angle = self.apex_angles[0]
            self.g2.apex_angle = self.apex_angles[1]
            self.g3.apex_angle = self.apex_angles[2]

            delta_ADC = self.prism_stack.trace(
                wls=wls,
                incidence_angle=self.incidence_angle
                if self.incidence_angle is not None
                else 0 * u.deg,
            ).to(u.degree)
        finally:
            self.g1.apex_angle = old_angles[0]
            self.g2.apex_angle = old_angles[1]
            self.g3.apex_angle = old_angles[2]

        ax.plot(
            wls,
            delta_ADC,
            "k--",
            linewidth=2,
            label=f"ADC (α₁={self.apex_angles[0]:.2f}, α₂={self.apex_angles[1]:.2f}, α₃={self.apex_angles[2]:.2f})",
        )

        delta_air = (
            self.beam_comp
            * self.z_max
            * (
                self.air_model.refractive_index(wls)
                - self.air_model.refractive_index(self.wl_0)
            )
        ).si.to(u.degree)

        ax.plot(
            wls,
            delta_air,
            "r-",
            linewidth=2,
            label=f"Air (zero at {self.wl_0:.3f})",
        )

        ax.axhline(y=0, color="k", ls="--", lw=1, alpha=0.5)
        ax.axvline(x=self.wl_0.to(u.um).value, color="k", ls="--", lw=1, alpha=0.5)
        ax.set_ylabel("Effective dispersion angle\n(degrees)")
        ax.legend(frameon=True)
        ax.grid(True, alpha=0.3)
        ax.set_xticklabels([])

        # Plot 3: Residuals
        ax = axs[2]
        residuals_curve = self.residuals_function(wls=wls)
        ax.plot(
            wls,
            residuals_curve,
            "b-",
            linewidth=2,
            label="Residuals"
            + f"\nRMS: {self.residuals_RMS:.2f}"
            + f"\nPTV: {self.residuals_PTV:.2f}",
        )

        ax.axhline(y=0, color="k", ls="--", lw=1, alpha=0.5)
        ax.axvline(x=self.wl_0.to(u.um).value, color="k", ls="--", lw=1, alpha=0.5)
        ax.set_xlabel("Wavelength (microns)")
        ax.set_ylabel("Angular residuals\n(mas)")
        ax.legend(frameon=True)
        ax.grid(True, alpha=0.3)

        if save_filename:
            fig.savefig(save_filename, dpi=150)

        if show:
            plt.show()

        return fig

    def __str__(self) -> str:
        return f"""
            Compound prism rotational ADC with {self.g1}, {self.g2}, and {self.g3} glasses.
            Apex angles: {[f"{a:.2f}" for a in self.apex_angles] if self.apex_angles is not None else "None"}.
            Provides zero beam deviation at {self.wl_0:.3f}.
            """

    def __repr__(self) -> str:
        return (
            f"ADC3(prism_stack = {self.prism_stack}, "
            + f"air_model = {self.air_model}, "
            + f"min_wl = {self.min_wl:.3f}, max_wl = {self.max_wl:.3f}, "
            + f"D_telescope = {self.D_telescope:.3f}, "
            + f"D_ADC = {self.D_ADC:.3f}, "
            + f"z_max = {self.z_max:.3f}, "
            + ")"
        )
