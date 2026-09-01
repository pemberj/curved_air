from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import TYPE_CHECKING, Any

import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from tqdm import tqdm

if TYPE_CHECKING:
    from collections.abc import Callable

    from astropy.units import Quantity
    from numpy.typing import NDArray

    from curved_air.glass import Glass
    from curved_air.instruments.ADC import ADC, PrismStack
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
    def p1(self) -> Prism:
        return self.prism_stack.prism_list[0]

    @property
    def p2(self) -> Prism:
        return self.prism_stack.prism_list[1]

    @property
    def p3(self) -> Prism:
        return self.prism_stack.prism_list[2]

    @property
    def g1(self) -> Glass:
        return self.prism_stack.prism_list[0].glass

    @property
    def g2(self) -> Glass:
        return self.prism_stack.prism_list[1].glass

    @property
    def g3(self) -> Glass:
        return self.prism_stack.prism_list[2].glass

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
    def residuals_RMS(self) -> float:
        assert self.residuals_function is not None
        RMS: float = np.sqrt(np.mean(self.residuals_function(wls=self.wls(100)) ** 2))

        return RMS

    @property
    def residuals_RSS(self) -> float:
        assert self.residuals_function is not None
        RSS: float = np.sqrt(np.sum(self.residuals_function(wls=self.wls(100)) ** 2))

        return RSS

    @property
    def residuals_PTV(self) -> float:
        assert self.residuals_function is not None
        PTV: float = abs(
            max(self.residuals_function(wls=self.wls(100)))
            - min(self.residuals_function(wls=self.wls(100)))
        )

        return PTV

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
        wl_min: float = self.min_wl.to(u.um).value
        wl_max: float = self.max_wl.to(u.um).value
        wl_med: float = np.median(wls.to(u.um).value)

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
            old_angles: list[ScalarQuantity] = [
                self.p1.apex_angle.copy(),
                self.p2.apex_angle.copy(),
                self.p3.apex_angle.copy(),
            ]

            try:
                self.p1.apex_angle: Quantity = alpha1 * u.deg
                self.p2.apex_angle: Quantity = alpha2 * u.deg
                self.p3.apex_angle: Quantity = alpha3 * u.deg

                # Compute the effective deviation angle from the prism stack
                delta_ADC: ArrayQuantity = self.prism_stack.trace(
                    wls=wls,
                    incidence_angle=incidence_angle,
                ).to(u.degree)

                # Compute the air dispersion relative to the zero-deviation wavelength
                delta_air: ArrayQuantity = (
                    self.beam_comp
                    * self.z_max
                    * (
                        self.air_model.refractive_index(wls)
                        - self.air_model.refractive_index(wl_0)
                    )
                ).si.to(u.degree)

                # Compute residuals
                residuals_vec: ArrayQuantity = (delta_ADC - delta_air) * u.radian
                # Convert back to on-sky equivalent angle

                # TODO: but this results in a unitless quantity!
                residuals_vec: NDArray[np.floating] = residuals_vec / (
                    self.z_max * self.beam_comp
                )

                if return_residuals:
                    return residuals_vec

                # Compute optimization criterion
                if self.criterion == "PTV":
                    return abs(max(residuals_vec) - min(residuals_vec)).value
                if self.criterion == "RMS":
                    return np.sqrt(np.mean(residuals_vec**2))
                if self.criterion == "RSS":
                    return np.sqrt(np.sum(residuals_vec**2))
                if self.criterion == "max_abs":
                    return max(np.abs(residuals_vec))

                return np.nan

            finally:
                # Restore original apex angles
                self.p1.apex_angle = old_angles[0]
                self.p2.apex_angle = old_angles[1]
                self.p3.apex_angle = old_angles[2]

        # Initial guess: use small positive and alternating negative angles,
        # zero deviation at mid wavelength, normal incidence
        x0: list[float] = [
            wl_med,
            10.0,  # alpha1
            -6.0,  # alpha2 (negative to provide opposite dispersion)
            3.0,  # alpha3 (positive to fine-tune)
            0.0,  # incidence angle
        ]

        bounds: list[tuple[float, float]] = [
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
        def residuals_function(wls: ArrayQuantity) -> float | Quantity:
            assert self.apex_angles is not None
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
            "p1": self.p1,
            "p2": self.p2,
            "p3": self.p3,
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

    @classmethod
    def optimise_materials_staged(
        cls,
        glasses: list[Glass],
        min_wl: ScalarQuantity = 390 * u.nm,
        max_wl: ScalarQuantity = 950 * u.nm,
        D_telescope: ScalarQuantity = 0.8 * u.m,
        D_ADC: ScalarQuantity = 15 * u.mm,
        z_max: ScalarQuantity = 65 * u.degree,
        air_model: Model | None = None,
        criterion: str = "PTV",
        top_n_pairs: int = 20,
        num_samples_pairs: int = 20,
        num_samples_adc3: int = 20,
        max_residuals: ScalarQuantity = 100 * u.mas,
        allow_repeated_glasses: bool = False,
    ) -> tuple[list[ADC3], list[dict[str, Any]]]:
        """
        Run a staged three-element ADC material search.

        Stage 1 optimizes all two-glass combinations and keeps the best
        ``top_n_pairs`` by the selected criterion.

        Stage 2 adds each possible third glass to the retained pairs and
        re-optimizes the full three-prism design.
        """

        from curved_air.instruments.ADC import ADC, PrismStack  # noqa: PLC0415
        from curved_air.optics.dispersers import Prism  # noqa: PLC0415

        if air_model is None:
            air_model = Air()

        if top_n_pairs < 1:
            raise ValueError("top_n_pairs must be >= 1")

        def adc2_metric(adc: ADC) -> float:
            if criterion == "RMS":
                return adc.residuals_RMS.to(u.mas).value
            if criterion == "RSS":
                return adc.residuals_RSS.to(u.mas).value
            if criterion == "max_abs":
                if adc.residuals_function is None:
                    return np.inf
                residuals = np.abs(adc.residuals_function(wls=adc.wls(100)))
                return float(np.max(residuals))

            return adc.residuals_PTV.to(u.mas).value

        def adc3_metric(adc: ADC3) -> float:
            if criterion == "RMS":
                return adc.residuals_RMS.to(u.mas).value
            if criterion == "RSS":
                return adc.residuals_RSS.to(u.mas).value
            if criterion == "max_abs":
                if adc.residuals_function is None:
                    return np.inf
                residuals = np.abs(adc.residuals_function(wls=adc.wls(100)))
                return float(np.max(residuals))

            return adc.residuals_PTV.to(u.mas).value

        pair_candidates: list[dict[str, Any]] = []
        n = len(glasses)

        for g1, g2 in tqdm(
            combinations(glasses, 2),
            desc="Stage 1/2: Optimising 2-glass ADCs",
            total=int((n * (n - 1)) / 2),
            unit="pair",
            ncols=100,
        ):
            adc2 = ADC(
                prism_stack=PrismStack([g1, g2]),
                min_wl=min_wl,
                max_wl=max_wl,
                D_telescope=D_telescope,
                D_ADC=D_ADC,
                z_max=z_max,
                air_model=air_model,
                criterion=criterion,
            )

            try:
                adc2.optimise(num_samples=num_samples_pairs)
            except (ValueError, RuntimeError, TypeError):
                continue

            pair_candidates.append(
                {
                    "g1": g1,
                    "g2": g2,
                    "pair_metric_mas": adc2_metric(adc2),
                }
            )

        if not pair_candidates:
            return [], []

        pair_candidates.sort(key=lambda x: x["pair_metric_mas"])
        top_pairs = pair_candidates[: min(top_n_pairs, len(pair_candidates))]

        solutions: list[ADC3] = []
        results: list[dict[str, Any]] = []

        num_third_options = len(glasses) if allow_repeated_glasses else max(n - 2, 0)

        with tqdm(
            total=len(top_pairs) * num_third_options,
            desc="Stage 2/2: Expanding to 3-glass ADCs",
            unit="adc3",
            ncols=100,
        ) as pbar:
            for rank, pair in enumerate(top_pairs, start=1):
                g1 = pair["g1"]
                g2 = pair["g2"]

                for g3 in glasses:
                    if not allow_repeated_glasses and (
                        g3.name == g1.name or g3.name == g2.name
                    ):
                        continue

                    prism_stack = PrismStack(
                        prism_list=[
                            Prism(glass=g1, apex_angle=10 * u.deg),
                            Prism(glass=g2, apex_angle=-6 * u.deg),
                            Prism(glass=g3, apex_angle=3 * u.deg),
                        ]
                    )

                    adc3 = cls(
                        prism_stack=prism_stack,
                        min_wl=min_wl,
                        max_wl=max_wl,
                        D_telescope=D_telescope,
                        D_ADC=D_ADC,
                        z_max=z_max,
                        air_model=air_model,
                        criterion=criterion,
                    )

                    try:
                        optimisation_result = adc3.optimise(
                            num_samples=num_samples_adc3
                        )
                    except (ValueError, RuntimeError, TypeError):
                        pbar.update(1)
                        continue

                    metric_mas = adc3_metric(adc3)
                    if metric_mas <= max_residuals.to(u.mas).value:
                        assert adc3.apex_angles is not None
                        assert adc3.incidence_angle is not None
                        assert adc3.wl_0 is not None

                        solutions.append(adc3)
                        results.append(
                            {
                                "Glass 1": g1.name,
                                "Glass 2": g2.name,
                                "Glass 3": g3.name,
                                "Pair rank": rank,
                                "Pair metric (mas)": pair["pair_metric_mas"],
                                "Apex angles (deg)": [
                                    adc3.apex_angles[0].to(u.deg).value,
                                    adc3.apex_angles[1].to(u.deg).value,
                                    adc3.apex_angles[2].to(u.deg).value,
                                ],
                                "Incidence angle (deg)": adc3.incidence_angle.to(
                                    u.deg
                                ).value,
                                "Zero-deviation wavelength (um)": adc3.wl_0.to(
                                    u.um
                                ).value,
                                "RMS residuals (mas)": adc3.residuals_RMS,
                                "RSS residuals (mas)": adc3.residuals_RSS,
                                "PTV residuals (mas)": adc3.residuals_PTV,
                                "Criterion metric (mas)": metric_mas,
                                "Optimization success": optimisation_result[
                                    "optimization_success"
                                ]
                                if optimisation_result is not None
                                else False,
                                "Optimization message": optimisation_result[
                                    "optimization_message"
                                ]
                                if optimisation_result is not None
                                else "",
                            }
                        )

                    pbar.update(1)

        results.sort(key=lambda x: x["Criterion metric (mas)"])

        return solutions, results

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
        ax: plt.Axes = axs[0]
        ax.plot(wls, self.g1.n(wls), label=f"{self.g1.name}", linewidth=2)
        ax.plot(wls, self.g2.n(wls), label=f"{self.g2.name}", linewidth=2)
        ax.plot(wls, self.g3.n(wls), label=f"{self.g3.name}", linewidth=2)
        ax.set_ylabel("Refractive index $n$")
        ax.legend(frameon=True)
        ax.grid(True, alpha=0.3)
        ax.set_xticklabels([])

        # Plot 2: Effective dispersion
        ax: plt.Axes = axs[1]

        # Update prism angles and trace through the stack
        old_angles = [
            self.p1.apex_angle.copy(),
            self.p2.apex_angle.copy(),
            self.p3.apex_angle.copy(),
        ]

        try:
            self.p1.apex_angle: ScalarQuantity = self.apex_angles[0]
            self.p2.apex_angle: ScalarQuantity = self.apex_angles[1]
            self.p3.apex_angle: ScalarQuantity = self.apex_angles[2]

            delta_ADC = self.prism_stack.trace(
                wls=wls,
                incidence_angle=self.incidence_angle
                if self.incidence_angle is not None
                else 0 * u.deg,
            ).to(u.degree)
        finally:
            self.p1.apex_angle: ScalarQuantity = old_angles[0]
            self.p2.apex_angle: ScalarQuantity = old_angles[1]
            self.p3.apex_angle: ScalarQuantity = old_angles[2]

        ax.plot(
            wls,
            delta_ADC,
            "k--",
            linewidth=2,
            label=f"ADC ($\\alpha_1$={self.apex_angles[0]:.2f}, $\\alpha_2$={self.apex_angles[1]:.2f}, $\\alpha_3$={self.apex_angles[2]:.2f})",
        )

        delta_air: ArrayQuantity = (
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
        ax.set_ylabel("Effective dispersion angle (degrees)")
        ax.legend(frameon=True)
        ax.grid(True, alpha=0.3)
        ax.set_xticklabels([])

        # Plot 3: Residuals
        ax: plt.Axes = axs[2]
        residuals_curve = self.residuals_function(wls=wls)
        ax.plot(
            wls,
            residuals_curve,
            "b-",
            linewidth=2,
            label="Residuals"
            + f"\nRMS: {self.residuals_RMS.to(u.mas):.2f}"
            + f"\nPTV: {self.residuals_PTV.to(u.mas):.2f}",
        )

        ax.axhline(y=0, color="k", ls="--", lw=1, alpha=0.5)
        ax.axvline(x=self.wl_0.to(u.um).value, color="k", ls="--", lw=1, alpha=0.5)
        ax.set_xlabel("Wavelength (microns)")
        ax.set_ylabel("Angular residuals (mas)")
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
