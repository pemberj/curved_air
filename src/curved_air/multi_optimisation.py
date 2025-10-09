"""
curved_air

multi_optimisation

This module contains functions for optimising the design of multi-element prisms using
a catalogue of glasses. This code is largely unfinished...
"""

import itertools

import astropy.units as u
import numpy as np
import scipy.optimize
from numpy.typing import ArrayLike
from typing import TYPE_CHECKING

from curved_air.glass import GlassCatalogue
from curved_air.optics.dispersers import Prism
from curved_air.instruments.ADC import PrismStack

if TYPE_CHECKING:
    from astropy.units import Quantity


def constant_dispersion(
    glass_catalogue: GlassCatalogue,
    wavelengths: np.ndarray | None = None,
    num_glasses: int = 3,
    max_apex_ratio: float = 3.0,  # TODO: use this!  # noqa: ARG001
    target_dispersion: float = 0.0,
    dispersion_range: float = 0.1,
) -> dict:
    """
    TODO: Move this funtion into the GlassCatalogue class as a method?
    """

    if len(glass_catalogue) >= 20 and num_glasses >= 3:  # noqa: PLR2004
        n: int = len(glass_catalogue)
        r: int = num_glasses

        num_combinations = (
            np.math.factorial(n + r - 1)
            / np.math.factorial(r)
            / np.math.factorial(n - 1)
        )

        print(
            "-" * 20
            + "\nWARNING!\n"
            + f"Optimising a {num_glasses}-element compound prism "
            + f"using a catalogue of {len(glass_catalogue)} materials.\n"
            + f"This will result in {round(num_combinations)} iterations.\n"
            + "You may wish to try with fewer materials in your catalogue."
        )

    if wavelengths is None:
        wavelengths: np.ndarray = np.linspace(0.39, 0.95, 20) * u.um  # default value

    combinations: itertools.combinations_with_replacement = (
        itertools.combinations_with_replacement(glass_catalogue.glass_list, num_glasses)
    )

    results: dict = {"glasses": [], "i": [], "A": [], "m": [], "b": [], "residuals": []}

    for combination in combinations:
        if np.any(np.array(combination[:-1]) == np.array(combination[1:])):
            # Skip any combinations with adjacent prisms of the same glass
            continue

        def residuals(
            iAmb: list[float] = [0] + [None] * num_glasses + [1, 0],
        ) -> ArrayLike:
            i, *A, m, b = iAmb
            i *= u.deg
            # i = iAmb[0] * u.deg
            # A = iAmb[1:-2]
            # m, b = iAmb[-2:]
            m *= u.rad / u.um  # Give 'm' units of radians per micron
            b *= u.rad  # Give 'b' units of radians

            # print(f"{i=} {A=} {m=} {b=}")

            stack: PrismStack = PrismStack(
                [
                    Prism(glass=glass_catalogue[g], apex_angle=a * u.deg)
                    for g, a in zip(combination, A, strict=False)  # noqa: B023
                ]
            )
            dispersion: Quantity = stack.trace(wavelengths, incidence_angle=i).to("deg")
            return np.sum(
                np.power((dispersion - (m * wavelengths + b)).to("deg").value, 2)
            )

        A_start: np.ndarray = np.ones(num_glasses) * 10
        A_start[1::2] *= -1.0
        A_start: list[float] = list(A_start)

        bounds: list[tuple[float, float]] = (
            [(-0.05, 0.05)]
            + [(-10.0, 10.0)] * num_glasses
            + [
                (
                    target_dispersion - dispersion_range,
                    target_dispersion + dispersion_range,
                ),
                (-0.05, 0.05),
            ]
        )

        #     i      A     m  b
        x0: list[float] = [0, *A_start, 1, 0]

        res = scipy.optimize.minimize(
            fun=residuals,
            x0=x0,
            bounds=bounds,
            # options = {"maxiter": 25},
        )

        if not res.success:
            print(f"{combination}\n{res.message}")
            continue

        i: Quantity = res.x[0] * u.deg
        A: list[float] = res.x[1:-2]
        m: Quantity = res.x[-2] * u.rad / u.um
        b: Quantity = res.x[-1] * u.rad

        results["glasses"].append(combination)
        results["i"].append(i)
        results["A"].append([round(a, 4) for a in A])
        results["m"].append(m)
        results["b"].append(b)
        results["residuals"].append(residuals(iAmb=res.x))

    return results


def main() -> None: ...


if __name__ == "__main__":
    main()
