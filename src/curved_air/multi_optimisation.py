"""
TODO:

Move this funtion into the GlassCatalogue class as a method.
"""

import numpy as np
from numpy.typing import ArrayLike
import itertools
import scipy.optimize
import astropy.units as u

from curved_air.glass import GlassCatalogue
from curved_air.optics import Prism, PrismStack


def constant_dispersion(glass_catalogue: GlassCatalogue = None,
                        wavelength_range: ArrayLike = None,
                        num_glasses: int = 3,
                        max_apex_ratio: float = 3.,
                        target_dispersion: float = 0.,
                        dispersion_range: float = 0.1):
    """
    """

    if len(glass_catalogue) >= 20 and num_glasses >= 3:
        n = len(glass_catalogue)
        r = num_glasses

        num_combinations = np.math.factorial(n + r - 1) / np.math.factorial(r)\
                                                        / np.math.factorial(n-1)

        print("-"*20+f"\nWARNING!\n"\
            +f"Optimising a {num_glasses}-element compound prism "\
            +f"using a catalogue of {len(glass_catalogue)} materials.\n"\
            +f"This will result in {round(num_combinations)} iterations.\n"\
            +"You may wish to try with fewer materials in your catalogue.")


    if wavelength_range is None:
        wavelength_range = np.linspace(0.39, 0.95, 20) * u.um # default value

    combinations = itertools.combinations_with_replacement(
                                        glass_catalogue.glass_list, num_glasses)

    results = {"glasses": [], "i": [], "A": [], "m": [], "b": [], "residuals": []}

    for combination in combinations:

        if np.any(np.array(combination[:-1]) == np.array(combination[1:])):
            # Skip any combinations with adjacent prisms of the same glass
            continue

        def residuals(iAmb = [0.] + [None]*num_glasses + [1., 0.]) -> ArrayLike:
            i = iAmb[0] * u.deg
            A = iAmb[1:-2]
            m, b = iAmb[-2:]
            m *= u.rad / u.um # Give 'm' units of radians per micron
            b *= u.rad # Give 'b' units of radians

            # print(f"{i=} {A=} {m=} {b=}")

            stack = PrismStack([Prism(glass=glass_catalogue[g], apex_angle=a*u.deg)\
                                        for g, a in zip(combination, A)])
            dispersion = stack.trace(wavelength_range, incidence_angle=i).to("deg")
            return np.sum(np.power((dispersion - (m * wavelength_range + b)).to("deg").value, 2))

        A_start = np.ones(num_glasses) * 10
        A_start[1::2] *= -1.
        A_start = list(A_start)

        bounds = [(-0.05, 0.05)]\
               + [(-10., 10.)] * num_glasses\
               + [(target_dispersion - dispersion_range,
                   target_dispersion + dispersion_range),
                  (-0.05, 0.05),]

        #      i        A       m   b
        x0 = [[0.] + A_start + [1., 0.]]

        res = scipy.optimize.minimize(fun=residuals, x0=x0, bounds=bounds,
                            # options={"maxiter": 25},
                            )

        if not res.success:
            print(f"{combination}\n{res.message}")
            continue

        i = res.x[0] * u.deg
        A = res.x[1:-2]
        m, b = res.x[-2:]
        m *= u.rad / u.um # Give 'm' units of radians per micron
        b *= u.rad # Give 'b' units of radians

        results["glasses"].append(combination)
        results["i"].append(i)
        results["A"].append([round(a, 4) for a in A])
        results["m"].append(m)
        results["b"].append(b)
        results["residuals"].append(residuals(iAmb=res.x))

    return results




def main() -> None:
    
    ...


if __name__ == "__main__":
    
    main()