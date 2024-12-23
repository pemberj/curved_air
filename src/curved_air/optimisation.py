"""
Several functions to find optimal solutions for compound prisms, achromatic
lenses, etc.
"""

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import curve_fit
import scipy.optimize
import astropy.units as u
from astropy.units import Quantity, UnitConversionError
from collections.abc import Callable

from curved_air.glass import Glass, GlassCatalogue
from curved_air.optics import Prism, PrismStack


def find_overlapping_refrective_index(
    glass_catalogue: GlassCatalogue = None,
    target_wavelength: Quantity = 0.55 * u.um,
    minwl: Quantity = 0.39 * u.um,
    maxwl: Quantity = 0.95 * u.um,
    # fractional_margin: float = 0.1,
) -> dict[str, list]:
    """
    A function to iterate through glass combinations to find pairings that have
    the same refractive index at a particular wavelength)
    """

    results = {"g1": [], "g2": [], "wls": [], "delta_wl": []}

    for _, glass1 in glass_catalogue.items():
        for _, glass2 in glass_catalogue.items():
            if glass2.is_the_same_as(glass1):
                continue

            # skip_flag = False
            # if glass2.name in results["g1"]:
            #     continue
            # if glass1.name in glass2.name or glass2.name in glass1.name:
            #     continue
            # for name1 in glass1.alternative_names:
            #     if name1 in glass2.alternative_names:
            #         skip_flag = True
            # if skip_flag: continue

            # minwl = target_wavelength - fractional_margin * target_wavelength
            # maxwl = target_wavelength + fractional_margin * target_wavelength

            test_wavelengths = np.linspace(minwl, maxwl, 100)

            n1 = glass1.n(test_wavelengths)
            n2 = glass2.n(test_wavelengths)

            diff = n1 - n2

            zwls = test_wavelengths[np.where(diff[:-1] / diff[1:] < 0)]
            # This line looks at where the 'diff' array changes sign, and
            # returns the corresponding wavelengths - where the two refracive
            # indices are cross over

            if len(zwls) == 0:
                continue

            for zwl in zwls:
                delta_wl = abs(zwl - target_wavelength)
                # if (delta_wl >= (1 - fractional_margin) * target_wavelength) or \
                #    (delta_wl <= (1 + fractional_margin) * target_wavelength):

            results["g1"].append(glass1.name)
            results["g2"].append(glass2.name)
            results["wls"].append(zwls)
            results["delta_wl"].append(delta_wl)

    return results


def anomalous_dispersion(
    glass_catalogue: GlassCatalogue = None,
    wavelength_range: ArrayLike = None,
    num_glasses: int = 2,
    max_apex_ratio: float = 1.05,
) -> dict[str, list]:
    """ """

    if wavelength_range is None:
        wavelength_range = np.linspace(0.39, 0.95, 20)  # default value

    results = {
        "g1": [],
        "g2": [],
        "apex_ratio": [],
        "where_anomalous": [],
        "sum_anomalous": [],
    }

    for _, glass1 in glass_catalogue.items():
        for _, glass2 in glass_catalogue.items():
            skip_flag = False
            if glass2.name in results["g1"]:
                continue
            if glass1.name in glass2.name or glass2.name in glass1.name:
                continue
            for name1 in glass1.alternative_names:
                if name1 in glass2.alternative_names:
                    skip_flag = True
            if skip_flag:
                continue

            try:
                n1 = glass1.n(wavelength_range)
                n2 = glass2.n(wavelength_range)
            except Exception as e:
                print(e)
                return None

            if num_glasses == 2:  # noqa: PLR2004

                def residuals(k: float = 1.0) -> float:
                    """
                    Here the definition of what merits a good system is defined

                    TODO: Define a number of potentially useful merit functions
                    so that this is the only part that needs to change between
                    optimising systems with different goals.

                    eg.
                    - constant dispersion
                    - linear dispersion
                    - targeting anomalous dispersion
                    - Matching a particular dispersion curve
                    """
                    # effective_n = (n1 - k * n2)
                    # effective_n = effective_n / (effective_n[0] / abs(effective_n[0]))
                    # dispersion = np.diff(effective_n)
                    # where_anomalous = dispersion[np.where(dispersion > 0)]
                    # return np.sum(where_anomalous)

                    p1 = Prism(glass=glass1, apex_angle=10 * u.deg)  # noqa: B023
                    p2 = Prism(glass=glass2, apex_angle=-k * 10 * u.deg)  # noqa: B023

                    compound_prism = PrismStack(prism_list=[p1, p2])

                    effective_n = compound_prism.trace(wavelength_range * u.um).value

                    effective_n = effective_n / (effective_n[0] / abs(effective_n[0]))
                    dispersion = np.diff(effective_n)
                    where_anomalous = dispersion[np.where(dispersion > 0)]
                    return np.sum(where_anomalous)

            elif num_glasses >= 3:  # noqa: PLR2004
                raise NotImplementedError

            res = scipy.optimize.minimize(
                fun=residuals,
                x0=1.0,
                bounds=[(1.0 / max_apex_ratio, max_apex_ratio)],
                options={"maxiter": 50},
            )

            if not res.success:
                continue

            k = res.x

            effective_n = n1 - k * n2
            effective_n = effective_n / (effective_n[0] / abs(effective_n[0]))
            dispersion = np.diff(effective_n)
            where_anomalous = wavelength_range[np.where(dispersion > 0)]

            if len(where_anomalous) >= 1:
                results["g1"].append(glass1.name)
                results["g2"].append(glass2.name)
                results["apex_ratio"].append(k)
                results["where_anomalous"].append(where_anomalous)
                results["sum_anomalous"].append(
                    np.sum(np.abs(dispersion[np.where(dispersion < 0)]))
                )

    return results


def constant_dispersion(
    glass_catalogue: GlassCatalogue = None,
    wavelength_range: ArrayLike = None,
    num_glasses: int = 2,
    max_apex_ratio: float = 2.0,
    target_dispersion: float = 0.0,
    dispersion_range: float = 0.1,
) -> dict[str, list]:
    """
    NOTE: target_dispersion (slope of the linear effective refractive index curve)
    is in units of *radians per wavelength step*

    TODO: Add the possibility of any number of glasses:
        - When looping through the glasses, create a PrismStack, populate it with
        each combination of glasses, then trace the rays
        - Will likely need to use very small input catalogues if using more than
        two glasses in a stack, the number of possible combinations grows very
        quickly!

    A simple first-pass optimisation routine for finding two glasses for which a
    linear combination of refractive indicies can produce a linear effective
    refractive index over a given wavelength range.

    This analysis can be the starting point in designing low-dispersion compound
    prisms for eg. slitless wide-field spectroscopy or wavelength-resolved
    exposure meters
    """

    if wavelength_range is None:
        wavelength_range = np.linspace(0.39, 0.95, 20)  # default value

    # def residuals(wls: ArrayLike = wavelength_range,
    #               glass1: Glass = None,
    #               glass2: Glass = None,
    #               k: float = 1.25,
    #               m: float = 1.,
    #               b: float = 0.) -> ArrayLike:

    #     return (glass1.n(wls) - k * glass2.n(wls)) - (m * wls + b)

    results = {"g1": [], "g2": [], "k": [], "m": [], "b": [], "residuals": []}

    for _, glass1 in glass_catalogue.items():
        for _, glass2 in glass_catalogue.items():
            skip_flag = False
            if glass2.name in results["g1"]:
                continue
            if glass1.name in glass2.name or glass2.name in glass1.name:
                continue
            for name1 in glass1.alternative_names:
                if name1 in glass2.alternative_names:
                    skip_flag = True
            if skip_flag:
                continue

            try:
                n1 = glass1.n(wavelength_range)
                n2 = glass2.n(wavelength_range)
            except Exception as e:
                print(e)
                return None

            if num_glasses == 2:  # noqa: PLR2004

                def residuals(
                    kmbi: ArrayLike = [1.0, target_dispersion, 0.0, 0.0],
                ) -> float:
                    k, m, b, i = kmbi
                    return np.sum(
                        np.power((n1 - k * n2) - (m * wavelength_range + b), 2)  # noqa: B023
                    )

            # For more than two glasses, create a PrismStack object, and use the
            # .trace() method from this to trace the rays (angle only)
            # TODO: Maybe this needs some more inputs? Incidence angle on the
            # first surface, for instance!
            # TODO: ACTUALLY, need to re-think the full constant_dispersion function
            elif num_glasses >= 3:  # noqa: PLR2004
                raise NotImplementedError

            res = scipy.optimize.minimize(
                fun=residuals,
                x0=[1.0, 5e-4, 0.0, 0],
                bounds=[
                    (1.0 / max_apex_ratio, max_apex_ratio),
                    (
                        target_dispersion - dispersion_range,
                        target_dispersion + dispersion_range,
                    ),
                    (-np.inf, np.inf),
                    (0.0, 0.0),
                ],
                options={"maxiter": 20},
            )

            if not res.success:
                continue

            k, m, b, i = res.x

            results["g1"].append(glass1.name)
            results["g2"].append(glass2.name)
            results["k"].append(k)
            results["m"].append(m)
            results["b"].append(b)
            results["residuals"].append(residuals(kmbi=[k, m, b, i]))

    return results


def match_target_refractive_index(
    g1: Glass = None,
    g2: Glass = None,
    min_wl: Quantity = 385 * u.nm,
    max_wl: Quantity = 950 * u.nm,
    num_samples: int = 20,
    target_refractive_index: Callable | None = None,
) -> PrismStack:
    min_wl = min_wl.to(u.um).value
    max_wl = max_wl.to(u.um).value

    try:
        assert min_wl >= g1.wlmin
        assert min_wl >= g2.wlmin
        assert max_wl <= g1.wlmax
        assert max_wl <= g2.wlmax
    except AssertionError as e:
        print(e)
    except UnitConversionError as e:
        print(e)

    wls = np.linspace(min_wl, max_wl, num_samples)

    def residuals(
        wls: list[float],  # Wavelengths in nanometers!
        alpha_deg: float = 10,
        lam_0_nm: float = 632.8,
    ) -> ArrayLike:
        wls = wls * u.um

        alpha = alpha_deg * u.deg
        lam_0 = lam_0_nm * u.nm

        # Setting epsilon here ensures that the compound prism will have zero
        # deviation at the wavelength lam_0
        epsilon = -np.arcsin((g1.n(lam_0) - g2.n(lam_0)) * np.sin(alpha)) / (
            g2.n(lam_0) - 1
        )

        deviation = (
            (
                np.arcsin((g1.n(wls) - g2.n(wls)) * np.sin(alpha))
                + (g2.n(wls) - 1) * epsilon
            )
            .to(u.radian)
            .value
        )

        target_deviation = target_refractive_index(wls) - target_refractive_index(lam_0)

        residuals = deviation - target_deviation

        return residuals  # noqa: RET504

    mid_wl = ((min_wl + max_wl) / 2) * 1000
    min_wl = min_wl * 1000
    max_wl = max_wl * 1000

    try:
        [alpha_fit, lam_0_fit], cov = curve_fit(
            f=residuals,
            xdata=wls,
            ydata=[0.0] * len(wls),
            p0=[1.0, mid_wl],
            bounds=[
                [-np.inf, mid_wl - 2 * abs(mid_wl - min_wl)],
                [np.inf, mid_wl + 2 * abs(mid_wl - max_wl)],
            ],
        )
    except Exception as e:
        print(e)

    # print(alpha_fit, lam_0_fit)

    prism_1_apex = np.degrees(alpha_fit)
    epsilon_fit = (g1.n(lam_0_fit) - 1) / (g2.n(lam_0_fit) - 1)
    prism_2_apex = epsilon_fit * prism_1_apex

    return PrismStack(
        prism_list=[
            Prism(glass=g1, apex_angle=prism_1_apex * u.deg),
            Prism(glass=g2, apex_angle=prism_2_apex * u.deg),
        ]
    )


def main() -> None: ...


if __name__ == "__main__":
    main()
