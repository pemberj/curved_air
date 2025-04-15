# /// script
# dependencies = [
#     "curved_air>=0.1.0",
# ]
# [tool.uv.sources]
# curved_air = { path = "..", editable = true }
# ///

from pathlib import Path

import numpy as np  # noqa: F401
from astropy import units as u
from astropy.units import Quantity
from matplotlib import pyplot as plt

from curved_air.plotting import plot_style

plt.style.use(plot_style)

SCRIPT_PATH = Path(__file__).parent


def main() -> None:
    field_angles = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * u.deg

    s1_incidence = [
        54.7,
        54.70617841727080,
        54.72470895880937,
        54.75557750944267,
        54.79876059591663,
        54.85422546450420,
        54.92193018877607,
        55.00182380666923,
        55.09384648576214,
        55.19792971545569,
        55.31399652456620,
    ] * u.deg

    s1_refraction = [
        30.03796869421788,
        30.04049805587760,
        30.04808228888068,
        30.06070984809417,
        30.07836152753366,
        30.10101051393918,
        30.12862246122656,
        30.16115558526819,
        30.19856077831588,
        30.24078174224563,
        30.28775513968292,
    ] * u.deg

    s2_incidence = [
        29.96203130578211,
        29.97087243216663,
        29.99737797543764,
        30.04149458476417,
        30.10313386151088,
        30.18217312691666,
        30.27845647439575,
        30.39179608502146,
        30.52197377999999,
        30.66874278011219,
        30.83182963928486,
    ] * u.deg

    s2_refraction = [
        54.51487220254448,
        54.53639004332054,
        54.60095676584825,
        54.70861283207348,
        54.85942856633275,
        55.05350854668020,
        55.29099787791974,
        55.57209048554004,
        55.89703963087082,
        56.26617092747520,
        56.67989824455353,
    ] * u.deg

    fig = plt.figure()
    ax = fig.gca()

    nominal_incidence = 54.7 * u.deg
    nominal_apex = 60 * u.deg

    effective_apex = np.array(s1_refraction.value) + np.array(s2_incidence.value)

    posneg_field_angles = list(-field_angles.value)[::-1] + list(field_angles.value)
    posneg_effective_apex = list(effective_apex)[::-1] + list(effective_apex)

    plt.axhline(y=nominal_apex.value, color="k", linestyle="--", alpha=0.5)

    # ax.plot(field_angles, s1_incidence, label="S1 incidence angle")
    ax.plot(
        posneg_field_angles,
        posneg_effective_apex,
        label="S1 refraction angle",
    )
    # ax.plot(field_angles, s2_refraction, label="S2 refraction angle")

    smoothx = np.linspace(-10, 10, 100) * u.deg

    def func(field_angles: Quantity) -> Quantity:
        return nominal_apex + (74 * (1 - np.cos(field_angles))) * u.deg

    ax.plot(smoothx, func(smoothx))

    plt.show()


if __name__ == "__main__":
    main()
