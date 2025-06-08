# /// script
# dependencies = [
#     "curved_air>=0.1.0",
# ]
# [tool.uv.sources]
# curved_air = { path = "..", editable = true }
# ///

from pathlib import Path

import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt

from curved_air.optics import Grating
from curved_air.plotting import plot_style, stroke, wavelength_to_rgb  # noqa: F401
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import ArrayLike
    from astropy.units import Quantity

plt.style.use(plot_style)

SCRIPT_PATH = Path(__file__).parent


def main() -> None:
    """ """

    MARVEL_grating: Grating = Grating(
        catalogue_number="53-*-417E",
        groove_frequency=52.67 / u.mm,
        blaze_angle=69.7 * u.deg,
        gamma_angle=0.3 * u.deg,
        min_wl=385 * u.nm,
        max_wl=950 * u.nm,
    )

    fig = plt.figure()
    ax = fig.gca()

    ms = MARVEL_grating.diffraction_orders

    for m in ms:
        if m % 10 != 0:
            continue

        blaze_wavelength: Quantity = MARVEL_grating.blaze_wavelength(m=m)
        wl1, wl2 = (
            MARVEL_grating.wavelength_at_angle(
                m=m, angle=MARVEL_grating.blaze_angle + 4 * u.deg
            ),
            MARVEL_grating.wavelength_at_angle(
                m=m, angle=MARVEL_grating.blaze_angle - 4 * u.deg
            ),
        )

        wavelengths = np.linspace(wl1.to(u.nm).value, wl2.to(u.nm).value, 200) * u.nm
        angles = [
            MARVEL_grating.diffraction_angle(m=m, wl=wl).to(u.deg).value
            for wl in wavelengths
        ] * u.deg

        efficiencies: ArrayLike = MARVEL_grating.relative_intensity(m=m, wl=wavelengths)

        ax.plot(
            angles.to(u.deg).value,
            efficiencies,
            color=wavelength_to_rgb(blaze_wavelength),
            path_effects=stroke(thickness=4),
        )

    ax.set_xlabel("")
    ax.set_ylabel("Relative intensity")

    ax.legend()

    plt.show()


if __name__ == "__main__":
    main()
