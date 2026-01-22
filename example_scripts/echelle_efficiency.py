# /// script
# dependencies = [
#     "curved_air>=0.1.0",
# ]
# [tool.uv.sources]
# curved_air = { path = "..", editable = true }
# ///

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt

from curved_air.optics.dispersers import EchelleGrating
from curved_air.plotting import plot_style, stroke, wavelength_to_rgb  # noqa: F401

if TYPE_CHECKING:
    from astropy.units import Quantity
    from numpy.typing import ArrayLike

    from curved_air.type_hinting import ArrayQuantity

plt.style.use(plot_style)

SCRIPT_PATH = Path(__file__).parent


def main() -> None:
    """ """

    MARVEL_grating: EchelleGrating = EchelleGrating(
        catalogue_number="53-*-417E",
        groove_frequency=52.67 / u.mm,
        blaze_angle=69.7 * u.deg,
        gamma_angle=0.3 * u.deg,
        min_wl=385 * u.nm,
        max_wl=950 * u.nm,
    )

    fig: plt.Figure = plt.figure()
    ax: plt.Axes = fig.gca()

    ms: list[int] = MARVEL_grating.diffraction_orders

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

        wavelengths: ArrayQuantity = (
            np.linspace(wl1.to(u.nm).value, wl2.to(u.nm).value, 200) * u.nm
        )
        angles: ArrayQuantity = [
            MARVEL_grating.diffraction_angle(m=m, wl=wl).to(u.deg).value
            for wl in wavelengths
        ] * u.deg

        efficiencies: ArrayLike = MARVEL_grating.relative_intensity(wl=wavelengths)

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
