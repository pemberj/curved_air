# /// script
# dependencies = [
#     "curved_air>=0.1.0",
# ]
# [tool.uv.sources]
# curved_air = { path = "..", editable = true }
# ///

"""
This script can be used to calculate various things about echelle spectrographs
"""

import astropy.units as u
import numpy as np
from astropy.units import Quantity

from curved_air.optics.dispersers import EchelleGrating


def MCE_fill_detector(
    grating: EchelleGrating,
    opening_angle: Quantity = 7.46 * u.deg,
    num_wavelengths: int = 5,
) -> None:
    """
    Generate a table of wavelengths for different echelle orders that can be entered
    into the Zemax Mutli-Configuration Editor to fill the detector of an echelle
    spectrograph in dispersion direction.

    TODO: Directly generate a Zemax .mce file? What is the format?
    """

    angles: np.ndarray = (
        np.linspace(-opening_angle / 2, opening_angle / 2, num_wavelengths)
        + grating.blaze_angle
    )

    ms: list[int] = []
    wls: list[list[float]] = [[] for _ in range(len(angles))]

    for m in grating.diffraction_orders[::]:
        ms.append(m)

        for i, angle in enumerate(angles):
            wl_at_angle: Quantity = grating.wavelength_at_angle(m=m, angle=angle)
            wls[i].append(wl_at_angle.to(u.um).value)

    skip: int = -len(ms) // 12

    print("\t".join([f"{m}" for m in ms[::skip]]))
    for iwls in wls:
        print("\t".join([f"{wl:.5f}" for wl in iwls[::skip]]))


def main() -> None:
    MARVEL_grating = EchelleGrating(
        groove_frequency=52.67 / u.mm,
        blaze_angle=69.7 * u.deg,
        gamma_angle=0.35 * u.deg,
        min_wl=380 * u.nm,
        max_wl=950 * u.nm,
    )

    MCE_fill_detector(MARVEL_grating, opening_angle=7.46 * u.deg, num_wavelengths=5)


if __name__ == "__main__":
    main()
