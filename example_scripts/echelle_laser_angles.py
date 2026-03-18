# /// script
# dependencies = [
#     "curved_air>=0.1.0",
# ]
# [tool.uv.sources]
# curved_air = { path = "..", editable = true }
# ///

"""
Calculate the diffraction angles for a selection of laser wavelengths on the 2ES echelle
grating. Used to determine the best wavelength for alignment of the echelle grating and
camera system.

Choosing a laser which is diffracted at a small angle, the grating can be placed in the
spectrograph and aligned such that the diffracted laser light is at the same angle as
if reflected from a flat mirror. This puts the grating at the correct gamma angle, with
the blaze angle to be adjusted at a later stage.
"""

from typing import TypeAlias

import astropy.units as u
from astropy.units import Quantity

from curved_air.optics.dispersers import EchelleGrating

ScalarQuantity: TypeAlias = Quantity
ArrayQuantity: TypeAlias = Quantity


def main() -> None:
    TwoES_grating = EchelleGrating(
        groove_frequency=31.6 / u.mm,
        blaze_angle=75.5 * u.deg,
        gamma_angle=0.55 * u.deg,
        min_wl=370 * u.nm,
        max_wl=950 * u.nm,
    )

    laser_wavelengths: list[ScalarQuantity] = [
        405 * u.nm,
        543 * u.nm,
        632.8 * u.nm,
        635 * u.nm,
        780 * u.nm,
    ]

    for wl in laser_wavelengths:
        m: int = TwoES_grating.order_number(wl=wl, quantised=True)
        angle: ScalarQuantity = TwoES_grating.diffraction_angle(wl=wl, m=m)
        if wl == 632.8 * u.nm:
            print("--> ", end="")
        else:
            print("    ", end="")
        print(f"{wl:.1f} at {angle:.2f} in order {m}")


if __name__ == "__main__":
    main()
