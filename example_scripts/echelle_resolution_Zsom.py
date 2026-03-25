# /// script
# dependencies = [
#     "curved_air>=0.1.0",
# ]
# [tool.uv.sources]
# curved_air = { path = "..", editable = true }
# ///

""" """

from typing import TypeAlias

import astropy.units as u
from astropy.units import Quantity

from curved_air.optics.dispersers import EchelleGrating

ScalarQuantity: TypeAlias = Quantity
ArrayQuantity: TypeAlias = Quantity


def main() -> None:

    grating = EchelleGrating(
        groove_frequency=31.6 / u.mm,
        blaze_angle=76 * u.deg,
        gamma_angle=0 * u.deg,
        min_wl=370 * u.nm,
        max_wl=950 * u.nm,
    )

    wavelengths: list[ScalarQuantity] = [
        649.8 * u.nm,
        649.8 * u.nm - 3.367 * u.nm,
        649.8 * u.nm - 3.373 * u.nm,
    ]

    for wl in wavelengths:
        m: int = grating.order_number(wl=wl, quantised=True)
        angle: ScalarQuantity = grating.diffraction_angle(wl=wl, m=m)
        if wl == 632.8 * u.nm:
            print("--> ", end="")
        else:
            print("    ", end="")
        print(f"{wl:.3f} at {angle:.5f} in order {m}")


if __name__ == "__main__":
    main()
