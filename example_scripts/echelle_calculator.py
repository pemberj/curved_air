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

from typing import TypeAlias

import astropy.units as u
from astropy.units import Quantity

from curved_air.optics.dispersers import EchelleGrating

ScalarQuantity: TypeAlias = Quantity
ArrayQuantity: TypeAlias = Quantity


def main() -> None:
    # MARVEL_grating = EchelleGrating(
    #     groove_frequency=52.67 / u.mm,
    #     blaze_angle=69.7 * u.deg,
    #     gamma_angle=0.35 * u.deg,
    #     min_wl=390 * u.nm,
    #     max_wl=850 * u.nm,
    # )

    # print(MARVEL_grating.order_number(wl=632.8 * u.nm, quantised=False))

    TwoES_grating = EchelleGrating(
        groove_frequency=31.6 / u.mm,
        blaze_angle=75.5 * u.deg,
        gamma_angle=0.55 * u.deg,
        min_wl=370 * u.nm,
        max_wl=950 * u.nm,
    )

    opening_angle: ScalarQuantity = 4.5 * u.deg
    max_angle: ScalarQuantity = TwoES_grating.blaze_angle + opening_angle / 2
    min_angle: ScalarQuantity = TwoES_grating.blaze_angle - opening_angle / 2
    pickoff_angle: ScalarQuantity = 5 * u.deg

    for m in TwoES_grating.diffraction_orders[::]:
        min_bound: ScalarQuantity = TwoES_grating.wavelength_at_angle(
            m=m, angle=min_angle
        )
        max_bound: ScalarQuantity = TwoES_grating.wavelength_at_angle(
            m=m, angle=max_angle
        )

        short_pickoff: ScalarQuantity = TwoES_grating.wavelength_at_angle(
            m, angle=TwoES_grating.blaze_angle - pickoff_angle
        )
        long_pickoff: ScalarQuantity = TwoES_grating.wavelength_at_angle(
            m, angle=TwoES_grating.blaze_angle + pickoff_angle
        )

        print("-" * 40)
        print(f"Order {m}")
        print(f"Bounds: {min_bound:.5f} - {max_bound:.5f}")
        print(f"Short pickoff: {short_pickoff:.5f}")
        print(f"Long pickoff: {long_pickoff:.5f}")


if __name__ == "__main__":
    main()
