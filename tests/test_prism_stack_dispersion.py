from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from astropy import units as u

from curved_air.glass import GlassCatalogue
from curved_air.instruments.ADC import PrismStack
from curved_air.optics.dispersers import Prism

if TYPE_CHECKING:
    from curved_air.type_hinting import ScalarQuantity


def test_two_prism_null_dispersion() -> None:
    gc = GlassCatalogue("Schott")
    p1 = Prism(glass=gc["N-BK7"], apex_angle=10 * u.deg)
    p2 = Prism(glass=gc["N-BK7"], apex_angle=-10 * u.deg)

    stack = PrismStack(prism_list=[p1, p2])

    wl: ScalarQuantity = 0.5 * u.um

    angle: ScalarQuantity = stack.trace(
        wls=wl,
        incidence_angle=0 * u.deg,
        # debug_level=1,
    )

    target: ScalarQuantity = 0 * u.deg

    assert np.isclose(angle.to(u.deg).value, target.to(u.deg).value, atol=1e-12), (
        f"{angle=:.5g} =/= {target=:.5g}"
    )

    print("Two-prism null dispersion test passed.")


def test_two_prism_dispersion() -> None:
    gc = GlassCatalogue(["Schott", "Ohara"])
    p1 = Prism(glass=gc["N-BK7"], apex_angle=10 * u.deg)
    p2 = Prism(glass=gc["F2"], apex_angle=-10 * u.deg)

    stack = PrismStack(prism_list=[p1, p2])

    wl: ScalarQuantity = 0.55 * u.um
    angle: ScalarQuantity = abs(
        stack.trace(
            wls=wl,
            incidence_angle=0 * u.deg,
            # debug_level=1,
        )
    )

    target: ScalarQuantity = 1.06113 * u.deg  # From Zemax

    assert np.isclose(angle.to(u.deg).value, target.to(u.deg).value, atol=3e-4), (
        f"{angle=:.5g} =/= {target=:.5g}"
    )

    print("Two prism dispersion test passed.")


def test_three_prism_dispersion() -> None:
    gc = GlassCatalogue(["Schott", "Ohara"])
    p1 = Prism(glass=gc["N-BK7"], apex_angle=10 * u.deg)
    p2 = Prism(glass=gc["F2"], apex_angle=-20 * u.deg)
    p3 = Prism(glass=gc["K10"], apex_angle=10 * u.deg)

    stack = PrismStack(prism_list=[p1, p2, p3])

    wl: ScalarQuantity = 0.55 * u.um

    angle: ScalarQuantity = stack.trace(
        wls=wl,
        incidence_angle=0 * u.deg,
        # debug_level=1,
    )

    target: ScalarQuantity = 2.28226 * u.deg  # From Zemax

    assert np.isclose(angle.to(u.deg).value, target.to(u.deg).value, atol=7e-4), (
        f"{angle=:.5g} =/= {target=:.5g}"
    )

    print("Three prism dispersion test passed.")


if __name__ == "__main__":
    test_two_prism_null_dispersion()
    test_two_prism_dispersion()
    test_three_prism_dispersion()
