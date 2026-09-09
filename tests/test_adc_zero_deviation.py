from typing import TYPE_CHECKING

import numpy as np
from astropy import units as u

from curved_air.glass import GlassCatalogue
from curved_air.instruments.ADC import PrismStack
from curved_air.optics.dispersers import Prism

if TYPE_CHECKING:
    from curved_air.type_hinting import ScalarQuantity


def test_two_prism_zero_deviation_angle_solver() -> None:
    gc = GlassCatalogue(["Schott", "Ohara"])
    p1 = Prism(glass=gc["N-BK7"], apex_angle=10 * u.deg)
    p2 = Prism(glass=gc["F2"], apex_angle=0 * u.deg)
    wl_0 = 0.5893 * u.um

    stack = PrismStack(prism_list=[p1, p2])
    last_angle: ScalarQuantity = stack.compute_last_apex_angle(wl_0=wl_0)

    weighted_sum: ScalarQuantity = (
        p1.glass.refractive_index(wl_0) - 1
    ) * p1.apex_angle + (p2.glass.refractive_index(wl_0) - 1) * last_angle

    assert np.isclose(weighted_sum.to(u.deg).value, 0.0, atol=1e-12)

    print("Two prism zero deviation angle solver test passed.")


def test_three_prism_zero_deviation_angle_solver() -> None:
    gc = GlassCatalogue(["Schott", "Ohara"])
    p1 = Prism(glass=gc["N-BK7"], apex_angle=8 * u.deg)
    p2 = Prism(glass=gc["F2"], apex_angle=-4 * u.deg)
    p3 = Prism(glass=gc["K10"], apex_angle=0 * u.deg)
    wl_0: ScalarQuantity = 0.4861 * u.um

    stack = PrismStack(prism_list=[p1, p2, p3])
    last_angle: ScalarQuantity = stack.compute_last_apex_angle(wl_0=wl_0)

    weighted_sum: ScalarQuantity = (
        sum(
            (prism.glass.refractive_index(wl_0) - 1) * angle
            for prism, angle in zip(
                stack.prism_list[:-1], [p1.apex_angle, p2.apex_angle], strict=True
            )
        )
        + (p3.glass.refractive_index(wl_0) - 1) * last_angle
    )

    assert np.isclose(weighted_sum.to(u.deg).value, 0.0, atol=1e-12)

    print("Three prism zero deviation angle solver test passed.")


if __name__ == "__main__":
    test_two_prism_zero_deviation_angle_solver()
    test_three_prism_zero_deviation_angle_solver()
