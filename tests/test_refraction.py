from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from astropy import units as u
from rich import print  # noqa: A004

from curved_air.glass import Glass, GlassCatalogue
from curved_air.optics.dispersers import Prism, _refract
from curved_air.other import Air

if TYPE_CHECKING:
    from curved_air.type_hinting import ScalarQuantity


def test_refractive_index() -> None:
    gc = GlassCatalogue("Schott")
    glass: Glass = gc["N-BK7"]

    wl: ScalarQuantity = 0.55 * u.um

    target: float = 1.5168

    assert np.isclose(glass.refractive_index(wl), target, atol=2e-3), (
        f"{glass.refractive_index(wl)=:.5g}, {target=:.5g}"
    )

    (print("Refractive index test passed."),)


def test_refraction() -> None:
    gc = GlassCatalogue("Schott")

    wl: ScalarQuantity = 0.55 * u.um

    air = Air(air_model="zemax")

    angle: ScalarQuantity = _refract(
        n_1=air.refractive_index(wl),
        n_2=gc["N-BK7"].refractive_index(wl),
        theta_in=10 * u.deg,
        wavelength=wl,
    )

    target: ScalarQuantity = 6.56627 * u.deg

    assert np.isclose(angle.to(u.deg).value, target.to(u.deg).value, atol=2e-3), (
        f"{angle=:.5g} =/= {target=:.5g}"
    )

    print("Refraction test passed.")


def test_prism() -> None:
    gc = GlassCatalogue("Schott")

    wl: ScalarQuantity = 0.55 * u.um

    prism = Prism(glass=gc["F2"], apex_angle=60 * u.deg)

    incidence_angle: ScalarQuantity = prism.minimum_deviation_incidence_angle(wl=wl)

    angle: ScalarQuantity = _refract(
        n_1=Air(air_model="zemax").refractive_index(wl),
        n_2=gc["F2"].refractive_index(wl),
        theta_in=incidence_angle,
        wavelength=wl,
    )

    target: ScalarQuantity = 30 * u.deg

    assert np.isclose(angle.to(u.deg).value, target.to(u.deg).value, atol=1e-2), (
        f"{angle=:.5g} =/= {target=:.5g}"
    )

    angle: ScalarQuantity = prism.trace(
        wls=wl,
        incidence_angle=incidence_angle,
    )

    target: ScalarQuantity = -1 * incidence_angle

    assert np.isclose(angle.to(u.deg).value, target.to(u.deg).value, atol=5e-2), (
        f"{angle=:.5g} =/= {target=:.5g}"
    )

    print("Prism test passed.")


if __name__ == "__main__":
    test_refractive_index()
    test_refraction()
    test_prism()
