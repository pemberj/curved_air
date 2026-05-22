"""
Convert between optical fringes and radius-of-curvature tolerance, via calculation of
the nominal, flatter, and steeper sagitta.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from astropy import units as u
from rich import print as rprint

from curved_air.utils import SurfaceTolerance

if TYPE_CHECKING:
    from curved_air.type_hinting import ScalarQuantity


def main() -> None:

    R1: ScalarQuantity = 673.923 * u.mm
    R2: ScalarQuantity = 2173.047 * u.mm
    D: ScalarQuantity = 194 * u.mm

    t1f = SurfaceTolerance(
        name="S1 (fringes specified)", nominal_radius=R1, diameter=D, fringes=18
    )

    t2f = SurfaceTolerance(
        name="S2 (fringes specified)", nominal_radius=R2, diameter=D, fringes=5
    )

    t1t = SurfaceTolerance(
        name="S1 (tolerances specified)",
        nominal_radius=R1,
        diameter=D,
        positive_tolerance=0.55 * u.mm,
    )

    t2t = SurfaceTolerance(
        name="S2 (tolerances specified)",
        nominal_radius=R2,
        diameter=D,
        positive_tolerance=1.58 * u.mm,
    )

    rprint(f"{t1f}")
    rprint(f"{t2f}")
    rprint(f"{t1t}")
    rprint(f"{t2t}")


if __name__ == "__main__":
    main()
