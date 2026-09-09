"""
This module provides functions to estimate the on-sky density of stars as a function of
magnitude, photometric band, and Galactic coordinates. It also provides functions to
compute the required field of view to achieve a desired number of stars, and vice versa.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from astropy import units as u

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from curved_air.type_hinting import ScalarQuantity

Band = Literal["G", "V"]
SkyRegion = Literal[
    "All sky average",
    "Galactic plane |b| < 10°",
    "Low latitude 10° ≤ |b| < 20°",
    "Mid latitude 20° ≤ |b| < 40°",
    "High latitude 40° ≤ |b| < 60°",
    "Polar |b| ≥ 60°",
]


@dataclass(frozen=True)
class SkyRegionModel:
    label: SkyRegion
    density_scale: float
    equivalent_coverage: float | None = None


SKY_REGION_MODELS: tuple[SkyRegionModel, ...] = (
    SkyRegionModel("All sky average", 1.00, None),
    SkyRegionModel("Galactic plane |b| < 10°", 2.20, 0.50),
    SkyRegionModel("Low latitude 10° ≤ |b| < 20°", 1.55, 0.68),
    SkyRegionModel("Mid latitude 20° ≤ |b| < 40°", 1.00, 0.95),
    SkyRegionModel("High latitude 40° ≤ |b| < 60°", 0.62, 0.99),
    SkyRegionModel("Polar |b| ≥ 60°", 0.42, 0.99),
)


def base_cumulative_density(magnitude: float, band: Band = "G") -> float:
    """
    Approximate cumulative stellar surface density brighter than `magnitude`.

    Returns stars per square degree. The model is intentionally simple for webapp use:
    a smooth broken exponential in magnitude calibrated to reasonable guide-star counts
    across bright-to-moderately-faint limits. `V` is treated as slightly sparser than `G`.
    """

    mag = float(magnitude)
    if mag < -2:  # noqa: PLR2004
        raise ValueError("Magnitude is outside the supported range.")

    if mag <= 6:  # noqa: PLR2004
        log10_density: float = -3.30 + 0.60 * mag
    elif mag <= 12:  # noqa: PLR2004
        log10_density: float = 0.30 + 0.34 * (mag - 6)
    elif mag <= 18:  # noqa: PLR2004
        log10_density: float = 2.34 + 0.24 * (mag - 12)
    else:
        raise ValueError("Magnitude is outside the supported range -2 to 18.")

    density: float = 10**log10_density
    if band == "V":
        density *= 0.82
    elif band != "G":
        raise ValueError(f"Unsupported band: {band}")

    return density


def limiting_magnitude_for_area(
    area: ScalarQuantity,
    band: Band = "G",
    region: SkyRegionModel = SKY_REGION_MODELS[0],
    magnitude_min: float = -2.0,
    magnitude_max: float = 18.0,
) -> float:
    target_density: float = (1 / area.to(u.deg**2)).value
    scale: float = region.density_scale
    base_target_density = target_density / scale

    if band == "V":
        base_target_density /= 0.82
    elif band != "G":
        raise ValueError(f"Unsupported band: {band}")

    density_min: float = base_cumulative_density(magnitude_min, band="G")
    density_max: float = base_cumulative_density(magnitude_max, band="G")

    if base_target_density < density_min:
        return magnitude_min
    if base_target_density > density_max:
        return magnitude_max

    magnitudes: NDArray = np.linspace(magnitude_min, magnitude_max, 2000)
    densities: NDArray = np.array(
        [base_cumulative_density(mag, band="G") for mag in magnitudes]
    )
    return float(np.interp(base_target_density, densities, magnitudes))
