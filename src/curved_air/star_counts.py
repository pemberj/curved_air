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


SKY_REGION_MODELS: tuple[SkyRegionModel, ...] = (
    SkyRegionModel("All sky average", 1.00),
    SkyRegionModel("Galactic plane |b| < 10°", 2.20),
    SkyRegionModel("Low latitude 10° ≤ |b| < 20°", 1.55),
    SkyRegionModel("Mid latitude 20° ≤ |b| < 40°", 1.00),
    SkyRegionModel("High latitude 40° ≤ |b| < 60°", 0.62),
    SkyRegionModel("Polar |b| ≥ 60°", 0.42),
)

DEFAULT_SKY_COVERAGE_TO_REGION: dict[int, SkyRegion] = {
    50: "All sky average",
    68: "Mid latitude 20° ≤ |b| < 40°",
    95: "High latitude 40° ≤ |b| < 60°",
    99: "Polar |b| ≥ 60°",
}


def sky_region_scale(region: SkyRegion) -> float:
    for model in SKY_REGION_MODELS:
        if model.label == region:
            return model.density_scale
    raise ValueError(f"Unknown sky region: {region}")


def coverage_region(coverage_percent: int) -> SkyRegion:
    if coverage_percent not in DEFAULT_SKY_COVERAGE_TO_REGION:
        raise ValueError("Coverage must be one of 50, 68, 95, or 99 percent.")
    return DEFAULT_SKY_COVERAGE_TO_REGION[coverage_percent]


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


def cumulative_density(
    magnitude: float,
    band: Band = "G",
    region: SkyRegion = "All sky average",
) -> float:
    return base_cumulative_density(magnitude=magnitude, band=band) * sky_region_scale(
        region
    )


def required_field_area(
    magnitude: float,
    band: Band = "G",
    region: SkyRegion = "All sky average",
) -> ScalarQuantity:
    density: float = cumulative_density(magnitude=magnitude, band=band, region=region)
    return (1 / density) * u.deg**2


def equivalent_circular_diameter(area: ScalarQuantity) -> ScalarQuantity:
    radius: ScalarQuantity = np.sqrt(area / np.pi)
    return (2 * radius).to(u.arcmin)


def area_from_circular_diameter(diameter: ScalarQuantity) -> ScalarQuantity:
    radius: ScalarQuantity = diameter / 2
    return (np.pi * radius**2).to(u.deg**2)


def limiting_magnitude_for_area(
    area: ScalarQuantity,
    band: Band = "G",
    region: SkyRegion = "All sky average",
    magnitude_min: float = -2.0,
    magnitude_max: float = 18.0,
) -> float:
    target_density: float = (1 / area.to(u.deg**2)).value
    scale: float = sky_region_scale(region)
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
