""" """

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from astropy import units as u

if TYPE_CHECKING:
    from astropy.units import Quantity
    from matplotlib import pyplot as plt

    from curved_air.type_hinting import ScalarQuantity


@dataclass
class Detector:
    name: str = "STA1600"
    pixel_size: ScalarQuantity = 9 * u.um
    nx: int = 10560
    ny: int = 10560

    def __post_init__(self) -> None:
        if isinstance(self.pixel_size, int | float):
            self.pixel_size: ScalarQuantity = self.pixel_size * u.um

        if not isinstance(self.nx, int) or not isinstance(self.ny, int):
            raise TypeError("nx and ny must be integers.")

        if self.nx <= 0 or self.ny <= 0:
            raise ValueError("nx and ny must be positive integers.")

    @property
    def size(self) -> tuple[Quantity, Quantity]:
        return (self.pixel_size * self.nx, self.pixel_size * self.ny)

    def plot_edge(self, ax: plt.Axes, name: bool = False) -> None:
        halfx: float = self.size[0].to(u.mm).value / 2
        halfy: float = self.size[1].to(u.mm).value / 2

        ax.plot(
            [-halfx, -halfx, halfx, halfx, -halfx],
            [-halfy, halfy, halfy, -halfy, -halfy],
            ls="--",
            lw=4,
            color="k",
            alpha=0.25,
        )

        if name:
            ax.text(
                -(halfx),
                -(halfy + 1),
                self.name,
                fontsize=18,
                ha="right",
                va="top",
                alpha=0.5,
            )


STA1600 = Detector(
    name="STA1600",
    pixel_size=9 * u.um,
    nx=10560,
    ny=10560,
)

H4RG_15 = Detector(
    name="H4RG 15um",
    pixel_size=15 * u.um,
    nx=4096,
    ny=4096,
)

H4RG_10 = Detector(
    name="H4RG 10um",
    pixel_size=10 * u.um,
    nx=4096,
    ny=4096,
)

H2RG_18 = Detector(
    name="H2RG 18um",
    pixel_size=18 * u.um,
    nx=2048,
    ny=2048,
)
