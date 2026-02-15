""" """

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

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
    reference: Any | None = None

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
                x=0,
                y=-(halfy + 1),
                s=self.name,
                fontsize=18,
                ha="center",
                va="top",
                alpha=0.5,
            )


STA1600 = Detector(
    name="STA1600",
    pixel_size=9 * u.um,
    nx=10560,
    ny=10560,
    reference="http://www.sta-inc.net/wp-content/uploads/2010/08/STA1600LN_011614_v3.pdf",
)

H4RG_15 = Detector(
    name="H4RG 15um",
    pixel_size=15 * u.um,
    nx=4096,
    ny=4096,
    reference="https://www.teledyne-si.com/what-we-do/products-and-services/hawaii-4rg",
)

H4RG_10 = Detector(
    name="H4RG 10um",
    pixel_size=10 * u.um,
    nx=4096,
    ny=4096,
    reference="https://www.teledyne-si.com/what-we-do/products-and-services/hawaii-4rg",
)

H2RG_18 = Detector(
    name="H2RG 18um",
    pixel_size=18 * u.um,
    nx=2048,
    ny=2048,
    reference="https://www.teledyne-si.com/en-us/Products-and-Services_/Documents/Infrared%20and%20Visible%20FPAs/TSI-0855%20H2RG%20Brochure-25Feb2022.pdf",
)

e2v_9k = Detector(
    name="e2v CCD290-99",
    pixel_size=10 * u.um,
    nx=9216,
    ny=9232,
    reference="https://www.teledynespaceimaging.com/en-us/Products_/Pages/CCD290-99.aspx",
)

DETECTORS: dict[str, Detector] = {
    d.name: d for d in [STA1600, H4RG_15, H4RG_10, H2RG_18, e2v_9k]
}
