""" """

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from astropy import units as u

if TYPE_CHECKING:
    from astropy.units import Quantity
    from matplotlib import pyplot as plt

    from curved_air.type_hinting import ScalarQuantity


@dataclass
class Detector:
    name: str = ""
    pixel_size: ScalarQuantity = 9 * u.um
    nx: int = 1024
    ny: int = 1024
    gain: ScalarQuantity = 1 * u.electron / u.photon  # e¯ / photon
    dark_noise: ScalarQuantity = 0 * u.electron / u.second  # e¯ / s
    read_noise: ScalarQuantity = 0 * u.electron  # e¯
    QE: Path | str | float | None = None  # Quantum efficiency, optional
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

CCD4k15 = Detector(
    name="4k CCD (15um)", pixel_size=15 * u.um, nx=4096, ny=4096, reference=""
)

CCD4k9 = Detector(
    name="4k CCD (9um)", pixel_size=9 * u.um, nx=4096, ny=4096, reference=""
)

CCD2k15 = Detector(
    name="2k CCD (15um)", pixel_size=15 * u.um, nx=2048, ny=2048, reference=""
)

iKon_L = Detector(
    name="iKon-L 936",
    pixel_size=13.5 * u.um,
    nx=2048,
    ny=2048,
    reference="https://andor.oxinst.com/assets/uploads/products/andor/documents/andor-ikon-l-936-specifications.pdf",
)

iKon_M = Detector(
    name="iKon-M 934",
    pixel_size=13 * u.um,
    nx=1024,
    ny=1024,
    reference="https://andor.oxinst.com/assets/uploads/products/andor/documents/andor-ikon-m-934-specifications.pdf",
)

KAF8300 = Detector(
    name="Kodak KAF-8300",
    pixel_size=5.4 * u.micron,
    nx=3326,
    ny=2504,
    gain=0.45 / u.photon,
    dark_noise=0.02 * u.electron / u.second,
    read_noise=9.0 * u.electron,
    QE=0.55,
)

ICX825 = Detector(
    name="Sony ICX825",
    pixel_size=6.45 * u.micron,
    nx=2448,
    ny=2048,
    gain=0.7 / u.photon,
    dark_noise=0.01 * u.electron / u.second,
    read_noise=6.5 * u.electron,
    QE=0.77,
)

ICX674 = Detector(
    name="Sony ICX674",
    pixel_size=4.54 * u.micron,
    nx=2448,
    ny=2048,
    gain=0.7 / u.photon,
    dark_noise=0.01 * u.electron / u.second,
    read_noise=6.0 * u.electron,
    QE=0.77,
)

CCD_DETECTORS: dict[str, Detector] = {
    d.name: d
    for d in [
        STA1600,
        H4RG_15,
        H4RG_10,
        H2RG_18,
        e2v_9k,
        CCD4k15,
        CCD4k9,
        CCD2k15,
        iKon_L,
        iKon_M,
        KAF8300,
        ICX825,
        ICX674,
    ]
}


IMX265 = Detector(
    name="Sony IMX265 (CMOS)",
    pixel_size=3.45 * u.micron,
    nx=2448,
    ny=2048,
    gain=1 / u.photon,
    dark_noise=3.2 * u.electron / u.second,
    read_noise=2.2 * u.electron,
    QE=Path(__file__).parent / ".." / "data" / "IMX265_QE.csv",
)

IMX174 = Detector(
    name="Sony IMX174 (CMOS)",
    pixel_size=5.86 * u.micron,
    nx=1936,
    ny=1216,
    gain=0.9 / u.photon,
    dark_noise=4.0 * u.electron / u.second,
    read_noise=3.5 * u.electron,
    QE=0.76,
)

IMX178 = Detector(
    name="Sony IMX178 (CMOS)",
    pixel_size=2.4 * u.micron,
    nx=3096,
    ny=2080,
    gain=0.9 / u.photon,
    dark_noise=2.0 * u.electron / u.second,
    read_noise=2.1 * u.electron,
    QE=0.8,
)

IMX290 = Detector(
    name="Sony IMX290 (CMOS)",
    pixel_size=2.9 * u.micron,
    nx=1936,
    ny=1096,
    gain=1.0 / u.photon,
    dark_noise=0.8 * u.electron / u.second,
    read_noise=1.5 * u.electron,
    QE=0.8,
)

IMX462 = Detector(
    name="Sony IMX462 (CMOS)",
    pixel_size=2.9 * u.micron,
    nx=1936,
    ny=1096,
    gain=1.0 / u.photon,
    dark_noise=0.7 * u.electron / u.second,
    read_noise=1.2 * u.electron,
    QE=0.82,
)

IMX585 = Detector(
    name="Sony IMX585 (CMOS)",
    pixel_size=2.9 * u.micron,
    nx=4032,
    ny=3040,
    gain=1.0 / u.photon,
    dark_noise=0.5 * u.electron / u.second,
    read_noise=1.1 * u.electron,
    QE=0.91,
)

CMOS_DETECTORS: dict[str, Detector] = {
    d.name: d
    for d in [
        IMX265,
        IMX174,
        IMX178,
        IMX290,
        IMX462,
        IMX585,
    ]
}


GSENSE2020BSI = Detector(
    name="Gpixel GSENSE2020BSI (sCMOS)",
    pixel_size=6.5 * u.micron,
    nx=2048,
    ny=2048,
    gain=0.95 / u.photon,
    dark_noise=0.2 * u.electron / u.second,
    read_noise=1.6 * u.electron,
    QE=0.95,
)
GSENSE4040BSI = Detector(
    name="Gpixel GSENSE4040BSI (sCMOS)",
    pixel_size=9.0 * u.micron,
    nx=4096,
    ny=4096,
    gain=0.95 / u.photon,
    dark_noise=0.15 * u.electron / u.second,
    read_noise=1.3 * u.electron,
    QE=0.95,
)

SCMOS_DETECTORS: dict[str, Detector] = {
    d.name: d
    for d in [
        GSENSE2020BSI,
        GSENSE4040BSI,
    ]
}


DETECTORS: dict[str, Detector] = {**CCD_DETECTORS, **CMOS_DETECTORS, **SCMOS_DETECTORS}
