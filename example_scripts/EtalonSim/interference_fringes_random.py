# /// script
# dependencies = [
#     "curved_air>=0.1.0",
# ]
# [tool.uv.sources]
# curved_air = { path = "../..", editable = true }
# ///

from pathlib import Path
from typing import TYPE_CHECKING

import astropy.units as u
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import griddata

from curved_air.beams import SphericalBeam
from curved_air.plotting import plot_style

if TYPE_CHECKING:
    from astropy.units import Quantity

plt.style.use(plot_style)

SCRIPT_NAME: Path = Path(__file__).stem
SCRIPT_PATH: Path = Path(__file__).parent


def main() -> None:
    """ """
    source_spacing: Quantity = 0.5 * u.mm
    distance_to_detector = 0.5 * u.m
    detector_num_pix = 10800
    pixel_size = 9 * u.um
    detector_x = detector_num_pix * pixel_size

    num_samples = 100_000

    x, y = (np.random.rand(2, num_samples) - 0.5) * detector_x

    # HeNe laser
    wavelength = 632.8 * u.nm

    z = [distance_to_detector.to("mm").value] * len(x) * u.mm

    beam_1 = SphericalBeam(wavelength=wavelength)
    E_1 = beam_1.E_field(x - source_spacing / 2, y, z)
    beam_2 = SphericalBeam(wavelength=wavelength)
    E_2 = beam_2.E_field(x + source_spacing / 2, y, z)

    interference = E_1 + E_2

    intensity = np.real(interference * np.conj(interference))

    # Set interpolation bounds
    xplot = np.linspace(
        -detector_x.to("mm").value / 2, detector_x.to("mm").value / 2, 2_000
    )
    yplot = np.linspace(
        -detector_x.to("mm").value / 2, detector_x.to("mm").value / 2, 2_000
    )

    # Interpolate data into a regular grid
    zplot = griddata(
        (x.to("mm").value, y.to("mm").value),
        intensity,
        (xplot[None, :], yplot[:, None]),
        method="cubic",
    )

    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()

    contour = ax.pcolormesh(xplot, yplot, zplot, cmap="afmhot")  # noqa: F841
    # ax.set_xticks([])
    # ax.set_yticks([])

    # for s in ["top", "bottom", "left", "right"]:
    #     ax.spines[s].set_visible(False)

    Path(SCRIPT_PATH / "plots").mkdir(parents=True, exist_ok=True)
    plt.savefig(SCRIPT_PATH / "plots" / f"{SCRIPT_NAME}.png")

    plt.show()


if __name__ == "__main__":
    main()
