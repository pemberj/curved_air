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

from curved_air.beams import GaussianBeam
from curved_air.plotting import plot_style

if TYPE_CHECKING:
    from astropy.units import Quantity

plt.style.use(plot_style)

SCRIPT_NAME: Path = Path(__file__).stem
SCRIPT_PATH: Path = Path(__file__).parent


def main() -> None:
    mirror_spacing: Quantity = 6 * u.mm  # mm
    round_trip = 2 * mirror_spacing
    round_trip_mm = round_trip.to("mm").value
    num_round_trips = 100  # number of reflections on the OUTPUT surface
    mirror_reflectivity = 0.95

    z0 = 0

    Zs = np.arange(z0, z0 + (num_round_trips) * round_trip_mm, round_trip_mm) * u.mm

    reflectivity = np.array([mirror_reflectivity ** (n) for n in range(1, len(Zs) + 1)])

    # Don't use round numbers for this plot
    plt.rcParams["axes.autolimit_mode"] = "data"

    fig = plt.figure()
    ax = fig.gca()

    wavelength = (mirror_spacing / int(mirror_spacing / (780 * u.nm))).to("nm")

    r = 5 * u.mm  # radius at which to evaluate curvature, radial phase

    # waist_radii = [2.5, 3, 4, 5, 6] * u.mm
    waist_radii = np.linspace(0.04, 6, 150) * u.mm
    # Zoom in to close-to-zero region
    # waist_radii = np.linspace(0.05, 1.75, 50) * u.mm

    summed_average_phase_differences = []
    for w_0 in waist_radii:
        beam = GaussianBeam(wavelength=wavelength, w_0=w_0)

        gouy = np.angle(beam.E_gouy_integrated(r=r, z=Zs)).value * reflectivity
        radi = np.angle(beam.E_radial_integrated(r=r, z=Zs)).value * reflectivity

        summed_average_phase_differences.append(np.sum(gouy + radi))

    ax.plot(
        waist_radii,
        summed_average_phase_differences,
        label=f"$z_0={z0 * u.mm.to('m')}$, $\\lambda=${wavelength:.0f}",
    )

    # ax.set_yscale("log")

    ax.legend()

    ax.set_xlabel("Beam waist radius, $w_0$ [mm]")
    ax.set_ylabel(f"Sum of average phase\nover {num_round_trips} round trips")

    Path(SCRIPT_PATH / "plots").mkdir(exist_ok=True)
    plt.savefig(SCRIPT_PATH / "plots" / f"{SCRIPT_NAME}.png")

    plt.show()


if __name__ == "__main__":
    main()
