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

    max_radius = 6 * u.mm

    w_0 = 5 * u.mm

    wavelength = mirror_spacing / int(mirror_spacing / (780 * u.nm))

    beam = GaussianBeam(wavelength=wavelength, w_0=w_0)

    Zs = np.arange(0, round_trip_mm * num_round_trips, round_trip_mm) * u.mm
    reflectivity = np.array([mirror_reflectivity ** (n) for n in range(1, len(Zs) + 1)])

    gouy = np.angle(beam.E_gouy_integrated(r=max_radius, z=Zs))
    radi = np.angle(beam.E_radial_integrated(r=max_radius, z=Zs))

    gouy = reflectivity * np.array(gouy)
    radi = reflectivity * np.array(radi)

    # Don't use round numbers for this plot
    plt.rcParams["axes.autolimit_mode"] = "data"

    fig = plt.figure()
    ax = fig.gca()

    ax.plot(Zs, gouy, label="Gouy phase")
    ax.plot(Zs, radi, label="Radial phase")

    ax.set_xlabel("Z (mm)")
    ax.set_ylabel("Phase (rad)")

    # ax.plot(Zs, gouy + radi)

    Path(SCRIPT_PATH / "plots").mkdir(parents=True, exist_ok=True)
    plt.savefig(SCRIPT_PATH / "plots" / f"{SCRIPT_NAME}.png")

    plt.show()


if __name__ == "__main__":
    main()
