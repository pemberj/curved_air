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
    """ """
    mirror_spacing: Quantity = 6 * u.mm  # mm
    round_trip = 2 * mirror_spacing
    round_trip_mm = round_trip.to("mm").value
    num_round_trips = 100  # number of reflections on the OUTPUT surface
    mirror_reflectivity = 0.95

    r = 5 * u.mm  # maximum radius of beam

    w_0 = 4 * u.mm
    wavelength = mirror_spacing / int(mirror_spacing / (780 * u.nm))

    beam = GaussianBeam(wavelength=wavelength, w_0=w_0)

    Zs = np.arange(0, beam.z_R.to("mm").value / 10, round_trip_mm) * u.mm
    reflectivity = np.array([mirror_reflectivity ** (n) for n in range(1, len(Zs) + 1)])

    gouy = np.angle(beam.E_gouy_integrated(r=r, z=Zs)).value * reflectivity
    radi = np.angle(beam.E_radial_integrated(r=r, z=Zs)).value * reflectivity

    # Don't use round numbers for this plot
    plt.rcParams["axes.autolimit_mode"] = "data"

    fig = plt.figure()
    ax = fig.gca()

    G = ax.plot(Zs, gouy, ls="--", label="Gouy phase")
    R = ax.plot(Zs, radi, ls="-.", color=G[0].get_color(), label="Radial phase")  # noqa: F841

    ax.plot(Zs, gouy + radi, color=G[0].get_color(), alpha=0.5, label="Sum")

    ax.legend()

    ax.set_title(
        "Contributions of Gouy and radial phase,\n"
        f"reduced by mirror reflection for $w_0=${w_0}"
    )
    ax.set_xlabel(f"Propagation distance z [mm], {num_round_trips} round trips")
    ax.set_ylabel("Phase offset [cycles]")

    Path(SCRIPT_PATH / "plots").mkdir(exist_ok=True)
    plt.savefig(SCRIPT_PATH / "plots" / f"{SCRIPT_NAME}.png")

    plt.show()


if __name__ == "__main__":
    main()
