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
    mirror_spacing: Quantity = (
        600 * u.mm
    )  # mm # NOTE changed mirror spacing to see very large distance from waist
    round_trip = 2 * mirror_spacing
    round_trip_mm = round_trip.to("mm").value
    num_round_trips = 100  # number of reflections on the OUTPUT surface

    Zs = np.arange(0, (num_round_trips) * round_trip_mm, round_trip_mm) * u.mm

    # mirror_reflectivity = 0.95
    # reflectivity = np.array([mirror_reflectivity ** (n) for n in range(1, len(Zs) + 1)])

    # Don't use round numbers for this plot
    plt.rcParams["axes.autolimit_mode"] = "data"

    fig = plt.figure()
    ax = fig.gca()

    wavelength = mirror_spacing / int(mirror_spacing / (780 * u.nm))

    r = 5 * u.mm  # radius at which to evaluate curvature, radial phase

    waist_radii = [2, 2.5, 3, 4] * u.mm
    # Zoom in to close-to-zero region
    # waist_radii = np.linspace(0.5, 2.5, 15) * u.mm

    summed_average_phase_differences = []
    for w_0 in waist_radii:
        beam = GaussianBeam(wavelength=wavelength, w_0=w_0)

        gouy = np.angle(beam.E_gouy_integrated(r=r, z=Zs))  # * reflectivity NOTE
        radi = np.angle(beam.E_radial_integrated(r=r, z=Zs))  # * reflectivity NOTE

        summed_average_phase_differences.append(np.sum(gouy + radi))

        G = ax.plot(0, 0)
        ax.plot(Zs.to(u.m), np.abs(gouy), color=G[0].get_color(), ls="--")
        ax.plot(Zs.to(u.m), np.abs(radi), color=G[0].get_color(), ls="-.")
        ax.fill_between(
            Zs.to(u.m).value,
            np.abs(gouy.to(u.rad).value),
            np.abs(radi.to(u.rad).value),
            color=G[0].get_color(),
            alpha=0.25,
            hatch="//",
            label=f"$w_0=${w_0} ($z_R\\approx${beam.z_R:.0f})",
        )

    ax.plot(0, 0, color="k", alpha=0.75, ls="--", label="Gouy phase")
    ax.plot(0, 0, color="k", alpha=0.75, ls="-.", label="radial phase (absolute)")

    # ax.set_title(f"Gouy phase and radial phase\nintegrated out to $r={r}$")

    ax.set_xlabel("propagation distance z [m]")
    ax.set_ylabel("average phase [cycles]")

    ax.legend()

    Path(SCRIPT_PATH / "plots").mkdir(parents=True, exist_ok=True)
    plt.savefig(SCRIPT_PATH / "plots" / f"{SCRIPT_NAME}.png", dpi=300)

    plt.show()

    # plt.savefig(f"plots/{os.path.basename(__file__).split('.')[0]}.png")


if __name__ == "__main__":
    main()
