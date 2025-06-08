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

    r = 5 * u.mm  # radius at which to evaluate curvature, radial phase

    # Don't use round numbers for this plot
    plt.rcParams["axes.autolimit_mode"] = "data"

    fig = plt.figure()
    ax = fig.gca()

    wavelength = mirror_spacing / int(mirror_spacing / (780 * u.nm))

    for z0 in [-1000, -500, -220.84, 0, 500]:
        Zs = np.arange(z0, z0 + (num_round_trips) * round_trip_mm, round_trip_mm) * u.mm

        reflectivity = np.array(
            [mirror_reflectivity ** (n) for n in range(1, len(Zs) + 1)]
        )

        waist_radii = np.linspace(0.5, 8, 100) * u.mm
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
            label=f"$z_0 = ${(z0 * u.mm).to(u.m)}",
        )

        if int(z0) == z0:
            text = f"$z_0 = ${(z0 * u.mm).to(u.m)}"
        else:
            text = f"$z_0 = ${z0 * u.mm:.2f}"

        ax.text(
            waist_radii[-1].value + 0.25,
            summed_average_phase_differences[-1],
            s=text,
            fontsize=14,
            ha="left",
            va="center",
            # bbox={
            #     "boxstyle": "round,pad=0.3",
            #     "facecolor": "white",
            #     "edgecolor": "black",
            #     "alpha": 0.5,
            # },
        )

    ax.set_xlim(0, 11)

    ax.set_xlabel("Beam waist radius, $w_0$ (mm)", size=14)
    ax.set_ylabel(
        f"Integrated phase over {num_round_trips} round trips\n(waves)", size=14
    )

    # ax.legend(frameon=True, loc="upper left")

    Path(SCRIPT_PATH / "plots").mkdir(exist_ok=True)
    plt.savefig(SCRIPT_PATH / "plots" / f"{SCRIPT_NAME}.png")

    plt.show()


if __name__ == "__main__":
    main()
