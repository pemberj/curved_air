# /// script
# dependencies = [
#     "curved_air>=0.1.0",
# ]
# [tool.uv.sources]
# curved_air = { path = "..", editable = true }
# ///

import re
from pathlib import Path

import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt

from curved_air.glass import GlassCatalogue
from curved_air.optics import Prism
from curved_air.plotting import plot_style, wavelength_to_rgb

plt.style.use(plot_style)

SCRIPT_PATH = Path(__file__).parent


def main() -> None:
    agf_dir = SCRIPT_PATH / "../data/AGFs"

    agf_files = sorted(
        [
            f
            for f in agf_dir.glob("*.[Aa][Gg][Ff]")
            if re.search(pattern="schott", string=f.name)
        ]
    )
    if not agf_files:
        raise FileNotFoundError("No AGF files found in the data directory.")

    gc = GlassCatalogue().read_AGF(agf_filename=agf_files)

    F2 = gc["F2"]

    prism = Prism(
        glass=F2,
        apex_angle=60 * u.deg,
        min_wl=390 * u.nm,
        max_wl=950 * u.nm,
    )

    print(prism.incidence_angle)

    wls = np.linspace(390, 950, 100) * u.nm

    fig = plt.figure("Prism deviation angles and common laser wavelengths")
    ax = fig.gca()

    deltas = prism.exitance_angle(wls)

    ax.plot(
        wls,
        deltas,
        label="Prism deviation angle",
    )

    ax.axhline(
        y=(max(deltas) + min(deltas)).to(u.deg).value / 2,
        color="k",
        linestyle="--",
        alpha=0.5,
        label=f"Incidence angle: {prism.incidence_angle:.3g}",
    )

    ax.axvline(
        x=494.3,
        color="k",
        linestyle="-.",
        alpha=0.5,
        label="Minimum deviation wavelength: 494.3 nm",
    )

    laser_wls = {
        "405nm": 405 * u.nm,
        "Green HeNe": 543.5 * u.nm,
        "Red HeNe": 632.8 * u.nm,
        "Fibre fault tester": 650 * u.nm,
        "Rubidium": 780 * u.nm,
    }

    for label, laser_wl in laser_wls.items():
        ax.plot(
            laser_wl,
            prism.exitance_angle(laser_wl),
            lw=0,
            marker=".",
            color=wavelength_to_rgb(laser_wl),
            label=label,
            zorder=10,
        )

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Deviation angle (deg)")

    ax.legend()

    plt.show()


if __name__ == "__main__":
    main()
