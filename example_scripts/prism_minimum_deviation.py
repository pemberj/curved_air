# /// script
# dependencies = [
#     "curved_air>=0.1.0",
# ]
# [tool.uv.sources]
# curved_air = { path = "..", editable = true }
# ///

import re
from pathlib import Path

import matplotlib.patheffects as pe
import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt

from curved_air.glass import GlassCatalogue
from curved_air.optics import Prism
from curved_air.plotting import plot_style, wavelength_to_rgb

plt.style.use(plot_style)

SCRIPT_PATH = Path(__file__).parent


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


def plot_mid_wavelength() -> None:
    """ """

    prism = Prism(
        glass=F2,
        apex_angle=60 * u.deg,
        min_wl=390 * u.nm,
        max_wl=950 * u.nm,
    )

    fig = plt.figure("Prism deviation angles as a function of wavelength")
    ax = fig.gca()

    wls = np.linspace(390, 950, 100) * u.nm

    deltas = prism.deviation_angle(wls)

    ax.plot(
        wls,
        deltas,
        label="Prism deviation angle",
    )

    x0, x1 = 300, 1000
    y0, y1 = 45, 53

    for wl in [390 * u.nm, 494.3 * u.nm, 950 * u.nm]:
        ax.plot(
            [x0, wl.value, wl.value, wl.value],
            [
                prism.deviation_angle(wl.value * u.nm).value,
                prism.deviation_angle(wl.value * u.nm).value,
                y0,
                y0,
            ],
            ls="--",
            color="k",
            alpha=0.5,
            zorder=-1,
        )

    ax.plot(
        494.3 * u.nm,
        prism.deviation_angle(494.3 * u.nm),
        marker=".",
        color=wavelength_to_rgb(494.3 * u.nm, fade_factor=0.5),
        lw=0,
    )

    ax.text(
        494.3 + 25,
        prism.deviation_angle(494.3 * u.nm).value,
        "Mid-wavelength: 494.3 nm",
        ha="left",
        va="center",
        fontsize=12,
        bbox={
            "boxstyle": "round,pad=0.3",
            "fc": "white",
            "ec": "black",
            "lw": 1,
            "alpha": 0.75,
        },
    )

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Deviation angle (deg)")

    ax.legend()

    plt.show()


def plot_deviation_angle_by_wavelength() -> None:
    """ """

    prism = Prism(
        glass=F2,
        apex_angle=60 * u.deg,
        min_wl=390 * u.nm,
        max_wl=950 * u.nm,
    )

    fig = plt.figure("Prism deviation angles and common laser wavelengths")
    ax = fig.gca()

    wls = np.linspace(390, 950, 100) * u.nm

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


def plot_deviation_angle_by_incidence_angle() -> None:
    """ """

    prism = Prism(
        glass=F2,
        apex_angle=60 * u.deg,
        min_wl=390 * u.nm,
        max_wl=950 * u.nm,
    )

    fig = plt.figure("Prism deviation angles as a function of incidence angle")
    ax = fig.gca()

    incidence_angles = np.linspace(40, 75, 100) * u.deg
    # wavelengths = [380 * u.nm, 494.3 * u.nm, 950 * u.nm]

    wavelength = 494.3 * u.nm

    deltas = prism.deviation_angle(wl=wavelength, incidence_angle=incidence_angles)

    ax.plot(
        incidence_angles,
        deltas,
        label=f"Deviation angle\n{prism.apex_angle.value:.3g}$\\degree$ {prism.glass.name} prism\n$\\lambda={wavelength.value:.1f}$ nm",
        color=wavelength_to_rgb(wavelength),
        path_effects=[pe.Stroke(linewidth=4, foreground="k"), pe.Normal()],
    )

    x0, x1 = 35, 80
    y0, y1 = 46, 58

    i = prism.minimum_deviation_incidence_angle(wl=wavelength).value
    d = prism.minimum_deviation_angle(wl=wavelength).value

    ax.plot(
        [x0, i, i, i],
        [d, d, y0, y0],
        color="k",
        linestyle="--",
        alpha=0.5,
        zorder=-1,
    )

    ax.text(
        i + 1,
        d - 0.25,
        f"Minimum deviation\nincidence angle: {i:.3g}$\\degree$",
        ha="left",
        va="top",
        fontsize=14,
        bbox={
            "boxstyle": "round,pad=0.3",
            "fc": "white",
            "ec": "black",
            "lw": 1,
            "alpha": 0.75,
        },
    )

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)

    ax.set_xlabel("Incidence angle (deg)")
    ax.set_ylabel("Deviation angle (deg)")

    ax.legend()

    plt.show()


if __name__ == "__main__":
    plot_mid_wavelength()

    # plot_deviation_angle_by_wavelength()

    # plot_deviation_angle_by_incidence_angle()
