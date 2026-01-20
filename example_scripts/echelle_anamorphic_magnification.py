# /// script
# dependencies = [
#     "curved_air>=0.1.0",
# ]
# [tool.uv.sources]
# curved_air = { path = "..", editable = true }
# ///

"""
This script calculates the anamorphic magnification for several echelles

This calculation depends only on the grating parameters: the angular spread of the free
spectral range (FSR) and the angle at which the grating is used.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import astropy.units as u
from matplotlib import pyplot as plt

from curved_air.plotting import plot_style
from curved_air.utils import richardson_gratings

if TYPE_CHECKING:
    from curved_air.optics.dispersers import EchelleGrating
    from curved_air.type_hinting import ScalarQuantity

plt.style.use(plot_style)

SCRIPT_PATH = Path(__file__).parent
OUT_PATH = Path("/home/jake/Nextcloud-IvS/Thesis Plots")

markers = ["o", "s", "D", "P", "v", "^", "<", ">", "h", "H", "p", "*", "+", "X"]


def main() -> None:
    gratings: list[EchelleGrating] = richardson_gratings()

    fig: plt.Figure = plt.figure("Anamorphic magnification", figsize=(8, 6))
    ax: plt.Axes = fig.gca()

    for i, g in enumerate(gratings):
        if g.gpmm.value >= 100:  # noqa: PLR2004
            continue

        g.min_wl: ScalarQuantity = 380 * u.nm
        g.max_wl: ScalarQuantity = 950 * u.nm

        ax.plot(
            g.R,
            g.relative_anamorphic_magnification(),
            lw=0,
            marker=markers[i],
            markersize=15,
            label=f"R={g.R}, {g.gpmm.value} g/mm",
            alpha=0.75,
        )

        if g.catalogue_number == "53-*-417E":
            ax.plot(
                g.R + 0.025,
                g.relative_anamorphic_magnification(),
                marker="o",
                markersize=35,
                markeredgecolor="r",
                markeredgewidth=2.6,
                markerfacecolor="none",
                lw=0,
            )

    ax.set_xlim(1, 7)
    ax.set_ylim(1, 2)

    ax.legend(loc="lower right", frameon=True)
    ax.set_xlabel("Grating R-number", size=16)
    ax.set_ylabel(
        "Relative anamorphic magnification of the\n"
        + "ends of the FSR of the reddest spectral order",
        size=16,
    )
    plt.show()


if __name__ == "__main__":
    main()
