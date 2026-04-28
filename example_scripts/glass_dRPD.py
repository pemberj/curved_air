# /// script
# dependencies = [
#     "curved_air>=0.1.0",
#     "PyQT6"
# ]
# [tool.uv.sources]
# curved_air = { path = "..", editable = true }
# ///

"""
A script to calculate and plot the differential relative partial dispersion between
glasses.

This can be a useful metric for glass selection, and is described in detail here:
http://www.primeoptics.com.au/Reference/dRPD_description.pdf
"""

import re
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u

from curved_air.glass import GlassCatalogue
from curved_air.plotting import plot_style

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from curved_air.type_hinting import ArrayQuantity

plt.style.use(plot_style)


def main() -> None:
    script_path: Path = Path(__file__).parent
    agf_dir: Path = script_path / "../data/AGFs"

    agf_files: list[Path] = sorted(
        [f for f in agf_dir.glob("*.[Aa][Gg][Ff]") if re.search("ohara|schott", f.name)]
    )

    if not agf_files:
        raise FileNotFoundError("No AGF files found in the data directory.")

    catalogue: GlassCatalogue = (
        GlassCatalogue()
        .read_AGF(agf_file=agf_files)
        .filter(
            # status_description=["standard | preferred"],
            min_wl=390 * u.nm,
            max_wl=950 * u.nm,
            # min_transmission = 0.97,
            # min_average_transmission = 0.99,
        )
    )

    wls: ArrayQuantity = np.linspace(0.39, 2, 30) * u.um

    reference_glass: str = "S-FPL51"
    query_glasses: list[str] = [
        "N-BK7",
        "N-LLF6",
        "N-SF5",
        "CAF2",
        "F_SILICA",
        "N-BAK2",
        "N-FK54",
        "S-BAL35",
        "PBL6",
        "PBM2Y",
        "F2",
        "BSL7Y",
        "S-FPL51",
        "S-FPL53",
        "S-FPL55",
        "S-LAH97",
    ]

    ref_dV: NDArray[np.floating] = catalogue[reference_glass].dV(wls)  # noqa: F841
    ref_dRPD: NDArray[np.floating] = catalogue[reference_glass].dRPD(wls)

    _fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)

    for g in catalogue:
        for q in query_glasses:
            if q not in g.alternative_names:
                continue

            y0: NDArray[np.floating] = g.dV(wls)  # - ref_dV
            y1: NDArray[np.floating] = g.dRPD(wls) - ref_dRPD

            if max(np.abs(y1)) > 0.01:  # noqa: PLR2004
                continue

            axs[0].plot(wls.value, y0, label=g.name, alpha=0.5)
            axs[0].text(
                wls[-1].value, y0[-1], g.name, fontsize=10, va="center", ha="left"
            )

            axs[1].plot(wls.value, y1, label=g.name, alpha=0.5)
            axs[1].text(
                wls[-1].value, y1[-1], g.name, fontsize=10, va="center", ha="left"
            )

    axs[0].set_ylabel("dV")
    axs[1].set_xlabel("Wavelength (um)")
    axs[1].set_ylabel("dRPD - ref_dRPD")

    plt.show()

    # fig = plt.figure()
    # ax = fig.gca()

    # for g in catalogue:
    #     x = np.mean(g.dV(wls) - ref_dV)
    #     y = np.mean(g.dRPD(wls)- ref_dRPD)

    #     # if max(np.abs(y1)) > 0.01:
    #     #     continue

    #     ax.plot(x, y, marker=".", markersize=10, label=g.name, alpha=0.5)
    #     ax.text(x, y, f"  {g.name}", fontsize=8, va="center", ha="left")

    # ax.set_xlim(ax.get_xlim()[::-1])
    # ax.set_xlabel("Average (dV - ref_dV)")
    # ax.set_ylabel("Average (dRPD - ref_dRPD)")
    # plt.show()


if __name__ == "__main__":
    main()
