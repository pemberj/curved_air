"""
A script to calculate and plot the differential relative partial dispersion between
glasses.

This can be a useful metric for glass selection, and is described in detail here:
http://www.primeoptics.com.au/Reference/dRPD_description.pdf
"""

import re
import sys
from pathlib import Path

import numpy as np
from astropy import units as u
import matplotlib.pyplot as plt

from curved_air.glass import GlassCatalogue
from curved_air.plot_style import plot_style

plt.style.use(plot_style)


script_path = Path(__file__).parent
sys.path.append(script_path / "../src")


def main() -> None:
    agf_dir: Path = script_path / "../data/AGFs"

    agf_files: list[Path] = sorted(
        [f for f in agf_dir.glob("*.[Aa][Gg][Ff]") if re.search("ohara|schott", f)]
    )

    if not agf_files:
        raise FileNotFoundError("No AGF files found in the data directory.")

    catalogue: GlassCatalogue = (
        GlassCatalogue()
        .read_AGF(agf_filename=agf_files)
        .filter(
            status_description=["standard | preferred"],
            min_wl=390 * u.nm,
            max_wl=950 * u.nm,
            # min_transmission = 0.97,
            # min_average_transmission = 0.99,
        )
    )

    wls: np.ndarray = np.linspace(0.39, 2, 30)

    reference_glass = "S-FPL51"
    query_glasses = [
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

    ref_dV = catalogue[reference_glass].dV(wls)  # noqa: F841
    ref_dRPD = catalogue[reference_glass].dRPD(wls)

    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(12, 8))

    for g in catalogue:
        for q in query_glasses:
            if q not in g.alternative_names:
                continue

            y0 = g.dV(wls)  # - ref_dV
            y1 = g.dRPD(wls) - ref_dRPD

            if max(np.abs(y1)) > 0.01:  # noqa: PLR2004
                continue

            axs[0].plot(wls, y0, label=g.name, alpha=0.5)
            axs[0].text(wls[-1], y0[-1], g.name, fontsize=8, va="center", ha="left")

            axs[1].plot(wls, y1, label=g.name, alpha=0.5)
            axs[1].text(wls[-1], y1[-1], g.name, fontsize=8, va="center", ha="left")

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
