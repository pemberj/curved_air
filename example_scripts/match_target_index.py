# /// script
# dependencies = [
#     "curved_air>=0.1.0",
# ]
# [tool.uv.sources]
# curved_air = { path = "..", editable = true }
# ///

"""
Code grabbed from and old main() function in optimisation.py

"""

import re
from pathlib import Path

import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt

from curved_air.glass import GlassCatalogue
from curved_air.optimisation import match_target_refractive_index


def main() -> None:
    agf_dir: Path = Path(__file__).parent / "../data/AGFs"

    agf_files = sorted(
        [
            f
            for f in agf_dir.glob("*.[Aa][Gg][Ff]")
            if re.search(pattern="ohara|schott", string=f.name)
        ]
    )
    if not agf_files:
        raise FileNotFoundError("No AGF files found in the data directory.")

    gc = (
        GlassCatalogue()
        .read_AGF(agf_filename=agf_files)
        .filter(
            # status_description=["standard", "preferred"],
            min_wl=390 * u.nm,
            max_wl=950 * u.nm,
            min_transmission=0.97,
            min_average_transmission=0.99,
        )
    )

    ADC = match_target_refractive_index(
        g1=gc["K10"],
        g2=gc["S-FPL51"],
        min_wl=390 * u.nm,
        max_wl=950 * u.nm,
        num_samples=50,
        target_refractive_index=lambda _wl: 1.5,
    )

    for p in ADC.prism_list:
        print(p)

    wls = np.linspace(0.39, 0.95, 100) * u.um

    out_rays = ADC.trace(wls=wls)

    fig = plt.figure()
    ax = fig.gca()

    ax.plot(wls, out_rays.to(u.degree))
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("Deviation angle (deg)")

    plt.show()


if __name__ == "__main__":
    main()
