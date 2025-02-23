# /// script
# dependencies = [
#     "curved_air>=0.1.0",
# ]
# [tool.uv.sources]
# curved_air = { path = "..", editable = true }
# ///

"""
A script demonstrating the method of reading in one or more glass catalogues
from manufacturer-provided AGF files, filtering the catalogue, and outputting
this to a new AGF file for use in Zemax.
"""

import re
from pathlib import Path

import numpy as np
from astropy import units as u
import matplotlib.pyplot as plt

# from glass_select.glass import GlassCatalogue, _optimise_ADC
from curved_air.glass import GlassCatalogue
from curved_air.plotting import plot_style

plt.style.use(plot_style)


def main() -> None:
    script_path = Path(__file__).parent
    agf_dir = script_path / "../data/AGFs"

    agf_files: list[Path] = sorted(
        [
            f
            for f in agf_dir.glob("*.[Aa][Gg][Ff]")
            if re.search(pattern="ohara|schott", string=f.name)
        ]
    )

    if not agf_files:
        raise FileNotFoundError("No AGF files found in the data directory.")

    gc: GlassCatalogue = (
        GlassCatalogue()
        .read_AGF(agf_filename=agf_files)
        .filter(
            status_description=["standard", "preferred"],
            min_wl=390 * u.nm,
            max_wl=950 * u.nm,
            min_transmission=0.97,
            min_average_transmission=0.99,
        )
    )

    print(f"{gc["K10"].rii_dispersion_coefficients = }")
    print(f"{gc["K10"].formula = }")

    print(f"{gc["K10"].n() = }")
    print(f"{gc['K10'].n(0.5) = }")
    print(f"{gc['K10'].n(0.5*u.um) = }")
    print(f"{gc['K10'].n(500*u.nm) = }")

    wls = np.linspace(0.39, 0.95, 100) * u.um

    plt.plot(wls, gc["K10"].n(wls))
    plt.show()


if __name__ == "__main__":
    main()
