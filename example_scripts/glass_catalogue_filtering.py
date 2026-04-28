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

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u

# from glass_select.glass import GlassCatalogue, _optimise_ADC
from curved_air.glass import GlassCatalogue
from curved_air.plotting import plot_style

plt.style.use(plot_style)


def main() -> None:
    script_path: Path = Path(__file__).parent
    agf_dir: Path = script_path / "../data/AGFs"

    print(agf_dir)
    print(agf_dir.glob("*.[Aa][Gg][Ff]"))

    agf_files: list[Path] = sorted(
        [f for f in agf_dir.glob("*.[Aa][Gg][Ff]") if re.search("ohara|schott", f.name)]
    )

    if not agf_files:
        raise FileNotFoundError("No AGF files found in the data directory.")

    gc: GlassCatalogue = GlassCatalogue().read_AGF(agf_file=agf_files)

    gc: GlassCatalogue = gc.filter(
        status_description=["standard", "preferred"],
        min_wl=390 * u.nm,
        max_wl=950 * u.nm,
        min_transmission=0.96,
        min_average_transmission=0.97,
    )

    gc.plot_Abbe_diagram()
    gc.plot_transmission(wls=np.linspace(0.39, 0.45, 100) * u.um)

    print(gc)

    # gc.write_AGF("./OUTPUT_CATALOGUE.AGF")


if __name__ == "__main__":
    main()
