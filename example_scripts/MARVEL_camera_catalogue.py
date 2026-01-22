# /// script
# dependencies = [
#     "curved_air>=0.1.0",
#     "rich>=10.0.0",
# ]
# [tool.uv.sources]
# curved_air = { path = "..", editable = true }
# ///

"""
A script used to filter input glass catalogues and output the filtered catalogue to a
new AGF file for use in optimising the MARVEL spectrograph camera design.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np  # noqa: F401
from astropy import units as u

from curved_air.glass import GlassCatalogue
from curved_air.plotting import plot_style

plt.style.use(plot_style)


def main() -> None:
    script_path: Path = Path(__file__).parent
    agf_dir: Path = script_path / "../data/AGFs"

    agf_files: list[Path] = sorted(
        [
            f
            for f in agf_dir.glob("*.[Aa][Gg][Ff]")
            if re.search(pattern="ohara|schott|nikon", string=f.name)
        ]
    )
    MARVEL_gc: GlassCatalogue = GlassCatalogue().read_AGF(agf_filename=agf_files)

    colours: list[str] = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    colours: dict[str, str] = {
        "schott": colours[0],
        "ohara": colours[1],
        "corning": colours[2],
        "nikon": colours[3],
    }

    MARVEL_gc: GlassCatalogue = MARVEL_gc.filter(
        # status_description=["standard", "preferred"],
        min_wl=390 * u.nm,
        max_wl=950 * u.nm,
        min_transmission=0.985,
        min_average_transmission=0.995,
    )

    MARVEL_gc.plot_Abbe_diagram()
    MARVEL_gc.plot_transmission(wls=np.linspace(0.39, 0.45, 100) * u.um)

    # gc.write_AGF("./OUTPUT_CATALOGUE.AGF")


if __name__ == "__main__":
    main()
