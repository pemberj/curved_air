# /// script
# dependencies = [
#     "curved_air>=0.1.0",
# ]
# [tool.uv.sources]
# curved_air = { path = "..", editable = true }
# ///

"""
- Create an input catalogue of glasses from AGF files
- Filter the catalogue for glasses with a minimum transmission of 98% and an average
  transmission of at least 99% over the visible range
- Iterate over all unique glass combinations in the catalogue, optimise an ADC for each,
  and list any that have angular residuals less than 25 milliarcseconds
"""

import re
from pathlib import Path

import astropy.units as u
from matplotlib import pyplot as plt

from curved_air.glass import GlassCatalogue
from curved_air.plotting import plot_style

plt.style.use(plot_style)


def main() -> None:
    script_path = Path(__file__).parent
    agf_dir: Path = script_path / "../data/AGFs"

    agf_files = sorted(
        [f for f in agf_dir.glob("*.[Aa][Gg][Ff]") if re.search("ohara|schott", f.name)]
    )

    if not agf_files:
        raise FileNotFoundError(f"No AGF files found in {agf_dir}")

    print(agf_files)

    catalogue = (
        GlassCatalogue()
        .read_AGF(agf_filename=agf_files)
        .filter(
            min_wl=380 * u.nm,
            max_wl=950 * u.nm,
            min_transmission=0.98,
            min_average_transmission=0.99,
            # status_description="preferred",
        )
    )

    # catalogue.plot_Abbe_diagram()

    catalogue.compute_ADCs(max_residuals=25 * u.mas)


if __name__ == "__main__":
    main()
