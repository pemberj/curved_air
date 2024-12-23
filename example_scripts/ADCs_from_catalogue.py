""" """

import re
import sys
from pathlib import Path

import astropy.units as u
from matplotlib import pyplot as plt

from curved_air.glass import GlassCatalogue
from curved_air.plot_style import plot_style

plt.style.use(plot_style)

script_path = Path(__file__).parent
sys.path.append(script_path / "../src")


def main() -> None:
    agf_dir: Path = script_path / "../data/AGFs"

    agf_files = sorted(
        [f for f in agf_dir.glob("*.[Aa][Gg][Ff]") if re.search("ohara|schott", f)]
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
            min_transmission=0.97,
            min_average_transmission=0.99,
        )
    )

    # catalogue.plot_Abbe_diagram()

    catalogue.compute_ADCs(max_residuals=25 * u.mas)


if __name__ == "__main__":
    main()
