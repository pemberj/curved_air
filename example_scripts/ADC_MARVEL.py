"""
A script to optimise and plot a report on a rotational atmospheric dispersion corrector
for the visible range. These glasses are used for the ADCs for the MARVEL facility.
"""

import re
import sys
from pathlib import Path

from astropy import units as u
import matplotlib.pyplot as plt

from curved_air.glass import GlassCatalogue
from curved_air.optics import PrismStack, ADC
from curved_air.plot_style import plot_style

plt.style.use(plot_style)


script_path = Path(__file__).parent
sys.path.append(script_path / "../src")


def main() -> None:
    agf_dir: Path = script_path / "../data/AGFs"

    agf_files: list[Path] = sorted(
        [
            str(f)
            for f in agf_dir.glob("*.[Aa][Gg][Ff]")
            if re.search("ohara|schott", f.name)
        ]
    )

    if not agf_files:
        raise FileNotFoundError("No AGF files found in the data directory.")

    catalogue: GlassCatalogue = GlassCatalogue().read_AGF(agf_filename=agf_files)

    example_ADC: ADC = ADC(
        prism_stack=PrismStack([catalogue["K10"], catalogue["S-FPL51"]]),
        min_wl=390 * u.nm,
        max_wl=950 * u.nm,
        D_telescope=0.8 * u.m,
        D_ADC=15 * u.mm,
        z_max=75 * u.degree,
        criterion="PTV",
    )

    example_ADC.optimise()
    example_ADC.plot()


if __name__ == "__main__":
    main()
