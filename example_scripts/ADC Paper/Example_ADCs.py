"""
A script to optimise and plot a report on a rotational atmospheric dispersion
corrector for the visible range. These glasses are used for the ADCs for the
MARVEL facility.
"""

import os
import sys

script_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(script_path, "../../src"))

from glob import glob
import re


from astropy import units as u

import matplotlib.pyplot as plt

# from glass_select.glass import GlassCatalogue, _optimise_ADC
from curved_air.glass import GlassCatalogue
from curved_air.optics import PrismStack, ADC

from curved_air.plot_style import plot_style

plt.style.use(plot_style)


def main() -> None:
    agf_files = sorted(
        [
            f
            for f in glob(f"{script_path}/../../data/AGFs/*.[Aa][Gg][Ff]")
            if re.search("ohara|schott", f)
        ]
    )

    if not agf_files:
        raise FileNotFoundError("No AGF files found in the data directory.")

    catalogue = GlassCatalogue().read_AGF(agf_filename=agf_files)

    MARVEL_ADC = ADC(
        prism_stack=PrismStack([catalogue["K10"], catalogue["S-FPL51"]]),
        min_wl=390 * u.nm,
        max_wl=950 * u.nm,
        D_telescope=0.8 * u.m,
        D_ADC=15 * u.mm,
        z_max=75 * u.degree,
        criterion="PTV",
    )

    MARVEL_ADC.optimise()
    MARVEL_ADC.plot(save_filename=f"{script_path}/ADC_K10_FPL51.svg", show=False)

    example_ADC = ADC(
        prism_stack=PrismStack([catalogue["N-KZFS4HT"], catalogue["N-SK4"]]),
        min_wl=390 * u.nm,
        max_wl=950 * u.nm,
        D_telescope=0.8 * u.m,
        D_ADC=15 * u.mm,
        z_max=75 * u.degree,
        criterion="RMS",
    )

    example_ADC.optimise()
    example_ADC.plot(save_filename=f"{script_path}/ADC_KZFS4HT_SK4.svg", show=False)

    example_ADC = ADC(
        prism_stack=PrismStack([catalogue["N-LASF41"], catalogue["N-BAF10"]]),
        min_wl=390 * u.nm,
        max_wl=950 * u.nm,
        D_telescope=0.8 * u.m,
        D_ADC=15 * u.mm,
        z_max=75 * u.degree,
        criterion="RMS",
    )

    example_ADC.optimise()
    example_ADC.plot(save_filename=f"{script_path}/ADC_LASF41_BAF10.svg", show=False)


if __name__ == "__main__":
    main()
