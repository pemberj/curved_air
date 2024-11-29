"""
A script demonstrating the method of reading in one or more glass catalogues
from manufacturer-provided AGF files, filtering the catalogue, and outputting
this to a new AGF file for use in Zemax.
"""

import os
import sys
script_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(script_path, "../src"))

from glob import glob
import re

import numpy as np
from numpy.typing import ArrayLike

from astropy import units as u
from astropy.units import Quantity

import matplotlib.pyplot as plt

# from glass_select.glass import GlassCatalogue, _optimise_ADC
from curved_air.glass import GlassCatalogue

from curved_air.plot_style import plot_style
plt.style.use(plot_style)


def main() -> None:

    agf_files = sorted(
        [f for f in glob(pathname="../data/AGFs/*.[Aa][Gg][Ff]")\
                        if re.search(pattern="ohara|schott", string=f)])
    
    if not agf_files:
        raise FileNotFoundError("No AGF files found in the data directory.")
    
    gc = GlassCatalogue()\
        .read_AGF(agf_filename=agf_files)\
            .filter(
                    status_description = ["standard", "preferred"],
                    min_wl = 390 * u.nm,
                    max_wl = 950 * u.nm,
                    min_transmission = 0.97,
                    min_average_transmission = 0.99,
                )\
            .plot_Abbe_diagram()\
            .plot_transmission(wls = np.linspace(0.39, 0.45, 100) * u.um)\
            
    print(gc)

    # gc.write_AGF("./OUTPUT_CATALOGUE.AGF")


if __name__ == "__main__":

    main()
