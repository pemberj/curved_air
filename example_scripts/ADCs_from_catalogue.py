"""

"""

import os
import sys
script_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(script_path, "../src"))

import re
from glob import glob

import numpy as np
from numpy.typing import ArrayLike

import astropy.units as u
from astropy.units import Quantity

from matplotlib import pyplot as plt

from curved_air.glass import GlassCatalogue
from curved_air.plot_style import plot_style

plt.style.use(plot_style)
 
 
 
 
 
def main() -> None: 
    agf_files = sorted(
            [
                f for f in glob("../data/AGFs/*.[Aa][Gg][Ff]")\
                                                if re.search("ohara|schott", f)
            ]
        )
    
    if not agf_files:
        raise FileNotFoundError("No AGF files found in ../data/AGFs")
    
    print(agf_files)

    catalogue = GlassCatalogue()\
        .read_AGF(agf_filename = agf_files)\
        .filter(
            min_wl = 380*u.nm,
            max_wl = 950*u.nm,
            min_transmission = 0.97,
            min_average_transmission = 0.99,
            
            )
    
    # catalogue.plot_Abbe_diagram()
    
    catalogue.compute_ADCs(max_residuals=25 * u.mas)
    
    
if __name__ == "__main__":
    main()
