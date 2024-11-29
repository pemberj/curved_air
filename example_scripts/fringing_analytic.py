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

from curved_air.optics import Etalon

from curved_air.plot_style import plot_style
plt.style.use(plot_style)


def main():

    BSC = Etalon(n=1.5, l=20*u.mm, θ=0*u.degree, R_1=0.02, R_2=0.02)

    print(BSC)

    fig = plt.figure()
    ax = fig.gca()

    λs = np.linspace(499.99, 500.01, 200) * u.nm

    intensity = BSC.intensity(λs)

    ax.plot(λs, intensity / np.max(intensity))
    ax.axhline(y=1, color="k", linestyle="--", alpha=0.25)

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalised Intensity")
    ax.set_ylim(0, 1.1)
    ax.set_xlim(min(λs).value, max(λs).value)
    
    plt.show()
    # plt.savefig(f"plots/{os.path.basename(__file__).split('.')[0]}.png")
    


if __name__ == "__main__":
    main()
