"""
This script calculates the required height of the transfer mirror in an echelle
spectrograph as a function of the collimator parameters and the parameters of
the echelle grating.
"""

import os
import sys
script_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(script_path, "../src"))

import numpy as np
from numpy.typing import ArrayLike

import astropy.units as u
from astropy.units import Quantity

from matplotlib import pyplot as plt

from curved_air.optics import Grating
from curved_air.utils import calculate_transfer_mirror_height


def main() -> None:
    
    fig = plt.figure("Transfer mirror height", figsize=(10 ,8))
    ax = fig.gca()
    
    combinations = ([31.6, 31.6, 52.67, 13.33], [63.5, 76, 69.7, 80.6])
    
    for gpmm, blaze in np.transpose(combinations):
            fls = np.linspace(400, 1200, 20)
            heights = []
            for fl in fls:
    
                grating = Grating(
                    groove_frequency = gpmm / u.mm,
                    blaze_angle = blaze * u.deg,
                    gamma_angle = 0.35 * u.deg,
                    min_wl = 390 * u.nm,
                    max_wl = 850 * u.nm
                    )
            
                image_height =\
                    calculate_transfer_mirror_height(grating, fl * u.mm)
                    
                heights.append(image_height.to(u.mm).value)
            ax.plot(fls, heights, label=f"{gpmm:.4g} g/mm, R{np.tan(blaze * u.deg):.2g} ($\\theta_B$ = {blaze:.3g}°)")
                    
                # print(f"{gpmm = :.1f}\t{blaze = :.1f}\t{fl = :.0f}\t{image_height = :.3f}")

    ax.legend()
    ax.set_title("Transfer mirror height as a function of collimator focal length\n($\lambda_{max}$ = "+f"{grating.max_wl:.0f})")
    ax.set_xlabel("Collimator focal length (mm)")
    ax.set_ylabel("Required transfer mirror height (mm)")
    ax.set_ylim(0, ax.get_ylim()[1])
    plt.show()



if __name__ == "__main__":
    main()
