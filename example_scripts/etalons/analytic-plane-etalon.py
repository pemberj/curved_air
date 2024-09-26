"""


"""


import os
import sys
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../..")
    ))

import numpy as np
import astropy.units as u

from matplotlib import pyplot as plt
from matplotlib import font_manager as fm

from curved_air.optics import Etalon
from curved_air.utils import FWHM
from curved_air.plot_style import plot_style

plt.style.use(plot_style)


def main():

    etalon = Etalon()

    fig = plt.figure()
    ax = fig.gca()

    # Single peak:
    # λs = np.linspace(499.998, 500.002, 500) * u.nm
    c = 6e-3 / 7692 * 1e9
    w = 0.005
    λs = np.linspace(c - w/2, c + w/2, 501) * u.nm

    print("Analytic finesse:", etalon.finesse)

    # Three peaks:
    # λs = np.linspace(499.975, 500.025, 200) * u.nm

    intensity = np.real(etalon.intensity(λs))
    peak_fwhm = np.diff(FWHM(λs.to("pm").value, intensity))[0] * u.pm

    ax.plot(λs.to("pm").value, intensity / np.max(intensity),
            label = f"Analytic\nreflection-limited peak\nFWHM={peak_fwhm:.4f}")

    print(
        "Finesse calculated from FWHM and Δλ: "+\
         f"{etalon.delta_lambda() / peak_fwhm}"
         )

    etalon.R
    etalon.T
    etalon.F_coef
    etalon.finesse
    etalon.R_max

    print(etalon)
    
    print(repr(etalon))
    
    ax.legend()
    ax.set_title("Plane-wave illumination (analytic description)")
    ax.set_xlim(min(λs).to("pm").value, max(λs).to("pm").value)
    ax.set_xlabel("Wavelength (pm)")
    #ax.set_yticklabels([])
    ax.set_ylim(-0.1, 1.1)
    
    plt.show()

    # plt.savefig(f"plots/{os.path.basename(__file__).split('.')[0]}.png")



if __name__ == "__main__":
    main()
