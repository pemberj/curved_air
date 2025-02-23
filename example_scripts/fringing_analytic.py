# /// script
# dependencies = [
#     "curved_air>=0.1.0",
# ]
# [tool.uv.sources]
# curved_air = { path = "..", editable = true }
# ///

"""
A script demonstrating the method of reading in one or more glass catalogues
from manufacturer-provided AGF files, filtering the catalogue, and outputting
this to a new AGF file for use in Zemax.
"""

import numpy as np
from astropy import units as u
import matplotlib.pyplot as plt

from curved_air.optics import Etalon
from curved_air.plotting import plot_style

plt.style.use(plot_style)


def main() -> None:
    BSC = Etalon(n=1.5, thickness=20 * u.mm, theta=0 * u.degree, R_1=0.02, R_2=0.02)

    print(BSC)

    fig = plt.figure()
    ax = fig.gca()

    wavelengths = np.linspace(499.99, 500.01, 200) * u.nm

    intensity = BSC.intensity(wavelengths)

    ax.plot(wavelengths, intensity / np.max(intensity))
    ax.axhline(y=1, color="k", linestyle="--", alpha=0.25)

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalised Intensity")
    ax.set_ylim(0, 1.1)
    ax.set_xlim(min(wavelengths).value, max(wavelengths).value)

    plt.show()
    # plt.savefig(f"plots/{os.path.basename(__file__).split('.')[0]}.png")


if __name__ == "__main__":
    main()
