# /// script
# dependencies = [
#     "curved_air>=0.1.0",
# ]
# [tool.uv.sources]
# curved_air = { path = "..", editable = true }
# ///

"""
This script uses the analytic description of a plane-wave-illuminated etalon and
calculates its wavelength-dependent transmission.
"""

import numpy as np
import astropy.units as u
from matplotlib import pyplot as plt

from curved_air.optics import Etalon
from curved_air.utils import FWHM
from curved_air.plotting import plot_style

plt.style.use(plot_style)


def main() -> None:
    etalon = Etalon()

    fig = plt.figure()
    ax = fig.gca()

    # Single peak:
    # λs = np.linspace(499.998, 500.002, 500) * u.nm
    c = 6e-3 / 7692 * 1e9
    w = 0.01
    wavelengths = np.linspace(c - w / 2, c + w / 2, 501) * u.nm

    print("Analytic finesse:", etalon.finesse)

    # Three peaks:
    # wavelengths = np.linspace(499.975, 500.025, 200) * u.nm

    intensity = np.real(etalon.intensity(wavelengths))
    peak_fwhm = np.diff(FWHM(wavelengths.to("pm").value, intensity))[0] * u.pm

    ax.plot(
        wavelengths.to("pm").value,
        intensity / np.max(intensity),
        label=f"Analytic\nreflection-limited peak\nFWHM={peak_fwhm:.4f}",
    )

    print(
        "Finesse calculated from FWHM and Δλ: " + f"{etalon.delta_lambda() / peak_fwhm}"
    )

    print(etalon)

    print(repr(etalon))

    ax.legend()
    ax.set_title("Plane-wave illumination (analytic description)")
    ax.set_xlim(min(wavelengths).to("pm").value, max(wavelengths).to("pm").value)
    ax.set_xlabel("Wavelength (pm)")
    # ax.set_yticklabels([])
    ax.set_ylim(-0.1, 1.1)

    plt.show()

    # plt.savefig(f"plots/{Path(__file__).name}.png")


if __name__ == "__main__":
    main()
