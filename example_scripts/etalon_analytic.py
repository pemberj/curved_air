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

from typing import TYPE_CHECKING

import astropy.units as u
import numpy as np
from matplotlib import pyplot as plt

from curved_air.instruments.etalon import Etalon
from curved_air.plotting import plot_style
from curved_air.utils import FWHM

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from curved_air.type_hinting import ArrayQuantity, ScalarQuantity

plt.style.use(plot_style)


def main() -> None:
    etalon = Etalon()

    fig: plt.Figure = plt.figure()
    ax: plt.Axes = fig.gca()

    # Single peak:
    c: float = 6e-3 / 7692 * 1e9
    w: float = 0.01
    wavelengths: ArrayQuantity = np.linspace(c - w / 2, c + w / 2, 501) * u.nm

    print("Analytic finesse:", etalon.finesse)

    # Three peaks:
    # wavelengths: ArrayQuantity = np.linspace(499.975, 500.025, 501) * u.nm

    intensity: NDArray = np.real(etalon.intensity(wavelengths))
    peak_fwhm: ScalarQuantity = (
        np.diff(FWHM(wavelengths.to("pm").value, intensity))[0] * u.pm
    )

    ax.plot(
        wavelengths.to("pm").value,
        intensity / np.max(intensity),
        label=f"Analytic\nreflection-limited peak\nFWHM={peak_fwhm:.4f}",
    )

    print(
        "Finesse calculated from FWHM and Δλ: "
        + f"{etalon.delta_lambda(wavelengths[len(wavelengths) // 2]) / peak_fwhm}"
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
