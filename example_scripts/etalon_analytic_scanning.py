#! uv run

"""
Generate the transmission spectrum of an etalon using an analytic description, showing
the peak FWHM, FSR, finesse, and the effect of illuminating at different angles to scan
the peaks across a wavelength range.
"""

from typing import TYPE_CHECKING

import astropy.units as u
import numpy as np
from matplotlib import pyplot as plt

from curved_air.instruments.etalon import Etalon
from curved_air.plotting import plot_style

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from curved_air.type_hinting import ArrayQuantity, ScalarQuantity

plt.style.use(plot_style)


def main() -> None:
    etalon = Etalon(thickness=5 * u.mm)

    print(etalon)
    return

    fig: plt.Figure = plt.figure(figsize=(8, 4), dpi=300)
    ax: plt.Axes = fig.gca()

    central_wavelength: ScalarQuantity = 500 * u.nm

    c: float = etalon.thickness.to(u.nm).value / round(
        (etalon.thickness / central_wavelength).si.value
    )
    w: float = 0.06
    wavelengths: ArrayQuantity = np.linspace(c - w / 2, c + w / 2, 2501) * u.nm

    intensity: NDArray = np.real(etalon.intensity(wavelengths))

    zero = ax.plot(
        (wavelengths - central_wavelength).to("pm").value,
        intensity / np.max(intensity),
        lw=1.2,
    )

    max_angle = 0.6 * u.deg

    for angle in np.linspace(0, max_angle.to(u.deg).value, 16) * u.deg:
        _etalon = Etalon(thickness=5 * u.mm, theta=angle)
        shifted_intensity: NDArray = np.real(_etalon.intensity(wavelengths))
        ax.plot(
            (wavelengths - central_wavelength).to("pm").value,
            shifted_intensity / np.max(shifted_intensity),
            color=zero[0].get_color(),
            alpha=1 - (angle / max_angle).value,
            lw=1.2,
        )

    # ax.set_title("")
    ax.set_xlim(
        min(wavelengths - central_wavelength).to("pm").value,
        max(wavelengths - central_wavelength).to("pm").value,
    )
    ax.set_xlabel(f"Wavelength - {central_wavelength:.0f} (pm)")
    ax.set_ylabel("Normalised intensity")
    ax.set_ylim(-0.1, 1.2)

    plt.show()

    # plt.savefig(f"plots/{Path(__file__).name}.png")


if __name__ == "__main__":
    main()
