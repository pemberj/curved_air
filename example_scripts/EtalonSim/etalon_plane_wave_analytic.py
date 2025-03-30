# /// script
# dependencies = [
#     "curved_air>=0.1.0",
# ]
# [tool.uv.sources]
# curved_air = { path = "../..", editable = true }
# ///

from pathlib import Path

import astropy.units as u
import numpy as np
from matplotlib import pyplot as plt

from curved_air.optics import Etalon
from curved_air.plotting import plot_style
from curved_air.utils import FWHM

plt.style.use(plot_style)

SCRIPT_NAME: Path = Path(__file__).stem
SCRIPT_PATH: Path = Path(__file__).parent


def main() -> None:
    etalon = Etalon()

    fig = plt.figure()
    ax = fig.gca()

    # Single peak:
    # wavelengths = np.linspace(499.998, 500.002, 500) * u.nm
    c = 6e-3 / 7692 * 1e9
    w = 0.005
    wavelengths = np.linspace(c - w / 2, c + w / 2, 501) * u.nm

    print("Analytic finesse:", etalon.finesse)

    # Three peaks:
    # wavelengths = np.linspace(499.975, 500.025, 200) * u.nm

    intensity = np.real(etalon.intensity(wavelengths))
    peak_fwhm = np.diff(FWHM(wavelengths.to("pm").value, intensity))[0] * u.pm

    ax.plot(
        wavelengths.to("pm").value,
        intensity / np.max(intensity),
        label="Analytic\nreflection-limited peak\n" + f"FWHM={peak_fwhm:.4f}",
    )

    print(
        "Finesse calculated from FWHM and $\\Delta\\lambda$:",
        etalon.delta_lambda() / peak_fwhm,
    )

    print(etalon)

    ax.legend()
    ax.set_title("Plane-wave illumination (analytic description)")
    ax.set_xlim(min(wavelengths).to("pm").value, max(wavelengths).to("pm").value)
    ax.set_xlabel("Wavelength (pm)")
    # ax.set_yticklabels([])
    ax.set_ylim(-0.1, 1.1)

    Path(SCRIPT_PATH / "plots").mkdir(parents=True, exist_ok=True)
    plt.savefig(SCRIPT_PATH / "plots" / f"{SCRIPT_NAME}.png")

    plt.show()


if __name__ == "__main__":
    main()
