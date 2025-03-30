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

plt.style.use(plot_style)

SCRIPT_NAME: Path = Path(__file__).stem
SCRIPT_PATH: Path = Path(__file__).parent


def main() -> None:
    etalon = Etalon(
        n=1.5,
        thickness=25.4 * u.mm,
        theta=0 * u.degree,
        R_1=0.02,
        R_2=None,
    )

    print(etalon.finesse)

    fig = plt.figure()
    ax = fig.gca()

    wavelengths = np.linspace(499.99, 500.01, 200) * u.nm

    intensity = etalon.intensity(wavelengths)

    ax.plot(wavelengths, intensity / np.max(intensity))

    ax.set_title("Plane-wave illumination")
    ax.set_xlabel("Wavelength (nm)")
    # ax.set_yticklabels([])
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlim(min(wavelengths).value, max(wavelengths).value)

    Path(SCRIPT_PATH / "plots").mkdir(parents=True, exist_ok=True)
    plt.savefig(SCRIPT_PATH / "plots" / f"{SCRIPT_NAME}.png")

    plt.show()


if __name__ == "__main__":
    main()
