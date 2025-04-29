# /// script
# dependencies = [
#     "curved_air>=0.1.0",
# ]
# [tool.uv.sources]
# curved_air = { path = "../..", editable = true }
# ///

from itertools import cycle
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
    fig = plt.figure(figsize=(8, 5))
    ax = fig.gca()

    plt.rcParams["axes.autolimit_mode"] = "data"

    # Three peaks:
    wavelengths = np.linspace(499.975, 500.025, 1000) * u.nm

    reflectivities = [0.9, 0.6, 0.32, 0.172]
    line_styles = cycle(["-", "--", "-.", ":"])

    for R in reflectivities:
        etalon = Etalon(R_1=R)
        intensity = np.real(etalon.intensity(wavelengths))
        ax.plot(
            wavelengths.to("pm").value,
            intensity,
            ls=next(line_styles),
            label=f"$R_1 = R_2 = {R:.2f}$",
        )

    # ax.axhline(y=0.5, color="k", ls="--", alpha=0.25)

    ax.legend(frameon=True, ncol=4, fontsize=12, loc="upper center")
    # ax.set_title("Analytic plane-wave illuminated etalon spectrum")
    ax.set_xlim(min(wavelengths).to("pm").value, max(wavelengths).to("pm").value)
    ax.set_ylim(-0.025, 1.15)

    ax.set_ylabel("Relative transmitted inttensity")
    ax.set_xlabel("Wavelength (pm)")

    Path(SCRIPT_PATH / "plots").mkdir(parents=True, exist_ok=True)
    plt.savefig(SCRIPT_PATH / "plots" / f"{SCRIPT_NAME}.png")


if __name__ == "__main__":
    main()
