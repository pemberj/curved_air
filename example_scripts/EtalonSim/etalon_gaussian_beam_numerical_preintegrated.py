# /// script
# dependencies = [
#     "curved_air>=0.1.0",
# ]
# [tool.uv.sources]
# curved_air = { path = "../..", editable = true }
# ///

from pathlib import Path
from typing import TYPE_CHECKING

import astropy.units as u
import numpy as np
from matplotlib import pyplot as plt

from curved_air.beams import GaussianBeam
from curved_air.optics import Etalon
from curved_air.plotting import plot_style
from curved_air.utils import FWHM

if TYPE_CHECKING:
    from astropy.units import Quantity

plt.style.use(plot_style)

SCRIPT_NAME: Path = Path(__file__).stem
SCRIPT_PATH: Path = Path(__file__).parent


def main() -> None:
    mirror_spacing: Quantity = 6 * u.mm  # mm
    round_trip = 2 * mirror_spacing
    round_trip_mm = round_trip.to("mm").value
    num_round_trips = 150  # number of reflections on the OUTPUT surface
    mirror_reflectivity = 0.95

    reflectivity = np.array(
        [mirror_reflectivity ** (n) for n in range(1, num_round_trips)]
    )

    Zs = np.arange(0, round_trip_mm * (num_round_trips - 1), round_trip_mm) * u.mm

    # Single peak:
    c = 6e-3 / 7692 * 1e9
    w = 0.005
    wavelengths = np.linspace(c - w / 2, c + w / 2, 201) * u.nm

    output = []

    for wl in wavelengths:
        beam = GaussianBeam(wavelength=wl, E_0=1, w_0=2 * u.mm, n=1, debug_level=0)

        results = beam.summed_E_field(z=Zs, exclude="none")

        dimmed = np.array(
            [result * R for result, R in zip(results, reflectivity, strict=False)]
        )
        intensity = np.real(np.sum(dimmed) * np.conj(np.sum(dimmed)))
        # Need sqrt to make it give the rigth answer - not sure why
        output.append(intensity**0.5)

    fig = plt.figure()
    ax = fig.gca()

    analytic_etalon = Etalon()
    distance_between_peaks = analytic_etalon.delta_lambda()

    values = output / np.max(output) - (1 - np.sqrt(mirror_reflectivity))

    peak_fwhm = np.diff(FWHM(wavelengths.to("pm").value, values))[0] * u.pm
    calculated_finesse = distance_between_peaks / peak_fwhm

    ax.plot(
        wavelengths.to(u.nm).value,
        values,
        label=f"Calculated finesse: {calculated_finesse:.1f}",
    )
    ax.set_title("Gaussian beam illumination")
    ax.set_xlabel("Wavelength (nm)")
    # ax.set_yticklabels([])
    ax.set_xlim(min(wavelengths).to(u.nm).value, max(wavelengths).to(u.nm).value)
    ax.set_ylim(0, 1)

    ax.legend()

    Path(SCRIPT_PATH / "plots").mkdir(exist_ok=True)
    plt.savefig(SCRIPT_PATH / "plots" / f"{SCRIPT_NAME}.png")

    plt.show()


if __name__ == "__main__":
    main()
