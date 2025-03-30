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

from curved_air.beams import GaussianBeam, PlanarBeam, SphericalBeam  # noqa: F401
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
    num_round_trips = 3000  # number of reflections on the OUTPUT surface
    mirror_reflectivity = 0.95

    analytic_etalon = Etalon(thickness=mirror_spacing, R_1=mirror_reflectivity)

    reflectivity = np.array(
        [mirror_reflectivity ** (n) for n in range(1, num_round_trips)]
    )

    Zs = np.arange(0, round_trip_mm * (num_round_trips - 1), round_trip_mm) * u.mm
    # two same-length placeholder arrays for the plane-wave version
    Xs = np.zeros_like(Zs)
    Ys = np.zeros_like(Zs)

    c = 6e-3 / 7692 * 1e9
    w = 0.01
    wavelengths = np.linspace(c - w / 2, c + w / 2, 301) * u.nm

    # Three peaks:
    # wavelengths = np.linspace(499.975, 500.025, 200) * u.nm

    output = {"gaussian": [], "planar": []}

    for wavelength in wavelengths:
        beam = GaussianBeam(
            wavelength=wavelength,
            E_0=1,
            w_0=4 * u.mm,
            n=1,
            debug_level=0,
        )
        results = beam.summed_E_field(z=Zs)
        dimmed = results * reflectivity
        intensity = np.real(np.sum(dimmed) * np.conj(np.sum(dimmed)))
        output["gaussian"].append(
            intensity**0.5
        )  # I don't know why taking the square root here makes it work as expected!

        # Now the same for the planar beam (only evaluated on-axis)
        beam = PlanarBeam(wavelength=wavelength, plane_normal=(0, 0, 1))
        results = np.real(beam.E_field(x=Xs, y=Ys, z=Zs))
        dimmed = results * reflectivity
        intensity = np.real(np.sum(dimmed) * np.conj(np.sum(dimmed)))
        output["planar"].append(intensity**0.5)

    fig = plt.figure()
    ax = fig.gca()

    distance_between_peaks = analytic_etalon.delta_lambda()
    analytic_intensity = analytic_etalon.intensity(wavelengths)

    analytic_fwhm = (
        np.diff(FWHM(wavelengths.to("pm").value, np.real(analytic_intensity)))[0] * u.pm
    )
    analytic_finesse = distance_between_peaks / analytic_fwhm

    ax.plot(
        wavelengths,
        analytic_intensity / np.max(analytic_intensity),
        label=f"Analytic\nreflection-limited peak\nFWHM={analytic_fwhm:.4f}\nFinesse={analytic_finesse:.2f}",
    )

    for _, [beam_type, values] in enumerate(output.items()):
        # Why am I getting a zero-offset in the numerical ones?
        # Corresponds to the very first mirror reflection
        values = values / np.max(values) - (1 - np.sqrt(mirror_reflectivity))  # noqa: PLW2901

        peak_fwhm = np.diff(FWHM(wavelengths.to("pm").value, values))[0] * u.pm
        calculated_finesse = distance_between_peaks / peak_fwhm

        ax.plot(
            wavelengths,
            values / np.max(values),
            ls="--",
            label=f"{beam_type}\nFWHM={peak_fwhm:.4f}\n"
            + f"Finesse={calculated_finesse:.2f}",
        )

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalised intensity")
    # ax.set_yticklabels([])
    ax.set_xlim(min(wavelengths).to("nm").value, max(wavelengths).to("nm").value)
    ax.set_ylim(-0.05, 1.1)
    ax.legend()

    Path(SCRIPT_PATH / "plots").mkdir(parents=True, exist_ok=True)
    plt.savefig(SCRIPT_PATH / "plots" / f"{SCRIPT_NAME}.png")

    plt.show()


if __name__ == "__main__":
    main()
