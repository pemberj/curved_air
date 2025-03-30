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

from curved_air.beams import SphericalBeam
from curved_air.plotting import plot_style

if TYPE_CHECKING:
    from astropy.units import Quantity

plt.style.use(plot_style)

SCRIPT_NAME: Path = Path(__file__).stem
SCRIPT_PATH: Path = Path(__file__).parent


def main() -> None:
    f_number = 10

    mirror_spacing: Quantity = 10 * u.um  # Glue gap thickness in beamsplitter cube
    mirror_reflectivity = 0.01  # Calculated reflectivity due to refractive
    # index mismatch between the glue and the glass
    num_round_trips = 11

    reflectivity = np.array([mirror_reflectivity**i for i in range(1, num_round_trips)])
    baseline_transmission = 1 - np.sum(reflectivity)
    print(f"{baseline_transmission = }")

    # Work out where to sample the beam - starting from the back surface of the BSC
    backfocus = 60.5 * u.mm
    start_z = -(backfocus + mirror_spacing)
    stop_z = start_z + (2 * mirror_spacing) * (num_round_trips - 1)

    # numpy doesn't enjoy working with astropy.units sometimes...
    Zs = (
        np.linspace(
            start_z.to("mm").value, stop_z.to("mm").value, (num_round_trips - 1)
        )
        * u.mm
    )

    wavelengths = np.linspace(499.9, 500.1, 100) * u.nm

    output = []

    for wavelength in wavelengths:
        beam = SphericalBeam(
            wavelength=wavelength,
            E_0=1,
            n=1,
            plane_normal=(0, 0, 1),
            f_number=f_number,
            debug_level=0,
        )

        results = beam.summed_E_field(z=Zs, max_radius=2 * u.mm)

        # Multiply by the reflectivity attenuation
        dimmed = np.array(
            [result * R for result, R in zip(results, reflectivity, strict=False)]
        )

        # Get the intensity of the beam (I = E * E†)
        intensity = np.real(np.sum(dimmed) * np.conj(np.sum(dimmed)))

        output.append(intensity)

    output = np.array(output)

    fig = plt.figure()
    ax = fig.gca()

    # Don't use round numbers for this plot
    plt.rcParams["axes.autolimit_mode"] = "data"

    transmission = baseline_transmission - output
    ax.plot(wavelengths, transmission, label="")

    # ax.set_title("Beamsplitter cube etalon, gaussian beam illumination")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Relative transmission")
    # ax.set_yticklabels([])
    ax.set_ylim(min(transmission), max(transmission))
    ax.set_xlim(min(wavelengths).value, max(wavelengths).value)
    # ax.legend()

    Path(SCRIPT_PATH / "plots").mkdir(parents=True, exist_ok=True)
    plt.savefig(SCRIPT_PATH / "plots" / f"{SCRIPT_NAME}.png")

    plt.show()


if __name__ == "__main__":
    main()
