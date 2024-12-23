"""
This script uses numerical methods to calculate the wavelength-dependent
transmission of a Fabry-Pérot etalon. It compares the numerical results (using
both a plane-wave beam and a Gaussian beam) to the analytic description. The
plane-wave-illuminated etalon should give the same result as the analytic
version.
"""

import sys
from pathlib import Path

import numpy as np
import astropy.units as u
from matplotlib import pyplot as plt

from curved_air.beams import PlanarBeam, SphericalBeam
from curved_air.optics import Etalon
from curved_air.plot_style import plot_style

plt.style.use(plot_style)


script_path: Path = Path(__file__).parent
sys.path.append(script_path / "../src")


def main() -> None:
    n = 1.5
    mirror_spacing = 20 * u.mm
    round_trip = 2 * mirror_spacing * n
    round_trip_mm = round_trip.to(u.mm).value
    num_round_trips = 20
    mirror_reflectivity = 0.02
    f_number = 10

    analytic_BSC = Etalon(
        n=n,
        thickness=mirror_spacing * n,
        theta=0 * u.degree,
        R_1=mirror_reflectivity,
        R_2=None,
    )

    reflectivity = np.array(
        [mirror_reflectivity ** (n) for n in range(1, num_round_trips)]
    )

    Zs = np.arange(0, round_trip_mm * (num_round_trips - 1), round_trip_mm) * u.mm

    c = 6e-3 / 7692 * 1e9
    w = 0.1
    wavelengths = np.linspace(c - w / 2, c + w / 2, 301) * u.nm

    # Three peaks:
    # wavelengths = np.linspace(499.975, 500.025, 500) * u.nm

    output = {}

    for wavelength in wavelengths:
        # beam = GaussianBeam(
        #     λ=wavelength,
        #     E_0=1,
        #     w_0=1*u.um,
        #     n=1.5,
        #     debug_level=0
        #     )
        # results = beam.summed_E_field(z=Zs)
        # dimmed = results * reflectivity
        # intensity = np.real(np.sum(dimmed) * np.conj(np.sum(dimmed)))
        # # I don't know why taking the square root here makes it work as expected
        # if "Gaussian" not in output:
        #     output["Gaussian"] = []
        # output["Gaussian"].append(intensity)

        # Now the same for the planar beam (only evaluated on-axis)
        beam = PlanarBeam(
            wavelength=wavelength, E_0=1, n=n, plane_normal=(0, 0, 1), debug_level=0
        )
        results = beam.E_field(x=np.zeros_like(Zs), y=np.zeros_like(Zs), z=Zs)
        dimmed = results * reflectivity
        intensity = np.real(np.sum(dimmed) * np.conj(np.sum(dimmed)))
        if "Planar" not in output:
            output["Planar"] = []
        output["Planar"].append(intensity)

        beam = SphericalBeam(
            wavelength=wavelength,
            E_0=1,
            f_number=f_number,
            n=n,
            plane_normal=(0, 0, 1),
            debug_level=0,
        )
        results = beam.summed_E_field(z=Zs)
        dimmed = results * reflectivity
        intensity = np.real(np.sum(dimmed) * np.conj(np.sum(dimmed)))
        if "Spherical" not in output:
            output["Spherical"] = []
        output["Spherical"].append(intensity)

    fig = plt.figure()
    ax = fig.gca()

    distance_between_peaks = analytic_BSC.delta_lambda()  # noqa: F841
    analytic_intensity = analytic_BSC.intensity(wavelengths)

    ax.plot(
        wavelengths,
        analytic_intensity,  # / np.max(analytic_intensity),
        label="Analytic\nreflection-limited peak",
    )

    for _, [beam_type, values] in enumerate(output.items(), start=1):
        # Why am I getting a zero-offset in the numerical ones?
        # Corresponds to the very first mirror reflection
        # values = (1 - np.sqrt(mirror_reflectivity)) + (1 - values)
        # values = values / np.max(values)
        norm_values = values / np.max(values)
        # values = np.array(values)

        ax.plot(wavelengths, norm_values, ls="--", label=f"{beam_type}")

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalised intensity")
    # ax.set_yticklabels([])

    # ax.set_xlim(min(λs).to("nm").value, max(λs).to("nm").value)
    # ax.set_ylim(-0.05, 1.1)

    ax.legend()

    plt.show()
    # plt.savefig(f"plots/{os.path.basename(__file__).split('.')[0]}.png")


if __name__ == "__main__":
    main()
