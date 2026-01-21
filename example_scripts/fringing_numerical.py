# /// script
# dependencies = [
#     "curved_air>=0.1.0",
# ]
# [tool.uv.sources]
# curved_air = { path = "..", editable = true }
# ///

"""
This script uses numerical methods to calculate the wavelength-dependent
transmission of a Fabry-Pérot etalon. It compares the numerical results (using
both a plane-wave beam and a Gaussian beam) to the analytic description. The
plane-wave-illuminated etalon should give the same result as the analytic
version.
"""

from typing import TYPE_CHECKING

import astropy.units as u
import numpy as np
from matplotlib import pyplot as plt

from curved_air.beams import PlanarBeam, SphericalBeam
from curved_air.instruments.etalon import Etalon
from curved_air.plotting import plot_style

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from curved_air.type_hinting import ArrayQuantity, ScalarQuantity

plt.style.use(plot_style)


def main() -> None:
    n: float = 1.5
    mirror_spacing: ScalarQuantity = 20 * u.mm
    round_trip: ScalarQuantity = 2 * mirror_spacing * n
    round_trip_mm: float = round_trip.to(u.mm).value
    num_round_trips: int = 20
    mirror_reflectivity: float = 0.02
    f_number: float = 10

    analytic_BSC = Etalon(
        n=n,
        thickness=mirror_spacing * n,
        theta=0 * u.degree,
        R_1=mirror_reflectivity,
        R_2=None,
    )

    reflectivity: NDArray = np.array(
        [mirror_reflectivity ** (n) for n in range(1, num_round_trips)]
    )

    Zs: ArrayQuantity = (
        np.arange(0, round_trip_mm * (num_round_trips - 1), round_trip_mm) * u.mm
    )
    c: float = 6e-3 / 7692 * 1e9
    w: float = 0.1
    wavelengths: ArrayQuantity = np.linspace(c - w / 2, c + w / 2, 301) * u.nm

    # Three peaks:
    # wavelengths = np.linspace(499.975, 500.025, 500) * u.nm

    output: dict[str, list[NDArray]] = {}

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
        results: NDArray = beam.E_field(x=np.zeros_like(Zs), y=np.zeros_like(Zs), z=Zs)
        dimmed: NDArray = results * reflectivity
        intensity: float = np.real(np.sum(dimmed) * np.conj(np.sum(dimmed)))
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
        results: NDArray = np.array(beam.summed_E_field(z=Zs))
        dimmed: NDArray = results * reflectivity
        intensity: float = np.real(np.sum(dimmed) * np.conj(np.sum(dimmed)))
        if "Spherical" not in output:
            output["Spherical"] = []
        output["Spherical"].append(intensity)

    fig: plt.Figure = plt.figure(figsize=(8, 6))
    ax: plt.Axes = fig.gca()

    distance_between_peaks: ScalarQuantity = analytic_BSC.delta_lambda()  # noqa: F841
    analytic_intensity: float = analytic_BSC.intensity(wavelengths)

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
        norm_values: NDArray = values / np.max(values)
        # values = np.array(values)

        if beam_type == "Spherical":
            _label = f"Spherical beam at f/{f_number}"
        else:
            _label = f"{beam_type} beam"

        ax.plot(wavelengths, norm_values, ls="--", label=f"{beam_type}")

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalised intensity")
    # ax.set_yticklabels([])

    # ax.set_xlim(min(wavelengths).to("nm").value, max(wavelengths).to("nm").value)
    # ax.set_ylim(-0.05, 1.1)

    ax.legend()

    plt.show()
    # plt.savefig(f"plots/{os.path.basename(__file__).split('.')[0]}.png")


if __name__ == "__main__":
    main()
