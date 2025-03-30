# /// script
# dependencies = [
#     "curved_air>=0.1.0",
# ]
# [tool.uv.sources]
# curved_air = { path = "../..", editable = true }
# ///


"""
A script to generate the expected etalon effect arising from internal reflections in an
antireflection-coated beamsplitter cube.

Here the definition of a Gaussian beam is used to model a beam focused by a telescope.
In this (f/6) beam is a beamsplitter cube used to examine both the telescope's field
of view and to look at an optical fiber in the telescope's prime focus.

Due to the low reflectivity inside the cube, and the ever-changing curvature of the
beam's wavefronts at successive round-trip reflections, the effect is miniscule compared
to the same beamsplitter cube placed in a collimated (~plane-wave) beam.
"""

from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.units import Quantity
from matplotlib import pyplot as plt

from curved_air.beams import GaussianBeam
from curved_air.plotting import plot_style

plt.style.use(plot_style)

SCRIPT_NAME: Path = Path(__file__).stem
SCRIPT_PATH: Path = Path(__file__).parent


def beam_waist(
    wavelength: Quantity = 0.5,
    f_number: float = 6,
    n: float = 1.5,
) -> Quantity:
    """
    Calculate the beam waist (w_0) for a beam with an input f-number, far away from the
    beam waist (i.e. at z >> z_R)

    Args:
        lam (float, optional): [description]. Defaults to 0.5.
        f_number (int, optional): [description]. Defaults to 6.
    """

    return (wavelength / (np.pi * n * np.arcsin(1 / (2 * n * f_number)))).to("um")


def main() -> None:
    # I want to define the Gaussian beam for each wavelength based on an f-number rather than the
    # beam waist. The beam waist for a beam of the same f/# is different for each wavelength.
    f_number = 10

    mirror_spacing = 25.4 * u.mm  # 1 inch beamsplitter cube
    num_round_trips = (
        50  # even 3 reflections gets very low power left indeed, here I go overboard
    )
    mirror_reflectivity = (
        0.02  # 2% reflection from antireflection-coated outer cube surface
    )

    reflectivity = np.array([mirror_reflectivity**i for i in range(1, num_round_trips)])
    baseline_transmission = 1 - np.sum(reflectivity)

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

    wavelengths = np.linspace(499.99, 500.01, 1_000) * u.nm

    output = []

    for wavelength in wavelengths:
        # Create a GaussianBeam for a given wavelength
        beam = GaussianBeam(
            wavelength=wavelength,
            E_0=1,
            w_0=beam_waist(wavelength=wavelength, f_number=f_number, n=1.5),
            n=1,
            debug_level=0,
        )

        results = beam.summed_E_field(z=Zs)

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
