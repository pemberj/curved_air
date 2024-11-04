"""
This script uses numerical methods to calculate the wavelength-dependent
transmission of a Fabry-Pérot etalon. It compares the numerical results (using
both a plane-wave beam and a Gaussian beam) to the analytic description. The
plane-wave-illuminated etalon should give the same result as the analytic
version.
"""

import os
import sys
script_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(script_path, "../.."))

import numpy as np
from numpy.typing import ArrayLike

import astropy.units as u
from astropy.units import Quantity

from matplotlib import pyplot as plt

from curved_air.utils import FWHM
from curved_air.beams import PlanarBeam, GaussianBeam, SphericalBeam
from curved_air.optics import Etalon
from curved_air.plot_style import plot_style

plt.style.use(plot_style)


def main() -> None:

    mirror_spacing = 6 * u.mm #mm
    n = 1
    round_trip = 2 * mirror_spacing * n
    round_trip_mm = round_trip.to("mm").value
    num_round_trips = 30 #number of reflections on the OUTPUT surface
    mirror_reflectivity = 0.8
    f_number = 100
    num_wavelengths = 500
    z_0 = 0 * u.mm
    # z_0 = -1 * round_trip * num_round_trips
    z_0_mm = z_0.to("mm").value

    analytic_etalon = Etalon(
        l = mirror_spacing,
        n = n,
        θ = 0 * u.degree,
        R_1 = mirror_reflectivity,
        )

    reflectivity = np.array(
        [mirror_reflectivity**(n) for n in range(1, num_round_trips)]
        )

    Zs = np.arange(
        z_0_mm,
        z_0_mm + round_trip_mm * (num_round_trips-1),
        round_trip_mm
        ) * u.mm

    # Calculate central wavelength of peak closest to 780nm
    c = (mirror_spacing / int((mirror_spacing)/(500*u.nm))).to(u.nm).value
    w = (50 * u.pm).to(u.nm).value
    λs = np.linspace(c - w/2, c + w/2, num_wavelengths) * u.nm

    output = {}

    for λ in λs:
        
        # beam = GaussianBeam(
        #     λ=λ,
        #     E_0=1,
        #     n=1,
        #     w_0=4*u.mm,
        #     debug_level=1
        #     )
        # results = beam.summed_E_field(z=Zs) * reflectivity
        # intensity = np.real(np.sum(results) * np.conj(np.sum(results)))
        # if "Gaussian" not in output:
        #     output["Gaussian"] = []
        # output["Gaussian"].append(intensity)

        # Now the same for the planar beam (only evaluated on-axis)
        beam = PlanarBeam(
            λ=λ,
            E_0=1,
            n=1,
            plane_normal=(0,0,1),
            debug_level=0
            )
        results =\
            beam.E_field(x=np.zeros_like(Zs), y=np.zeros_like(Zs), z=Zs)\
                * reflectivity
        intensity = np.real(np.sum(results) * np.conj(np.sum(results)))
        if "Planar" not in output:
            output["Planar"] = []
        output["Planar"].append(intensity)
        
        beam = SphericalBeam(
            λ=λ,
            E_0=1,
            f_number=f_number,
            n=n,
            plane_normal=(0,0,1),
            debug_level=0
            )
        results = beam.summed_E_field(z=Zs, max_radius=10*u.mm)
        dimmed = results * reflectivity
        intensity = np.real(np.sum(dimmed) * np.conj(np.sum(dimmed)))
        if "Spherical" not in output:
            output["Spherical"] = []
        output["Spherical"].append(intensity)


    plt.rcParams["axes.formatter.useoffset"] = False

    fig = plt.figure()
    ax = fig.gca()

    distance_between_peaks = analytic_etalon.delta_lambda()
    analytic_intensity = analytic_etalon.intensity(λs)

    analytic_fwhm = np.diff(
        FWHM(λs.to("pm").value, np.real(analytic_intensity))
        )[0] * u.pm
    analytic_finesse = distance_between_peaks/analytic_fwhm

    ax.plot(λs, analytic_intensity / np.max(analytic_intensity),
            label="Analytic\nreflection-limited peak\n"+\
                 f"FWHM={analytic_fwhm:.4f}\nFinesse={analytic_finesse:.2f}")

    for _, [beam_type, values] in enumerate(output.items(), start=1):
        if beam_type in ["Planar", "Spherical"]:
            values -= np.min(values)
            values /= np.max(values)
        # values = 1 - values
        
        peak_fwhm = np.diff(FWHM(λs.to("pm").value, values))[0] * u.pm
        calculated_finesse = distance_between_peaks/peak_fwhm

        ax.plot(λs, values, ls="--",
                label=f"{beam_type}\nFWHM={peak_fwhm:.4f}\n"\
                     +f"Finesse={calculated_finesse:.2f}")

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalised intensity")
    #ax.set_yticklabels([])
    ax.set_xlim(min(λs).to("nm").value, max(λs).to("nm").value)
    # ax.set_ylim(-0.05, 1.1)
    ax.legend()

    plt.show()
    # plt.savefig(f"plots/{os.path.basename(__file__).split('.')[0]}.png")
    


if __name__ == "__main__":
    main()
