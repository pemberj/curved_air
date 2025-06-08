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

from curved_air.beams import GaussianBeam
from curved_air.plotting import plot_style

plt.style.use(plot_style)

SCRIPT_NAME: Path = Path(__file__).stem
SCRIPT_PATH: Path = Path(__file__).parent


def main() -> None:
    """
    mirror_spacing = 6*u.mm #mm
    round_trip = 2 * mirror_spacing
    round_trip_mm = round_trip.to("mm").value
    num_round_trips = 100 #number of reflections on the OUTPUT surface
    mirror_reflectivity = 0.95

    r = 5 * u.mm # maximum radius of beam

    w_0 = 4 * u.mm
    """

    mirror_spacing = 6 * u.mm
    num_round_trips = 100
    wavelength = mirror_spacing / int(mirror_spacing / (780 * u.nm))

    waist_radii = np.linspace(0.005, 15, 200) * u.mm

    z_Rs = []
    R_halfs = []  # beam wavefront radius in middle of etalon
    w_halfs = []
    Gouy_halfs = []
    radial_halfs = []
    amp_halfs = []
    for w_0 in waist_radii:
        beam = GaussianBeam(wavelength=wavelength, w_0=w_0)
        z_Rs.append(beam.z_R.si.value)
        R_halfs.append(beam.R(z=mirror_spacing * num_round_trips / 2).si.value)
        w_halfs.append(beam.w(z=mirror_spacing * num_round_trips / 2).si.value)
        Gouy_halfs.append(beam.Gouy_phase(z=mirror_spacing * num_round_trips / 2))
        radial_halfs.append(
            beam.radial_phase(r=5 * u.mm, z=mirror_spacing * num_round_trips / 2)
        )
        amp_halfs.append(
            beam.amplitude(r=5 * u.mm, z=mirror_spacing * num_round_trips / 2)
        )

    fig = plt.figure()
    ax = fig.gca()

    ax.plot(waist_radii.value, z_Rs / max(z_Rs), label="$z_R$, Rayleigh length")
    ax.plot(
        waist_radii.value,
        R_halfs / max(R_halfs),
        label="R($z_{mid}$), radius of wavefront curvature",
    )
    ax.plot(
        waist_radii.value, w_halfs / max(w_halfs), label="w($z_{mid}$), beam radius"
    )
    ax.plot(
        waist_radii.value,
        Gouy_halfs / max(Gouy_halfs),
        label="$\\Psi$($z_{mid}$), Gouy phase",
    )
    ax.plot(
        waist_radii.value,
        radial_halfs / max(radial_halfs),
        label="radial($z_{mid}$), radial phase @ r=5mm",
    )
    ax.plot(
        waist_radii.value,
        amp_halfs / max(amp_halfs),
        label="amp($z_{mid}$), E-field amplitude @ r=5mm",
    )

    ax.legend()

    # ax.set_title("Normalised beam properties as a function of waist radius $w_0$")
    ax.set_xlabel("Beam waist $w_0$")
    ax.set_ylabel("Normalised value")

    ax.set_xlim(0, max(waist_radii).value)
    ax.set_ylim(-0.05, 1.5)

    Path(SCRIPT_PATH / "plots").mkdir(parents=True, exist_ok=True)
    plt.savefig(SCRIPT_PATH / "plots" / f"{SCRIPT_NAME}.png")


if __name__ == "__main__":
    main()
