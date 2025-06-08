# /// script
# dependencies = [
#     "curved_air>=0.1.0",
# ]
# [tool.uv.sources]
# curved_air = { path = "..", editable = true }
# ///

import re
from pathlib import Path

import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt

from curved_air.glass import GlassCatalogue
from curved_air.optics import Prism
from curved_air.plotting import plot_style, stroke, wavelength_to_rgb  # noqa: F401

plt.style.use(plot_style)

SCRIPT_PATH = Path(__file__).parent


agf_dir = SCRIPT_PATH / "../data/AGFs"

agf_files = sorted(
    [
        f
        for f in agf_dir.glob("*.[Aa][Gg][Ff]")
        if re.search(pattern="schott", string=f.name)
    ]
)
if not agf_files:
    raise FileNotFoundError("No AGF files found in the data directory.")

gc = GlassCatalogue().read_AGF(agf_filename=agf_files)

F2 = gc["F2"]


def main() -> None:
    """ """

    prism = Prism(
        glass=F2,
        apex_angle=60 * u.deg,
        min_wl=390 * u.nm,
        max_wl=950 * u.nm,
    )

    fig = plt.figure()
    ax = fig.gca()

    incidence_angles = np.linspace(40, 75, 100) * u.deg

    wl1 = 500 * u.nm
    wl2 = 550 * u.nm

    angles1 = prism.deviation_angle(wl=wl1, incidence_angle=incidence_angles)
    angles2 = prism.deviation_angle(wl=wl2, incidence_angle=incidence_angles)
    dispersion = (angles2 - angles1) / (wl2 - wl1)

    ax.plot(
        incidence_angles,
        dispersion,
        label="Prism deviation angle",
    )

    ax.axvline(
        x=prism.incidence_angle.value,
        color="k",
        linestyle="--",
        label="Apex angle",
    )

    ax.set_xlabel("Incidence angle (deg)")
    ax.set_ylabel("Angular dispersion (deg / nm)")

    ax.legend()

    plt.show()


if __name__ == "__main__":
    main()
