# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "curved-air>=0.1.0",
#     "rich>=10.0.0",
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
from curved_air.optics import EchelleSpectrograph, Grating, Prism
from curved_air.other import _atomic_lines
from curved_air.plotting import plot_style, stroke, wavelength_to_rgb

plt.style.use(plot_style)

SCRIPT_PATH = Path(__file__).parent


agf_dir = SCRIPT_PATH / "../data/AGFs"

agf_files = [
    f
    for f in agf_dir.glob("*.[Aa][Gg][Ff]")
    if re.search(pattern="schott", string=f.name)
]

if not agf_files:
    raise FileNotFoundError("No AGF files found in the data directory.")

schott = GlassCatalogue().read_AGF(agf_filename=agf_files)


def main() -> None:
    MARVEL = EchelleSpectrograph(
        name="MARVEL Spectrograph",
        grating=Grating(
            name="MARVEL Grating",
            catalogue_number="53-*-417E",
            groove_frequency=52.67 / u.mm,
            ruled_width=400 * u.mm,
            blaze_angle=69.7 * u.deg,
            gamma_angle=0.3 * u.deg,
        ),
        cross_disperser=Prism(
            name="MARVEL Prism",
            glass=schott["F2"],
            apex_angle=60 * u.deg,
        ),
        min_wl=380 * u.nm,
        max_wl=950 * u.nm,
        collimator_focal_length=1125 * u.mm,
        camera_focal_length=730 * u.mm,
        slit_width=40 * u.um,
        slit_length=0.5 * u.mm,
        slit_native_focal_ratio=3.85,
    )

    print(f"{MARVEL.slit_magnification = }")

    print(f"{MARVEL.slit_length * MARVEL.slit_magnification = }")
    print(f"{MARVEL.reddest_order_image_plane_separation = }")

    print(f"{MARVEL.slit_width = }")
    print(f"{MARVEL.slit_sampling = }")

    # assert MARVEL.slit_length < MARVEL.reddest_order_image_plane_separation

    fig = plt.figure("MARVEL spectral format", figsize=(12, 12))
    ax = fig.gca()

    MARVEL.plot_detector_edge(ax=ax)

    for i in range(1):  # number of fibres
        offset = i * 130 * u.um * MARVEL.slit_magnification
        for m in MARVEL.spectral_orders:
            wls = MARVEL.order_wavelengths(m=m)
            mean_wl = np.mean(wls)
            # print(f"{m = }, mean_wl={mean_wl.to(u.nm):.1f}")

            # edge_wls = MARVEL.order_wavelengths(m=m, FSR_fraction=3)

            # xs, ys = MARVEL.image_plane_position(m=m, wavelength=edge_wls)
            # ax.plot(xs + offset, ys, color=wavelength_to_rgb(mean_wl, fade_factor=0.3))

            xs, ys = MARVEL.image_plane_position(m=m, wavelength=wls)
            ax.plot(
                xs + offset,
                ys,
                color=wavelength_to_rgb(mean_wl, fade_factor=0.8),
                lw=3,
                path_effects=stroke(thickness=3.5),
            )

            min_FSR_wl, max_FSR_wl = MARVEL.grating.order_FSR_bounds(m=m)

            for name, wl in _atomic_lines.items():
                # for name, wl in _fraunhofer_lines.items():
                if min_FSR_wl <= wl <= max_FSR_wl:
                    x, y = MARVEL.image_plane_position(m=m, wavelength=wl)

                    ax.text(
                        x.value,
                        y.value,
                        f"{name}",
                        fontsize=12,
                        ha="center",
                        va="center",
                        bbox={
                            "facecolor": "w",
                            "edgecolor": "w",
                            "pad": 0.5,
                            "alpha": 0.8,
                        },
                    )

    ax.set_xlim(50, -50)
    ax.set_ylim(-50, 50)

    # ax.patch.set_facecolor("black")
    ax.axis("off")

    plt.show()


if __name__ == "__main__":
    main()
