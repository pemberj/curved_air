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
from curved_air.plotting import plot_style, stroke, wavelength_to_rgb

plt.style.use(plot_style)

SCRIPT_PATH = Path(__file__).parent


agf_dir = SCRIPT_PATH / "../data/AGFs"

agf_files = [
    f
    for f in agf_dir.glob("*.[Aa][Gg][Ff]")
    if re.search(pattern="MARVEL_CAMERA", string=f.name)
]

if not agf_files:
    raise FileNotFoundError("No AGF files found in the data directory.")

ohara = GlassCatalogue().read_AGF(agf_filename=agf_files)

NEID_grating = Grating(
    name="NEID Grating",
    catalogue_number="",
    groove_frequency=31.6 / u.mm,
    ruled_width=800 * u.mm,
    blaze_angle=76 * u.deg,
    gamma_angle=0.5 * u.deg,
)


def main() -> None:
    NEID = EchelleSpectrograph(
        name="NEID Spectrograph",
        grating=NEID_grating,
        cross_disperser=Prism(
            name="MARVEL Prism",
            glass=ohara["PBM2Y"],
            apex_angle=60 * u.deg,
        ),
        min_wl=380 * u.nm,
        max_wl=930 * u.nm,
        collimator_focal_length=1550 * u.mm,
        camera_focal_length=600 * u.mm,
        slit_width=40 * u.um,
        slit_length=0.25 * u.mm,
        slit_native_focal_ratio=3.85,
        detector_nx=9216,
        detector_ny=9231,
        detector_pixel_size=10 * u.um,
    )

    print(f"{NEID.slit_magnification = }")

    print(f"{NEID.slit_length * NEID.slit_magnification = }")
    print(f"{NEID.reddest_order_image_plane_separation = }")

    print(f"{NEID.slit_width = }")
    print(f"{NEID.slit_sampling = }")

    # assert NEID.slit_length < NEID.reddest_order_image_plane_separation

    fig = plt.figure("NEID spectral format", figsize=(8, 8))
    ax = fig.gca()

    NEID.plot_detector_edge(ax=ax)

    for m in NEID.spectral_orders:
        wls = NEID.order_wavelengths(m=m)
        mean_wl = np.mean(wls)
        # print(f"{m = }, mean_wl={mean_wl.to(u.nm):.1f}")

        edge_wls = NEID.order_wavelengths(m=m, FSR_fraction=4)

        xs, ys = NEID.image_plane_position(m=m, wavelength=edge_wls)
        ax.plot(ys, xs, color=wavelength_to_rgb(mean_wl, fade_factor=0.2))

        xs, ys = NEID.image_plane_position(m=m, wavelength=wls)
        ax.plot(
            ys,
            xs,
            color=wavelength_to_rgb(mean_wl, fade_factor=0.8),
            lw=4,
            path_effects=stroke(thickness=5),
        )

    ax.set_xlim(-50, 50)
    ax.set_ylim(50, -50)

    # ax.patch.set_facecolor("black")
    ax.axis("off")

    plt.show()


if __name__ == "__main__":
    main()
