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
from typing import TYPE_CHECKING

import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt

from curved_air.detectors import Detector
from curved_air.glass import GlassCatalogue
from curved_air.instruments.echelle import EchelleSpectrograph
from curved_air.optics.dispersers import EchelleGrating, Prism
from curved_air.plotting import plot_style, stroke, wavelength_to_rgb
from curved_air.type_hinting import ScalarQuantity

if TYPE_CHECKING:
    from curved_air.type_hinting import ArrayQuantity

plt.style.use(plot_style)

SCRIPT_PATH: Path = Path(__file__).parent


agf_dir: Path = SCRIPT_PATH / "../data/AGFs"

agf_files: list[Path] = [
    f
    for f in agf_dir.glob("*.[Aa][Gg][Ff]")
    if re.search(pattern="MARVEL_CAMERA", string=f.name)
]

if not agf_files:
    raise FileNotFoundError("No AGF files found in the data directory.")

ohara: GlassCatalogue = GlassCatalogue().read_AGF(agf_filename=agf_files)

MIN_WL: ScalarQuantity = 380 * u.nm
MAX_WL: ScalarQuantity = 930 * u.nm

NEID_GRATING: EchelleGrating = EchelleGrating(
    name="NEID Grating",
    catalogue_number="",
    groove_frequency=31.6 / u.mm,
    ruled_width=800 * u.mm,
    blaze_angle=76 * u.deg,
    gamma_angle=0.5 * u.deg,
    min_wl=MIN_WL,
    max_wl=MAX_WL,
)

NEID_PRISM: Prism = Prism(
    name="MARVEL Prism",
    glass=ohara["PBM2Y"],
    apex_angle=60 * u.deg,
    min_wl=MIN_WL,
    max_wl=MAX_WL,
)


def main() -> None:
    NEID: EchelleSpectrograph = EchelleSpectrograph(
        name="NEID Spectrograph",
        grating=NEID_GRATING,
        cross_disperser=NEID_PRISM,
        min_wl=MIN_WL,
        max_wl=MAX_WL,
        collimator_focal_length=1550 * u.mm,
        camera_focal_length=600 * u.mm,
        slit_width=40 * u.um,
        slit_length=0.25 * u.mm,
        slit_native_focal_ratio=3.85,
        detector=Detector(
            name="",
            nx=9216,
            ny=9231,
            pixel_size=10 * u.um,
        ),
    )

    NEID.print_parameters()

    fig: plt.Figure = plt.figure("NEID spectral format", figsize=(8, 8))
    ax: plt.Axes = fig.gca()

    NEID.plot_detector_edge(ax=ax)

    for m in NEID.spectral_orders:
        wls: ArrayQuantity = NEID.order_wavelengths(m=m)
        mean_wl: ScalarQuantity = np.mean(wls.value) * wls.unit
        # print(f"{m = }, mean_wl={mean_wl.to(u.nm):.1f}")

        edge_wls: ArrayQuantity = NEID.order_wavelengths(m=m, FSR_fraction=4)

        xs, ys = np.transpose(NEID.image_plane_position(m=m, wavelength=edge_wls))
        ax.plot(
            ys,
            xs,
            color=wavelength_to_rgb(mean_wl, fade_factor=0.2),
        )

        xs, ys = np.transpose(NEID.image_plane_position(m=m, wavelength=wls))
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
