# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "curved-air>=0.2.0",
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
from curved_air.type_hinting import ScalarQuantity

if TYPE_CHECKING:
    from astropy.units import Quantity

from curved_air.glass import GlassCatalogue
from curved_air.instruments.echelle import DoublePassEchelleSpectrograph
from curved_air.optics.dispersers import EchelleGrating, Prism
from curved_air.other import _atomic_lines
from curved_air.plotting import plot_style, stroke, wavelength_to_rgb  # noqa: F401

plt.style.use(plot_style)

SCRIPT_PATH: Path = Path(__file__).parent


agf_dir: Path = SCRIPT_PATH / "../data/AGFs"

agf_files: list[Path] = [
    f
    for f in agf_dir.glob("*.[Aa][Gg][Ff]")
    if re.search(pattern="schott", string=f.name)
]

if not agf_files:
    raise FileNotFoundError("No AGF files found in the data directory.")

schott: GlassCatalogue = GlassCatalogue().read_AGF(agf_filename=agf_files)

MIN_WL: ScalarQuantity = 380 * u.nm
MAX_WL: ScalarQuantity = 950 * u.nm

MARVEL_GRATING: EchelleGrating = EchelleGrating(
    name="MARVEL Grating",
    catalogue_number="53-*-417E",
    groove_frequency=52.67 / u.mm,
    ruled_width=50 * u.mm,
    blaze_angle=69.7 * u.deg,
    gamma_angle=1.2 * u.deg,
    min_wl=MIN_WL,
    max_wl=MAX_WL,
)

PRISM: Prism = Prism(
    name="45° N-BK7 Prism",
    glass=schott["N-BK7"],
    apex_angle=45 * u.deg,
    incidence_angle=36.3 * u.deg,
    min_wl=MIN_WL,
    max_wl=MAX_WL,
)

DETECTOR: Detector = Detector(
    name="MiniMARVEL Detector",
    pixel_size=2.315 * u.um,
    nx=5644,
    ny=8288,
)


def main() -> None:
    MiniMARVEL: DoublePassEchelleSpectrograph = DoublePassEchelleSpectrograph(
        name="MiniMARVEL Double-Pass Single-Mode Spectrograph",
        grating=MARVEL_GRATING,
        cross_disperser=PRISM,
        min_wl=MIN_WL,
        max_wl=MAX_WL,
        collimator_focal_length=200 * u.mm,
        camera_focal_length=200 * u.mm,
        slit_width=5 * u.um,
        slit_length=5 * u.um,
        slit_native_focal_ratio=6.5,
        detector=DETECTOR,
    )

    MiniMARVEL.print_parameters()

    fig: plt.Figure = plt.figure("MiniMARVEL spectral format", figsize=(8, 10))
    ax: plt.Axes = fig.gca()

    MiniMARVEL.plot_detector_edge(ax=ax, name=False)

    ax.text(
        -6.25,
        -9.5,
        "Detector edge",
        fontsize=20,
        ha="left",
        va="bottom",
        alpha=0.5,
    )

    ax.set_xlim(-9.5, 9.5)
    ax.set_ylim(-12.25, 12.25)

    for f in range(1):  # number of fibres
        offset: Quantity = f * 130 * u.um * MiniMARVEL.slit_magnification
        for m in MiniMARVEL.spectral_orders:
            wls: Quantity = MiniMARVEL.order_wavelengths(m=m)

            # # just 5 wavelengths
            # wls = np.linspace(wls[0].to(u.nm).value, wls[-1].to(u.nm).value, num=5)
            # wls *= u.nm

            mean_wl: Quantity = np.mean(wls)  # ty: ignore[invalid-assignment]

            xs, ys = np.transpose(MiniMARVEL.image_plane_position(m=m, wavelength=wls))

            ax.plot(
                -xs + offset,
                ys,
                color=wavelength_to_rgb(mean_wl, fade_factor=0.8),
                lw=2.5,
                # lw=0,
                # marker=".",
                path_effects=stroke(thickness=3),
            )

            if m % 5 == 0:
                if m == 90:  # noqa: PLR2004
                    mtext: str = f"{m = }"
                    wltext: str = f"$\\lambda = {wls[0].to(u.nm):.0f}$"
                else:
                    mtext: str = f"{m}"
                    wltext: str = f"{wls[0].to(u.nm).value:.0f}"

                if abs(ys[0].value) > ax.get_ylim()[1]:
                    continue

                ax.text(
                    (-xs[0] + offset).value - 0.25,
                    ys[0].value - 0.25,
                    mtext,
                    fontsize=16,
                    ha="right",
                    va="top",
                    bbox={
                        "facecolor": "w",
                        "edgecolor": "none",
                        "alpha": 0.85,
                        "boxstyle": "round,pad=0.25",
                    },
                )

                if abs(ys[-1].value) > ax.get_ylim()[1]:
                    continue

                ax.text(
                    (-xs[-1] + offset).value - 0.25,
                    ys[-1].value + 0.25,
                    wltext,
                    fontsize=16,
                    ha="right",
                    va="bottom",
                    bbox={
                        "facecolor": "w",
                        "edgecolor": "none",
                        "alpha": 0.85,
                        "boxstyle": "round,pad=0.25",
                    },
                )

            min_FSR_wl, max_FSR_wl = MiniMARVEL.grating.order_FSR_bounds(m=m)

            for name, wl in _atomic_lines.items():
                # for name, wl in _fraunhofer_lines.items():
                if min_FSR_wl <= wl <= max_FSR_wl:
                    x, y = MiniMARVEL.image_plane_position(m=m, wavelength=wl)

                    if abs(y.value) + 1 > ax.get_ylim()[1]:
                        continue

                    ax.text(
                        -x.value,
                        y.value,
                        f"{name}",
                        fontsize=16,
                        ha="center",
                        va="center",
                        bbox={
                            "facecolor": "w",
                            "edgecolor": "k",
                            "alpha": 0.85,
                            "boxstyle": "round,pad=0.25",
                        },
                    )

    ax.set_aspect("equal")

    ax.set_xlabel("Image plane position (mm)", size=16)
    ax.set_ylabel("Image plane position (mm)", size=16)

    plt.axis("off")

    plt.show()


if __name__ == "__main__":
    main()
