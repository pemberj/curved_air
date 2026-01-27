# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "curved-air>=0.1.0",
#     "rich>=10.0.0",
# ]
# [tool.uv.sources]
# curved_air = { path = "../..", editable = true }
# ///
from typing import TYPE_CHECKING

import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt

from curved_air.detectors import Detector
from curved_air.glass import GlassCatalogue
from curved_air.instruments.echelle import EchelleSpectrograph
from curved_air.optics.dispersers import EchelleGrating, Prism
from curved_air.other import _atomic_lines
from curved_air.plotting import plot_style, stroke, wavelength_to_rgb
from curved_air.type_hinting import ScalarQuantity

if TYPE_CHECKING:
    from curved_air.type_hinting import ArrayQuantity

plt.style.use(plot_style)

PBM2Y = GlassCatalogue("MARVEL_CAMERA")["PBM2Y"]

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
    name="NEID Prism",
    glass=PBM2Y,
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
            name="NEID Detector",
            nx=9216,
            ny=9231,
            pixel_size=10 * u.um,
        ),
    )

    NEID.print_parameters()

    fig: plt.Figure = plt.figure("NEID spectral format", figsize=(10, 10))
    ax: plt.Axes = fig.gca()

    b: float = 50  # size of plot

    NEID.plot_detector_edge(ax=ax)

    for m in NEID.spectral_orders:
        wls: ArrayQuantity = NEID.order_wavelengths(m=m)
        mean_wl: ScalarQuantity = np.mean(wls.value) * wls.unit
        # print(f"{m = }, mean_wl={mean_wl.to(u.nm):.1f}")

        xs, ys = np.transpose(NEID.image_plane_position(m=m, wavelength=wls))
        ax.plot(
            xs,
            ys,
            color=wavelength_to_rgb(mean_wl, fade_factor=0.8),
            lw=3.5,
            path_effects=stroke(thickness=5),
        )

        if m % 5 == 0:
            if m == 160:  # noqa: PLR2004
                mtext = f"{m = }"
                wltext = f"$\\lambda = {wls[0].to(u.nm):.0f}$"
            else:
                mtext = f"{m}"
                wltext = f"{wls[0].to(u.nm).value:.0f}"

            if abs(ys[0].value) + 5 > b:
                continue

            ax.text(
                xs[0].value,
                ys[0].value - 1.5,
                mtext,
                fontsize=14,
                ha="right",
                va="top",
            )

            if abs(ys[-1].value) + 5 > b:
                continue

            ax.text(
                xs[-1].value,
                ys[-1].value + 1.5,
                wltext,
                fontsize=14,
                ha="right",
                va="bottom",
            )

        min_FSR_wl, max_FSR_wl = NEID.grating.order_FSR_bounds(m=m)

        for name, wl in _atomic_lines.items():
            if min_FSR_wl <= wl <= max_FSR_wl:
                x, y = NEID.image_plane_position(m=m, wavelength=wl)

                ax.text(
                    x.value,
                    y.value,
                    f"{name}",
                    fontsize=14,
                    ha="center",
                    va="center",
                    bbox={
                        "facecolor": "w",
                        "edgecolor": "k",
                        "alpha": 0.75,
                        "boxstyle": "round,pad=0.25",
                    },
                )

    ax.set_xlim(b, -b)
    ax.set_ylim(-b, b)

    # ax.patch.set_facecolor("black")
    ax.axis("off")

    plt.show()


if __name__ == "__main__":
    main()
