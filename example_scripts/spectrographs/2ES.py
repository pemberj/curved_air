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
from curved_air.glass import Glass, GlassCatalogue
from curved_air.instruments.echelle import EchelleSpectrograph
from curved_air.optics.dispersers import EchelleGrating, Prism, TwoPrisms
from curved_air.other import _atomic_lines
from curved_air.plotting import plot_style, stroke, wavelength_to_rgb

if TYPE_CHECKING:
    from curved_air.type_hinting import ArrayQuantity, ScalarQuantity

plt.style.use(plot_style)


F2: Glass = GlassCatalogue("Schott")["F2"]


def main(plot: bool = False) -> None:
    TwoES: EchelleSpectrograph = EchelleSpectrograph(
        name="2ES Spectrograph",
        grating=EchelleGrating(
            catalogue_number="53-*-417E",
            groove_frequency=31.6 / u.mm,
            ruled_width=400 * u.mm,
            blaze_angle=76 * u.deg,
            gamma_angle=0.5 * u.deg,
        ),
        cross_disperser=TwoPrisms(
            prism1=Prism(
                glass=F2,
                apex_angle=48.7 * u.deg,
            ),
            prism2=Prism(
                glass=F2,
                apex_angle=48.7 * u.deg,
            ),
        ),
        detector=Detector(
            name="STA1600",
            pixel_size=9 * u.um,
            nx=10560,
            ny=10560,
        ),
        min_wl=375 * u.nm,
        max_wl=900 * u.nm,
        beam_diameter=100 * u.mm,
        camera_focal_length=570 * u.mm,
        slit_width=32 * u.um,
        slit_length=128 * u.um,
        slit_native_focal_ratio=5,
    ).print_parameters()

    if plot:
        fig: plt.Figure = plt.figure("2ES spectral format", figsize=(10, 10))
        ax: plt.Axes = fig.gca()

        TwoES.plot_detector_edge(ax=ax)

        b = 50

        ax.set_xlim(b, -b)
        ax.set_ylim(-b, b)

        ax.set_aspect("equal")

        for m in TwoES.spectral_orders:
            wls: ArrayQuantity = TwoES.order_wavelengths(m=m)
            mean_wl: ScalarQuantity = np.mean(wls.value) * wls.unit

            xs, ys = np.transpose(TwoES.image_plane_position(m=m, wavelength=wls))
            ax.plot(
                xs,
                ys,
                color=wavelength_to_rgb(mean_wl, fade_factor=0.8),
                lw=2.5,
                path_effects=stroke(thickness=5),
            )

            if m % 5 == 0:
                if m == 165:  # noqa: PLR2004
                    mtext = f"{m = }"
                    wltext = f"$\\lambda = {wls[0].to(u.nm):.0f}$"
                else:
                    mtext = f"{m}"
                    wltext = f"{wls[0].to(u.nm).value:.0f}"

                if abs(ys[0].value) + 5 > b:
                    continue

                ax.text(
                    xs[0].value,
                    ys[0].value - 2,
                    mtext,
                    fontsize=14,
                    ha="center",
                    va="top",
                )

                if abs(ys[-1].value) + 5 > b:
                    continue

                ax.text(
                    xs[-1].value,
                    ys[-1].value + 2,
                    wltext,
                    fontsize=14,
                    ha="center",
                    va="bottom",
                )

            min_FSR_wl, max_FSR_wl = TwoES.grating.order_FSR_bounds(m=m)

            for name, wl in _atomic_lines.items():
                if min_FSR_wl <= wl <= max_FSR_wl:
                    x, y = TwoES.image_plane_position(m=m, wavelength=wl)

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

        # ax.patch.set_facecolor("black")
        ax.axis("off")

        # plt.savefig(Path(__file__).parent / "TwoES-spectral-format.png", dpi=300)

        plt.show()


if __name__ == "__main__":
    main(plot=True)
