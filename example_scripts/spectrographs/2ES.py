# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "curved-air>=0.1.0",
#     "rich>=10.0.0",
# ]
# [tool.uv.sources]
# curved_air = { path = "../..", editable = true }
# ///

import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt

from curved_air.detectors import Detector
from curved_air.glass import Glass, GlassCatalogue
from curved_air.instruments.echelle import EchelleSpectrograph
from curved_air.optics.dispersers import EchelleGrating, Prism, TwoPrisms
from curved_air.other import (  # noqa: F401
    _atomic_lines,
    _balmer_lines,
    _fraunhofer_lines,
)
from curved_air.plotting import plot_style, stroke, wavelength_to_rgb  # noqa: F401
from curved_air.type_hinting import ArrayQuantity, ScalarQuantity

# from curved_air.utils import agf_files

plt.style.use(plot_style)


F2: Glass = GlassCatalogue("schott")["F2"]

MIN_WL: ScalarQuantity = 370 * u.nm
MAX_WL: ScalarQuantity = 900 * u.nm

TwoES_GRATING: EchelleGrating = EchelleGrating(
    name="TwoES Grating",
    catalogue_number="53-*-417E",
    groove_frequency=31.6 / u.mm,
    ruled_width=800 * u.mm,
    blaze_angle=76 * u.deg,
    gamma_angle=0.5 * u.deg,
    min_wl=MIN_WL,
    max_wl=MAX_WL,
)

TwoES_PRISMS: TwoPrisms = TwoPrisms(
    prism1=Prism(
        name="TwoES Prism 1",
        glass=F2,
        apex_angle=30 * u.deg,
        incidence_angle=60 * u.deg,
    ),
    prism2=Prism(
        name="TwoES Prism 2",
        glass=F2,
        apex_angle=30 * u.deg,
        incidence_angle=500 * u.deg,
    ),
    min_wl=MIN_WL,
    max_wl=MAX_WL,
)

TwoES_DETECTOR: Detector = Detector(
    name="STA1600",
    pixel_size=9 * u.um,
    nx=10560,
    ny=10560,
)


def main(plot: bool = False) -> None:
    TwoES: EchelleSpectrograph = EchelleSpectrograph(
        name="2ES Spectrograph",
        grating=TwoES_GRATING,
        cross_disperser=TwoES_PRISMS,
        detector=TwoES_DETECTOR,
        min_wl=MIN_WL,
        max_wl=MAX_WL,
        collimator_focal_length=1000 * u.mm,
        camera_focal_length=700 * u.mm,
        slit_width=32 * u.um,
        slit_length=128 * u.um,
        slit_native_focal_ratio=4,
    ).print_parameters()

    if plot:
        fig: plt.Figure = plt.figure("2ES spectral format", figsize=(10, 10))
        ax: plt.Axes = fig.gca()

        TwoES.plot_detector_edge(ax=ax)

        b = 50

        ax.set_xlim(b, -b)
        ax.set_ylim(-b, b)

        ax.set_aspect("equal")

        for f in range(1):  # number of fibres
            offset: ScalarQuantity = f * 130 * u.um * TwoES.slit_magnification
            for m in TwoES.spectral_orders:
                wls: ArrayQuantity = TwoES.order_wavelengths(m=m)
                mean_wl: ScalarQuantity = np.mean(wls.value) * wls.unit

                xs, ys = np.transpose(TwoES.image_plane_position(m=m, wavelength=wls))
                ax.plot(
                    xs + offset,
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
                        (xs[0] + offset).value,
                        ys[0].value - 2,
                        mtext,
                        fontsize=14,
                        ha="center",
                        va="top",
                    )

                    if abs(ys[-1].value) + 5 > b:
                        continue

                    ax.text(
                        (xs[-1] + offset).value,
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
