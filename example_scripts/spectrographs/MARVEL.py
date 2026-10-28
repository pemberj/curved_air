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

from curved_air.detectors import e2v_9k
from curved_air.glass import GlassCatalogue
from curved_air.instruments.echelle import EchelleSpectrograph
from curved_air.optics.dispersers import EchelleGrating, Prism
from curved_air.other import (  # noqa: F401
    _atomic_lines,
    _balmer_lines,
    _fraunhofer_lines,
)
from curved_air.plotting import plot_style, stroke, wavelength_to_rgb  # noqa: F401

if TYPE_CHECKING:
    from curved_air.type_hinting import ArrayQuantity, ScalarQuantity

# from curved_air.utils import agf_files

plt.style.use(plot_style)


MARVEL: EchelleSpectrograph = EchelleSpectrograph(
    name="MARVEL Spectrograph",
    grating=EchelleGrating(
        name="MARVEL Grating",
        catalogue_number="53-*-417E",
        groove_frequency=52.67 / u.mm,
        ruled_width=400 * u.mm,
        blaze_angle=69.7 * u.deg,
        gamma_angle=0.3 * u.deg,
    ),
    cross_disperser=Prism(
        name="MARVEL Prism",
        glass=GlassCatalogue("Schott")["F2"],
        apex_angle=60 * u.deg,
    ),
    detector=e2v_9k,
    min_wl=380 * u.nm,
    max_wl=950 * u.nm,
    beam_diameter=160 * u.mm,
    camera_focal_length=730 * u.mm,
    slit_width=40 * u.um,
    slit_length=0.5 * u.mm,
    slit_native_focal_ratio=3.85,
)


def main(plot: bool = False) -> None:

    MARVEL.print_parameters()

    if plot:
        fig: plt.Figure = plt.figure("MARVEL spectral format", figsize=(10, 10))
        ax: plt.Axes = fig.gca()

        MARVEL.plot_detector_edge(ax=ax)

        b = 50

        ax.set_xlim(b, -b)
        ax.set_ylim(-b, b)

        ax.set_aspect("equal")

        for f in range(1):  # number of fibres
            offset: ScalarQuantity = f * 130 * u.um * MARVEL.slit_magnification
            for m in MARVEL.spectral_orders:
                wls: ArrayQuantity = MARVEL.order_wavelengths(m=m)
                mean_wl: ScalarQuantity = np.mean(wls.value) * wls.unit

                xs, ys = np.transpose(MARVEL.image_plane_position(m=m, wavelength=wls))
                ax.plot(
                    xs + offset,
                    ys,
                    color=wavelength_to_rgb(mean_wl, fade_factor=0.8),
                    lw=3.5,
                    # path_effects=stroke(thickness=3.5),
                )

                if m % 5 == 0:
                    if m == 90:  # noqa: PLR2004
                        mtext = f"{m = }"
                        wltext = f"$\\lambda = {wls[0].to(u.nm):.0f}$"
                    else:
                        mtext = f"{m}"
                        wltext = f"{wls[0].to(u.nm).value:.0f}"

                    if abs(ys[0].value) + 5 > b:
                        continue

                    ax.text(
                        (xs[0] + offset).value,
                        ys[0].value - 1.5,
                        mtext,
                        fontsize=14,
                        ha="center",
                        va="top",
                    )

                    if abs(ys[-1].value) + 5 > b:
                        continue

                    ax.text(
                        (xs[-1] + offset).value,
                        ys[-1].value + 1.5,
                        wltext,
                        fontsize=14,
                        ha="center",
                        va="bottom",
                    )

                min_FSR_wl, max_FSR_wl = MARVEL.grating.order_FSR_bounds(m=m)

                for name, wl in _atomic_lines.items():
                    if min_FSR_wl <= wl <= max_FSR_wl:
                        x, y = MARVEL.image_plane_position(m=m, wavelength=wl)

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

        # plt.savefig(Path(__file__).parent / "MARVEL-spectral-format.png", dpi=300)

        plt.show()


if __name__ == "__main__":
    main(plot=True)
