# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "curved-air>=0.1.0",
# ]
# [tool.uv.sources]
# curved_air = { path = "../..", editable = true }
# ///
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt

from curved_air.detectors import H2RG_18, H4RG_15
from curved_air.instruments.echelle import EchelleSpectrograph
from curved_air.optics.dispersers import EchelleGrating, SimpleGrating
from curved_air.other import _atomic_lines, _IR_bands  # noqa: F401
from curved_air.plotting import plot_style, stroke, wavelength_to_rgb
from curved_air.utils import out_of_bounds

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from curved_air.type_hinting import ArrayQuantity, ScalarQuantity

plt.style.use(plot_style)

SCRIPT_PATH: Path = Path(__file__).parent


def main(plot: bool = True) -> None:
    MIN_WL: ScalarQuantity = 1 * u.um
    MAX_WL: ScalarQuantity = 1.8 * u.um

    ECHELLE: EchelleGrating = EchelleGrating(
        name="Echelle Grating",
        catalogue_number="53-*-174E",
        groove_frequency=31.6 / u.mm,
        ruled_width=800 * u.mm,
        blaze_angle=76 * u.deg,
        gamma_angle=0.5 * u.deg,
        min_wl=MIN_WL,
        max_wl=MAX_WL,
    )

    VPH: SimpleGrating = SimpleGrating(
        name="VPH Cross-disperser",
        groove_frequency=150 / u.mm,
        min_wl=MIN_WL,
        max_wl=MAX_WL,
    )

    spectrograph: EchelleSpectrograph = EchelleSpectrograph(
        name="LFAST Test",
        grating=ECHELLE,
        cross_disperser=VPH,
        detector=H4RG_15,
        min_wl=MIN_WL,
        max_wl=MAX_WL,
        # collimator_focal_length=1000 * u.mm,
        camera_focal_length=400 * u.mm,
        slit_width=54 * u.um,
        slit_length=126 * u.um,
        slit_native_focal_ratio=2.7,
    ).print_parameters()

    if plot:
        fig: plt.Figure = plt.figure(
            "Spectral format",
            figsize=(10, 10),
            dpi=96,
            constrained_layout=True,
        )
        ax: plt.Axes = fig.gca()

        b: int = 60
        ax.set_xlim(b, -b)
        ax.set_ylim(-b, b)

        for m in spectrograph.spectral_orders:
            wls: ArrayQuantity = spectrograph.order_wavelengths(m=m)
            mean_wl: ScalarQuantity = np.mean(wls.value) * wls.unit

            xs, ys = np.transpose(
                spectrograph.image_plane_position(m=m, wavelength=wls)
            )

            ax.plot(
                xs,
                ys,
                color=wavelength_to_rgb(mean_wl, fade_factor=0.8),
                lw=2.5,
                path_effects=stroke(thickness=3.5),
            )

            slit_xs: ArrayQuantity = xs[::5]
            slit_ys: ArrayQuantity = ys[::5]

            sw: ScalarQuantity = spectrograph.slit_image_width
            sl: ScalarQuantity = spectrograph.slit_image_length

            for sx, sy in zip(slit_xs, slit_ys, strict=False):
                slit_corner_xs: list[float] = [
                    (sx - sl / 2).value,
                    (sx + sl / 2).value,
                    (sx + sl / 2).value,
                    (sx - sl / 2).value,
                    (sx - sl / 2).value,
                ] * u.mm
                slit_corner_ys: list[float] = [
                    (sy - sw / 2).value,
                    (sy - sw / 2).value,
                    (sy + sw / 2).value,
                    (sy + sw / 2).value,
                    (sy - sw / 2).value,
                ] * u.mm

                ax.plot(
                    slit_corner_xs,
                    slit_corner_ys,
                    lw=1.5,
                    ls="--",
                    color="k",
                    alpha=0.5,
                    zorder=-1,
                )

            if m % 5 == 0:
                if m in [35, 55]:  # noqa: PLR2004
                    mtext: str = f"{m = }"
                    wltext: str = f"$\\lambda = {wls[0].to(u.nm):.0f}$"
                else:
                    mtext: str = f"{m}"
                    wltext: str = f"{wls[0].to(u.nm).value:.0f}"

                if not out_of_bounds(ax=ax, y=ys[0]):
                    ax.text(
                        xs[0].value,
                        ys[0].value - 1.5,
                        mtext,
                        fontsize=14,
                        ha="center",
                        va="top",
                    )

                if not out_of_bounds(ax=ax, y=ys[-1]):
                    ax.text(
                        xs[-1].value,
                        ys[-1].value + 1.5,
                        wltext,
                        fontsize=14,
                        ha="center",
                        va="bottom",
                    )

            # Plot the locations of interesting spectral lines
            # min_FSR_wl, max_FSR_wl = spectrograph.grating.order_FSR_bounds(m=m)

            band_colors: dict[str, str] = {
                "I": "violet",
                "Y": "blue",
                "J": "green",
                "H": "orangered",
                "K": "red",
            }

            for name, (wl1, wl2) in _IR_bands.items():
                wls_mask: NDArray[np.bool_] = (wl1 <= wls) & (wls <= wl2)
                if np.any(wls_mask):
                    ax.plot(
                        xs[wls_mask],
                        ys[wls_mask],
                        lw=4,
                        label=name,
                        color=band_colors.get(name, "black"),
                        alpha=0.75,
                    )

        ax.set_aspect("equal")
        ax.axis("off")

        H4RG_15.plot_edge(ax=ax, name=True)
        H2RG_18.plot_edge(ax=ax, name=True)

        # plt.savefig(SCRIPT_PATH / "LFAST_test_spectral_format.png", dpi=300)

        plt.show()


if __name__ == "__main__":
    main(plot=True)
