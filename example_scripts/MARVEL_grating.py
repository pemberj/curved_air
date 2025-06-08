# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "curved-air>=0.1.0",
#     "rich>=10.0.0",
# ]
# [tool.uv.sources]
# curved_air = { path = "..", editable = true }
# ///


import numpy as np
from astropy import units as u
from astropy.units import Quantity  # noqa: F401
from matplotlib import pyplot as plt

from curved_air.optics import Grating
from curved_air.plotting import plot_style, stroke, wavelength_to_rgb

plt.style.use(plot_style)


def plot_FSR(grating: Grating) -> None:
    """
    Plot the free spectral range of the grating as a function of wavelength.
    """

    fig = plt.figure("Free spectral range")
    ax = fig.gca()

    ms = grating.diffraction_orders

    blaze_wavelengths = grating.blaze_wavelength(ms)
    blaze_wavelengths = [b.to(u.um).value for b in blaze_wavelengths]

    FSRs = grating.order_FSR(ms)
    FSRs = [f.to(u.nm).value for f in FSRs]

    ax.plot(
        blaze_wavelengths,
        FSRs,
        marker=".",
        markersize=10,
        markeredgecolor="k",
        markeredgewidth=0.5,
        alpha=0.5,
        # label=f"R={g.R}, {g.gpmm.value} g/mm: {len(ms)} orders",
    )

    ax.set_xlabel("Blaze wavelength $\\lambda$ [microns]")
    ax.set_ylabel("Free spectral range $\\Delta\\lambda_{FSR}$ [nm]")

    ax.legend()

    plt.show()


def plot_spectral_format(grating: Grating) -> None:
    """ """
    fig = plt.figure("Spectral format")
    ax = fig.gca()

    for m in grating.diffraction_orders[::-1]:
        wl_bounds = grating.order_FSR_bounds(m=m, FSR_fraction=1)
        wls = np.linspace(
            wl_bounds[0].to(u.nm).value,
            wl_bounds[1].to(u.nm).value,
            100,
        )

        angles = (
            grating.diffraction_angle(wl=wl_bounds[0], m=m),
            grating.diffraction_angle(wl=wl_bounds[1], m=m),
        )

        angles = np.linspace(
            angles[0].to(u.deg).value,
            angles[1].to(u.deg).value,
            100,
        )

        ax.plot(
            [m] * len(angles),
            angles,
            color=wavelength_to_rgb(np.mean(wls), fade_factor=0.5),
            label=f"m={m}",
            path_effects=stroke(thickness=4),
        )

        if m % 5 == 0:
            ax.text(
                m,
                angles[0] - 0.25,
                f"{np.mean(wls):.0f}nm",
                ha="center",
                va="top",
                fontsize=10,
            )

    ax.set_xlabel("Diffraction order $m$")
    ax.set_ylabel("Diffraction angle [deg]")

    ax.set_xlim(grating.max_order + 5, grating.min_order - 5)
    ax.set_ylim(-5, 5)

    plt.show()


def plot_ideal_grating_parameters() -> None:
    """
    Plot the ideal grating.
    """

    fig = plt.figure("Ideal grating")
    ax = fig.gca()

    MARVEL_grating: Grating = Grating(
        name="MARVEL nominal",
        catalogue_number="53-*-417E",
        groove_frequency=52.67 / u.mm,
        blaze_angle=69.7 * u.deg,
        gamma_angle=0.3 * u.deg,
        min_wl=385 * u.nm,
        max_wl=950 * u.nm,
    )

    ideal_grating: Grating = Grating(
        name="MARVEL ideal",
        catalogue_number="_",
        groove_frequency=80 / u.mm,
        blaze_angle=76 * u.deg,
        gamma_angle=0.3 * u.deg,
        min_wl=385 * u.nm,
        max_wl=950 * u.nm,
    )

    for g in [MARVEL_grating, ideal_grating]:
        slit_magnification: float = g.R / 2.7

        print(g.name)
        print(slit_magnification)
        print(g.reddest_order_separation / slit_magnification)

        marker = ax.plot(
            g.R,
            g.reddest_order_separation,
            markersize=15,
            markeredgecolor="k",
            markeredgewidth=0.5,
            lw=0,
            alpha=0.75,
            label=f"R={g.R}, {g.gpmm.value} g/mm",
        )

        # if slit_magnification != 1:
        ax.plot(
            g.R,
            g.reddest_order_separation / slit_magnification,
            markersize=5,
            markeredgecolor="k",
            markeredgewidth=0.5,
            lw=0,
            alpha=0.75,
            color=marker[0].get_color(),
        )

        # ax.plot(
        #     [g.R, g.R],
        #     [
        #         (g.reddest_order_separation / slit_magnification).value,
        #         g.reddest_order_separation.value,
        #     ],
        #     linestyle=":",
        #     color="k",
        #     alpha=0.25,
        #     zorder=-10,
        # )

    plt.show()


def main() -> None:
    MARVEL_grating: Grating = Grating(
        catalogue_number="53-*-417E",
        groove_frequency=52.67 / u.mm,
        blaze_angle=69.7 * u.deg,
        gamma_angle=0.3 * u.deg,
        min_wl=385 * u.nm,
        max_wl=950 * u.nm,
    )

    NEID_grating: Grating = Grating(
        catalogue_number="53-*-417E",
        groove_frequency=31.6 / u.mm,
        blaze_angle=76 * u.deg,
        gamma_angle=0.3 * u.deg,
        min_wl=385 * u.nm,
        max_wl=950 * u.nm,
    )

    # plot_FSR(MARVEL_grating)
    plot_spectral_format(MARVEL_grating)

    plot_spectral_format(NEID_grating)

    # plot_ideal_grating_parameters()


if __name__ == "__main__":
    main()
