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

from astropy import units as u
from astropy.units import Quantity  # noqa: F401
from matplotlib import pyplot as plt
from rich import print  # noqa: A004

from curved_air.glass import GlassCatalogue
from curved_air.optics import Grating, Prism
from curved_air.plotting import plot_style, wavelength_to_rgb
from curved_air.utils import richardson_gratings

plt.style.use(plot_style)

SCRIPT_PATH = Path(__file__).parent
OUT_PATH = Path("/home/jake/Nextcloud-IvS/Thesis Plots")

agf_dir = SCRIPT_PATH / "../data/AGFs"

agf_files = [
    f
    for f in agf_dir.glob("*.[Aa][Gg][Ff]")
    if re.search(pattern="schott", string=f.name)
]

if not agf_files:
    raise FileNotFoundError("No AGF files found in the data directory.")

schott = GlassCatalogue().read_AGF(agf_filename=agf_files)


markers = ["o", "s", "D", "P", "v", "^", "<", ">", "h", "H", "p", "*", "+", "X"]


def plot_grating_options(gratings: list[Grating]) -> None:
    fig = plt.figure("Grating options")
    ax = fig.gca()

    for i, g in enumerate(gratings):
        ax.plot(
            g.gpmm,
            g.R,
            marker=markers[i],
            markersize=15,
            markeredgecolor="k",
            markeredgewidth=0.5,
            lw=0,
            alpha=0.75,
            label=f"R={g.R}, {g.gpmm.value} g/mm",
        )

        if g.catalogue_number == "53-*-417E":
            ax.plot(
                g.gpmm.value + 0.35,
                g.R,
                marker="o",
                markersize=35,
                markeredgecolor="r",
                markeredgewidth=2.6,
                markerfacecolor="none",
                lw=0,
            )

    ax.set_xlabel("Grooves per mm")
    ax.set_ylabel("R number")

    ax.legend()

    plt.savefig(
        Path(OUT_PATH / "grating-options.png"),
        dpi=300,
    )


def plot_wavelength_vs_m(gratings: list[Grating]) -> None:
    """
    Plot the blaze wavelength as a function of order number.
    """
    import matplotlib.pyplot as plt

    fig = plt.figure("Wavelength as a function of order number")
    ax = fig.gca()

    for i, g in enumerate(gratings):
        ms = g.diffraction_orders
        blaze_wavelengths = g.blaze_wavelength(ms)
        blaze_wavelengths = [b.to(u.um).value for b in blaze_wavelengths]
        ax.plot(
            ms,
            blaze_wavelengths,
            marker=markers[i],
            markersize=5,
            markeredgecolor="k",
            markeredgewidth=0.5,
            alpha=0.5,
            label=f"R={g.R}, {g.gpmm.value} g/mm: {len(ms)} orders",
        )

    ax.set_xlabel("Order number $m$")
    ax.set_ylabel("Blaze wavelength $\\lambda$ [microns]")

    ax.legend()

    plt.savefig(
        Path(OUT_PATH / "gratings-wavelength-vs-m.png"),
        dpi=300,
    )


def plot_opening_angle_vs_m(gratings: list[Grating]) -> None:
    """_summary_

    Args:
        gratings (list[Grating]): _description_
    """
    fig = plt.figure("Order opening angle as a function of order number")
    ax = fig.gca()

    for i, g in enumerate(gratings):
        ms = g.diffraction_orders
        angles = g.order_opening_angle(ms)
        angles = [a.to(u.deg).value for a in angles]

        ax.plot(
            ms,
            angles,
            marker=markers[i],
            markersize=5,
            markeredgecolor="k",
            markeredgewidth=0.5,
            alpha=0.5,
            label=f"R={g.R}, {g.gpmm.value} g/mm",
        )

    ax.set_xlabel("Order number $m$")
    ax.set_ylabel("Order FSR opening angle [degrees]")

    ax.legend()

    plt.savefig(
        Path(OUT_PATH / "gratings-opening-angle-vs-m.png"),
        dpi=300,
    )


def plot_opening_angle_vs_blaze_wavelength(gratings: list[Grating]) -> None:
    """_summary_

    Args:
        gratings (list[Grating]): _description_
    """
    fig = plt.figure("Opening angle as a function of blaze wavelength")
    ax = fig.gca()

    for i, g in enumerate(gratings):
        ms = g.diffraction_orders
        blaze_wavelengths = g.blaze_wavelength(ms)
        blaze_wavelengths = [b.to(u.um).value for b in blaze_wavelengths]

        ms = g.diffraction_orders
        angles = g.order_opening_angle(ms)
        angles = [a.to(u.deg).value for a in angles]

        ax.plot(
            blaze_wavelengths,
            angles,
            marker=markers[i],
            markersize=5,
            markeredgecolor="k",
            markeredgewidth=0.5,
            alpha=0.5,
            label=f"R={g.R}, {g.gpmm.value} g/mm: {len(ms)} orders",
        )

    ax.set_xlabel("Blaze wavelength $\\lambda$ [microns]")
    ax.set_ylabel("Order FSR opening angle [degrees]")

    ax.legend()

    plt.savefig(
        Path(OUT_PATH / "gratings-opening-angle-vs-blaze.png"),
        dpi=300,
    )


def plot_opening_angle_vs_R_and_gpmm(gratings: list[Grating]) -> None:
    """_summary_

    Args:
        gratings (list[Grating]): _description_
    """
    fig = plt.figure("Order opening angle as a function of R-number")
    ax = fig.gca()

    fig2 = plt.figure("Order opening angle as a function of groove density")
    ax2 = fig2.gca()

    for i, g in enumerate(gratings):
        Rs = g.R
        gpmms = g.gpmm
        ms = g.diffraction_orders
        angles = g.order_opening_angle(ms)
        angles = max([a.to(u.deg).value for a in angles])

        ax.plot(
            Rs,
            angles,
            marker=markers[i],
            markersize=15,
            markeredgecolor="k",
            markeredgewidth=0.5,
            lw=0,
            alpha=0.75,
            label=f"R={g.R}, {g.gpmm.value} g/mm",
        )

        ax2.plot(
            gpmms,
            angles,
            marker=markers[i],
            markersize=15,
            markeredgecolor="k",
            markeredgewidth=0.5,
            lw=0,
            alpha=0.75,
            label=f"R={g.R}, {g.gpmm.value} g/mm",
        )

    ax.set_xlabel("Grating R-number")
    ax2.set_xlabel("Grating groove density [grooves/mm]")
    ax.set_ylabel("Reddest-wavelength order FSR opening angle [degrees]")
    ax2.set_ylabel("Reddest-wavelength order FSR opening angle [degrees]")

    ax.legend()
    ax2.legend()

    plt.savefig(
        Path(OUT_PATH / "gratings-opening-angle-vs-R-and-gpmm.png"),
        dpi=300,
    )


def plot_opening_angle_times_R_number(gratings: list[Grating]) -> None:
    """
    Args:
        gratings (list[Grating]): _description_
    """
    fig = plt.figure("Order opening angle times R-number as a function of R-number")
    ax = fig.gca()

    for i, g in enumerate(gratings):
        ms = g.diffraction_orders
        red_m = min(ms)
        angles = g.order_opening_angle(ms)
        angles = max([a.to(u.deg).value for a in angles])

        ax.plot(
            g.R / red_m * g.gpmm,
            angles / g.R,
            marker=markers[i],
            markersize=15,
            markeredgecolor="k",
            markeredgewidth=0.5,
            lw=0,
            alpha=0.75,
            label=f"R={g.R}, {g.gpmm.value} g/mm",
        )

    ax.set_xlabel("Grating R-number")
    ax.set_ylabel("Reddest-wavelength order FSR opening angle times R-number [degrees]")

    ax.legend()

    plt.savefig(
        Path(OUT_PATH / "grating-opening-angle-times-R.png"),
        dpi=300,
    )


def plot_angular_dispersion_vs_R_and_gpmm(gratings: list[Grating]) -> None:
    fig = plt.figure("Angular dispersion as a function of R-number")
    ax = fig.gca()

    for i, g in enumerate(gratings):
        ax.plot(
            g.R,
            g.angular_dispersion(m=g.order_number(g.max_wl)).to(u.degree / u.nm),
            marker=markers[i],
            markersize=15,
            markeredgecolor="k",
            markeredgewidth=0.5,
            lw=0,
            alpha=0.75,
            label=f"R={g.R}, {g.gpmm.value} g/mm",
        )

    ax.set_xlabel("Grating R-number")
    ax.set_ylabel("Angular dispersion [degrees/nm] divided by R-number")

    ax.legend()

    plt.savefig(
        Path(OUT_PATH / "gratings-angular-dispersion-vs-R-and-gpmm.png"),
        dpi=300,
    )


def plot_reddest_orders_separation(gratings: list[Grating]) -> None:
    fig: plt.figure = plt.figure("Reddest orders separation")
    ax: plt.axes = fig.gca()

    for i, g in enumerate(gratings):
        slit_magnification = g.R / 2.7

        marker = ax.plot(
            g.R,
            g.reddest_order_separation,
            marker=markers[i],
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
            marker=markers[i],
            markersize=5,
            markeredgecolor="k",
            markeredgewidth=0.5,
            lw=0,
            alpha=0.75,
            color=marker[0].get_color(),
        )

        ax.plot(
            [g.R, g.R],
            [
                (g.reddest_order_separation / slit_magnification).value,
                g.reddest_order_separation.value,
            ],
            linestyle=":",
            color="k",
            alpha=0.25,
            zorder=-10,
        )

        color = wavelength_to_rgb(wavelength=650 * u.nm, fade_factor=0.6)

        if g.catalogue_number == "53-*-417E":
            ax.plot(
                g.R + 0.025,
                g.reddest_order_separation / slit_magnification,
                marker="o",
                markersize=35,
                markeredgecolor=color,
                markeredgewidth=2.6,
                markerfacecolor="none",
                lw=0,
            )

            ax.arrow(
                g.R + 0.025,
                (g.reddest_order_separation / slit_magnification).value + 5,
                0,
                15,
                head_width=0.1,
                head_length=3,
                fc=color,
                ec=color,
                lw=2,
                capstyle="round",
            )

            ax.arrow(
                g.R + 0.025,
                (g.reddest_order_separation / slit_magnification).value - 5,
                0,
                -11,
                head_width=0.1,
                head_length=3,
                fc=color,
                ec=color,
                lw=2,
                capstyle="round",
            )

    ax.text(2.3, 50, "Lower required cross-dispersion", ha="left", fontsize=14)
    ax.text(2.3, 2, "Higher required cross-dispersion", ha="left", fontsize=14)

    ax.set_xlabel("Grating R-number")
    ax.set_ylabel(
        "Reddest orders' separation [nm]\n(and equivalent after slit magnification)"
    )

    ax.set_ylim(0, 70)

    ax.legend()

    plt.savefig(
        Path(OUT_PATH / "grating-reddest-orders-separation.png"),
        dpi=300,
    )


def plot_required_cross_dispersion(gratings: list[Grating]) -> None:
    """ """
    prism = Prism(
        glass=schott["F2"],
        apex_angle=60 * u.deg,
    )
    angular_dispersion_950 = prism.local_angular_dispersion(wl=950 * u.nm).to(
        u.deg / u.nm
    )
    print(f"{angular_dispersion_950 = }")

    slit_length = 1.08 * u.mm

    # MARVEL magnification
    slit_angular_size = (slit_length * (u.radian / (1125 * u.mm))).to(u.deg)

    yoffset = slit_angular_size

    fig: plt.figure = plt.figure("Required cross-dispersion", figsize=(8, 5))
    ax: plt.axes = fig.gca()

    for i, g in enumerate(gratings):
        slit_magnification = g.R / 2.7

        slit_angular_size = (
            slit_length * (u.radian / (1125 * u.mm)) * slit_magnification
        ).to(u.deg)

        print(f"{g.R = }")
        print(f"{slit_angular_size = }")

        reddest_m = g.min_order

        wl1 = g.blaze_wavelength(reddest_m + 1)
        wl2 = g.blaze_wavelength(reddest_m)

        # print(wl1, wl2)

        reddest_angular_spread = prism.angular_spread(min_wl=wl1, max_wl=wl2).to(u.deg)
        delta_wavelength = (wl2 - wl1).to(u.nm)

        # print(f"{reddest_angular_spread / delta_wavelength = }")

        y = slit_angular_size / reddest_angular_spread

        print(y)

        _markers = ax.plot(
            g.R,
            y,
            marker=markers[i],
            markersize=15,
            markeredgecolor="k",
            markeredgewidth=0.5,
            lw=0,
            alpha=0.75,
            label=f"R={g.R}, {g.gpmm.value} g/mm",
        )

        color = wavelength_to_rgb(wavelength=650 * u.nm, fade_factor=0.6)

        if g.catalogue_number == "53-*-417E":
            ax.plot(
                g.R + 0.025,
                y,
                marker="o",
                markersize=35,
                markeredgecolor=color,
                markeredgewidth=2.6,
                markerfacecolor="none",
                lw=0,
            )

    ax.axhline(
        y=1,
        color="k",
        xmax=0.71,
        linestyle="--",
        alpha=0.25,
        zorder=-10,
    )

    ax.text(
        5.4,
        1,
        "Single $60\\degree$ F2 prism",
        ha="right",
        va="bottom",
        fontsize=12,
        color="k",
        alpha=0.75,
    )

    ax.set_xlabel("Grating R-number")
    ax.set_ylabel(
        "Required cross-dispersion\nrelative to a single $60\\degree$ F2 prism"
    )

    ax.set_xlim(1.5, 7)
    ax.set_ylim(0.2, 15)
    ax.set_yscale("log")

    ax.text(3.5, 5, "Higher required cross-dispersion", ha="center", fontsize=14)
    ax.text(3.5, 0.3, "Lower required cross-dispersion", ha="center", fontsize=14)

    ax.legend(frameon=True, fontsize=12, markerscale=0.8, loc="lower right")

    plt.savefig(
        Path(OUT_PATH / "grating-required-cross-dispersion.png"),
        dpi=300,
    )

    # plt.show()


def main() -> None:
    gratings: list[Grating] = richardson_gratings(
        selection_query="Ruled_width>=400 and Grooves_per_mm<=110",
    )

    for g in gratings:
        g.min_wl = 380 * u.nm
        g.max_wl = 950 * u.nm

        g.sign_of_blaze = -1
        print(f"{g}")
        # print(f"Aspect ratio: {g.aspect_ratio()}")
        # grid = order_grid(grating=g)
        # print(grid)

    # plot_grating_options(gratings)
    # plot_wavelength_vs_m(gratings)
    # plot_opening_angle_vs_m(gratings)
    # plot_opening_angle_vs_blaze_wavelength(gratings)
    # plot_opening_angle_vs_R_and_gpmm(gratings)

    # plot_opening_angle_times_R_number(gratings)

    # plot_angular_dispersion_vs_R_and_gpmm(gratings)
    # plot_reddest_orders_separation(gratings)

    # plot_order_separation(gratings)

    plot_required_cross_dispersion(gratings)


if __name__ == "__main__":
    main()
