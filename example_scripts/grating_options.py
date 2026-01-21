# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "curved-air>=0.1.0",
# ]
# [tool.uv.sources]
# curved_air = { path = "..", editable = true }
# ///
import re
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from astropy import units as u
from astropy.units import Quantity  # noqa: F401
from matplotlib import pyplot as plt
from rich import print as rprint  # noqa: A004, F401

from curved_air.glass import GlassCatalogue
from curved_air.optics.dispersers import EchelleGrating, Prism
from curved_air.plotting import plot_style, wavelength_to_rgb
from curved_air.utils import richardson_gratings

if TYPE_CHECKING:
    from curved_air.type_hinting import ArrayQuantity, ScalarQuantity
plt.style.use(plot_style)

SCRIPT_PATH = Path(__file__).parent
OUT_PATH = Path("/home/japember/Projects/grating_options_plots")
OUT_PATH.mkdir(parents=True, exist_ok=True)

agf_dir = SCRIPT_PATH / "../data/AGFs"

agf_files = [
    f
    for f in agf_dir.glob("*.[Aa][Gg][Ff]")
    if re.search(pattern="schott", string=f.name)
]

if not agf_files:
    raise FileNotFoundError("No AGF files found in the data directory.")

schott: GlassCatalogue = GlassCatalogue().read_AGF(agf_filename=agf_files)


markers: list[str] = [
    "o",
    "s",
    "D",
    "P",
    "v",
    "^",
    "<",
    ">",
    "h",
    "H",
    "p",
    "*",
    "+",
    "X",
]


def plot_grating_options(gratings: list[EchelleGrating]) -> None:
    fig: plt.Figure = plt.figure("Grating options")
    ax: plt.Axes = fig.gca()

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


def plot_wavelength_vs_m(gratings: list[EchelleGrating]) -> None:
    """
    Plot the blaze wavelength as a function of order number.
    """

    fig: plt.Figure = plt.figure("Wavelength as a function of order number")
    ax: plt.Axes = fig.gca()

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


def plot_opening_angle_vs_m(gratings: list[EchelleGrating]) -> None:
    """_summary_

    Args:
        gratings (list[EchelleGrating]): _description_
    """
    fig: plt.Figure = plt.figure("Order opening angle as a function of order number")
    ax: plt.Axes = fig.gca()

    for i, g in enumerate(gratings):
        ms: list[int] = g.diffraction_orders
        angles: list[float] = [a.to(u.deg).value for a in g.order_opening_angle(ms)]

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


def plot_opening_angle_vs_blaze_wavelength(gratings: list[EchelleGrating]) -> None:
    """_summary_

    Args:
        gratings (list[EchelleGrating]): _description_
    """
    fig: plt.Figure = plt.figure("Opening angle as a function of blaze wavelength")
    ax: plt.Axes = fig.gca()

    for i, g in enumerate(gratings):
        ms: list[int] = g.diffraction_orders
        blaze_wavelengths: ArrayQuantity = g.blaze_wavelength(ms)
        blaze_wavelengths: list[float] = [b.to(u.um).value for b in blaze_wavelengths]
        angles: list[float] = [a.to(u.deg).value for a in g.order_opening_angle(ms)]

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


def plot_opening_angle_vs_R(gratings: list[EchelleGrating]) -> None:
    """_summary_

    Args:
        gratings (list[EchelleGrating]): _description_
    """
    fig: plt.Figure = plt.figure("Order opening angle as a function of R-number")
    ax: plt.Axes = fig.gca()

    for i, g in enumerate(gratings):
        ms: list[int] = g.diffraction_orders
        angles: float = max([a.to(u.deg).value for a in g.order_opening_angle(ms)])

        ax.plot(
            g.R,
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
    ax.set_ylabel("Reddest-wavelength order FSR opening angle [degrees]")

    ax.legend()

    plt.savefig(
        Path(OUT_PATH / "gratings-opening-angle-vs-R.png"),
        dpi=300,
    )


def plot_opening_angle_vs_gpmm(gratings: list[EchelleGrating]) -> None:
    """_summary_

    Args:
        gratings (list[EchelleGrating]): _description_
    """

    fig: plt.Figure = plt.figure("Order opening angle as a function of groove density")
    ax: plt.Axes = fig.gca()

    for i, g in enumerate(gratings):
        ms: list[int] = g.diffraction_orders
        angles: float = max([a.to(u.deg).value for a in g.order_opening_angle(ms)])

        ax.plot(
            g.gpmm,
            angles,
            marker=markers[i],
            markersize=15,
            markeredgecolor="k",
            markeredgewidth=0.5,
            lw=0,
            alpha=0.75,
            label=f"R={g.R}, {g.gpmm.value} g/mm",
        )

    ax.set_xlabel("Grating groove density [grooves/mm]")
    ax.set_ylabel("Reddest-wavelength order FSR opening angle [degrees]")

    ax.legend()

    plt.savefig(
        Path(OUT_PATH / "gratings-opening-angle-vs-gpmm.png"),
        dpi=300,
    )


def plot_opening_angle_times_R_number(gratings: list[EchelleGrating]) -> None:
    """
    Args:
        gratings (list[EchelleGrating]): _description_
    """
    fig: plt.Figure = plt.figure(
        "Order opening angle times R-number as a function of R-number"
    )
    ax: plt.Axes = fig.gca()

    for i, g in enumerate(gratings):
        ms: list[int] = g.diffraction_orders
        red_m: int = min(ms)
        angles: float = max([a.to(u.deg).value for a in g.order_opening_angle(ms)])

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


def plot_angular_dispersion_vs_R_and_gpmm(gratings: list[EchelleGrating]) -> None:
    fig: plt.Figure = plt.figure("Angular dispersion as a function of R-number")
    ax: plt.Axes = fig.gca()

    for i, g in enumerate(gratings):
        ax.plot(
            g.R,
            g.angular_dispersion(m=g.order_number(g.max_wl, quantised=True)).to(
                u.degree / u.nm
            ),
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


def plot_reddest_orders_separation(gratings: list[EchelleGrating]) -> None:
    fig: plt.Figure = plt.figure("Reddest orders separation")
    ax: plt.Axes = fig.gca()

    for i, g in enumerate(gratings):
        slit_magnification: float = g.R / 2.7

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

        color: tuple[float, float, float] = wavelength_to_rgb(
            wavelength=650 * u.nm, fade_factor=0.6
        )

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


def plot_required_cross_dispersion(gratings: list[EchelleGrating]) -> None:
    """ """
    prism = Prism(
        glass=schott["F2"],
        apex_angle=60 * u.deg,
    )
    angular_dispersion_950: ScalarQuantity = prism.local_angular_dispersion(
        wl=950 * u.nm
    ).to(u.deg / u.nm)
    rprint(f"{angular_dispersion_950 = }")

    slit_length: ScalarQuantity = 1.08 * u.mm

    # MARVEL magnification
    slit_angular_size: ScalarQuantity = (slit_length * (u.radian / (1125 * u.mm))).to(
        u.deg
    )

    # yoffset = slit_angular_size

    fig: plt.Figure = plt.figure("Required cross-dispersion", figsize=(8, 5))
    ax: plt.Axes = fig.gca()

    for i, g in enumerate(gratings):
        slit_magnification: float = g.R / 2.7

        slit_angular_size: ScalarQuantity = (
            slit_length * (u.radian / (1125 * u.mm)) * slit_magnification
        ).to(u.deg)

        rprint(f"{g.R = }")
        rprint(f"{slit_angular_size = }")

        reddest_m: int = g.min_order

        wl1: ScalarQuantity = g.blaze_wavelength(reddest_m + 1)
        wl2: ScalarQuantity = g.blaze_wavelength(reddest_m)

        # rprint(wl1, wl2)

        reddest_angular_spread: ScalarQuantity = prism.angular_spread(
            min_wl=wl1, max_wl=wl2
        ).to(u.deg)
        # delta_wavelength: Quantity = (wl2 - wl1).to(u.nm)

        # rprint(f"{reddest_angular_spread / delta_wavelength = }")

        y: float = (slit_angular_size / reddest_angular_spread).value

        rprint(y)

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

        color: tuple[float, float, float] = wavelength_to_rgb(
            wavelength=650 * u.nm, fade_factor=0.6
        )

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
    gratings: list[EchelleGrating] = richardson_gratings(
        selection_query="Ruled_width>=400 and Grooves_per_mm<=110",
    )

    for g in gratings:
        g.min_wl: ScalarQuantity = 380 * u.nm
        g.max_wl: ScalarQuantity = 950 * u.nm

        g.sign_of_blaze: Literal[1, -1] = -1
        rprint(f"{g}")
        # rprint(f"Aspect ratio: {g.aspect_ratio()}")
        # grid = order_grid(grating=g)
        # rprint(grid)

    plot_grating_options(gratings)
    plot_wavelength_vs_m(gratings)
    plot_opening_angle_vs_m(gratings)
    plot_opening_angle_vs_blaze_wavelength(gratings)
    plot_opening_angle_vs_R(gratings)
    plot_opening_angle_vs_gpmm(gratings)

    plot_opening_angle_times_R_number(gratings)

    plot_angular_dispersion_vs_R_and_gpmm(gratings)
    plot_reddest_orders_separation(gratings)

    plot_required_cross_dispersion(gratings)


if __name__ == "__main__":
    main()
