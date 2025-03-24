# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "curved-air>=0.1.0",
#     "rich>=10.0.0",
# ]
# [tool.uv.sources]
# curved_air = { path = "..", editable = true }
# ///


from astropy import units as u
from astropy.units import Quantity  # noqa: F401
from matplotlib import pyplot as plt
from rich import print  # noqa: A004

from curved_air.optics import Grating
from curved_air.utils import richardson_gratings

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

    plt.show()


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

    plt.show()


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

    plt.show()


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

    plt.show()


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

    plt.show()


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

    plt.show()


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

    plt.show()


def plot_reddest_orders_separation(gratings: list[Grating]) -> None:
    fig: plt.figure = plt.figure("Reddest orders separation")
    ax: plt.axes = fig.gca()

    for i, g in enumerate(gratings):
        ax.plot(
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

        if g.catalogue_number == "53-*-417E":
            ax.plot(
                g.R + 0.025,
                g.reddest_order_separation,
                marker="o",
                markersize=35,
                markeredgecolor="r",
                markeredgewidth=2.6,
                markerfacecolor="none",
                lw=0,
            )

    ax.text(3.75, 50, "Lower required cross-dispersion", ha="center", fontsize=14)
    ax.text(3.75, 5, "Higher required cross-dispersion", ha="center", fontsize=14)

    ax.set_xlabel("Grating R-number")
    ax.set_ylabel("Reddest orders' separation [nm]")

    ax.legend()

    plt.show()


def main() -> None:
    gratings: list[Grating] = richardson_gratings(
        selection_query="Ruled_width>=400 and Grooves_per_mm<=110",
    )

    for g in gratings:
        g.sign_of_blaze = -1
        print(f"{g}")
        # print(f"Aspect ratio: {g.aspect_ratio()}")
        # grid = order_grid(grating=g)
        # print(grid)

    plot_grating_options(gratings)
    # plot_wavelength_vs_m(gratings)
    # plot_opening_angle_vs_m(gratings)
    # plot_opening_angle_vs_blaze_wavelength(gratings)
    # plot_opening_angle_vs_R_and_gpmm(gratings)

    # plot_opening_angle_times_R_number(gratings)

    # plot_angular_dispersion_vs_R_and_gpmm(gratings)
    # plot_reddest_orders_separation(gratings)


if __name__ == "__main__":
    main()
