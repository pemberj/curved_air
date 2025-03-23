# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "curved-air>=0.1.0",
#     "rich>=10.0.0",
# ]
# [tool.uv.sources]
# curved_air = { path = "..", editable = true }
# ///


from matplotlib import pyplot as plt
from astropy import units as u
from astropy.units import Quantity  # noqa: F401
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
            alpha=0.5,
            label=f"R={g.R}, {g.gpmm.value} g/mm",
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


def plot_angle_vs_m(gratings: list[Grating]) -> None:
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
    plot_wavelength_vs_m(gratings)
    # plot_angle_vs_m(gratings)
    # plot_opening_angle_vs_blaze_wavelength(gratings)


if __name__ == "__main__":
    main()
