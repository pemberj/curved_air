"""
curved_air

utils

Utility functions
"""

from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u
from matplotlib import pyplot as plt

from curved_air.optics import Grating


# Echelledesigner / read-csv.py
def richardson_gratings(
    filename: str | None = None,
    plot: bool = False,
    selection_query: str = "Ruled_width >= 400",
) -> list[Grating]:
    """
    Load grating parameters from a csv catalogue from Richardson Gratings.
    """

    if not filename:
        data_dir = Path(__file__).parent / ".." / ".." / "data"
        files = list(data_dir.glob("richardson-echelle-T3.csv"))
        assert len(files) == 1
        file = files[0]
    df = pd.read_csv(file, delimiter=" ")

    selection = df.query(selection_query)

    gratings: list[Grating] = [
        Grating(
            catalogue_number=c,
            groove_frequency=float(gpmm) / u.mm,
            blaze_angle=float(theta[:-1]) * u.deg,
        )
        for c, gpmm, theta in zip(
            selection["Catalog_number"],
            selection["Grooves_per_mm"],
            selection["Nominal_blaze_angle"],
            strict=False,
        )
    ]

    if plot:
        fig = plt.figure("Wavelength as a function of order number")
        ax = fig.gca()

        for g in gratings:
            ms = g.diffraction_orders
            blaze_wavelengths = g.blaze_wavelength(ms)
            blaze_wavelengths = [b.to(u.um).value for b in blaze_wavelengths]
            ax.plot(
                ms,
                blaze_wavelengths,
                marker=".",
                markersize=10,
                markeredgecolor="k",
                markeredgewidth=0.5,
                alpha=0.5,
                label=f"R={g.R}, {g.gpmm.value} g/mm: {len(ms)} orders",
            )

        ax.set_xlabel("Order number $m$")
        ax.set_ylabel("Blaze wavelength $\\lambda$ [microns]")

        ax.legend()

        fig = plt.figure("Order opening angle as a function of order number")
        ax = fig.gca()

        for g in gratings:
            ms = g.diffraction_orders
            angles = g.order_opening_angle(ms)
            angles = [a.to(u.deg).value for a in angles]

            ax.plot(
                ms,
                angles,
                marker=".",
                markersize=10,
                markeredgecolor="k",
                markeredgewidth=0.5,
                alpha=0.5,
                label=f"R={g.R}, {g.gpmm.value} g/mm",
            )

        ax.set_xlabel("Order number $m$")
        ax.set_ylabel("Order FSR opening angle [degrees]")

        ax.legend()

        fig = plt.figure("Opening angle as a function of blaze wavelength")
        ax = fig.gca()

        for g in gratings:
            ms = g.diffraction_orders
            blaze_wavelengths = g.blaze_wavelength(ms)
            blaze_wavelengths = [b.to(u.um).value for b in blaze_wavelengths]

            ms = g.diffraction_orders
            angles = g.order_opening_angle(ms)
            angles = [a.to(u.deg).value for a in angles]

            ax.plot(
                blaze_wavelengths,
                angles,
                marker=".",
                markersize=10,
                markeredgecolor="k",
                markeredgewidth=0.5,
                alpha=0.5,
                label=f"R={g.R}, {g.gpmm.value} g/mm: {len(ms)} orders",
            )

        ax.set_xlabel("Blaze wavelength $\\lambda$ [microns]")
        ax.set_ylabel("Order FSR opening angle [degrees]")

        ax.legend()

        plt.show()

    return gratings


# EchelleDesigner / order_gird.py
def order_grid(
    grating: Grating = None,
    min_wl: u.Quantity = 380 * u.nm,
    max_wl: u.Quantity = 870 * u.nm,
    num_configs: int = 12,
) -> list | None:
    """
    This function can be used to generate an evenly-spaced (in order m number) grid of
    spectral orders for use in optical design software, eg. Zemax.
    """

    if grating is None:
        grating = (
            Grating(
                groove_frequency=52.67 / u.mm,
                blaze_angle=69 * u.deg,
                gamma_angle=0.36 * u.deg,
            ),
        )
        print(f"No grating provided, using default grating: {grating}")

    blaze = grating.blaze_angle
    groove_density = grating.groove_frequency
    gamma = grating.gamma

    m_times_lambda = np.cos(gamma) * (1 / groove_density) * 2 * np.sin(blaze)

    sign_of_blaze = -int(blaze / abs(blaze))

    # print(f"{sign_of_blaze = }")

    min_order = int(-1 * sign_of_blaze * int((m_times_lambda / min_wl) - 0.5))
    max_order = int(-1 * sign_of_blaze * int((m_times_lambda / max_wl) + 0.5))

    # print(f"{min_order = }")
    # print(f"{max_order = }")

    min_wl_blaze = (m_times_lambda / min_order).to(u.nm)
    max_wl_blaze = (m_times_lambda / max_order).to(u.nm)
    # print(f"{min_wl_blaze = !s}")
    # print(f"{max_wl_blaze = !s}")

    alt_increment = int(((max_order - 1) - (min_order + 1)) / (num_configs - 4))
    # print(f"{alt_increment = }")

    alt_grid = []

    alt_grid.append(int(min_order))
    alt_grid.append(int(min_order + 1))

    for i in range(1, num_configs - 3):
        m = min_order + 1 + i * alt_increment
        alt_grid.append(m)

    alt_grid.append(int(max_order - 1))
    alt_grid.append(int(max_order))

    # print(f"{alt_grid=}")
    # print(f"{len(alt_grid)=}")
    return {
        "sign_of_blaze": sign_of_blaze,
        "min_order": min_order,
        "max_order": max_order,
        "min_wl_blaze": min_wl_blaze,
        "max_wl_blaze": max_wl_blaze,
        "alt_increment": alt_increment,
        "alt_grid": alt_grid,
    }


# EchelleDesigner / transfer_mirror_length.py
def calculate_transfer_mirror_height(
    grating: Grating,
    collimator_focal_length: u.Quantity,
) -> u.Quantity:
    image_height = (
        collimator_focal_length
        / u.radian
        * grating.order_opening_angle(m=grating.min_order)
    ).si

    # print("Total required height of transfer mirror:"\
    #     +f"{image_height.to(u.mm) = :.3f}")

    return image_height.to(u.mm)


# EtalonSim / helper.py
def FWHM_old(
    y_values: list | np.ndarray, x_values: list | np.ndarray | None = None
) -> float:
    """
    Calculate the FWHM of a series of numerical values on a constant x-grid
    """

    norm = y_values / np.nanmax(y_values)
    half = abs(norm - max(norm) / 2)  # Create zero-crossings at half max value
    zero_crossings = []
    zero_crossings.append(np.argmin(half[: len(half) // 2]))  # First zero-crossing
    zero_crossings.append(
        np.argmin(half[len(half) // 2 :]) + len(half) // 2
    )  # Second zero-crossing

    fwhm_steps = np.diff(zero_crossings)[0]

    if x_values is not None:
        x_step = np.diff(x_values)[0]
        return fwhm_steps * x_step
    return fwhm_steps


def lin_interp(
    x: list | np.ndarray, y: list | np.ndarray, i: int, half: float
) -> float:
    return x[i] + (x[i + 1] - x[i]) * ((half - y[i]) / (y[i + 1] - y[i]))


def FWHM(x_values: list | np.ndarray, y_values: list | np.ndarray) -> tuple[float]:
    """ """

    y_values -= min(y_values)
    y_values /= max(y_values)

    signs = np.sign(np.add(y_values, -0.5))
    zero_crossings = (signs[0:-2]) != signs[1:-1]
    zero_crossings_i = np.where(zero_crossings)[0]

    return (
        lin_interp(x_values, y_values, zero_crossings_i[0], 0.5),
        lin_interp(x_values, y_values, zero_crossings_i[1], 0.5),
    )


def Gaussian(x: list | np.ndarray, x0: float = 0, sigma: float = 1) -> np.ndarray:
    """
    Generate a simple Gaussian distribution
    """

    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
        -(1 / 2) * ((x - x0) / sigma) ** 2
    )


def animate_results(
    results,  # noqa: ANN001
    vmin: float = 0,
    vmax: float = 1,
    cmap: str = "jet",
    title: str = "",
    save: bool = False,
) -> None:
    """ """

    from matplotlib import animation

    fig = plt.figure()
    ax = fig.gca()

    ims = []
    for r in results:
        im = ax.imshow(r, animated=True, vmin=vmin, vmax=vmax, cmap=cmap)

        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=60, repeat_delay=0)

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

    if save:
        # Set up formatting for the movie files
        writer = animation.writers["ffmpeg"]

        ani.save(f"{title}_animation.mp4", writer=writer(fps=30))
    else:
        plt.show()


def main() -> None: ...


if __name__ == "__main__":
    main()
