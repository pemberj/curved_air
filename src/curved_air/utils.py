"""
curved_air

utils

This module contains utility functions that are used throughout the curved_air package.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.units import Quantity
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from curved_air.optics.dispersers import EchelleGrating
from curved_air.type_hinting import ScalarQuantity

DEFAULT_WAVELENGTH: ScalarQuantity = 632.8 * u.nm


@dataclass
class SurfaceTolerance:
    name: str = ""
    nominal_radius: ScalarQuantity = 1000 * u.mm
    diameter: ScalarQuantity = 100 * u.mm
    wavelength: ScalarQuantity = DEFAULT_WAVELENGTH
    fringes: float | None = None
    positive_tolerance: ScalarQuantity | None = None
    negative_tolerance: ScalarQuantity | None = None

    def __post_init__(self) -> None:

        if (
            self.fringes is None
            and self.negative_tolerance is None
            and self.positive_tolerance is None
        ):
            raise ValueError("You must specify either fringes or tolerances.")

        if self.fringes is not None:
            assert (
                self.positive_tolerance is None and self.negative_tolerance is None
            ), "You cannot specify both fringes and tolerances."

            self.positive_tolerance: ScalarQuantity = abs(
                sagitta2radius(
                    radius2sagitta(self.nominal_radius, self.diameter)
                    + self.fringes * self.wavelength / 2,
                    self.diameter,
                )
                - 1 * self.nominal_radius
            )
            self.negative_tolerance: ScalarQuantity = abs(
                sagitta2radius(
                    radius2sagitta(self.nominal_radius, self.diameter)
                    - self.fringes * self.wavelength / 2,
                    self.diameter,
                )
                - 1 * self.nominal_radius
            )

        else:
            # If only one of the tolerances is specified, assume the tolerances are symmetrical
            if self.positive_tolerance is not None and self.negative_tolerance is None:
                self.negative_tolerance: ScalarQuantity = self.positive_tolerance

            elif (
                self.negative_tolerance is not None and self.positive_tolerance is None
            ):
                self.positive_tolerance: ScalarQuantity = self.negative_tolerance

            self.fringes: float = (self.sagitta_delta / (self.wavelength / 2)).si.value

    @property
    def steeper_radius(self) -> ScalarQuantity:
        assert self.positive_tolerance is not None
        return self.nominal_radius + 1 * self.positive_tolerance

    @property
    def flatter_radius(self) -> ScalarQuantity:
        assert self.negative_tolerance is not None
        return self.nominal_radius - 1 * self.negative_tolerance

    @property
    def nominal_sagitta(self) -> ScalarQuantity:
        return radius2sagitta(self.nominal_radius, self.diameter)

    @property
    def flatter_sagitta(self) -> ScalarQuantity:
        return radius2sagitta(self.flatter_radius, self.diameter)

    @property
    def steeper_sagitta(self) -> ScalarQuantity:
        return radius2sagitta(self.steeper_radius, self.diameter)

    @property
    def sagitta_delta(self) -> ScalarQuantity:
        if self.fringes is not None:
            return abs(self.fringes) * self.wavelength / 2
        if self.positive_tolerance is not None and self.negative_tolerance is not None:
            return max(
                abs(self.nominal_sagitta - self.steeper_sagitta),
                abs(self.nominal_sagitta - self.flatter_sagitta),
            )
        raise ValueError("Unable to calculate sagitta delta. Please check your inputs.")

    def __dict__(self) -> dict[str, ScalarQuantity | float | None]:
        return {
            "nominal_radius": self.nominal_radius,
            "diameter": self.diameter,
            "wavelength": self.wavelength,
            "fringes": self.fringes,
            "positive_tolerance": self.positive_tolerance,
            "negative_tolerance": self.negative_tolerance,
            "semi_diameter": self.semi_diameter,
            "nominal_sagitta": self.nominal_sagitta,
            "sagitta_delta": self.sagitta_delta,
            "flatter_sagitta": self.flatter_sagitta,
            "steeper_sagitta": self.steeper_sagitta,
            "steeper_radius": self.steeper_radius,
            "flatter_radius": self.flatter_radius,
        }

    def __repr__(self) -> str:
        return f"SurfaceTolerance(nominal_radius={self.nominal_radius}, diameter={self.diameter}, wavelength={self.wavelength}, fringes={self.fringes}, positive_tolerance={self.positive_tolerance}, negative_tolerance={self.negative_tolerance})"

    def __str__(self) -> str:
        return f"""Surface Tolerance for {f"{self.name}" or "Unnamed Surface"}:
        Ø {self.diameter:.6g}
        {self.fringes:.2g} Fringes at λ = {self.wavelength:.4g}
        R {self.nominal_radius:.7g} {self.positive_tolerance:+.4g} / {-self.negative_tolerance:+.4g}
        Nominal sag: {self.nominal_sagitta:.5g}
        """  # ty:ignore[unsupported-operator]


def radius2sagitta(
    radius: ScalarQuantity,
    diameter: ScalarQuantity,
) -> ScalarQuantity:

    return np.abs(radius) - np.sqrt(radius**2 - (diameter / 2) ** 2)


def sagitta2radius(
    sagitta: ScalarQuantity,
    diameter: ScalarQuantity,
) -> ScalarQuantity:

    return ((diameter / 2) ** 2 + sagitta**2) / (2 * sagitta)


def agf_files(search: str = "") -> list[Path]:
    UTILS_PATH: Path = Path(__file__).parent

    agf_dir: Path = UTILS_PATH / "agfs"

    files: list[Path] = [
        f
        for f in agf_dir.glob("*.[Aa][Gg][Ff]")
        if re.search(pattern=search, string=f.name)
    ]

    if not files:
        raise FileNotFoundError("No AGF files found in the data directory.")

    return files


def richardson_gratings(
    filename: str | None = None,
    selection_query: str = "Ruled_width >= 400",
) -> list[EchelleGrating]:
    """
    Load grating parameters from a csv catalogue from Richardson Gratings.
    """

    if not filename:
        data_dir = Path(__file__).parent / "gratings"
        files = list(data_dir.glob("richardson-echelle-T3.csv"))
        assert len(files) == 1
        file = files[0]
    df = pd.read_csv(file, delimiter=" ")

    selection = df.query(selection_query)

    gratings: list[EchelleGrating] = [
        EchelleGrating(
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

    return gratings


# EchelleDesigner / order_gird.py
def order_grid(
    grating: EchelleGrating | None = None,
    min_wl: ScalarQuantity = 380 * u.nm,
    max_wl: ScalarQuantity = 870 * u.nm,
    num_configs: int = 12,
) -> dict[str, Any]:
    """
    This function can be used to generate an evenly-spaced (in order m number) grid of
    spectral orders for use in optical design software, eg. Zemax.
    """

    if grating is None:
        grating: EchelleGrating = EchelleGrating(
            groove_frequency=52.67 / u.mm,
            blaze_angle=69 * u.deg,
            gamma_angle=0.36 * u.deg,
        )
        print(f"No grating provided, using default grating: {grating}")

    blaze: ScalarQuantity = grating.blaze_angle
    groove_density: ScalarQuantity = grating.groove_frequency
    gamma: ScalarQuantity = grating.gamma

    m_times_lambda: float = np.cos(gamma) * (1 / groove_density) * 2 * np.sin(blaze)

    sign_of_blaze: int = -int(blaze / abs(blaze))

    # print(f"{sign_of_blaze = }")

    min_order: int = int(-1 * sign_of_blaze * int((m_times_lambda / min_wl) - 0.5))
    max_order: int = int(-1 * sign_of_blaze * int((m_times_lambda / max_wl) + 0.5))

    # print(f"{min_order = }")
    # print(f"{max_order = }")

    min_wl_blaze: ScalarQuantity = (m_times_lambda / min_order).to(u.nm)
    max_wl_blaze: ScalarQuantity = (m_times_lambda / max_order).to(u.nm)
    # print(f"{min_wl_blaze = !s}")
    # print(f"{max_wl_blaze = !s}")

    alt_increment: int = int(((max_order - 1) - (min_order + 1)) / (num_configs - 4))
    # print(f"{alt_increment = }")

    alt_grid: list[int] = []

    alt_grid.append(int(min_order))
    alt_grid.append(int(min_order + 1))

    for i in range(1, num_configs - 3):
        m: int = min_order + 1 + i * alt_increment
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
    grating: EchelleGrating,
    collimator_focal_length: ScalarQuantity,
) -> ScalarQuantity:
    """
    Calculate the height of the (echelle-dispersed) image at the transfer focus
    """
    image_height: ScalarQuantity = (
        collimator_focal_length
        / u.radian
        * grating.order_opening_angle(m=grating.min_order)
    ).si

    # print("Total required height of transfer mirror:"\
    #     +f"{image_height.to(u.mm) = :.3f}")

    return image_height.to(u.mm)


# EtalonSim / helper.py
def FWHM_old(
    y_values: list[float] | NDArray[np.floating],
    x_values: list[float] | NDArray[np.floating] | None = None,
) -> float:
    """
    Calculate the FWHM of a series of numerical values on a constant x-grid
    """

    norm: NDArray[np.floating] = y_values / np.nanmax(y_values)

    # Create zero-crossings at half max value
    half: NDArray[np.floating] = abs(norm - max(norm) / 2)

    zero_crossings: list[int] = []
    # First zero-crossing
    zero_crossings.append(int(np.argmin(half[: len(half) // 2])))
    # Second zero-crossing
    zero_crossings.append(int(np.argmin(half[len(half) // 2 :]) + len(half) // 2))

    fwhm_steps: float = np.diff(zero_crossings)[0]

    if x_values is not None:
        x_step: float = np.diff(x_values)[0]
        return fwhm_steps * x_step
    return fwhm_steps


def lin_interp(
    x: list[float] | NDArray[np.floating],
    y: list[float] | NDArray[np.floating],
    i: int,
    half: float,
) -> float:
    return x[i] + (x[i + 1] - x[i]) * ((half - y[i]) / (y[i + 1] - y[i]))


def FWHM(
    x_values: list | NDArray[np.floating],
    y_values: list | NDArray[np.floating],
) -> tuple[float, float]:
    """ """

    y_values -= min(y_values)
    y_values /= max(y_values)

    signs: NDArray[np.int_] = np.sign(np.add(y_values, -0.5))  # Array of +1, -1
    zero_crossings: NDArray[np.bool_] = signs[0:-2] != signs[1:-1]  # Array of bools
    zero_crossings_i: NDArray[np.int_] = np.where(zero_crossings)[0]

    if len(zero_crossings_i) < 2:  # noqa: PLR2004
        return np.nan, np.nan

    return (
        lin_interp(x_values, y_values, zero_crossings_i[0], 0.5),
        lin_interp(x_values, y_values, zero_crossings_i[1], 0.5),
    )


def Gaussian(
    x: list[float] | NDArray[np.floating],
    x0: float = 0,
    sigma: float = 1,
) -> NDArray[np.floating]:
    """
    Generate a simple Gaussian distribution
    """

    if isinstance(x, list):
        x: NDArray[np.floating] = np.array(list)

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

    from matplotlib import animation  # noqa: PLC0415

    fig: plt.Figure = plt.figure()
    ax: plt.Axes = fig.gca()

    ims: list[list[plt.AxesImage]] = []
    for r in results:
        im: plt.AxesImage = ax.imshow(
            r,
            animated=True,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )

        ims.append([im])

    ani: plt.ArtistAnimation = animation.ArtistAnimation(
        fig,
        ims,
        interval=60,
        repeat_delay=0,
    )

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

    if save:
        # Set up formatting for the movie files
        writer = animation.writers["ffmpeg"]

        ani.save(f"{title}_animation.mp4", writer=writer(fps=30))
    else:
        plt.show()


def out_of_bounds(
    ax: plt.Axes,
    x: ScalarQuantity | float | None = None,
    y: ScalarQuantity | float | None = None,
) -> bool:
    """ """

    assert x is not None or y is not None, "At least one of xs or ys must be provided."

    if isinstance(x, Quantity):
        x: float = x.value
    if isinstance(y, Quantity):
        y: float = y.value

    if x is not None:  # noqa: SIM102
        if abs(x) + 5 > ax.get_xlim()[1] or abs(x) + 5 < ax.get_xlim()[0]:
            return True

    if y is not None:  # noqa: SIM102
        if abs(y) + 5 > ax.get_ylim()[1] or abs(y) + 5 < ax.get_ylim()[0]:
            return True

    return False


def main() -> None: ...


if __name__ == "__main__":
    main()
