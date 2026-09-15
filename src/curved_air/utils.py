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
import streamlit as st
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
    selection_query: str | None = "Ruled_width >= 400",
) -> list[EchelleGrating]:
    """
    Load grating parameters from a csv catalogue from Richardson Gratings.
    """

    if not filename:
        data_dir: Path = Path(__file__).parent / "gratings"
        files: list[Path] = list(data_dir.glob("richardson-echelle-T3.csv"))
        assert len(files) == 1
        file: Path = files[0]
    df: pd.DataFrame = pd.read_csv(file, delimiter=" ")

    selection: pd.DataFrame = df.query(selection_query) if selection_query else df

    gratings: list[EchelleGrating] = [
        EchelleGrating(
            catalogue_number=g["Catalog_number"],
            groove_frequency=float(g["Grooves_per_mm"]) / u.mm,
            blaze_angle=float(g["Nominal_blaze_angle"][:-1]) * u.deg,
            ruled_width=float(g["Ruled_width"]) * u.mm,
            groove_length=float(g["Groove_length"]) * u.mm,
        )
        for g in selection.to_dict(orient="records")
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
    sigma: float | None = None,
    FWHM: float | None = None,
) -> NDArray[np.floating]:
    """
    Generate a simple Gaussian distribution
    """

    if sigma is None and FWHM is None:
        raise ValueError("You must specify either sigma or FWHM.")

    if FWHM is not None:
        assert sigma is None, "You cannot specify both sigma and FWHM."
        sigma: float = FWHM / (2 * np.sqrt(2 * np.log(2)))

    if isinstance(x, list):
        x: NDArray[np.floating] = np.array(list)

    assert isinstance(sigma, float | int)

    profile: NDArray = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
        -(1 / 2) * ((x - x0) / sigma) ** 2
    )

    return profile / np.nanmax(profile)


def SuperGaussian(
    x: list[float] | NDArray[np.floating],
    x0: float = 0,
    sigma: float | None = None,
    FWHM: float | None = None,
    n: int = 2,
) -> NDArray[np.floating]:
    """
    Generate a simple Super-Gaussian distribution
    """

    if sigma is None and FWHM is None:
        raise ValueError("You must specify either sigma or FWHM.")

    if FWHM is not None:
        assert sigma is None, "You cannot specify both sigma and FWHM."
        # For a Super-Gaussian, the half-max condition
        # exp(-0.5 * (w/sigma)^(2n)) = 0.5 gives w = sigma * (2 ln 2)^(1/(2n)),
        # so FWHM = 2w -> sigma = FWHM / (2 * (2 ln 2)^(1/(2n))). This reduces to the
        # standard Gaussian conversion only when n = 1.
        sigma: float = FWHM / (2 * (2 * np.log(2)) ** (1 / (2 * n)))

    if isinstance(x, list):
        x: NDArray[np.floating] = np.array(list)

    assert isinstance(sigma, float | int)

    profile: NDArray = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
        -(1 / 2) * ((x - x0) / sigma) ** (2 * n)
    )

    return profile / np.nanmax(profile)


def Lorentzian(
    x: list[float] | NDArray[np.floating],
    x0: float = 0,
    gamma: float | None = None,
    FWHM: float | None = None,
) -> NDArray[np.floating]:
    """
    Generate a simple Lorentzian distribution
    """

    if gamma is None and FWHM is None:
        raise ValueError("You must specify either gamma or FWHM.")

    if FWHM is not None:
        assert gamma is None, "You cannot specify both gamma and FWHM."
        gamma: float = FWHM / 2

    if isinstance(x, list):
        x: NDArray[np.floating] = np.array(list)

    assert isinstance(gamma, float | int)

    profile: NDArray = (1 / np.pi) * (gamma / ((x - x0) ** 2 + gamma**2))

    return profile / np.nanmax(profile)


def Voigt(
    x: list[float] | NDArray[np.floating],
    x0: float = 0,
    sigma: float | None = None,
    gamma: float | None = None,
    FWHM: float | None = None,
) -> NDArray[np.floating]:
    """
    Generate a simple (true, convolved) Voigt distribution.

    Either:
    - specify `sigma` and `gamma` directly, in which case their absolute values set
      the widths of the Gaussian and Lorentzian components being convolved, or
    - specify `FWHM` together with `sigma` and `gamma`, in which case only the RATIO
      of `sigma` to `gamma` is used to set the relative Gaussian/Lorentzian
      contribution, while their absolute scale is determined (using the Olivero &
      Longbothum (1977) approximation) so that the resulting profile has (very
      closely) the given `FWHM`. Because a true Voigt profile is a convolution,
      scaling `sigma` and `gamma` by the same factor scales the resulting FWHM by
      exactly that factor, so this rescaling is exact to within the (very small)
      accuracy limits of the approximation used to relate FWHM to sigma and gamma.
    """

    from scipy.special import wofz  # noqa: PLC0415

    if FWHM is None:
        if sigma is None or gamma is None:
            raise ValueError(
                "You must specify both sigma and gamma, and optionally FWHM."
            )
    else:
        if sigma is None or gamma is None:
            raise ValueError(
                "You must specify both sigma and gamma (their ratio sets the "
                "relative Gaussian/Lorentzian contribution) together with FWHM."
            )

        # Only the ratio of sigma to gamma matters here: it sets the relative
        # Gaussian/Lorentzian broadening magnitudes. A true Voigt profile's FWHM
        # scales exactly linearly with (sigma, gamma) when both are scaled by the
        # same factor (since it is a convolution), so we can rescale to hit the
        # requested `FWHM` (to within the accuracy of the Olivero & Longbothum
        # approximation used to relate FWHM to sigma and gamma).
        ratio: float = sigma / gamma

        gaussian_fwhm: float = ratio * (2 * np.sqrt(2 * np.log(2)))
        lorentzian_fwhm: float = 2

        total_fwhm: float = 0.5346 * lorentzian_fwhm + np.sqrt(
            0.2166 * lorentzian_fwhm**2 + gaussian_fwhm**2
        )

        scale: float = FWHM / total_fwhm
        gamma = scale
        sigma = ratio * scale

    if isinstance(x, list):
        x: NDArray[np.floating] = np.array(list)

    assert isinstance(sigma, float | int)
    assert isinstance(gamma, float | int)

    profile: NDArray = np.real(wofz(((x - x0) + 1j * gamma) / (sigma * np.sqrt(2)))) / (
        sigma * np.sqrt(2 * np.pi)
    )

    return profile / np.nanmax(profile)


def PseudoVoigt(
    x: list[float] | NDArray[np.floating],
    x0: float = 0,
    sigma: float | None = None,
    gamma: float | None = None,
    FWHM: float | None = None,
) -> NDArray[np.floating]:
    """
    Generate a simple Pseudo-Voigt distribution.

    Either:
    - specify `sigma` and `gamma` directly, in which case their absolute values set the
      (independent) widths of the Gaussian and Lorentzian components, or
    - specify `FWHM` together with `sigma` and `gamma`, in which case only the RATIO of
      `sigma` to `gamma` is used, via the Thompson-Cox-Hastings approximation, to set
      the mixing fraction (`eta`) between the Gaussian and Lorentzian components. `FWHM`
      itself then sets the width shared by both components (as is standard for a
      pseudo-Voigt profile), so that the resulting profile has (very closely, to within
      the accuracy of the approximation) the given FWHM.
    """

    if FWHM is None:
        if sigma is None or gamma is None:
            raise ValueError(
                "You must specify both sigma and gamma, and optionally FWHM."
            )

        eta: float = (
            1.36603 * (gamma / (gamma + sigma))
            - 0.47719 * (gamma / (gamma + sigma)) ** 2
            + 0.11116 * (gamma / (gamma + sigma)) ** 3
        )
    else:
        if sigma is None or gamma is None:
            raise ValueError(
                "You must specify both sigma and gamma (their ratio sets the "
                "relative Gaussian/Lorentzian contribution) together with FWHM."
            )

        # Only the ratio of sigma to gamma matters here: it sets the relative
        # Gaussian/Lorentzian broadening magnitudes, from which the mixing
        # fraction `eta` is derived below (Thompson-Cox-Hastings
        # approximation). `FWHM` is then used directly as the width shared by
        # both components, so the resulting profile has (approximately) the
        # requested FWHM.
        ratio: float = sigma / gamma

        gaussian_fwhm: float = ratio * (2 * np.sqrt(2 * np.log(2)))
        lorentzian_fwhm: float = 2

        total_fwhm: float = (
            gaussian_fwhm**5
            + 2.69269 * gaussian_fwhm**4 * lorentzian_fwhm
            + 2.42843 * gaussian_fwhm**3 * lorentzian_fwhm**2
            + 4.47163 * gaussian_fwhm**2 * lorentzian_fwhm**3
            + 0.07842 * gaussian_fwhm * lorentzian_fwhm**4
            + lorentzian_fwhm**5
        ) ** (1 / 5)

        eta: float = (
            1.36603 * (lorentzian_fwhm / total_fwhm)
            - 0.47719 * (lorentzian_fwhm / total_fwhm) ** 2
            + 0.11116 * (lorentzian_fwhm / total_fwhm) ** 3
        )

        sigma: float = FWHM / (2 * np.sqrt(2 * np.log(2)))
        gamma: float = FWHM / 2

    if isinstance(x, list):
        x: NDArray[np.floating] = np.array(list)

    assert isinstance(sigma, float | int)
    assert isinstance(gamma, float | int)

    profile: NDArray = eta * Lorentzian(x, x0=x0, gamma=gamma) + (1 - eta) * Gaussian(
        x, x0=x0, sigma=sigma
    )

    return profile / np.nanmax(profile)


def Moffat(
    x: list[float] | NDArray[np.floating],
    x0: float = 0,
    alpha: float | None = None,
    beta: float = 1,
    FWHM: float | None = None,
) -> NDArray[np.floating]:
    """
    Generate a simple Moffat distribution
    """

    if beta == 1:
        beta += 1e-6  # Avoid division by zero

    assert beta > 1, "Beta must be greater than 1 for a valid Moffat distribution."

    if alpha is None and FWHM is None:
        raise ValueError("You must specify either alpha or FWHM.")

    if FWHM is not None:
        assert alpha is None, "You cannot specify both alpha and FWHM."
        alpha: float = FWHM / (2 * np.sqrt(2 ** (1 / beta) - 1))

    if isinstance(x, list):
        x: NDArray[np.floating] = np.array(list)

    assert isinstance(alpha, float | int)
    assert isinstance(beta, float | int)

    profile: NDArray = (
        (beta - 1) / (np.pi * alpha**2) * (1 + ((x - x0) / alpha) ** 2) ** (-beta)
    )

    return profile / np.nanmax(profile)


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


def st_quantity_input(
    name: str,
    possible_units: list[str],
    min_value: float | None = None,
    max_value: float | None = None,
    value: float = 0.0,
    step: float = 1.0,
    format: str | None = None,  # noqa: A002
    default_unit: str | None = None,
    disabled: bool = False,
    help: str | None = None,  # noqa: A002
    key: str = "",
    unit_input_text: str = "Unit",
    inverse: bool = False,
) -> ScalarQuantity:
    """
    A custom Streamlit input widget for entering a numeric quantity with a value and
    unit.

    This uses AstroPy Units to parse the unit
    """

    for unit in possible_units:
        try:
            u.Unit(unit)
        except ValueError as e:
            raise ValueError(f"Invalid unit: '{unit}'") from e

    if default_unit is not None and default_unit not in possible_units:
        raise ValueError("Default unit must be one of the possible units.")

    cols = st.columns([1, 1.5])
    with cols[0]:
        value: float = st.number_input(
            f"{name or 'Value'}",
            min_value=min_value,
            max_value=max_value,
            value=value,
            step=step,
            format=format,
            width=200,
            disabled=disabled,
            help=help,
            key=f"{key}_value",
        )
    with cols[1]:
        unit: str | None = st.segmented_control(
            unit_input_text,
            possible_units,
            default=possible_units[0] if default_unit is None else default_unit,
            disabled=disabled,
            key=f"{key}_unit",
        )
    if unit is None:
        st.warning("Please select a unit.")
        st.stop()

    if inverse:
        st.session_state[key] = value / u.Unit(unit)
        return value / u.Unit(unit)

    st.session_state[key] = value * u.Unit(unit)
    return value * u.Unit(unit)


def plot_distributions() -> None:

    xs: NDArray = np.linspace(-4, 4, 300)

    # Use TkAgg backend for non-interactive plotting
    plt.switch_backend("TkAgg")

    fig: plt.Figure = plt.figure()
    ax: plt.Axes = fig.gca()

    ax.axhline(0.5, color="black", ls="--", alpha=0.5)

    ax.plot(xs, Gaussian(xs, x0=0, FWHM=1), label="Gaussian")
    ax.plot(xs, Lorentzian(xs, x0=0, FWHM=1), label="Lorentzian")

    ax.plot(xs, SuperGaussian(xs, x0=0, FWHM=1, n=2), label="Super-Gaussian, n=2")

    ax.plot(
        xs,
        Voigt(xs, x0=0, FWHM=1, sigma=0.5, gamma=0.5),
        label="Voigt, sigma=0.5, gamma=0.5",
    )

    ax.plot(
        xs,
        PseudoVoigt(xs, x0=0, FWHM=1, sigma=0.5, gamma=0.5),
        label="Pseudo-Voigt, sigma=0.5, gamma=0.5",
    )

    for beta in [2, 4.765]:
        ax.plot(xs, Moffat(xs, x0=0, FWHM=1, beta=beta), label=f"Moffat, beta={beta}")

    ax.legend()

    plt.show()


if __name__ == "__main__":
    # plot_distributions()
    ...
