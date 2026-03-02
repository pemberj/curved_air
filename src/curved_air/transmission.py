from dataclasses import dataclass, field
from math import factorial
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.units import Quantity
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from curved_air.glass import Glass
from curved_air.plotting import plot_style
from curved_air.type_hinting import ArrayQuantity, ScalarQuantity

plt.style.use(plot_style)


DATA_PATH: Path = Path(__file__).parent / "transmission_data"


@dataclass
class Transmission:
    name: str
    wavelengths: Quantity = field(default_factory=lambda: [0.55] * u.um)
    transmissions: list[float] | NDArray = field(default_factory=lambda: [1.0])

    def __post_init__(self) -> None:
        assert len(self.wavelengths) == len(self.transmissions)

    #     self.wavelengths = self.wavelengths * 1e-9
    #     self.transmissions = self.transmissions / 100

    def plot(self) -> None:
        """
        Plot transmission as a function of wavelength
        """
        plt.plot(self.wavelengths, self.transmissions, label=self.name)
        plt.xlabel("Wavelength [microns]")
        plt.ylabel("Transmission [%]")
        plt.legend()
        plt.show()

    def T(self, wl: Quantity) -> np.ndarray:
        if isinstance(wl, Quantity):
            wl: float = wl.to(u.um).value

        return np.interp(
            x=wl, xp=self.wavelengths.to(u.um).value, fp=self.transmissions
        )

    @property
    def peak(self) -> float:
        return np.max(self.transmissions)

    @property
    def str_peak(self) -> str:
        return f"peak {100 * self.peak:.1f}%"


@dataclass
class FlatTransmission(Transmission):
    transmission: float = 1.0

    def __post_init__(self) -> None:
        self.transmissions: list[float] = [self.transmission]

    @property
    def str_peak(self) -> str:
        return f"flat {100 * self.transmissions[0]:.1f}%"


def mirror(
    file: Path,
    name: str,
    num_reflections: int = 1,
    smooth: bool = True,
) -> Transmission:
    assert file.exists(), f"File {file} does not exist"

    wavelengths, efficiencies = np.transpose(np.genfromtxt(file, delimiter=", "))

    wavelengths: Quantity = wavelengths * u.nm
    efficiencies: np.ndarray = efficiencies**num_reflections

    if smooth:
        efficiencies: np.ndarray = savitzky_golay(efficiencies)

    return Transmission(
        name=name,
        wavelengths=wavelengths.to(u.um),
        transmissions=efficiencies,
    )


def zemax(
    file: Path,
    name: str,
) -> Transmission:
    """
    Load transmission data exported from Zemax, single configuration only.
    """
    if file is None:
        file: Path = next(iter((DATA_PATH / "example_zemax").glob("frontend.txt")))

    assert file.exists(), f"File {file} does not exist"

    wavelengths: list[float] = []
    transmissions: list[float] = []

    df: pd.DataFrame = pd.read_csv(
        file,
        skiprows=14,
        delimiter="    ",
        encoding="utf-16",
        names=["a", "b", "c"],
        engine="python",
    )

    wls: list[float] = [i.rstrip(":\t") for i in df["b"][:12].values]
    trs: list[float] = [i.rstrip(":\t") for i in df["c"][:12].values]

    for wl in wls:
        wavelengths.append(float(wl))
    for tr in trs:
        transmissions.append(float(tr))

    wavelengths: np.ndarray = np.array(wavelengths) * u.um
    transmissions: np.ndarray = np.array(transmissions)

    return Transmission(name=name, wavelengths=wavelengths, transmissions=transmissions)


def zemax_multi(
    files: Path | list[Path],
    name: str,
) -> Transmission:
    if isinstance(files, Path):
        files: list[Path] = list(files.glob("*.txt"))

    assert isinstance(files, list), "Files must be a list of Paths"
    for f in files:
        assert f.exists(), f"File {f} does not exist"
    assert len(files) > 0, "You must provide at least one file"

    wavelengths: list[float] = []
    transmissions: list[float] = []

    for f in files:
        # config_num = int(f[len("./spectrograph/") :].split("c")[1].split(".")[0])
        df: pd.DataFrame = pd.read_csv(
            f,
            skiprows=14,
            delimiter="    ",
            encoding="utf-16",
            names=["a", "b", "c"],
            engine="python",
        )

        wls: list[float] = [i.rstrip(":\t") for i in df["b"][:5].values]
        trs: list[float] = [i.rstrip(":\t") for i in df["c"][:5].values]

        for wl in wls:
            wavelengths.append(float(wl))
        for tr in trs:
            transmissions.append(float(tr))

    wavelengths: ArrayQuantity = np.array(wavelengths) * u.um
    transmissions: NDArray = np.array(transmissions)

    return Transmission(
        name=name,
        wavelengths=wavelengths,
        transmissions=transmissions,
    )


def from_file(
    name: str,
    file: Path,
    wavelength_unit: u.Unit = u.nm,
) -> Transmission:
    wavelengths, transmissions = np.transpose(np.genfromtxt(file))
    wavelengths: Quantity = wavelengths * wavelength_unit

    return Transmission(
        name=name,
        wavelengths=wavelengths,
        transmissions=transmissions,
    )


def glass(
    name: str,
    wavelengths: ArrayQuantity,
    glass: Glass,
    thickness: ScalarQuantity,
) -> Transmission:
    """
    Compute the transmission of a glass component, given the glass type and thickness.
    """

    transmissions: list[float] = [glass.transmission(wl=wl) for wl in wavelengths]

    # Assuming 10mm is the thickness for which the transmission is given! Should check...
    transmissions: np.ndarray = (
        np.array(transmissions) ** (thickness.to(u.mm).value) / 10
    )

    return Transmission(
        name=name,
        wavelengths=wavelengths,
        transmissions=transmissions,
    )


def savitzky_golay(
    y: np.ndarray,
    window_size: int = 11,
    order: int = 3,
    deriv: int = 0,
    rate: float = 1,
) -> np.ndarray:
    """
    https://stackoverflow.com/questions/22988882/how-to-smooth-a-curve-in-python
    https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter
    """

    assert window_size % 2 == 1 and window_size >= 1, (
        "Window_size size must be a positive odd number"
    )
    assert window_size >= order + 2, (
        "Window_size is too small for the polynomials order"
    )

    order_range = range(order + 1)
    half_window = (window_size - 1) // 2

    # precompute coefficients
    b = np.asmatrix(
        [[k**i for i in order_range] for k in range(-half_window, half_window + 1)]
    )
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)

    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1 : half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1 : -1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode="valid")
