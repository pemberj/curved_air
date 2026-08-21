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
from curved_air.optics.dispersers import EchelleGrating
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


def transmission_plot(
    Ts: list[Transmission | None],
    min_wl: Quantity = 370 * u.nm,
    max_wl: Quantity = 950 * u.nm,
    num_wls: int = 200,
) -> plt.Figure:
    """
    Generate a plot of the progressive transmission of the MARVEL spectrograph, i.e.
    show the curve of transmission as light passes through each component in the system.
    """

    master_wls: NDArray = (
        np.linspace(min_wl.to(u.nm).value, max_wl.to(u.nm).value, num_wls) * u.nm
    )

    running_transmission: NDArray = np.ones(num_wls)

    fig: plt.Figure = plt.figure(figsize=(7, 9), tight_layout=True)
    ax: plt.Axes = fig.gca()

    def plot_transmission(
        ax: plt.Axes,
        T: Transmission,
        running_transmission: NDArray,
    ) -> np.ndarray:
        x: NDArray = master_wls
        y: NDArray = running_transmission * T.T(wl=master_wls)

        ax.plot(x.to(u.um).value, y, label=f"{T.name} ({T.str_peak})", alpha=0.75)
        ax.fill_between(x.to(u.um).value, 0, y, color="k", alpha=0.05)

        return y

    # Loop over all the components and plot their transmission
    for T in Ts:
        if T is None:
            continue
        print(f"Computing {T.name} transmission...")
        running_transmission: NDArray = plot_transmission(
            ax=ax, T=T, running_transmission=running_transmission
        )

    # Plot the echelle blaze function last
    print("Computing blaze function of echelle orders...")
    ax.plot(0, 0, color="k", alpha=0.25, label="Echelle (blaze peaks flat 75%)")
    echelle: EchelleGrating = EchelleGrating(
        catalogue_number="",
        groove_frequency=31.6 / u.mm,
        blaze_angle=76 * u.deg,
        gamma_angle=0.55 * u.deg,
        min_wl=0.375 * u.um,
        max_wl=0.9 * u.um,
        peak_efficiency=0.75,
    )

    blaze_efficiencies: dict[int, tuple[Quantity, list[float]]] = (
        echelle.blaze_efficiency()
    )

    for _m, (wl, be) in blaze_efficiencies.items():
        R_T: NDArray = np.interp(x=wl, xp=master_wls, fp=running_transmission)
        ax.fill_between(
            x=wl.to(u.um).value,
            y1=0,
            y2=R_T * be,
            color="k",
            alpha=0.25,
        )

    ax.legend(loc="upper right")

    ax.set_xlim(np.min(master_wls.to(u.um).value), np.max(master_wls.to(u.um).value))
    ax.set_ylim(0, 1.25)
    # ax.set_yscale("log")

    ax.set_xlabel("Wavelength (microns)")
    ax.set_ylabel("Transmission")

    return fig


def total_transmission(
    T: list[Transmission | None],
    min_wl: Quantity = 370 * u.nm,
    max_wl: Quantity = 950 * u.nm,
    num_wls: int = 300,
    save: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute and plot the total transmission of the system. Optionally save the plot, and
    return the data in tabulated form for incorporation into a PyEchelle model or
    similar.
    """

    master_wls: NDArray = (
        np.linspace(min_wl.to(u.um).value, max_wl.to(u.um).value, num_wls) * u.um
    )

    running_transmission: NDArray = np.ones(num_wls)

    for _T in T:
        if _T is None:
            continue
        running_transmission *= T.T(wl=master_wls)

    # Account for echelle blaze function (flat at peak value only)
    # This will be used to add total system efficiency to PyEchelle, where the
    # shape of the blaze function is already accounted for
    running_transmission *= FlatTransmission(name="", transmission=0.75).T(
        wl=master_wls
    )

    fig: plt.Figure = plt.figure(
        "Total transmission", figsize=(7, 5), tight_layout=True
    )
    ax: plt.Axes = fig.gca()
    ax.plot(master_wls, running_transmission)
    ax.set_xlim(np.min(master_wls).value, np.max(master_wls).value)
    ax.set_ylim(0, ax.get_ylim()[1])
    # ax.set_title("MARVEL Total Transmission")
    ax.set_xlabel("Wavelength (microns)")
    ax.set_ylabel("Fractional transmission")

    if save:
        for ext in ["png", "pdf"]:
            plt.savefig(f"2ES_total_transmission.{ext}")

    else:
        plt.show()

    return master_wls.to(u.um).value, running_transmission
