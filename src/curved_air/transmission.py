from dataclasses import dataclass, field
from math import factorial
from pathlib import Path
from typing import Literal, Self

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

    _wavelengths: ArrayQuantity = field(default_factory=lambda: [0.55] * u.um)
    _transmissions: NDArray[np.floating] = field(
        default_factory=lambda: np.array([1.0])
    )

    num_passes: int = 1
    smooth: bool = True
    scale_factor: float = 1.0

    _file: Path | None = None

    def __post_init__(self) -> None:
        assert len(self._wavelengths) == len(self._transmissions)

        if isinstance(self._transmissions, list):
            self._transmissions: NDArray = np.array(self._transmissions)

    @property
    def wavelengths(self) -> ArrayQuantity:
        if self.smooth:
            return savitzky_golay(self._wavelengths.to(u.um).value) * u.um
        return self._wavelengths

    @property
    def transmissions(self) -> NDArray[np.floating]:
        if self.smooth:
            try:
                return (
                    savitzky_golay(self._transmissions) * self.scale_factor
                ) ** self.num_passes
            except ValueError:
                raise Warning(
                    "Failed to compute smooth transmission with Savitsky-Golay filter"
                ) from None

        return (self._transmissions * self.scale_factor) ** self.num_passes

    def plot(self) -> None:
        """Plot transmission as a function of wavelength"""

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
        return float(np.max(self.transmissions))

    @property
    def str_peak(self) -> str:
        return f"peak {100 * self.peak:.3g}%"


@dataclass
class FlatTransmission(Transmission):
    transmission: float = 1.0
    smooth: Literal[False] = False

    def __post_init__(self) -> None:
        self._transmissions: NDArray[np.floating] = np.array([self.transmission])

    @property
    def str_peak(self) -> str:
        return f"flat {100 * self.transmissions[0]:.3g}%"


@dataclass
class CSVTransmission(Transmission):
    def load_file(self, file: Path) -> Self:
        assert file.exists(), f"File {file} does not exist"

        self._file: Path = file

        wavelengths, efficiencies = np.transpose(np.genfromtxt(file, delimiter=", "))

        self._wavelengths: ArrayQuantity = wavelengths * u.nm
        self._transmissions: NDArray[np.floating] = efficiencies

        return self


@dataclass
class ZemaxTransmission(Transmission):
    def load_file(self, file: Path) -> Self:
        """
        Load transmission data exported from Zemax, single configuration only.
        """

        assert file.exists(), f"File {file} does not exist"

        self._file: Path = file

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

        self._wavelengths: NDArray = np.array(wavelengths) * u.um
        self._transmissions: NDArray = np.array(transmissions)

        return self


@dataclass
class ZemaxMultiTransmission(Transmission):
    def load_file(self, files: Path | list[Path]) -> Self:
        """ """

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

        self._wavelengths: ArrayQuantity = np.array(wavelengths) * u.um
        self._transmissions: NDArray = np.array(transmissions)

        return self


@dataclass
class GlassTransmission(Transmission):
    glass: Glass = field(default_factory=Glass)
    thickness: ScalarQuantity = 10 * u.um
    reference_thickness: ScalarQuantity = 10 * u.mm

    @property
    def transmission(self) -> NDArray[np.floating]:
        glass_T: NDArray[np.floating] = np.array(
            [self.glass.transmission(wl=wl) for wl in self._wavelengths]
        )

        return (
            np.array(glass_T) ** (self.thickness.to(u.mm).value)
            / self.reference_thickness.to(u.mm).value
        )


@dataclass
class FileTransmission(Transmission):
    def load_file(self, file: Path, wavelength_unit: u.Unit = u.nm) -> Self:
        """
        Load transmission data from a generic file
        """

        wavelengths, transmissions = np.transpose(np.genfromtxt(file))
        self._wavelengths: Quantity = wavelengths * wavelength_unit
        self._transmissions: NDArray[np.floating] = transmissions

        return self


def savitzky_golay(
    y: NDArray[np.floating],
    window_size: int = 11,
    order: int = 3,
    deriv: int = 0,
    rate: float = 1,
) -> NDArray:
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
    Ts: list[Transmission | EchelleGrating | None],
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

    def _plot_transmission(
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
        if isinstance(T, EchelleGrating):
            continue
        assert isinstance(T, Transmission), f"Expected Transmission, got {type(T)}"
        running_transmission: NDArray = _plot_transmission(
            ax=ax, T=T, running_transmission=running_transmission
        )

    if any(isinstance(T, EchelleGrating) for T in Ts):
        # Make sure there is only one EchelleGrating in the list
        assert sum(isinstance(T, EchelleGrating) for T in Ts) == 1, (
            "There can only be one EchelleGrating in the list"
        )
        echelle: EchelleGrating = next(T for T in Ts if isinstance(T, EchelleGrating))

        # Plot nothing for the echelle, but add a legend entry for it
        ax.plot(
            0,
            0,
            color="k",
            alpha=0.25,
            label=f"Echelle (blaze peaks flat {(echelle.peak_efficiency * 100):.3g}%)",
        )
        # echelle: EchelleGrating = EchelleGrating(
        #     catalogue_number="",
        #     groove_frequency=31.6 / u.mm,
        #     blaze_angle=76 * u.deg,
        #     gamma_angle=0.55 * u.deg,
        #     min_wl=0.375 * u.um,
        #     max_wl=0.9 * u.um,
        #     peak_efficiency=0.75,
        # )

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
        if isinstance(_T, EchelleGrating):
            continue
        assert isinstance(_T, Transmission), f"Expected Transmission, got {_T}"
        running_transmission *= _T.T(wl=master_wls)

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


MARVEL_TRANSMISSION: list[Transmission | EchelleGrating | None] = [
    # Telescope
    CSVTransmission(
        name="Telescope mirrors (ASA)",
        num_passes=3,
    ).load_file(
        file=next(iter(DATA_PATH.glob("ASA_Protected_Enhanced_Aluminium.csv")))
    ),
    # Frontend
    ZemaxTransmission(name="Frontend optics (incl. 5% split for guiding)").load_file(
        DATA_PATH / "example_zemax" / "frontend.txt"
    ),
    # FibreAperture
    FlatTransmission(name="Fibre aperture loss", transmission=0.80),
    # Fibre,
    Transmission(
        name="Fibre transmission & coupling from\ncircular to octagonal fibre",
        _wavelengths=np.array([390, 400, 500, 700, 920]) * u.nm,
        _transmissions=np.array(
            [
                0.91 ** (15 / 10) * 0.92,
                0.920 ** (15 / 10) * 0.92,
                0.963 ** (15 / 10) * 0.92,
                0.987 ** (15 / 10) * 0.92,
                0.991 ** (15 / 10) * 0.92,
            ]
        ),
    ),
    # Spectrograph,
    ZemaxMultiTransmission(name="Spectrograph, excluding echelle").load_file(
        list((DATA_PATH / "example_zemax_multi").glob("c[123456789].txt"))
    ),
    # Detector,
    FileTransmission(name="STA1600LN Detector QE").load_file(
        file=next(iter(DATA_PATH.glob("STA1600LN*.txt"))),
        wavelength_unit=u.nm,
    ),
    # Echelle,
    EchelleGrating(
        name="MARVEL nominal",
        catalogue_number="53-*-417E",
        groove_frequency=52.67 / u.mm,
        blaze_angle=69.7 * u.deg,
        gamma_angle=0.3 * u.deg,
        min_wl=370 * u.nm,
        max_wl=950 * u.nm,
        peak_efficiency=0.75,
    ),
    None,
]
