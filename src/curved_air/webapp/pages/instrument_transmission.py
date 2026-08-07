from pathlib import Path

import numpy as np
import streamlit as st
from astropy import units as u
from astropy.units import Quantity
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from curved_air.optics.dispersers import EchelleGrating
from curved_air.plotting import plot_style
from curved_air.transmission import (
    FlatTransmission,
    Transmission,
    from_file,
    mirror,
    zemax,
    zemax_multi,
)

plt.style.use(plot_style)

DATA_PATH: Path = Path(__file__).parent / ".." / ".." / "transmission_data"


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


def main() -> None:
    Telescope: Transmission = mirror(
        file=next(iter(DATA_PATH.glob("ASA_Protected_Enhanced_Aluminium.csv"))),
        name="Telescope mirrors (ASA)",
        num_reflections=3,
    )

    Frontend: Transmission = zemax(
        file=DATA_PATH / "example_zemax" / "frontend.txt",
        name="Frontend optics (incl. 5% split for guiding)",
    )

    FibreAperture: Transmission = FlatTransmission(
        name="Fibre aperture loss", transmission=0.80
    )

    Fibre: Transmission = Transmission(
        name="Fibre transmission & coupling from\ncircular to octagonal fibre",
        wavelengths=np.array([390, 400, 500, 700, 920]) * u.nm,
        transmissions=np.array(
            [
                0.91 ** (15 / 10) * 0.92,
                0.920 ** (15 / 10) * 0.92,
                0.963 ** (15 / 10) * 0.92,
                0.987 ** (15 / 10) * 0.92,
                0.991 ** (15 / 10) * 0.92,
            ]
        ),
    )

    Slicer = FlatTransmission(name="Slicer/double scrambler", transmission=0.9)

    Spectrograph: Transmission = zemax_multi(
        files=list((DATA_PATH / "example_zemax_multi").glob("c[123456789].txt")),
        name="Spectrograph, excluding echelle",
    )

    Detector: Transmission = from_file(
        name="STA1600LN Detector QE",
        file=next(iter(DATA_PATH.glob("STA1600LN*.txt"))),
        wavelength_unit=u.nm,
    )

    Ts: list[Transmission | None] = [
        # atmosphere(), # Not yet implemented
        Telescope,
        Frontend,
        FibreAperture,
        Fibre,
        Slicer,
        Spectrograph,
        Detector,
        None,
        None,
        None,
    ]

    fig: plt.Figure = transmission_plot(
        Ts=Ts,
        min_wl=0.37 * u.um,
        max_wl=0.92 * u.um,
        num_wls=500,
    )

    st.pyplot(fig)

    # total: tuple[np.ndarray, np.ndarray] = total_transmission(
    #     min_wl=0.37 * u.um,
    #     max_wl=0.92 * u.um,
    #     num_wls=500,
    #     save=False,
    # )

    # np.savetxt(
    #     "2ES_total_transmission.csv",
    #     np.transpose(total),
    #     fmt="%.5f",
    #     delimiter=",",
    #     header="Wavelength (nm), Transmission",
    # )


if __name__ == "__main__":
    main()
