from pathlib import Path

import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt

from curved_air.plotting import plot_style
from curved_air.transmission import (
    FlatTransmission,
    Transmission,
    from_file,
    mirror,
    transmission_plot,
    # total_transmission,
    zemax,
    zemax_multi,
)

plt.style.use(plot_style)

DATA_PATH: Path = Path(__file__).parent / ".." / ".." / "transmission_data"


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

    transmission_plot(
        Ts=Ts,
        min_wl=0.37 * u.um,
        max_wl=0.92 * u.um,
        num_wls=500,
    )

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
