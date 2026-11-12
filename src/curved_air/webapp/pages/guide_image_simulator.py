#! uv run streamlit run

from itertools import cycle
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from astropy import units as u
from astropy.modeling.models import BlackBody
from astropy.units import Quantity
from numpy.typing import NDArray
from plotly import graph_objects as go

from curved_air.detectors import DETECTORS, Detector
from curved_air.plotting import plot_style
from curved_air.type_hinting import ArrayQuantity, ScalarQuantity
from curved_air.utils import st_quantity_input
from curved_air.webapp import badges, bmac

plt.style.use(plot_style)

SCRIPT_PATH: Path = Path(__file__).parent
DATA_PATH: Path = SCRIPT_PATH / ".." / ".." / "data"


# This dictionary provides a reference for the absolute flux of magnitude 0 stars across
# several bands. There is probably a better way to do this...
band_properties: dict[str, dict[str, ScalarQuantity]] = {
    "U": {
        "lambda_eff": 365 * u.nm,
        "dlambda": 66 * u.nm,
        "standard_flux": 1810 * u.Jy,  # mag 0
    },
    "B": {
        "lambda_eff": 445 * u.nm,
        "dlambda": 94 * u.nm,
        "standard_flux": 4260 * u.Jy,  # mag 0
    },
    "V": {
        "lambda_eff": 551 * u.nm,
        "dlambda": 88 * u.nm,
        "standard_flux": 3640 * u.Jy,  # mag 0
    },
    "R": {
        "lambda_eff": 658 * u.nm,
        "dlambda": 138 * u.nm,
        "standard_flux": 3080 * u.Jy,  # mag 0
    },
}


mirror_coatings: list[Path] = [
    DATA_PATH / "ASA_Protected_Enhanced_Aluminium.csv",
]


def gaussian_seeing(
    fwhm: ScalarQuantity = 1.0 * u.arcsec,
    center: tuple[float, float] = (0.0, 0.0),
    plate_scale: ScalarQuantity = 0.08594 * u.arcsec / u.micron,
    pixel_size: ScalarQuantity = 3.45 * u.micron,
    num_pixels: int = 50,
) -> NDArray:
    """
    From the input parameters, creates a 2D array (image) of the appropriate size of the
    seeing disk (parametrised with an angular FWHM) as imaged through a telescope onto a
    detector.
    """

    sigma: ScalarQuantity = fwhm / (2 * np.sqrt(2 * np.log(2)))
    sigma_pixels: ScalarQuantity = sigma / plate_scale / pixel_size

    x: NDArray = np.linspace(-num_pixels / 2, num_pixels / 2 - 1, num_pixels)
    # x = np.arange(start=-num_pixels / 2, stop=num_pixels / 2, step=1)
    y: NDArray = x[:, np.newaxis]

    gaussian: NDArray = np.exp(
        -((x - center[0]) ** 2 + (y - center[1]) ** 2) / (2 * sigma_pixels**2)
    )
    return gaussian / np.sum(gaussian)


def PSF(
    fwhm: ScalarQuantity = 3.0 * u.micron,
    pixel_size: ScalarQuantity = 3.45 * u.micron,
    num_pixels: int = 20,
) -> NDArray:
    """
    Creates a 2D array (image) of a Gaussian point spread function.
    """

    sigma: ScalarQuantity = fwhm / (2 * np.sqrt(2 * np.log(2)))
    sigma_pixels: ScalarQuantity = sigma / pixel_size

    x: NDArray = np.linspace(-num_pixels / 2, num_pixels / 2 - 1, num_pixels)
    # x = np.arange(start=-num_pixels / 2, stop=num_pixels / 2, step=1)
    y: NDArray = x[:, np.newaxis]

    gaussian: NDArray = np.exp(-(x**2 + y**2) / (2 * sigma_pixels**2))
    return gaussian / np.sum(gaussian)


def photon_flux(
    magnitude: ScalarQuantity = 10 * u.mag,
    wavelengths: ArrayQuantity | None = None,
    temperature: ScalarQuantity = 7000 * u.K,
) -> NDArray:
    """
    A simple calculation of the photon flux from a star of a given magnitude.
    Uses standard value of V-band flux for m=0 star from Bessel 1979

    Calulcations as per
    https://lweb.cfa.harvard.edu/~dfabricant/huchra/ay145/mags.html

    TODO: Make this more accurate by using the actual bandpass of the filter?
    TODO: Make this variable with spectral type!
    """

    if wavelengths is None:
        np.linspace(380, 950, 100) * u.nm

    assert isinstance(wavelengths, Quantity)

    dlambda = (wavelengths[1].value - wavelengths[0].value) * u.nm

    # Calculating the blackbody radiation of the observed star over the full wavelenghth
    # range and converting to photon/s/m²/sr
    bb = BlackBody(temperature)
    bbJy: ArrayQuantity = bb(wavelengths).to(u.Jy / u.sr)
    bbPh: ArrayQuantity = (
        bbJy.to(
            u.photon / u.s / u.m**2 / u.nm / u.sr,
            equivalencies=u.spectral_density(wavelengths),
        )
        * dlambda
    )

    # The same calculation but only for the V band, this will be used to scale the flux
    # V band wavelength range and spacing between wavelengths
    wave_V: ArrayQuantity = np.arange(492, 580, dlambda.to(u.nm).value) * u.nm
    bbJy_V: ArrayQuantity = bb(wave_V).to(u.Jy / u.sr)
    bbPh_V: ArrayQuantity = np.sum(
        bbJy_V.to(
            u.photon / u.s / u.m**2 / u.nm / u.sr,
            equivalencies=u.spectral_density(wave_V),
        )
        * dlambda
    )

    # Reference flux for a magnitude 0 star in the V band
    ref_V: ScalarQuantity = band_properties["V"]["standard_flux"]

    # Scaling the flux from the blackbody function -> units: photon/s/m²
    ref_V_ph: ScalarQuantity = np.sum(
        ref_V.to(
            u.photon / u.s / u.m**2 / u.nm,
            equivalencies=u.spectral_density(wave_V),
        )
        * dlambda
    )

    scaler = ref_V_ph / bbPh_V
    bbPh_scaled = bbPh * scaler

    # Adjusting the flux for the correct apparent magnitude
    return bbPh_scaled * 10 ** (magnitude.value / -2.512)


def telescope_area(
    telescope_diameter: ScalarQuantity = 0.8 * u.meter,
    secondary_diameter: ScalarQuantity = 0.2 * u.meter,
) -> ScalarQuantity:
    """
    Calculates the area of the telescope aperture.
    """

    return np.pi * (telescope_diameter / 2) ** 2 - np.pi * (secondary_diameter / 2) ** 2


def read_noise(
    image: NDArray | None = None,
    num_pixels: int = 50,
    noise_level: ScalarQuantity = 2.2 * u.electron,
    plot_hist: bool = False,
) -> NDArray:
    """
    Computes and returns a 2D array (image) containing the read noise of a CMOS
    detector.

    Assumes Poissonian noise, which seems to be appropriate for a CMOS detector.
    """

    if not image:
        image: NDArray = np.zeros((num_pixels, num_pixels))

    # For a CCD - Gaussian noise
    # noise = np.random.normal(0, noise_level.to(u.electron).value, image.shape)

    # For a CMOS - Poissonian noise?
    noise: NDArray = np.random.poisson(
        lam=noise_level.to(u.electron).value,
        size=image.shape,
    )

    if plot_hist:
        plt.hist(noise.ravel(), bins=100, label="Read noise")
        plt.legend()
        plt.show()

    return noise


def dark_noise(
    image: NDArray | None = None,
    num_pixels: int = 50,
    exposure_time: ScalarQuantity = 2 * u.second,
    dark_noise: ScalarQuantity = 3.2 * u.electron / u.second,
    plot_hist: bool = False,
) -> NDArray:
    """
    Computes and returns a 2D array (image) containing the dark-current induced
    noise of a CMOS detector.
    """

    if not image:
        image: NDArray = np.zeros((num_pixels, num_pixels))

    DC: NDArray = np.random.poisson(
        lam=(dark_noise * exposure_time).to(u.electron).value,
        size=image.shape,
    )

    if plot_hist:
        plt.hist(DC.ravel(), bins=100, label="Dark noise")
        plt.legend()
        plt.show()

    return DC


def compute_centroid(
    image: NDArray,
    ROI_size: int | None = None,
) -> tuple[float, float]:
    """
    Computes the intensity-weighted centroid of an image.

    Optionally can take a region of interest (ROI) size to crop the image.
    """
    original_image_shape = image.shape

    if ROI_size:
        # Crop the image to the region of interest
        image: NDArray = image[
            image.shape[0] // 2 - ROI_size // 2 : image.shape[0] // 2 + ROI_size // 2,
            image.shape[1] // 2 - ROI_size // 2 : image.shape[1] // 2 + ROI_size // 2,
        ]

    x: NDArray = np.arange(0, image.shape[0], 1, float)
    y: NDArray = np.arange(0, image.shape[1], 1, float)
    x, y = np.meshgrid(x, y)
    total_intensity: float = np.sum(image)

    centroid_x: int = np.sum(x * image) / total_intensity
    centroid_y: int = np.sum(y * image) / total_intensity

    if ROI_size:
        centroid_x += original_image_shape[0] // 2 - ROI_size // 2
        centroid_y += original_image_shape[1] // 2 - ROI_size // 2

    return centroid_x, centroid_y


def square_coordinates(w: float = 50) -> tuple[NDArray, NDArray]:
    xs: NDArray = np.array([w / 2, w / 2, -w / 2, -w / 2, w / 2])
    ys: NDArray = np.array([w / 2, -w / 2, -w / 2, w / 2, w / 2])

    return xs, ys


def mirror_reflectivity(
    csv: Path | str | float,
    num_mirrors: int = 3,
    wavelengths: NDArray | None = None,
) -> NDArray:
    if wavelengths is None:
        wavelengths: ArrayQuantity = np.linspace(380, 950, 100) * u.nm

    if isinstance(csv, float):
        return np.ones_like(wavelengths.value) * csv**num_mirrors

    assert isinstance(csv, (str, Path))

    data: pd.DataFrame = pd.read_csv(csv, names=["wavelength", "reflectivity"])
    wave_data = data["wavelength"]
    efficiency = data["reflectivity"]

    refl_full: NDArray = np.interp(
        wavelengths.value, wave_data, efficiency, left=efficiency[0]
    )

    return refl_full**num_mirrors


def detector_QE(
    csv: Path | str | float | None = None,
    wavelengths: NDArray | None = None,
) -> NDArray:
    if wavelengths is None:
        wavelengths: ArrayQuantity = np.linspace(380, 950, 100) * u.nm

    if csv is None:
        return np.ones_like(wavelengths.value)

    if isinstance(csv, float):
        return np.ones_like(wavelengths.value) * csv

    assert isinstance(csv, (str, Path))

    data: pd.DataFrame = pd.read_csv(csv, names=["wavelength", "QE"], skiprows=1)
    wave_QE_data = data["wavelength"]
    quantum_efficiency = data["QE"]

    return np.interp(wavelengths.value, wave_QE_data, quantum_efficiency)


def offset(
    center_x: float = 35.0,
    center_y: float = 35.0,
    # pixel_size: Quantity = 3.45 * u.um,
) -> Quantity:
    delta_x: float = center_x - 35.0
    delta_y: float = center_y - 35.0
    return np.sqrt(delta_x**2 + delta_y**2)


def SNR(
    noise: NDArray,
    signal: NDArray,
) -> Quantity:
    return np.sum(signal) / np.sum(noise)


def guiding_efficiency(
    curve: Path | str | float = 0.05,
    wavelengths: NDArray | None = None,
) -> NDArray:
    if wavelengths is None:
        wavelengths: ArrayQuantity = np.linspace(380, 950, 100) * u.nm

    if isinstance(curve, float):
        return np.ones_like(wavelengths.value) * curve

    assert isinstance(curve, (str, Path))

    data: pd.DataFrame = pd.read_csv(curve, names=["wavelength", "efficiency"])
    wave_data = data["wavelength"]
    efficiency = data["efficiency"]

    return np.interp(wavelengths.value, wave_data, efficiency, left=efficiency[0])


def simulate_image(
    magnitude: float | ScalarQuantity = 11 * u.mag,
    blackbody_temperature: float | ScalarQuantity = 3200 * u.K,
    seeing_fwhm: float | ScalarQuantity = 2 * u.arcsec,
    psf_fwhm: float | ScalarQuantity = 2.5 * u.micron,
    exposure_time: float | ScalarQuantity = 0.1 * u.second,
    sky_background: float | ScalarQuantity = 22 * u.mag / u.arcsec**2,
    telescope_diameter: float | ScalarQuantity = 0.8 * u.meter,
    secondary_diameter: float | ScalarQuantity = 0.2 * u.meter,
    telescope_num_mirrors: int = 3,
    telescope_mirror_reflectivity: Path | str | float = 0.9,
    guide_focal_ratio: float = 3.5,
    guiding_split: Path | str | float = 0.05,
    detector: Detector = DETECTORS["IMX174"],
    image_size: int = 44,
    centroid_ROI: int = 30,
    fibre_size: float | ScalarQuantity | None = None,
) -> plt.Figure:
    """
    The main script.

    Simulates a star image on a detector, with all of the relevant parameters
    describing the brightness of the star, the telescope, the detector, etc.,
    and then computes the centroid of the resulting image.
    """

    if isinstance(magnitude, int | float):
        magnitude *= u.mag

    if isinstance(blackbody_temperature, int | float):
        blackbody_temperature *= u.K

    if isinstance(seeing_fwhm, int | float):
        seeing_fwhm *= u.arcsec

    if isinstance(psf_fwhm, int | float):
        psf_fwhm *= u.micron

    if isinstance(exposure_time, int | float):
        exposure_time *= u.second

    if isinstance(sky_background, int | float):
        sky_background *= u.mag / u.arcsec**2

    if isinstance(telescope_diameter, int | float):
        telescope_diameter *= u.meter

    if isinstance(secondary_diameter, int | float):
        secondary_diameter *= u.meter

    if isinstance(fibre_size, int | float):
        fibre_size *= u.micron

    wavelengths: ArrayQuantity = np.linspace(380, 950, 100) * u.nm

    plate_scale: ScalarQuantity = (
        1 * u.radian / (telescope_diameter * guide_focal_ratio)
    )
    plate_scale: ScalarQuantity = plate_scale.to(u.arcsec / u.um)

    exposure: NDArray = gaussian_seeing(
        fwhm=seeing_fwhm,
        plate_scale=plate_scale,
        pixel_size=detector.pixel_size,
        num_pixels=image_size,
    )

    psf: NDArray = PSF(fwhm=psf_fwhm)

    exposure: NDArray = cv2.filter2D(exposure, -1, psf)

    _photon_flux = photon_flux(
        magnitude=magnitude,
        wavelengths=wavelengths,
        temperature=blackbody_temperature,
    )

    _telescope_efficiency = mirror_reflectivity(
        wavelengths=wavelengths,
        num_mirrors=telescope_num_mirrors,
        csv=telescope_mirror_reflectivity,
    )

    _guiding_efficiency = guiding_efficiency(
        curve=guiding_split,
        wavelengths=wavelengths,
    )

    _detector_QE = detector_QE(
        wavelengths=wavelengths,
        csv=detector.QE,
    )

    flux = (
        _photon_flux
        * telescope_area(
            telescope_diameter=telescope_diameter,
            secondary_diameter=secondary_diameter,
        )
        * _telescope_efficiency
        * _guiding_efficiency
        * exposure_time
        * detector.gain
        * _detector_QE
    )
    exposure: NDArray = exposure * np.sum(flux)

    dark: NDArray = dark_noise(
        num_pixels=image_size,
        exposure_time=exposure_time,
        dark_noise=detector.dark_noise,
    )

    read: NDArray = read_noise(
        num_pixels=image_size,
        noise_level=detector.read_noise,
    )

    exposure += dark + read

    exposure -= np.median(dark) + np.median(read)

    fig: plt.Figure = plt.figure()
    ax: plt.Axes = fig.gca()

    ax.imshow(exposure)

    if fibre_size is not None:
        fibre = plt.Circle(
            (image_size // 2, image_size // 2),
            radius=((fibre_size / 2) / detector.pixel_size).si.value,
            color="red",
            fill=False,
            label="Science fibre",
        )
        ax.add_artist(fibre)

    cenx, ceny = compute_centroid(
        image=exposure,
        ROI_size=centroid_ROI,
    )

    ax.plot(
        cenx,
        ceny,
        lw=0,
        marker=".",
        markersize=15,
        color="red",
        label=f"Computed centroid: ({cenx:.2f}, {ceny:.2f})",
    )

    xs, ys = square_coordinates(
        w=centroid_ROI,
    )
    ax.plot(
        image_size / 2 + xs,
        image_size / 2 + ys,
        ls="--",
        color="w",
        alpha=0.5,
        label="ROI for centroid computation",
    )

    ax.plot(
        image_size / 2,
        image_size / 2,
        lw=0,
        marker="x",
        markersize=15,
        color="blue",
        label=f"True centroid: ({image_size / 2}, {image_size / 2})",
    )

    ax.legend(frameon=True, loc="lower center", fontsize=12)
    ax.set_title(
        "Simulated guiding image\n"
        f"Seeing FWHM = {seeing_fwhm:.2g}, "
        f"V={magnitude.value:.3g} Blackbody of T={blackbody_temperature:.4g}\n"
        f"{exposure_time:.3g} exposure on {detector.name} detector",
        size=17,
    )
    ax.set_xticks([])
    ax.set_yticks([])

    return fig


# Now start the actual webapp


st.title("Guide Image Simulator")

with st.expander("About this tool", expanded=False):
    f"""
    This tool generates simulated images for a telescope guiding system.

    You can use this tool for top-level design and specification of a guiding system,
    given target stars of different spectral type and brightness. The script takes into
    account the telescope size, number of mirrors, mirror coating, focal ratio,
    detector, and other parameters to compute a simulated image of a star on the guide
    detector. The simulated image is displayed with an overlay showing the true and
    computed centroid of the star image.

    Currently this tool only simulates with a set wavelength range of 380-950 nm. Please
    get in contact if you would like to see this configurable to extend to other
    bandpasses.

    All code is written by Jake Pember {badges}

    Please get in contact if you have any questions, suggestions, or find any bugs!

    You can also [Buy me a pint](https://buymeacoffee.com/pemberj) if you find this tool
    useful and want to support its development!
    """

    bmac()

settings_column, output_column = st.columns(2)

with settings_column:
    with st.expander("Target", expanded=True):
        cols = cycle(st.columns(2))

        with next(cols):
            st.number_input(
                "Magnitude",
                value=10.0,
                step=0.1,
                width=200,
                key="stellar_magnitude",
            )

        with next(cols):
            st.number_input(
                "Blackbody temperature (K)",
                value=5777,
                step=1,
                width=200,
                key="stellar_temperature",
            )

        with st.popover("Show blackbody spectrum"):
            wavelengths: ArrayQuantity = np.linspace(380, 950, 30) * u.nm
            blackbody_flux: NDArray = photon_flux(
                magnitude=st.session_state.get("stellar_magnitude", 10.0) * u.mag,
                wavelengths=wavelengths,
                temperature=st.session_state.get("stellar_temperature", 5777) * u.K,
            )

            fig: go.Figure = go.Figure()
            fig.add_trace(
                trace=go.Scatter(
                    x=wavelengths.to(u.nm).value,
                    y=blackbody_flux,
                    mode="lines",
                )
            )

            fig.update_layout(
                title=f"Blackbody spectrum T={st.session_state.get('stellar_temperature', 5777)} K",
                xaxis_title="Wavelength (nm)",
                yaxis_title="Blackbody flux (photons/s/m²)",
            )

            st.plotly_chart(fig)

    with st.expander("Seeing", expanded=True):
        cols = cycle(st.columns([2, 1]))

        with next(cols):
            st_quantity_input(
                "Angular seeing FWHM",
                min_value=0.0,
                value=1.0,
                step=0.1,
                default_unit="arcsec",
                possible_units=["arcsec", "arcmin", "mrad", "degree"],
                key="seeing_fwhm",
            )

    with st.expander("Telescope", expanded=True):
        cols = cycle(st.columns(2))

        with next(cols):
            st_quantity_input(
                "Primary mirror diameter",
                min_value=0.0,
                value=0.8,
                step=0.1,
                default_unit="m",
                possible_units=["mm", "cm", "m"],
                key="telescope_diameter",
            )

            st_quantity_input(
                "Secondary mirror diameter",
                min_value=0.0,
                value=0.2,
                step=0.1,
                default_unit="m",
                possible_units=["mm", "cm", "m"],
                key="telescope_secondary_diameter",
            )

        with next(cols):
            st.number_input(
                "Number of mirrors",
                min_value=1,
                value=3,
                step=1,
                width=200,
                key="telescope_num_mirrors",
            )

            st.selectbox(
                "Mirror coating",
                options=[*[coating.name for coating in mirror_coatings], "Custom"],
                index=0,
                width=400,
                key="telescope_mirror_coating",
            )

            if st.session_state.get("telescope_mirror_coating", "Custom") != "Custom":
                coating: Path = next(
                    c
                    for c in mirror_coatings
                    if c.name == st.session_state["telescope_mirror_coating"]
                )

                st.session_state["telescope_mirror_reflectivity"] = coating

                with st.popover("Show mirror reflectivity curve"):
                    # Display a small plot of the mirror reflectivity curve

                    wavelengths: ArrayQuantity = np.linspace(380, 950, 30) * u.nm
                    reflectivity: NDArray[np.floating] = mirror_reflectivity(
                        coating, wavelengths=wavelengths
                    )

                    fig: go.Figure = go.Figure()
                    fig.add_trace(
                        trace=go.Scatter(
                            x=wavelengths.to(u.nm).value,
                            y=reflectivity,
                            mode="lines",
                        )
                    )

                    fig.update_layout(
                        title=f"{coating.name}",
                        xaxis_title="Wavelength (nm)",
                        yaxis_title="Reflectivity",
                    )

                    st.plotly_chart(fig)

            else:
                st.file_uploader(
                    "Upload mirror reflectivity curve (CSV)",
                    type=["csv"],
                    key="telescope_mirror_reflectivity",
                    help="CSV file with two columns: wavelength (nm) and reflectivity (0-1)",
                )

    with st.expander("Guide system", expanded=True):
        cols = cycle(st.columns(2))

        with next(cols):
            st.number_input(
                "Guide system focal ratio",
                min_value=0.1,
                value=3.5,
                step=0.1,
                width=200,
                key="guide_focal_ratio",
            )

            st.session_state["guide_plate_scale"] = (
                1
                * u.radian
                / (
                    st.session_state["telescope_diameter"]
                    * st.session_state["guide_focal_ratio"]
                )
            ).to(u.arcsec / u.um)

            st_quantity_input(
                "PSF size (FWHM)",
                min_value=0.0,
                value=10.0,
                step=0.1,
                default_unit="µm",
                possible_units=["µm", "mm"],
                key="psf_fwhm",
            )

        with next(cols):
            st.number_input(
                "Guiding split",
                min_value=0.0,
                max_value=1.0,
                value=0.05,
                step=0.01,
                width=200,
                key="guiding_split",
                help="The fraction of light to be sent to the guide camera",
            )

            st_quantity_input(
                "Exposure time",
                min_value=0.01,
                value=0.25,
                default_unit="s",
                possible_units=["ms", "s", "m"],
                key="exposure_time",
            )

    with st.expander("Detector", expanded=True):
        st.selectbox(
            "Detector",
            options=[*list(DETECTORS), "Custom"],
            index=0,
            width=400,
            key="detector_preset",
        )

        if st.session_state.get("detector_preset", "Custom") != "Custom":
            try:
                detector: Detector = DETECTORS[st.session_state["detector_preset"]]
            except KeyError:
                st.error(
                    f"Detector {st.session_state['detector_preset']} not found in DETECTORS."
                )
                detector: Detector = DETECTORS["IMX174"]

            st.session_state["detector"] = detector

            subcols = cycle(st.columns(5))
            with next(subcols):
                st.metric("Pixel Size", value=f"{detector.pixel_size.to(u.um):.3g}")
            with next(subcols):
                st.metric("Gain", value=f"{detector.gain.value:.2g}")
            with next(subcols):
                st.metric(
                    "Dark Noise",
                    value=f"{detector.dark_noise.to(u.electron / u.s).value:.3g}e$^-$/s",
                )
            with next(subcols):
                st.metric(
                    "Read Noise",
                    value=f"{detector.read_noise.to(u.electron).value:.3g}e$^-$",
                )
            with next(subcols):
                if isinstance(detector.QE, (str, Path)):
                    st.metric("Quantum Efficiency", "From file")

                    with st.popover("Show QE Curve"):
                        # Display a small plot of the QE curve

                        wavelengths: ArrayQuantity = np.linspace(380, 950, 30) * u.nm
                        QE: NDArray[np.floating] = detector_QE(
                            detector.QE, wavelengths=wavelengths
                        )

                        fig: go.Figure = go.Figure()
                        fig.add_trace(
                            trace=go.Scatter(
                                x=wavelengths.to(u.nm).value,
                                y=QE,
                                mode="lines",
                            )
                        )

                        fig.update_layout(
                            title=f"{detector.name}",
                            xaxis_title="Wavelength (nm)",
                            yaxis_title="Quantum Efficiency",
                        )

                        st.plotly_chart(fig)

                else:
                    st.metric("Quantum Efficiency", value=f"{detector.QE}")

        else:
            if "detector" not in st.session_state:
                st.session_state["detector"] = Detector(
                    name="Custom",
                    pixel_size=3.45 * u.micron,
                    gain=1.0 / u.photon,
                    dark_noise=3.2 * u.electron / u.second,
                    read_noise=2.2 * u.electron,
                    QE=0.8,
                )

            cols = cycle(st.columns(3))

            with next(cols):
                custom_detector_name: str = st.text_input(
                    "Detector name",
                    value="Custom",
                    width=200,
                )

            with next(cols):
                custom_pixel_size: float = st.number_input(
                    "Pixel size (µm)",
                    min_value=0.0,
                    value=st.session_state.detector.pixel_size.to(u.micron).value,
                    step=0.01,
                    width=200,
                )

            with next(cols):
                custom_gain: float = st.number_input(
                    "Gain (e¯/photon)",
                    min_value=0.0,
                    value=st.session_state.detector.gain.to(1 / u.photon).value,
                    step=0.1,
                    width=200,
                )

            with next(cols):
                custom_read_noise: float = st.number_input(
                    "Read noise (e¯)",
                    min_value=0.0,
                    value=st.session_state.detector.read_noise.to(u.electron).value,
                    step=0.1,
                    width=200,
                )

            with next(cols):
                custom_dark_noise: float = st.number_input(
                    "Dark noise (e¯/s)",
                    min_value=0.0,
                    value=st.session_state.detector.dark_noise.to(
                        u.electron / u.second
                    ).value,
                    step=0.1,
                    width=200,
                )

            cols = cycle(st.columns(3))

            with next(cols):
                custom_qe = st.number_input(
                    "Quantum efficiency (flat)",
                    min_value=0.0,
                    value=st.session_state.detector.QE
                    if isinstance(st.session_state.detector.QE, (int, float))
                    else 0.8,
                    step=0.01,
                    max_value=1.0,
                    width=200,
                    disabled=st.session_state.get("custom_qe", False),
                )

            with next(cols):
                st.container(height=12, border=False)
                st.toggle("Upload QE curve?", key="custom_qe")

                if st.session_state.get("custom_qe", False):
                    st.warning("Not implemented yet! Using flat QE of 0.8 for now.")
                    custom_qe = 0.8

            detector: Detector = Detector(
                name=custom_detector_name,
                pixel_size=custom_pixel_size * u.micron,
                gain=custom_gain * (1 / u.photon),
                dark_noise=custom_dark_noise * (u.electron / u.second),
                read_noise=custom_read_noise * u.electron,
                QE=st.session_state.get("detector_qe", None),
            )

    with st.expander("Image settings", expanded=True):
        cols = cycle(st.columns(4))

        with next(cols):
            st.number_input(
                "Image size (pixels)",
                min_value=1,
                value=50,
                step=1,
                width=200,
                key="image_size",
            )

        with next(cols):
            st.number_input(
                "Centroid ROI size (pixels)",
                min_value=1,
                value=30,
                step=1,
                width=200,
                key="centroid_ROI",
            )

        cols = st.columns([1, 3])

        with cols[0]:
            st.container(height=12, border=False)
            st.toggle("Show fibre?", key="show_fibre_overlay")

        with cols[1]:
            st_quantity_input(
                "Fibre size",
                min_value=0.0,
                value=40.0,
                step=0.1,
                default_unit="µm",
                possible_units=["µm", "mm"],
                disabled=not st.session_state.get("show_fibre_overlay", False),
                key="fibre_size",
            )

        if not st.session_state.get("show_fibre_overlay", False):
            st.session_state["fibre_size"] = None


fig: plt.Figure = simulate_image(
    magnitude=st.session_state["stellar_magnitude"],
    blackbody_temperature=st.session_state["stellar_temperature"],
    seeing_fwhm=st.session_state["seeing_fwhm"],
    psf_fwhm=st.session_state["psf_fwhm"],
    exposure_time=st.session_state["exposure_time"],
    sky_background=22 * u.mag / u.arcsec**2,
    telescope_diameter=st.session_state["telescope_diameter"],
    secondary_diameter=st.session_state["telescope_secondary_diameter"],
    telescope_num_mirrors=st.session_state["telescope_num_mirrors"],
    guide_focal_ratio=st.session_state["guide_focal_ratio"],
    guiding_split=st.session_state["guiding_split"],
    detector=detector,
    image_size=st.session_state["image_size"],
    centroid_ROI=st.session_state["centroid_ROI"],
    fibre_size=st.session_state.get("fibre_size", None),
)


with output_column:
    st.pyplot(fig)
