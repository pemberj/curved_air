#! uv run streamlit run
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.modeling.models import BlackBody
from astropy.units import Quantity
from numpy.typing import NDArray

from curved_air.plotting import plot_style
from curved_air.type_hinting import ArrayQuantity, ScalarQuantity

plt.style.use(plot_style)

SCRIPT_PATH = Path(__file__).parent


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


def gaussian_seeing(
    fwhm: ScalarQuantity = 1.0 * u.arcsec,
    center: tuple[float, float] = (0.0, 0.0),
    plate_scale: ScalarQuantity = 0.08594 * u.arcsec / u.micron,
    pixel_size: ScalarQuantity = 3.45 * u.micron,
    num_pixels: int = 50,
) -> NDArray:
    """
    From the input parameters, creates a 2D array (image) of the expected size
    of the seeing disk at a detector.
    """

    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    sigma_pixels = sigma / plate_scale / pixel_size

    x = np.linspace(-num_pixels / 2, num_pixels / 2 - 1, num_pixels)
    # x = np.arange(start=-num_pixels / 2, stop=num_pixels / 2, step=1)
    y = x[:, np.newaxis]

    gaussian = np.exp(
        -((x - center[0]) ** 2 + (y - center[1]) ** 2) / (2 * sigma_pixels**2)
    )
    return gaussian / np.sum(gaussian)


def PSF(
    fwhm: ScalarQuantity = 3.0 * u.micron,
    pixel_size: ScalarQuantity = 3.45 * u.micron,
    num_pixels: int = 20,
) -> NDArray:
    """
    Creates a 2D array (image) of a point spread function.

    Assumes a Gaussian PSF.
    """

    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    sigma_pixels = sigma / pixel_size

    x = np.linspace(-num_pixels / 2, num_pixels / 2 - 1, num_pixels)
    # x = np.arange(start=-num_pixels / 2, stop=num_pixels / 2, step=1)
    y = x[:, np.newaxis]

    gaussian = np.exp(-(x**2 + y**2) / (2 * sigma_pixels**2))
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

    assert isinstance(wavelengths, ArrayQuantity)

    dlambda = (wavelengths[1].value - wavelengths[0].value) * u.nm

    # Calculating the blackbody radiation of the observed star over the full wavelenghth
    # range and converting to photon/s/m²/sr
    bb = BlackBody(temperature)
    bbJy = bb(wavelengths).to(u.Jy / u.sr)
    bbPh = (
        bbJy.to(
            u.photon / u.s / u.m**2 / u.nm / u.sr,
            equivalencies=u.spectral_density(wavelengths),
        )
        * dlambda
    )

    # The same calculation but only for the V band, this will be used to scale the flux
    # V band wavelength range and spacing between wavelengths
    wave_V = np.arange(492, 580, dlambda.to(u.nm).value) * u.nm
    bbJy_V = bb(wave_V).to(u.Jy / u.sr)
    bbPh_V = np.sum(
        bbJy_V.to(
            u.photon / u.s / u.m**2 / u.nm / u.sr,
            equivalencies=u.spectral_density(wave_V),
        )
        * dlambda
    )

    # Reference flux for a magnitude 0 star in the V band
    ref_V = band_properties["V"]["standard_flux"]

    # Scaling the flux from the blackbody function -> units: photon/s/m²
    ref_V_ph = np.sum(
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
    noise = np.random.poisson(
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
    dark_current: ScalarQuantity = 3.2 * u.electron / u.second,
    plot_hist: bool = False,
) -> NDArray:
    """
    Computes and returns a 2D array (image) containing the dark-current induced
    noise of a CMOS detector.
    """

    if not image:
        image: NDArray = np.zeros((num_pixels, num_pixels))

    DC: NDArray = np.random.poisson(
        lam=(dark_current * exposure_time).to(u.electron).value,
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


def telescope_efficiency(
    csv: Path | str,
    num_mirrors: int = 3,
    wavelengths: NDArray | None = None,
) -> NDArray:
    if wavelengths is None:
        wavelengths: ArrayQuantity = np.linspace(380, 950, 100) * u.nm

    data: pd.DataFrame = pd.read_csv(csv, names=["wavelength", "reflectivity"])
    wave_data = data["wavelength"]
    efficiency = data["reflectivity"]

    refl_full = np.interp(wavelengths.value, wave_data, efficiency, left=efficiency[0])

    return refl_full**num_mirrors


def detector_QE(
    csv: Path | str,
    wavelengths: NDArray | None = None,
) -> NDArray:
    if wavelengths is None:
        wavelengths = np.linspace(380, 950, 100) * u.nm

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
    csv: Path | str,
    wavelengths: NDArray | None = None,
) -> NDArray:
    if wavelengths is None:
        wavelengths = np.linspace(380, 950, 100) * u.nm

    data: pd.DataFrame = pd.read_csv(csv, names=["wavelength", "efficiency"])
    wave_data = data["wavelength"]
    efficiency = data["efficiency"]

    return np.interp(wavelengths.value, wave_data, efficiency, left=efficiency[0])


def simulate_image(
    magnitude: float | Quantity = 11 * u.mag,
    blackbody_temperature: float | Quantity = 3200 * u.K,
    seeing_fwhm: float | Quantity = 2 * u.arcsec,
    exposure_time: float | Quantity = 0.1 * u.second,
    sky_background: float | Quantity = 22 * u.mag / u.arcsec**2,
) -> None:
    """
    The main script.

    Simulates a star image on a detector, with all of the relevant parameters
    describing the brightness of the star, the telescope, the detector, etc.,
    and then computes the centroid of the resulting image.
    """

    if isinstance(magnitude, (int, float)):
        magnitude *= u.mag

    if isinstance(blackbody_temperature, (int, float)):
        blackbody_temperature *= u.K

    if isinstance(seeing_fwhm, (int, float)):
        seeing_fwhm *= u.arcsec

    if isinstance(exposure_time, (int, float)):
        exposure_time *= u.second

    if isinstance(sky_background, (int, float)):
        sky_background *= u.mag / u.arcsec**2

    mirror_R = SCRIPT_PATH / "data" / "ASA_Protected_Enhanced_Aluminium.csv"
    guiding_split = SCRIPT_PATH / "data" / "Tafelmeier_6_final.csv"
    QE = SCRIPT_PATH / "data" / "IMX265_QE.csv"

    image_size = 44
    centroid_ROI = 30

    wavelengths: ArrayQuantity = np.linspace(380, 950, 100) * u.nm

    # Optical performance
    psf_fwhm: ScalarQuantity = 2.5 * u.micron

    # Telescope properties
    telescope_diameter: ScalarQuantity = 0.8 * u.meter
    secondary_diameter: ScalarQuantity = 0.2 * u.meter
    mirrors = 3  # number of mirrors throughout telescope

    guide_focal_ratio = 3.5
    plate_scale: ScalarQuantity = (
        1 * u.radian / (telescope_diameter * guide_focal_ratio)
    )
    plate_scale: ScalarQuantity = plate_scale.to(u.arcsec / u.um)

    # Detector properties
    pixel_size: ScalarQuantity = 3.45 * u.um
    gain: ScalarQuantity = 1 / u.photon  # ?
    dark_current: ScalarQuantity = 3.2 * u.electron / u.second
    read_noise_level: ScalarQuantity = 2.2 * u.electron

    exposure: NDArray = gaussian_seeing(
        fwhm=seeing_fwhm,
        plate_scale=plate_scale,
        pixel_size=pixel_size,
        num_pixels=image_size,
    )

    psf: NDArray = PSF(fwhm=psf_fwhm)

    exposure: NDArray = cv2.filter2D(exposure, -1, psf)

    _photon_flux = photon_flux(
        magnitude=magnitude, wavelengths=wavelengths, temperature=blackbody_temperature
    )

    _telescope_efficiency = telescope_efficiency(
        wavelengths=wavelengths, num_mirrors=mirrors, csv=mirror_R
    )

    _guiding_efficiency = guiding_efficiency(csv=guiding_split, wavelengths=wavelengths)

    _detector_QE = detector_QE(wavelengths=wavelengths, csv=QE)

    flux = (
        _photon_flux
        * telescope_area(
            telescope_diameter=telescope_diameter, secondary_diameter=secondary_diameter
        )
        * _telescope_efficiency
        * _guiding_efficiency
        * exposure_time
        * gain
        * _detector_QE
    )
    exposure *= np.sum(flux)

    dark = dark_noise(
        num_pixels=image_size, exposure_time=exposure_time, dark_current=dark_current
    )

    read = read_noise(num_pixels=image_size, noise_level=read_noise_level)

    exposure += dark + read

    exposure -= np.median(dark) + np.median(read)

    _fig, axs = plt.subplots(
        nrows=2, ncols=1, figsize=(10, 10), gridspec_kw={"height_ratios": [3, 1]}
    )

    ax: plt.Axes = axs[0]

    ax.imshow(exposure)

    fibre_radius = 20 * u.um

    fibre = plt.Circle(
        (image_size // 2, image_size // 2),
        radius=(fibre_radius / pixel_size).si.value,
        color="red",
        fill=False,
        label="Science fibre",
    )
    ax.add_artist(fibre)

    cenx, ceny = compute_centroid(image=exposure, ROI_size=centroid_ROI)

    ax.plot(
        cenx,
        ceny,
        lw=0,
        marker=".",
        markersize=15,
        color="red",
        label=f"Computed centroid: ({cenx:.2f}, {ceny:.2f})",
    )

    xs, ys = square_coordinates(w=centroid_ROI)
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

    ax.legend()
    ax.set_title(
        "Simulated guiding image\n"
        f"Seeing FWHM = {seeing_fwhm:.2g}, "
        f"V={magnitude.value:.3g} Blackbody of T={blackbody_temperature:.4g}\n"
        f"{exposure_time:.3g} exposure on Sony IMX265 CMOS detector",
        size=17,
    )
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[1]
    ax.set_ylabel("Signal", size=15)
    ax.set_xlabel("Pixel", size=15)
    for i in exposure:
        ax.plot(i, alpha=0.25)

    for ext in ["png", "pdf"]:
        plt.savefig(SCRIPT_PATH / f"image.{ext}")
    plt.close()


def plot_curves(
    magnitude: float | Quantity = 11 * u.mag,
    blackbody_temperature: float | Quantity = 3200 * u.K,
    seeing_fwhm: float | Quantity = 2 * u.arcsec,
    exposure_time: float | Quantity = 0.1 * u.second,
) -> None:
    """ """

    if isinstance(magnitude, (int, float)):
        magnitude *= u.mag

    if isinstance(blackbody_temperature, (int, float)):
        blackbody_temperature *= u.K

    if isinstance(seeing_fwhm, (int, float)):
        seeing_fwhm *= u.arcsec

    if isinstance(exposure_time, (int, float)):
        exposure_time *= u.second

    mirror_R = SCRIPT_PATH / "data" / "ASA_Protected_Enhanced_Aluminium.csv"
    guiding_split = SCRIPT_PATH / "data" / "Tafelmeier_6_final.csv"
    QE = SCRIPT_PATH / "data" / "IMX265_QE.csv"

    image_size = 70

    wavelengths: ArrayQuantity = np.linspace(380, 950, 1_000) * u.nm

    # Optical performance
    psf_fwhm: ScalarQuantity = 2.5 * u.micron

    # Telescope properties
    telescope_diameter: ScalarQuantity = 0.8 * u.meter
    secondary_diameter: ScalarQuantity = 0.2 * u.meter
    mirrors = 3  # number of mirrors throughout telescope

    guide_focal_ratio = 3.5
    plate_scale: ScalarQuantity = (
        1 * u.radian / (telescope_diameter * guide_focal_ratio)
    )
    plate_scale: ScalarQuantity = plate_scale.to(u.arcsec / u.um)

    # Detector properties
    pixel_size: ScalarQuantity = 3.45 * u.um
    gain: ScalarQuantity = 1 / u.photon  # ?
    dark_current: ScalarQuantity = 3.2 * u.electron / u.second
    read_noise_level: ScalarQuantity = 2.2 * u.electron

    exposure: NDArray = gaussian_seeing(
        fwhm=seeing_fwhm,
        plate_scale=plate_scale,
        pixel_size=pixel_size,
        num_pixels=image_size,
    )

    psf: NDArray = PSF(fwhm=psf_fwhm)
    exposure: NDArray = cv2.filter2D(exposure, -1, psf)

    _photon_flux = photon_flux(
        magnitude=magnitude, wavelengths=wavelengths, temperature=blackbody_temperature
    )

    _telescope_efficiency = telescope_efficiency(
        wavelengths=wavelengths, num_mirrors=mirrors, csv=mirror_R
    )

    _guiding_efficiency = guiding_efficiency(csv=guiding_split, wavelengths=wavelengths)
    _detector_QE = detector_QE(wavelengths=wavelengths, csv=QE)

    flux = (
        _photon_flux
        * telescope_area(
            telescope_diameter=telescope_diameter,
            secondary_diameter=secondary_diameter,
        )
        * _telescope_efficiency
        * _guiding_efficiency
        * exposure_time
        * gain
        * _detector_QE
    )

    exposure *= np.sum(flux)

    dark: NDArray = dark_noise(
        num_pixels=image_size, exposure_time=exposure_time, dark_current=dark_current
    )

    read: NDArray = read_noise(num_pixels=image_size, noise_level=read_noise_level)

    exposure += dark + read
    exposure -= np.median(dark) + np.median(read)

    fig: plt.Figure = plt.figure(figsize=(8, 8))
    ax: plt.Axes = fig.gca()

    _normalised_photon_flux = _photon_flux / np.max(_photon_flux)

    ax.plot(
        wavelengths,
        _normalised_photon_flux,
        label=f"Normalised input flux (Blackbody of T={blackbody_temperature:.4g})",
    )
    ax.plot(wavelengths, _telescope_efficiency, label="Telescope efficiency")
    ax.plot(wavelengths, _guiding_efficiency, label="Guiding split (coating)")
    ax.plot(wavelengths, _detector_QE, label="Detector QE")

    _total_efficiency = _telescope_efficiency * _guiding_efficiency * _detector_QE
    ax.plot(wavelengths, _total_efficiency, label="Total efficiency")

    _total_flux = (
        _photon_flux * _telescope_efficiency * _guiding_efficiency * _detector_QE
    )
    _normalised_total_flux = _total_flux / np.max(_total_flux)

    ax.plot(wavelengths, _normalised_total_flux, label="Normalised guiding flux")

    ax.set_ylim(-0.05, 1.3)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Efficiency / Normalised Flux")
    ax.legend()

    for ext in ["png", "pdf"]:
        plt.savefig(SCRIPT_PATH / f"curves.{ext}")
    plt.close()


def plot_QE_and_coating() -> None:
    """ """

    guiding_split: Path = SCRIPT_PATH / "data" / "Tafelmeier_6_final.csv"

    measured_guiding_split: Path = (
        SCRIPT_PATH / "data" / "MARVEL_beamsplitter_split.csv"
    )

    QE: Path = SCRIPT_PATH / "data" / "IMX265_QE.csv"

    wavelengths: ArrayQuantity = np.linspace(380, 950, 1_000) * u.nm

    _guiding_efficiency = guiding_efficiency(
        csv=guiding_split,
        wavelengths=wavelengths,
    )
    _measured_guiding_efficiency = guiding_efficiency(
        csv=measured_guiding_split,
        wavelengths=wavelengths,
    )
    _detector_QE = detector_QE(
        csv=QE,
        wavelengths=wavelengths,
    )

    fig: plt.Figure = plt.figure(figsize=(8, 4))
    ax1: plt.Axes = fig.gca()
    ax2: plt.Axes = ax1.twinx()

    QEline = ax1.plot(
        wavelengths,
        _detector_QE,
        color="tab:blue",
        label="Detector QE",
        alpha=0.8,
    )
    BSline = ax2.plot(
        wavelengths,
        _guiding_efficiency,
        color="tab:orange",
        ls="--",
        label="Guiding split\n(design)",
        alpha=0.8,
    )
    BSline_measured = ax2.plot(
        wavelengths,
        _measured_guiding_efficiency,
        color="tab:orange",
        ls="-",
        label="Guiding split\n(measured)",
        alpha=0.8,
    )

    ax1.tick_params(axis="y", labelcolor=QEline[0].get_color())
    ax2.tick_params(axis="y", labelcolor=BSline_measured[0].get_color())

    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel("Detector quantum efficiency", color=QEline[0].get_color())
    ax2.set_ylabel("Guiding split efficiency", color=BSline[0].get_color())

    lines = QEline + BSline + BSline_measured
    labels = [str(line.get_label()) for line in lines]
    ax1.legend(lines, labels)

    for ext in ["png", "pdf"]:
        plt.savefig(SCRIPT_PATH / f"QE_coating_curves.{ext}")
    plt.close()


def main() -> None:
    simulate_image(
        magnitude=11 * u.mag,
        blackbody_temperature=3200 * u.K,
        seeing_fwhm=1.5 * u.arcsec,
        exposure_time=0.1 * u.second,
    )

    plot_curves()
    plot_QE_and_coating()


if __name__ == "__main__":
    main()
