import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt

from curved_air.detectors import Detector
from curved_air.instruments.echelle import EchelleSpectrograph
from curved_air.optics.dispersers import EchelleGrating, SimpleGrating
from curved_air.plotting import plot_style, wavelength_to_rgb

plt.style.use(plot_style)


def main() -> None:

    simple_echelle = EchelleSpectrograph(
        name="Simple echelle spectrograph",
        grating=EchelleGrating(
            groove_frequency=31.6 / u.mm,
            blaze_angle=76 * u.deg,
            gamma_angle=0.5 * u.deg,
        ),
        cross_disperser=SimpleGrating(
            groove_frequency=200 / u.mm,
        ),
        detector=Detector(
            name="4k Detector",
            pixel_size=9 * u.um,
            nx=4096,
            ny=4096,
        ),
        min_wl=400 * u.nm,
        max_wl=800 * u.nm,
        beam_diameter=100 * u.mm,
        camera_focal_length=400 * u.mm,
        slit_width=50 * u.um,
        slit_length=200 * u.um,
        slit_native_focal_ratio=4,
    )

    simple_echelle.print_parameters()

    fig: plt.Figure = plt.figure()
    ax: plt.Axes = fig.gca()

    simple_echelle.plot_detector_edge(ax=ax)

    for m in simple_echelle.spectral_orders:
        wls = simple_echelle.order_wavelengths(m=m, num=20)
        xs, ys = np.transpose(simple_echelle.image_plane_position(m=m, wavelength=wls))
        ax.plot(xs, ys, color=wavelength_to_rgb(wls[0]))

    ax.set_aspect("equal")
    ax.axis("off")

    plt.show()


if __name__ == "__main__":
    main()
