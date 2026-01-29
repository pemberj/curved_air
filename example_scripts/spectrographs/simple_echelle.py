import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt

from curved_air.detectors import Detector
from curved_air.instruments.echelle import EchelleSpectrograph
from curved_air.optics.dispersers import EchelleGrating, SimpleGrating
from curved_air.plotting import plot_style, wavelength_to_rgb

plt.style.use(plot_style)

MIN_WL = 400 * u.nm
MAX_WL = 800 * u.nm

simple_echelle = EchelleSpectrograph(
    name="Simple echelle",
    grating=EchelleGrating(
        name="Echelle",
        groove_frequency=31.6 / u.mm,
        blaze_angle=76 * u.deg,
        gamma_angle=0.5 * u.deg,
        min_wl=MIN_WL,
        max_wl=MAX_WL,
    ),
    cross_disperser=SimpleGrating(
        name="Cross-disperser",
        groove_frequency=200 / u.mm,
        min_wl=MIN_WL,
        max_wl=MAX_WL,
    ),
    detector=Detector(
        name="4k Detector",
        pixel_size=9 * u.um,
        nx=4096,
        ny=4096,
    ),
    min_wl=MIN_WL,
    max_wl=MAX_WL,
    collimator_focal_length=800 * u.mm,
    camera_focal_length=400 * u.mm,
    slit_width=50 * u.um,
    slit_length=200 * u.um,
    slit_native_focal_ratio=4,
)
simple_echelle.print_parameters()

fig = plt.figure()
ax = fig.gca()

simple_echelle.plot_detector_edge(ax=ax)

for m in simple_echelle.spectral_orders:
    wls = simple_echelle.order_wavelengths(m=m)
    xs, ys = np.transpose(simple_echelle.image_plane_position(m=m, wavelength=wls))
    ax.plot(xs, ys, color=wavelength_to_rgb(wls[0]))


ax.set_xlim(
    -simple_echelle.detector.size[0].to(u.mm).value / 2 - 5,
    simple_echelle.detector.size[0].to(u.mm).value / 2 + 5,
)
ax.set_ylim(
    -simple_echelle.detector.size[1].to(u.mm).value / 2 - 5,
    simple_echelle.detector.size[1].to(u.mm).value / 2 + 5,
)
ax.set_aspect("equal")
ax.axis("off")

plt.show()
