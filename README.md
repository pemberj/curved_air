# Curved Air

A repository of code developed during my PhD thesis, applicable to several areas of optical design tasks.

Contains code for optimisation of compound prisms (in particular for development of atmospheric dispersion correctors [ADCs]); narrowing down the options for the main parameters of an echelle spectrograph; and for numerical simulation of Fabry-Pérot etalons.


## Installation

I recommend using [`uv`](https://astral.sh/uv) to manage Python packages, dependencies
and virtual environments. To run code in this repository with `uv`:

```bash
git clone https://github.com/pemberj/curved_air
```

And simply:

```bash
uv run ./example_scripts/spectrographs/simple_echelle.py
```

`uv` will pull and install all the relevant dependencies into a virtual environment and
run the code.


## Using `curved_air` objects to model optical systems

Throughout this repository I have used [AstroPy](https://www.astropy.org/) `units` to manage physical quantities. Most functions should also accept regular floats or integers but will assume a default unit (and inform the user about this assumption).

```python
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
```
