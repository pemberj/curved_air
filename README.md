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
```

### Output:
```
Collimated beam diameter:          100 mm
Slit magnification factor:         1x
Slit length at detector:           22.2 pixels
Bluest orders separation:          23.2 pixels
Reddest orders separation:         92 pixels
Slit width (object plane):         50 um
Slit length (object plane):        200 um
Slit width (image plane):          50 um
Sampling of slit image:            5.56 pixels
Spectral resolution @ 600 nm:      64,273
```

![](./example_scripts/spectrographs/simple_echelle.svg)
