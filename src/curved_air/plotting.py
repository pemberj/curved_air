"""
A place for functions to plot image arrays, orders, matching wavelength lines,
wavelength solution functions,
"""

from typing import overload
import colorsys

import numpy as np
from astropy import units as u
from astropy.units import Quantity
import matplotlib.font_manager as fm
from pyfonts import load_font

# load Quicksand font
url = (
    "https://github.com/andrew-paglinawan/QuicksandFamily/blob/master/"
    + "fonts/statics/Quicksand-Regular.ttf"
)
quicksand = load_font(font_url=f"{url}?raw=true")
fm.FontManager.addfont(fm.fontManager, path=quicksand._file)

lw = 1.3

plot_style = {
    # Set font to the beautiful Quicksand:
    # https://github.com/andrew-paglinawan/QuicksandFamily
    "font.family": "Quicksand",
    # Also try to use this font for mathtext characters
    "mathtext.default": "regular",
    # Set axis text label sizes
    "axes.labelsize": 12,
    "axes.titlesize": 16,
    # Axis spine line widths
    "axes.linewidth": lw,
    # Set the default axis limits to be round numbers
    # Use plt.rcParams["axes.autolimit_mode"] = "data" to disable
    "axes.autolimit_mode": "round_numbers",
    "axes.unicode_minus": True,
    # x tick properties
    "xtick.top": True,
    "xtick.bottom": True,
    "xtick.major.size": 8,
    "xtick.major.width": lw,
    "xtick.minor.visible": True,
    "xtick.minor.size": 4,
    "xtick.minor.width": lw,
    "xtick.direction": "in",
    "xtick.major.pad": 10,
    # y tick properties
    "ytick.left": True,
    "ytick.right": True,
    "ytick.major.size": 8,
    "ytick.major.width": lw,
    "ytick.minor.visible": True,
    "ytick.minor.size": 4,
    "ytick.minor.width": lw,
    "ytick.direction": "in",
    "ytick.major.pad": 10,
    # Tick text label sizes
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    # Legend properties
    "legend.frameon": False,
    "legend.fontsize": 14,
    "legend.labelspacing": 0.25,
    "legend.handletextpad": 0.25,
    "figure.titlesize": 18,
    "figure.figsize": (8, 6),
    "figure.dpi": 96,
    "figure.constrained_layout.use": True,
    # Default properties for plotting lines, markers, scatterpoints, etc.
    "lines.linewidth": 2,
    "lines.markeredgecolor": "k",
    "lines.markersize": 16,
    "lines.solid_capstyle": "round",
    "lines.dash_capstyle": "round",
    "scatter.marker": ".",
    "scatter.edgecolors": "k",
    "errorbar.capsize": 3,
    "hist.bins": 20,
}


@overload
def wavelength_to_rgb(
    wavelength: float, gamma: float, fade_factor: float
) -> tuple[float, float, float]: ...


@overload
def wavelength_to_rgb(
    wavelength: list, gamma: float, fade_factor: float
) -> list[tuple[float, float, float]]: ...


def wavelength_to_rgb(
    wavelength: Quantity = 0.55 * u.um,
    gamma: float = 0.8,
    fade_factor: float = 0.8,
) -> tuple[float, float, float] | list[tuple[float, float, float]]:
    """This converts a given wavelength of light to an approximate RGB color value. The
    wavelength must be given in nanometers in the range from 380 nm through 750 nm
    (789 THz through 400 THz).

    Based on code by Dan Bruton (http://www.physics.sfasu.edu/astro/color/spectra.html)
    """

    if isinstance(wavelength, list | np.ndarray):
        return [
            wavelength_to_rgb(wl, gamma=gamma, face_factor=fade_factor)
            for wl in wavelength
        ]

    try:
        wavelength = wavelength.to("nm")
    except AttributeError:
        # Make sure `wls` is given in nanometers if no units are passed
        assert 5 < wavelength < 5000  # noqa: PLR2004

    if wavelength >= 380 and wavelength <= 440:  # noqa: PLR2004
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        r = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        g = 0.0
        b = (1.0 * attenuation) ** gamma
    elif wavelength >= 440 and wavelength <= 490:  # noqa: PLR2004
        r = 0.0
        g = ((wavelength - 440) / (490 - 440)) ** gamma
        b = 1.0
    elif wavelength >= 490 and wavelength <= 510:  # noqa: PLR2004
        r = 0.0
        g = 1.0
        b = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif wavelength >= 510 and wavelength <= 580:  # noqa: PLR2004
        r = ((wavelength - 510) / (580 - 510)) ** gamma
        g = 1.0
        b = 0.0
    elif wavelength >= 580 and wavelength <= 645:  # noqa: PLR2004
        r = 1.0
        g = (-(wavelength - 645) / (645 - 580)) ** gamma
        b = 0.0
    elif wavelength >= 645 and wavelength <= 750:  # noqa: PLR2004
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        r = (1.0 * attenuation) ** gamma
        g = 0.0
        b = 0.0
    else:
        attenuation = 0.3
        r = 1.0 * attenuation
        g = 1.0 * attenuation
        b = 1.0 * attenuation

    return fade((r, g, b), fade_factor=fade_factor)


def fade(
    rgb: tuple[float, float, float],
    fade_factor: float = 0.8,
) -> tuple[float, float, float]:
    """
    Applies a saturation fade to a given RGB colour tuple.
    """

    h, s, v = colorsys.rgb_to_hsv(*rgb)

    return colorsys.hsv_to_rgb(h=h, s=fade_factor * s, v=v)
