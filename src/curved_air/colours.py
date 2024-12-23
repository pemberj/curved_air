import numpy as np
from astropy import units as u
from astropy.units import Quantity


def wavelength_to_rgb(
    wls: Quantity = 0.55 * u.um,
    gamma: float = 0.8,
) -> tuple[float, float, float] | list[tuple[float, float, float]]:
    """This converts a given wavelength of light to an approximate RGB color value. The
    wavelength must be given in nanometers in the range from 380 nm through 750 nm
    (789 THz through 400 THz).

    Based on code by Dan Bruton (http://www.physics.sfasu.edu/astro/color/spectra.html)
    """

    if not isinstance(wls, list | np.ndarray):
        wls = np.array([wls])

    colors: list = []

    try:
        wls = wls.to("nm")
    except AttributeError:
        assert 5 < wls < 5000  # Make sure it is given in nanometers if no units

    for wl in wls:
        if wl >= 380 and wl <= 440:  # noqa: PLR2004
            attenuation = 0.3 + 0.7 * (wl - 380) / (440 - 380)
            R = ((-(wl - 440) / (440 - 380)) * attenuation) ** gamma
            G = 0.0
            B = (1.0 * attenuation) ** gamma
        elif wl >= 440 and wl <= 490:  # noqa: PLR2004
            R = 0.0
            G = ((wl - 440) / (490 - 440)) ** gamma
            B = 1.0
        elif wl >= 490 and wl <= 510:  # noqa: PLR2004
            R = 0.0
            G = 1.0
            B = (-(wl - 510) / (510 - 490)) ** gamma
        elif wl >= 510 and wl <= 580:  # noqa: PLR2004
            R = ((wl - 510) / (580 - 510)) ** gamma
            G = 1.0
            B = 0.0
        elif wl >= 580 and wl <= 645:  # noqa: PLR2004
            R = 1.0
            G = (-(wl - 645) / (645 - 580)) ** gamma
            B = 0.0
        elif wl >= 645 and wl <= 750:  # noqa: PLR2004
            attenuation = 0.3 + 0.7 * (750 - wl) / (750 - 645)
            R = (1.0 * attenuation) ** gamma
            G = 0.0
            B = 0.0
        else:
            attenuation = 0.3
            R = 1.0 * attenuation
            G = 1.0 * attenuation
            B = 1.0 * attenuation

        colors.append((R, G, B))
    return colors
