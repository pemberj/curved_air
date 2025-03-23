# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "curved-air>=0.1.0",
#     "rich>=10.0.0",
# ]
# [tool.uv.sources]
# curved_air = { path = "..", editable = true }
# ///


from matplotlib import pyplot as plt
from astropy import units as u
from astropy.units import Quantity  # noqa: F401

from curved_air.optics import Grating


def plot_FSR(grating: Grating) -> None:
    """
    Plot the free spectral range of the grating as a function of wavelength.
    """

    fig = plt.figure("Free spectral range")
    ax = fig.gca()

    ms = grating.diffraction_orders

    blaze_wavelengths = grating.blaze_wavelength(ms)
    blaze_wavelengths = [b.to(u.um).value for b in blaze_wavelengths]

    FSRs = grating.order_FSR(ms)
    FSRs = [f.to(u.nm).value for f in FSRs]

    ax.plot(
        blaze_wavelengths,
        FSRs,
        marker=".",
        markersize=10,
        markeredgecolor="k",
        markeredgewidth=0.5,
        alpha=0.5,
        # label=f"R={g.R}, {g.gpmm.value} g/mm: {len(ms)} orders",
    )

    ax.set_xlabel("Blaze wavelength $\\lambda$ [microns]")
    ax.set_ylabel("Free spectral range $\\Delta\\lambda_{FSR}$ [nm]")

    ax.legend()

    plt.show()


def main() -> None:
    MARVEL_grating: Grating = Grating(
        catalogue_number="53-*-417E",
        groove_frequency=52.67 / u.mm,
        blaze_angle=69.7 * u.deg,
        gamma_angle=0 * u.deg,
        min_wl=390 * u.nm,
        max_wl=950 * u.nm,
    )

    plot_FSR(MARVEL_grating)


if __name__ == "__main__":
    main()
