from typing import TYPE_CHECKING

from astropy import units as u
from matplotlib import pyplot as plt

from curved_air.detectors import Detector
from curved_air.glass import GlassCatalogue
from curved_air.instruments.LRS import LowResolutionSpectrograph
from curved_air.optics.dispersers import Prism
from curved_air.plotting import plot_style

if TYPE_CHECKING:
    from curved_air.type_hinting import ScalarQuantity

plt.style.use(plot_style)


def main() -> None:
    MIN_WL: ScalarQuantity = 390 * u.nm
    MAX_WL: ScalarQuantity = 950 * u.nm

    lrs = LowResolutionSpectrograph(
        name="Test LRS",
        disperser=Prism(
            glass=GlassCatalogue("Schott")["F2"],
            apex_angle=30 * u.deg,
            min_wl=MIN_WL,
            max_wl=MAX_WL,
        ),
        # disperser=SimpleGrating(groove_frequency=300 / u.mm),
        detector=Detector(pixel_size=15 * u.um, nx=2048, ny=2048),
        min_wl=MIN_WL,
        max_wl=MAX_WL,
        beam_diameter=100 * u.mm,
        collimator_focal_length=800 * u.mm,
        slit_width=50 * u.um,
        slit_length=5 * u.mm,
        slit_native_focal_ratio=4,
        camera_focal_length=600 * u.mm,
    )

    lrs.print_parameters()

    fig: plt.Figure = plt.figure()
    ax: plt.Axes = fig.gca()

    lrs.plot_detector_edge(ax=ax)
    lrs.plot_spectrum(ax=ax, num_wavelengths=300)

    plt.show()


if __name__ == "__main__":
    main()
