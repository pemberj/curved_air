"""

"""

import os
import sys
script_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(script_path, "../.."))

from glob import glob
import re

import numpy as np
from numpy.typing import ArrayLike

from astropy import units as u
from astropy.units import Quantity

import matplotlib.pyplot as plt

from curved_air.glass import GlassCatalogue
from curved_air.optics import Prism

from curved_air.plot_style import plot_style
plt.style.use(plot_style)


def main():


    agf_files = sorted([
        f for f in glob("./data/AGFs/*.[Aa][Gg][Ff]")
            if re.search("schott", f)
            ])
    catalogue = GlassCatalogue().read_AGF(agf_files)


    XD = Prism(
        glass = catalogue["F2"],
        apex_angle = 60 * u.deg,
        min_wl = 0.39 * u.um,
        max_wl = 0.95 * u.um,
    )
    
    print(XD.incidence_angle)
    
    # wls = np.linspace(0.39, 0.95, 50) * u.um
    
    # fig = plt.figure()
    # ax = fig.gca()
    
    # i_angles = np.linspace(40, 70) * u.deg
    
    
    # for w in wls:
    #     e_angles = [((XD.apex_angle + d) / 2).to(u.deg).value for d in XD.deviation_angle(wl = w, incidence_angle = i_angles)]
    #     ax.plot(i_angles, e_angles, label=f"{w:.2f}")
    #     ax.axvline(x=XD.minimum_deviation_incidence_angle(wl = w).to(u.deg).value, color="k", ls="--")
    
    # ax.set_xlabel("Prism incidence angle (deg)")
    # ax.set_ylabel("Prism exitance angle (deg)")
    
    # ax.legend()
    
    # plt.show()

    
    
    
    
    ## A plot of the exitance angle as a function of wavelength
    
    # fig = plt.figure()
    # ax = fig.gca()
    
    # angles = [((XD.apex_angle + d) / 2).to(u.deg).value for d in XD.deviation_angle(wls)] * u.deg
    # ax.plot(wls, angles)
    # ax.axhline(y=XD.incidence_angle.to(u.deg).value, color="k", ls="--")
    # ax.text(x=wls[-1].value, y=XD.incidence_angle.to(u.deg).value,
    #         s=f"Incidence angle: {XD.incidence_angle:.2f}", ha="right", va="bottom", size=16)
    
    # ax.set_xlabel("Wavelength (um)")
    # ax.set_ylabel("Prism exitance angle (deg)")
    
    # plt.show()
    
    
    
    
    
    # Now what we really care about: a plot of detector XD position as a function of wavelength
    # To provide information on order spacing
    
    camera_focal_length = 730 * u.mm
    plate_scale = u.radian / camera_focal_length
    # camera_f_ratio = 4.76
    # injection_f_ratio = 3.85
    # slit_magnification = camera_f_ratio / injection_f_ratio
    
    slit_magnification = 1.13
    
    required_gap_pixels = 5
    pixel_size = 9 * u.um
    
    
    fibre_spacing = 130 * u.um
    fibre_width = 40 * u.um
    raw_slit_length = 4 * fibre_spacing + fibre_width
    print(f"{raw_slit_length = }")
    
    slit_length = 620 * u.um - required_gap_pixels * pixel_size/slit_magnification
    
    sigma = 1 / (52.67 / u.mm)
    gamma = 0.3 * u.deg
    theta = 69.7 * u.deg
    
    m_times_lambda = (sigma * 2 * np.sin(theta) * np.cos(gamma)).to(u.um)
    
    wls = np.linspace(0.38, 0.95, 50) * u.um
    m_min = int(m_times_lambda / np.max(wls) - 1)
    m_max = int(m_times_lambda / np.min(wls) + 1)
    
    ms = np.arange(m_min, m_max, 1)
    blaze_wls = [(m_times_lambda / m).to(u.um).value for m in ms] * u.um
    
    
    fig = plt.figure()
    ax = fig.gca()
    
    # angles2 = [((XD.apex_angle + d - XD.incidence_angle)).to(u.deg).value for d in XD.deviation_angle(blaze_wls)] * u.deg
    angles = [XD.trace(wl).value for wl in blaze_wls] * u.deg
    # detector_position2 = (angles2 / plate_scale).to(u.mm)
    detector_position = (angles / plate_scale).to(u.mm)
    # order_spacing2 = np.abs(np.diff(detector_position2))
    order_spacing = np.abs(np.diff(detector_position))
    
    
    ax.plot(blaze_wls[1:], order_spacing.to(u.um), marker="o", markersize=5)
    # ax.plot(blaze_wls[1:], order_spacing2.to(u.um), marker="o", markersize=5)
    ax.axhline(y=(slit_length * slit_magnification + required_gap_pixels * pixel_size).value, color="k", ls="--")
    ax.text(x=ax.get_xlim()[0], y=(slit_magnification * slit_length).value,
            s=f"  {slit_length:.0f} slit + {required_gap_pixels:.0f} pixel gap\n", ha="left", va="bottom", size=16)
    
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("Inter-order spacing (microns)")
    
    plt.show()
    


if __name__ == "__main__":

    main()