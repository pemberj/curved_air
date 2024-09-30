
from glob import glob
import numpy as np
import pandas as pd
from astropy import units as u
from matplotlib import pyplot as plt

try:
    from curved_air.optics import Grating
except ImportError:
    from optics import Grating


# Echelledesigner / read-csv.py
def load_csv(
    filename: str | None = None,
    plot: bool = False
    ) -> list[Grating]:
    
    if not filename:
        file = glob("./richardson-echelle-T3.csv")[0]
    df = pd.read_csv(file, delimiter=" ")

    selection = df.query("Ruled_width >= 400")

    gratings = [
        Grating(
            catalogue_number = c,
            groove_frequency = float(gpmm) / u.mm,
            blaze_angle = float(theta[:-1]) * u.deg,
            )\
                for c, gpmm, theta in zip(
                    selection["Catalog_number"],
                    selection["Grooves_per_mm"],
                    selection["Nominal_blaze_angle"]
                    )
        ]

    if plot:
        for g in gratings:
            print(g)
            print(g.aspect_ratio())

        min_wl = 0.385 * u.um
        max_wl = 0.950 * u.um

        fig = plt.figure()
        ax = fig.gca()

        for g in gratings:
            ms = g.diffraction_orders(min_wl = min_wl, max_wl = max_wl)
            blaze_wavelengths = g.blaze_wavelength(ms)
            blaze_wavelengths = [b.to(u.um).value for b in blaze_wavelengths]
            ax.plot(ms, blaze_wavelengths, label=f"R={g.R}, {g.gpmm.value} g/mm")

        ax.set_xlabel("Order number $m$")
        ax.set_ylabel("Blaze wavelength $\lambda$ ($\mu m$)")

        ax.legend()
        plt.show()


        fig = plt.figure()
        ax = fig.gca()

        for g in gratings:
            ms = g.diffraction_orders(min_wl = min_wl, max_wl = max_wl)
            angles = g.order_opening_angle(ms)
            angles = [a.to(u.deg).value for a in angles]

            ax.plot(ms, angles, label=f"R={g.R}, {g.gpmm.value} g/mm")

        ax.set_xlabel("Order number $m$")
        ax.set_ylabel("Order opening angle (FSR) (degrees)")

        ax.legend()
        plt.show()
        
    return gratings


# EchelleDesigner / order_gird.py

def order_grid(
    grating: Grating =\
        Grating(
            groove_frequency = 52.67/u.mm,
            blaze_angle = 69*u.deg,
            gamma_angle = 0.36*u.deg
            ),
    min_wl: u.Quantity = 380*u.nm,
    max_wl: u.Quantity = 870*u.nm,
    num_configs: int = 12,
    ):

    blaze = grating.blaze_angle
    groove_density = grating.groove_frequency
    gamma = grating.gamma

    m_times_lambda = np.cos(gamma) * (1 / groove_density) * 2 * np.sin(blaze)

    sign_of_blaze = blaze / abs(blaze)

    print(f"{sign_of_blaze = }")

    min_order = -1 * sign_of_blaze * int((m_times_lambda / min_wl) - 0.5)
    max_order = -1 * sign_of_blaze * int((m_times_lambda / max_wl) + 0.5)

    print(f"{min_order = }")
    print(f"{max_order = }")

    min_wl_blaze = (m_times_lambda / min_order).to(u.nm)
    max_wl_blaze = (m_times_lambda / max_order).to(u.nm)
    print(f"{min_wl_blaze = }")
    print(f"{max_wl_blaze = }")

    alt_increment = int(((max_order - 1) - (min_order + 1)) / (num_configs - 4))
    print(f"{alt_increment = }")

    alt_grid = []
    
    alt_grid.append(int(min_order))
    alt_grid.append(int(min_order + 1))

    for i in range(1, num_configs - 3):
        m = min_order + 1 + i * alt_increment
        alt_grid.append(int(m.value))

    alt_grid.append(int(max_order - 1))
    alt_grid.append(int(max_order))

    print(f"{alt_grid=}")
    print(f"{len(alt_grid)=}")
    return
    
    alt_orders = [min_order.value, min_order.value + 1]\
      + list(range(min_order.value + 1 + alt_increment, max_order.value - 1, num_configs - 4))\
      + [max_order.value - 1, max_order.value]
    print(alt_orders)
   
    
# EchelleDesigner / transfer_mirror_length.py
def calculate_transfer_mirror_height(
    grating: Grating,
    collimator_focal_length: u.Quantity,
    ) -> u.Quantity:
    
    image_height =\
        (collimator_focal_length / u.radian\
            * grating.order_opening_angle(m=grating.min_order)).si
    
    # print("Total required height of transfer mirror:"\
    #     +f"{image_height.to(u.mm) = :.3f}")
    
    return image_height.to(u.mm)


# EtalonSim / helper.py
def FWHM_old(y_values, x_values=None):
    """
    Calculate the FWHM of a series of numerical values on a constant x-grid
    """

    norm = y_values / np.nanmax(y_values)
    half = abs(norm - max(norm) / 2) # Create zero-crossings at half max value
    zero_crossings = []
    zero_crossings.append(np.argmin(half[:len(half)//2])) # First zero-crossing
    zero_crossings.append(np.argmin(half[len(half)//2:]) + len(half)//2) # Second zero-crossing

    fwhm_steps = np.diff(zero_crossings)[0]

    if x_values is not None:
        x_step = np.diff(x_values)[0]
        return fwhm_steps * x_step
    else:
        return fwhm_steps


def lin_interp(x, y, i, half):
    return x[i] + (x[i+1] - x[i]) * ((half - y[i]) / (y[i+1] - y[i]))


def FWHM(x_values, y_values):
    """
    """
    
    y_values -= min(y_values)
    y_values /= max(y_values)

    signs = np.sign(np.add(y_values, -0.5))
    zero_crossings = (signs[0:-2]) != signs[1:-1]
    zero_crossings_i = np.where(zero_crossings)[0]
    return [lin_interp(x_values, y_values, zero_crossings_i[0], 0.5),
            lin_interp(x_values, y_values, zero_crossings_i[1], 0.5)]


def Gaussian(x, x0=0, sigma=1):
    """
    Generate a simple Gaussian distribution
    """
    
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(1/2) * ((x - x0) / sigma) ** 2)


def animate_results(results=None, vmin=0, vmax=1, cmap="jet", title="", save=False):
    """
    """
    
    from matplotlib import animation

    fig = plt.figure()
    ax = fig.gca()

    ims = []
    for r in results:
        im = ax.imshow(r, animated=True, vmin=vmin, vmax=vmax, cmap=cmap)

        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=60, repeat_delay=0)

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

    if save:
        # Set up formatting for the movie files
        writer = animation.writers["ffmpeg"]

        ani.save(f"{title}_animation.mp4", writer=writer(fps=30))
    else:
        plt.show()



def main() -> None:
    
    ...


if __name__ == "__main__":
        
    main()
