# /// script
# dependencies = [
#     "curved_air>=0.1.0",
# ]
# [tool.uv.sources]
# curved_air = { path = "..", editable = true }
# ///

""" """

import numpy as np
from matplotlib import pyplot as plt
from numpy.fft import fft, fftshift, ifft  # noqa: F401

from curved_air.plotting import plot_style
from curved_air.utils import Gaussian

plt.style.use(plot_style)

plt.rcParams["axes.autolimit_mode"] = "data"


def main() -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

    lw = 3
    alpha = 0.75

    sigma = 1 / np.sqrt(2 * np.pi)
    fft_sigma = 1 / (2 * np.pi * sigma)

    grid = np.linspace(-10 * sigma, 10 * sigma, 2_000)

    g = Gaussian(x=grid, x0=0, sigma=sigma)
    ax1.plot(
        grid / sigma,
        g,
        lw=lw,
        alpha=alpha,
        label="Gaussian",
    )

    g_fft = fft(g)
    g_fft_shifted = fftshift(g_fft)
    freq = np.fft.fftfreq(n=grid.size, d=(grid[1] - grid[0]))
    freq_shifted = fftshift(freq)

    ax2.plot(
        freq_shifted / fft_sigma,
        np.abs(g_fft_shifted),
        lw=lw,
        alpha=alpha,
        label="Fourier Transform",
    )

    for sigma_sampling in [1, 2, 3]:
        g_fft_shifted_bandwidth_limited = g_fft_shifted.copy()

        g_fft_shifted_bandwidth_limited[
            np.abs(freq_shifted) > sigma_sampling * fft_sigma
        ] = 0

        ax2.plot(
            freq_shifted / fft_sigma,
            np.abs(g_fft_shifted_bandwidth_limited),
            lw=lw,
            alpha=alpha,
            label=f"Bandwidth-limited to {sigma_sampling}-" + "$\\sigma^{-1}$",
        )

        g_bandwidth_limited = ifft(fftshift(g_fft_shifted_bandwidth_limited)).real
        ax1.plot(
            grid / sigma,
            g_bandwidth_limited,
            lw=lw,
            alpha=alpha,
            label=f"Reconstructed ($\\sigma=${sigma_sampling} pixels)",
        )

    ax1.legend()
    ax2.legend()

    ax1.set_xlabel("Spatial coordinate ($\\sigma$)")
    ax2.set_xlabel("Spatial frequency ($\\sigma^{-1}$)")

    for ax in [ax1, ax2]:
        ax.set_xlim(-6, 6)
        # ax.set_ylabel("")

    plt.show()


if __name__ == "__main__":
    main()
