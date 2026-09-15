"""
curved_air

utils

This module contains utility functions that are used throughout the curved_air package.
"""

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray


def Gaussian1D(
    x: list[float] | NDArray[np.floating],
    x0: float = 0,
    sigma: float | None = None,
    FWHM: float | None = None,
) -> NDArray[np.floating]:
    """
    Generate a simple Gaussian distribution
    """

    if sigma is None and FWHM is None:
        raise ValueError("You must specify either sigma or FWHM.")

    if FWHM is not None:
        assert sigma is None, "You cannot specify both sigma and FWHM."
        sigma: float = FWHM / (2 * np.sqrt(2 * np.log(2)))

    if isinstance(x, list):
        x: NDArray[np.floating] = np.array(list)

    assert isinstance(sigma, float | int)

    profile: NDArray = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
        -(1 / 2) * ((x - x0) / sigma) ** 2
    )

    return profile / np.nanmax(profile)


def SuperGaussian1D(
    x: list[float] | NDArray[np.floating],
    x0: float = 0,
    sigma: float | None = None,
    FWHM: float | None = None,
    n: int = 2,
) -> NDArray[np.floating]:
    """
    Generate a simple Super-Gaussian distribution
    """

    if sigma is None and FWHM is None:
        raise ValueError("You must specify either sigma or FWHM.")

    if FWHM is not None:
        assert sigma is None, "You cannot specify both sigma and FWHM."
        # For a Super-Gaussian, the half-max condition
        # exp(-0.5 * (w/sigma)^(2n)) = 0.5 gives w = sigma * (2 ln 2)^(1/(2n)),
        # so FWHM = 2w -> sigma = FWHM / (2 * (2 ln 2)^(1/(2n))). This reduces to the
        # standard Gaussian conversion only when n = 1.
        sigma: float = FWHM / (2 * (2 * np.log(2)) ** (1 / (2 * n)))

    if isinstance(x, list):
        x: NDArray[np.floating] = np.array(list)

    assert isinstance(sigma, float | int)

    profile: NDArray = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
        -(1 / 2) * ((x - x0) / sigma) ** (2 * n)
    )

    return profile / np.nanmax(profile)


def Lorentzian1D(
    x: list[float] | NDArray[np.floating],
    x0: float = 0,
    gamma: float | None = None,
    FWHM: float | None = None,
) -> NDArray[np.floating]:
    """
    Generate a simple Lorentzian distribution
    """

    if gamma is None and FWHM is None:
        raise ValueError("You must specify either gamma or FWHM.")

    if FWHM is not None:
        assert gamma is None, "You cannot specify both gamma and FWHM."
        gamma: float = FWHM / 2

    if isinstance(x, list):
        x: NDArray[np.floating] = np.array(list)

    assert isinstance(gamma, float | int)

    profile: NDArray = (1 / np.pi) * (gamma / ((x - x0) ** 2 + gamma**2))

    return profile / np.nanmax(profile)


def Voigt1D(
    x: list[float] | NDArray[np.floating],
    x0: float = 0,
    sigma: float | None = None,
    gamma: float | None = None,
    FWHM: float | None = None,
) -> NDArray[np.floating]:
    """
    Generate a simple (true, convolved) Voigt distribution.

    Either:
    - specify `sigma` and `gamma` directly, in which case their absolute values set
      the widths of the Gaussian and Lorentzian components being convolved, or
    - specify `FWHM` together with `sigma` and `gamma`, in which case only the RATIO
      of `sigma` to `gamma` is used to set the relative Gaussian/Lorentzian
      contribution, while their absolute scale is determined (using the Olivero &
      Longbothum (1977) approximation) so that the resulting profile has (very
      closely) the given `FWHM`. Because a true Voigt profile is a convolution,
      scaling `sigma` and `gamma` by the same factor scales the resulting FWHM by
      exactly that factor, so this rescaling is exact to within the (very small)
      accuracy limits of the approximation used to relate FWHM to sigma and gamma.
    """

    from scipy.special import wofz  # noqa: PLC0415

    if FWHM is None:
        if sigma is None or gamma is None:
            raise ValueError(
                "You must specify both sigma and gamma, and optionally FWHM."
            )
    else:
        if sigma is None or gamma is None:
            raise ValueError(
                "You must specify both sigma and gamma (their ratio sets the "
                "relative Gaussian/Lorentzian contribution) together with FWHM."
            )

        # Only the ratio of sigma to gamma matters here: it sets the relative
        # Gaussian/Lorentzian broadening magnitudes. A true Voigt profile's FWHM
        # scales exactly linearly with (sigma, gamma) when both are scaled by the
        # same factor (since it is a convolution), so we can rescale to hit the
        # requested `FWHM` (to within the accuracy of the Olivero & Longbothum
        # approximation used to relate FWHM to sigma and gamma).
        ratio: float = sigma / gamma

        gaussian_fwhm: float = ratio * (2 * np.sqrt(2 * np.log(2)))
        lorentzian_fwhm: float = 2

        total_fwhm: float = 0.5346 * lorentzian_fwhm + np.sqrt(
            0.2166 * lorentzian_fwhm**2 + gaussian_fwhm**2
        )

        scale: float = FWHM / total_fwhm
        gamma = scale
        sigma = ratio * scale

    if isinstance(x, list):
        x: NDArray[np.floating] = np.array(list)

    assert isinstance(sigma, float | int)
    assert isinstance(gamma, float | int)

    profile: NDArray = np.real(wofz(((x - x0) + 1j * gamma) / (sigma * np.sqrt(2)))) / (
        sigma * np.sqrt(2 * np.pi)
    )

    return profile / np.nanmax(profile)


def PseudoVoigt1D(
    x: list[float] | NDArray[np.floating],
    x0: float = 0,
    sigma: float | None = None,
    gamma: float | None = None,
    FWHM: float | None = None,
) -> NDArray[np.floating]:
    """
    Generate a simple Pseudo-Voigt distribution.

    Either:
    - specify `sigma` and `gamma` directly, in which case their absolute values set the
      (independent) widths of the Gaussian and Lorentzian components, or
    - specify `FWHM` together with `sigma` and `gamma`, in which case only the RATIO of
      `sigma` to `gamma` is used, via the Thompson-Cox-Hastings approximation, to set
      the mixing fraction (`eta`) between the Gaussian and Lorentzian components. `FWHM`
      itself then sets the width shared by both components (as is standard for a
      pseudo-Voigt profile), so that the resulting profile has (very closely, to within
      the accuracy of the approximation) the given FWHM.
    """

    if FWHM is None:
        if sigma is None or gamma is None:
            raise ValueError(
                "You must specify both sigma and gamma, and optionally FWHM."
            )

        eta: float = (
            1.36603 * (gamma / (gamma + sigma))
            - 0.47719 * (gamma / (gamma + sigma)) ** 2
            + 0.11116 * (gamma / (gamma + sigma)) ** 3
        )
    else:
        if sigma is None or gamma is None:
            raise ValueError(
                "You must specify both sigma and gamma (their ratio sets the "
                "relative Gaussian/Lorentzian contribution) together with FWHM."
            )

        # Only the ratio of sigma to gamma matters here: it sets the relative
        # Gaussian/Lorentzian broadening magnitudes, from which the mixing
        # fraction `eta` is derived below (Thompson-Cox-Hastings
        # approximation). `FWHM` is then used directly as the width shared by
        # both components, so the resulting profile has (approximately) the
        # requested FWHM.
        ratio: float = sigma / gamma

        gaussian_fwhm: float = ratio * (2 * np.sqrt(2 * np.log(2)))
        lorentzian_fwhm: float = 2

        total_fwhm: float = (
            gaussian_fwhm**5
            + 2.69269 * gaussian_fwhm**4 * lorentzian_fwhm
            + 2.42843 * gaussian_fwhm**3 * lorentzian_fwhm**2
            + 4.47163 * gaussian_fwhm**2 * lorentzian_fwhm**3
            + 0.07842 * gaussian_fwhm * lorentzian_fwhm**4
            + lorentzian_fwhm**5
        ) ** (1 / 5)

        eta: float = (
            1.36603 * (lorentzian_fwhm / total_fwhm)
            - 0.47719 * (lorentzian_fwhm / total_fwhm) ** 2
            + 0.11116 * (lorentzian_fwhm / total_fwhm) ** 3
        )

        sigma: float = FWHM / (2 * np.sqrt(2 * np.log(2)))
        gamma: float = FWHM / 2

    if isinstance(x, list):
        x: NDArray[np.floating] = np.array(list)

    assert isinstance(sigma, float | int)
    assert isinstance(gamma, float | int)

    profile: NDArray = eta * Lorentzian1D(x, x0=x0, gamma=gamma) + (
        1 - eta
    ) * Gaussian1D(x, x0=x0, sigma=sigma)

    return profile / np.nanmax(profile)


def Moffat1D(
    x: list[float] | NDArray[np.floating],
    x0: float = 0,
    alpha: float | None = None,
    beta: float = 1,
    FWHM: float | None = None,
) -> NDArray[np.floating]:
    """
    Generate a simple Moffat distribution
    """

    if beta == 1:
        beta += 1e-6  # Avoid division by zero

    assert beta > 1, "Beta must be greater than 1 for a valid Moffat distribution."

    if alpha is None and FWHM is None:
        raise ValueError("You must specify either alpha or FWHM.")

    if FWHM is not None:
        assert alpha is None, "You cannot specify both alpha and FWHM."
        alpha: float = FWHM / (2 * np.sqrt(2 ** (1 / beta) - 1))

    if isinstance(x, list):
        x: NDArray[np.floating] = np.array(list)

    assert isinstance(alpha, float | int)
    assert isinstance(beta, float | int)

    profile: NDArray = (
        (beta - 1) / (np.pi * alpha**2) * (1 + ((x - x0) / alpha) ** 2) ** (-beta)
    )

    return profile / np.nanmax(profile)


def plot_distributions() -> None:

    xs: NDArray = np.linspace(-4, 4, 300)

    # Use TkAgg backend for non-interactive plotting
    plt.switch_backend("TkAgg")

    fig: plt.Figure = plt.figure()
    ax: plt.Axes = fig.gca()

    ax.axhline(0.5, color="black", ls="--", alpha=0.5)

    ax.plot(xs, Gaussian1D(xs, x0=0, FWHM=1), label="Gaussian")
    ax.plot(xs, Lorentzian1D(xs, x0=0, FWHM=1), label="Lorentzian")

    ax.plot(xs, SuperGaussian1D(xs, x0=0, FWHM=1, n=2), label="Super-Gaussian, n=2")

    ax.plot(
        xs,
        Voigt1D(xs, x0=0, FWHM=1, sigma=0.5, gamma=0.5),
        label="Voigt, sigma=0.5, gamma=0.5",
    )

    ax.plot(
        xs,
        PseudoVoigt1D(xs, x0=0, FWHM=1, sigma=0.5, gamma=0.5),
        label="Pseudo-Voigt, sigma=0.5, gamma=0.5",
    )

    for beta in [2, 4.765]:
        ax.plot(xs, Moffat1D(xs, x0=0, FWHM=1, beta=beta), label=f"Moffat, beta={beta}")

    ax.legend()

    plt.show()


if __name__ == "__main__":
    plot_distributions()
