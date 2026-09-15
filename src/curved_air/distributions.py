"""
curved_air

distributions

This module contains 1D and 2D distributions commonly used in optics.
"""

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from curved_air.plotting import plot_style

plt.style.use(plot_style)
# Use TkAgg backend for non-interactive plotting
plt.switch_backend("TkAgg")


def _as_grid(
    x: list[float] | NDArray[np.floating],
    y: list[float] | NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Coerce `x` and `y` into a 2D grid. If both are already 2D arrays of matching
    shape (e.g. produced by `np.meshgrid`), they are returned as-is; if both are 1D,
    they are combined into a 2D grid via `np.meshgrid`.
    """

    x: NDArray = np.asarray(x, dtype=float)
    y: NDArray = np.asarray(y, dtype=float)

    assert x.ndim == 1 and y.ndim == 1, "Both x and y must be 1D arrays."

    x, y = np.meshgrid(x, y)

    return x, y


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
        x: NDArray[np.floating] = np.array(x)

    assert isinstance(sigma, float | int)

    profile: NDArray = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
        -(1 / 2) * ((x - x0) / sigma) ** 2
    )

    return profile / np.nanmax(profile)


def Gaussian2D(
    x: list[float] | NDArray[np.floating],
    y: list[float] | NDArray[np.floating],
    x0: float = 0,
    y0: float = 0,
    sigma: float | None = None,
    sigma_y: float | None = None,
    FWHM: float | None = None,
    FWHM_y: float | None = None,
) -> NDArray[np.floating]:
    """
    Generate a simple (elliptical) 2D Gaussian distribution.

    Isotropic (circularly symmetric) by default: `sigma_y`/`FWHM_y` default to the
    x-direction `sigma`/`FWHM`. Pass `sigma_y` and/or `FWHM_y` explicitly for an
    elliptical profile.
    """

    if sigma_y is None and FWHM_y is None:
        sigma_y = sigma
        FWHM_y = FWHM

    x, y = _as_grid(x, y)

    profile: NDArray = Gaussian1D(x, x0=x0, sigma=sigma, FWHM=FWHM) * Gaussian1D(
        y, x0=y0, sigma=sigma_y, FWHM=FWHM_y
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
        x: NDArray[np.floating] = np.array(x)

    assert isinstance(sigma, float | int)

    profile: NDArray = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
        -(1 / 2) * ((x - x0) / sigma) ** (2 * n)
    )

    return profile / np.nanmax(profile)


def SuperGaussian2D(
    x: list[float] | NDArray[np.floating],
    y: list[float] | NDArray[np.floating],
    x0: float = 0,
    y0: float = 0,
    sigma: float | None = None,
    sigma_y: float | None = None,
    FWHM: float | None = None,
    FWHM_y: float | None = None,
    n: int = 2,
) -> NDArray[np.floating]:
    """
    Generate a simple (elliptical) 2D Super-Gaussian distribution, i.e. the standard
    flat-top/super-Gaussian beam profile

        I(x, y) = exp(-0.5 * (((x-x0)/sigma)^2 + ((y-y0)/sigma_y)^2)^n)

    Note this uses the combined elliptical radius (the standard optics convention for
    super-Gaussian/flat-top beams), NOT the product of two 1D Super-Gaussians (which
    would not be radially symmetric even in the isotropic case, for n != 1).

    Isotropic (circularly symmetric) by default: `sigma_y`/`FWHM_y` default to the
    x-direction `sigma`/`FWHM`. Pass `sigma_y` and/or `FWHM_y` explicitly for an
    elliptical profile.
    """

    if sigma is None and FWHM is None:
        raise ValueError("You must specify either sigma or FWHM.")

    if FWHM is not None:
        assert sigma is None, "You cannot specify both sigma and FWHM."
        sigma: float = FWHM / (2 * (2 * np.log(2)) ** (1 / (2 * n)))

    if sigma_y is None and FWHM_y is None:
        sigma_y = sigma
    elif FWHM_y is not None:
        assert sigma_y is None, "You cannot specify both sigma_y and FWHM_y."
        sigma_y: float = FWHM_y / (2 * (2 * np.log(2)) ** (1 / (2 * n)))

    x, y = _as_grid(x, y)

    assert isinstance(sigma, float | int)
    assert isinstance(sigma_y, float | int)

    rho_squared: NDArray = ((x - x0) / sigma) ** 2 + ((y - y0) / sigma_y) ** 2
    profile: NDArray = np.exp(-0.5 * rho_squared**n)

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

    assert isinstance(x, (list, np.ndarray)), "x must be a list or numpy array."

    if isinstance(x, list):
        x: NDArray[np.floating] = np.array(x)

    assert isinstance(gamma, float | int)

    profile: NDArray = (1 / np.pi) * (gamma / ((x - x0) ** 2 + gamma**2))

    return profile / np.nanmax(profile)


def Lorentzian2D(
    x: list[float] | NDArray[np.floating],
    y: list[float] | NDArray[np.floating],
    x0: float = 0,
    y0: float = 0,
    gamma: float | None = None,
    gamma_y: float | None = None,
    FWHM: float | None = None,
    FWHM_y: float | None = None,
) -> NDArray[np.floating]:
    """
    Generate a simple (elliptical) 2D Lorentzian distribution, using the combined
    elliptical radius

        profile(x, y) = 1 / (1 + ((x-x0)/gamma)^2 + ((y-y0)/gamma_y)^2)

    Note this is a peak-normalized shape function (like the other functions in this
    module), NOT the properly area-normalized 2D Cauchy distribution used internally
    by `Voigt2D` for its convolution.

    Isotropic (circularly symmetric) by default: `gamma_y`/`FWHM_y` default to the
    x-direction `gamma`/`FWHM`. Pass `gamma_y` and/or `FWHM_y` explicitly for an
    elliptical profile.
    """

    if gamma is None and FWHM is None:
        raise ValueError("You must specify either gamma or FWHM.")

    if FWHM is not None:
        assert gamma is None, "You cannot specify both gamma and FWHM."
        gamma: float = FWHM / 2

    if gamma_y is None and FWHM_y is None:
        gamma_y = gamma
    elif FWHM_y is not None:
        assert gamma_y is None, "You cannot specify both gamma_y and FWHM_y."
        gamma_y: float = FWHM_y / 2

    x, y = _as_grid(x, y)

    assert isinstance(gamma, float | int)
    assert isinstance(gamma_y, float | int)

    rho_squared: NDArray = ((x - x0) / gamma) ** 2 + ((y - y0) / gamma_y) ** 2
    profile: NDArray = 1 / (1 + rho_squared)

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

    assert isinstance(x, (list, np.ndarray)), "x must be a list or numpy array."
    if isinstance(x, list):
        x: NDArray[np.floating] = np.array(x)

    assert isinstance(sigma, float | int)
    assert isinstance(gamma, float | int)

    profile: NDArray = np.real(wofz(((x - x0) + 1j * gamma) / (sigma * np.sqrt(2)))) / (
        sigma * np.sqrt(2 * np.pi)
    )

    return profile / np.nanmax(profile)


def Voigt2D(
    x: list[float] | NDArray[np.floating],
    y: list[float] | NDArray[np.floating],
    x0: float = 0,
    y0: float = 0,
    sigma: float | None = None,
    gamma: float | None = None,
    sigma_y: float | None = None,
    gamma_y: float | None = None,
    FWHM: float | None = None,
    FWHM_y: float | None = None,
) -> NDArray[np.floating]:
    """
    Generate a simple (true, convolved) 2D Voigt distribution: the exact 2D
    convolution (evaluated numerically via FFT) of an elliptical 2D Gaussian and an
    elliptical 2D Lorentzian (Cauchy) distribution.

    Either:
    - specify `sigma`/`gamma` (and, for ellipticity, `sigma_y`/`gamma_y`) directly, in
      which case their absolute values set the widths of the Gaussian and Lorentzian
      components being convolved, or
    - specify `FWHM` together with `sigma` and `gamma`, in which case only the RATIO
      of `sigma` to `gamma` is used to set the relative Gaussian/Lorentzian
      contribution along x (using the same Olivero & Longbothum (1977) approximation
      as `Voigt1D`). For the y-direction, either leave `sigma_y`/`gamma_y`/`FWHM_y`
      unset for an isotropic profile (the same ratio and `FWHM` are reused), or pass
      `FWHM_y` (optionally together with `sigma_y` and `gamma_y`, to also set a
      different ratio for y) for an elliptical profile.

    Note that, unlike the 1D case, using `FWHM`/`FWHM_y` here is only an
    approximation: because a genuinely elliptical 2D convolution does not reduce to
    independent 1D convolutions along x and y, the numerically measured FWHM along
    each axis of the result will be close to, but not exactly, the requested value.

    Because the Lorentzian/Cauchy distribution has long tails, `x` and `y` should
    span a domain that comfortably extends beyond the region of interest, to avoid
    truncation artifacts near the edges of the returned profile.
    """

    from scipy.interpolate import RegularGridInterpolator  # noqa: PLC0415
    from scipy.signal import fftconvolve  # noqa: PLC0415

    if FWHM is None:
        if FWHM_y is not None:
            raise ValueError(
                "FWHM_y can only be specified together with FWHM (i.e. in FWHM mode)."
            )
        if sigma is None or gamma is None:
            raise ValueError(
                "You must specify both sigma and gamma, and optionally FWHM."
            )

        ratio: float = sigma / gamma
    else:
        if sigma is None or gamma is None:
            raise ValueError(
                "You must specify both sigma and gamma (their ratio sets the "
                "relative Gaussian/Lorentzian contribution) together with FWHM."
            )

        ratio = sigma / gamma

        gaussian_fwhm: float = ratio * (2 * np.sqrt(2 * np.log(2)))
        lorentzian_fwhm: float = 2

        total_fwhm: float = 0.5346 * lorentzian_fwhm + np.sqrt(
            0.2166 * lorentzian_fwhm**2 + gaussian_fwhm**2
        )

        scale: float = FWHM / total_fwhm
        gamma = scale
        sigma = ratio * scale

    if sigma_y is None and gamma_y is None and FWHM_y is None:
        sigma_y, gamma_y = sigma, gamma
    elif FWHM_y is not None:
        if (sigma_y is None) != (gamma_y is None):
            raise ValueError(
                "You must specify both sigma_y and gamma_y together, or neither, "
                "to set a y-direction ratio different from x."
            )

        assert isinstance(sigma_y, float | int)
        assert isinstance(gamma_y, float | int)

        ratio_y: float = (sigma_y / gamma_y) if sigma_y is not None else ratio

        gaussian_fwhm_y: float = ratio_y * (2 * np.sqrt(2 * np.log(2)))
        lorentzian_fwhm_y: float = 2

        total_fwhm_y: float = 0.5346 * lorentzian_fwhm_y + np.sqrt(
            0.2166 * lorentzian_fwhm_y**2 + gaussian_fwhm_y**2
        )

        scale_y: float = FWHM_y / total_fwhm_y
        gamma_y = scale_y
        sigma_y = ratio_y * scale_y
    else:
        if sigma_y is None or gamma_y is None:
            raise ValueError(
                "You must specify both sigma_y and gamma_y for the y-direction."
            )

    assert isinstance(sigma, float | int)
    assert isinstance(gamma, float | int)
    assert isinstance(sigma_y, float | int)
    assert isinstance(gamma_y, float | int)

    x, y = _as_grid(x, y)

    # Infer the (assumed uniform) grid spacing from the requested output grid.
    dx: float = float(x[0, 1] - x[0, 0]) if x.shape[1] > 1 else 1.0
    dy: float = float(y[1, 0] - y[0, 0]) if y.shape[0] > 1 else 1.0

    # Build a symmetric, zero-centred, padded grid for the FFT convolution: wide
    # enough to cover the requested domain plus generous padding (in units of the
    # largest width parameter) to capture the Lorentzian's long tails and avoid
    # truncation artifacts.
    pad: float = 30 * max(sigma, gamma, sigma_y, gamma_y)
    half_extent_x: float = np.nanmax(np.abs(x - x0)) + pad
    half_extent_y: float = np.nanmax(np.abs(y - y0)) + pad

    n_u: int = 2 * int(np.ceil(half_extent_x / dx)) + 1
    n_v: int = 2 * int(np.ceil(half_extent_y / dy)) + 1

    u: NDArray = (np.arange(n_u) - n_u // 2) * dx
    v: NDArray = (np.arange(n_v) - n_v // 2) * dy
    U, V = np.meshgrid(u, v)

    # Area-normalized elliptical 2D Gaussian and Lorentzian (Cauchy) PDFs, so that
    # their FFT convolution is a proper convolution integral.
    gaussian_kernel: NDArray = (1 / (2 * np.pi * sigma * sigma_y)) * np.exp(
        -0.5 * (U**2 / sigma**2 + V**2 / sigma_y**2)
    )
    lorentzian_kernel: NDArray = (1 / (2 * np.pi * gamma * gamma_y)) * (
        1 + U**2 / gamma**2 + V**2 / gamma_y**2
    ) ** (-1.5)

    convolved: NDArray = (
        fftconvolve(gaussian_kernel, lorentzian_kernel, mode="same") * dx * dy
    )

    # Interpolate the (zero-centred) convolution back onto the requested,
    # (x0, y0)-offset output grid.
    interpolator = RegularGridInterpolator(
        (v, u), convolved, bounds_error=False, fill_value=0.0
    )
    points: NDArray = np.stack([(y - y0).ravel(), (x - x0).ravel()], axis=-1)
    profile: NDArray = interpolator(points).reshape(x.shape)

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

    assert isinstance(x, (list, np.ndarray)), "x must be a list or numpy array."
    if isinstance(x, list):
        x: NDArray[np.floating] = np.array(x)

    assert isinstance(sigma, float | int)
    assert isinstance(gamma, float | int)

    profile: NDArray = eta * Lorentzian1D(x, x0=x0, gamma=gamma) + (
        1 - eta
    ) * Gaussian1D(x, x0=x0, sigma=sigma)

    return profile / np.nanmax(profile)


def PseudoVoigt2D(
    x: list[float] | NDArray[np.floating],
    y: list[float] | NDArray[np.floating],
    x0: float = 0,
    y0: float = 0,
    sigma: float | None = None,
    gamma: float | None = None,
    sigma_y: float | None = None,
    gamma_y: float | None = None,
    FWHM: float | None = None,
    FWHM_y: float | None = None,
) -> NDArray[np.floating]:
    """
    Generate a simple (elliptical) 2D Pseudo-Voigt distribution:

        PV(x, y) = eta * Lorentzian2D(x, y; gamma, gamma_y)
                   + (1 - eta) * Gaussian2D(x, y; sigma, sigma_y)

    A single mixing fraction `eta` (computed exactly as in `PseudoVoigt1D`, from the
    x-direction `sigma`/`gamma`, or their ratio together with `FWHM`) is shared by
    both the Gaussian and Lorentzian components along x and y.

    Isotropic (circularly symmetric) by default: `sigma_y`/`gamma_y` (direct mode) or
    `FWHM_y` (FWHM mode) default to the x-direction values. Pass them explicitly for
    an elliptical profile.
    """

    if FWHM is None:
        if FWHM_y is not None:
            raise ValueError(
                "FWHM_y can only be specified together with FWHM (i.e. in FWHM mode)."
            )
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

    if sigma_y is None and gamma_y is None and FWHM_y is None:
        sigma_y = sigma
        gamma_y = gamma
    elif FWHM_y is not None:
        assert sigma_y is None and gamma_y is None, (
            "You cannot specify sigma_y/gamma_y together with FWHM_y."
        )
        sigma_y = FWHM_y / (2 * np.sqrt(2 * np.log(2)))
        gamma_y = FWHM_y / 2
    else:
        if sigma_y is None or gamma_y is None:
            raise ValueError(
                "You must specify both sigma_y and gamma_y for the y-direction."
            )

    profile: NDArray = eta * Lorentzian2D(
        x, y, x0=x0, y0=y0, gamma=gamma, gamma_y=gamma_y
    ) + (1 - eta) * Gaussian2D(x, y, x0=x0, y0=y0, sigma=sigma, sigma_y=sigma_y)

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
        x: NDArray[np.floating] = np.array(x)

    assert isinstance(alpha, float | int)
    assert isinstance(beta, float | int)

    profile: NDArray = (
        (beta - 1) / (np.pi * alpha**2) * (1 + ((x - x0) / alpha) ** 2) ** (-beta)
    )

    return profile / np.nanmax(profile)


def Moffat2D(
    x: list[float] | NDArray[np.floating],
    y: list[float] | NDArray[np.floating],
    x0: float = 0,
    y0: float = 0,
    alpha: float | None = None,
    alpha_y: float | None = None,
    beta: float = 1,
    FWHM: float | None = None,
    FWHM_y: float | None = None,
) -> NDArray[np.floating]:
    """
    Generate a simple (elliptical) 2D Moffat distribution, using the combined
    elliptical radius

        profile(x, y) = (1 + ((x-x0)/alpha)^2 + ((y-y0)/alpha_y)^2)^(-beta)

    Isotropic (circularly symmetric) by default: `alpha_y`/`FWHM_y` default to the
    x-direction `alpha`/`FWHM`. Pass `alpha_y` and/or `FWHM_y` explicitly for an
    elliptical profile.
    """

    if beta == 1:
        beta += 1e-6  # Avoid division by zero

    assert beta > 1, "Beta must be greater than 1 for a valid Moffat distribution."

    if alpha is None and FWHM is None:
        raise ValueError("You must specify either alpha or FWHM.")

    if FWHM is not None:
        assert alpha is None, "You cannot specify both alpha and FWHM."
        alpha: float = FWHM / (2 * np.sqrt(2 ** (1 / beta) - 1))

    if alpha_y is None and FWHM_y is None:
        alpha_y = alpha
    elif FWHM_y is not None:
        assert alpha_y is None, "You cannot specify both alpha_y and FWHM_y."
        alpha_y: float = FWHM_y / (2 * np.sqrt(2 ** (1 / beta) - 1))

    x, y = _as_grid(x, y)

    assert isinstance(alpha, float | int)
    assert isinstance(alpha_y, float | int)
    assert isinstance(beta, float | int)

    rho_squared: NDArray = ((x - x0) / alpha) ** 2 + ((y - y0) / alpha_y) ** 2
    profile: NDArray = (1 + rho_squared) ** (-beta)

    return profile / np.nanmax(profile)


def plot_distributions1D() -> None:

    xs: NDArray = np.linspace(-3, 3, 300)

    fig: plt.Figure = plt.figure(figsize=(12, 8))
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

    ax.set_xlim(-3, 3)
    ax.set_ylim(-0.05, 1.05)

    ax.legend()

    plt.show()


def plot_distributions2D() -> None:

    xs: NDArray = np.linspace(-3, 3, 300)
    ys: NDArray = np.linspace(-3, 3, 300)

    _fig, axs = plt.subplots(2, 3, figsize=(12, 8))

    ax = axs[0, 0]
    ax.imshow(Gaussian2D(xs, ys, x0=0, y0=0, FWHM=1))
    ax.set_title("Gaussian2D")

    ax = axs[0, 1]
    ax.imshow(Lorentzian2D(xs, ys, x0=0, y0=0, FWHM=1))
    ax.set_title("Lorentzian2D")

    ax = axs[0, 2]
    ax.imshow(SuperGaussian2D(xs, ys, x0=0, y0=0, FWHM=1, n=2))
    ax.set_title("SuperGaussian2D, n=2")

    ax = axs[1, 0]
    ax.imshow(Voigt2D(xs, ys, x0=0, y0=0, FWHM=1, sigma=0.5, gamma=0.5))
    ax.set_title("Voigt2D")

    ax = axs[1, 1]
    ax.imshow(PseudoVoigt2D(xs, ys, x0=0, y0=0, FWHM=1, sigma=0.5, gamma=0.5))
    ax.set_title("PseudoVoigt2D")

    ax = axs[1, 2]
    ax.imshow(Moffat2D(xs, ys, x0=0, y0=0, FWHM=1, beta=2))
    ax.set_title("Moffat2D, beta=2")

    plt.show()


if __name__ == "__main__":
    plot_distributions1D()
    # plot_distributions2D()
