"""
curved_air

interferograms

fitting

This module contains functions for fitting various surfaces (plane, sphere, cylinder,
saddle, trefoil) to 2D data such as interferograms. Each of these functions returns the
fitted surface, the residuals between the fitted surface and the original data, and a
dictionary of the fit coefficients.

Also included is the required code for constructing grids over which the best-fit
surfaces are to be calculated.
"""

import numpy as np
from numpy.typing import NDArray


def make_grid(x_or_y: NDArray, z_shape: tuple[int, int]) -> NDArray:
    """
    Create a 2D grid from a 1D array of x or y coordinates.
    """
    nrows, ncols = z_shape
    if x_or_y.ndim == 1:
        if x_or_y.size == ncols:
            return np.tile(x_or_y, (nrows, 1))
        if x_or_y.size == nrows:
            return np.tile(x_or_y[:, None], (1, ncols))
        raise ValueError(
            "1D x_or_y must have length equal to number of rows or columns"
        )
    if x_or_y.shape == z_shape:
        return x_or_y
    raise ValueError(
        "x_or_y must be 1D (length = nrows or ncols) or 2D matching data shape"
    )


def best_fit_plane(
    data: NDArray,
    mask: NDArray | None = None,
    x: NDArray | None = None,
    y: NDArray | None = None,
) -> tuple[NDArray, NDArray, dict[str, float]]:
    """
    Fit a best-fit plane to a 2D interferogram and return the fitted plane and residual.
    """

    z: NDArray = np.asarray(data, dtype=float)
    assert z.ndim == 2, "data must be a 2D array"  # noqa: PLR2004

    nrows, ncols = z.shape

    if x is None:
        x_grid: NDArray = np.tile(np.arange(ncols, dtype=float), (nrows, 1))
    else:
        x_grid: NDArray = make_grid(x_or_y=x, z_shape=z.shape)

    if y is None:
        y_grid: NDArray = np.tile(np.arange(nrows, dtype=float)[:, None], (1, ncols))
    else:
        y_grid: NDArray = make_grid(x_or_y=y, z_shape=z.shape)

    # Now we have x_grid, y_grid, and z all as 2D arrays of the same shape

    # Create a mask of valid points, apply the user-provided mask if given
    valid: NDArray = np.isfinite(z)
    if mask is not None:
        m: NDArray = np.asarray(mask, dtype=bool)
        if m.shape != z.shape:
            raise ValueError("mask must match data shape")
        valid &= m

    assert np.count_nonzero(valid) >= 3, "Not enough valid points to fit a plane!"  # noqa: PLR2004

    xv: NDArray = x_grid[valid]
    yv: NDArray = y_grid[valid]
    zv: NDArray = z[valid]

    # Linear least-squares fit to: z = Ax + By + C
    A_matrix = np.column_stack((xv, yv, np.ones_like(xv)))
    coeffs, *_ = np.linalg.lstsq(A_matrix, zv, rcond=None)
    A_coeff, B_coeff, C_coeff = coeffs

    fitted: NDArray = A_coeff * x_grid + B_coeff * y_grid + C_coeff
    residual: NDArray = z - fitted

    return fitted, residual, {"A": A_coeff, "B": B_coeff, "C": C_coeff}


def best_fit_sphere(
    data: NDArray,
    mask: NDArray | None = None,
    x: NDArray | None = None,
    y: NDArray | None = None,
) -> tuple[NDArray, NDArray, dict[str, float]]:
    """
    Fit a best-fit sphere to a 2D interferogram and return the fitted sphere and residual.
    """

    z: NDArray = np.asarray(data, dtype=float)
    assert z.ndim == 2, "Data must be a 2D array!"  # noqa: PLR2004

    nrows, ncols = z.shape

    if x is None:
        x_grid: NDArray = np.tile(np.arange(ncols, dtype=float), (nrows, 1))
    else:
        x_grid: NDArray = make_grid(x_or_y=x, z_shape=z.shape)

    if y is None:
        y_grid: NDArray = np.tile(np.arange(nrows, dtype=float)[:, None], (1, ncols))
    else:
        y_grid: NDArray = make_grid(x_or_y=y, z_shape=z.shape)

    # Now we have x_grid, y_grid, and z all as 2D arrays of the same shape

    # Create a mask of valid points, apply the user-provided mask if given
    valid: NDArray = np.isfinite(z)
    if mask is not None:
        m: NDArray = np.asarray(mask, dtype=bool)
        if m.shape != z.shape:
            raise ValueError("mask must match data shape")
        valid &= m

    # Quick check to ensure there are enough valid points to fit a sphere
    assert np.count_nonzero(valid) >= 4, "Not enough valid points to fit a sphere!"  # noqa: PLR2004

    xv: NDArray = x_grid[valid]
    yv: NDArray = y_grid[valid]
    zv: NDArray = z[valid]

    # Linear least-squares fit to: x^2 + y^2 + z^2 + Ax + By + Cz + D = 0
    A = np.column_stack((xv, yv, zv, np.ones_like(xv)))
    b = -(xv * xv + yv * yv + zv * zv)
    coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
    a, b_, c, d = coeffs

    x0 = -0.5 * a
    y0 = -0.5 * b_
    z0 = -0.5 * c
    rad_sq = x0 * x0 + y0 * y0 + z0 * z0 - d
    if rad_sq <= 0:
        raise ValueError("fitted sphere has non-positive radius squared")
    R = np.sqrt(rad_sq)

    rr_sq = R * R - (x_grid - x0) ** 2 - (y_grid - y0) ** 2
    rr_sq = np.clip(rr_sq, 0.0, None)

    # Choose the sphere branch that best matches data on valid points.
    z_plus = z0 + np.sqrt(rr_sq)
    z_minus = z0 - np.sqrt(rr_sq)
    err_plus = np.nanmean((z[valid] - z_plus[valid]) ** 2)
    err_minus = np.nanmean((z[valid] - z_minus[valid]) ** 2)
    fitted = z_plus if err_plus <= err_minus else z_minus

    residual = z - fitted
    return fitted, residual, {"x0": x0, "y0": y0, "z0": z0, "R": R}


def best_fit_cylinder(
    data: NDArray,
    mask: NDArray | None = None,
    x: NDArray | None = None,
    y: NDArray | None = None,
) -> tuple[NDArray, NDArray, dict[str, float]]:
    """
    Fit a best-fit cylinder to a 2D interferogram and return the fitted cylinder and residual.
    """

    z: NDArray = np.asarray(data, dtype=float)
    assert z.ndim == 2, "Data must be a 2D array!"  # noqa: PLR2004

    nrows, ncols = z.shape

    if x is None:
        x_grid: NDArray = np.tile(np.arange(ncols, dtype=float), (nrows, 1))
    else:
        x_grid: NDArray = make_grid(x_or_y=x, z_shape=z.shape)

    if y is None:
        y_grid: NDArray = np.tile(np.arange(nrows, dtype=float)[:, None], (1, ncols))
    else:
        y_grid: NDArray = make_grid(x_or_y=y, z_shape=z.shape)

    # Now we have x_grid, y_grid, and z all as 2D arrays of the same shape

    # Create a mask of valid points, apply the user-provided mask if given
    valid: NDArray = np.isfinite(z)
    if mask is not None:
        m: NDArray = np.asarray(mask, dtype=bool)
        if m.shape != z.shape:
            raise ValueError("mask must match data shape")
        valid &= m

    # Quick check to ensure there are enough valid points to fit a cylinder
    assert np.count_nonzero(valid) >= 3, "Not enough valid points to fit a cylinder!"  # noqa: PLR2004

    xv: NDArray = x_grid[valid]
    yv: NDArray = y_grid[valid]
    zv: NDArray = z[valid]

    # Linear least-squares fit to: z = Ax^2 + By + C
    A_matrix = np.column_stack((xv * xv, yv, np.ones_like(xv)))
    coeffs, *_ = np.linalg.lstsq(A_matrix, zv, rcond=None)
    A_coeff, B_coeff, C_coeff = coeffs

    fitted: NDArray = A_coeff * x_grid * x_grid + B_coeff * y_grid + C_coeff
    residual: NDArray = z - fitted

    return fitted, residual, {"A": A_coeff, "B": B_coeff, "C": C_coeff}


def best_fit_saddle(
    data: NDArray,
    mask: NDArray | None = None,
    x: NDArray | None = None,
    y: NDArray | None = None,
) -> tuple[NDArray, NDArray, dict[str, float]]:
    """
    Fit a best-fit saddle (general quadratic surface) to a 2D interferogram and
    return the fitted surface and residual.

    Model:
        z = A*x^2 + B*x*y + C*y^2 + D*x + E*y + F
    """

    z: NDArray = np.asarray(data, dtype=float)
    assert z.ndim == 2, "Data must be a 2D array!"  # noqa: PLR2004

    nrows, ncols = z.shape

    if x is None:
        x_grid: NDArray = np.tile(np.arange(ncols, dtype=float), (nrows, 1))
    else:
        x_grid: NDArray = make_grid(x_or_y=x, z_shape=z.shape)

    if y is None:
        y_grid: NDArray = np.tile(np.arange(nrows, dtype=float)[:, None], (1, ncols))
    else:
        y_grid: NDArray = make_grid(x_or_y=y, z_shape=z.shape)

    valid: NDArray = np.isfinite(z)
    if mask is not None:
        m: NDArray = np.asarray(mask, dtype=bool)
        if m.shape != z.shape:
            raise ValueError("mask must match data shape")
        valid &= m

    # Need at least one point per parameter for a stable 6-parameter fit.
    assert np.count_nonzero(valid) >= 6, "Not enough valid points to fit a saddle!"  # noqa: PLR2004

    xv: NDArray = x_grid[valid]
    yv: NDArray = y_grid[valid]
    zv: NDArray = z[valid]

    # Linear least-squares fit to: z = A*x^2 + B*x*y + C*y^2 + D*x + E*y + F
    A_matrix = np.column_stack((xv * xv, xv * yv, yv * yv, xv, yv, np.ones_like(xv)))
    coeffs, *_ = np.linalg.lstsq(A_matrix, zv, rcond=None)
    A_coeff, B_coeff, C_coeff, D_coeff, E_coeff, F_coeff = coeffs

    fitted: NDArray = (
        A_coeff * x_grid * x_grid
        + B_coeff * x_grid * y_grid
        + C_coeff * y_grid * y_grid
        + D_coeff * x_grid
        + E_coeff * y_grid
        + F_coeff
    )
    residual: NDArray = z - fitted

    # For z = Ax^2 + Bxy + Cy^2 + ..., B^2 - 4AC > 0 indicates a saddle.
    discriminant: float = B_coeff * B_coeff - 4.0 * A_coeff * C_coeff

    return (
        fitted,
        residual,
        {
            "A": A_coeff,
            "B": B_coeff,
            "C": C_coeff,
            "D": D_coeff,
            "E": E_coeff,
            "F": F_coeff,
            "discriminant": discriminant,
        },
    )


def best_fit_trefoil(
    data: NDArray,
    mask: NDArray | None = None,
    x: NDArray | None = None,
    y: NDArray | None = None,
) -> tuple[NDArray, NDArray, dict[str, float]]:
    """
    Fit a best-fit trefoil-like surface to a 2D interferogram and return
    the fitted surface and residual.

    Model (Cartesian trefoil basis + tilt + piston):
        z = T_cos * (x^3 - 3*x*y^2)
          + T_sin * (3*x^2*y - y^3)
          + P*x + Q*y + R

    The first two terms correspond to r^3*cos(3θ) and r^3*sin(3θ).
    """

    z: NDArray = np.asarray(data, dtype=float)
    assert z.ndim == 2, "Data must be a 2D array!"  # noqa: PLR2004

    nrows, ncols = z.shape

    if x is None:
        x_grid: NDArray = np.tile(np.arange(ncols, dtype=float), (nrows, 1))
    else:
        x_grid = make_grid(x_or_y=x, z_shape=z.shape)

    if y is None:
        y_grid: NDArray = np.tile(np.arange(nrows, dtype=float)[:, None], (1, ncols))
    else:
        y_grid = make_grid(x_or_y=y, z_shape=z.shape)

    valid: NDArray = np.isfinite(z)
    if mask is not None:
        m: NDArray = np.asarray(mask, dtype=bool)
        if m.shape != z.shape:
            raise ValueError("mask must match data shape")
        valid &= m

    # 5 parameters: T_cos, T_sin, P, Q, R
    assert np.count_nonzero(valid) >= 5, "Not enough valid points to fit trefoil!"  # noqa: PLR2004

    xv: NDArray = x_grid[valid]
    yv: NDArray = y_grid[valid]
    zv: NDArray = z[valid]

    trefoil_cos = xv * xv * xv - 3.0 * xv * yv * yv
    trefoil_sin = 3.0 * xv * xv * yv - yv * yv * yv

    A_matrix = np.column_stack((trefoil_cos, trefoil_sin, xv, yv, np.ones_like(xv)))
    coeffs, *_ = np.linalg.lstsq(A_matrix, zv, rcond=None)
    t_cos, t_sin, p_coeff, q_coeff, r_coeff = coeffs

    fitted: NDArray = (
        t_cos * (x_grid * x_grid * x_grid - 3.0 * x_grid * y_grid * y_grid)
        + t_sin * (3.0 * x_grid * x_grid * y_grid - y_grid * y_grid * y_grid)
        + p_coeff * x_grid
        + q_coeff * y_grid
        + r_coeff
    )
    residual: NDArray = z - fitted

    amplitude = float(np.hypot(t_cos, t_sin))
    orientation_rad = float(np.arctan2(t_sin, t_cos) / 3.0)

    return (
        fitted,
        residual,
        {
            "T_cos": t_cos,
            "T_sin": t_sin,
            "P": p_coeff,
            "Q": q_coeff,
            "R": r_coeff,
            "amplitude": amplitude,
            "orientation_rad": orientation_rad,
        },
    )
