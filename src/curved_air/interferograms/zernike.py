"""
curved_air

interferograms

zernike

This module contains functions for analyzing interferograms with Zernike polynomials
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from curved_air.interferograms.fitting import make_grid

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray


# Common names for the low-order Noll-indexed Zernike terms, as typically labelled by
# commercial interferometry software. Terms beyond this table fall back to a generic
# "Z{j} (n=.., m=..)" label - see `zernike_name`.
ZERNIKE_NOLL_NAMES: dict[int, str] = {
    1: "Piston",
    2: "Tilt X",
    3: "Tilt Y",
    4: "Defocus",
    5: "Astigmatism (oblique, 45°)",
    6: "Astigmatism (vertical, 0°)",
    7: "Coma Y (vertical)",
    8: "Coma X (horizontal)",
    9: "Trefoil Y (vertical)",
    10: "Trefoil X (oblique)",
    11: "Primary Spherical",
    12: "Secondary Astigmatism (vertical)",
    13: "Secondary Astigmatism (oblique)",
    14: "Quadrafoil (vertical)",
    15: "Quadrafoil (oblique)",
    16: "Secondary Coma X (horizontal)",
    17: "Secondary Coma Y (vertical)",
    18: "Secondary Trefoil X (oblique)",
    19: "Secondary Trefoil Y (vertical)",
    20: "Pentafoil (oblique)",
    21: "Pentafoil (vertical)",
    22: "Secondary Spherical",
}


def noll_to_n_m(j: int) -> tuple[int, int]:
    """
    Convert a 1-based Noll Zernike index `j` into its radial order `n` and signed
    azimuthal frequency `m` (m > 0 selects the cosine term, m < 0 selects the sine term,
    m == 0 is rotationally symmetric).
    """

    assert j >= 1, "Noll index j must be a positive integer (indices start at 1)"  # noqa: PLR2004

    n = 0
    j1: int = j - 1
    while j1 > n:
        n += 1
        j1 -= n

    m: int = (-1) ** j * ((n % 2) + 2 * ((j1 + ((n + 1) % 2)) // 2))
    return n, m


def zernike_name(j: int, n: int | None = None, m: int | None = None) -> str:
    """
    Return a human-readable name for the Zernike term with Noll index `j`, falling back
    to a generic "Z{j} (n=.., m=..)" label for terms of a sufficiently high order
    """

    if j in ZERNIKE_NOLL_NAMES:
        return ZERNIKE_NOLL_NAMES[j]

    if n is None or m is None:
        n, m = noll_to_n_m(j)
    return f"Z{j} (n={n}, m={m:+d})"


def zernike_radial(n: int, m_abs: int, rho: NDArray) -> NDArray:
    """
    Evaluate the (unnormalized) Zernike radial polynomial R_n^|m|(rho).
    """

    R: NDArray = np.zeros_like(rho, dtype=float)
    for k in range((n - m_abs) // 2 + 1):
        coeff: float = (
            (-1) ** k
            * math.factorial(n - k)
            / (
                math.factorial(k)
                * math.factorial((n + m_abs) // 2 - k)
                * math.factorial((n - m_abs) // 2 - k)
            )
        )
        R += coeff * rho ** (n - 2 * k)
    return R


def zernike_noll(j: int, rho: NDArray, theta: NDArray) -> NDArray:
    """
    Evaluate the Noll-normalized Zernike polynomial Z_j(rho, theta).

    Terms are normalized so that each one (other than piston) has unit RMS over the unit
    disk. Because the basis is then orthonormal, a least-squares fit against it lets the
    total wavefront variance be decomposed exactly as the sum of each term's squared
    coefficient - this is what makes the per-term "relative weight" in
    `ZernikeFitResult.weights` meaningful.
    """

    n, m = noll_to_n_m(j)
    m_abs: int = abs(m)
    radial: NDArray = zernike_radial(n, m_abs, rho)

    if m == 0:
        norm: float = math.sqrt(n + 1)
        azimuthal: NDArray | float = 1.0
    elif m > 0:
        norm: float = math.sqrt(2 * (n + 1))
        azimuthal: NDArray = np.cos(m_abs * theta)
    else:
        norm: float = math.sqrt(2 * (n + 1))
        azimuthal: NDArray = np.sin(m_abs * theta)

    return norm * radial * azimuthal


def zernike_basis(max_term: int, rho: NDArray, theta: NDArray) -> NDArray:
    """
    Build a stack of Noll-ordered Zernike basis terms (Z_1 ... Z_max_term) evaluated
    over the given rho/theta grids.

    Returns an array of shape (max_term, *rho.shape).
    """

    return np.stack(
        [zernike_noll(j, rho, theta) for j in range(1, max_term + 1)],
        axis=0,
    )


@dataclass
class ZernikeFitResult:
    """
    The result of fitting Zernike polynomials to a 2D phase map.

    Holds the fitted coefficients (one per Noll term), the fitted surface and overall
    residual, and convenience methods for inspecting relative term weights or
    subtracting individual terms from the phase.
    """

    coefficients: NDArray
    terms: list[int]
    n_m: list[tuple[int, int]]
    names: list[str]
    center: tuple[float, float]
    radius: float
    rho: NDArray
    theta: NDArray
    domain_mask: NDArray
    fit_mask: NDArray
    data: NDArray
    fitted: NDArray
    residual: NDArray

    def term_surface(self, terms: int | Sequence[int]) -> NDArray:
        """
        Reconstruct the phase surface contributed by one or more fitted Zernike
        terms (specified by Noll index), evaluated over the fitted aperture.
        """

        term_list: list[int] = [terms] if isinstance(terms, int) else list(terms)
        unknown: list[int] = [t for t in term_list if t not in self.terms]
        if unknown:
            raise ValueError(f"Terms {unknown} were not included in this fit")

        surface: NDArray = np.zeros_like(self.rho, dtype=float)
        for t in term_list:
            idx: int = self.terms.index(t)
            surface += self.coefficients[idx] * zernike_noll(t, self.rho, self.theta)

        surface[~self.domain_mask] = np.nan
        return surface

    def remove(
        self,
        terms: int | Sequence[int],
        data: NDArray | None = None,
    ) -> tuple[NDArray, NDArray]:
        """
        Subtract one or more fitted Zernike terms (by Noll index) from phase data.

        Returns a tuple of (removed_surface, residual), where `residual` is the input
        data (or the data used in the original fit, if `data` is not supplied) minus the
        contribution of only the requested terms - any other fitted or unfitted
        structure is left in place.
        """

        removed_surface: NDArray = self.term_surface(terms)
        base_data: NDArray = (
            self.data if data is None else np.asarray(data, dtype=float)
        )
        residual: NDArray = base_data - removed_surface
        return removed_surface, residual

    def weights(self, exclude: Sequence[int] = (1,)) -> dict[int, float]:
        """
        Compute the relative weight (fraction of total wavefront variance) carried by
        each fitted Zernike term.

        Because the Zernike terms are fit here using an orthonormal (Noll) basis, the
        variance contributed by term `j` is simply `coefficient_j ** 2`, and its
        relative weight is that value divided by the sum across all included terms. By
        default, piston (term 1) is excluded, since it carries no wavefront error
        information; pass e.g. `exclude=(1, 2, 3)` to also exclude tilt.
        """

        variance: NDArray = self.coefficients**2
        included: NDArray = np.array([t not in exclude for t in self.terms], dtype=bool)
        total: float = float(np.sum(variance[included]))

        if total <= 0:
            return dict.fromkeys(self.terms, 0.0)

        return {
            t: (float(variance[i]) / total if included[i] else 0.0)
            for i, t in enumerate(self.terms)
        }

    def table(self, exclude_from_weights: Sequence[int] = (1,)) -> list[dict]:
        """
        Build a list of per-term dictionaries (term, n, m, name, coefficient, weight),
        suitable for direct conversion into a pandas DataFrame or display table, e.g.
        for showing relative Zernike term weights in the UI.
        """

        weights: dict[int, float] = self.weights(exclude=exclude_from_weights)
        return [
            {
                "term": t,
                # "n": n,
                # "m": m,
                "name": name,
                "coefficient": float(coeff),
                "weight": weights[t],
            }
            for t, (n, m), name, coeff in zip(
                self.terms, self.n_m, self.names, self.coefficients, strict=True
            )
        ]

    @property
    def rms_residual(self) -> float:
        """
        RMS of the residual phase, over points that were actually fit
        """

        valid: NDArray = self.fit_mask & np.isfinite(self.residual)
        return float(np.sqrt(np.mean(self.residual[valid] ** 2)))

    @property
    def ptv_residual(self) -> float:
        """
        Peak-to-valley of the residual phase, over points that were actually fit
        """

        valid: NDArray = self.fit_mask & np.isfinite(self.residual)
        vals: NDArray = self.residual[valid]
        return float(np.max(vals) - np.min(vals))


def fit_zernike(
    data: NDArray,
    max_term: int = 37,
    mask: NDArray | None = None,
    x: NDArray | None = None,
    y: NDArray | None = None,
    center: tuple[float, float] | None = None,
    radius: float | None = None,
) -> ZernikeFitResult:
    """
    Fit Noll-ordered Zernike polynomials (terms 1 .. max_term) to a 2D phase map,
    similar to the Zernike decomposition offered by commercial interferometry software

    Parameters
    ----------
    data:
        The 2D phase map to fit.
    max_term:
        The highest Noll term index to fit (inclusive). For example, an input of 12 fits
        terms 1 (piston) through 12 (secondary astigmatism).
    mask:
        An optional boolean array (same shape as `data`); False marks points to exclude
        from the fit.
    x, y:
        Optional 1D or 2D coordinate arrays (see `make_grid`). If omitted, pixel indices
        are used.
    center:
        Optional (x0, y0) center of the circular aperture used for fitting. If omitted,
        the centroid of the valid data points is used.
    radius:
        Optional radius of the circular aperture (in the same units as `x`/`y`). If
        omitted, the distance from `center` to the farthest valid data point is used, so
        the whole valid region sits inside the unit disk.

    Returns
    -------
    A `ZernikeFitResult` with per-term coefficients and names, the fitted surface, and
    the residual (data minus all fitted terms up to `max_term`).
    """

    z: NDArray = np.asarray(data, dtype=float)
    assert z.ndim == 2, "data must be a 2D array"  # noqa: PLR2004
    assert max_term >= 1, "max_term must be at least 1"  # noqa: PLR2004

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
        m_arr: NDArray = np.asarray(mask, dtype=bool)
        if m_arr.shape != z.shape:
            raise ValueError("mask must match data shape")
        valid &= m_arr

    assert np.count_nonzero(valid) >= max_term, (  # noqa: PLR2004
        "Not enough valid points to fit the requested number of Zernike terms!"
    )

    if center is None:
        x0: float = float(np.mean(x_grid[valid]))
        y0: float = float(np.mean(y_grid[valid]))
    else:
        x0, y0 = center

    dx: NDArray = x_grid - x0
    dy: NDArray = y_grid - y0

    if radius is None:
        r: float = float(np.sqrt(np.max(dx[valid] ** 2 + dy[valid] ** 2)))
        if r <= 0:
            raise ValueError("Could not determine a positive fit radius from the data")
    else:
        r = float(radius)
        assert r > 0, "radius must be positive"  # noqa: PLR2004

    rho: NDArray = np.sqrt(dx**2 + dy**2) / r
    theta: NDArray = np.arctan2(dy, dx)

    domain_mask: NDArray = rho <= 1.0 + 1e-9
    fit_mask: NDArray = valid & domain_mask

    assert np.count_nonzero(fit_mask) >= max_term, (  # noqa: PLR2004
        "Not enough valid points inside the fit aperture to fit the requested "
        "number of Zernike terms!"
    )

    basis: NDArray = zernike_basis(max_term, rho[fit_mask], theta[fit_mask])
    design_matrix: NDArray = basis.T  # shape (n_points, max_term)
    zv: NDArray = z[fit_mask]

    coefficients, *_ = np.linalg.lstsq(design_matrix, zv, rcond=None)

    terms: list[int] = list(range(1, max_term + 1))
    n_m: list[tuple[int, int]] = [noll_to_n_m(t) for t in terms]
    names: list[str] = [
        zernike_name(t, n=n, m=m) for t, (n, m) in zip(terms, n_m, strict=True)
    ]

    full_basis: NDArray = zernike_basis(max_term, rho, theta)
    fitted: NDArray = np.tensordot(coefficients, full_basis, axes=(0, 0))
    fitted[~domain_mask] = np.nan

    residual: NDArray = z - fitted

    return ZernikeFitResult(
        coefficients=coefficients,
        terms=terms,
        n_m=n_m,
        names=names,
        center=(x0, y0),
        radius=r,
        rho=rho,
        theta=theta,
        domain_mask=domain_mask,
        fit_mask=fit_mask,
        data=z,
        fitted=fitted,
        residual=residual,
    )
