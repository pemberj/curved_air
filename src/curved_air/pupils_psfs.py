"""
curved air

pupils & psfs

This module contains code that can be used to define pupil geometry for the purpose of
computing angular PSFs (point spread functions).

The geometry is defined using elementary units (circles, rectangles, polygons, etc.),
and these can be composed into more complex forms using boolean operators (union,
difference, etc.). The geometric definitions exist as mathematical objects independent
of any numerical sampling.

A SampledPupil object is created from a geometrically-defined pupil, which then creates
a NumPy array of the pupil sampled on a regular grid. From this, the PSF can be computed
using a Fourier transform.

Example code defining a simple circular-pupil telescope, as well as defining the
significantly more complex pupil of JWST is included.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from astropy import units as u
from numpy.typing import NDArray

from curved_air.geometry import Circle, Rectangle, RegularPolygon, Shape

if TYPE_CHECKING:
    from astropy.units import Quantity

    from curved_air.type_hinting import ArrayQuantity, ScalarQuantity

type FloatArray = NDArray[np.floating]
type ComplexArray = NDArray[np.complexfloating]
type BoolArray = NDArray[np.bool_]


# -- PupilGrid, Pupil, SampledPupl and PSF ---------------------------------------------


@dataclass
class PupilGrid:
    """
    A square Cartesian grid for sampling a telescope pupil

    `extent` may (and maybe should) be larger than the physical pupil to provide
    zero-padding and therefore better sampling in PSF-space
    """

    npix: int
    extent: ScalarQuantity
    center: tuple[ScalarQuantity, ScalarQuantity] = (0 * u.m, 0 * u.m)

    def __post_init__(self) -> None:
        if self.npix < 2:  # noqa: PLR2004
            raise ValueError("npix must be >= 2")
        if self.extent.ndim != 0:
            raise ValueError("Extent must be scalar")
        if self.extent <= 0 * self.extent.unit:
            self.extent = abs(self.extent)

    @property
    def pixel_size(self) -> ScalarQuantity:
        """
        Physical pupil-plane sampling interval
        """
        return self.extent / self.npix

    @property
    def x(self) -> ArrayQuantity:
        """
        1d x-axis coordinates of pixel centres
        """
        dx: ScalarQuantity = self.pixel_size
        c: ScalarQuantity = self.center[0].to(dx.unit)
        return (np.arange(self.npix) - (self.npix - 1) / 2) * dx + c

    @property
    def y(self) -> ArrayQuantity:
        """
        1d y-axis coordinates of pixel centres
        """
        dx: ScalarQuantity = self.pixel_size
        c: ScalarQuantity = self.center[1].to(dx.unit)
        return (np.arange(self.npix) - (self.npix - 1) / 2) * dx + c

    def coordinates(self) -> tuple[ArrayQuantity, ArrayQuantity]:
        """
        Return 2d meshgrid coordinates
        """
        return np.meshgrid(self.x, self.y, indexing="xy")  # ty: ignore

    def spatial_frequency(
        self,
    ) -> tuple[ArrayQuantity, ArrayQuantity]:
        """
        Return centred Fourier-plane spatial frequencies for each pixel

        The returned values have units of inverse length (cycles per unit length)!
        """

        dx: ScalarQuantity = self.pixel_size

        values: FloatArray = np.fft.fftshift(
            np.fft.fftfreq(
                self.npix,
                d=dx.to(dx.unit).value,
            )
        )

        values: ArrayQuantity = values / dx.unit

        # Rotationally symetric, and always a square grid, so return the same values for
        # both axes.
        return values, values


@dataclass
class Pupil:
    """
    which is a placeholder for now, and an (also optional) complex phase amplitude model
    which may be used for optimising pupil masks or something like that, in the future
    """

    geometry: Shape
    opd: object | None = None  # Placeholder
    amplitude: object | None = None  # Placeholder

    def sample(
        self,
        grid: PupilGrid,
        wavelength: Quantity | None = None,
    ) -> SampledPupil:
        """
        Sample the pupil over an already-defined PupilGrid
        """

        x, y = grid.coordinates()
        aperture: BoolArray = self.geometry.contains(x, y)

        amplitude: FloatArray = np.zeros(aperture.shape, dtype=float)

        if self.amplitude is None:
            amplitude[aperture] = 1.0
        else:
            raise NotImplementedError("Amplitude model is not yet implemented")
            value: FloatArray = np.asarray(self.amplitude(x, y), dtype=float)
            amplitude: FloatArray = np.where(aperture, value, 0.0)

        phase: FloatArray = np.zeros(aperture.shape, dtype=float)

        if self.opd is not None:
            raise NotImplementedError("OPD model is not yet implemented")
            if wavelength is None:
                raise ValueError("Wavelength is required when an OPD model is present")
            opd: ArrayQuantity = self.opd(x, y)
            phase: FloatArray = np.asarray(
                2 * np.pi * opd / wavelength,
                dtype=float,
            )
            phase: FloatArray = np.where(aperture, phase, 0.0)

        field: ComplexArray = amplitude * np.exp(1j * phase)

        return SampledPupil(field=field, grid=grid)

    def mask(self, grid: PupilGrid) -> BoolArray:
        """
        Return the binary geometrical pupil mask
        """

        x, y = grid.coordinates()

        return self.geometry.contains(x, y)


@dataclass(frozen=True)
class SampledPupil:
    """
    Complex pupil field sampled on a Cartesian grid
    """

    field: ComplexArray
    grid: PupilGrid

    @property
    def amplitude(self) -> FloatArray:
        return np.abs(self.field)

    @property
    def phase(self) -> FloatArray:
        return np.angle(self.field)

    def fft(self, *, shift: bool = True) -> ComplexArray:
        """
        Return the unscaled Fourier transform of the pupil field
        """

        result = np.fft.fft2(self.field)

        if shift:
            result = np.fft.fftshift(result)

        return result

    def psf(
        self,
        wavelength: ScalarQuantity,
        *,
        normalize: bool = True,
        shift: bool = True,
    ) -> PSF:
        """
        Compute the monochromatic Fraunhofer PSF for the pupil for the given wavelength
        """

        if wavelength.ndim != 0:
            raise ValueError("Wavelength must be scalar")

        image = self.fft(shift=shift)

        intensity: FloatArray = np.abs(image) ** 2

        if normalize:
            peak: float = intensity.max()

            if peak > 0:
                intensity: FloatArray = intensity / peak

        # For Fraunhofer diffraction:
        #     theta ~= lambda * f
        # where f is spatial frequency in cycles / length

        fx, fy = self.grid.spatial_frequency()

        theta_x: ArrayQuantity = (wavelength * fx).si * u.rad
        theta_y: ArrayQuantity = (wavelength * fy).si * u.rad

        return PSF(intensity=intensity, x=theta_x, y=theta_y, wavelength=wavelength)


@dataclass(frozen=True)
class PSF:
    """
    A monochromatic PSF sampled in angular coordinates

    TODO: With a telescope's focal length we can compute the plate scale and then sample
    the psf in spatial coordinates - that is much nicer in some cases
    """

    intensity: FloatArray
    x: ArrayQuantity
    y: ArrayQuantity
    wavelength: ScalarQuantity

    @property
    def pixel_scale(self) -> ScalarQuantity:
        """
        Angular separation between adjacent PSF samples
        """
        return abs(self.x[1] - self.x[0])

    @property
    def angular_sampling(self) -> ScalarQuantity:
        return self.wavelength / (self.x[-1] - self.x[0])


def telescope(
    primary_diameter: ScalarQuantity = 2.2 * u.m,
    secondary_diameter: ScalarQuantity = 0.844 * u.m,
    num_spiders: int = 4,
    spiders_width: ScalarQuantity = 50 * u.mm,
    spider_angles: ArrayQuantity | None = None,
    clocking_angle: ScalarQuantity = 0 * u.deg,
) -> Shape:
    """
    A convenience function to generate a simple telescope with round primary and
    secondary mirrors, and with four spider supports

    The default parameters correspond to the MPG/ESO 2.2m telescope at La Silla, Chile
    """

    if num_spiders < 0:
        raise ValueError("Number of spiders must be >= 0")

    primary = Circle(diameter=primary_diameter)
    secondary = Circle(diameter=secondary_diameter)

    running_shape: Shape = primary - secondary

    if num_spiders == 0:
        return running_shape

    if spider_angles is None:
        # Create evenly spaced spiders spanning 360°
        spider_angles: ArrayQuantity = (
            np.linspace(0, 360, num_spiders, endpoint=False) * u.deg
        )

    if spider_angles.ndim != 1:
        raise ValueError("Spider angles must be a 1-D array")

    if len(spider_angles) != num_spiders:
        raise ValueError("Number of spider angles must match number of spiders")

    spider_length: ScalarQuantity = primary_diameter / 2
    spider_offset: ScalarQuantity = spider_length / 2

    spiders: list[Shape] = [
        Rectangle(width=spiders_width, height=spider_length)
        .translate(dy=spider_offset)
        .rotate(angle)
        for angle in spider_angles
    ]

    for s in spiders:
        running_shape: Shape = running_shape - s

    return running_shape.rotate(clocking_angle)


MPGESO: Shape = telescope(
    primary_diameter=2.2 * u.m,
    secondary_diameter=0.844 * u.m,
    num_spiders=4,
)


def segmented_telescope(
    segment_size: ScalarQuantity = 1.32 * u.m,
    segment_gap_size: ScalarQuantity = 10 * u.mm,
    num_hexagon_rings: int | None = 2,
    clocking_angle: ScalarQuantity = 0 * u.deg,
    max_radius: ScalarQuantity | None = None,
) -> Shape:
    """
    A convenience function to model compound-mirror telescopes with hexagonal mirror
    segments, such as the James Webb Space Telescope (JWST), the Keck telescopes, the
    European Extremely Large Telescope (ELT), etc.
    """

    HEX_FACTOR: float = 1.15470  # Factor between circumradius and flat-to-flat distance

    if num_hexagon_rings is None:
        assert max_radius is not None, (
            "max_radius must be specified if num_hexagon_rings is None"
        )
        num_hexagon_rings: int = int(np.ceil((2 * max_radius) / segment_size))

    UNIT_MIRROR: Shape = RegularPolygon(
        num_sides=6, circumradius=(segment_size * HEX_FACTOR) / 2
    )

    # Flat-to-flat spacing
    X_SPACING: ScalarQuantity = segment_size + 1 * segment_gap_size
    Y_SPACING: ScalarQuantity = (segment_size * HEX_FACTOR + segment_gap_size) / 2

    # Calculate primary mirror Shape based on num_hexagon_rings
    primary: Shape = UNIT_MIRROR

    # Axial coordinates (q, r)
    for q in range(-num_hexagon_rings, num_hexagon_rings + 1):
        for r in range(-num_hexagon_rings, num_hexagon_rings + 1):
            if max(abs(q), abs(r), abs(q + r)) <= num_hexagon_rings:
                x: ScalarQuantity = X_SPACING * (q + r / 2)
                y: ScalarQuantity = 1.5 * Y_SPACING * r
                if max_radius is not None and np.sqrt(x**2 + y**2) > max_radius:
                    # Skip segments outside the max_radius
                    continue
                primary |= UNIT_MIRROR.translate(dx=x, dy=y)
                # else:
                #     print(f"{q=}, {r=} not added because outside max_radius")

    telescope: Shape = (primary).rotate(clocking_angle)

    return telescope


JWST_SPIDERS: Shape = (
    Rectangle(width=80 * u.mm, height=4 * u.m).translate(dy=2 * u.m)
    | Rectangle(width=80 * u.mm, height=4 * u.m)
    .translate(dy=-2 * u.m)
    .rotate(30 * u.deg)
    | Rectangle(width=80 * u.mm, height=4 * u.m)
    .translate(dy=-2 * u.m)
    .rotate(-30 * u.deg)
)

JWST_SECONDARY: Shape = RegularPolygon(num_sides=6, circumradius=0.7621 * u.m)

JWST: Shape = (
    segmented_telescope(
        segment_size=1.32 * u.m,
        segment_gap_size=40 * u.mm,
        num_hexagon_rings=2,
    )
    - JWST_SECONDARY
).rotate(30 * u.deg) - JWST_SPIDERS

ELT: Shape = (
    segmented_telescope(
        segment_size=1.45 * u.m,
        max_radius=19.5 * u.m,
        segment_gap_size=40 * u.mm,
        num_hexagon_rings=None,  # The function calculates what is needed here
    )
    - Circle(diameter=4.25 * u.m)
    - (
        Rectangle(width=0.5 * u.m, height=45 * u.m)
        | Rectangle(width=0.5 * u.m, height=45 * u.m).rotate(60 * u.deg)
        | Rectangle(width=0.5 * u.m, height=45 * u.m).rotate(120 * u.deg)
    )
)


def test() -> None:

    from matplotlib import pyplot as plt  # noqa: PLC0415

    pupil = Pupil(geometry=JWST)

    grid = PupilGrid(npix=1600, extent=25 * u.m)

    sampled: SampledPupil = pupil.sample(grid)

    _fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    ax: plt.Axes = axs[0]

    ax.imshow(
        sampled.amplitude,
        cmap="gray",
        extent=[
            grid.x[0].to(u.m).value,
            grid.x[-1].to(u.m).value,
            grid.y[0].to(u.m).value,
            grid.y[-1].to(u.m).value,
        ],
    )

    ax.set_xlabel("Pupil position [m]")
    ax.set_ylabel("Pupil position [m]")

    width: ScalarQuantity = (8 * u.m).to(u.m).value
    ax.set_xlim(-width / 2, width / 2)
    ax.set_ylim(-width / 2, width / 2)

    psf: PSF = sampled.psf(wavelength=500 * u.nm)

    ax: plt.Axes = axs[1]

    ax.imshow(
        np.log10(psf.intensity),
        cmap="gray",
        extent=[
            psf.x[0].to(u.arcsec).value,
            psf.x[-1].to(u.arcsec).value,
            psf.y[0].to(u.arcsec).value,
            psf.y[-1].to(u.arcsec).value,
        ],
        vmin=-5,
        vmax=0,
    )

    ax.set_xlabel("Angular position [arcsec]")
    ax.set_ylabel("Angular position [arcsec]")

    width = (2 * u.arcsec).to(u.arcsec).value
    ax.set_xlim(-width / 2, width / 2)
    ax.set_ylim(-width / 2, width / 2)

    plt.show()


if __name__ == "__main__":
    test()
