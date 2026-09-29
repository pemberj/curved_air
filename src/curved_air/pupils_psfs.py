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

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from astropy import units as u
from numpy.typing import NDArray

if TYPE_CHECKING:
    from astropy.units import Quantity

    from curved_air.type_hinting import ArrayQuantity, ScalarQuantity

type FloatArray = NDArray[np.floating]
type ComplexArray = NDArray[np.complexfloating]
type BoolArray = NDArray[np.bool_]


# -- Shapes ----------------------------------------------------------------------------


class Shape(ABC):
    """Base class describing a generic geometrical region"""

    @abstractmethod
    def contains(self, x: ArrayQuantity, y: ArrayQuantity) -> BoolArray:
        """Return True where (x, y) lies inside the shape"""
        ...

    def __sub__(self, other: Shape) -> Shape:
        return Difference(self, other)

    def __or__(self, other: Shape) -> Shape:
        return Union(self, other)

    def __and__(self, other: Shape) -> Shape:
        return Intersection(self, other)

    def __invert__(self) -> Shape:
        return Complement(self)

    def translate(
        self,
        dx: ScalarQuantity = 0 * u.mm,
        dy: ScalarQuantity = 0 * u.mm,
    ) -> Shape:
        return TransformedShape(self, translation=(-1 * dx, -1 * dy))

    def rotate(self, angle: ScalarQuantity) -> Shape:
        return TransformedShape(self, rotation=angle)


@dataclass
class Circle(Shape):
    diameter: ScalarQuantity

    def __post_init__(self) -> None:
        if self.diameter.ndim != 0:
            raise ValueError("Radius must be scalar")
        if self.diameter <= 0 * self.diameter.unit:
            self.diameter = abs(self.diameter)

    @property
    def radius(self) -> ScalarQuantity:
        return self.diameter / 2

    def contains(self, x: ArrayQuantity, y: ArrayQuantity) -> BoolArray:
        # Get everyhing into the same units
        x: ArrayQuantity = x.to(self.diameter.unit)
        y: ArrayQuantity = y.to(self.diameter.unit)

        return np.asarray(x**2 + y**2 <= self.radius**2)


@dataclass
class Ellipse(Shape):
    semi_major: ScalarQuantity
    semi_minor: ScalarQuantity

    def __post_init__(self) -> None:
        if self.semi_major.ndim != 0 or self.semi_minor.ndim != 0:
            raise ValueError("Ellipse dimensions must be scalar")
        if self.semi_major <= 0 * self.semi_major.unit:
            self.semi_major = abs(self.semi_major)
        if self.semi_minor <= 0 * self.semi_minor.unit:
            self.semi_minor = abs(self.semi_minor)

    def contains(self, x: ScalarQuantity, y: ScalarQuantity) -> BoolArray:
        # Get everyhing into the same units
        x: ArrayQuantity = x.to(self.semi_major.unit)
        y: ArrayQuantity = y.to(self.semi_major.unit)
        a: ScalarQuantity = self.semi_major
        b: ScalarQuantity = self.semi_minor.to(self.semi_major.unit)

        return np.asarray((x / a) ** 2 + (y / b) ** 2 <= 1)


@dataclass
class Rectangle(Shape):
    width: ScalarQuantity
    height: ScalarQuantity

    def __post_init__(self) -> None:
        if self.width.ndim != 0 or self.height.ndim != 0:
            raise ValueError("Rectangle dimensions must be scalar")
        if self.width <= 0 * self.width.unit:
            self.width = abs(self.width)
        if self.height <= 0 * self.height.unit:
            self.height = abs(self.height)

    def contains(self, x: ScalarQuantity, y: ScalarQuantity) -> BoolArray:
        # Get everyhing into the same units
        x: ArrayQuantity = x.to(self.width.unit)
        y: ArrayQuantity = y.to(self.width.unit)
        height = self.height.to(x.unit)

        return np.asarray((np.abs(x) <= self.width / 2) & (np.abs(y) <= height / 2))


@dataclass
class Polygon(Shape):
    """Simple polygon defined by vertices in its local coordinate system"""

    vertices: tuple[tuple[Quantity, Quantity], ...]

    def __post_init__(self) -> None:
        if len(self.vertices) < 3:  # noqa: PLR2004
            raise ValueError("A Polygon requires at least three vertices")

    def contains(self, x: Quantity, y: Quantity) -> BoolArray:
        unit: u.Unit = self.vertices[0][0].unit
        xv: FloatArray = np.asarray([vx.to(unit).value for vx, _ in self.vertices])
        yv: FloatArray = np.asarray([vy.to(unit).value for _, vy in self.vertices])

        xx: ArrayQuantity = x.to_value(unit)
        yy: ArrayQuantity = y.to_value(unit)

        # Ray-casting algorithm.
        inside: BoolArray = np.zeros(np.broadcast(xx, yy).shape, dtype=bool)
        for i in range(len(xv)):
            j: int = (i - 1) % len(xv)
            intersects = ((yv[i] > yy) != (yv[j] > yy)) & (
                xx < (xv[j] - xv[i]) * (yy - yv[i]) / (yv[j] - yv[i]) + xv[i]
            )
            # TODO: understand this
            inside ^= intersects

        return inside


@dataclass
class RegularPolygon(Polygon):
    """Regular polygon defined by number of sides and circumradius"""

    num_sides: int
    circumradius: ScalarQuantity
    vertices: tuple[tuple[Quantity, Quantity], ...] = field(init=False)

    def __post_init__(self) -> None:
        if self.num_sides < 3:  # noqa: PLR2004
            raise ValueError("A RegularPolygon requires at least three sides")
        if self.circumradius.ndim != 0:
            raise ValueError("Circumradius must be scalar")
        if self.circumradius <= 0 * self.circumradius.unit:
            self.circumradius = abs(self.circumradius)

        # Generate vertices
        angles: FloatArray = (
            np.linspace(0, 2 * np.pi, self.num_sides, endpoint=False) + np.pi / 2
        )
        vertices: list[tuple[Quantity, Quantity]] = [
            (
                self.circumradius * np.cos(angle),
                self.circumradius * np.sin(angle),
            )
            for angle in angles
        ]
        object.__setattr__(self, "vertices", tuple(vertices))


# -- Boolean shape composition ---------------------------------------------------------

# All of these subclasses represent the result of a boolean operation on two shapes, for
# instance adding two shapes together to form a more complex geometry


@dataclass(frozen=True)
class Union(Shape):
    left: Shape
    right: Shape

    def contains(self, x: ArrayQuantity, y: ArrayQuantity) -> BoolArray:
        return self.left.contains(x, y) | self.right.contains(x, y)


@dataclass(frozen=True)
class Intersection(Shape):
    left: Shape
    right: Shape

    def contains(self, x: ArrayQuantity, y: ArrayQuantity) -> BoolArray:
        return self.left.contains(x, y) & self.right.contains(x, y)


@dataclass(frozen=True)
class Difference(Shape):
    left: Shape
    right: Shape

    def contains(self, x: ArrayQuantity, y: ArrayQuantity) -> BoolArray:
        return self.left.contains(x, y) & ~self.right.contains(x, y)


@dataclass(frozen=True)
class Complement(Shape):
    # Maybe rename to Inverse?
    shape: Shape

    def contains(self, x: ArrayQuantity, y: ArrayQuantity) -> BoolArray:
        return ~self.shape.contains(x, y)


# -- Geometrical transformations -------------------------------------------------------


@dataclass
class TransformedShape(Shape):
    """
    This class represents the result of transforming an existing shape with either a
    translation or a rotation.

    TODO: Add mirroring and scaling transformations? Perhaps each should be its own
    class?
    """

    shape: Shape
    translation: tuple[ScalarQuantity, ScalarQuantity] = (0 * u.m, 0 * u.m)
    rotation: ScalarQuantity = 0 * u.deg

    def contains(self, x: ArrayQuantity, y: ArrayQuantity) -> BoolArray:
        # Transform world coordinates into the shape's local coordinate system.
        dx: ScalarQuantity = self.translation[0].to(x.unit)
        dy: ScalarQuantity = self.translation[1].to(y.unit)

        x_local: ScalarQuantity = x - dx
        y_local: ScalarQuantity = y - dy

        theta: ScalarQuantity = self.rotation.to_value(u.rad)
        cos_t: float = np.cos(theta)
        sin_t: float = np.sin(theta)

        # Inverse rotation.
        xr: ScalarQuantity = cos_t * x_local + sin_t * y_local
        yr: ScalarQuantity = -sin_t * x_local + cos_t * y_local

        return self.shape.contains(xr, yr)


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
        """Physical pupil-plane sampling interval."""
        return self.extent / self.npix

    @property
    def x(self) -> ArrayQuantity:
        """1-D x coordinates of pixel centres."""
        dx: ScalarQuantity = self.pixel_size
        c: ScalarQuantity = self.center[0].to(dx.unit)
        return (np.arange(self.npix) - (self.npix - 1) / 2) * dx + c

    @property
    def y(self) -> ArrayQuantity:
        """1-D y coordinates of pixel centres."""
        dx: ScalarQuantity = self.pixel_size
        c: ScalarQuantity = self.center[1].to(dx.unit)
        return (np.arange(self.npix) - (self.npix - 1) / 2) * dx + c

    def coordinates(self) -> tuple[ArrayQuantity, ArrayQuantity]:
        """Return 2-D meshgrid coordinates."""
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
    A description of an optical pupil, which includes some geometry formed by an
    arbitrary collection of Shapes, an (optional) optical path difference (OPD) model,
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
        """ """

        x, y = grid.coordinates()
        aperture: BoolArray = self.geometry.contains(x, y)

        amplitude: FloatArray = np.zeros(aperture.shape, dtype=float)

        if self.amplitude is None:
            amplitude[aperture] = 1.0
        else:
            value: FloatArray = np.asarray(self.amplitude(x, y), dtype=float)  # ty: ignore[call-non-callable]
            amplitude: FloatArray = np.where(aperture, value, 0.0)

        phase: FloatArray = np.zeros(aperture.shape, dtype=float)

        if self.opd is not None:
            if wavelength is None:
                raise ValueError("Wavelength is required when an OPD model is present")
            opd: ArrayQuantity = self.opd(x, y)  # ty: ignore
            phase: FloatArray = np.asarray(
                2 * np.pi * opd / wavelength,
                dtype=float,
            )
            phase: FloatArray = np.where(aperture, phase, 0.0)

        field: ComplexArray = amplitude * np.exp(1j * phase)

        return SampledPupil(field=field, grid=grid)

    def mask(self, grid: PupilGrid) -> BoolArray:
        """Return the binary geometrical pupil mask."""

        x, y = grid.coordinates()

        return self.geometry.contains(x, y)


@dataclass(frozen=True)
class SampledPupil:
    """Complex pupil field sampled on a Cartesian grid"""

    field: ComplexArray
    grid: PupilGrid

    @property
    def amplitude(self) -> FloatArray:
        return np.abs(self.field)

    @property
    def phase(self) -> FloatArray:
        return np.angle(self.field)

    def fft(self, *, shift: bool = True) -> ComplexArray:
        """Return the unscaled Fourier transform of the pupil field."""
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
        """Monochromatic Fraunhofer PSF"""

        # Angular sampling
        #     dtheta = wavelength / grid.extent

        # Total angular field of view
        #     N * dtheta = wavelength / grid.pixel_size.

        if wavelength.ndim != 0:
            raise ValueError("Wavelength must be scalar")

        image = self.fft(shift=shift)

        intensity: FloatArray = np.abs(image) ** 2

        if normalize:
            peak: float = intensity.max()

            if peak > 0:
                intensity: FloatArray = intensity / peak

        # FFT spatial frequency is cycles / length.
        # For Fraunhofer diffraction:
        #     theta ~= lambda * f
        # where f is spatial frequency in cycles / length.

        fx, fy = self.grid.spatial_frequency()

        theta_x: ArrayQuantity = (wavelength * fx).si * u.rad
        theta_y: ArrayQuantity = (wavelength * fy).si * u.rad

        return PSF(
            intensity=intensity,
            x=theta_x,
            y=theta_y,
        )


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

    @property
    def pixel_scale(self) -> ScalarQuantity:
        """Angular separation between adjacent PSF samples"""
        return abs(self.x[1] - self.x[0])


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


# Factor between circumradius and flat-to-flat distance for a regular hexagon
HEX_FACTOR: float = 1.15470
JWST_MIRROR_SIZE: ScalarQuantity = 1.32 * u.m

JWST_UNIT_MIRROR: Shape = RegularPolygon(
    num_sides=6, circumradius=(JWST_MIRROR_SIZE * HEX_FACTOR) / 2
)

X_SPACING: ScalarQuantity = 1.35 * u.m
Y_SPACING: ScalarQuantity = 0.86603 * X_SPACING

JWST: Shape = (
    JWST_UNIT_MIRROR.translate(dx=X_SPACING)
    | JWST_UNIT_MIRROR.translate(dx=X_SPACING * 2)
    | JWST_UNIT_MIRROR.translate(dx=-X_SPACING)
    | JWST_UNIT_MIRROR.translate(dx=-X_SPACING * 2)
    #
    | JWST_UNIT_MIRROR.translate(dx=X_SPACING / 2, dy=Y_SPACING)
    | JWST_UNIT_MIRROR.translate(dx=3 * X_SPACING / 2, dy=Y_SPACING)
    | JWST_UNIT_MIRROR.translate(dx=-X_SPACING / 2, dy=Y_SPACING)
    | JWST_UNIT_MIRROR.translate(dx=-3 * X_SPACING / 2, dy=Y_SPACING)
    #
    | JWST_UNIT_MIRROR.translate(dx=X_SPACING / 2, dy=-Y_SPACING)
    | JWST_UNIT_MIRROR.translate(dx=3 * X_SPACING / 2, dy=-Y_SPACING)
    | JWST_UNIT_MIRROR.translate(dx=-X_SPACING / 2, dy=-Y_SPACING)
    | JWST_UNIT_MIRROR.translate(dx=-3 * X_SPACING / 2, dy=-Y_SPACING)
    #
    | JWST_UNIT_MIRROR.translate(dy=2 * Y_SPACING)
    | JWST_UNIT_MIRROR.translate(dx=X_SPACING, dy=2 * Y_SPACING)
    | JWST_UNIT_MIRROR.translate(dx=-X_SPACING, dy=2 * Y_SPACING)
    #
    | JWST_UNIT_MIRROR.translate(dy=-2 * Y_SPACING)
    | JWST_UNIT_MIRROR.translate(dx=X_SPACING, dy=-2 * Y_SPACING)
    | JWST_UNIT_MIRROR.translate(dx=-X_SPACING, dy=-2 * Y_SPACING)
).rotate(30 * u.deg) - (
    Rectangle(width=80 * u.mm, height=4 * u.m).translate(dy=2 * u.m)
    | Rectangle(width=80 * u.mm, height=4 * u.m)
    .translate(dy=-2 * u.m)
    .rotate(30 * u.deg)
    | Rectangle(width=80 * u.mm, height=4 * u.m)
    .translate(dy=-2 * u.m)
    .rotate(-30 * u.deg)
)


def test() -> None:

    from matplotlib import pyplot as plt  # noqa: PLC0415

    pupil = Pupil(geometry=JWST)

    grid = PupilGrid(npix=1500, extent=15 * u.m)

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

    width = (3 * u.arcsec).to(u.arcsec).value
    ax.set_xlim(-width / 2, width / 2)
    ax.set_ylim(-width / 2, width / 2)

    plt.show()


if __name__ == "__main__":
    test()
