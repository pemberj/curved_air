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


class Shape(ABC):
    """
    Base class describing a generic geometrical region
    """

    @abstractmethod
    def contains(self, x: ArrayQuantity, y: ArrayQuantity) -> BoolArray:
        """
        Returns True if (x, y) lies inside the Shape, which may be complex and comprised
        of many simple Shapes with boolean operations between them. x and y are in the
        Shape's local coordinate system, and may represent a single point or an array of
        many points. The return value is a boolean array of the same shape as the input
        coordinates, indicating whether each point is inside the Shape.
        """
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

    def scale(
        self,
        sx: float = 1.0,
        sy: float = 1.0,
    ) -> Shape:
        return TransformedShape(self, scale=(1 / sx, 1 / sy))

    def mirror(self, mirror_x: bool = False, mirror_y: bool = False) -> Shape:
        return TransformedShape(self, mirror_x=mirror_x, mirror_y=mirror_y)


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
    """
    This class can be used to create polygons that are irregular, or where the vertices
    of the shape are known explicitly.
    """

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
            # Bitwise exclusive 'or' operator (XOR)
            inside ^= intersects

        return inside


@dataclass
class RegularPolygon(Polygon):
    """ """

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


"""
Boolean shape compositions

The following subclasses represent the result of a boolean operation on two shapes, for
instance adding two shapes together, taking their difference, or only the intersection
between them. With simple shapes and these basic operations you can form much more
complex geometrical shapes.
"""


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


@dataclass
class TransformedShape(Shape):
    """
    This class represents the result of transforming an existing Shape with either a
    translation or a rotation.

    This class is typically never called directly, but is rather returned when a
    transformation method is called on an existing Shape.

    Care should certainly be taken when applying multiple transformations at once!
    """

    shape: Shape
    translation: tuple[ScalarQuantity, ScalarQuantity] = (0 * u.m, 0 * u.m)
    rotation: ScalarQuantity = 0 * u.deg
    mirror_x: bool = False
    mirror_y: bool = False
    scale: tuple[float, float] = (
        1 * u.dimensionless_unscaled,
        1 * u.dimensionless_unscaled,
    )

    def contains(self, x: ArrayQuantity, y: ArrayQuantity) -> BoolArray:
        # Transform world coordinates into the shape's local coordinate system
        dx: ScalarQuantity = self.translation[0].to(x.unit)
        dy: ScalarQuantity = self.translation[1].to(y.unit)

        x_local: ScalarQuantity = x - dx
        y_local: ScalarQuantity = y - dy

        theta: ScalarQuantity = self.rotation.to_value(u.rad)
        cos_t: float = np.cos(theta)
        sin_t: float = np.sin(theta)

        # Rotation
        xr: ScalarQuantity = cos_t * x_local + sin_t * y_local
        yr: ScalarQuantity = -sin_t * x_local + cos_t * y_local

        # Mirroring
        if self.mirror_x:
            xr = -xr
        if self.mirror_y:
            yr = -yr

        # Scaling
        xr = xr * self.scale[0]
        yr = yr * self.scale[1]

        return self.shape.contains(xr, yr)
