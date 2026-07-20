from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy import units as u

from curved_air.type_hinting import ScalarQuantity

DESIGNS: list[str] = [
    "Newtonian",
    "Classical Cassegrain",
    "Ritchey-Chrétien",
    "Dall-Kirkham",
    # "Gregorian",
    "Pressmann-Camichel",
    # "Stigmatic combination",
]

CONIC_NOTES: dict[str, str] = {
    "Newtonian": "Parabolic primary (K1 = -1), flat secondary.",
    "Classical Cassegrain": "Parabolic primary (K1 = -1), hyperbolic convex secondary.",
    "Ritchey-Chrétien": "Both mirrors are hyperboloids; corrects for coma (aplanatic).",
    "Dall-Kirkham": "Elliptical primary, spherical secondary (K2 = 0). Not aplanatic.",
    # "Gregorian": "Parabolic primary (K1 = -1), elliptical concave secondary.",
    "Pressmann-Camichel": "Spherical primary (K1 = 0), hyperbolic convex secondary.",
    # "Stigmatic combination": "Conic constants chosen to eliminate both coma and astigmatism for a given field angle. Requires solving for both K1 and K2.",
}


@dataclass
class Mirror:
    diameter: ScalarQuantity | None = None
    radius: ScalarQuantity | None = None
    conic: float | None = None

    @property
    def focal_length(self) -> ScalarQuantity | None:
        if self.radius is not None:
            return self.radius / 2
        return None

    @property
    def f_number(self) -> float | None:
        if self.focal_length is not None:
            return (self.focal_length / self.diameter).si.value
        return None


@dataclass
class TelescopeDesign:
    name: str
    design: str
    M1: Mirror | None = None
    M2: Mirror | None = None
    system_focal_length: ScalarQuantity | None = None
    back_focal_distance: ScalarQuantity | None = None
    primary_secondary_separation: ScalarQuantity | None = None
    secondary_magnification: float | None = None
    maximum_field: ScalarQuantity | None = None

    extension_factor: float | None = None
    CT: float | None = None
    secondary_prime_focus_distance: ScalarQuantity | None = None
    secondary_system_focus_distance: ScalarQuantity | None = None

    def __post_init__(self) -> None:
        if self.design not in DESIGNS:
            raise ValueError(
                f"Invalid design: {self.design}. Must be one of {DESIGNS}."
            )

        assert isinstance(self.M1.diameter, ScalarQuantity), (
            "Primary diameter must be specified."
        )
        assert isinstance(self.M1.radius, ScalarQuantity), (
            "Primary radius must be specified."
        )

        assert self.system_focal_length is not None, (
            "System focal length must be specified."
        )

        assert self.back_focal_distance is not None, (
            "Back focal distance must be specified."
        )

        assert self.system_focal_length >= self.M1.focal_length, (
            "System focal length must be greater than primary focal length."
        )

        # if self.M1.focal_length is not None:
        #     self.M1.radius: ScalarQuantity = 2 * self.M1.focal_length

        self.secondary_magnification: float = (
            self.system_focal_length / self.M1.focal_length
        ).si.value

        self.compute_prescription()

    @property
    def system_f_number(self) -> float:
        assert self.system_focal_length is not None and self.M1.diameter is not None
        return (self.system_focal_length / self.M1.diameter).si.value

    @property
    def plate_scale(self) -> ScalarQuantity:
        assert self.system_focal_length is not None
        return (u.radian / self.system_focal_length).to(u.arcsec / u.mm)

    @property
    def angular_resolution(self) -> ScalarQuantity:
        assert self.system_focal_length is not None
        return (1.22 * u.arcsec * u.m / self.M1.diameter).to(u.arcsec)

    # @property
    # def angular_field_of_view(self) -> ScalarQuantity:
    #     assert self.maximum_field is not None
    #     return self.maximum_field

    @property
    def image_height(self) -> ScalarQuantity:
        assert self.system_focal_length is not None
        assert self.maximum_field is not None
        return (self.system_focal_length * np.tan(self.maximum_field / 2)).to(u.mm)

    @property
    def central_obstruction(self) -> float | None:

        if self.M2.diameter is None:
            self.compute_M2_diameter()

        if self.M2.diameter is not None:
            return (self.M2.diameter / self.M1.diameter).si.value
        raise ValueError(
            "Secondary diameter must be specified to compute central obstruction."
        )
        return 0

    def compute_M2_diameter(self, force_recompute: bool = False) -> None:

        assert self.M2 is not None

        if self.M2.diameter is not None and not force_recompute:
            return

        if (
            self.system_focal_length is None
            or self.primary_secondary_separation is None
        ):
            raise ValueError(
                "System focal length and primary-secondary separation must be specified to compute M2 diameter."
            )

        # secondary_half_height: ScalarQuantity = (
        #     self.system_focal_length / self.maximum_field
        # ) / 2

        assert self.secondary_prime_focus_distance is not None
        axial_secondary_diameter: ScalarQuantity = (
            self.secondary_prime_focus_distance
            * self.M1.diameter
            / self.M1.focal_length
        )

        self.M2.diameter: ScalarQuantity = 2 * axial_secondary_diameter

        # self.M2.diameter: ScalarQuantity = (
        #     self.system_f_number * self.secondary_system_focus_distance

        # assert self.system_focal_length is not None
        # assert self.primary_secondary_separation is not None
        # self.M2.diameter: ScalarQuantity = (
        #     self.system_focal_length - self.primary_secondary_separation
        # ) / self.system_f_number

    def compute_parameters(self) -> TelescopeDesign:
        if (
            self.M1.diameter == 0
            or self.M1.focal_length == 0
            or self.maximum_field == 0
        ):
            raise ValueError(
                "You must enter primary mirror diameter, primary mirror focal length, and maximum field."
            )

        if (
            self.design == "Stigmatic combination"
            and self.M1.conic is None
            and self.M2.conic is None
        ):
            raise ValueError("Please enter one - Primary or Secondary conic")

        if self.M2.radius is None:
            if self.system_focal_length is None or self.back_focal_distance is None:
                raise ValueError("Not enough parameters!")

            assert self.system_focal_length is not None
            assert self.back_focal_distance is not None

            # Extension factor
            self.extension_factor: float = (
                self.system_focal_length / self.M1.focal_length
            ).si.value

            p: ScalarQuantity = (self.M1.focal_length + self.back_focal_distance) / (
                self.extension_factor + 1
            )

            v: ScalarQuantity = p * self.extension_factor

            self.primary_secondary_separation: ScalarQuantity = self.M1.focal_length - p

            self.M2.radius: ScalarQuantity = v * 2 / (self.extension_factor - 1)

            self.M1.radius: ScalarQuantity = self.M1.focal_length * 2  # ty:ignore[unsupported-operator]

            self.CT: ScalarQuantity = (
                (self.extension_factor + 1.0) / (self.extension_factor - 1.0)
            ) ** 2

        elif self.M2.conic is None:
            if self.back_focal_distance is None:
                raise ValueError("Not enough parameters!")

            assert self.back_focal_distance is not None

            pp: ScalarQuantity = self.M1.focal_length + 1 * self.back_focal_distance

            p: ScalarQuantity = (
                pp + self.M2.radius - np.sqrt(pp * pp + self.M2.radius**2)
            ) / 2

            v: ScalarQuantity = pp - p

            # Extension factor
            self.extension_factor: float = (v / p).si.value

            self.system_focal_length: ScalarQuantity = (
                self.M1.focal_length * self.extension_factor
            )

            self.primary_secondary_separation: ScalarQuantity = (
                self.M1.focal_length - 1 * p
            )

            assert self.M1.focal_length is not None
            self.M1.radius: ScalarQuantity = self.M1.focal_length * 2

            self.secondary_system_focus_distance: ScalarQuantity = v

            if self.back_focal_distance is None:
                self.back_focal_distance: ScalarQuantity = (
                    v - self.primary_secondary_separation
                )

            self.CT: ScalarQuantity = (
                (self.extension_factor + 1) / (self.extension_factor - 1)
            ) ** 2

        else:
            if self.M2.conic <= 0:
                raise ValueError(
                    "Secondary mirror conic constant must be positive for this mode."
                )

            # Extension factor
            self.extension_factor: float = (np.sqrt(self.M2.conic) + 1) / (
                np.sqrt(self.M2.conic) - 1
            )

            v: ScalarQuantity = self.M2.radius * (self.extension_factor - 1) / 2

            p: ScalarQuantity = v / self.extension_factor

            self.system_focal_length: ScalarQuantity = (
                self.M1.focal_length * self.extension_factor
            )

            self.primary_secondary_separation: ScalarQuantity = self.M1.focal_length - p

            self.back_focal_distance: ScalarQuantity = (
                v - self.primary_secondary_separation
            )

            self.M1.radius: ScalarQuantity = self.M1.focal_length * 2  # ty:ignore[unsupported-operator]

            # Units here?
            self.CT = ((self.extension_factor + 1) / (self.extension_factor - 1)) ** 2

        assert self.M2.radius is not None

        self.secondary_prime_focus_distance: ScalarQuantity = p

        self.secondary_system_focus_distance: ScalarQuantity = v

        if self.back_focal_distance is None:
            self.back_focal_distance: ScalarQuantity = (
                v - self.primary_secondary_separation
            )

        return self

    def compute_conics(self) -> TelescopeDesign:

        assert self.M1 is not None
        assert self.M1.radius is not None
        assert self.M2 is not None
        assert self.M2.radius is not None

        assert self.CT is not None
        assert self.secondary_prime_focus_distance is not None
        assert self.primary_secondary_separation is not None
        assert self.extension_factor is not None

        match self.design:
            case "Classical Cassegrain":
                self.M1.conic = -1
                self.M2.conic: float = -self.CT

            case "Ritchey-Chrétien":
                self.M1.conic: float = -(
                    self.secondary_prime_focus_distance
                    * 2
                    / (self.primary_secondary_separation * self.extension_factor**2)
                    + 1
                ).si
                self.M2.conic: float = -(
                    self.extension_factor
                    * self.M1.radius
                    / (
                        self.primary_secondary_separation
                        * ((self.extension_factor - 1) ** 3)
                    )
                    + self.CT
                ).si

            case "Dall-Kirkham":
                self.M1.conic: float = -(
                    1
                    - self.M2.radius
                    / self.M1.radius
                    * ((self.extension_factor**2 - 1) ** 2)
                    / (self.extension_factor**4)
                ).si
                self.M2.conic = 0

            case "Pressmann-Camichel":
                self.M1.conic = 0
                self.M2.conic: float = (
                    self.M1.radius
                    / self.M2.radius
                    * (self.extension_factor / (self.extension_factor - 1)) ** 4
                    - self.CT
                ).si

            case "Gregorian":
                ...

            case "Stigmatic combination":
                if self.M1.conic is None:
                    assert self.M2.conic is not None
                    self.M1.conic: float = (
                        self.M2.radius
                        / self.M1.radius
                        * (((self.extension_factor - 1) / self.extension_factor) ** 4)
                        * (self.CT + self.M2.conic)
                        - 1
                    ).si
                else:
                    self.M2.conic: float = (
                        (1 + self.M1.conic)
                        * self.M1.radius
                        / (self.M1.radius - 2 * self.primary_secondary_separation)
                        * (self.extension_factor / (self.extension_factor - 1)) ** 3
                        - self.CT
                    ).si

            case _:
                raise ValueError(f"Unsupported design: {self.design}")
                return self

        return self

    def compute_prescription(self) -> TelescopeDesign:
        """
        Compute the telescope prescription depending on design type
        """

        if self.design == "Newtonian":
            assert self.M1 is not None
            assert self.M2 is not None
            assert self.system_focal_length is not None

            self.extension_factor: float = 1

            self.M1.conic = -1

            self.M2.radius: ScalarQuantity = np.inf * self.M1.radius.unit
            self.M2.conic = 0

            assert self.back_focal_distance is not None

            self.primary_secondary_separation: ScalarQuantity = (
                self.M1.focal_length - 1 * self.back_focal_distance
            )

            self.M2.diameter = (
                self.system_focal_length - self.primary_secondary_separation
            ) / self.system_f_number

            return self

        self.compute_parameters()
        self.compute_conics()

        return self

    @property
    def coma_coefficient(self) -> float:
        # Coma = ef * ef / 2.0 * (1.0 / (ef * ef) - (1.0 + C1) * l / (l * 2.0 - R1));

        assert self.extension_factor is not None
        assert self.M1.conic is not None
        assert self.primary_secondary_separation is not None

        return (
            self.extension_factor**2
            / 2
            * (
                (1 / self.extension_factor**2)
                - (1 + self.M1.conic)
                * self.primary_secondary_separation
                / (self.primary_secondary_separation * 2 - self.M1.radius)
            )
        )

    @property
    def coma(self) -> ScalarQuantity:
        # UH = FH / 2.0;
        # ComaArc = Coma * UH * fst * fst * 77349.0;

        assert self.maximum_field is not None

        return (
            self.coma_coefficient
            * (self.maximum_field.to(u.rad) / 2)
            * (1 / self.system_f_number) ** 2
            * (3 / 8)
        ).to(u.arcsecond)

    @property
    def astigmatism_coefficient(self) -> float:
        # Ast = ef * (
        #   (l * 2.0 - ef * R1) / (ef * ef * (l * 2.0 - R1)) - Math.pow((l / (l * 2.0 - R1)), 2) * (C1 + 1.0));

        assert self.extension_factor is not None
        assert self.primary_secondary_separation is not None
        assert self.M1.radius is not None
        assert self.M1.conic is not None

        return self.extension_factor * (
            (
                self.primary_secondary_separation * 2
                - self.extension_factor * self.M1.radius
            )
            / (
                self.extension_factor**2
                * (self.primary_secondary_separation * 2 - self.M1.radius)
            )
            - (
                self.primary_secondary_separation
                / (self.primary_secondary_separation * 2 - self.M1.radius)
            )
            ** 2
            * (self.M1.conic + 1)
        )

    @property
    def astigmatism(self) -> ScalarQuantity:
        # UH = FH / 2.0;
        # AstA = Ast * UH * UH * fst * 103132.0;

        return (
            self.astigmatism_coefficient
            * (self.maximum_field.to(u.rad) / 2) ** 2
            * (1 / self.system_f_number)
            / (2 * u.rad)
        ).to(u.arcsecond)

    @property
    def petzval_curvature(self) -> float:
        assert self.extension_factor is not None
        assert self.M1.radius is not None
        assert self.primary_secondary_separation is not None

        return (
            self.extension_factor
            * (
                1
                + self.M1.radius
                * (self.extension_factor - 1)
                / (
                    self.extension_factor
                    * (self.primary_secondary_separation * 2 - self.M1.radius)
                )
            )
        ).si

    @property
    def field_curvature(self) -> float:
        return self.petzval_curvature - 2 * self.astigmatism_coefficient

    @property
    def field_curvature_radius(self) -> ScalarQuantity:
        assert self.system_focal_length is not None

        return self.system_focal_length / self.field_curvature

    @property
    def primary_edge_exponent(self) -> float:
        assert self.M1.conic is not None
        assert self.M1.diameter is not None
        assert self.M1.radius is not None

        return self.M1.conic / 32 * (self.M1.diameter / 2) ** 4 / self.M1.radius**3

    @property
    def secondary_edge_exponent(self) -> float:
        assert self.M2.conic is not None
        assert self.M2.diameter is not None
        assert self.M2.radius is not None

        return self.M2.conic / 32 * (self.M2.diameter / 2) ** 4 / self.M2.radius**3

    @property
    def primary_edge_slope(self) -> float:
        assert self.M1.conic is not None
        assert self.M1.diameter is not None
        assert self.M1.radius is not None

        return self.M1.conic * (self.M1.diameter / 2) ** 3 / (self.M1.radius**3 * 4)

    @property
    def secondary_edge_slope(self) -> float:
        assert self.M2.conic is not None
        assert self.M2.diameter is not None
        assert self.M2.radius is not None

        return self.M2.conic / 4 * (self.M2.diameter / 2) ** 3 / self.M2.radius**3

    @property
    def prescription_summary(self) -> str:
        return f"""{self.design} telescope
        D={self.M1.diameter:.1f}
        F={self.system_focal_length:.1f}
        f/{self.system_f_number:.1f}
        central obstruction {self.central_obstruction:.2%}
        M1: R={self.M1.radius:.1f} K={self.M1.conic:.3f}
        M2: R={self.M2.radius:.1f} K={self.M2.conic:.3f}
        """
