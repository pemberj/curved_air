from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from astropy import units as u
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from astropy.units import Quantity


@dataclass
class Etalon:
    """Description of plane wave illumination of an etalon a la Wikipedia"""

    n: float = 1
    thickness: Quantity = 6 * u.mm
    theta: Quantity = 0 * u.degree
    R_1: float = 0.95

    R_2: float | None = None

    def __post_init__(self) -> None:
        if self.R_2 is None:
            self.R_2 = self.R_1

        self.calculate_properties()

    @cached_property
    def R(self) -> float:
        return np.sqrt(self.R_1 * self.R_2)

    @cached_property
    def T(self) -> float:
        return 1 - self.R

    @cached_property
    def F_coef(self) -> float:
        return (4 * self.R) / ((1 - self.R) ** 2)

    @cached_property
    def finesse(self) -> float:
        return np.pi / (2 * np.arcsin(1 / np.sqrt(self.F_coef)))

    @cached_property
    def R_max(self) -> float:
        """
        Maximum reflectivity, R_max
        """
        return (4 * self.R) / ((1 + self.R) ** 2)

    def delta_lambda(self, wavelength: Quantity = 780 * u.nm) -> Quantity:
        """
        Calculate the free spectral range of the etalon, defaults to wavelength = 780nm
        """
        n_g = 1
        return (
            (wavelength**2)
            / (2 * n_g * self.thickness * np.cos(self.theta) + wavelength)
        ).to("pm")

    def delta(self, wavelength: Quantity) -> float:
        return (
            (2 * np.pi / wavelength) * 2 * self.n * self.thickness * np.cos(self.theta)
        ).si.value

    def Transmittance(self, wavelength: Quantity) -> float:
        """
        Transmittance function, sometimes T_e
        Transmitted beam intensity as a function of wavelength
        """
        return 1 / (1 + self.F_coef * (np.sin(self.delta(wavelength) / 2)) ** 2)

    def intensity(self, wavelength: Quantity) -> float:
        """Alias for Transmittance"""
        return self.Transmittance(wavelength)

    def amplitude(self, m: int, wavelength: Quantity) -> float:
        return self.T * (self.R**m) * np.exp(1j * m * self.delta(wavelength))

    def summed_amplitudes(self, wavelength: Quantity) -> float:
        """
        The geometric infinite sum of ampllitudes over m from 0 to infinity
        """
        return self.T / (1 - self.R * np.exp(1j * self.delta(wavelength)))

    def calculate_properties(self) -> None:
        """
        This function just calls each of the propert: Quantityy values in order to cache
        the calculated values
        """
        self.R  # noqa: B018
        self.T  # noqa: B018
        self.F_coef  # noqa: B018
        self.finesse  # noqa: B018
        self.R_max  # noqa: B018

        return

    def __str__(self) -> str:
        return "Etalon:\n" + "".join(
            [f"{p:>12}: {val:<6.3g}\n" for p, val in self.__dict__.items()]
        )

    def __repr__(self) -> str:
        return (
            f"Etalon(n={self.n}, l={self.thickness}, θ={self.theta}, "
            + f"R_1={self.R_1}, R_2={self.R_2})"
        )
