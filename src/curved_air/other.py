"""
curved_air

other

Contains functions that model the refractive index of air, the airmass of the atmosphere
under different angles, etc.

Also contains wavelength definitions of Fraunhofer lines
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import overload

import numpy as np
import ref_index
from astropy import units as u
from astropy.units import Quantity
from numpy.typing import ArrayLike

from curved_air.typing import ArrayQuantity, ScalarQuantity

_balmer_lines: dict[str, ScalarQuantity] = {
    "H$\\alpha$": 656.279 * u.nm,
    "H$\\beta$": 486.135 * u.nm,
    "H$\\gamma$": 434.0472 * u.nm,
    "H$\\delta$": 410.1734 * u.nm,
    "H$\\epsilon$": 397.0075 * u.nm,
    "H$\\zeta$": 388.9064 * u.nm,
    "H$\\eta$": 383.5397 * u.nm,
    "(Balmer break)": 364.5 * u.nm,
}

# TODO: How to handle these separate line lists? Lots of overlap

_fraunhofer_lines: dict[str, ScalarQuantity] = {
    "t": 1013.98 * u.nm,
    "s": 852.11 * u.nm,
    "y": 898.765 * u.nm,
    "Z": 822.696 * u.nm,
    "A'": 768.2 * u.nm,
    "A": 759.370 * u.nm,
    "r": 706.52 * u.nm,
    "B": 686.719 * u.nm,
    # "C": 656.281 * u.nm, # Halpha
    "C'": 643.85 * u.nm,
    "a": 627.661 * u.nm,
    # "D1": 589.592 * u.nm,
    # "D2": 588.995 * u.nm,
    # "d$_{Na}$": 587.5618 * u.nm,
    "E2": 527.039 * u.nm,
    "b1": 518.362 * u.nm,
    "b2": 517.270 * u.nm,
    "b3": 516.891 * u.nm,
    "b4": 516.733 * u.nm,
    "c": 495.761 * u.nm,
    # "F": 486.134 * u.nm, # Hbeta
    "F'": 479.99 * u.nm,
    "d$_{Fe}$": 466.814 * u.nm,
    "e": 438.355 * u.nm,
    "g": 435.84 * u.nm,
    # "G1": 434.047 * u.nm, # Hgamma
    "G2&3": 430.78 * u.nm,
    # "h": 410.175 * u.nm, # Hdelta
    # "H": 396.847 * u.nm, # Ca-H
    # "K": 393.366 * u.nm, # Ca-K
    "L": 382.044 * u.nm,
    "i": 365.01 * u.nm,
    "N": 358.121 * u.nm,
    "P": 336.112 * u.nm,
    "T": 302.108 * u.nm,
}

_atomic_lines: dict[str, ScalarQuantity] = {
    # Balmer lines of Hydrogen
    "H$\\alpha$": 656.279 * u.nm,
    "H$\\beta$": 486.135 * u.nm,
    "H$\\gamma$": 434.0472 * u.nm,
    "H$\\delta$": 410.1734 * u.nm,
    "H$\\epsilon$": 397.0075 * u.nm,
    "H$\\zeta$": 388.9064 * u.nm,
    "H$\\eta$": 383.5397 * u.nm,
    "(Balmer break)": 364.5 * u.nm,
    # Sodium
    "Na $D_1$": 589.592 * u.nm,
    "Na $D_2$": 588.995 * u.nm,
    "NaI": 819.12515 * u.nm,
    # Silicon
    "SiIII$_1$": 455.2622 * u.nm,
    "SiIII$_2$": 456.7840 * u.nm,
    "SiIII$_3$": 457.4757 * u.nm,
    # Calcium misc
    "CaI$_{422}$": 422.67270 * u.nm,
    "CaI$_{616}$": 616.42055 * u.nm,
    # Calcium-II doublet
    "CaII-K": 393.36614 * u.nm,
    "CaII-H": 396.84673 * u.nm,
    # Calcium Triplet
    "CaII$_1$": 849.802 * u.nm,
    "CaII$_2$": 854.209 * u.nm,
    "CaII$_3$": 866.2140 * u.nm,
    # Iron
    "FeI$_1$": 438.48318 * u.nm,
    "FeI$_2$": 440.60371 * u.nm,
    "FeI$_{869}$": 869.13867 * u.nm,
    "FeH": 9940 * u.nm,
    # Helium d-line
    "He-d": 587.5618 * u.nm,
    # Manganese
    "MnI": 539.47 * u.nm,
    # Magnesium
    "MgI": 517.4141 * u.nm,
    # Rubidium
    "Rb-B$_1$": 792.9781 * u.nm,
    "Rb-B$_2$": 796.97918 * u.nm,
    # Calcium hydride
    "CaH2": 683.1862 * u.nm,
    "CaH3": 697.69239 * u.nm,
}

_IR_bands: dict[str, tuple[ScalarQuantity, ScalarQuantity]] = {
    "I": (731 * u.nm, 880 * u.nm),
    "Y": (1 * u.um, 1.15 * u.um),
    "J": (1.2 * u.um, 1.4 * u.um),
    "H": (1.5 * u.um, 1.8 * u.um),
    "K": (2 * u.um, 2.5 * u.um),
}


"""
Wrappers to functions from the `ref_index` library

These wrappers now take `astropy.units.Quantity` objects in, to allow input parameters
with any appropriate unit to be used, and no confusion about Celcius or Kevlin, etc.
It does come with the overhead of requiring `astropy` but for me and others who
regularly use `astropy` anyway, it is well worth it.

Original paper: Ciddor 1996, https://doi.org/10.1364/AO.35.001566
"""


@overload
def ciddor(
    wavelength: ScalarQuantity,
    T: ScalarQuantity,
    p: ScalarQuantity,
    h: float = 0.0,
    x_c: float = 450,
) -> float: ...


@overload
def ciddor(
    wavelength: ArrayQuantity,
    T: ScalarQuantity,
    p: ScalarQuantity,
    h: float = 0.0,
    x_c: float = 450,
) -> list[float]: ...


@overload
def ciddor(
    wavelength: ScalarQuantity,
    T: ArrayQuantity,
    p: ScalarQuantity,
    h: float = 0.0,
    x_c: float = 450,
) -> list[float]: ...


@overload
def ciddor(
    wavelength: ScalarQuantity,
    T: ScalarQuantity,
    p: ArrayQuantity,
    h: float = 0.0,
    x_c: float = 450,
) -> list[float]: ...


def ciddor(
    wavelength: ScalarQuantity | ArrayQuantity = 0.6328 * u.um,
    T: ScalarQuantity | ArrayQuantity = 10 * u.deg_C,
    p: ScalarQuantity | ArrayQuantity = 101_325 * u.Pa,
    h: float = 0.0,
    x_c: float = 450.0,
) -> float | list[float]:
    """
    A wrapper for the `ref_index.ciddor` function from the `ref_index` library.
    Takes input parameters with astropy.units units.

    First checks the validity of input parameters and then passes them into the
    `ref_index.ciddor` function.
    """

    if isinstance(wavelength, float):
        print("No units provided for wavelength. Assuming microns.")
        wavelength *= u.um

    assert isinstance(wavelength, Quantity)

    if not wavelength.isscalar:
        assert T.isscalar, (
            "Only one array input is allowed, not both wavelength and temperature"
        )
        assert p.isscalar, (
            "Only one array input is allowed, not both wavelength and pressure"
        )
        return [ciddor(wavelength=wl, T=T, p=p, h=h, x_c=x_c) for wl in wavelength]

    if not T.isscalar:
        assert p.isscalar, (
            "Only one array input is allowed, not both temperature amd pressure"
        )
        return [ciddor(wavelength=wavelength, T=iT, p=p, h=h, x_c=x_c) for iT in T]

    if not p.isscalar:
        return [ciddor(wavelength=wavelength, T=T, p=ip, h=h, x_c=x_c) for ip in p]

    assert 0.3 <= wavelength.to("um").value <= 1.7  # noqa: PLR2004
    T_C: Quantity = T.to("deg_C", equivalencies=u.temperature())
    assert -40 <= T_C.value <= 100  # noqa: PLR2004
    assert 10_000 <= p.to("Pa").value <= 140_000  # noqa: PLR2004
    assert 0 <= h <= 1
    assert 0 <= x_c <= 2_000  # noqa: PLR2004

    return ref_index.ciddor(
        wave=wavelength.to("nm").value,
        t=T_C.value,
        p=p.to("Pa").value,
        rh=h * 100,
        co2=x_c,
        # warn = True, # Doesn't work - bug in package code
    )


@overload
def edlen(
    wavelength: ScalarQuantity, T: ScalarQuantity, p: ScalarQuantity, h: float = 0.0
) -> float: ...


@overload
def edlen(
    wavelength: ArrayQuantity, T: ScalarQuantity, p: ScalarQuantity, h: float = 0.0
) -> list[float]: ...


@overload
def edlen(
    wavelength: ScalarQuantity, T: ArrayQuantity, p: ScalarQuantity, h: float = 0.0
) -> list[float]: ...


@overload
def edlen(
    wavelength: ScalarQuantity, T: ScalarQuantity, p: ArrayQuantity, h: float = 0.0
) -> list[float]: ...


def edlen(
    wavelength: ScalarQuantity | ArrayQuantity = 656 * u.nm,
    T: ScalarQuantity | ArrayQuantity = 10 * u.deg_C,
    p: ScalarQuantity | ArrayQuantity = 101_325 * u.Pa,
    h: float = 0.0,
) -> float | list[float]:
    """
    A wrapper for the `ref_index.edlen` function from the `ref_index` library.
    Takes input parameters with astropy.units units.

    First checks the validity of input parameters and then passes them into the
    `ref_index.edlen` function.
    """

    if isinstance(wavelength, float):
        print("Not units provided for wavelength. Assuming microns")
        wavelength *= u.um

    assert isinstance(wavelength, Quantity)

    if not wavelength.isscalar:
        # lowmask: np.ndarray = np.array(wavelength.to("um").value) >= 0.3  # noqa: PLR2004
        # highmask: np.ndarray = np.array(wavelength.to("um").value) <= 1.7  # noqa: PLR2004
        # assert np.all(lowmask) and np.all(highmask)
        assert T.isscalar, (
            "Only one array input is allowed, not both wavelength and temperature"
        )
        assert p.isscalar, (
            "Only one array input is allowed, not both wavelength and pressure"
        )
        return [edlen(wl, T=T, p=p, h=h) for wl in wavelength]

    if not T.isscalar:
        assert p.isscalar, (
            "Only one array input is allowed, not both temperature amd pressure"
        )
        return [edlen(wavelength, T=iT, p=p, h=h) for iT in T]

    if not p.isscalar:
        return [edlen(wavelength, T=T, p=ip, h=h) for ip in p]

    assert 0.3 <= wavelength.to("um").value <= 1.7  # noqa: PLR2004
    T_C: Quantity = T.to("deg_C", equivalencies=u.temperature())
    assert -40 <= T_C.value <= 100  # noqa: PLR2004
    assert 10_000 <= p.to("Pa").value <= 140_000  # noqa: PLR2004
    assert 0 <= h <= 1

    return ref_index.edlen(
        wave=wavelength.to("nm").value,
        t=T_C.value,
        p=p.to("Pa").value,
        rh=h * 100,
        # warn = True, # Doesn't work - bug in package code
    )


air_refractive_index_formulae: dict[str, Callable] = {"ciddor": ciddor, "edlen": edlen}


"""
Models of refractive index
"""


@dataclass
class Model:
    """
    A generic model material, acts as a base class for more specific materials
    """

    tabulated: np.ndarray | None = None
    function: Callable | None = None

    @overload
    def refractive_index(self, wl: ScalarQuantity) -> float: ...

    @overload
    def refractive_index(self, wl: ArrayQuantity) -> np.ndarray: ...

    @overload
    def refractive_index(self, wl: None) -> Callable: ...

    @overload
    def refractive_index(self) -> Callable: ...

    def refractive_index(
        self,
        wl: Quantity | None = None,
    ) -> Callable | np.ndarray | float:
        if self.function:
            if wl is None:
                return self.function
            return self.function(wl)

        if self.tabulated:
            if wl is None:
                return lambda x: np.interp(
                    x,
                    self.tabulated[:, 0],  # type: ignore[non-subscriptable]
                    self.tabulated[:, 1],  # type: ignore[non-subscriptable]
                )
            return np.interp(wl, self.tabulated[:, 0], self.tabulated[:, 1])

        print("No function or tabulated values found")
        raise NotImplementedError

    @overload
    def n(self, wl: Quantity) -> float: ...

    @overload
    def n(self, wl: None) -> Callable: ...

    def n(self, wl: Quantity | None = None) -> Callable | float:
        if wl:
            return self.refractive_index(wl)

        return self.refractive_index

    @property
    def n_D(self) -> float:
        wl_D: Quantity = 587.6 * u.nm

        return self.n(wl_D)


@dataclass
class Air(Model):
    """
    Model the refractive index of air as a function of wavelength

    ciddor (Default)
    edlen
    zemax
    """

    air_model: str = "ciddor"
    temperature: float | None = None
    pressure: float | None = None
    humidity: float | None = None
    altitude: float | None = None

    def __post_init__(self) -> None:
        if self.air_model:
            f: Callable = air_refractive_index_formulae[self.air_model]
            super().__init__(function=f)

        else:
            super().__init__()


@dataclass
class Linear(Model):
    """
    A simple linear model - linear refractive index as a function of wavelength
    """

    m: float = 0
    b: float = 1

    def __post_init__(self) -> None:
        def f(wls: Quantity) -> float | np.ndarray:
            return self.m * wls + self.b

        super().__init__(function=f)


@dataclass
class Constant(Model):
    """
    A very simple constant model
    Constant refractive index, regardless of wavelength
    """

    c: float = 1

    def __post_init__(self) -> None:
        if self.c:

            def f(_wls: Quantity | None = None) -> float:
                return self.c

            super().__init__(function=f)


"""
Atmospheric models - how to scale refractive index with zenith angle?
Airmass models only, they do not depend on wavelength!
"""


def _plane_parallel(z: Quantity = 67 * u.deg) -> ArrayLike:
    """
    Zeroth-order approximation of airmass as a function of zenith distance
    """

    return np.sec(z)


def _young_irvine(z: Quantity = 67 * u.deg) -> ArrayLike:
    """
    https://en.wikipedia.org/wiki/Air_mass_(astronomy)

    Usable results up to z~80 degrees
    """

    return np.sec(z) * (1 - 0.0012 * (np.sec(z) ** 2 - 1))


def _hardie(z: Quantity = 67 * u.deg) -> ArrayLike:
    """
    https://en.wikipedia.org/wiki/Air_mass_(astronomy)

    Good for zenith angles up to ~85 degrees
    """

    return (
        np.sec(z)
        - 0.0018167 * (np.sec(z) - 1)
        - 0.0028750 * (np.sec(z) - 1) ** 2
        - 0.0008083 * (np.sec(z) - 1) ** 3
    )


def _rozenberg(z: Quantity = 67 * u.deg) -> ArrayLike:
    """
    https://en.wikipedia.org/wiki/Air_mass_(astronomy)


    """

    return 1 / (np.cos(z) + 0.025 * np.exp(-11 * np.cos(z)))


def _kasten_young(z: Quantity = 67 * u.deg) -> ArrayLike:
    """
    https://en.wikipedia.org/wiki/Air_mass_(astronomy)


    """

    return 1 / (np.cos(z) + 0.50572 * (6.007995 * u.deg + 90 * u.deg - z) ** -1.6364)


def _young(z: Quantity = 67 * u.deg) -> ArrayLike:
    """
    https://en.wikipedia.org/wiki/Air_mass_(astronomy)


    """

    return (1.002432 * np.cos(z) ** 2 + 0.148386 * np.cos(z) + 0.0096467) / (
        np.cos(z) ** 3 + 0.149864 * np.cos(z) + 0.000303978
    )


def _pickering(z: Quantity = 67 * u.deg) -> ArrayLike:
    """
    https://en.wikipedia.org/wiki/Air_mass_(astronomy)


    """

    h = 90 * u.deg - z

    return 1 / np.sin(h + 244 / (165 + 47 * h**1.1))


def _(wls: ArrayLike) -> ArrayLike:
    raise NotImplementedError


atmosphere_model_functions: dict[str, Callable] = {
    "plane-parallel": _plane_parallel,
    "young-irvine": _young_irvine,
    "hardie": _hardie,
    "rozenberg": _rozenberg,
    "kasten-young": _kasten_young,
    "young": _young,
    "pickering": _pickering,
    "_": _,
}


@dataclass
class Atmosphere:
    # TODO: should this class be a subclass of `Model`?
    """
    Model the refraction by the Earth's atmosphere

    Depends on the model of refractive index of air and the physical model of
    the atmosphere - how to scale the refractive index of air depending on the
    angle under which you are looking through the atmosphere.
    """

    atmosphere_model: str = "plane-parallel"
    zenith_distance: Quantity = 67.0 * u.deg
    air_model: Air = field(default_factory=Air)

    def __post_init__(self) -> None:
        if self.atmosphere_model:
            self.refraction_function = atmosphere_model_functions[self.atmosphere_model]

    @overload
    def refractive_index(self, wls: ScalarQuantity) -> float: ...

    @overload
    def refractive_index(self, wls: ArrayQuantity) -> np.ndarray: ...

    def refractive_index(self, wls: Quantity) -> float | np.ndarray:
        """
        Access the underlying refractive index model
        """

        return self.air_model.refractive_index(wls)

    @property
    def refraction(
        self,
        wls: Quantity | None = None,
        zs: Quantity | None = None,
    ) -> None:
        """
        Taking in an array of wavelengths and an array of zenith distances,
        return the angular refraction a ray of light passing through the
        atmosphere would undergo.
        """


def main() -> None: ...


if __name__ == "__main__":
    main()
