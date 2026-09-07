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
from numpy.typing import NDArray

from curved_air.type_hinting import ArrayQuantity, ScalarQuantity

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
    # "C": 656.281 * u.nm,  # Halpha
    "C'": 643.85 * u.nm,
    "a": 627.661 * u.nm,
    # "D1": 589.592 * u.nm,
    # "D2": 588.995 * u.nm,
    # "d": 587.5618 * u.nm, # Sodium
    "E2": 527.039 * u.nm,
    "b1": 518.362 * u.nm,
    "b2": 517.270 * u.nm,
    "b3": 516.891 * u.nm,
    "b4": 516.733 * u.nm,
    "c": 495.761 * u.nm,
    # "F": 486.134 * u.nm,  # Hbeta
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

material_CTEs: dict[str, ScalarQuantity] = {
    "Acetal": 59.2 * 1e-6 / u.K,
    "Acrylic": 70.0 * 1e-6 / u.K,
    "ALLVAR Alloy 30": -30.0 * 1e-6 / u.K,
    "Aluminium 1100": 23.6 * 1e-6 / u.K,
    "Aluminium 2011": 23.0 * 1e-6 / u.K,
    "Aluminium 2024": 22.9 * 1e-6 / u.K,
    "Aluminium 5086": 23.8 * 1e-6 / u.K,
    "Aluminium 6061": 23.6 * 1e-6 / u.K,
    "Aluminium 7075": 23.4 * 1e-6 / u.K,
    "Aluminium A356": 21.5 * 1e-6 / u.K,
    "Aluminium Bronze": 16.4 * 1e-6 / u.K,
    "Aluminium Nitride": 4.6 * 1e-6 / u.K,
    "Alumina (99.5%)": 7.2 * 1e-6 / u.K,
    "Beryllium": 11.3 * 1e-6 / u.K,
    "Bismuth": 13.4 * 1e-6 / u.K,
    "Carbon Fiber Composite": 0.1 * 1e-6 / u.K,
    "Chromium": 5.94 * 1e-6 / u.K,
    "Cobalt": 13.0 * 1e-6 / u.K,
    "Concrete": 12.0 * 1e-6 / u.K,
    "Copper": 16.5 * 1e-6 / u.K,
    "Copper Alloy C11000 (Electrolytic Tough Pitch)": 17.0 * 1e-6 / u.K,
    "Copper Alloy C17200 (Beryllium-Copper)": 16.7 * 1e-6 / u.K,
    "Copper Alloy C22000 (Commercial Bronze, 90%)": 18.4 * 1e-6 / u.K,
    "Copper Alloy C23000 (Red Brass, 85%)": 18.7 * 1e-6 / u.K,
    "Copper Alloy C26000 (Cartridge Brass)": 19.9 * 1e-6 / u.K,
    "Copper Alloy C27000 (Yellow Brass)": 20.3 * 1e-6 / u.K,
    "Copper Alloy C36000 (Free-Cutting Brass)": 20.5 * 1e-6 / u.K,
    "Copper Alloy C51000 (Phosphor Bronze, 5% A)": 17.8 * 1e-6 / u.K,
    "Copper Alloy C62300 (Aluminium Bronze, 9%)": 16.2 * 1e-6 / u.K,
    "Copper Alloy C71500 (Copper-Nickel, 30%)": 16.2 * 1e-6 / u.K,
    "Copper Alloy C93200 (Bearing Bronze)": 16.2 * 1e-6 / u.K,
    "Cordierite": 0.05 * 1e-6 / u.K,
    "CuCrZr": 17.1 * 1e-6 / u.K,
    "Cermet": 7.6 * 1e-6 / u.K,
    "Diamond": 1.0 * 1e-6 / u.K,
    "Delrin (Polyoxymethylene, POM)": 8.46 * 1e-6 / u.K,
    "Epoxy Low (Example)": 60.0 * 1e-6 / u.K,
    "Epoxy High (Example)": 120.0 * 1e-6 / u.K,
    "Fosterite": 9.7 * 1e-6 / u.K,
    "G-3 (Lengthwise)": 14.94 * 1e-6 / u.K,
    "G-3 (Crosswise)": 18.0 * 1e-6 / u.K,
    "G-5/G-9 (Lengthwise)": 14.94 * 1e-6 / u.K,
    "G-5/G-9 (Crosswise)": 18.0 * 1e-6 / u.K,
    "G-7 (Lengthwise)": 12.96 * 1e-6 / u.K,
    "G-7 (Crosswise)": 16.2 * 1e-6 / u.K,
    "G-10 (Lengthwise)": 9.9 * 1e-6 / u.K,
    "G-10 (Crosswise)": 11.88 * 1e-6 / u.K,
    "G-11 (Lengthwise)": 12.96 * 1e-6 / u.K,
    "G-11 (Crosswise)": 14.94 * 1e-6 / u.K,
    "Gold": 14.2 * 1e-6 / u.K,
    "Granite": 7.9 * 1e-6 / u.K,
    "Graphite": 4.5 * 1e-6 / u.K,
    "Hastelloy C-276": 11.2 * 1e-6 / u.K,
    "Haynes Alloy 25": 12.8 * 1e-6 / u.K,
    "Haynes Alloy 188": 12.1 * 1e-6 / u.K,
    "Haynes Alloy 230": 11.8 * 1e-6 / u.K,
    "Haynes Alloy 242": 10.8 * 1e-6 / u.K,
    "Haynes Alloy 244": 10.7 * 1e-6 / u.K,
    "Inconel 625": 12.8 * 1e-6 / u.K,
    "Inconel 718": 13.0 * 1e-6 / u.K,
    "Indium": 33.0 * 1e-6 / u.K,
    "Invar": 1.6 * 1e-6 / u.K,
    "Invar, Super": 7.2 * 1e-2 * 1e-6 / u.K,
    "Iridium": 6.5 * 1e-6 / u.K,
    "Iron": 11.8 * 1e-6 / u.K,
    "Iron, Cast": 11.4 * 1e-6 / u.K,
    "Iron, Ductile, Grade 60-40-18": 11.2 * 1e-6 / u.K,
    "Iron, Ductile, Grade 80-55-06": 10.6 * 1e-6 / u.K,
    "Kovar": 5.1 * 1e-6 / u.K,
    "Lead": 29.3 * 1e-6 / u.K,
    "Lead, Antimonial (6%)": 27.2 * 1e-6 / u.K,
    "Lithium": 46.0 * 1e-6 / u.K,
    "Magnesium": 26.0 * 1e-6 / u.K,
    "Magnesium, Alloy AZ31B": 26.0 * 1e-6 / u.K,
    "Manganese": 22.0 * 1e-6 / u.K,
    "Mica": 3.0 * 1e-6 / u.K,
    "Molybdenum": 4.8 * 1e-6 / u.K,
    "Monel": 15.7 * 1e-6 / u.K,
    "Monel 400": 13.9 * 1e-6 / u.K,
    "Nickel": 13.5 * 1e-6 / u.K,
    "Nickel 200": 13.3 * 1e-6 / u.K,
    "Niobium": 7.02 * 1e-6 / u.K,
    "Nylon": 110.0 * 1e-6 / u.K,
    "Nylon 6/6": 99.0 * 1e-6 / u.K,
    "Osmium": 5.5 * 1e-6 / u.K,
    "PEEK": 50.0 * 1e-6 / u.K,
    "Platinum": 9.0 * 1e-6 / u.K,
    "Polyester": 125.0 * 1e-6 / u.K,
    "Polyethylene (PE)": 110.0 * 1e-6 / u.K,
    "Polypropelene (PP)": 150.0 * 1e-6 / u.K,
    "Polystyrene (PS)": 70.0 * 1e-6 / u.K,
    "Rubber": 80.0 * 1e-6 / u.K,
    "Sapphire": 7.7 * 1e-6 / u.K,
    "Silicon": 2.6 * 1e-6 / u.K,
    "Silicon Carbide": 3.7 * 1e-6 / u.K,
    "Silicon Nitride": 2.8 * 1e-6 / u.K,
    "Silver": 19.7 * 1e-6 / u.K,
    "Steatite": 7.7 * 1e-6 / u.K,
    "Steel, Alloy A36": 11.7 * 1e-6 / u.K,
    "Steel, Alloy 1010": 12.2 * 1e-6 / u.K,
    "Steel, Alloy 1020": 11.7 * 1e-6 / u.K,
    "Steel, Alloy 1025": 12.0 * 1e-6 / u.K,
    "Steel, Alloy 1040": 11.3 * 1e-6 / u.K,
    "Steel, Alloy 1045": 11.6 * 1e-6 / u.K,
    "Steel, Alloy 4140": 12.3 * 1e-6 / u.K,
    "Steel, Alloy 4340": 12.3 * 1e-6 / u.K,
    "Steel, Carbon Steel": 10.8 * 1e-6 / u.K,
    "Steel, Mild": 10.6 * 1e-6 / u.K,
    "Steel, Stainless 304": 17.2 * 1e-6 / u.K,
    "Steel, Stainless 316": 15.9 * 1e-6 / u.K,
    "Steel, Stainless 405": 10.8 * 1e-6 / u.K,
    "Steel, Stainless 440A": 10.2 * 1e-6 / u.K,
    "Steel, Stainless 12-7PH": 11.0 * 1e-6 / u.K,
    "Steel, Tool Steel D2 (Annealed)": 10.5 * 1e-6 / u.K,
    "Steel, Tool Steel T1": 9.7 * 1e-6 / u.K,
    "Steel, Tool Steel M2": 10.1 * 1e-6 / u.K,
    "Steel, Tool Steel W1": 10.4 * 1e-6 / u.K,
    "Steel, Tool Steel O6 (Annealed)": 13.6 * 1e-6 / u.K,
    "Tantalum": 6.48 * 1e-6 / u.K,
    "Teflon (PTFE)": 112.0 * 1e-6 / u.K,
    "Thallium": 29.9 * 1e-6 / u.K,
    "Thulium": 13.5 * 1e-6 / u.K,
    "Tin": 23.8 * 1e-6 / u.K,
    "Titanium (99.9% Ti)": 8.41 * 1e-6 / u.K,
    "Titanium, Grade 2": 8.6 * 1e-6 / u.K,
    "Titanium, Ti-6Al-4V Grade 5, Annealed": 8.6 * 1e-6 / u.K,
    "Titanium Ti-5Al-2.5Sn": 9.4 * 1e-6 / u.K,
    "Titanium Ti-8Mn": 8.6 * 1e-6 / u.K,
    "Tungsten": 4.5 * 1e-6 / u.K,
    "Uranium": 13.5 * 1e-6 / u.K,
    "Wax": 6.5 * 1e-6 / u.K,
    "Ytterbium": 26.0 * 1e-6 / u.K,
    "Yttria": 7.2 * 1e-6 / u.K,
    "Yttrium": 11.0 * 1e-6 / u.K,
    "Zerodur, Class 2": 0.1 * 1e-6 / u.K,
    "Zerodur, Class 1": 0.05 * 1e-6 / u.K,
    "Zerodur, Class 0": 0.02 * 1e-6 / u.K,
    "Zerodur, Class 0 SPECIAL": 0.01 * 1e-6 / u.K,
    "Zerodur, Class 0 EXTREME": 0.007 * 1e-6 / u.K,
    "Zinc": 23.0 * 1e-6 / u.K,
    "Zirconium": 5.5 * 1e-6 / u.K,
    "Zirconia": 10.5 * 1e-6 / u.K,
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
) -> float | NDArray[np.floating]:
) -> float | NDArray[np.floating]:
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
        return np.array(
            [ciddor(wavelength=wl, T=T, p=p, h=h, x_c=x_c) for wl in wavelength]
        )
        return np.array(
            [ciddor(wavelength=wl, T=T, p=p, h=h, x_c=x_c) for wl in wavelength]
        )

    if not T.isscalar:
        assert p.isscalar, (
            "Only one array input is allowed, not both temperature amd pressure"
        )
        return np.array(
            [ciddor(wavelength=wavelength, T=iT, p=p, h=h, x_c=x_c) for iT in T]
        )
        return np.array(
            [ciddor(wavelength=wavelength, T=iT, p=p, h=h, x_c=x_c) for iT in T]
        )

    if not p.isscalar:
        return np.array(
            [ciddor(wavelength=wavelength, T=T, p=ip, h=h, x_c=x_c) for ip in p]
        )
        return np.array(
            [ciddor(wavelength=wavelength, T=T, p=ip, h=h, x_c=x_c) for ip in p]
        )

    assert 0.3 <= wavelength.to("um").value <= 1.7  # noqa: PLR2004
    T_C: Quantity = T.to("deg_C", equivalencies=u.temperature())
    assert -40 <= T_C.value <= 100  # noqa: PLR2004
    assert 10_000 <= p.to("Pa").value <= 140_000  # noqa: PLR2004
    assert 0 <= h <= 1
    assert 0 <= x_c <= 2_000  # noqa: PLR2004

    return float(
        ref_index.ciddor(
            wave=wavelength.to("nm").value,
            t=T_C.value,
            p=p.to("Pa").value,
            rh=h * 100,
            co2=x_c,
            # warn = True, # Doesn't work - bug in package code
        )
    return float(
        ref_index.ciddor(
            wave=wavelength.to("nm").value,
            t=T_C.value,
            p=p.to("Pa").value,
            rh=h * 100,
            co2=x_c,
            # warn = True, # Doesn't work - bug in package code
        )
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
) -> float | NDArray[np.floating]:
) -> float | NDArray[np.floating]:
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
        return np.array([edlen(wl, T=T, p=p, h=h) for wl in wavelength])
        return np.array([edlen(wl, T=T, p=p, h=h) for wl in wavelength])

    if not T.isscalar:
        assert p.isscalar, (
            "Only one array input is allowed, not both temperature amd pressure"
        )
        return np.array([edlen(wavelength, T=iT, p=p, h=h) for iT in T])
        return np.array([edlen(wavelength, T=iT, p=p, h=h) for iT in T])

    if not p.isscalar:
        return np.array([edlen(wavelength, T=T, p=ip, h=h) for ip in p])
        return np.array([edlen(wavelength, T=T, p=ip, h=h) for ip in p])

    assert 0.3 <= wavelength.to("um").value <= 1.7  # noqa: PLR2004
    T_C: Quantity = T.to("deg_C", equivalencies=u.temperature())
    assert -40 <= T_C.value <= 100  # noqa: PLR2004
    assert 10_000 <= p.to("Pa").value <= 140_000  # noqa: PLR2004
    assert 0 <= h <= 1

    return float(
        ref_index.edlen(
            wave=wavelength.to("nm").value,
            t=T_C.value,
            p=p.to("Pa").value,
            rh=h * 100,
            # warn = True, # Doesn't work - bug in package code
        )
    return float(
        ref_index.edlen(
            wave=wavelength.to("nm").value,
            t=T_C.value,
            p=p.to("Pa").value,
            rh=h * 100,
            # warn = True, # Doesn't work - bug in package code
        )
    )


@overload
def zemax(
    wl: ScalarQuantity,
    T: ScalarQuantity,
    p: ScalarQuantity,
    h: float | None = None,
    x_c: float | None = None,
) -> float: ...


@overload
def zemax(
    wl: ArrayQuantity,
    T: ScalarQuantity,
    p: ScalarQuantity,
    h: float | None = None,
    x_c: float | None = None,
) -> list[float]: ...


def zemax(
    wl: ScalarQuantity | ArrayQuantity = 656 * u.nm,
    T: ScalarQuantity | ArrayQuantity = 15 * u.deg_C,
    p: ScalarQuantity | ArrayQuantity = 101_325 * u.Pa,
    h: float | None = None,
    x_c: float | None = None,
) -> float | NDArray[np.floating]:
    """
    Zemax definition of the refractive index of air

    formula from F. Kophlrausch, Praktiche Physik, 1968, Vol. 1, Page 408
    """

    if h is not None:
        print("Warning: h parameter is not used in the Zemax formula. Ignoring it.")

    if x_c is not None:
        print("Warning: x_c parameter is not used in the Zemax formula. Ignoring it.")

    wl: float = wl.to(u.um).value

    T: float = T.to(u.deg_C, equivalencies=u.temperature()).value

    relative_pressure: Quantity = p.to(u.Pa) / (101_325 * u.Pa)
    P: float = relative_pressure.value

    n_ref: float = (
        1
        + (
            6432.8
            + ((2949810 * wl**2) / (146 * wl**2 - 1))
            + ((25540 * wl**2) / (41 * wl**2 - 1))
        )
        * 1e-8
    )

    return 1 + ((n_ref - 1) * P) / (1 + (T - 15) * 3.4785e-3)


air_refractive_index_formulae: dict[str, Callable] = {
    "ciddor": ciddor,
    "edlen": edlen,
    "zemax": zemax,
}


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

    m: ScalarQuantity = 0 / u.um
    b: float = 1

    def __post_init__(self) -> None:

        assert isinstance(self.m, Quantity)
        assert self.m.unit.is_equivalent(u.um**-1), "m must have units of 1/wavelength"

        def f(wls: Quantity) -> float | NDArray:
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

            def f(_wls: Quantity | None = None) -> float | NDArray:
                if _wls is None:
                    return self.c
                if isinstance(_wls, Quantity):
                    if _wls.isscalar:
                        return self.c
                    return np.ones_like(_wls).value * self.c
                if isinstance(_wls, NDArray | list):
                    return np.ones_like(_wls).value * self.c
                return self.c

            super().__init__(function=f)


"""
Atmospheric models - how to scale refractive index with zenith angle?
Airmass models only, they do not depend on wavelength!
"""


def _plane_parallel(z: Quantity = 67 * u.deg) -> NDArray[np.floating]:
    """
    Zeroth-order approximation of airmass as a function of zenith distance
    """

    return np.sec(z)


def _young_irvine(z: Quantity = 67 * u.deg) -> NDArray[np.floating]:
    """https://en.wikipedia.org/wiki/Air_mass_(astronomy)tronomy)

    Usable results up to z~80 degrees
    """

    return np.sec(z) * (1 - 0.0012 * (np.sec(z) ** 2 - 1))


def _hardie(z: Quantity = 67 * u.deg) -> NDArray[np.floating]:
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


def _rozenberg(z: Quantity = 67 * u.deg) -> NDArray[np.floating]:
    """
    https://en.wikipedia.org/wiki/Air_mass_(astronomy)
    """

    return 1 / (np.cos(z) + 0.025 * np.exp(-11 * np.cos(z)))


def _kasten_young(z: Quantity = 67 * u.deg) -> NDArray[np.floating]:
    """
    https://en.wikipedia.org/wiki/Air_mass_(astronomy)
    """

    return 1 / (np.cos(z) + 0.50572 * (6.007995 * u.deg + 90 * u.deg - z) ** -1.6364)


def _young(z: Quantity = 67 * u.deg) -> NDArray[np.floating]:
    """
    https://en.wikipedia.org/wiki/Air_mass_(astronomy)
    """

    return (1.002432 * np.cos(z) ** 2 + 0.148386 * np.cos(z) + 0.0096467) / (
        np.cos(z) ** 3 + 0.149864 * np.cos(z) + 0.000303978
    )


def _pickering(z: Quantity = 67 * u.deg) -> NDArray[np.floating]:
    """
    https://en.wikipedia.org/wiki/Air_mass_(astronomy)
    """

    h = 90 * u.deg - z

    return 1 / np.sin(h + 244 / (165 + 47 * h**1.1))


def _(z: Quantity = 67 * u.deg) -> NDArray[np.floating]:
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
