"""
Contains functions that model the refractive index of air, the airmass of the
atmosphere under different angles, ...

Contains wavelength definitions of Fraunhofer lines

"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike

from astropy import units as u
from astropy.units import Quantity

import ref_index



_fraunhofer_lines = {
    "i":  365.01 * u.nm,
    "h":  404.66 * u.nm,
    "g":  435.84 * u.nm,
    "F'": 479.99 * u.nm,
    "F":  486.13 * u.nm,
    "e":  546.07 * u.nm,
    "d":  587.56 * u.nm,
    "D":  589.3  * u.nm,
    "C'": 643.85 * u.nm,
    "C":  656.27 * u.nm,
    "r":  706.52 * u.nm,
    "A'": 768.2  * u.nm,
    "s":  852.11 * u.nm,
    "t": 1013.98 * u.nm,
    }



"""
Wrappers to functions from the `ref_index` library

These wrappers now take `astropy.units.Quantity` objects in, to allow input
parameters with any appropriate unit to be used, and no confusion about Celcius
or Kevlin, etc. It does come with the overhead of requiring `astropy` but for me
and others who regularly use `astropy` anyway, it is well worth it.

Original paper: Ciddor 1996, https://doi.org/10.1364/AO.35.001566
"""

def ciddor(
        λ: Quantity = 0.6328 * u.um,
        T: Quantity = 10 * u.deg_C,
        p: Quantity = 101_325 * u.Pa,
        h: float = 0.,
        x_c: float = 450.
        ) -> float:    
    """
    A wrapper for the `ref_index.ciddor` function from the `ref_index` library.
    Takes input parameters with astropy.units units.
    
    First checks the validity of input parameters and then passes them into the
    `ref_index.ciddor` function.
    """
    
    if isinstance(λ, (float, int)):
        λ *= u.um

    if isinstance(λ.value, (float, int)):
        assert 0.3 <= λ.to("um").value <= 1.7
    elif isinstance(λ.value, (list, np.ndarray)):
        lowmask = np.array(λ.to("um").value) >= 0.3
        highmask = np.array(λ.to("um").value) <= 1.7
        assert np.all(lowmask) and np.all(highmask)

    T_C = T.to("deg_C", equivalencies=u.temperature())
    assert -40 <= T_C.value <= 100
    assert 10_000 <= p.to("Pa").value <= 140_000
    assert 0 <= h <= 1
    assert 0 <= x_c <= 2_000
    
    return ref_index.ciddor(
        wave = λ.to("nm").value,
        t = T_C.value,
        p = p.to("Pa").value,
        rh = h * 100,
        co2 = x_c,
        # warn = True, # Doesn't work - bug in package code
        )



def edlen(
        λ: Quantity = 656 * u.nm,
        T: Quantity = 10 * u.deg_C,
        p: Quantity = 101_325 * u.Pa,
        h: float = 0.,
        ) -> float:    
    """
    A wrapper for the `ref_index.edlen` function from the `ref_index` library.
    Takes input parameters with astropy.units units.
    
    First checks the validity of input parameters and then passes them into the
    `ref_index.edlen` function.
    """

    if isinstance(λ.value, (float, int)):
        assert 0.3 <= λ.to("um").value <= 1.7
    elif isinstance(λ.value, (list, np.ndarray)):
        lowmask = np.array(λ.to("um").value) >= 0.3
        highmask = np.array(λ.to("um").value) <= 1.7
        assert np.all(lowmask) and np.all(highmask)

    T_C = T.to("deg_C", equivalencies=u.temperature())
    assert -40 <= T_C.value <= 100
    assert 10_000 <= p.to("Pa").value <= 140_000
    assert 0 <= h <= 1
    
    return ref_index.edlen(
        wave = λ.to("nm").value,
        t = T_C.value,
        p = p.to("Pa").value,
        rh = h * 100,
        # warn = True, # Doesn't work - bug in package code
        )



air_refractive_index_formulae = {"ciddor": ciddor, "edlen": edlen}


"""
Models of refractive index

"""



@dataclass
class Model:
    """
    A generic model material, acts as a base class for more specific materials
    """

    tabulated: ArrayLike = None
    function: Callable = None


    def refractive_index(self, wl: Quantity = None) -> Callable | float:

        if self.function:
            if wl is None:
                return self.function
            else:
                return self.function(wl)
            
        elif self.tabulated:
            if wl is None:
                return self.tabulated
            else:
                return np.interp(wl, self.tabulated[:, 0], self.tabulated[:, 1])

        else:
            print("No function or tabulated values found")
            raise NotImplementedError


    def n(self, wl: Quantity = None) -> Callable | float:

        if wl:
            return self.refractive_index(wl)
        
        return self.refractive_index
    
    
    @property
    def n_D(self) -> float:
        
        wl_D = 587.6 * u.nm
        
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
    pressure:    float | None = None
    humidity:    float | None = None
    altitude:    float | None = None

    def __post_init__(self):
        if self.air_model:
            f = air_refractive_index_formulae[self.air_model]
            super().__init__(function = f)

        else:
            super().__init__()


@dataclass
class Linear(Model):
    """
    A simple linear model - linear refractive index as a function of wavelength
    """

    m: float = None
    b: float = None


    def __post_init__(self):

        if self.m and self.b:
            def f(wls: ArrayLike):
                return self.m * wls + self.b

            super().__init__(function = f)


@dataclass
class Constant(Model):
    """
    A very simple constant model
    Constant refractive index, regardless of wavelength
    """

    c: float = None

    def __post_init__(self):

        if self.c:
            def f(_wls: ArrayLike = None):
                return self.c

            super().__init__(function = f)
            
            
            


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

    return np.sec(z)\
           - 0.0018167 * (np.sec(z) - 1)\
           - 0.0028750 * (np.sec(z) - 1) ** 2\
           - 0.0008083 * (np.sec(z) - 1) ** 3


def _rozenberg(z: Quantity = 67 * u.deg) -> ArrayLike:
    """
    https://en.wikipedia.org/wiki/Air_mass_(astronomy)
    
    
    """

    return 1 / (np.cos(z) + 0.025 * np.exp(-11 * np.cos(z)))



def _kasten_young(z: Quantity = 67 * u.deg) -> ArrayLike:
    """
    https://en.wikipedia.org/wiki/Air_mass_(astronomy)
    
    
    """

    return 1 / (np.cos(z)\
                + 0.50572 * (6.007995 * u.deg + 90 * u.deg - z) ** -1.6364 )


def _young(z: Quantity = 67 * u.deg) -> ArrayLike:
    """
    https://en.wikipedia.org/wiki/Air_mass_(astronomy)
    
    
    """

    return (1.002432 * np.cos(z) ** 2 + 0.148386 * np.cos(z) + 0.0096467)\
         / (np.cos(z) ** 3 + 0.149864 * np.cos(z) + 0.000303978)


def _pickering(z: Quantity = 67 * u.deg) -> ArrayLike:
    """
    https://en.wikipedia.org/wiki/Air_mass_(astronomy)
    
    
    """

    h = 90 * u.deg - z

    return 1 / np.sin(h + 244 / (165 + 47 * h ** 1.1))



def _(wls: ArrayLike) -> ArrayLike:

    raise NotImplementedError


atmosphere_model_functions = {
                              "plane-parallel": _plane_parallel,
                              "young-irvine": _young_irvine,
                              "hardie": _hardie,
                              "rozenberg": _rozenberg,
                              "kasten-young": _kasten_young,
                              "young": _young,
                              "pickering": _pickering,
                              "_": _
                             }


@dataclass
class Atmosphere():
    # TODO: should this class be a subclass of `Model`?
    """
    Model the refraction by the Earth's atmosphere

    Depends on the model of refractive index of air and the physical model of
    the atmosphere - how to scale the refractive index of air depending on the
    angle under which you are looking through the atmosphere.
    """

    atmosphere_model: str = "plane-parallel"
    zenith_distance: Quantity = 67. * u.deg
    air_model: Air = field(default_factory = Air)
        
    def __post_init__(self):
        if self.atmosphere_model:
            self.refraction_function = atmosphere_model_functions[self.atmosphere_model]


    @property
    def refractive_index(self, wls: ArrayLike) -> ArrayLike:
        """
        Access the underlying refractive index model
        """

        return self.air_model.refractive_index(wls)


    @property
    def refraction(self, wls: ArrayLike = None, zs: ArrayLike = None) -> ArrayLike:
        """
        Taking in an array of wavelengths and an array of zenith distances,
        return the angular refraction a ray of light passing through the
        atmosphere would undergo.
        """

        ...



def main():
    
    ...


if __name__ == "__main__":

    main()
