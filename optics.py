"""
Originally from 'EchelleDesigner' project
"""

from dataclasses import dataclass, field
from functools import cached_property
from typing import Callable, Union, Iterable

import numpy as np
from numpy.typing import ArrayLike

from astropy import units as u
from astropy.units import Quantity

from scipy.optimize import minimize

try:
    from curved_air.glass import Glass
    from curved_air.other import Model, Air
except ImportError:
    from glass import Glass
    from other import Model, Air



@dataclass
class Prism:

    glass: Glass = None
    apex_angle: Quantity = 60 * u.deg
    min_wl: Quantity = 390 * u.nm
    max_wl: Quantity = 950 * u.nm
    incidence_angle: Quantity = None
    
    
    def __post_init__(self):

        assert self.glass is not None
        
        if self.incidence_angle is None:        
            if self.min_wl is not None and self.max_wl is not None:
                self.incidence_angle = self.find_incidence_angle(
                    min_wl = self.min_wl,
                    max_wl = self.max_wl
                    )
            else:
                self.incidence_angle = self.minimum_deviation_incidence_angle()


    @property
    def refractive_index(self) -> Callable:

        return self.glass.refractive_index


    @property
    def n(self) -> Callable:
        """Alias of refractive_index"""

        return self.refractive_index


    def deviation_angle(
        self,
        wl: ArrayLike = 0.6328 * u.um,
        incidence_angle: Quantity = None,
        ) -> Quantity:
        
        if incidence_angle is None:
            incidence_angle = self.incidence_angle
            
        i = incidence_angle

        return i - self.apex_angle + np.arcsin(
                        self.glass.n(wl) * np.sin(
                            self.apex_angle - np.arcsin(
                                np.sin(i) / self.glass.n(wl)
                                )
                            )
                        )


    def find_incidence_angle(
            self,
            min_wl: Quantity = 385 * u.nm,
            max_wl: Quantity = 950 * u.nm,
            ) -> Quantity:
        
        min_wl_incidence = self.minimum_deviation_incidence_angle(wl=min_wl)
        max_wl_incidence = self.minimum_deviation_incidence_angle(wl=max_wl)
        
        return ((min_wl_incidence + max_wl_incidence) / 2).to(u.deg)


    def minimum_deviation_angle(
            self, wl: ArrayLike = 0.6328 * u.um
            ) -> Quantity:
        
        minimum_deviation_angle = 2 * np.arcsin(self.glass.n(wl)\
                                    * np.sin(self.apex_angle / 2))\
                                                            - self.apex_angle

        return minimum_deviation_angle.to(u.deg)

    
    def minimum_deviation_incidence_angle(
            self,
            wl: ArrayLike = 632.8 * u.nm
            ) -> Quantity:
        
        """
        Calculate the incidence angle that provides the minimum deviation
        condition for the prism at a given wavelength.

        Parameters
        ----------
        wl : ArrayLike, optional
            Single or multiple wavelengths, with linear units from astropy.units
            The default value is 0.6328 um, the wavelength of a HeNe laser.

        Returns
        -------
        A single angle or an array of incidence angles at which minimum
        deviation occurs
        """

        incidence_angle = (self.apex_angle + self.minimum_deviation_angle(wl)) / 2

        return incidence_angle
    
    def trace(
        self,
        wls: ArrayLike = None,
        incidence_angle: Quantity = None,
        debug_level: float = 0,
        ) -> ArrayLike:
        
        """
        Trace a set of rays through the Prism, solving Snell's Law at each
        interface
        """

        if wls is None:
            wls = [0.4861, 0.5893, 0.6563] * u.um

        if incidence_angle is None:
            incidence_angle = self.incidence_angle

        try:
            rays = incidence_angle * np.ones(len(wls))
        except TypeError: # wls is a single value
            rays = incidence_angle

        previous_angle = 0 * u.degree

        rays = _refract(
            n_1 = Air().refractive_index(wls),
            n_2 = self.refractive_index,
            theta_in = rays - previous_angle,
            wavelength = wls,
            )
        
            
        if debug_level > 0:
            print(f"Refracting through a {self.glass} prism of {self.apex_angle}")

            
        # Refract through the last surface into the exit medium
        rays = _refract(
            n_1 = self.refractive_index(wls),
            n_2 = Air().refractive_index(wls),
            theta_in = rays - self.apex_angle,
            wavelength = wls,
            )

        return rays.to(u.degree)


@dataclass
class PrismStack:
    """
    TODO:
    - [ ] Add method effective_refractive_index(), to return the effective
    refractive index of the PrismStack. How to define this? Take the angle
    between front and back surfaces? But what if that angle is zero? Refractive
    index is effectively infinite... Need to have an "effective apex angle" in
    there as well somewhere... Take the average of absolute apex angles of each
    of the components? Need to think about the best way to define this so that
    it generalises well, but this method will be SO USEFUL in optimising -> you
    can then just match this (or a scaled/offset version) to any desired
    refractive index vector, and drive the difference to be small.
    """

    prism_list: list[Prism] = field(default_factory = list)
    entrance_medium: Model = Air()
    exit_medium: Model = Air()


    def add_prism(self, prism: Prism):

        self.prism_list.append(prism)


    def trace(
        self,
        wls: ArrayLike = None,
        incidence_angle: Quantity = 0 * u.deg,
        debug_level: float = 0,
        ) -> ArrayLike:
        
        """
        Trace a set of rays through the PrismStack, solving Snell's Law at each
        interface
        """

        if wls is None:
            wls = [0.4861, 0.5893, 0.6563] * u.um

        try:
            rays = incidence_angle * np.ones(len(wls))
        except TypeError: # wls is a single value
            rays = incidence_angle

        previous_material = self.entrance_medium
        previous_angle = 0 * u.degree

        for n, prism in enumerate(self.prism_list):

            rays = _refract(
                n_1 = previous_material.refractive_index(wls),
                n_2 = prism.refractive_index,
                theta_in = rays - previous_angle,
                wavelength = wls,
                )
            
            if debug_level > 0:
                print(f"Refracting through a {prism.glass} prism of {prism.apex_angle}")

            previous_material = prism.glass
            previous_angle = prism.apex_angle
            

            # print(rays.to("degree"))

        # Refract through the last surface into the exit medium
        rays = _refract(
            n_1 = previous_material.refractive_index(wls),
            n_2 = self.exit_medium.refractive_index(wls),
            theta_in = rays - prism.apex_angle,
            wavelength = wls,
            )

        return rays.to(u.degree)


    def zero_deviation_wavelengths(self,
                                   min_wl: Quantity = 0.39 * u.um,
                                   max_wl: Quantity = 1.50 * u.um,
                                   num_wls: float = 100,
                                   incidence_angle: Quantity = 0*u.deg) -> ArrayLike:
        """
        This method traces the dispersion of light through a compound prism and
        returns the wavelengths at which the prism does not deviate the light.
        These zero-deviation wavelengths are interesting in design for
        astronomical ADCs, and may be a fast way to check that a design is
        working the way it should.
        """

        if isinstance(min_wl, Quantity) and isinstance(max_wl, Quantity):
            wls = np.linspace(min_wl.to("um").value,
                              max_wl.to("um").value, num_wls) * u.um
        elif isinstance(min_wl, (int, float)) and isinstance(max_wl, (int, float)):
            wls = np.linspace(min_wl, max_wl, num_wls) * u.um
        else:
            raise TypeError("min_wl and max_wl parameters must be int, float, "\
                +"or astropy.units.quantity.Quantity with appropriate units.")

        out_angles = self.trace(wls=wls, incidence_angle=incidence_angle)
        zwls = wls[np.where(np.array(out_angles.value)[:-1]\
                          / np.array(out_angles.value)[1:] < 0)]

        if len(zwls) >= 2:
            return zwls
        elif len(zwls) == 1:
            return zwls[0]
        else:
            return None


@dataclass
class ADC:
    
    prism_stack: PrismStack
    air_model: Model = Air()
    min_wl: Quantity = 390 * u.nm
    max_wl: Quantity = 950 * u.nm
    D_telescope: Quantity = 0.8 * u.m
    D_ADC: Quantity = 15 * u.mm
    z_max: Quantity = 65 * u.degree
    criterion: str = "PTV" # "RMS", "RSS", "PTV", "max_abs"

    G_alpha_air:    Quantity | None = None
    alpha:          Quantity | None = None
    beta:           Quantity | None = None
    wl_0:           Quantity | None = None
    
    residuals_function: Callable | None = None
    
    
    @property
    def g1(self) -> Glass:
        return self.prism_stack.prism_list[0]
    
    
    @property
    def g2(self) -> Glass:
        return self.prism_stack.prism_list[1]
    
    
    @property
    def epsilon(self) -> Quantity:
        if self.wl_0 is None:
            return None
        return (self.g1.refractive_index(self.wl_0) - 1)\
             / (self.g2.refractive_index(self.wl_0) - 1)
    
    
    @property
    def alpha2(self) -> Quantity:
        return self.alpha + self.beta
    
    
    @property
    def beam_comp(self) -> Quantity:
        return self.D_telescope / self.D_ADC
    
    
    @property
    def magnification_at_ADC(self) -> Quantity:
        return self.beam_comp
    
    
    def wls(self, n: int = 100) -> Iterable:
        return np.linspace(
            self.min_wl.to(u.um).value,
            self.max_wl.to(u.um).value,
            100
            ) * u.um
    
    
    @property
    def residuals_RMS(self) -> Quantity:
        
        RMS = np.sqrt(np.mean(self.residuals_function(wls = self.wls(100))**2))
        
        return RMS.to(u.mas)
    
    
    @property
    def residuals_RSS(self) -> Quantity:
        
        RSS = np.sqrt(np.sum(self.residuals_function(wls = self.wls(100))**2))
    
        return RSS.to(u.mas)
    
    
    @property
    def residuals_PTV(self) -> Quantity:
        
        PTV = abs(
            max(self.residuals_function(wls = self.wls(100)))
          - min(self.residuals_function(wls = self.wls(100)))
          )
        
        return PTV.to(u.mas)
    
    
    def optimise(
        self,
        plot: bool = False,
        num_samples: int = 20,
        ) -> None:
        """
        TODO:
        
        - [ ] Instead of computing residuals based on the difference between
        refractive indices, actually trace rays through the ADC
        
        - [ ] 
        
        """
        
        
        if len(self.prism_stack.prism_list) != 2:
            raise NotImplementedError("This method is only implemented for "+\
                "two-component compound prisms.")
            
            
        g1 = self.prism_stack.prism_list[0]
        g2 = self.prism_stack.prism_list[1]
    
        wls = self.wls(n = num_samples)

        A_mean = 1 - 2.57064513e-07
        B_mean = 1.39479963e-04
        G = 1e8
        self.G_alpha_air = G * np.arctan((1 / G) *\
            (A_mean * np.tan(self.z_max) + B_mean * np.tan(self.z_max) ** 3))

        # G_alpha_air is the required scaled prism angle for a thin prism model
        # of atmospheric dispersion. It is roughly equal to
        # Atan(z_max) + Btan^3(z_max), which is in turn roughly equal to
        # tan(z_max) but the extra terms are important when we are talking about
        # dispersion residuals anything less than about a hundred
        # milliarcseconds (in the atmosphere, so once scaled that respresents
        # residuals of several hundred mas - you can decide which approximation
        # if any is appropriate to your application.

        # Approximation:
        #G_alpha_air = np.tan(np.radians(z_max))

        # Find a linear combination of refractive index curves that best match
        # the refractive index of air, in which the apex angles of the two prism
        def residuals(
            x: tuple[float, float] = (1, 0.6328), # alpha, wl_0 - no units
            wls: ArrayLike = wls,
            return_residuals: bool = False,
            verbose: bool = False,
            ):
            
            # Fully parameterised by alpha and wl_0
            alpha, wl_0 = x
            
            alpha = alpha * u.degree
            wl_0 = wl_0 * u.um
            
            if verbose:
                print(f"Trying: {alpha = }, {wl_0 = }")

            epsilon = (g1.refractive_index(wl_0) - 1)\
                    / (g2.refractive_index(wl_0) - 1)

            delta_air = (
                self.beam_comp * self.G_alpha_air * (
                    self.air_model.refractive_index(wls)
                  - self.air_model.refractive_index(wl_0)
                    )
                ).si.to(u.degree)
            
            delta_ADC = (
                g1.refractive_index(wls)
              - epsilon * g2.refractive_index(wls)
              + epsilon - 1
              ) * alpha

            residuals = (delta_ADC - delta_air) * u.radian
            # Put it back in equivalent on-sky angle
            residuals /= (self.G_alpha_air * self.beam_comp)
            
            if return_residuals:
                return residuals.to(u.mas)

            elif self.criterion == "PTV":
                return abs(max(residuals.to(u.mas)) - min(residuals.to(u.mas)))
            elif self.criterion == "RMS":
                return np.sqrt(np.mean(residuals.to(u.mas)**2))
            elif self.criterion == "RSS":
                return np.sqrt(np.sum(np.square(residuals.to(u.mas).value)))
            elif self.criterion == "max_abs":
                return max(np.abs(residuals.to(u.mas)))

        
        try:
            result = minimize(
                fun = residuals,
                x0 = [1., np.median(wls.to(u.um).value)],
                bounds = [    #min     #max
                            (-np.inf, np.inf),
                            (0.4, 1.5), # um
                        ]
                    )

            alpha_fit, wl_0_fit = result.x

        except Exception as e:
            print(e)
            print(f"Unable to converge on solution.")
            return None
        
        # residuals_PTV = residuals(x = result.x)
        # residuals_curve = residuals(x = result.x, return_residuals=True)
        
        def residuals_function(wls: ArrayLike = wls):
            return residuals(
                x = (
                    self.alpha.to(u.degree).value,
                    self.wl_0.to(u.um).value,
                    ),
                wls = wls,
                return_residuals = True
                )
        
        self.residuals_function = residuals_function
        

        # Solve for individual prism apex angle from fitted constant
        self.first_apex_angle = alpha_fit * u.degree / 2
        epsilon_fit = (g1.refractive_index(wl_0_fit) - 1)\
                    / (g2.refractive_index(wl_0_fit) - 1)
        self.second_apex_angle = epsilon_fit * self.first_apex_angle
        
        self.alpha = alpha_fit * u.degree
        self.beta = self.second_apex_angle - self.first_apex_angle
        self.wl_0 = wl_0_fit * u.um
        
        
        results = {
                    "g1": g1,
                    "g2": g2,

                    "D_telescope": self.D_telescope,
                    "D_ADC": self.D_ADC,
                    "G_alpha_air": self.G_alpha_air,
                    "beam_comp": self.beam_comp,

                    "z_max": self.z_max,
                    "min_wl": self.min_wl,
                    "max_wl": self.max_wl,

                    "first_apex_angle": self.first_apex_angle,
                    "second_apex_angle": self.second_apex_angle,

                    "alpha_fit": self.alpha,
                    "wl_0_fit": self.wl_0,
                    "epsilon_fit": self.epsilon,

                    "residuals_rms": self.residuals_RMS,
                    "residuals_rss": self.residuals_RSS,
                    "residuals_ptv": self.residuals_PTV,
                }
        
        if plot:            
            self.plot()
            
        return results
            
            
    def plot(self, num_samples: int = 100):
        
        wls = self.wls(n = num_samples)
            
        from matplotlib import pyplot as plt
        
        plt.rcParams["axes.labelsize"] = 14
        plt.rcParams["legend.fontsize"] = 14
    
        min_thickness = 10 * u.mm
        gap = 6 * u.mm
        D_ADC = 25.4 * u.mm
        
        fig, axs = plt.subplots(4, 1, figsize=(8, 10), tight_layout=True)

        fig.suptitle(f"ADC design for {self.g1.name} and {self.g2.name}", size=18)

        # Plot refractive indices of the two glasses
        ax = axs[0]
        
        ax.plot(wls, self.g1.n(wls), label=f"{self.g1.name}")
        ax.plot(wls, self.g2.n(wls), label=f"{self.g2.name}")
            
        ax.set_xticklabels([])
        # ax.ticklabel_format(axis="y", useOffset=1.00000000001)
        ax.set_ylabel("Refractive index $n$")
        ax.legend()

        # Plot scaled difference against the normalised refractivity of air
        ax = axs[1]
        
        ADC_index = self.alpha * (
                            self.g1.n(wls)
                        - self.epsilon * self.g2.n(wls)
                        + self.epsilon - 1
                        )
        
        ADC_index = ADC_index.to(u.degree)
        
        ax.plot(wls, ADC_index, "k--",
            label="2 $\\times$ ("+\
                    f"$n_1 - {self.epsilon:.5g}n_2 {self.epsilon - 1:+.3g}$)"+\
                    f" {self.alpha/2:.3g}$^o$")
        
        d_air = self.beam_comp * self.G_alpha_air * (
                                self.air_model.refractive_index(wls)
                              - self.air_model.refractive_index(self.wl_0)
                                )
        
        d_air = d_air.to(u.degree)
        
        ax.plot(wls, d_air, "r-",
            label="$\\frac{D_{tel}}{D_{ADC}}$ ( $n_{air} - n_{air}$(" \
                +f"{(self.wl_0.to(u.nm)):.0f}) ) $\\times$ "\
                +f"tan({self.z_max:.0f})")
        
        ax.axhline(y=0, color="k", ls="--", lw=1)
        ax.axvline(x=self.wl_0.to(u.um).value, color="k", ls="--", lw=1)
        ax.set_xticklabels([])
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.set_ylabel("Refractivity $\\approx n-1$")
        ax.legend()

        # residuals_curve = (ADC_index - d_air).to(u.mas)
        residuals_curve = self.residuals_function(wls = wls)

        # Plot angular residuals at maximum zenith angle
        ax = axs[2]
        ax.plot(wls, residuals_curve,
            label="Residuals"\
            +"\nRMS: " + f"{self.residuals_RMS:.2f}"\
            +"\nPTV: " + f"{self.residuals_PTV:.2f}")
        
        ax.axhline(y=0, color="k", ls="--", lw=1)
        ax.axvline(x=self.wl_0.to(u.um).value, color="k", ls="--", lw=1)
        ax.set_xlabel("Wavelength (microns)")
        ax.set_ylabel("Angular dispersion\n"\
                     +"residuals at $z_{max}$\n"\
                     +"$\Delta \delta$ (mas)")
        ax.legend()
        
        # Plot cross-sectional layout of physical compound prisms
        ax = axs[3]
        
        ax.axhline(y=0, c="k", ls="--", lw=1)

        sign = self.first_apex_angle / abs(self.first_apex_angle)
        R_ADC = D_ADC / 2

        x1 = D_ADC * np.sin(np.radians(self.first_apex_angle))
        x2 = D_ADC * np.sin(np.radians(self.second_apex_angle))
        if x1 < min_thickness:
            d = (min_thickness.value - x1.value) / 2
        else:
            d = 0

        # Plot first Amici prism
        ax.plot(
            [0, 0],
            [-R_ADC.value, R_ADC.value],
            color = "k", lw = 1.3
            )
        ax.plot(
            [d, d + x1.value],
            [sign * R_ADC.value, -sign * R_ADC.value],
            color = "k", lw = 1.3
            )
        ax.plot(
            [d + x1.value + d, d + d + x2.value],
            [-sign * R_ADC.value, sign * R_ADC.value],
            color = "k", lw = 1.3
            )
        ax.plot(
            [0, d + d + x2.value],
            [sign * R_ADC.value,  sign * R_ADC.value],
            color = "k", lw = 1.3
            )
        ax.plot(
            [0, x1.value + 2 * d],
            [-sign * R_ADC.value, -sign * R_ADC.value],
            color = "k", lw = 1.3
            )

        ax.annotate(
            text = f"{D_ADC:.3g}",
            xy = (-gap.value, R_ADC.value),
            xytext = (-gap.value, 0),
            arrowprops = dict(arrowstyle="->"),
            ha = "center",
            va = "center",
            bbox = dict(boxstyle="round", fc="white", ec="black", pad=0.2)
            )
        ax.annotate(
            text = f"{D_ADC:.3g}",
            xy = (-gap.value, -R_ADC.value),
            xytext = (-gap.value, 0),
            arrowprops = dict(arrowstyle="->"),
            ha = "center",
            va = "center",
            bbox = dict(boxstyle="round", fc="white", ec="black", pad=0.2))


        # Plot second Amici prism
        xoffset = (max(x1, x2) + 2 * d*u.mm + gap).value
        ax.plot(
            [xoffset + 0, xoffset + 0],
            [-R_ADC.value, R_ADC.value],
            color = "k", lw = 1.3
            )
        ax.plot(
            [xoffset + 0 + d, xoffset + x1.value + d],
            [-sign * R_ADC.value, sign * R_ADC.value],
            color = "k", lw = 1.3
            )
        ax.plot(
            [xoffset + d + x1.value + d, xoffset + d + d + x2.value],
            [sign * R_ADC.value, -sign * R_ADC.value],
            color = "k", lw = 1.3
            )
        ax.plot(
            [xoffset + 0, xoffset + x1.value + 2 * d],
            [sign * R_ADC.value, sign * R_ADC.value],
            color = "k", lw = 1.3
            )
        ax.plot(
            [xoffset + 0, xoffset + x2.value + 2 * d],
            [-sign * R_ADC.value, -sign * R_ADC.value],
            color = "k", lw = 1.3
            )

        # Plot annotations of distances
        xtotal = xoffset + (max(x1, x2) + 2 * d*u.mm).value
        ax.annotate(
            text = f"{xtotal:.3g}",
            xy = (0, -R_ADC.value / 2),
            xytext = (xtotal / 2, -R_ADC.value / 2),
            arrowprops = dict(arrowstyle="->"),
            va = "center",
            ha = "center",
            bbox = dict(boxstyle="round", fc="white", ec="black", pad=0.2)
            )
        
        ax.annotate(
            text = f"{xtotal:.3g}",
            xy = (xtotal, -R_ADC.value / 2),
            xytext = (xtotal / 2, -R_ADC.value / 2),
            arrowprops = dict(arrowstyle="->"),
            va = "center",
            ha = "center",
            bbox = dict(boxstyle="round", fc="white", ec="black", pad=0.2)
            )
        
        # For the legend only
        ax.plot(0, 0, lw=0,
            label = "Apex angles:\n"\
                    + f"$\\alpha_1 = ${self.first_apex_angle:.3f}\n"\
                    + f"$\\alpha_2 = ${self.second_apex_angle:.3f}")
        
        #ax.plot(0, 0, label=f"Minimum compound\nprism thickness: "\
            # +"{abs(x1) + abs(max(abs(x1-x2), abs(x2-x1))):.3f}", lw=0)
        
        ax.axis("equal")
        ax.set_xlabel("Optical axis distance (mm)")
        ax.set_ylabel("Lateral axis\ndistance (mm)")
        
        ax.legend()
        
        plt.show()


def _refract(
    n_1: Union[int, float, Callable],
    n_2: Union[int, float, Callable],
    theta_in: Quantity = 0 * u.deg,
    wavelength: Quantity = None,
    debug_level: float = 0.
    ) -> Quantity:

    # Compute the refracted angle of a ray or rays at a single material interface
    # This can handle EITHER an array of input angles at a single wavelength
    # OR a single input angle at a range of wavelengths

    if wavelength is None:
        _n_1 = n_1
        _n_2 = n_2

    else:
        try: _n_1 = n_1(wavelength)
        except TypeError: _n_1 = n_1
        try: _n_2 = n_2(wavelength)
        except TypeError: _n_2 = n_2

    # By this point, _n_1 and _n_2 should be either floats, ints, or lists/arrays of such
    assert not callable(_n_1)
    assert not callable(_n_1)

    # print(f"Refracting at {theta_in.to('degree')} from {_n_1 = } to {_n_2 = }")

    theta_out = np.arcsin((_n_1 / _n_2) * np.sin(theta_in))

    return theta_out.to(u.degree)


@dataclass
class Grating:
    catalogue_number: str = ""
    groove_frequency: Quantity = 31.6 / u.mm
    blaze_angle: Quantity = 76 * u.deg
    gamma_angle: Quantity = 0 * u.deg
    
    min_wl: Quantity = 390 * u.nm
    max_wl: Quantity = 950 * u.nm

    R: float = field(init = False)
    m_times_lambda: Quantity = field(init = False)

    def __post_init__(self):
        self.R = round(np.tan(self.blaze_angle).value, 1)
        self.m_times_lambda = (2 * np.cos(self.gamma) * self.sigma * np.sin(self.blaze_angle)).to(u.um)

    @property
    def gamma(self) -> Quantity: return self.gamma_angle
    
    @property
    def theta_B(self) -> Quantity: return self.blaze_angle

    @property
    def groove_period(self) -> Quantity: return 1 / self.groove_frequency

    @property
    def sigma(self) -> Quantity: return self.groove_period.to(u.um)

    @property
    def gpmm(self) -> Quantity: return self.groove_frequency.to(1 / u.mm)
    
    @property
    def max_order(self): return int((self.m_times_lambda / self.min_wl) - 0.5)
    
    @property
    def min_order(self): return int((self.m_times_lambda / self.max_wl) + 0.5)

    @property
    def order_containing_500nm(self, integer = True) -> int:
        
        if integer:
            return self.order_number(wl = 500 * u.nm, quantised = True)

        return self.order_number(wl = 500 * u.nm)


    def blaze_wavelength(self, m: int | list[int] | ArrayLike) -> Quantity:

        if isinstance(m, list | tuple):
            return [self.blaze_wavelength(im) for im in m]

        return (self.m_times_lambda / m).to(u.um)
    

    def order_number(
        self,
        wl: int | float | Quantity = 500 * u.nm,
        quantised: bool = False,
        ) -> int | list[int] | float | list[float]:
        
        if isinstance(wl, tuple) or (isinstance(wl, Quantity) and  wl.size > 1):
            print(f"Iterating over wavelengths {wl}")
            return [self.order_number(iwl) for iwl in wl]

        if isinstance(wl, int | float):
            print(f"Input was unitless value {wl}. Defaulting to {wl * u.um}")
            wl *= u.um

        if quantised:
            return int(round(float((self.m_times_lambda / wl).si), 0))

        return float((self.m_times_lambda / wl).si)
    
    
    @property
    def diffraction_orders(self) -> list[int]:
        
        min_m = self.order_number(self.max_wl, quantised=True)
        max_m = self.order_number(self.min_wl, quantised=True)
        
        return list(range(min_m, max_m + 1))


    def order_FSR(self, m: int, FSR_fraction: float = 1.) -> Quantity:

        m_wavelength = self.blaze_wavelength(m = m)
        m_plus_one_wavelength = self.blaze_wavelength(m = m+1)

        return FSR_fraction * (m_wavelength - m_plus_one_wavelength).to(u.nm)
    

    def order_FSR_bounds(self, m: int, FSR_fraction: float = 1.) -> Quantity:

        FSR = self.order_FSR(m = m, FSR_fraction = FSR_fraction).to(u.nm).value
        blaze_wavelength = self.blaze_wavelength(m = m).to(u.nm).value

        return (blaze_wavelength - FSR / 2, blaze_wavelength + FSR / 2) * u.nm


    def order_opening_angle(
            self,
            m: int | list[int],
            FSR_fraction: float = 1.,
            return_bounds = False,
            ) -> Quantity:
        
        if isinstance(m, list | tuple):
            return [self.order_opening_angle(im) for im in m]

        FSR_bounds = self.order_FSR_bounds(m = m, FSR_fraction = FSR_fraction)

        # angles = (self.diffraction_angle(m, wl = FSR_bounds[0]),
        #           self.diffraction_angle(m, wl = FSR_bounds[1]))

        angles = self.diffraction_angle(m = m, wl = FSR_bounds)

        if return_bounds:
            return (angles)

        return max(angles) - min(angles)



    def diffraction_angle(
            self,
            m: int,
            wl: Quantity = 550 * u.nm,
            ) -> Quantity:
        
        if isinstance(wl, Quantity) and  wl.size > 1:
            return [self.diffraction_angle(m, iwl) for iwl in wl]
        
        if isinstance(wl, list | tuple):
            return [self.diffraction_angle(iwl) for iwl in wl]


        if isinstance(wl, float):
            print(f"No wavelength units passed ({wl}). Assuming nanometers")
            wl *= u.nm
        
        arcsin_argument = ((m * wl).to(u.um).value\
            / (np.cos(self.gamma) * self.sigma.to(u.um).value))\
                - np.sin(self.blaze_angle)
        return (
            np.arcsin(arcsin_argument) - self.blaze_angle).to(u.deg)
    

    def wavelength_at_angle(self, m: int, angle: Quantity) -> Quantity:

        # angle += self.blaze_angle

        return (np.cos(self.gamma) * self.sigma * (np.sin(self.blaze_angle) + np.sin(angle))) / m


    def order_FSR_bounds_from_angle(self, m: int) -> tuple[Quantity]:

        angle = self.order_opening_angle(m = m) / 2

        return (self.wavelength_at_angle(m = m, angle = self.blaze_angle - angle),
                self.wavelength_at_angle(m = m, angle = self.blaze_angle + angle))
    

    def aspect_ratio(
            self,
            min_wl: Quantity = 385 * u.nm,
            max_wl: Quantity = 950 * u.nm,
            ):

        blue_angle = self.order_opening_angle(m=self.order_number(min_wl)).to(u.deg)
        red_angle = self.order_opening_angle(m=self.order_number(max_wl)).to(u.deg)

        print(f"{blue_angle=}")
        print(f"{red_angle=}")

        return red_angle / blue_angle


# EtalonSim / Etalon.py
@dataclass
class Etalon():
    """Description of plane wave illumination of an etalon a la Wikipedia"""
    
    n: float = 1
    l: Quantity = 6 * u.mm
    θ: Quantity = 0 * u.degree
    R_1: float = 0.95
    
    R_2: float | None = None

    
    def __post_init__(self):
        
        if self.R_2 is None:
            self.R_2 = self.R_1


    @cached_property
    def R(self) -> float: return np.sqrt(self.R_1 * self.R_2)
    
    
    @cached_property
    def T(self):
        return 1 - self.R


    @cached_property
    def F_coef(self):
        return (4 * self.R) / ((1 - self.R)**2)


    @cached_property
    def finesse(self):
        return np.pi / (2 * np.arcsin(1 / np.sqrt(self.F_coef)))


    @cached_property
    def R_max(self):
        """
        Maximum reflectivity, R_max
        """
        return (4 * self.R) / ((1 + self.R) ** 2)


    def delta_lambda(self, λ=780*u.nm):
        """
        Free spectral range of the etalon, defaults to λ = 780nm
        """
        n_g = 1
        return ((λ ** 2) / (2 * n_g * self.l * np.cos(self.θ) + λ)).to("pm")


    def delta(self, λ):
        return ((2 * np.pi / λ) * 2 * self.n * self.l * np.cos(self.θ)).si.value


    def Transmittance(self, λ):
        """
        Transmittance function, sometimes T_e
        Transmitted beam intensity as a function of wavelength
        """
        return 1 / (1 + self.F_coef * (np.sin(self.delta(λ) / 2))**2)


    def intensity(self, λ):
        return self.Transmittance(λ)


    def amplitude(self, m, λ):
        return self.T * (self.R ** m) * np.exp(1j * m * self.delta(λ))


    def summed_amplitudes(self, λ):
        """
        The geometric infinite sum of ampllitudes over m from 0 to infinity
        """
        return self.T / (1 - self.R * np.exp(1j * self.delta(λ)))


    def __str__(self) -> str:

        vals = "".join(
                [
                   f"{p:>12}: {val:<6.3g}\n" for p, val in self.__dict__.items()
                ]
            )

        return "Etalon:\n" + vals


    def __repr__(self) -> str:
        
        return f"Etalon(n={self.n}, l={self.l}, θ={self.θ}, "+\
               f"R_1={self.R_1}, R_2={self.R_2})"




def main() -> None:
    
    ...


if __name__ == "__main__":
    main()
