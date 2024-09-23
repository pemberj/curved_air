# EtalonSim / GaussianBeam.py

"""
A simple implementation of Gaussian Beams using Python.

Dependencies: numpy, astropy (for units), matplotlib (for plots)
"""

import numpy as np
import astropy.units as u


class GaussianBeam():
    """
    """


    def __init__(self, λ=633*u.nm, E_0=1, w_0=10*u.um, n=1, debug_level=0):
        """
        Args:
            E_0 (int, optional): [description]. Defaults to 1.
            w_0 ([type], optional): [description]. Defaults to 10*u.um.
            n (int, optional): [description]. Defaults to 1.
            debug_level (int, optional): [description]. Defaults to 0.
        """

        self.debug_level = debug_level

        self.λ = λ.si
        self.E_0 = E_0
        self.w_0 = w_0.si
        self.n = n

        if debug_level > 0:
            for p, val in self.__dict__.items():
                print(f"{p}: {val}")

    @property
    def k(self):
        """
        k, wavenumber
        """

        return ((2 * np.pi * self.n) / self.λ).si

    @property
    def z_R(self):
        """
        z_R, Rayleigh range, sometimes z_0
        (not to be confused with a distance to the beam waist)
        """

        return ((np.pi * self.n * self.w_0**2) / self.λ).si


    def q(self, z=0*u.mm):
        """
        q, complex beam parameter
        """

        return z + 1j * self.z_R


    def w_squared(self, z=0*u.mm):
        """
        Square of the beam width
        """

        _z = (z / self.z_R).si

        if self.debug_level > 0:
            print(f"z / self.z_0: {_z}")
            print(f"w(z)^2: {(self.w_0**2 * (1 + (_z)**2)).si}")

        return (self.w_0**2 * (1 + (_z)**2)).si


    def w(self, z=0*u.mm):
        """
        w, beam width
        Defined as the radius at which the field amplitudes fall to 1/e of their axial values
        (i.e., where the intensity values fall to 1/e2 of their axial values)
        """

        if self.debug_level > 0:
            print(f"w(z): {np.sqrt(self.w_squared(z)).si}")

        return np.sqrt(self.w_squared(z)).si


    def R(self, z=0*u.mm):
        """
        R, radius of curvature of the wavefront
        """

        _z = (self.z_R / z).si

        if self.debug_level > 0:
            print(f"z_0 / z: {_z}")
            print(f"R(z): {(z * (1 + (_z)**2)).si}")

        R = (z * (1 + (_z)**2)).si
        # Account for infinite radius of curvature at beam waist
        R[np.isnan(R)] = np.inf * u.m

        return R


    def curvature(self, z=0*u.mm):
        """
        Curvature of the wavefront at a distance z from the beam waist
        """

        return 1 / self.R(z)
    

    def sag(self, z=0*u.mm, D=4*u.mm):
        """
        Returns the sag (sagitta) between the wavefront at a distance z from the beam waist and at
        a diameter D from the propagation axis

        Modified definition to avoid a singularity at z=0, infinite radius
        """

        R = self.R(z)

        return R - np.sign(R) * np.sqrt(R**2 - (D/2)**2)


    def amplitude(self, r=None, x=0*u.mm, y=0*u.mm, z=0*u.mm):
        """
        The amplitude of the electric field at the given coordinates
        """
        if r is None:
            r = np.sqrt(x**2 + y**2).si
        else:
            r = r.si

        return self.E_0 * (self.w_0 / self.w(z)).si * np.exp((-r**2 / (self.w_squared(z))).si)


    def Gouy_phase(self, z=0*u.mm):
        """
        The Gouy phase at a distance z from the beam waist
        """

        if self.debug_level > 0:
            print(f"Gouy phase, arctan(z/z_0): {np.arctan((z / self.z_R).si)}")

        return np.arctan((z / self.z_R).si) / u.radian


    def longitudinal_phase(self, z=0*u.mm):
        """
        The longitudinal phase of the E field at a distance z from the beam waist
        """

        return (self.k * z).si


    def radial_phase(self, r=None, x=0*u.mm, y=0*u.mm, z=0*u.mm):
        """
        The radial phase of the E field at the given coordinates
        """
        if r is None:
            r = np.sqrt(x**2 + y**2)
        else:
            r = r.si

        return (self.k.si * (r**2).si) / (2 * self.R(z).si)


    def E_field(self, x=0*u.mm, y=0*u.mm, z=0*u.mm):
        """
        Returns the complex E field of the wave, encoding amplitude and phase, at the given coordinates
        """

        return self.amplitude(x, y, z) * np.exp(-1j * (self.longitudinal_phase(z)\
                                                    + self.radial_phase(x, y, z)\
                                                    - self.Gouy_phase(z)))


    def E_field_partial(self, r=None, x=0*u.mm, y=0*u.mm, z=0*u.mm, include="amplitude longitudinal"):
        """
        A function to construct partial Gaussian beams, for instance including only Gouy phase or
        only radial phase, for inspection of the contributions of each phase component, while taking
        in to account the amplitude variations of these contributions.

        Args:
            r, radius from the optical axis in units of length. Defaults to None.
            x ([type], optional): [description]. Defaults to 0*u.mm.
            y ([type], optional): [description]. Defaults to 0*u.mm.
            z ([type], optional): [description]. Defaults to 0*u.mm.
            include (str, optional): [description]. Defaults to "amplitude longitudinal".
        """

        if r is None:
            r = np.sqrt(x**2 + y**2)
        else:
            r = r.si

        E = 1.

        if "amplitude" in include.lower():
            # print("Computing amplitude")
            E = E * self.amplitude(r=r, z=z)

        if "longitudinal" in include.lower():
            # print("Computing longitudinal phase")
            E = E * np.exp(-1j * self.longitudinal_phase(z=z))
            
        if "radial" in include.lower():
            # print("Computing radial phase")
            E = E * np.exp(-1j * self.radial_phase(r=r, z=z))

        if "gouy" in include.lower():
            # print("Computing Gouy phase")
            E = E * np.exp(+1j * self.Gouy_phase(z=z))

        return E


    def E_gouy_integrated(self, r=np.inf, z=0*u.mm):
        """
        A function that returns the E-field incorporating only the Gouy phase component integrated
        from r=0 to the given r in the plane a given distance z from the beam waist.
        """

        ret = self.E_0 * self.w_0 * self.w(z=z) * np.pi * np.exp(1j * self.Gouy_phase(z=z))

        if np.isinf(r):
            return ret

        else:
            exponent = -((r**2) / (self.w(z)**2)).si
            ret *= (1 - np.exp(exponent))
            return ret


    def E_radial_integrated(self, r=np.inf, z=0*u.mm):
        """
        A function that returns the E-field incorporating only the radial phase component integrated
        from r=0 to the given radius r in the plane a given distance z from the beam waist.
        """

        ret = self.E_0 * (self.w_0 / self.w(z=z)) * np.pi\
            / ((1 / (self.w(z=z)**2)) + ((1j * self.k) / (2 * self.R(z=z))))

        if np.isinf(r):
            return ret

        else:
            exponent1 = -r**2 * (1 / self.w(z=z)**2).si
            exponent2 = -r**2 * ((1j * self.k) / (2 * self.R(z=z))).si
            exponent = exponent1 + exponent2
            ret *= (1 - np.exp(exponent))
            return ret


    def summed_E_field(self, z=0*u.mm, exclude="none", debug_level=0):
        """
        For a beam symmetric about the optical axis. Integrating over r from 0 to infinity gives the
        summed (inegrated) E field across a plane at z. Used for calculating the interference of
        full waves in devices like a Fabry-Perot etalon.
        """

        A = ((u.m**2 / self.w(z)**2) + 1j*((self.k * u.m**2) / (2 * self.R(z))))
        integral_over_r = np.pi / A

        # baseline result, including only the amplitude and longitudinal phase, ie. a plane wave beam
        ret = self.E_0 * (self.w_0 / self.w(z)) * np.exp(-1j * self.longitudinal_phase(z))

        try:
            if "gouy phase" not in exclude.lower():
                if debug_level > 0: print("Including Gouy phase")
                ret *= np.exp(1j * self.Gouy_phase(z))

            else: print("Returning results without Gouy phase...")

            if "radial" not in exclude.lower():
                if debug_level > 0: print("Including radial effects (wavefront curvature, intensity variations)")
                ret *= integral_over_r

            else: print("Returning results with no radial effects...")

            return ret.si

        except Exception as e:
            print(e)
            print("Assuming full Gaussian beam...")
            
            return (self.E_0 * (self.w_0 / self.w(z))\
                             * np.exp(-1j * self.longitudinal_phase(z))\
                             * np.exp(1j * self.Gouy_phase(z))\
                             * integral_over_r).si
    
    
# EtalonSim / PlanarBeam.py
    
import numpy as np
import astropy.units as u
import pprint
pp = pprint.PrettyPrinter(indent=4)

class PlanarBeam():
    """
    An implementation of a very simple plane electromagnetic wave
    """

    def __init__(self, λ=633*u.nm, E_0=1, n=1, plane_normal=(0,0,1), debug_level=0):

        self.λ = λ
        self.E_0 = E_0
        self.n = n
        self.plane_normal = plane_normal
        self.debug_level = debug_level
        
        self.k_scalar = (2* np.pi / λ) * np.linalg.norm(self.plane_normal)
        self.k_vector = self.k_scalar * self.plane_normal

        if debug_level > 0:
            pp.pprint(self.__dict__)


    def E_field(self, x=0*u.mm, y=0*u.mm, z=0*u.mm):
        """
        https://en.wikipedia.org/wiki/Electromagnetic_wave_equation
        """

        x, y, z = x.to("mm").value, y.to("mm").value, z.to("mm").value
        
        # Position vector r
        r_unitless = np.array([x, y, z])
        # r = r_unitless * u.mm
        k_vector_unitless = self.k_vector.to("1 / mm").value

        return self.E_0 * np.exp(-1j * np.dot(k_vector_unitless, r_unitless))
   
    
# EtalonSim / SphericalBeam.py

import numpy as np
import astropy.units as u
import pprint
pp = pprint.PrettyPrinter(indent=4)

class SphericalBeam():
    """
    An implementation of a very simple spherical electromagnetic wave
    """

    def __init__(self, λ=633*u.nm, 
                       E_0=1,
                       n=1,
                       plane_normal=(0,0,1),
                       opening_angle=np.inf * u.deg,
                       debug_level=0):

        self.λ = λ
        self.E_0 = E_0
        self.n = n
        self.plane_normal = plane_normal
        self.opening_angle = opening_angle
        self.debug_level = debug_level
        
        self.k_scalar = (2* np.pi / λ) * np.linalg.norm(self.plane_normal)
        self.k_vector = self.k_scalar * self.plane_normal

        if debug_level > 0:
            pp.pprint(self.__dict__)


    def E_field(self, x=0*u.mm, y=0*u.mm, z=0*u.mm):
        """
        https://en.wikipedia.org/wiki/Electromagnetic_wave_equation
        """

        x, y, z = x.to("mm").value, y.to("mm").value, z.to("mm").value

        mask = np.arctan((np.sqrt(np.sum(np.array([x, y])**2, axis=0)))/z)\
                                    >= self.opening_angle.to("radian").value
        
        # length of vectors x,y,z
        rho_unitless = np.sqrt(np.sum(np.array([x, y, z])**2, axis=0))
        k_scalar_unitless = self.k_scalar.to("1/mm").value

        return np.ma.array(self.E_0 * np.exp(-1j * k_scalar_unitless * rho_unitless),
                           mask=mask,
                           fill_value=0.).filled() # .filled() method
                           # returns normal array, filling in the masked values!



    def E_field_array(self, Xs=[0] * u.mm, Ys=[0] * u.mm, Zs=[0] * u.mm):
        """
        https://en.wikipedia.org/wiki/Electromagnetic_wave_equation
        """

        Xs, Ys, Zs = Xs.to("mm").value, Ys.to("mm").value, Zs.to("mm").value

        coords = np.vstack(np.meshgrid(Xs, Ys, Zs)).reshape(3,-1).T

        result = [self.E_field(x=x*u.mm, y=y*u.mm, z=z*u.mm) for x, y, z in coords]

        return(result)


    def E_field_summed_preintegrated(self, z=0*u.mm):
        """
        Preintegrate, exploiting axial symmetry
        """

        raise NotImplementedError
        return






def main() -> None:
    
    ...



if __name__ == "__main__":

    main()
