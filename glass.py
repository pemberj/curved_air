"""
The core of the library - this file contains definitions of Glass and
GlassCatalogue classes, which are used to hold information about optical glasses,
to filter a collection of glasses by certain criteria, to inspect and plot
properties of single glasses or collections of glasses, and to output a filtered
or selected collection to an external file for use in Zemax/OpticStudio

Some code originally forked from
https://github.com/polyanskiy/refractiveindex.info-scripts
agf2yml.py
"""


import numpy as np
from glob import glob
import re
from itertools import combinations

from astropy import units as u

# Type hinting
from typing import TypeVar
SelfGlass = TypeVar("SelfGlass", bound="Glass")
SelfGlassCatalogue = TypeVar("SelfGlassCatalogue", bound="Glass")
from typing import Iterator, Generator
from numpy.typing import ArrayLike
from astropy.units import Quantity

from dataclasses import dataclass, field
import matplotlib.pyplot as plt

# import matplotlib.font_manager as fm
# fm.fontManager.addfont(path="./Quicksand_regular.ttf")

try:
    from curved_air.plot_style import plot_style
    from curved_air.other import _fraunhofer_lines
except ImportError:
    from plot_style import plot_style
    from other import _fraunhofer_lines
plt.style.use(plot_style)


@dataclass
class Glass:
    """
    A class to hold all the parameters for a single type of optical glass
    """

    name: str = ""
    formula: int = -1
    glasscode: float = -1.
    nd: float = -1.
    vd: float = -1.
    status: int = -1 # 1:Standard, 2:Preferred, 3:Special, 4:Obsolete, 5:Melt
    meltfreq: float = -1.

    wlmin: float = -1.
    wlmax: float = -1.

    disp_formula_coefficients: list[float] = field(default_factory = list)
    thermal_disp_coefficients: list[float] = field(default_factory = list)

    cte1: float = -1. #CTE in range -30C - +70C [1E-6/K]
    cte2: float = -1. #CTE in rance +20C - +300C [1E-6/K]
    density: float = -1. # [g/cm^3]
    dpgf: float = -1. #dPgF

    CR: float = -1. # Climatic resistance: 1(high) - 4(low)
    FR: float = -1. # Stain resistance: 0(high) - 5(low)
    SR: float = -1. # Acid resistance: 1(high) - 4(low) and 51-53(very low)
    AR: float = -1. # Alkali resistance: 1(high) - 4(low)
    PR: float = -1. # Phosphate resistance: 1(high) - 4(low)

    wl: list[float] = field(default_factory = list)
    IT: list[float] = field(default_factory = list)
    thickness: list[float] = field(default_factory = list)

    comments: str = ""

    original_file: str = ""
    
    def __post_init__(self) -> None:
        if not self.glasscode:
            self.generate_glasscode()
            
        if self.nd != -1. and self.vd in [-1., 0.]:
            self.vd = (self.n_d - 1) / (self.n_F - self.n_C)
            
        # if self.n_d == -1:
        #     print(f"WARNING! No refractive index provided for glass {self.name}")

    @property
    def Vd(self) -> float: return self.vd

    @property
    def V_d(self) -> float: return self.vd

    @property
    def Abbe_number(self) -> float: return self.vd

    @property
    def n_d(self) -> float: return self.n("d")
    
    @property
    def n_F(self) -> float: return self.n("F")
    
    @property
    def n_C(self) -> float: return self.n("C")
    
    @property
    def WL(self) -> list[float]: return self.wl


    @property
    def alternative_names(self) -> list:
        """
        A short function to generate a list of all possible alternative names
        (by stripping commonly used characters)

        Returns:
            list: a list containing the alternative names of the glass (strings)
        """

        prefixes = ["N-", "S-", "LITHOTEC-"]
        characters = [" ", "*", "ultra"]
        brackets = "[{([\])}]"

        alt_names = [self.name]

        for p in prefixes:
            if p in self.name:
                alt_names.append("".join(self.name.split(p)[1:]))
        for c in characters:
            if c in self.name:
                alt_names.append("".join(self.name.split(c)))
        alt_names.append(re.sub(brackets, "", self.name))

        so_far = alt_names[:]
        # Do it again to catch names with more than one unwanted character/substring
        for alt in so_far:
            for c in characters:
                alt_names.append("".join(alt.split(c)))

        # Convert to set to remove duplicates, remove rempty strings
        return [n for n in list(set(alt_names)) if n != ""]


    @property
    def status_description(self) -> str:
        status_dictionary: dict[int, str] =\
        {
            1: "Standard",
            2: "Preferred",
            3: "Special",
            4: "Obsolete",
            5: "Melt"
        }
        
        try:
            return status_dictionary[self.status]
        except KeyError:
            return "unknown"


    @property
    def rii_formula_number(self) -> int:

        return _dispersion_formula_number(self.formula)


    @property
    def rii_dispersion_coefficients(self) -> list[float]:

        return _dispersion_coefficients(
            AGF_formula_number = self.formula,
            coefficients = self.disp_formula_coefficients
            )


    def generate_glasscode(self):
        if not self.glasscode:
            self.glasscode = int((self.n_d - 1) * 1e3) * 1e3\
                           + int((self.V_d * 10) * 1e3) / 1e3
            return self
        else:
            return self


    def transmission(self, wl: ArrayLike = None) -> tuple[ArrayLike]:

        if isinstance(wl, Quantity):
            wl = wl.to(u.um).value
        elif isinstance(wl, list):
            if isinstance(wl[0], Quantity):
                wl = [_iwl.to(u.um).value for _iwl in wl]

        WL = [w for w in self.wl if w > 0]
        IT = self.IT
        if wl is None:
            wl = WL

        try:
            return np.interp(wl, xp=WL, fp=IT, left=0., right=0.)
        except:
            print(f"Could not compute transmission for {self.name}")
            print(f"{self.wl = }")
            print(f"{self.IT = }")
            return


    def refractive_index(
        self,
        wls: ArrayLike | float | Quantity | str = None,
        ) -> ArrayLike:
        
        if wls is None:
            return self.nd
        
        if isinstance(wls, str):
            try:
                wls = _fraunhofer_lines[wls]
            except KeyError:
                print(f"Unknown wavelength string {wls}. Valid options are"+\
                      f"{', '.join(_fraunhofer_lines.keys())}")
                print("Using default wavelength of 588 nm")
                return self.n_d

        return _refractive_index(
            lam = wls,
            coefficients = self.rii_dispersion_coefficients,
            formula_number = self.formula
            )


    def n(self, wls: ArrayLike | float | Quantity | str = None) -> ArrayLike:

        return self.refractive_index(wls)


    def dn(self, wls: ArrayLike) -> ArrayLike:
        """
        Differential refractive index, a similar metric to Abbe number/dispersion

        Parameters
        ----------
        wl : ArrayLike
            Wavelengths in microns

        Returns
        -------
        ArrayLike
        
        https://www.primeoptics.com.au/Reference/dRPD_description.pdf
        """

        if isinstance(wls, (list, np.ndarray)) and len(wls) == 1:
            wls = wls[0]

        elif isinstance(wls, (int, float)):
            wls = np.array([wls - 1e-10, wls, wls + 1e-10])
            return (np.gradient(self.refractive_index(wls))\
                        / np.gradient(wls))[1]

        return np.gradient(self.refractive_index(wls))\
                        / np.gradient(wls)


    def dV(self, wls: ArrayLike) -> ArrayLike:
        """Differential Abbe number

        Parameters
        ----------
        wls : ArrayLike
            Wavelengths in microns

        Returns
        -------
        ArrayLike
        
        https://www.primeoptics.com.au/Reference/dRPD_description.pdf
        """

        if isinstance(wls, (list, np.ndarray)) and len(wls) == 1:
            wls = wls[0]

        elif isinstance(wls, (int, float)):
            wls = np.array([wls - 1e-10, wls, wls + 1e-10])

            return np.mean((self.refractive_index(wls[1]) - 1)\
                        / (-self.dn(wls)))

        return (self.refractive_index(wls) - 1) / (-self.dn(wls))


    def dRPD(self, wls: ArrayLike = None) -> ArrayLike:
        """[summary]

        Parameters
        ----------
        wl : ArrayLike
            Wavelengths in microns

        Returns
        -------
        ArrayLike
        
        https://www.primeoptics.com.au/Reference/dRPD_description.pdf
        """
        
        if wls is None:
            wls = self.wl

        if isinstance(wls, (list, np.ndarray)) and len(wls) == 1:
            wls = wls[0]

        if isinstance(wls, (int, float)):
            wls = np.array([wls - 1e-10, wls, wls + 1e-10])

        return self.dV(wls - np.gradient(wls)) / self.dV(wls)


    def plot(
        self,
        metric: str = "n",
        wls: ArrayLike = None,
        axis: plt.Axes = None,
        show: bool = True,
        debug_level: float = 0
        ) -> SelfGlass:

        if wls is None:
            wls = np.linspace(0.39, 0.95, 50) * u.um

        metrics = {
            "n": self.n,
            "refractive_index": self.refractive_index,
            "transmission": self.transmission,
            "dn": self.dn,
            "dRPD": self.dRPD,
            "dV": self.dV,
        }

        if metric not in metrics.keys():
            if debug_level:
                print("Unknown metric."\
                   + f"Valid options are {', '.join(metrics.keys())}")
            return self

        try:
            y = metrics[metric](wls)
        except Exception as e:
            print(e)

        if y is None:
            if debug_level:
                print("Unable to compute"\
                   + f"{metric} over {wls} for glass {self.name}")
            return self

        if not axis:
            fig = plt.figure()
            ax = fig.gca()
        else:
            ax = axis
        
        ax.plot(wls, y, label=f"{metric}")
        ax.set_xlabel("Wavelength (um)")
        ax.set_ylabel(f"{metric[0].upper()}{metric[1:]}")

        if show:
            plt.show()

        return self
    
    
    def rename(self, new_name: str) -> SelfGlass:
        self.name = new_name
        return self


    def is_the_same_as(self, other: None) -> bool:
        """
        A function to check whether two glasses are the same as each other.
        Used to determine whether an existing glass in a catalogue is the same
        as one potentially being added.

        Currently, this function only checks the glass name (which is likely
        identical as this is usally what will trigger a function call in the
        first place).

        TODO: Maybe need a function that determines if two glasses are
        negligably close but not identical.

        Parameters
        ----------
        self : Glass
            the glass used to call the method
        other : Glass
            the glass being compared against the current one
        """

        return (self.name == other.name)\
           and (self.n_d == other.n_d)\
           and (self.V_d == other.V_d)


    def to_agf_format(self) -> str:
        try:
            output = ""
            output += f"NM {self.name} {self.formula:.0f} {self.glasscode:.3f} {self.n_d:.5f} {self.V_d:.2f} {self.status:.0f} {self.meltfreq:.0f}\n"
            output += f"GC {self.comments}\n"
            output += f"ED {self.cte1:.6f} {self.cte2:.6f} {self.density:.6f} {self.dpgf:.6f}\n"
            output += f"CD {' '.join([f'{c:.9E}' for c in self.disp_formula_coefficients if c is not None])}\n"
            output += f"TD {' '.join([f'{c:.9E}' for c in self.thermal_disp_coefficients if c is not None])}\n"
            output += f"OD {self.CR:.4f} {self.FR:.4f} {self.SR:.4f} {self.AR:.4f} {self.PR:.4f}\n"
            output += f"LD {self.wlmin:.5E} {self.wlmax:.5E}\n"
            for wl, it, thickness in zip(self.wl, self.IT, self.thickness):
                output += f"IT {wl:.5E} {it:.5E} {thickness:.5E}\n"

            return output

        except TypeError:
            print(f"Errors occurred when attempting to process glass {self.name}")
            return ""


    def summary(self) -> str:
        return "-"*25 + "\n"\
             + f"Original file: {self.original_file}\n"\
             + f"Glass: {self.name}\n"\
             + f"n_d = {self.nd}\n"\
             + f"V_d = {self.vd}\n"\
             + f"Status {self.status}: {self.status_description}"


    def __str__(self) -> str:
        return self.name
        # return f"Original file: {self.original_file}\n"\
        #      + f"Glass: {self.name}\n"\
        #      + f"n_d = {self.nd}\n"\
        #      + f"V_d = {self.vd}\n"\
        #      + f"Status: {self.status}: {self.status_description}\n"\
        #      + "-"*20


class GlassCatalogue(dict):
    """
    * generate plots:
        - Abbe diagram (change reference wavelength)
        - Etc.
    """


    def __init__(self, *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)
        self.glass_list = list(super().keys())


    def __setitem__(self, key, value) -> None:
        self.glass_list.append(key)
        super().__setitem__(key, value)


    def __iter__(self) -> Iterator:
        return iter(list(self.values()))


    def itervalues(self) -> Generator:
        return (self[key] for key in self)


    def replace(self, key: str = None, replacement: Glass = None) -> None:
        if replacement:
            self[key] = replacement
            
            
    @property
    def glasses(self) -> list[Glass]:
        return list(self.values())


    @property
    def number_of_glasses(self) -> int:
        
        return len(self.__dict__)


    def read_AGF(self, agf_filename: str = None) -> SelfGlassCatalogue:

        if isinstance(agf_filename, (list, np.ndarray)):
            for f in agf_filename:
                self.read_AGF(f)
            return self

        original_file = agf_filename.split("/")[-1]

        print(f"Reading file: {original_file:<20}")
        try:
            agf = open(file=agf_filename, mode="r", encoding="utf-8", newline="\r\n")
            # for line in agf:
            #     print(line, end="")
            #     break
        except UnicodeDecodeError:
            agf = open(file=agf_filename, mode="r", encoding="utf-16", newline="\r\n")
            # for line in agf:
            #     print(line, end="")
            #     break

        # print(f"{len(agf.readlines())} lines...")
        
        def _add_glass(g: Glass) -> None:
            
            # print(g.name)
            if not g.name in self.keys():
                self[g.name] = g
            else:
                print(f"A glass with the same name ({g.name})"\
                     + "already exists in the catalogue!")
                if self[g.name].is_the_same_as(g):
                    print(f"The two glasses are identical. Skipping...")
                    
                else:
                    print(f"Glass {g.name} is different from the existing one")
                    
                    print(f"Renaming existing {self[g.name]} to "\
                        + f"{self[g.name].name}_{self[g.name].original_file}...")
                    self[f"{g.name}_{self[g.name].original_file}"]\
                        = self[g.name]\
                            .rename(new_name=f"{self[g.name].name}_{self[g.name].original_file}")
                    del self[g.name]
                        
                    print(f"Adding new glass {g.name} as "\
                        + f"{g.name}_{g.original_file} to the catalogue...")
                    self[f"{g.name}_{g.original_file}"]\
                        = self[g.name] = g.rename(new_name=f"{g.name}_{g.original_file}")
    

        #  Empty variable in which a Glass object will be stored            
        g = None
        
        for _, line in enumerate(agf):
            
            data = [k.strip() for k in line.split(" ")]
            
            if data:
                if data[0] == "CC":
                    # Comment at the start of a catalogue file
                    print(" ".join(data[1:]))

                elif data[0] == "NM":
                    # Found a new "Name" line or reached the end of the file!
                    # Store the previous data in the catalogue
                    if g is not None:
                        _add_glass(g)

                    # Create a new empty glass
                    g = Glass(
                        name = data[1],
                        original_file = original_file
                        )

                    try: g.formula = int(float(data[2]))
                    except ValueError: ...

                    try: g.nd = float(data[4])
                    except ValueError: g.nd = -1.
                        
                    if data[5] != "":
                        try: g.vd = float(data[5])
                        except ValueError: ...
                    else:
                        try: g.vd = float(data[6])
                        except ValueError: ...

                    try:
                        glasscode = float(data[3])
                        if glasscode in [-1., 0., 1.]: g.generate_glasscode()
                        else: g.glasscode = glasscode
                    except ValueError:
                        g.generate_glasscode()

                    if len(data) >= 8:
                        try: g.status = int(float(data[7][0]))
                        except ValueError: ...

                    if len(data) >= 9:# and data.count("-") == 0:
                        try: g.meltfreq = float(data[8])
                        except ValueError: ...
                    # else: g.meltfreq = 0

                    continue

                elif data[0] == "LD":

                    try: g.wlmin = float(data[1])
                    except ValueError: ...

                    try: g.wlmax = float(data[2])
                    except ValueError: ...

                    continue

                elif data[0] == "CD":
                    try:
                        # Ignoring empty strings - may need to change?
                        # Or at least check the length of the list if computing
                        # something from these coefficients

                        # Unused coefficients SHOULD be reported as zero
                        g.disp_formula_coefficients\
                            = [float(d) for d in data[1:] if d != ""]
                    except ValueError: ...

                    continue

                elif data[0] == "TD":
                    try:
                        # Ignoring empty strings - may need to change?
                        # Or at least check the length of the list if computing
                        # something from these coefficients
                        g.thermal_disp_coefficients\
                            = [float(d) for d in data[1:] if d != ""]
                    except ValueError: ...

                    continue

                elif data[0] == "ED":
                    try: g.cte1 = float(data[1])
                    except ValueError: ...

                    try: g.cte2 = float(data[2])
                    except ValueError: ...

                    try: g.density = float(data[3])
                    except ValueError: ...

                    try: g.dpgf = float(data[4])
                    except ValueError: ...

                    continue

                elif data[0] == "OD":
                    #? = data[1]
                    if len(data)>2:
                        try:
                            g.CR = float(data[2])
                        except ValueError:
                            if g.CR == "3-4":
                                g.CR = 3.5

                        try: g.FR = int(float(data[3]))
                        except ValueError: ...

                        try: g.SR = float(data[4])
                        except ValueError: ...

                        try: g.AR = float(data[5])
                        except ValueError: ...

                        try: g.PR = float(data[6])
                        except ValueError: ...

                    continue

                elif data[0]=="IT" and data[1]!="" and data[2]!="" and data[3]!="":
                    g.wl.append(float(data[1]))
                    g.IT.append(float(data[2]))
                    g.thickness.append(float(data[3]))
                    continue

                elif data[0] == "GC":
                    g.comments = " ".join([d for d in data[1:] if d != ""])
                    continue
                
            else: # data is empty
            # In a normal AGF file, there are no empty lines, so this is likely
            # the end of the file
                if g is not None:
                    _add_glass(g)

        return self


    def write_AGF(self, filename: str = "./output.agf") -> SelfGlassCatalogue:
        """
        Maybe define an external function or even just a dictionary to generate
        the required line-structure for the AGF file. I just want to do what
        `catalogues.read_AGF` is doing, but in reverse.

        Come to think of it, maybe `read_AGF` belongs in this class? Workflow
        would be: create empty GlassCatalogue(), call GC.read_AGF() to populate,
        OR: create GlassCatalogue(agf_files=["./schott.agf", "./ohara.agf"]) to
        initialise the GC with these to begin with. That might be nice!
        """

        if not filename:
            print("No target filename specified. Printing AGF output...")
            for _, glass in self.items():
                print(glass.to_agf_format(), end="")
            return self

        with open(filename, "w") as f:
            print(f"Writing to {filename}...")
            for _, glass in self.items():
                f.write(glass.to_agf_format())

        return self


    def find(
        self, name: str = None, case_sensitive: bool = False,
        n: float = None, dn: float = 0.01, n_wavelength: float = 0.588,
        V_d: float = None, dV_d: float = 1.,
            ) -> SelfGlassCatalogue:
        """
        A simple method to search inside a GlassCatalogue

        This function will find anything matching an input name, and anything
        with refractive index (at a given wavelength) or Abbe number matching
        within a particular range.

        N-BK7 defaults: wavelength=0.588, n=1.5168, dn=0.01, V=64.17, dV=1.

        TODO:
        * RegEx (regular expression) name search (matching with wildcards, etc.)

        Parameters
        ----------
        name : str, optional
            [description], by default None
        case_sensitive : bool, optional
            [description], by default False
        n : float, optional
            [description], by default None
        dn : float, optional
            [description], by default 0.01
        n_wavelength : float, optional
            [description], by default 0.588
        V_d : float, optional
            [description], by default None
        dV_d : float, optional
            [description], by default 1.

        Returns
        -------
        GlassCatalogue
            Returns a new GlassCatalogue containing only the glasses that match
            the search criteria
        """

        # Create an empty GlassCatalogue to store our results in
        results = GlassCatalogue()

        for _, glass in self.items():
            if name:
                if not case_sensitive: name = name.lower()
                for possible_match in glass.alternative_names:
                    if not case_sensitive: possible_match = possible_match.lower()
                    if name == possible_match or name in possible_match:
                        results[glass.name] = glass
                        continue

            if n:
                if not dn: dn = 0.01
                if abs(n - glass.n(n_wavelength)) < dn:
                    results[glass.name] = glass
                    continue

            if V_d:
                if not dV_d: dV_d = 1.
                if abs(V_d - glass.V_d) < dV_d:
                    results[glass.name] = glass
                    continue

        return results


    def filter(
        self,
        
        # Allowable wavelengths (tied to transmission)
        min_wl: Quantity = 380 * u.nm,
        max_wl: Quantity = 950 * u.nm,
        
        # Refractive index
        min_n: float = None,
        max_n: float = None,
        n_wavelength: Quantity = "d",
        
        # Abbe number
        min_V_d: float = None,
        max_V_d: float = None,
        
        # Glass status
        status_code: list[int | float] = None,
        status_description: list[str] = None,
        
        # Transsmission
        min_transmission: float = None,
        max_transmission: float = None,
        min_average_transmission: float = None,
        
        # CTE
        min_CTE: float = None,
        max_CTE: float = None,

            ) -> SelfGlassCatalogue:
            

        """
        For now, a very simple filtering method, providing a way to return
        only those glasses that match particular cutoff criteria based on
        values of n and V_d.

        TODO:
        * Filter by transmission (minimum value at a particular wavelength,
        minimum value anywhere in a range, or mean/median over a range higher
        than a particular value)
        * Filter by CTE and other physical properties (environmental resistance?)
        * Filter by more sophisticated glass properties (partial dispersions,
        dRPD, etc.)
        """

        if not (min_n or max_n or min_V_d or max_V_d\
                or status_code or status_description\
                or min_transmission or max_transmission\
                or min_average_transmission\
                or min_CTE or max_CTE):
            print("No filter criteria provided.")
            return self

        # Create an empty GlassCatalogue to store our results in
        results = GlassCatalogue()

        for _, glass in self.items():
            
            if min_wl is not None and max_wl is not None:
                try:
                    if glass.wlmin > min_wl.to("um").value\
                    or glass.wlmax < max_wl.to("um").value:
                        continue
                except TypeError: ...

            if status_code:
                try:
                    if isinstance(status_code, int)\
                                              or isinstance(status_code, float):
                        if glass.status != status_code:
                            continue
                    else: # It is a list or ArrayLike
                        if glass.status not in status_code:
                            continue
                except TypeError: ...

            if status_description:
                try:
                    if isinstance(status_description, str):
                        if status_description.lower()\
                                        not in glass.status_description.lower():
                            continue
                    else: # It is list or ArrayLike
                        status_description\
                                     = [sd.lower() for sd in status_description]
                        for sd in status_description:
                            if sd not in glass.status_description.lower():
                                continue
                except TypeError: ...

            if min_n:
                try:
                    if glass.n(n_wavelength) < min_n:
                        print(f"{glass.n(n_wavelength) = }, skipping...")
                        continue
                except TypeError: ...
            
            if max_n:
                try:
                    if glass.n(n_wavelength) > max_n:
                        continue
                except TypeError: ...

            if min_V_d and not max_V_d:
                try:
                    if glass.V_d < min_V_d:
                        continue
                except TypeError: ...
            
            if max_V_d and not min_V_d:
                try:
                    if glass.V_d > max_V_d:
                        continue
                except TypeError: ...

            if min_transmission or max_transmission or min_average_transmission:
                try:
                    assert (min_wl is not None) and (max_wl is not None)
                except AssertionError: ...

                transmission_wls = np.linspace(min_wl.to("um").value,
                                               max_wl.to("um").value, 20)
                transmission = glass.transmission(wl=transmission_wls)

            if min_transmission:
                try:
                    if min(transmission) < min_transmission:
                        continue
                except TypeError: ...

            if max_transmission:
                try:
                    if max(glass.IT[min_wl.to("um").value < glass.wl\
                                   < max_wl.to("um").value]) > max_transmission:
                        continue
                except TypeError: ...

            if min_average_transmission:
                try:
                    if np.mean(transmission) < min_average_transmission:
                        continue
                except TypeError: ...

            # If it hasn't violated any of the condition yet...
            results[glass.name] = glass

        print(f"Filter returned {len(results)} / {len(self.glass_list)}")
        return results


    def plot_Abbe_diagram(
        self,
        axis: plt.Axes | None = None,
        interactive: bool = True,
        ) -> SelfGlassCatalogue:
        """
        TODO:
        - [ ] Add interactivity to the plot, with hover-over labels
        """
        

        if not axis:
            fig = plt.figure()
            ax = fig.gca()
        else: ax = axis
        
        names = [g.name for g in self.glasses]

        # for _, glass in self.items():
        points = ax.scatter(
            [g.Abbe_number for g in self.glasses],
            [g.n_d for g in self.glasses],
            s=50,
            alpha=0.5
            )

        ax.set_xlim(ax.get_xlim()[::-1])
        
        if interactive:
            # code from here: 
            # https://stackoverflow.com/questions/7908636/how-to-add-hovering-annotations-to-a-plot
            annot = ax.annotate(
                "",
                xy=(0, 0),
                xytext=(20, 20),
                textcoords="offset points",
                bbox=dict(boxstyle="round", fc="w"),
                arrowprops=dict(arrowstyle="->")
                )
            annot.set_visible(False)

            def update_annot(ind):
                
                pos = points.get_offsets()[ind["ind"][0]]
                annot.xy = pos
                text = f"{names[ind['ind'][0]]}"
                # text = "{}, {}".format(" ".join(list(map(str,ind["ind"]))), 
                #                     " ".join([names[n] for n in ind["ind"]]))
                annot.set_text(text)
                annot.get_bbox_patch().set_alpha(0.4)
                

            def hover(event):
                vis = annot.get_visible()
                if event.inaxes == ax:
                    cont, ind = points.contains(event)
                    if cont:
                        update_annot(ind)
                        annot.set_visible(True)
                        fig.canvas.draw_idle()
                    else:
                        if vis:
                            annot.set_visible(False)
                            fig.canvas.draw_idle()

            fig.canvas.mpl_connect("motion_notify_event", hover)

        plt.show()
        
        return self


    def plot_transmission(
        self, axis: plt.Axes | None = None,
        wls: ArrayLike | Quantity | None = None,
        ) -> SelfGlassCatalogue:

        if not axis:
            fig = plt.figure()
            ax = fig.gca()
        else: ax = axis

        for _, glass in self.items():
            glass.plot(wls=wls, metric="transmission", axis=ax, show=False)
            
        ax.set_xlabel("Wavelength (microns)")
        ax.set_ylabel("Transmission")

        plt.show()
        
        return self
    
    
    def compute_ADCs(
        self,
        min_wl: Quantity = 380 * u.nm,
        max_wl: Quantity = 950 * u.nm,
        num_samples: int = 20,
        max_residuals: Quantity = 100 * u.mas,
        ):
        """
        Optimise ADCs for all possible combinations of glasses in the catalogue
        """
        
        try:
            from curved_air.optics import PrismStack, ADC
        except ImportError:
            from optics import PrismStack, ADC
        
        results = []
        
        columns = [
            "Glass 1",
            "Glass 2",
            "Apex angle",
            "Zero-deviation wavelength",
            "PTV Residuals"
            ]
        
        print(
            f"{columns[0]:<12}\t"+\
            f"{columns[1]:<12}\t"+\
            f"{columns[2]:<12}\t"+\
            f"{columns[3]:<24}\t"+\
            f"{columns[4]:<12}"
            )
        
        for g1, g2 in combinations(self.glasses, 2):
            
            adc = ADC(
                prism_stack = PrismStack([g1, g2]),
                min_wl = min_wl,
                max_wl = max_wl,
                D_telescope = 0.8 * u.m,
                D_ADC = 15 * u.mm,
                z_max = 65 * u.degree,
                criterion = "PTV",
                )
            
            adc.optimise(num_samples = num_samples)
            
            if adc.residuals_PTV <= max_residuals:
            
                results.append(adc)
                print(
                    f"{g1.name:<12}\t"+\
                    f"{g2.name:<12}\t"+\
                    f'{f"{adc.alpha:.5g}":<12}\t'+\
                    f'{f"{adc.wl_0:.5g}":<24}\t'+\
                    f'{f"{adc.residuals_PTV:.5g}":<12}\t'
                    )
                
        return results


    def __str__(self) -> str:
        return "\n".join([f.summary() for f in self.values()])


    def __repr__(self) -> str:
        return "\n".join([f.summary() for f in self.values()])


def _dispersion_formula_number(AGF_formula_number: int) -> int:
    """
    Converts between AGF formula numbers and refractiveindex.INFO formula
    numbers. The two systems use different conventions for numbering.
    """
    
    if AGF_formula_number in [1, 11, 12, 13]:
        return 3
    
    elif AGF_formula_number in [2, 6]:
        return 2

    elif AGF_formula_number == 3:
        return 4

    else:
        raise ValueError
        # return 0


def _dispersion_coefficients(
    AGF_formula_number: int,
    coefficients: list[float]
    ) -> list[float]:
    """
    Adapted from the refractiveindex.INFO database-generation scripts. I follow
    the same conventions in formula numbering as that database.
    """

    if AGF_formula_number == 1: # AGF formula number
        # rii_formula_number = 3
        coeff_list = [None, 2, -2, -4, -6, -8]
        rii_formula_coefficients = []
        for i, k in zip(coefficients, coeff_list[:len(coefficients)]):
            rii_formula_coefficients.append(i)
            if k is not None:
                rii_formula_coefficients.append(k)
                    
    elif AGF_formula_number in [2, 6]:
        # rii_formula_number = 2
        rii_formula_coefficients = [0] + [c for c in coefficients]
                
    elif AGF_formula_number == 3:
        # rii_formula_number = 4
        c = coefficients
        rii_formula_coefficients = [c[0]+1, c[1], 2, c[2], 2, c[3], 0, c[4], 2]

    elif AGF_formula_number == 11:
        # rii_formula_number = 3
        coeff_list = [None, 2, -2, -4, -6, -8, -10, -12]
        rii_formula_coefficients = []
        for c, k in zip(coefficients, coeff_list):
            rii_formula_coefficients.append(c)
            if k is not None:
                rii_formula_coefficients.append(k)

    elif AGF_formula_number == 12:
        # rii_formula_number = 3
        coeff_list = [None, 2, -2, -4, -6, -8, 4, 6]
        rii_formula_coefficients = []
        for c, k in zip(coefficients, coeff_list):
            rii_formula_coefficients.append(c)
            if k is not None:
                rii_formula_coefficients.append(k)
    
    elif AGF_formula_number == 13:
        # rii_formula_number = 3
        coeff_list = [None, 2, 4, -2, -4, -6, -8, -10, -12]
        rii_formula_coefficients = []
        for c, k in zip(coefficients, coeff_list):
            rii_formula_coefficients.append(c)
            if k is not None:
                rii_formula_coefficients.append(k)

    else:
        rii_formula_coefficients = []

    return rii_formula_coefficients


def _refractive_index(
    lam: ArrayLike,
    coefficients: list[float],
    formula_number: int
    ) -> ArrayLike:
    """A function to compute the wavelength-dependent refractive index of a
    material based on coefficients
    
    This uses the formula numbering convention of the refractiveindex.INFO
    database: 1 is the Sellmeier formula with squared coefficients in the
    denominators, 2 is the Sellmeier formula without squares in the
    denominators, and 3 is a polynomial formula used by some glass manufacturers

    Args:
        wavelengths   (list): [description]
        coefficients  (list): [description]
        formula_number (int): [description]

    Returns:
        list: [description]
    """

    try:
        lam = lam.to("um")
    except AttributeError:
        ...

    lam = np.array(lam)

    C = []
    C.append(0) # Set C[0] as padding, for parity in numbering
    for c in coefficients:
        C.append(c) # Set C[1] through C[len(coefficients)]
    for i in range(18 - len(C)):
        C.append(0) # Pad C to a length of 18
    
    if formula_number == 1:
        return np.sqrt(1 + C[1] + (C[2] * lam**2) / (lam**2 - C[3]**2)
                                + (C[4] * lam**2) / (lam**2 - C[5]**2)
                                + (C[6] * lam**2) / (lam**2 - C[7]**2)
                                + (C[8] * lam**2) / (lam**2 - C[9]**2)
                                + (C[10] * lam**2) / (lam**2 - C[11]**2)
                                + (C[12] * lam**2) / (lam**2 - C[13]**2)
                                + (C[14] * lam**2) / (lam**2 - C[15]**2)
                                + (C[16] * lam**2) / (lam**2 - C[17]**2))
    
    if formula_number == 2:
        return np.sqrt(1 + C[1] + (C[2] * lam**2) / (lam**2 - C[3])
                                + (C[4] * lam**2) / (lam**2 - C[5])
                                + (C[6] * lam**2) / (lam**2 - C[7])
                                + (C[8] * lam**2) / (lam**2 - C[9])
                                + (C[10] * lam**2) / (lam**2 - C[11])
                                + (C[12] * lam**2) / (lam**2 - C[13])
                                + (C[14] * lam**2) / (lam**2 - C[15])
                                + (C[16] * lam**2) / (lam**2 - C[17]))
    
    if formula_number == 3:
        return np.sqrt(C[1] + C[2] * lam ** C[3]
                            + C[4] * lam ** C[5]
                            + C[6] * lam ** C[7]
                            + C[8] * lam ** C[9]
                            + C[10] * lam ** C[11]
                            + C[12] * lam ** C[13]
                            + C[14] * lam ** C[15]
                            + C[16] * lam ** C[17])



def main() -> None:
    
    agf_files = sorted(
            [
                f for f in glob("./curved_air/AGFs/*.[Aa][Gg][Ff]")\
                                                if re.search("ohara|schott", f)
            ]
        )

    catalogue = GlassCatalogue()\
        .read_AGF(agf_filename = agf_files)\
        .filter(
            min_wl=380 * u.nm,
            max_wl=950 * u.nm,
            min_transmission = 0.97,
            min_average_transmission = 0.99,
            
            )
    
    # catalogue.plot_Abbe_diagram()
    
    catalogue.compute_ADCs(max_residuals=25 * u.mas)


if __name__ == "__main__":
    main()
