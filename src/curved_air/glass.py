"""
curved_air

glass

Contains classes Glass and GlassCatalogue, which are used to hold information about
optical glasses, to filter a collection of glasses by certain criteria, to inspect
and plot properties of single glasses or collections of glasses, and to output
a filtered or selected collection to an external file for use in Zemax/OpticStudio
and other optical design software.

The Glass class contains all the parameters for a single type of optical glass,
including refractive index, Abbe number, and environmental resistance properties.
The GlassCatalogue class is a dictionary-like object that holds multiple Glass objects
and provides methods for filtering and searching through them. The classes also provide
methods for reading and writing catalogues in AGF format (suitable for working in
Zemax/OpticStudio), as well as for plotting various properties of the glasses.

Some code originally forked from
https://github.com/polyanskiy/refractiveindex.info-scripts
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, overload

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.units import Quantity, UnitConversionError
from tqdm import tqdm

from curved_air.other import Air, Model, _fraunhofer_lines
from curved_air.plotting import plot_style

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterator
    from typing import Any  # noqa: F401

    from matplotlib.backend_bases import MouseEvent
    from numpy.typing import ArrayLike, NDArray

    from curved_air.type_hinting import ArrayQuantity, ScalarQuantity

plt.style.use(plot_style)


status_dictionary: dict[int, str] = {
    1: "Standard",
    2: "Preferred",
    3: "Special",
    4: "Obsolete",
    5: "Melt",
}

dCF: dict[str, ScalarQuantity] = {
    "d": 587.5618 * u.nm,
    "C": 656.281 * u.nm,
    "F": 486.134 * u.nm,
}

GLASS_CATALOGUES: dict[str, Path] = {
    p.stem: p for p in sorted((Path(__file__).parent / "agfs").glob("*.[Aa][Gg][Ff]"))
}


@dataclass
class Glass:
    """
    A class to hold all the parameters for a single type of optical glass
    """

    name: str = "_default_"
    manufacturer: str = ""
    formula: int = -1
    glasscode: float = -1.0
    nd: float = -1.0
    vd: float = -1.0
    status: int = -1  # 1:Standard, 2:Preferred, 3:Special, 4:Obsolete, 5:Melt
    meltfreq: float = -1.0

    wlmin: float = -1.0
    wlmax: float = -1.0

    disp_formula_coefficients: list[float] = field(default_factory=list)
    thermal_disp_coefficients: list[float] = field(default_factory=list)

    cte1: float = -1.0  # CTE in range -30C - +70C [1E-6/K]
    cte2: float = -1.0  # CTE in rance +20C - +300C [1E-6/K]
    density: float = -1.0  # [g/cm^3]
    dpgf: float = -1.0  # dPgF

    CR: float = -1.0  # Climatic resistance: 1(high) - 4(low)
    FR: float = -1.0  # Stain resistance: 0(high) - 5(low)
    SR: float = -1.0  # Acid resistance: 1(high) - 4(low) and 51-53(very low)
    AR: float = -1.0  # Alkali resistance: 1(high) - 4(low)
    PR: float = -1.0  # Phosphate resistance: 1(high) - 4(low)

    wl: list[float] = field(default_factory=list)
    IT: list[float] = field(default_factory=list)
    thickness: list[float] = field(default_factory=list)

    comments: str = ""

    original_file: str = ""

    def __post_init__(self) -> None:
        if not self.glasscode:
            self.generate_glasscode()

        if self.nd != -1.0 and self.vd in [-1.0, 0.0]:
            self.vd = (self.n_d - 1) / (self.n_F - self.n_C)

        if self.original_file:
            self.manufacturer = self.original_file.split(".")[0]

        # if self.n_d == -1:
        #     print(f"WARNING! No refractive index provided for glass {self.name}")

    @property
    def Vd(self) -> float:
        return self.vd

    @property
    def V_d(self) -> float:
        return self.vd

    @property
    def Abbe_number(self) -> float:
        return self.vd

    @property
    def WL(self) -> list[float]:
        return self.wl

    @property
    def alternative_names(self) -> list:
        """
        A short function to generate a list of all possible alternative names
        (by stripping commonly used characters)

        Returns:
            list: a list containing the alternative names of the glass (strings)
        """

        prefixes: list[str] = ["N-", "S-", "LITHOTEC-"]
        characters: list[str] = [" ", "*", "ultra"]
        brackets: str = "[{()}]"

        alt_names: list[str] = [self.name]

        for p in prefixes:
            if p in self.name:
                alt_names.append("".join(self.name.split(p)[1:]))
        for c in characters:
            if c in self.name:
                alt_names.append("".join(self.name.split(c)))
        alt_names.append(re.sub(brackets, "", self.name))

        so_far: list[str] = alt_names[:]
        # Do it again to catch names with more than one unwanted character/substring
        for alt in so_far:
            for c in characters:
                alt_names.append("".join(alt.split(c)))

        # Convert to set to remove duplicates, remove rempty strings
        return [n for n in list(set(alt_names)) if n != ""]

    @property
    def status_description(self) -> str:
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
            AGF_formula_number=self.formula, coefficients=self.disp_formula_coefficients
        )

    def generate_glasscode(self) -> Glass:
        if not self.glasscode:
            self.glasscode = (
                int((self.n_d - 1) * 1e3) * 1e3 + int((self.V_d * 10) * 1e3) / 1e3
            )

        return self

    @overload
    def transmission(self, wl: ScalarQuantity) -> float: ...

    @overload
    def transmission(self, wl: ArrayQuantity) -> np.ndarray: ...

    def transmission(
        self,
        wl: ScalarQuantity | ArrayQuantity,
    ) -> float | np.ndarray:
        """
        The internal transmission of the glass at the specified wavelength(s)
        """
        if not isinstance(wl, Quantity):
            print(wl)
            assert isinstance(wl, int | float | list | tuple)

            print(
                f"Warning: no units provided for wavelength(s) {wl}. "
                + "Assuming default unit of microns (um)."
            )
            wl: Quantity = wl * u.um

        if wl.isscalar:
            wl: float = wl.to(u.um).value
        else:
            wl: list[float] = [float(iwl.to(u.um).value) for iwl in wl]

        if wl is None:
            wl: list[float] = self.wl

        try:
            return np.interp(wl, xp=self.wl, fp=self.IT, left=0.0, right=0.0)
        except Exception as e:
            print(e)
            print(f"Could not compute transmission for {self.name}")
            print(f"{self.wl = }")
            print(f"{self.IT = }")
            return np.array([np.nan] * len(wl))

    @overload
    def refractive_index(self, wls: ScalarQuantity | float | str) -> float: ...

    @overload
    def refractive_index(
        self,
        wls: ArrayQuantity | list[float] | list[str],
    ) -> NDArray[np.floating]: ...

    def refractive_index(
        self,
        wls: ScalarQuantity
        | ArrayQuantity
        | float
        | list[float]
        | str
        | list[str] = "d",
    ) -> float | NDArray[np.floating]:

        if wls is None:
            wls: str = "d"

        if isinstance(wls, str):
            try:
                wls: ScalarQuantity = (dCF | _fraunhofer_lines)[wls]
            except KeyError:
                print(
                    f"Unknown wavelength string {wls}. Valid options are"
                    + f"{', '.join((dCF | _fraunhofer_lines).keys())}"
                )
                print("Using default wavelength of 588 nm")
                wls: ScalarQuantity = dCF["d"]

        if isinstance(wls, list) and all(isinstance(wl, str) for wl in wls):
            return np.ndarray([self.refractive_index(wls=wl) for wl in wls])  # type: ignore

        try:
            return _refractive_index(
                wavelengths=wls,
                coefficients=self.rii_dispersion_coefficients,
                formula_number=self.rii_formula_number,
            )
        except ValueError as e:
            print(e)
            print(f"Could not compute refractive index for {self.name}")
            return np.nan

    @overload
    def n(self, wls: ScalarQuantity | float | str | list[str]) -> float: ...

    @overload
    def n(self, wls: ArrayQuantity) -> np.ndarray: ...

    def n(
        self,
        wls: ScalarQuantity | ArrayQuantity | float | str | list[str] = "d",
    ) -> float | ArrayLike:
        return self.refractive_index(wls=wls)

    @property
    def n_d(self) -> float:
        return self.refractive_index(wls="d")

    @property
    def n_F(self) -> float:
        return self.refractive_index(wls="F")

    @property
    def n_C(self) -> float:
        return self.refractive_index(wls="C")

    @overload
    def dn(self, wls: ScalarQuantity | float) -> float: ...

    @overload
    def dn(self, wls: ArrayQuantity | float) -> np.ndarray: ...

    def dn(self, wls: float | Quantity) -> float | np.ndarray:
        """
        Differential refractive index, a similar metric to Abbe number/dispersion

        https://www.primeoptics.com.au/Reference/dRPD_description.pdf
        """

        if isinstance(wls, Quantity):
            try:
                wls: np.ndarray = wls.to(u.um).value
            except UnitConversionError as e:
                raise e

        if isinstance(wls, list | np.ndarray) and len(wls) == 1:
            # If only one element, convert to a float
            wls: float = wls[0]  # ty:ignore[invalid-argument-type]

        elif isinstance(wls, int | float):
            wls: np.ndarray = np.array([wls - 1e-10, wls, wls + 1e-10])
            return (np.gradient(self.refractive_index(wls * u.um)) / np.gradient(wls))[
                1
            ]

        assert isinstance(wls, np.ndarray) | isinstance(wls, list)
        # Otherwise, return the gradient over the array
        return np.gradient(self.refractive_index(wls)) / np.gradient(wls)

    @overload
    def dV(self, wls: ScalarQuantity | float) -> float: ...

    @overload
    def dV(self, wls: ArrayQuantity) -> NDArray[np.floating]: ...

    def dV(self, wls: Quantity) -> float | NDArray[np.floating]:
        """
        Differential Abbe number

        https://www.primeoptics.com.au/Reference/dRPD_description.pdf
        """

        if isinstance(wls, Quantity):
            wls: np.ndarray = wls.to(u.um).value

        if len(wls) == 1:
            wls: float = wls[0]

        elif isinstance(wls, int | float):
            wls: np.ndarray = np.array([wls - 1e-10, wls, wls + 1e-10])

            return float(
                np.mean((self.refractive_index(wls[1]) - 1) / (-self.dn(wls * u.um)))
            )

        return (self.refractive_index(wls) - 1) / (-self.dn(wls))

    @overload
    def dRPD(self, wls: ScalarQuantity | float) -> float: ...

    @overload
    def dRPD(self, wls: ArrayQuantity) -> NDArray[np.floating]: ...

    def dRPD(self, wls: float | Quantity | None = None) -> float | NDArray[np.floating]:
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
            wls: float = self.wl[0]

        elif isinstance(wls, Quantity):
            if wls.isscalar:
                wls: float = wls.to(u.um).value
            wls: np.ndarray = wls.to(u.um).value

        if isinstance(wls, int | float):
            wls: np.ndarray = np.array([wls - 1e-10, wls, wls + 1e-10])

        return self.dV(wls - np.gradient(wls)) / self.dV(wls * u.um)

    def plot(
        self,
        metric: str = "n",
        wls: Quantity | None = None,
        axis: plt.Axes | None = None,
        show: bool = True,
        debug_level: float = 0,
    ) -> Glass:
        """ """

        if wls is None:
            wls: Quantity = np.linspace(0.39, 0.95, 50) * u.um

        metrics: dict[str, Callable] = {
            "n": self.n,
            "refractive_index": self.refractive_index,
            "transmission": self.transmission,
            "dn": self.dn,
            "dRPD": self.dRPD,
            "dV": self.dV,
        }

        if metric not in metrics:
            if debug_level:
                print(
                    "Unknown metric." + f"Valid options are {', '.join(metrics.keys())}"
                )
            return self

        try:
            y: float | list[float] = metrics[metric](wls)
        except Exception as e:
            print(e)

        if y is None:
            if debug_level:
                print(
                    "Unable to compute" + f"{metric} over {wls} for glass {self.name}"
                )
            return self

        if not axis:
            fig: plt.Figure = plt.figure()
            ax: plt.Axes = fig.gca()
        else:
            ax: plt.Axes = axis

        ax.plot(wls, y, label=f"{metric}")
        ax.set_xlabel("Wavelength (um)")
        ax.set_ylabel(f"{metric[0].upper()}{metric[1:]}")

        if show:
            plt.show()

        return self

    def rename(self, new_name: str) -> Glass:
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
        """

        return (
            (self.name == other.name)
            and (self.n_d == other.n_d)
            and (self.V_d == other.V_d)
        )

    def to_agf_format(self) -> str:
        try:
            output: str = ""
            output += f"NM {self.name} {self.formula:.0f} {self.glasscode:.3f} {self.n_d:.5f} {self.V_d:.2f} {self.status:.0f} {self.meltfreq:.0f}\n"
            output += f"GC {self.comments}\n"
            output += f"ED {self.cte1:.6f} {self.cte2:.6f} {self.density:.6f} {self.dpgf:.6f}\n"
            output += f"CD {' '.join([f'{c:.9E}' for c in self.disp_formula_coefficients if c is not None])}\n"
            output += f"TD {' '.join([f'{c:.9E}' for c in self.thermal_disp_coefficients if c is not None])}\n"
            output += f"OD {self.CR:.4f} {self.FR:.4f} {self.SR:.4f} {self.AR:.4f} {self.PR:.4f}\n"
            output += f"LD {self.wlmin:.5E} {self.wlmax:.5E}\n"
            for wl, it, thickness in zip(
                self.wl, self.IT, self.thickness, strict=False
            ):
                output += f"IT {wl:.5E} {it:.5E} {thickness:.5E}\n"

            return output

        except TypeError:
            print(f"Errors occurred when attempting to process glass {self.name}")
            return ""

    @property
    def summary(self) -> str:
        return (
            "-" * 25
            + "\n"
            + f"Original file: {self.original_file}\n"
            + f"Glass: {self.name}\n"
            + f"n_d = {self.nd}\n"
            + f"V_d = {self.vd}\n"
            + f"Status {self.status}: {self.status_description}"
        )

    def __str__(self) -> str:
        return self.name
        # return f"Original file: {self.original_file}\n"\
        #      + f"Glass: {self.name}\n"\
        #      + f"n_d = {self.nd}\n"\
        #      + f"V_d = {self.vd}\n"\
        #      + f"Status: {self.status}: {self.status_description}\n"\
        #      + "-"*20

    # def __repr__(self) -> str:
    #     return "Glass("
    #            ""


@dataclass
class AGFData:
    """
    A simple wrapper class to indicate that a string is in AGF format, and should be
    processed as such when passed to functions that can take multiple types of input
    (e.g. GlassCatalogue.read_AGF)
    """

    agf_string: str
    filename: str = ""


class GlassCatalogue(dict):
    """
    * generate plots:
        - Abbe diagram (change reference wavelength)
        - Etc.

    TODO:
    Take a string, Path, or list thereof as input and auto-generate GlassCatalogue from
    AGF file(s)
    """

    def __init__(
        self,
        agf: str
        | Path
        | list[str]
        | list[Path]
        | AGFData
        | list[AGFData]
        | None = None,
        *args,  # noqa: ANN002
        **kwargs,  # noqa: ANN003
    ) -> None:
        super().__init__(*args, **kwargs)
        self.glass_list: list[str] = list(super().keys())

        if agf:
            self.read_AGF(agf)

    @property
    def comment(self) -> str:
        return getattr(self, "_comment", "")

    def set_comment(self, value: str) -> None:
        self._comment = value

    def __setitem__(self, key: str, value: Glass) -> None:
        self.glass_list.append(key)
        super().__setitem__(key, value)

    def __iter__(self) -> Iterator[Glass]:
        return iter(list(self.values()))

    def itervalues(self) -> Generator[Glass]:
        return (self[key] for key in self)

    def replace(self, key: str, replacement: Glass) -> None:
        assert isinstance(replacement, Glass)
        self[key] = replacement

    @property
    def glasses(self) -> list[Glass]:
        return list(self.values())

    @property
    def number_of_glasses(self) -> int:
        return len(self.__dict__)

    def read_AGF(
        self,
        agf_file: Path | str | list[Path] | list[str] | AGFData | list[AGFData],
        debug_level: int = 0,
    ) -> GlassCatalogue:
        """
        This function reads in a Zemax-formatted AGF file (as provided by glass
        manufacturers) and creates a GlassCatalogue from this.

        TODO: if the passed-in string is NOT an AGF filename, search the AGFs using that
        term instead.
        """

        if isinstance(agf_file, AGFData):
            original_file: str = agf_file.filename
            agf: list[str] = agf_file.agf_string.splitlines()

        else:
            if isinstance(agf_file, str):
                if "." not in agf_file:
                    agf_file: Path = GLASS_CATALOGUES[agf_file]
                else:
                    agf_file: Path = Path(agf_file)

            if isinstance(agf_file, list):
                for f in agf_file:
                    self.read_AGF(f)
                return self

            assert isinstance(agf_file, Path)

            original_file: str = agf_file.name

            # print(f"Reading file: {original_file:<20}")

            p: Path = agf_file

            # TODO: First find the text encoding (with chardet), and then use `with p.open(.....) as agf:`
            try:
                agf = p.open(encoding="utf-8", newline="\r\n")
                for _, _line in enumerate(agf):
                    break  # Causes UnicodeDecodeError
            except UnicodeDecodeError:
                agf = p.open(encoding="utf-16", newline="\r\n")

        def _add_glass(g: Glass) -> None:
            if g.name not in self:
                self[g.name] = g
            else:
                print(
                    f"A glass with the same name ({g.name}) "
                    + "already exists in the catalogue!"
                )
                if self[g.name].is_the_same_as(g):
                    print("The two glasses are identical. Skipping...")

                else:
                    print(f"Glass {g.name} is different from the existing one")

                    print(
                        f"Renaming existing {self[g.name]} to "
                        + f"{self[g.name].name}_{self[g.name].original_file}..."
                    )
                    self[f"{g.name}_{self[g.name].original_file}"] = self[
                        g.name
                    ].rename(
                        new_name=f"{self[g.name].name}_{self[g.name].original_file}"
                    )
                    del self[g.name]

                    print(
                        f"Adding new glass {g.name} as "
                        + f"{g.name}_{g.original_file} to the catalogue..."
                    )
                    self[f"{g.name}_{g.original_file}"] = self[g.name] = g.rename(
                        new_name=f"{g.name}_{g.original_file}"
                    )

        #  Empty variable in which a Glass object will be stored
        g: Glass = None  # ty:ignore[invalid-assignment]

        for _, line in enumerate(agf):
            data = [k.strip() for k in line.split(" ")]

            if data:
                if data[0] == "CC":
                    # Comment at the start of a catalogue file
                    self.set_comment(" ".join(data[1:]))
                    if debug_level >= 1:
                        print(self.comment)

                elif data[0] == "NM":
                    # Found a new "Name" line or reached the end of the file!
                    # Store the previous data in the catalogue
                    if g is not None:
                        _add_glass(g)

                    # Create a new empty glass
                    g = Glass(name=data[1], original_file=original_file)

                    try:
                        g.formula = int(float(data[2]))
                        if debug_level >= 1:
                            print(f"Formula number: {g.formula}")
                    except ValueError:
                        ...

                    try:
                        g.nd = float(data[4])
                        if debug_level >= 1:
                            print(f"Refractive index: {g.nd}")
                    except ValueError as e:
                        if debug_level >= 1:
                            print(f"{e}, assigning refractive index = -1.0")
                        g.nd = -1.0

                    if data[5] != "":
                        try:
                            g.vd = float(data[5])
                            if debug_level >= 1:
                                print(f"Abbe number: {g.vd}")
                        except ValueError:
                            ...
                    else:
                        try:
                            g.vd = float(data[6])
                            if debug_level >= 1:
                                print(f"Abbe number: {g.vd}")
                        except ValueError:
                            ...

                    try:
                        glasscode = float(data[3])
                        if glasscode in [-1.0, 0.0, 1.0]:
                            g.generate_glasscode()
                        if debug_level >= 1:
                            print(f"Glass code: {glasscode}")
                        else:
                            g.glasscode = glasscode
                            if debug_level >= 1:
                                print(f"Glass code: {g.glasscode}")
                    except ValueError:
                        g.generate_glasscode()

                    if len(data) >= 8:  # noqa: PLR2004
                        try:
                            _status = float(data[7])
                            if debug_level >= 1:
                                print(f"_status: {_status}")
                            g.status = int(_status)
                            if debug_level >= 1:
                                print(f"int -> Status: {g.status}")
                        except ValueError:
                            ...

                    if len(data) >= 9:  # and data.count("-") == 0:  # noqa: PLR2004
                        try:
                            g.meltfreq = float(data[8])
                            if debug_level >= 1:
                                print(f"Melting frequency: {g.meltfreq}")
                        except ValueError:
                            ...
                    # else: g.meltfreq = 0

                    continue

                elif data[0] == "LD":
                    try:
                        g.wlmin = float(data[1])
                        if debug_level >= 1:
                            print(f"Minimum wavelength: {g.wlmin}")
                    except ValueError:
                        ...

                    try:
                        g.wlmax = float(data[2])
                        if debug_level >= 1:
                            print(f"Maximum wavelength: {g.wlmax}")
                    except ValueError:
                        ...

                    continue

                elif data[0] == "CD":
                    try:
                        # Ignoring empty strings - may need to change?
                        # Or at least check the length of the list if computing
                        # something from these coefficients

                        # Unused coefficients SHOULD be reported as zero
                        g.disp_formula_coefficients = [
                            float(d) for d in data[1:] if d != ""
                        ]
                        if debug_level >= 1:
                            print(
                                f"Dispersion formula coefficients: {g.disp_formula_coefficients}"
                            )
                    except ValueError:
                        ...

                    continue

                elif data[0] == "TD":
                    try:
                        # Ignoring empty strings - may need to change?
                        # Or at least check the length of the list if computing
                        # something from these coefficients
                        g.thermal_disp_coefficients = [
                            float(d) for d in data[1:] if d != ""
                        ]
                        if debug_level >= 1:
                            print(
                                f"Thermal dispersion coefficients: {g.thermal_disp_coefficients}"
                            )
                    except ValueError:
                        ...

                    continue

                elif data[0] == "ED":
                    try:
                        g.cte1 = float(data[1])
                        if debug_level >= 1:
                            print(f"CTE1: {g.cte1}")
                    except ValueError:
                        ...

                    try:
                        g.cte2 = float(data[2])
                        if debug_level >= 1:
                            print(f"CTE2: {g.cte2}")
                    except ValueError:
                        ...

                    try:
                        g.density = float(data[3])
                        if debug_level >= 1:
                            print(f"Density: {g.density}")
                    except ValueError:
                        ...

                    try:
                        g.dpgf = float(data[4])
                        if debug_level >= 1:
                            print(f"dPgF: {g.dpgf}")
                    except ValueError:
                        ...

                    continue

                elif data[0] == "OD":
                    # ? = data[1]
                    if len(data) > 2:  # noqa: PLR2004
                        try:
                            g.CR = float(data[2])
                            if debug_level >= 1:
                                print(f"CR: {g.CR}")
                        except ValueError:
                            if g.CR == "3-4":
                                g.CR = 3.5
                            if debug_level >= 1:
                                print(f"CR: {g.CR}")

                        try:
                            g.FR = int(float(data[3]))
                            if debug_level >= 1:
                                print(f"FR: {g.FR}")
                        except ValueError:
                            ...

                        try:
                            g.SR = float(data[4])
                            if debug_level >= 1:
                                print(f"SR: {g.SR}")
                        except ValueError:
                            ...

                        try:
                            g.AR = float(data[5])
                            if debug_level >= 1:
                                print(f"AR: {g.AR}")
                        except ValueError:
                            ...

                        try:
                            g.PR = float(data[6])
                            if debug_level >= 1:
                                print(f"PR: {g.PR}")
                        except ValueError:
                            ...
                        except IndexError:
                            # Missing data?
                            continue

                    continue

                elif (
                    data[0] == "IT"
                    and data[1] != ""
                    and data[2] != ""
                    and data[3] != ""
                ):
                    g.wl.append(float(data[1]))
                    g.IT.append(float(data[2]))
                    g.thickness.append(float(data[3]))
                    continue

                elif data[0] == "GC":
                    g.comments = " ".join([d for d in data[1:] if d != ""])
                    continue

            else:  # data is empty
                # In a normal AGF file, there are no empty lines, so this is likely
                # the end of the file
                if g is not None:
                    _add_glass(g)

        return self

    def to_AGF_string(self, comment: str = "") -> str:
        """
        Maybe define an external function or even just a dictionary to generate
        the required line-structure for the AGF file. I just want to do what
        `catalogues.read_AGF` is doing, but in reverse.

        Come to think of it, maybe `read_AGF` belongs in this class? Workflow
        would be: create empty GlassCatalogue(), call GC.read_AGF() to populate,
        OR: create GlassCatalogue(agf_files=["./schott.agf", "./ohara.agf"]) to
        initialise the GC with these to begin with. That might be nice!
        """

        out_string: str = ""
        for _, glass in self.items():
            if comment:
                out_string += f"CC {comment}\n"
            out_string += glass.to_agf_format()
        return out_string

    def write_AGF(
        self, comment: str = "", filename: str = "./output.agf"
    ) -> GlassCatalogue:
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
            print("No target filename specified. Printing raw output...")
            if comment:
                print(f"CC {comment}")
            for _, glass in self.items():
                print(glass.to_agf_format(), end="")
            return self

        p = Path(filename)
        with p.open(mode="w") as f:
            print(f"Writing to {filename}...")
            if comment:
                f.write(f"CC {comment}\n")
            for _, glass in self.items():
                f.write(glass.to_agf_format())

        return self

    def find(
        self,
        name: str | None = None,
        case_sensitive: bool = False,
        n: float | None = None,
        dn: float = 0.01,
        n_wavelength: float = 0.588,
        V_d: float | None = None,
        dV_d: float = 1.0,
    ) -> GlassCatalogue:
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
                if not case_sensitive:
                    name = name.lower()
                for possible_match in glass.alternative_names:
                    if not case_sensitive:
                        possible_match = possible_match.lower()  # noqa: PLW2901
                    if name == possible_match or name in possible_match:
                        results[glass.name] = glass
                        continue

            if n:
                if not dn:
                    dn = 0.01
                if abs(n - glass.n(n_wavelength)) < dn:
                    results[glass.name] = glass
                    continue

            if V_d:
                if not dV_d:
                    dV_d = 1.0
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
        min_n: float | None = None,
        max_n: float | None = None,
        n_wavelength: str | Quantity = "d",
        # Abbe number
        min_V_d: float | None = None,
        max_V_d: float | None = None,
        # Glass status
        status_code: float | list[int | float] | None = None,
        status_description: str | list[str] | None = None,
        # Transsmission
        min_transmission: float | None = None,
        max_transmission: float | None = None,
        min_average_transmission: float | None = None,
        # CTE
        min_CTE: float | None = None,
        max_CTE: float | None = None,
        debug_level: int = 0,
    ) -> GlassCatalogue:
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

        if debug_level >= 1:
            print("Filtering catalogue with criteria:")
            if min_n is not None:
                print(f"  Minimum n at {n_wavelength}: {min_n}")
            if max_n is not None:
                print(f"  Maximum n at {n_wavelength}: {max_n}")
            if min_V_d is not None:
                print(f"  Minimum V_d: {min_V_d}")
            if max_V_d is not None:
                print(f"  Maximum V_d: {max_V_d}")
            if status_code is not None:
                print(f"  Status code: {status_code}")
            if status_description is not None:
                print(f"  Status description: {status_description}")

            if min_transmission is not None or max_transmission is not None:
                print(f"  Wavelength range: {min_wl} - {max_wl}")
            if min_transmission is not None:
                print(f"  Minimum transmission: {min_transmission}")
            if max_transmission is not None:
                print(f"  Maximum transmission: {max_transmission}")
            if min_average_transmission is not None:
                print(f"  Minimum average transmission: {min_average_transmission}")

            if min_CTE is not None:
                print(f"  Minimum CTE: {min_CTE}")
            if max_CTE is not None:
                print(f"  Maximum CTE: {max_CTE}")

        if not (
            min_n
            or max_n
            or min_V_d
            or max_V_d
            or status_code
            or status_description
            or min_transmission
            or max_transmission
            or min_average_transmission
            or min_CTE
            or max_CTE
        ):
            print("No filter criteria provided.")
            return self

        # Create an empty GlassCatalogue to store our results in
        results: GlassCatalogue = GlassCatalogue()

        for _, glass in self.items():
            if min_wl is not None and max_wl is not None:
                try:
                    if (
                        glass.wlmin > min_wl.to("um").value
                        or glass.wlmax < max_wl.to("um").value
                    ):
                        continue
                except TypeError:
                    ...

            if status_code:
                try:
                    if isinstance(status_code, int | float):
                        if glass.status != status_code:
                            continue
                    else:  # It is a list or ArrayLike
                        if glass.status not in status_code:
                            continue
                except TypeError:
                    ...

            if status_description:
                try:
                    if isinstance(status_description, str):
                        if (
                            status_description.lower()
                            not in glass.status_description.lower()
                        ):
                            continue
                    else:  # It is list or ArrayLike
                        status_description = [sd.lower() for sd in status_description]
                        keep = False
                        for sd in status_description:
                            if sd in glass.status_description.lower():
                                keep = True
                        if not keep:
                            continue
                except TypeError:
                    ...

            if min_n:
                try:
                    if glass.n(n_wavelength) < min_n:
                        print(f"{glass.n(n_wavelength) = }, skipping...")
                        continue
                except TypeError:
                    ...

            if max_n:
                try:
                    if glass.n(n_wavelength) > max_n:
                        continue
                except TypeError:
                    ...

            if min_V_d and not max_V_d:
                try:
                    if glass.V_d < min_V_d:
                        continue
                except TypeError:
                    ...

            if max_V_d and not min_V_d:
                try:
                    if glass.V_d > max_V_d:
                        continue
                except TypeError:
                    ...

            if min_V_d and max_V_d:
                try:
                    if glass.V_d < min_V_d or glass.V_d > max_V_d:
                        continue
                except TypeError:
                    ...

            if min_transmission or max_transmission or min_average_transmission:
                try:
                    assert (min_wl is not None) and (max_wl is not None)
                except AssertionError:
                    ...

                transmission_wls: ArrayQuantity = (
                    np.linspace(min_wl.to("um").value, max_wl.to("um").value, 20) * u.um
                )
                transmission = glass.transmission(wl=transmission_wls)

            if min_transmission:
                try:
                    if min(transmission) <= min_transmission:
                        continue
                except TypeError:
                    ...

            if max_transmission:
                try:
                    if max(transmission) >= max_transmission:
                        continue
                except TypeError:
                    ...

            if min_average_transmission:
                try:
                    if np.mean(transmission) <= min_average_transmission:
                        continue
                except TypeError:
                    ...

            # If it hasn't violated any of the conditions yet...
            results[glass.name] = glass

        if debug_level >= 1:
            print(f"Filter returned {len(results)} / {len(self.glass_list)}")
        return results

    def plot_Abbe_diagram(
        self,
        axis: plt.Axes | None = None,
        interactive: bool = True,
    ) -> GlassCatalogue:
        """
        TODO:
        - [ ] Add interactivity to the plot, with hover-over labels
        """

        if axis is None:
            fig: plt.Figure = plt.figure()
            ax: plt.Axes = fig.gca()
        else:
            ax: plt.Axes = axis

        names: list[str] = [g.name for g in self.glasses]

        # Somewhat bulky code to color-code the points by the original file
        colors: list[str] = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        all_original_files: list[str] = [g.original_file for g in self.glasses]
        unique_original_files: list[str] = list(set(all_original_files))
        manufacturer_indices: dict[str, int] = {
            f: i for i, f in enumerate(unique_original_files)
        }
        cols: list[str] = [colors[manufacturer_indices[f]] for f in all_original_files]

        # for _, glass in self.items():
        points = ax.scatter(
            [g.Abbe_number for g in self.glasses],
            [g.n_d for g in self.glasses],
            marker="o",
            s=50,
            alpha=0.5,
            c=cols,
        )

        ax.set_xlim(ax.get_xlim()[::-1])
        ax.set_ylim(ax.get_ylim())

        for i, m in enumerate(unique_original_files):
            ax.scatter(0, 0, marker="o", s=50, c=colors[i], label=m)

        ax.legend()

        if interactive:
            # code from here:
            # https://stackoverflow.com/questions/7908636/how-to-add-hovering-annotations-to-a-plot
            annot = ax.annotate(
                "",
                xy=(0, 0),
                xytext=(20, 20),
                textcoords="offset points",
                bbox={"boxstyle": "round", "fc": "w"},
                arrowprops={"arrowstyle": "->"},
            )
            annot.set_visible(False)

            def update_annot(ind: dict) -> None:
                pos = points.get_offsets()[ind["ind"][0]]  # type: ignore
                annot.xy = pos  # type: ignore
                text = f"{names[ind['ind'][0]]}"
                # text = "{}, {}".format(" ".join(list(map(str,ind["ind"]))),
                #                     " ".join([names[n] for n in ind["ind"]]))
                annot.set_text(text)
                annot.get_bbox_patch().set_alpha(0.4)

            def hover(event: MouseEvent) -> None:
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

            fig.canvas.mpl_connect("motion_notify_event", hover)  # type: ignore

        plt.show()

        return self

    def plot_transmission(
        self,
        axis: plt.Axes | None = None,
        wls: ArrayLike | Quantity | None = None,
    ) -> GlassCatalogue:
        if axis is None:
            fig: plt.Figure = plt.figure()
            ax: plt.Axes = fig.gca()
        else:
            ax: plt.Axes = axis

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
        D_telescope: Quantity = 0.8 * u.m,
        D_ADC: Quantity = 15 * u.mm,
        z_max: Quantity = 65 * u.degree,
        air_model: Model | None = None,
        criterion: str = "max_abs",
        num_samples: int = 20,
        max_residuals: Quantity = 100 * u.mas,
    ) -> tuple[list, list]:
        """
        Optimise ADCs for all possible combinations of glasses in the catalogue
        """

        from curved_air.instruments.ADC import ADC, PrismStack  # noqa: PLC0415

        if air_model is None:
            air_model = Air()

        ADCs: list[ADC] = []
        results: list[dict] = []

        columns = [
            "Glass 1",
            "Glass 2",
            "Apex angle",
            "Compound angle",
            "Zero-deviation wavelength",
            "PTV Residuals",
        ]

        print_output = (
            f"{columns[0]:<16}\t"
            f"{columns[1]:<16}\t"
            f"{columns[2]:<16}\t"
            f"{columns[3]:<16}\t"
            f"{columns[4]:<16}\t"
            f"{columns[5]:<16}\n"
        )

        n: int = len(self.glasses)
        num_combinations = int((n * (n - 1)) / 2)

        for g1, g2 in tqdm(
            combinations(self.glasses, 2),
            desc="Optimising ADCs",
            total=num_combinations,
            position=0,
            leave=True,
            unit="ADC",
            ncols=100,
        ):
            adc: ADC = ADC(
                prism_stack=PrismStack([g1, g2]),
                min_wl=min_wl,
                max_wl=max_wl,
                D_telescope=D_telescope,
                D_ADC=D_ADC,
                z_max=z_max,
                air_model=air_model,
                criterion=criterion,
            )

            adc.optimise(num_samples=num_samples)

            if adc.residuals_PTV <= max_residuals:
                ADCs.append(adc)
                results.append(
                    {
                        columns[0]: g1.name,
                        columns[1]: g2.name,
                        columns[2]: f"{adc.alpha.to(u.deg).value:.5g}",
                        columns[3]: f"{adc.beta.to(u.deg).value:.5g}",
                        columns[4]: f"{adc.wl_0.to(u.um).value:.5g}",
                        columns[5]: f"{adc.residuals_PTV.to(u.mas).value:.5g}",
                    }
                )
                print_output += (
                    f"{g1.name:<16}\t"
                    f"{g2.name:<16}\t"
                    f"{adc.alpha:<16.5g}\t"
                    f"{adc.beta:<16.5g}\t"
                    f"{adc.wl_0:<16.5g}\t"
                    f"{adc.residuals_PTV:<16.5g}\n"
                )

        print(print_output)

        return ADCs, results

    def __add__(self, other: GlassCatalogue) -> GlassCatalogue:
        if not isinstance(other, GlassCatalogue):
            raise TypeError(f"Cannot add GlassCatalogue with {type(other)}")

        new_catalogue = GlassCatalogue()

        # Merge catalogues, keeping all unique glasses (by name) from both
        for _, glass in self.items():
            new_catalogue[glass.name] = glass

        for _, glass in other.items():
            if glass.name not in new_catalogue:
                new_catalogue[glass.name] = glass
            else:
                print(
                    f"Glass {glass.name} already exists in the new catalogue. Skipping..."
                )

        return new_catalogue

    def __str__(self) -> str:
        return "\n".join([f.summary for f in self.values()])

    def __repr__(self) -> str:
        return "\n".join([f.summary for f in self.values()])


def _dispersion_formula_number(AGF_formula_number: int) -> int:
    """
    Converts between AGF formula numbers and refractiveindex.INFO formula
    numbers. The two systems use different conventions for numbering.
    """

    if AGF_formula_number in [1, 11, 12, 13]:
        return 3

    if AGF_formula_number in [2, 6]:
        return 2

    if AGF_formula_number in [3]:
        return 4

    raise ValueError
    # return 0


def _dispersion_coefficients(
    AGF_formula_number: int,
    coefficients: list[float],
) -> list[float]:
    """
    Adapted from the refractiveindex.INFO database-generation scripts. I follow
    the same conventions in formula numbering as that database.
    """

    if AGF_formula_number == 1:  # AGF formula number
        # rii_formula_number = 3
        coeff_list: list[int | None] = [None, 2, -2, -4, -6, -8]
        rii_formula_coefficients: list[float] = []
        for i, k in zip(coefficients, coeff_list[: len(coefficients)], strict=False):
            rii_formula_coefficients.append(i)
            if k is not None:
                rii_formula_coefficients.append(k)

    elif AGF_formula_number in [2, 6]:
        # rii_formula_number = 2
        rii_formula_coefficients: list[float] = [0, *coefficients]

    elif AGF_formula_number == 3:  # noqa: PLR2004
        # rii_formula_number = 4
        c: list[float] = coefficients
        rii_formula_coefficients: list[float] = [
            c[0] + 1,
            c[1],
            2,
            c[2],
            2,
            c[3],
            0,
            c[4],
            2,
        ]

    elif AGF_formula_number == 11:  # noqa: PLR2004
        # rii_formula_number = 3
        coeff_list: list[int | None] = [None, 2, -2, -4, -6, -8, -10, -12]
        rii_formula_coefficients: list[float] = []
        for c, k in zip(coefficients, coeff_list, strict=False):
            rii_formula_coefficients.append(c)
            if k is not None:
                rii_formula_coefficients.append(k)

    elif AGF_formula_number == 12:  # noqa: PLR2004
        # rii_formula_number = 3
        coeff_list: list[int | None] = [None, 2, -2, -4, -6, -8, 4, 6]
        rii_formula_coefficients: list[float] = []
        for c, k in zip(coefficients, coeff_list, strict=False):
            rii_formula_coefficients.append(c)
            if k is not None:
                rii_formula_coefficients.append(k)

    elif AGF_formula_number == 13:  # noqa: PLR2004
        # rii_formula_number = 3
        coeff_list: list[int | None] = [None, 2, 4, -2, -4, -6, -8, -10, -12]
        rii_formula_coefficients: list[float] = []
        for c, k in zip(coefficients, coeff_list, strict=False):
            rii_formula_coefficients.append(c)
            if k is not None:
                rii_formula_coefficients.append(k)

    else:
        raise ValueError(f"Invalid AGF_formula_number: {AGF_formula_number}")

    return rii_formula_coefficients


def _refractive_index(
    wavelengths: Quantity | float | list[float] | ArrayLike,
    coefficients: list[float],
    formula_number: int,
) -> float | np.ndarray:
    """A function to compute the wavelength-dependent refractive index of a material
    based on coefficients

    This uses the formula numbering convention of the refractiveindex.INFO database:
    1 is the Sellmeier formula with squared coefficients in the denominators,
    2 is the Sellmeier formula without squares in the denominators, and
    3 is a polynomial formula used by some glass manufacturers.
    """

    if isinstance(wavelengths, Quantity):
        try:
            wavelengths: float | list[float] = wavelengths.to("um").value
        except UnitConversionError as e:
            raise e  # Wrong units, not compatible with units of wavelength
        except AttributeError as e:
            # Not an astropy.units.Quantity after all
            raise AttributeError("") from e

    wavelengths: np.ndarray = np.array(wavelengths)

    C: list[float] = [0] * 18  # ty:ignore[invalid-assignment]
    # C[0] = 0
    for i, c in enumerate(coefficients, start=1):
        C[i] = c
    # all C[len(coefficients)+1:] = 0

    if formula_number == 1:
        return np.sqrt(
            1
            + C[1]
            + (C[2] * wavelengths**2) / (wavelengths**2 - C[3] ** 2)
            + (C[4] * wavelengths**2) / (wavelengths**2 - C[5] ** 2)
            + (C[6] * wavelengths**2) / (wavelengths**2 - C[7] ** 2)
            + (C[8] * wavelengths**2) / (wavelengths**2 - C[9] ** 2)
            + (C[10] * wavelengths**2) / (wavelengths**2 - C[11] ** 2)
            + (C[12] * wavelengths**2) / (wavelengths**2 - C[13] ** 2)
            + (C[14] * wavelengths**2) / (wavelengths**2 - C[15] ** 2)
            + (C[16] * wavelengths**2) / (wavelengths**2 - C[17] ** 2)
        )

    if formula_number == 2:  # noqa: PLR2004
        return np.sqrt(
            1
            + C[1]
            + (C[2] * wavelengths**2) / (wavelengths**2 - C[3])
            + (C[4] * wavelengths**2) / (wavelengths**2 - C[5])
            + (C[6] * wavelengths**2) / (wavelengths**2 - C[7])
            + (C[8] * wavelengths**2) / (wavelengths**2 - C[9])
            + (C[10] * wavelengths**2) / (wavelengths**2 - C[11])
            + (C[12] * wavelengths**2) / (wavelengths**2 - C[13])
            + (C[14] * wavelengths**2) / (wavelengths**2 - C[15])
            + (C[16] * wavelengths**2) / (wavelengths**2 - C[17])
        )

    if formula_number == 3:  # noqa: PLR2004
        return np.sqrt(
            C[1]
            + C[2] * wavelengths ** C[3]
            + C[4] * wavelengths ** C[5]
            + C[6] * wavelengths ** C[7]
            + C[8] * wavelengths ** C[9]
            + C[10] * wavelengths ** C[11]
            + C[12] * wavelengths ** C[13]
            + C[14] * wavelengths ** C[15]
            + C[16] * wavelengths ** C[17]
        )

    print(f"Invalid formula_number: {formula_number}")
    return np.nan


def AGF_refractive_index(
    wavelengths: ScalarQuantity | ArrayQuantity | float | list[float] | NDArray,
    coefficients: list[float],
    formula_number: int,
) -> float | np.ndarray:
    """
    These formulae are defined by Zemax, and indexed in the AGF files with an integer.

    From AGF file documentation:

    NM: Glass name and other data. The dispersion formula number is:
    1 for Schott
    2 for Sellmeier 1
    3 for Herzberger
    4 for Sellmeier 2
    5 for Conrady
    6 for Sellmeier 3
    7 for Handbook of Optics 1
    8 for Handbook of Optics 2
    9 for Sellmeier 4
    10 for Extended
    11 for Sellmeier 5
    12 for Extended 2
    """

    if isinstance(wavelengths, Quantity):
        try:
            wavelengths: float | list[float] = wavelengths.to(u.um).value
        except UnitConversionError as e:
            raise e  # Wrong units, not compatible with units of wavelength
        except AttributeError as e:
            # Not an astropy.units.Quantity after all
            raise e

    if isinstance(wavelengths, list):
        wavelengths: NDArray = np.array(wavelengths)

    # Pad the coefficients with zeroes at index zero and to the maximum length required
    # by the longest formula
    C: list[float] = [0] * 11  # ty:ignore[invalid-assignment]
    # C[0] = 0
    for i, c in enumerate(coefficients, start=1):
        C[i] = c

    if formula_number == 1:  # Schott formula
        assert len(C) >= 6, "Schott formula requires at least 6 coefficients"  # noqa: PLR2004
        return np.sqrt(
            C[1]
            + C[2] * wavelengths**2
            + C[3] * wavelengths ** (-2)
            + C[4] * wavelengths ** (-4)
            + C[5] * wavelengths ** (-6)
            + C[6] * wavelengths ** (-8)
        )

    if formula_number == 2:  # Sellmeier 1  # noqa: PLR2004
        return np.sqrt(
            1
            + (C[1] * wavelengths**2) / (wavelengths**2 - C[2])
            + (C[3] * wavelengths**2) / (wavelengths**2 - C[4])
            + (C[5] * wavelengths**2) / (wavelengths**2 - C[6])
        )

    if formula_number == 3:  # Herzberger  # noqa: PLR2004
        L: float | NDArray = wavelengths**2 - 0.0281

        return (
            C[1]
            + C[2] * L
            + C[3] * L**2
            + C[4] * wavelengths**2
            + C[5] * wavelengths**4
            + C[6] * wavelengths**6
        )

    if formula_number == 4:  # Sellmeier 2  # noqa: PLR2004
        # Only four coefficients
        return np.sqrt(
            1
            + (C[1] * wavelengths**2) / (wavelengths**2 - C[2] ** 2)
            + (C[3] * wavelengths**2) / (wavelengths**2 - C[4] ** 2)
        )

    if formula_number == 5:  # Conrady  # noqa: PLR2004
        # Only 3 coefficients
        return C[1] + C[2] / wavelengths + C[3] / wavelengths**3.5

    if formula_number == 6:  # Sellmeier 3  # noqa: PLR2004
        # 6 Coefficients
        return np.sqrt(
            1
            + (C[1] * wavelengths**2) / (wavelengths**2 - C[2])
            + (C[3] * wavelengths**2) / (wavelengths**2 - C[4])
            + (C[5] * wavelengths**2) / (wavelengths**2 - C[6])
        )

    if formula_number == 7:  # Handbook of Optics 1  # noqa: PLR2004
        return np.sqrt(C[1] + C[2] / (wavelengths**2 - C[3]) + C[4] * wavelengths**2)

    if formula_number == 8:  # Handbook of Optics 2  # noqa: PLR2004
        return np.sqrt(
            C[1]
            + (C[2] * wavelengths**2) / (wavelengths**2 - C[3])
            - C[4] * wavelengths**2
        )

    if formula_number == 9:  # Sellmeier 4  # noqa: PLR2004
        return np.sqrt(
            C[1]
            + (C[2] * wavelengths**2) / (wavelengths**2 - C[3])
            + (C[4] * wavelengths**2) / (wavelengths**2 - C[5])
        )

    if formula_number == 10:  # Extended #noqa: PLR2004
        return (
            C[1]
            + C[2] * wavelengths**2
            + C[3] * wavelengths ** (-2)
            + C[4] * wavelengths ** (-4)
            + C[5] * wavelengths ** (-6)
            + C[6] * wavelengths ** (-8)
            + C[7] * wavelengths ** (-10)
            + C[8] * wavelengths ** (-12)
        )

    if formula_number == 11:  # Sellmeier 5  # noqa: PLR2004
        return np.sqrt(
            1
            + (C[1] * wavelengths**2) / (wavelengths**2 - C[2])
            + (C[3] * wavelengths**2) / (wavelengths**2 - C[4])
            + (C[5] * wavelengths**2) / (wavelengths**2 - C[6])
            + (C[7] * wavelengths**2) / (wavelengths**2 - C[8])
            + (C[9] * wavelengths**2) / (wavelengths**2 - C[10])
        )

    if formula_number == 12:  # Extended 2  # noqa: PLR2004
        return np.sqrt(
            C[1]
            + C[2] * wavelengths**2
            + C[3] * wavelengths ** (-2)
            + C[4] * wavelengths ** (-4)
            + C[5] * wavelengths ** (-6)
            + C[6] * wavelengths ** (-8)
            + C[7] * wavelengths ** (4)
            + C[8] * wavelengths ** (6)
        )

    if formula_number == 13:  # Extended 3  # noqa: PLR2004
        return np.sqrt(
            C[1]
            + C[2] * wavelengths**2
            + C[3] * wavelengths**4
            + C[4] * wavelengths ** (-2)
            + C[5] * wavelengths ** (-4)
            + C[6] * wavelengths ** (-6)
            + C[7] * wavelengths ** (-8)
            + C[8] * wavelengths ** (-10)
            + C[9] * wavelengths ** (-12)
        )

    return 0


def main() -> None: ...


if __name__ == "__main__":
    main()
