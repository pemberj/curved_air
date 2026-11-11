"""
curved_air

interferograms

convert

Various functions for reading in or writing out interferometer grid data
Includes functions to parse Zygo- and Fisba-exported tabulated data, and a function to
write out to Zemax-compatible grid phase or sag files.
"""

from __future__ import annotations

import logging
import re
import struct
from pathlib import Path
from typing import TYPE_CHECKING, Any, overload

import chardet
import numpy as np
from astropy import units as u
from astropy.units import Quantity
from matplotlib import pyplot as plt

from curved_air.interferograms import (
    GridSagData,
    IntensityData,
    InterferogramData,
    Metadata,
    PhaseData,
)
from curved_air.interferograms.utils import phase2sag

if TYPE_CHECKING:
    from numpy.typing import NDArray

VALID = 0
INVALID = 1

zemax_units: dict[str, int] = {"mm": 0, "cm": 1, "in": 2, "m": 3}

logger: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


def from_zygo(
    file: str | Path,
    which: str = "both",
    plot: bool = False,
) -> list[InterferogramData]:
    """
    Extract data from a Zygo binary data file.

    This is based on `metropro_extract.py`
    Original author: Nicholas Nell:
    https://gist.github.com/cosmonaut/af9839c7b6492eceaf3ac9cc96238962

    All of the logic for processing the binary data is based on the above script.
    """

    to_return: list[InterferogramData] = []

    if which not in ["intensity", "phase", "both"]:
        raise ValueError(
            "Invalid 'which' argument. Must be 'intensity', 'phase', or 'both' (default)."
        )

    if isinstance(file, str):
        file = Path(file)

    # Read file in raw bytes
    with file.open(mode="rb") as f:
        data: bytes = f.read()

    # "Nominal" header size in bytes
    hdr_size: int = 834

    if len(data) < 834:  # noqa: PLR2004
        raise ValueError("File too small to be MetroPro DAT format!")

    # Read first 4 bytes to determine "magic number"
    magic_number: int = int.from_bytes(data[0:4], byteorder="big")
    logger.debug(f"MAGIC NUMBER: 0x{magic_number:x}")

    if magic_number == 0x881B036F:  # noqa: PLR2004
        logger.debug("Detected header format 1")
    elif magic_number == 0x881B0370:  # noqa: PLR2004
        logger.debug("Detected header format 2")
    elif magic_number == 0x881B0371:  # noqa: PLR2004
        logger.debug("Detected header format 3")
        # Header size is 4096 bytes for format 3
        hdr_size = 4096
    else:
        raise ValueError(
            f"No valid magic number detected, incorrect dat file format!\nMAGIC NUMBER: 0x{magic_number:x}"
        )

    hdr_format: int = struct.unpack(">h", data[4:6])[0]
    logger.info(f"HEADER FORMAT: {hdr_format}")

    hsize: int = struct.unpack(">l", data[6:10])[0]
    logger.info(f"HEADER SIZE: {hsize}")

    # If the intensity data matrix is present then it begins immediately after the header
    if which == "intensity" or which == "both":
        logger.info(f"\n{'Processing intensity data ':.<50}")
        # Intensity data matrix header variables
        ac_org_x: int = struct.unpack(">h", data[48:50])[0]
        ac_org_y: int = struct.unpack(">h", data[50:52])[0]
        ac_width: int = struct.unpack(">h", data[52:54])[0]
        ac_height: int = struct.unpack(">h", data[54:56])[0]
        ac_n_buckets: int = struct.unpack(">h", data[56:58])[0]
        ac_range: int = struct.unpack(">h", data[58:60])[0]
        ac_n_bytes: int = struct.unpack(">l", data[60:64])[0]

        if ac_n_bytes > 0:
            logger.info("INTENSITY DATA MATRIX HEADER DATA")
            logger.info(f"X/Y ORIGIN: {ac_org_x}, {ac_org_y}")
            logger.info(f"WIDTH: {ac_width}")
            logger.info(f"HEIGHT: {ac_height}")
            logger.info(f"N BUCKETS: {ac_n_buckets}")
            logger.info(f"RANGE: {ac_range}")
            logger.info(f"AC TOTAL BYTES: {ac_n_bytes}")

            wavelength: float = struct.unpack(">f", data[168:172])[0]
            wavelength = (wavelength * u.m).to(u.nm)  # Convert to nm

            intensity_metadata: dict[str, Any] = {
                "FILE": file,
                "GRD": (ac_width, ac_height),
                "TYPE": "SUR",
                "WVL": wavelength,
                "SSZ": ac_n_buckets,  # Sample size
                "NDA": ...,  # Null data (identifies points that should be ignored)
            }

            intensity: NDArray = np.frombuffer(
                data[hdr_size : hdr_size + ac_n_bytes], dtype=">u2"
            ).copy()
            intensity: NDArray = intensity.reshape(ac_height, ac_width)

            # Supress 0xffff (masked) bytes
            ind: NDArray = intensity > ac_range  # booleans!
            intensity[ind] = 0

            if plot:
                fig1: plt.Figure = plt.figure("Intensity map", dpi=96)
                ax1: plt.Axes = fig1.gca()
                ax1.set_title("Intensity Map")
                img1: plt.AxesImage = ax1.imshow(intensity)
                fig1.colorbar(img1)
                # plt.savefig(file + ".intensity.png")

            to_return.append(
                IntensityData(
                    data=intensity, metadata=Metadata(raw_dict=intensity_metadata)
                )
            )

        else:
            logger.info("No intensity data matrix found in the file.")
            # to_return.append((None, None))

    if which == "phase" or which == "both":
        logger.info(f"{'Processing phase data ':.<50}")
        # Phase data matrix header variables
        cn_org_x: int = struct.unpack(">h", data[64:66])[0]
        cn_org_y: int = struct.unpack(">h", data[66:68])[0]
        cn_width: int = struct.unpack(">h", data[68:70])[0]
        cn_height: int = struct.unpack(">h", data[70:72])[0]
        cn_n_bytes: int = struct.unpack(">l", data[72:76])[0]
        intf_scale_factor: float = struct.unpack(">f", data[164:168])[0]
        wavelength_in: float = struct.unpack(">f", data[168:172])[0]
        obliquity_factor: float = struct.unpack(">f", data[176:180])[0]
        phase_res: int = struct.unpack(">h", data[218:220])[0]

        match hdr_format:
            case 1:
                if phase_res == 0:
                    R: int = 4096
                elif phase_res == 1:
                    R: int = 32768
                else:
                    logger.error("INVALID PHASE RES")

            case 2 | 3:
                if phase_res == 0:
                    R: int = 4096
                elif phase_res == 1:
                    R: int = 32768
                elif phase_res == 2:  # noqa: PLR2004
                    R: int = 131072
                else:
                    logger.error("INVALID PHASE RES")

            case _:
                logger.error("Invalid header format.")

        # If phase data matrix is present, then process it.
        if cn_n_bytes > 0:
            logger.info("PHASE MATRIX HEADER DATA")
            logger.info(f"X/Y ORIGIN: {cn_org_x}, {cn_org_y}")
            logger.info(f"WIDTH / HEIGHT: {cn_width}, {cn_height}")
            logger.info(f"INTF_SCALE_FACTOR: {intf_scale_factor}")
            logger.info(f"WAVELENGTH IN: {(wavelength_in * u.m).to(u.nm).value:.6g}")
            logger.info(f"OBLIQUITY FACTOR: {obliquity_factor}")
            logger.info(f"PHASE RES: {phase_res}")
            logger.info(f"PHASE RES ELEMENTS (R): {R}")
            logger.info(f"CN TOTAL BYTES: {cn_n_bytes}")

            phase_metadata: dict[str, Any] = {
                "FILE": file,
                "GRD": (cn_width, cn_height),
                "TYPE": "SUR",
                "WVL": (wavelength_in * u.m).to(u.nm),  # Measurement wavelength
                "SSZ": ...,  # Sample size
                "NDA": ...,  # Null data (identifies points that should be ignored)
            }

            phase: NDArray = np.frombuffer(
                data[hdr_size + ac_n_bytes : hdr_size + ac_n_bytes + cn_n_bytes],
                dtype=">i4",
            ).copy()
            phase: NDArray = phase.reshape(cn_height, cn_width)
            ind: NDArray = phase >= 0x7FFFFFF8  # booleans!  # noqa: PLR2004
            phase[ind] = 0
            # This is now the phase in units of "zygos"

            # Convert to units of waves
            phase: NDArray = phase * ((intf_scale_factor * obliquity_factor) / R)
            # It is also possible to use the formula (W*S*O)/R to have the
            # units in meters where W is wavelength is given in the
            # "wavelength_in" header variable in units of meters.

            if plot:
                fig2: plt.Figure = plt.figure("Phase map", dpi=96)
                ax2: plt.Axes = fig2.gca()
                img2: plt.AxesImage = ax2.imshow(phase)
                fig2.colorbar(img2)
                ax2.set_title("Phase Map (waves)")
                # plt.savefig(file + ".phase.png")

            to_return.append(
                PhaseData(data=phase, metadata=Metadata(raw_dict=phase_metadata))
            )

        else:
            logger.info("No phase data matrix found in the file.")
            # to_return.append((None, None))

    if len(to_return) == 0:
        raise ValueError("No data extracted from file.")

    plt.show()
    logger.info("")

    return to_return


def from_fisba_binary(file: str | Path) -> InterferogramData:
    """
    TODO: Parse header, comments, and data
    TODO: Look up Fisba file format specifications
    """

    raise NotImplementedError("from_fisba_binary is not yet implemented.")


def from_fisba_csv(file: str | Path) -> InterferogramData:
    r"""
    Read a 2ES interferogram text export like the provided sample.

    Expected structure (loosely):
      - Header lines (may include "PixelSize=...", "Row=... Col=...", etc.)
      - A column header line starting with "Row\\Col" followed by integer column indices
      - Then data lines: "<row_index>\\t<val0>\\t<val1>..."; missing/blank cells -> NaN

    Returns
    -------
    data : NDArray, shape (n_rows, n_cols), dtype float
        2D array indexed by row/col (0..max header col), filled with NaN where absent.
        Note: Row indices in the file can be in any order; the array is sized based on
        the max row/col indices encountered and filled accordingly.
    metadata : dict
        Parsed metadata (e.g., PixelSize, declared Row/Col dimensions, etc.).
    """
    metadata: dict[str, int | float] = {}
    col_indices: list[int] | None = None

    # Holds sparse values until we know shape
    values: dict[tuple[int, int], float] = {}  # (row, col) -> float
    max_r = -1
    max_c = -1

    # Capture common metadata patterns
    re_pixelsize: re.Pattern[str] = re.compile(
        r"^\s*PixelSize\s*=\s*([+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)\s*$",
    )
    re_dim: re.Pattern[str] = re.compile(
        r"^\s*Row\s*=\s*(\d+)\s*Col\s*=\s*(\d+)\s*$",
    )

    if isinstance(file, str):
        file = Path(file)

    with file.open(mode="r", encoding="utf-8", errors="replace") as f:
        logger.info(f"Reading Fisba CSV file: {file}")
        logger.info(f"{'-' * 80}")

        for i, raw_line in enumerate(f):
            line: str = raw_line.rstrip("\n")

            if i == 0:
                # Output the first line (commonly a comment) to the log
                logger.info(f"{line}")

            # Skip fully empty lines
            if not line.strip():
                continue

            # Metadata
            m: re.Match[str] | None = re_pixelsize.match(line)
            if m:
                logger.info(f"Found PixelSize in line {i}: {line}")
                metadata["PixelSize"] = float(m.group(1))
                continue

            m: re.Match[str] | None = re_dim.match(line)
            if m:
                logger.info(f"Found Row/Col in line {i}: {line}")
                metadata["Rows"] = int(m.group(1))
                metadata["Cols"] = int(m.group(2))
                continue

            # Column header
            if line.lstrip().startswith("Row\\Col"):
                parts: list[str] = line.split("\t")
                # after "Row\Col" come integer column indices
                col_indices: list[int] = []
                for p in parts[1:]:
                    p: str = p.strip()  # noqa: PLW2901
                    if p == "":
                        continue
                    try:
                        col_indices.append(int(p))
                    except ValueError:
                        # ignore non-integer clutter
                        pass
                if col_indices:
                    max_c: int = max(max_c, *col_indices)
                continue

            # Data row lines: first field is row index
            parts: list[str] = line.split("\t")
            if len(parts) == 0:
                continue

            row_str: str = parts[0].strip()
            try:
                r = int(row_str)
            except ValueError:
                # Not a data row; ignore
                continue

            max_r: int = max(max_r, r)

            # If we haven't seen the "Row\Col" header, fall back to sequential columns
            if col_indices is None:
                # columns 0..N-1 for remaining fields
                for j, cell in enumerate(parts[1:]):
                    s: str = cell.strip()
                    if s == "":
                        continue
                    try:
                        v = float(s)
                    except ValueError:
                        continue
                    values[(r, j)] = v
                    max_c: int = max(max_c, j)
                continue

            # Map each cell to the corresponding column index
            for j, cell in enumerate(parts[1:]):
                if j >= len(col_indices):
                    break
                c: int = col_indices[j]
                s: str = cell.strip()
                if s == "":
                    continue  # missing -> NaN
                try:
                    v = float(s)
                except ValueError:
                    continue
                values[(r, c)] = v
                max_c: int = max(max_c, c)
            continue

            logger.info(f"Line {i}: {line}")

    # Decide final shape:
    # Prefer declared sizes if present and consistent; otherwise use encountered maxima.
    n_rows: int = metadata.get("Rows", max_r + 1)  # ty:ignore[invalid-assignment]
    n_cols: int = metadata.get("Cols", max_c + 1)  # ty:ignore[invalid-assignment]

    if n_rows <= 0 or n_cols <= 0:
        raise ValueError(f"No table data found in {file!r}")

    data: NDArray = np.full((n_rows, n_cols), np.nan, dtype=float)
    for (r, c), v in values.items():
        if 0 <= r < n_rows and 0 <= c < n_cols:
            data[r, c] = v

    # return data, metadata
    return InterferogramData(data=data, metadata=Metadata(raw_dict=metadata))


def from_zemax_grid_sag(file: Path | str) -> GridSagData:
    """
    Import an existing Zemax grid sag file
    """

    raw: list[float] = []

    if isinstance(file, str):
        file = Path(file)

    with file.open("rb") as data:
        detector = chardet.UniversalDetector()
        for line in data:
            detector.feed(line)
            if detector.done:
                break
        detector.close()
    encoding: str | None = detector.result["encoding"]

    if encoding is None:
        logger.warning(f"Could not detect encoding for {file!r}, defaulting to utf-8")
        encoding = "utf-8"
    else:
        logger.info(f"Detected encoding: {encoding}")

    with file.open(mode="r", encoding=encoding) as data:
        header: list[str] = data.readlines()[:2]

        _comment = header[0].split("\n")[0]
        if _comment:
            logger.info(f"{_comment}")
        try:
            # nx ny delx dely unitflag xdec ydec
            nx, ny, delx, dely, unitflag, xdec, ydec = map(float, header[1].split(" "))
            nx = int(nx)
            ny = int(ny)
            delx = float(delx)
            dely = float(dely)
            unitflag = int(unitflag)
            data_start: int = 2
            xdec = float(xdec)
            ydec = float(ydec)

        except ValueError:
            logger.info("Trying to parse 'comment' line as header")
            logger.error(f"Invalid header format in {file!r}")

            _comment = ""
            try:
                # nx ny delx dely unitflag xdec ydec
                nx, ny, delx, dely, unitflag, xdec, ydec = map(
                    float, header[0].split(" ")
                )
                nx = int(nx)
                ny = int(ny)
                delx = float(delx)
                dely = float(dely)
                unitflag = int(unitflag)
                xdec = float(xdec)
                ydec = float(ydec)

                data_start: int = 1

            except ValueError as e:
                raise ValueError(f"Invalid header format in {file!r}") from e

        logger.info(
            f"Parsed header: nx={nx}, ny={ny}, delx={delx}, dely={dely}, unitflag={unitflag}, xdec={xdec}, ydec={ydec}"
        )

    with file.open(mode="r", encoding=encoding) as data:
        lines: list[str] = data.readlines()[data_start:]
        for line in lines:
            # z dz/dx dz/dy d2z/dxdy nodata
            z, _dzdx, _dzdy, _d2zdxdy, _nodata = line.split(" ")
            raw.append(float(z))

    # Truncate to expected size if there are extra lines; otherwise reshape will fail.
    # TODO: handle shorter than expected files
    if len(raw) != nx * ny:
        raw: list[float] = raw[: nx * ny]

    interferogram: NDArray = np.array(raw)
    interferogram: NDArray = interferogram.reshape((int(ny), int(nx)))

    Imin, Imax = np.nanmin(interferogram), np.nanmax(interferogram)
    PTV: float = Imax - Imin
    logger.info(f"Peak-to-valley: {PTV:.6g}")

    RMS: float = np.sqrt(np.nanmean(np.square(interferogram)))
    logger.info(f"RMS: {RMS:.6g}")

    # return _comment, interferogram
    return GridSagData(
        comment=_comment,
        data=interferogram,
        metadata=Metadata(
            rows=nx,
            cols=ny,
            grid_size=(nx, ny),
            pixel_size=delx,
            origin=(xdec, ydec),
        ),
    )


def to_int(
    data: InterferogramData | GridSagData | NDArray,
    metadata: Metadata | None = None,
) -> None:
    """
    Convert to INT format, which is compatible with several interferometer software
    packages, including MetroPro and Fisba.
    """


@overload
def to_zemax(
    data: InterferogramData | GridSagData | NDArray,
    file: str | Path,
    comment: str = "",
    size: float | tuple = 220.0,
    pixel_size: float | None = None,  # in mm
    decenter_x: float = 0.0,
    decenter_y: float = 0.0,
    dzdx: float = 0.0,
    dzdy: float = 0.0,
    d2zdxdy: float = 0.0,
    passed_unit: str | None = None,
    requested_unit: str | None = None,
    flag_invalid: list[float] | None = None,
    log_level: int = 0,
) -> None: ...


@overload
def to_zemax(
    data: InterferogramData | GridSagData | NDArray,
    file: None = None,
    comment: str = "",
    size: float | tuple = 220.0,
    pixel_size: float | None = None,  # in mm
    decenter_x: float = 0.0,
    decenter_y: float = 0.0,
    dzdx: float = 0.0,
    dzdy: float = 0.0,
    d2zdxdy: float = 0.0,
    passed_unit: str | None = None,
    requested_unit: str | None = None,
    flag_invalid: list[float] | None = None,
    log_level: int = 0,
) -> str: ...


def to_zemax(
    data: InterferogramData | GridSagData | NDArray,
    file: str | Path | None = None,
    comment: str = "EXPORTED FROM CURVED_AIR: https://curvedair.jakepember.com",
    size: float
    | tuple[float, float]
    | Quantity
    | tuple[Quantity, Quantity] = 220.0,  # size of the surface in mm
    pixel_size: float | None = None,  # in mm
    decenter_x: float = 0.0,
    decenter_y: float = 0.0,
    dzdx: float = 0.0,
    dzdy: float = 0.0,
    d2zdxdy: float = 0.0,
    passed_unit: str | None = None,
    requested_unit: str | None = None,
    flag_invalid: list[float] | None = None,
    log_level: int = 0,
) -> str | None:
    """
    Convert the extracted data to Zemax format.

    While Zemax allows conversion of .int interferogram files to its native .dat format,
    it only converts directly to sag data. This script allows conversion of .int files
    to both sag and phase data.

    The script will read in both the tabulated grid data and the metadata from the .int
    file, and then write out the data to a .dat file in the format that Zemax expects.

    The only piece of information that is missing from the metedata, and needs to be
    provided by the user, is the size of the interferogram surface. This is passed as
    the ROI_size parameter to the read_interferogram function.
    """

    if requested_unit is None:
        requested_unit = "mm"

    if passed_unit is None:
        passed_unit = "mm"

    if isinstance(size, Quantity):
        size: float = size.to(requested_unit).value
    elif isinstance(size, tuple) and all(isinstance(s, Quantity) for s in size):
        size: tuple[float, float] = tuple(s.to(requested_unit).value for s in size)

    if pixel_size is not None and isinstance(pixel_size, Quantity):
        pixel_size: float = pixel_size.to(requested_unit).value

    if passed_unit is not None and passed_unit not in zemax_units:
        raise ValueError(
            f"Invalid passed_unit: {passed_unit!r}. Must be one of {list(zemax_units.keys())}."
        )

    if requested_unit is not None and requested_unit not in zemax_units:
        raise ValueError(
            f"Invalid requested_unit: {requested_unit!r}. Must be one of {list(zemax_units.keys())}."
        )

    if isinstance(data, InterferogramData | GridSagData):
        data: NDArray = data.data.copy()
    else:
        data: NDArray = data.copy()

    assert isinstance(data, np.ndarray)
    assert isinstance(data[0][0], int | float | np.floating | np.integer), (
        f"Data must be a 2D array of numbers (floats or integers). {data[0][0] = }, ({type(data[0][0])})"
    )

    if requested_unit is not None and passed_unit != requested_unit:
        # Scale the data to the requested unit
        logger.warning(
            f"{passed_unit = } is not equal to {requested_unit = }. "
            "The data will be converted to the requested unit."
        )
        data: NDArray = data * (u.Unit(passed_unit).to(u.Unit(requested_unit)))

    # gridx, gridy = metadata["GRD"]
    # spacingx = np.shape(data)[1] / gridx
    # spacingy = np.shape(data)[0] / gridy

    gridy, gridx = np.shape(data)

    if pixel_size is not None:
        spacingx: float = pixel_size
        spacingy: float = pixel_size
    else:
        if isinstance(size, tuple):
            size_x, size_y = size
            spacingx: float = size_x / np.shape(data)[1]
            spacingy: float = size_y / np.shape(data)[0]
        else:
            # Warning! In this case, pixels may not be square!
            spacingx: float = size / np.shape(data)[1]
            spacingy: float = size / np.shape(data)[0]

    if log_level >= logging.INFO:
        logger.info("Converting data to Zemax format...")

    data: NDArray = data.flatten()

    output_lines: list[str] = []
    output_lines.append(f"! {comment}\r\n")
    output_lines.append(
        f"{gridx} "
        f"{gridy} "
        f"{spacingx:.5E} "
        f"{spacingy:.5E} "
        f"{zemax_units[requested_unit] if requested_unit is not None else '0'} "
        f"{decenter_x} "
        f"{decenter_y}\r\n"
    )
    for d in data:
        if np.isnan(d):
            output_lines.append(f"{0.0:.6E} {dzdx} {dzdy} {d2zdxdy} {INVALID}\r\n")
        else:
            if flag_invalid and d in flag_invalid:
                output_lines.append(f"{d:.6E} {dzdx} {dzdy} {d2zdxdy} {INVALID}\r\n")
            else:
                output_lines.append(f"{d:.6E} {dzdx} {dzdy} {d2zdxdy} {VALID}\r\n")

    if file is None:
        return "".join(output_lines)

    if isinstance(file, str):
        file = Path(file)

    with file.open(mode="w") as f:
        for line in output_lines:
            f.write(line)

    if log_level >= logging.INFO:
        logger.info(f"Data saved to {file}.")

    return None


def test() -> None:
    """
    Test function to demonstrate usage.
    """
    file: str | Path = (
        "/home/japember/Nextcloud-IvS/Zemax/MARVEL/MARVEL Camera/As-Built MARVEL Lenses/data/C1.2_SN3_00c_386CX_IRR_UC.dat"
    )
    data: list[InterferogramData] = from_zygo(file)

    for d in data:
        if isinstance(d, IntensityData):
            _intensity_metadata = d.metadata
            _intensity_data = d.data
        elif isinstance(d, PhaseData):
            _phase_metadata = d.metadata
            phase_data: NDArray = d.data

    assert phase_data is not None, "No phase data extracted from file."

    print(f"{'Raw phase data ':.<50}")
    print(f"Max: {np.nanmax(phase_data):.6f} waves")
    print(f"Min: {np.nanmin(phase_data):.6f} waves")
    print(f"PV: {np.nanmax(phase_data) - np.nanmin(phase_data):.6f} waves")
    print(f"RMS: {np.nanstd(phase_data):.6f} waves")

    fig: plt.Figure = plt.figure("Resampled")
    ax: plt.Axes = fig.gca()
    img: plt.AxesImage = ax.imshow(phase_data, cmap="jet")
    cb: plt.Colorbar = fig.colorbar(img, ax=ax)
    cb.set_label("Phase (waves)")
    ax.set_title("Resampled Data")
    plt.show()

    surface_sag: NDArray = phase2sag(phase_data)

    print(f"\n{'Computed surface sag ':.<50}")
    print(f"Max: {np.nanmax(surface_sag):.6f} mm")
    print(f"Min: {np.nanmin(surface_sag):.6f} mm")
    print(f"PV: {np.nanmax(surface_sag) - np.nanmin(surface_sag):.6f} mm")
    print(f"RMS: {np.nanstd(surface_sag):.6f} mm")

    fig2: plt.Figure = plt.figure("Surface Sag")
    ax2: plt.Axes = fig2.gca()
    img2: plt.AxesImage = ax2.imshow(surface_sag, cmap="jet")
    cb: plt.Colorbar = fig2.colorbar(img2, ax=ax2)
    cb.set_label("Sag (mm)")
    ax2.set_title("Surface Sag Data")
    plt.show()

    # to_zemax(resample(phase_data), "test_output.dat")

    data: InterferogramData = from_fisba_csv(
        "/home/japember/Nextcloud-2ES/2ES Prism Interferograms/Fisba/2ES_P1S1.txt"
    )

    fig3: plt.Figure = plt.figure("2ES Fisba Data")
    ax3: plt.Axes = fig3.gca()
    img3: plt.AxesImage = ax3.imshow(data.data, cmap="jet")
    cb: plt.Colorbar = fig3.colorbar(img3, ax=ax3)
    cb.set_label("Phase (waves)")
    ax3.set_title("2ES Fisba Data")
    plt.show()

    data: GridSagData = from_zemax_grid_sag(
        "/home/japember/Nextcloud-IvS/Zemax/MARVEL/MARVEL Camera/As-Built MARVEL Lenses/output/MARVEL_C1.1.dat"
    )

    fig4: plt.Figure = plt.figure("Zemax Grid Sag Data")
    ax4: plt.Axes = fig4.gca()
    img4: plt.AxesImage = ax4.imshow(data.data, cmap="jet")
    cb: plt.Colorbar = fig4.colorbar(img4, ax=ax4)
    cb.set_label("Sag (mm)")
    ax4.set_title("Zemax Grid Sag Data")
    plt.show()


if __name__ == "__main__":
    test()
