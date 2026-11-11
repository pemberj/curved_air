"""
curved_air

interferograms

utils

This module contains various small and not-so-small utilities and convenience functions
used throughout the curved_air.interferograms code.
"""

import logging
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from astropy import units as u
from astropy.units import Quantity
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from PIL import Image

from curved_air.geometry import Shape

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


def rms(data: NDArray) -> float:
    """
    The root-mean-square value of an input array, across all axes
    """
    return np.sqrt(np.mean(np.power(data, 2)))


def show_image(
    image: NDArray,
    title: str,
    vmin: float,
    vmax: float,
) -> None:
    """
    A simple function to display a Matplotlib/PyPlot image of some input 2d data.
    """

    im = plt.imshow(image, cmap="jet", vmin=vmin, vmax=vmax)
    plt.colorbar(im)
    plt.title(title)
    plt.show()


def resample(
    data: NDArray,
    new_shape: tuple[int, int] | None = None,
    plot: bool = False,
) -> NDArray:
    """
    Resample the input data to a new shape.

    This is useful for reducing the size of the data for use within Zemax, where the
    high spatial resolution of the interferogram (in the pupil) is rarely required, and
    can result in a significant increase in computation time.
    """

    if not new_shape:
        logger.info("No resample size provided, using default (100, 100)")
        new_shape: tuple[int, int] = (100, 100)

    Imin = np.nanmin(data)
    Imax = np.nanmax(data)

    if plot:
        show_image(data, "Before resampling", vmin=Imin, vmax=Imax)

    data = np.array(Image.fromarray(data).resize(new_shape))

    if plot:
        show_image(data, "After resampling", vmin=Imin, vmax=Imax)

    return data


def line_profile(
    data: NDArray,
    start: tuple[float, float],
    end: tuple[float, float],
    num_points: int | None = None,
    half_width: int = 0,
) -> tuple[NDArray, NDArray]:
    """
    Sample a 2d array along a straight line between two points using bilinear
    interpolation.

    `start` and `end` are (row, col) pixel coordinates (floats, need not land on exact
    pixel centers). Returns a tuple of `(distance, values)`, where `distance` is the
    distance along the line in pixel units, measured from `start`.

    If `half_width` is greater than zero, the profile is additionally sampled along a
    band of parallel lines offset perpendicular to the main line, up to `half_width`
    pixels to either side. The median across that band is taken for each point along the
    line, which smooths the profile (useful for data with noise at high spatial
    frequencies).
    """

    # Unpack tuples
    row_start, col_start = start
    row_end, col_end = end

    length: float = float(np.hypot(row_end - row_start, col_end - col_start))
    if num_points is None:
        num_points: int = max(2, round(length) + 1)

    if half_width > 0:
        if length > 0:
            perp_row: float = -(col_end - col_start) / length
            perp_col: float = (row_end - row_start) / length
        else:
            perp_row, perp_col = 0.0, 0.0

        # Trace profiles along a set of parallel lines, up to +/- half_width
        offsets: NDArray = np.arange(-half_width, half_width + 1)
        band_values: NDArray = np.empty((offsets.size, num_points))
        for band_index, offset in enumerate(offsets):
            shifted_start: tuple[float, float] = (
                row_start + offset * perp_row,
                col_start + offset * perp_col,
            )
            shifted_end: tuple[float, float] = (
                row_end + offset * perp_row,
                col_end + offset * perp_col,
            )

            # Recursive function call!
            distance, band_values[band_index] = line_profile(
                data, shifted_start, shifted_end, num_points=num_points
            )

        # Get the median across all of the shifted profiles
        values: NDArray = np.nanmedian(band_values, axis=0)

        return distance, values

    fraction: NDArray = np.linspace(0.0, 1.0, num_points)
    rows: NDArray = row_start + fraction * (row_end - row_start)
    cols: NDArray = col_start + fraction * (col_end - col_start)

    num_rows, num_cols = data.shape
    rows_clipped: NDArray = np.clip(rows, 0, num_rows - 1)
    cols_clipped: NDArray = np.clip(cols, 0, num_cols - 1)

    row_floor: NDArray = np.floor(rows_clipped).astype(int)
    col_floor: NDArray = np.floor(cols_clipped).astype(int)
    row_ceil: NDArray = np.clip(row_floor + 1, 0, num_rows - 1)
    col_ceil: NDArray = np.clip(col_floor + 1, 0, num_cols - 1)

    row_frac: NDArray = rows_clipped - row_floor
    col_frac: NDArray = cols_clipped - col_floor

    top: NDArray = (
        data[row_floor, col_floor] * (1 - col_frac)
        + data[row_floor, col_ceil] * col_frac
    )

    bottom: NDArray = (
        data[row_ceil, col_floor] * (1 - col_frac) + data[row_ceil, col_ceil] * col_frac
    )

    values: NDArray = top * (1 - row_frac) + bottom * row_frac

    distance: NDArray = fraction * length

    return distance, values


def sag2phase(
    sag_data: NDArray,
    sag_unit: u.Unit = u.um,
    wavelength: Quantity | float = 632.8 * u.nm,
) -> NDArray:  # floats
    """
    Convert sag data to surface phase data.
    """

    if isinstance(sag_data[0][0], Quantity):
        quantity_flag: bool = True
    else:
        assert isinstance(
            sag_data, float | int | np.floating | np.integer | np.ndarray
        ), "Input sag_data must be a numpy array of floats or ints"
        quantity_flag: bool = False

    if quantity_flag:
        sag_data: Quantity = (sag_data.value * sag_unit).value
    else:
        sag_data: Quantity = (sag_data * sag_unit).value

    if isinstance(wavelength, float):
        logger.info("Assuming wavelength is in nanometers.")
        wavelength: Quantity = (wavelength * u.nm).to(sag_unit).value
    else:
        wavelength: Quantity = wavelength.to(sag_unit).value

    # Convert from sag to waves of phase
    return sag_data * (2 * np.pi) / wavelength


def phase2sag(
    phase_data: NDArray,  # ints
    wavelength: Quantity | float = 632.8 * u.nm,
) -> NDArray:  # floats
    """
    Convert phase data to surface sag data.

    This function assumes that the phase data is in units of waves and converts
    it to sag data in micrometers.

    TODO: Check the logic here, match the sag2phase function above
    """

    if isinstance(wavelength, float):
        logger.info("Assuming wavelength is in nanometers.")
        wavelength: Quantity = wavelength * u.nm

    # Convert from waves of phase to sag
    sag: Quantity = phase_data * (wavelength / (2 * np.pi))

    return sag.to(u.mm).value


def mask(data: NDArray, mask_geometry: Shape) -> None:
    """
    Apply an already-defined mask to a 2D data array. The mask is passed as a Shape
    object, with the geometry defined independent of any quantised grid. The validity of
    each pixel in the dat array is computed based on whether the pixel center lies
    within the mask geometry.

    I'm not sure yet how I will expose creation of the mask geometry to the user in the
    GUI
    """

    raise NotImplementedError("Mask functionality is not yet implemented.")


def process(data: NDArray, steps: list[str] | None) -> NDArray:
    """
    A simple wrapper function to take a list of processing steps (rotations and mirrors)
    and apply them, in order, to a 2d data array.
    """

    if steps is None:
        return data

    # Processing steps
    for step in steps:
        match step:
            case x if "ROT" in x:
                direction: str = step[3:]
                if direction == "CW":
                    data: NDArray = np.rot90(data, k=-1)
                elif direction == "CCW":
                    data: NDArray = np.rot90(data, k=1)
            case x if "MIRROR" in x:
                axis: str = step[6:]
                if axis.lower() in ("h", "horizontal"):
                    data: NDArray = np.fliplr(data)
                elif axis.lower() in ("v", "vertical"):
                    data: NDArray = np.flipud(data)
            case _:
                logger.warning(f"Unknown processing step: {step}")

    return data


def transform(
    data: NDArray,
    flip_direction: str | None = None,
    rotation_angle: float | None = None,
    crop_to_valid: bool = False,
    resample_size: int | tuple[int, int] | None = None,
) -> NDArray:
    """
    Transform the relevant data based on flips and rotations selected.
    """

    if crop_to_valid:
        # Crop the data to the valid region (non-NaN values)
        valid_mask: NDArray = ~np.isnan(data)
        zero_mask: NDArray = data != 0
        if np.any(valid_mask & zero_mask):
            rows: NDArray = np.where(np.any(valid_mask & zero_mask, axis=1))[0]
            cols: NDArray = np.where(np.any(valid_mask & zero_mask, axis=0))[0]
            data: NDArray = data[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]

    if flip_direction is not None:
        if flip_direction == "Horizontal":
            data: NDArray = data[:, ::-1]
        elif flip_direction == "Vertical":
            data: NDArray = data[::-1, :]
        elif flip_direction == "Both":
            data: NDArray = data[::-1, ::-1]

    if rotation_angle is not None:
        if rotation_angle == 0:  # noqa: PLR2004
            pass
        elif rotation_angle == 90:  # noqa: PLR2004
            data: NDArray = data.T[::-1, :]
        elif rotation_angle == 180:  # noqa: PLR2004
            data: NDArray = data[::-1, ::-1]
        elif rotation_angle == 270:  # noqa: PLR2004
            data: NDArray = data.T[:, ::-1]
        else:
            # Arbitrary-angle rotation (counter-clockwise), preserving array shape
            data = np.array(
                Image.fromarray(data.astype(np.float32)).rotate(
                    angle=rotation_angle,
                    resample=Image.Resampling.NEAREST,
                    expand=False,
                    fillcolor=np.nan,
                ),
            )

    if resample_size is not None:
        # Resample to an explicit shape when provided; int keeps square behavior.
        if isinstance(resample_size, tuple):
            data: NDArray = resample(data, new_shape=resample_size)
        else:
            data: NDArray = resample(data, new_shape=(resample_size, resample_size))

    return data


def x_offset(data: NDArray, offset: int) -> NDArray:
    """
    Shift the data in the x-direction by a given offset.
    Positive offsets shift to the right, negative offsets shift to the left.

    The output canvas is expanded with NaN padding so data is not clipped.
    """

    if offset == 0:  # noqa: PLR2004
        return data

    if offset > 0:
        data = np.hstack((np.full((data.shape[0], offset), np.nan), data))
    else:
        data = np.hstack((data, np.full((data.shape[0], -offset), np.nan)))

    return data


def y_offset(data: NDArray, offset: int) -> NDArray:
    """
    Shift the data in the y-direction by a given offset.
    Positive offsets shift down, negative offsets shift up.

    The output canvas is expanded with NaN padding so data is not clipped.
    """

    if offset == 0:  # noqa: PLR2004
        return data

    if offset > 0:
        data = np.vstack((np.full((offset, data.shape[1]), np.nan), data))
    else:
        data = np.vstack((data, np.full((-offset, data.shape[1]), np.nan)))

    return data


def save_plot(
    data: NDArray,
    filename: Path | str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """
    A wrapper function to save a plot of some input data
    """
    fig: plt.Figure = plt.figure(figsize=(8, 7))
    ax: plt.Axes = fig.gca()

    ax.imshow(data)
    ax.set_xlim(0, data.shape[1])
    ax.set_ylim(0, data.shape[0])

    cbar: plt.Colorbar = plt.colorbar(ax.imshow(data), ax=ax)
    cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.0f"))
    cbar.set_ticks(np.linspace(np.nanmin(data), np.nanmax(data), num=6))
    cbar.set_label("Surface deviation [nm]")

    valid_mask: NDArray = data != 0

    PTV: float = np.max(data[valid_mask]) - np.min(data[valid_mask])
    PTV_waves: float = PTV / 632.8
    if PTV_waves >= 0.5:  # noqa: PLR2004
        PTV_text: str = f"PTV: {PTV:.3g}nm = {PTV_waves:.3g}$\\lambda$"
    else:
        PTV_frac: float = 1 / PTV_waves
        PTV_text: str = f"PTV: {PTV:.3g}nm = $\\lambda$/{PTV_frac:.1f}"

    RMS: float = rms(data[valid_mask])
    RMS_waves: float = RMS / 632.8
    if RMS_waves >= 0.5:  # noqa: PLR2004
        RMS_text: str = f"RMS: {RMS:.3g}nm = {RMS_waves:.3g}$\\lambda$"
    else:
        RMS_frac: float = 1 / RMS_waves
        RMS_text: str = f"RMS: {RMS:.3g}nm = $\\lambda$/{RMS_frac:.1f}"

    ax.text(
        x=0.95,
        y=0.95,
        s=PTV_text,
        transform=ax.transAxes,
        color="w",
        fontsize=20,
        ha="right",
        va="top",
    )

    ax.text(
        x=0.95,
        y=0.9,
        s=RMS_text,
        transform=ax.transAxes,
        color="w",
        fontsize=20,
        ha="right",
        va="top",
    )

    if metadata is None:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    else:
        pixel_size: float | None = metadata.get("PixelSize", None)  # in mm

        if pixel_size is None:
            logger.warning("Pixel size not found in metadata, cannot label axes in mm.")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
        else:
            # Label x and y axes in mm
            ax.set_xticks(np.linspace(0, data.shape[1], num=6))
            x_labels: list[str] = [
                f"{tick * pixel_size:.1f}" for tick in ax.get_xticks()
            ]
            ax.set_xticklabels(x_labels)
            ax.set_xlabel("X [mm]")

            ax.set_yticks(np.linspace(0, data.shape[0], num=6))
            y_labels: list[str] = [
                f"{tick * pixel_size:.1f}" for tick in ax.get_yticks()
            ]
            ax.set_yticklabels(y_labels)
            ax.set_ylabel("Y [mm]")

    plt.savefig(f"{filename}", dpi=300)


def native_endian_array(data: NDArray) -> NDArray:
    """Ensure arrays use native byte order for pandas/pyarrow conversion."""

    if data.dtype.byteorder == ">":
        return data.byteswap().view(data.dtype.newbyteorder("="))
    if data.dtype.byteorder == "=" and not np.little_endian:
        return data.byteswap().view(data.dtype.newbyteorder("="))
    return data


def flag_edge_zeroes(data: NDArray) -> NDArray:
    """Mark only zero-valued regions connected to the array border as invalid."""

    zero_mask: NDArray = data == 0.0
    if not np.any(zero_mask):
        return data

    edge_mask: NDArray = np.zeros_like(zero_mask, dtype=bool)
    rows, cols = zero_mask.shape
    queue: deque[tuple[int, int]] = deque()

    def enqueue(row: int, col: int) -> None:
        if zero_mask[row, col] and not edge_mask[row, col]:
            edge_mask[row, col] = True
            queue.append((row, col))

    for col in range(cols):
        enqueue(0, col)
        enqueue(rows - 1, col)

    for row in range(rows):
        enqueue(row, 0)
        enqueue(row, cols - 1)

    while queue:
        row, col = queue.popleft()
        for next_row, next_col in (
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        ):
            if 0 <= next_row < rows and 0 <= next_col < cols:
                enqueue(next_row, next_col)

    data = data.copy()
    data[edge_mask] = np.nan
    return data
