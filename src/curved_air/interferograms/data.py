"""
curved_air

interferograms

data

This module contains classes for holding data from imported interferograms. These
classes are importable directly from the curved_air.interferograms module via
__init__.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class Metadata:
    """
    A simple subclass of dict to hold metadata extracted from interferogram files.
    This can be extended in the future to include validation, default values, or
    convenience methods for accessing specific metadata fields.
    """

    raw_dict: dict[str, Any] | None = None

    file: str | None = None

    type: Literal["SUR"] | None = None
    pixel_size: float | None = None
    wavelength: float | None = None

    grid_size: tuple[int, int] | None = None
    rows: int | None = None
    cols: int | None = None
    origin: tuple[float, float] | None = None

    sample_size: int | None = None
    null_data_value: float | None = None

    scale_factor: float | None = None

    obliquity_factor: float | None = None

    phase_resolution: int | None = None
    phase_resolution_elements: int | None = None

    total_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.raw_dict is not None:
            if self.file is None:
                self.file: str | None = self.raw_dict.get("FILE", None)

            grd: Any = self.raw_dict.get("GRD", None)
            if grd is None:
                grd_x: int | None = self.raw_dict.get("Cols", None)
                grd_y: int | None = self.raw_dict.get("Rows", None)
                grd: tuple[int, int] | None = (
                    (grd_x, grd_y) if grd_x is not None and grd_y is not None else None
                )
            if isinstance(grd, tuple) and len(grd) == 2:  # noqa: PLR2004
                self.grid_size: tuple[int, int] = (int(grd[0]), int(grd[1]))
                self.rows: int = self.grid_size[0]
                self.cols: int = self.grid_size[1]

            if self.type is None:
                self.type: str | None = self.raw_dict.get("TYPE", None)

            if self.pixel_size is None:
                self.pixel_size: Any = self.raw_dict.get("PixelSize", None)

            if self.wavelength is None:
                self.wavelength: Any = self.raw_dict.get("WVL", None)

            if self.sample_size is None:
                self.sample_size: Any = self.raw_dict.get("SSZ", None)

            if self.null_data_value is None:
                self.null_data_value: Any = self.raw_dict.get("NDA", None)

    def __dict__(self) -> dict[str, Any]:
        return {
            "FILE": self.file,
            "GRD": self.grid_size,
            "TYPE": self.type,
            "WVL": self.wavelength,
            "SSZ": self.sample_size,
            "NDA": self.null_data_value,
            "PixelSize": self.pixel_size,
            "Rows": self.rows,
            "Cols": self.cols,
        }


@dataclass
class InterferogramData:
    """
    A simple container for interferogram data and its associated metadata.

    This can be extended in the future to include methods for processing or visualizing
    the data, or for converting between different formats.
    """

    data: NDArray
    metadata: Metadata | None = None


class PhaseData(InterferogramData):
    """
    A subclass of InterferogramData to indicate that the data represents phase
    information
    """


class IntensityData(InterferogramData):
    """
    A subclass of InterferogramData to indicate that the data represents intensity
    information
    """


@dataclass
class GridSagData:
    """
    A simple container for grid sag data and its associated metadata.

    This can be extended in the future to include methods for processing or visualizing
    the data, or for converting between different formats.
    """

    comment: str
    data: NDArray
    metadata: Metadata | None = None
