"""
curved_air

zemax

This module contains fucntions to generate a simple Zemax ZMX file from a `curved_air`
`EchelleSpectrograph` object. This is a work in progress and may not behave exactly as
expected.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from astropy import units as u

from curved_air.instruments.echelle import (
    DoublePassEchelleSpectrograph,
    EchelleSpectrograph,
)
from curved_air.optics.dispersers import Prism, TwoPrisms

if TYPE_CHECKING:
    from curved_air.type_hinting import ScalarQuantity


def zemax_wavelengths(
    spectrograph: EchelleSpectrograph,
) -> list[tuple[int, list[ScalarQuantity]]]:
    """Return a list of wavelengths to use in Zemax file generation"""

    number_of_configs: int = 12

    out: list[tuple[int, list[ScalarQuantity]]] = []

    # Generate all the diffraction orders: 12 spanning the full wavelength range

    max_wl_order: int = spectrograph.grating.order_number(
        spectrograph.max_wl, quantised=True
    )
    min_wl_order: int = spectrograph.grating.order_number(
        spectrograph.min_wl, quantised=True
    )

    step: int = int((min_wl_order - max_wl_order) / (number_of_configs - 1))

    ms: list[int] = [
        max_wl_order,
        *[max_wl_order + 1 + (i - 1) * step for i in range(2, number_of_configs)],
        min_wl_order,
    ]

    for m in ms:
        wls: list[float] = (
            spectrograph.order_wavelengths(m=m, num=5, FSR_fraction=1.0)
            .to(u.nm)
            .to(u.um)
            .value
        )
        out.append((m, [w * u.um for w in wls]))

    return out


@dataclass
class RectangularAperture:
    """A simple square aperture class for Zemax export."""

    half_width: ScalarQuantity | float
    half_height: ScalarQuantity | float
    rotation: ScalarQuantity | float


@dataclass
class Surface:
    """A simple surface class for Zemax export."""

    comment: str
    surface_type: Literal["STANDARD", "PARAXIAL", "COORDBRK", "DGRATING"]
    thickness: ScalarQuantity | float = 0.0
    stop: bool = False
    params: dict[int, float] | None = None
    glass: str | None = None
    conic: float | None = None
    diam: ScalarQuantity | float | None = None
    rectangular_aperture: RectangularAperture | None = None
    mirror_angle: float | None = None
    dgrating: tuple[float, float] | None = None

    def zemax_lines(self, idx: int) -> list[str]:
        lines: list[str] = [
            f"SURF {idx}",
            f"  COMM {self.comment}",
        ]

        if self.stop:
            lines.append("  STOP")
        lines.extend(
            [
                f"  TYPE {self.surface_type}",
                "  FIMP ",
                # "",  # ? - Sometimes files have a blank line here?
                "  CURV 0.0 0 0.0 0.0 0",
                "  HIDE 0 0 0 0 0 0 0 0 0 0 0 0 0 0",
            ]
        )

        if self.mirror_angle is None:
            lines.append("  MIRR 2 0")
        else:
            lines.append(f"  MIRR 1 {self.mirror_angle:.6f}".rstrip("0").rstrip("."))

        if self.params is not None:
            for p_idx, p_val in sorted(self.params.items()):
                lines.append(f"  PARM {p_idx} {p_val:.12f}".rstrip("0").rstrip("."))

        if self.dgrating is not None:
            groove_density_per_um, order = self.dgrating
            lines.append(
                f"  PARM 1 {groove_density_per_um:.12f}".rstrip("0").rstrip(".")
            )
            lines.append(f"  PARM 2 {order:.12f}".rstrip("0").rstrip("."))

        lines.append(f"  DISZ {self.thickness:.12f}".rstrip("0").rstrip("."))

        if self.glass is not None:
            lines.append(f"  GLAS {self.glass} 0 0 1.5 40 0 0 0 0 0 0 ")

        if self.conic is not None:
            lines.append(f"  CONI {self.conic:.6f}")

        if self.diam is not None:
            lines.append(
                f"  DIAM {self.diam:.12f}".rstrip("0").rstrip(".") + ' 0 0 0 1 ""'
            )
            lines.append(
                f"  MEMA {self.diam:.12f}".rstrip("0").rstrip(".") + ' 0 0 0 1 ""'
            )
        else:
            lines.append('  DIAM 0 0 0 0 1 ""')
            lines.append('  MEMA 0 0 0 0 1 ""')

        if self.rectangular_aperture is not None:
            lines.append(
                "  SQAP"
                + f" {self.rectangular_aperture.half_width:.6f}".rstrip("0").rstrip(".")
                + f" {self.rectangular_aperture.half_height:.6f}".rstrip("0").rstrip(
                    "."
                )
                + f" {self.rectangular_aperture.rotation:.6f}".rstrip("0").rstrip(".")
            )

        lines.append("  POPS 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0")
        return lines


def zemax_surface_sequence(
    spectrograph: EchelleSpectrograph,
) -> list[list[str]]:
    beam_diameter_mm: float = spectrograph.beam_diameter.to(u.mm).value
    half_beam_mm: float = beam_diameter_mm / 2

    f_col: ScalarQuantity = (
        spectrograph.collimator_focal_length
        if spectrograph.collimator_focal_length is not None
        else 1125 * u.mm
    )
    f_col_mm: float = f_col.to(u.mm).value
    f_cam_mm: float = spectrograph.camera_focal_length.to(u.mm).value

    assert isinstance(spectrograph, DoublePassEchelleSpectrograph) or isinstance(
        spectrograph.cross_disperser, Prism | TwoPrisms
    ), "This script currently supports prism-based cross-dispersers only."

    assert isinstance(spectrograph.cross_disperser, Prism | TwoPrisms)

    if isinstance(spectrograph.cross_disperser, Prism):
        prism1: Prism = spectrograph.cross_disperser
        prism2: Prism | None = None
    else:
        assert isinstance(spectrograph.cross_disperser.prism2, Prism)
        prism1: Prism = spectrograph.cross_disperser.prism1
        prism2: Prism = spectrograph.cross_disperser.prism2

    echelle_order: int = -spectrograph.grating.order_number(
        spectrograph.max_wl, quantised=True
    )
    groove_density_per_um: float = (
        spectrograph.grating.groove_frequency.to(1 / u.mm).value / 1000
    )

    surfaces: list[Surface] = [
        Surface("Input", "STANDARD"),
        Surface("Flip coordinate system", "STANDARD", glass="MIRROR"),
        Surface("Collimate", "COORDBRK", -spectrograph.slit_native_focal_ratio),
        Surface(
            "Collimate",
            "PARAXIAL",
            params={1: spectrograph.slit_native_focal_ratio, 2: 1.0},
            diam=half_beam_mm / f_col_mm,
        ),
        Surface("Collimate", "COORDBRK", -spectrograph.slit_native_focal_ratio),
        Surface("Injection Pupil", "STANDARD", stop=True),
        # TODO: compute the collimator focal length or f/# in order to set the correct distances here (default 7 mm) - this is essentially a free parameter!
        Surface("Refocus", "COORDBRK", -7.0),
        Surface(
            "Refocus", "PARAXIAL", params={1: 7, 2: 1.0}, diam=half_beam_mm / f_col_mm
        ),
        Surface("Refocus", "COORDBRK", -7),
        Surface("to Collimator", "COORDBRK", -f_col_mm),
        Surface(
            "M1.1",
            "STANDARD",
            0.0,
            glass="MIRROR",
            conic=-1.0,
            diam=beam_diameter_mm * 1.5,
            mirror_angle=54.73,
        ),
        Surface("to Echelle", "COORDBRK", f_col_mm),
        Surface("Echelle decenter", "COORDBRK", params={1: 0.8 * half_beam_mm}),
        Surface("Echelle pupil", "STANDARD", stop=True, diam=half_beam_mm),
        Surface(
            "Gamma",
            "COORDBRK",
            params={4: spectrograph.grating.gamma_angle.to(u.deg).value},
        ),
        Surface(
            "Blaze",
            "COORDBRK",
            params={3: spectrograph.grating.blaze_angle.to(u.deg).value},
        ),
        Surface(
            "Echelle",
            "DGRATING",
            glass="MIRROR",
            diam=beam_diameter_mm * 1.45,
            rectangular_aperture=RectangularAperture(
                half_width=beam_diameter_mm * 0.5,
                half_height=beam_diameter_mm * 0.5 * spectrograph.grating.R,
                rotation=0.0,
            ),
            mirror_angle=75.0,
            dgrating=(groove_density_per_um, float(echelle_order)),
        ),
        Surface(
            "- Blaze",
            "COORDBRK",
            params={3: -spectrograph.grating.blaze_angle.to(u.deg).value},
        ),
        Surface(
            "- Gamma",
            "COORDBRK",
            params={4: -spectrograph.grating.gamma_angle.to(u.deg).value},
        ),
        Surface("- Echelle decenter", "COORDBRK", params={1: -0.8 * half_beam_mm}),
        Surface("to Collimator", "COORDBRK", -f_col_mm),
        Surface(
            "M1.2", "STANDARD", glass="MIRROR", conic=-1.0, diam=beam_diameter_mm * 1.5
        ),
        Surface("to Prism", "COORDBRK", f_col_mm),
        Surface(
            "Prism incidence angle",
            "COORDBRK",
            params={4: -prism1.incidence_angle.to(u.deg).value},
        ),
        Surface(
            "Prism S1",
            "STANDARD",
            beam_diameter_mm,
            glass=prism1.glass.name,
            diam=half_beam_mm,
            rectangular_aperture=RectangularAperture(
                half_width=beam_diameter_mm,
                half_height=np.tan(prism1.incidence_angle.to(u.deg)),
                rotation=0.0,
            ),
        ),
        Surface(
            "Prism apex angle",
            "COORDBRK",
            params={1: 90.0, 4: prism1.apex_angle.to(u.deg).value},
        ),
        Surface(
            "Prism S2",
            "STANDARD",
            0.0,
            glass=prism1.glass.name,
            diam=half_beam_mm,
            rectangular_aperture=RectangularAperture(
                half_width=beam_diameter_mm,
                half_height=np.tan(prism1.incidence_angle.to(u.deg)),
                rotation=0.0,
            ),
        ),
        Surface(
            "Post prism tilt",
            "COORDBRK",
            params={4: -prism1.incidence_angle.to(u.deg).value},
        ),
    ]

    if prism2 is not None:
        surfaces.extend(
            [
                Surface("to Prism 2", "COORDBRK", params={1: 5.0, 2: -15.0}),
                Surface(
                    "Prism 2 incidence angle",
                    "COORDBRK",
                    params={4: -prism2.incidence_angle.to(u.deg).value},
                ),
                Surface(
                    "Prism 2 S1",
                    "STANDARD",
                    beam_diameter_mm,
                    glass=prism2.glass.name,
                    diam=half_beam_mm,
                    rectangular_aperture=RectangularAperture(
                        half_width=beam_diameter_mm,
                        half_height=np.tan(prism2.incidence_angle.to(u.deg)),
                        rotation=0.0,
                    ),
                ),
                Surface(
                    "Prism 2 apex angle",
                    "COORDBRK",
                    params={1: 90.0, 4: prism2.apex_angle.to(u.deg).value},
                ),
                Surface(
                    "Prism 2 S2",
                    "STANDARD",
                    glass=prism2.glass.name,
                    diam=half_beam_mm,
                    rectangular_aperture=RectangularAperture(
                        half_width=beam_diameter_mm,
                        half_height=np.tan(prism2.incidence_angle.to(u.deg)),
                        rotation=0.0,
                    ),
                ),
                Surface(
                    "Post prism tilt",
                    "COORDBRK",
                    params={4: -prism2.incidence_angle.to(u.deg).value},
                ),
            ]
        )

    surfaces.extend(
        [
            Surface("to Camera", "COORDBRK", 100.0),
            Surface(
                "Camera", "PARAXIAL", params={1: f_cam_mm, 2: 1.0}, diam=half_beam_mm
            ),
            Surface("to Detector", "COORDBRK", f_cam_mm),
            Surface(
                "CCD",
                "STANDARD",
                rectangular_aperture=RectangularAperture(
                    half_width=spectrograph.detector.size[0].to(u.mm).value,
                    half_height=spectrograph.detector.size[1].to(u.mm).value,
                    rotation=0.0,
                ),
                mirror_angle=5.0,
            ),
        ]
    )

    return [s.zemax_lines(idx=i) for i, s in enumerate(surfaces)]


def generate_zemax_zmx(spectrograph: EchelleSpectrograph) -> str:
    """Generate a sequential Zemax ZMX text model from the current design."""

    mce_wavelengths: list[tuple[int, list[ScalarQuantity]]] = zemax_wavelengths(
        spectrograph
    )

    first_cfg: list[ScalarQuantity] = mce_wavelengths[0][1]
    timestamp: str = datetime.now(UTC).strftime("%Y%m%d%H%M%S")

    if isinstance(spectrograph.cross_disperser, TwoPrisms):
        glass_catalogues: set[str] = {
            spectrograph.prism1.glass.manufacturer,
            spectrograph.prism2.glass.manufacturer,
        }
    elif isinstance(spectrograph.cross_disperser, Prism):
        glass_catalogues: set[str] = {spectrograph.cross_disperser.glass.manufacturer}

    header: list[str] = [
        f"VERS {timestamp} 37328 1153887 01153887",  # Where do these numbers come from? The Zemax version and build number(?), but how to get them programmatically?
        "MODE SEQ",
        "NAME Echelle Spectrograph",
        "AUTH curved_air",
        "IWDP 0",
        "NOTE 0 Exported from curved_air: https://curvedair.jakepember.com/",
        "NOTE 4",  # Note off
        "PFIL 0 0 0",
        "LANG 0",
        "UNIT MM X W X CM MR CPMM LU ",
        f"OBNA {(1 / (2 * spectrograph.slit_native_focal_ratio)):.5f} 0",
        "ENVD 20 1e-08 1",  # Temperature and pressure?
        "GFAC 0 0",  # ?
        f"GCAT {' '.join(gc.upper() for gc in glass_catalogues)} MISC 0",
        "RAIM 0 0 1 1 0 0 0 0 0 1",
        "PUSH 0 0 0 0 0 0",
        "SDMA 0 0 0",
        "OMMA 1 1",
        "FTYP 1 1 1 5 1 0 0 1",
        "ROPD 2",
        "HYPR 0",
        "PICB 1",
        "XFLN 0",
        "YFLN 0",
        "FWGN 1",
        "VDXN 0",
        "VDYN 0",
        "VCXN 0",
        "VCYN 0",
        "VANN 0",
    ]

    for i, wl in enumerate(first_cfg, start=1):
        header.append(f"WAVM {i} {wl.to(u.um).value:.16f} 1")

    header.extend(
        [
            "PWAV 1",  # Primary wavelength
            "POLS 1 0 1 0 0 1 0",
            "GLRS 12 0",  # Global coordinate reference surface - don't hard code it!
            "GSTD 1 100.000 100.000 100.000 100.000 100.000 100.000 0 1 1 0 0 1 1 1 1 1 1",
            "NSCD 100 500 0 1E-06 10 1E-07 0 0 0 0 0 0 1000000 0 2",
            'COFN QF "COATING.DAT" "SCATTER_PROFILE.DAT" "ABG_DATA.DAT" "PROFILE.GRD"',
            "COFN COATING.DAT SCATTER_PROFILE.DAT ABG_DATA.DAT PROFILE.GRD",
            "LUID 197",
            "EERA 0 0 0 10",
        ]
    )

    surface_blocks: list[list[str]] = zemax_surface_sequence(spectrograph=spectrograph)

    mce_lines: list[str] = [
        f"CONF {len(mce_wavelengths)} 0 0 0 0 0 0 0 0 0",
        "BLNK ",
        "TOL TOFF   0   0 0.0000000000000000E+00 0.0000000000000000E+00   0 0 0 0 0",
        f"MNUM {len(mce_wavelengths)} 1",
    ]

    echelle_surface_idx: int | None = None
    for block in surface_blocks:
        if any(line.strip() == "COMM Echelle" for line in block):
            echelle_surface_idx = int(block[0].split()[1])
            break

    if echelle_surface_idx is None:
        raise RuntimeError("Could not identify echelle surface index for MCE setup.")

    for cfg_idx, (order, _wls) in enumerate(mce_wavelengths, start=1):
        mce_lines.append(
            "PRAM"
            + f" {echelle_surface_idx:>3}"
            + f" {cfg_idx:>3}"
            + f" {float(-order):.12E}"
            + ' 0 2 0 1 1 1.000000000000E+00 0.000000000000E+00 0 "" 0'
        )

    for cfg_idx, (_order, wls) in enumerate(mce_wavelengths, start=1):
        for w_idx, wl in enumerate(wls, start=1):
            mce_lines.append(
                "WAVE"
                + f" {cfg_idx:>3}"
                + f" {w_idx:>3}"
                + f" {wl.to(u.um).value:.12E}"
                + ' 0 0 0 1 1 1.000000000000E+00 0.000000000000E+00 0 "" 0'
            )

    all_lines: list[str] = header
    for block in surface_blocks:
        all_lines.extend(block)
    all_lines.extend(mce_lines)

    return "\n".join(all_lines) + "\n"


def main() -> None:
    """Main function to generate a Zemax ZMX file from a `curved_air` EchelleSpectrograph object."""

    spectrograph = EchelleSpectrograph()

    zmx_content: str = generate_zemax_zmx(spectrograph)

    OUTPUT_FILE: Path = Path("curved_air_export.zmx")
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        f.write(zmx_content)

    print("Zemax ZMX file generated: curved_air_export.zmx")


if __name__ == "__main__":
    main()
