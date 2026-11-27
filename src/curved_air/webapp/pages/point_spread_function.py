#! uv run streamlit run

"""
TODO:
  - Different wavelengths: generate PSF with up to three different wavelengths?
"""

from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st
from astropy import units as u

from curved_air.geometry import (
    HEX_FACTOR,
    Circle,
    Rectangle,  # noqa: F401
    RegularPolygon,
    Shape,
)
from curved_air.plotting import plot_style
from curved_air.pupils_psfs import (
    PSF,
    Pupil,
    PupilGrid,
    SampledPupil,
    segmented_telescope,
    telescope,  # noqa: F401
)
from curved_air.type_hinting import ArrayQuantity, ScalarQuantity  # noqa: F401
from curved_air.utils import st_quantity_input
from curved_air.webapp import badges, bmac

plt.style.use(plot_style)

SCRIPT_PATH: Path = Path(__file__).parent
DATA_PATH: Path = SCRIPT_PATH / ".." / ".." / "data"


st.set_page_config(
    page_title="Point Spread Function",
    layout="wide",
)

st.title("Point Spread Function")

with st.expander("About this tool", expanded=False):
    f"""
    This tool generates the idealised diffraction-only point spread function (PSF) for a
    given telescope pupil aperture.



    All code is written by Jake Pember {badges}

    Please get in contact if you have any questions, suggestions, or find any bugs!

    You can also [Buy me a pint](https://buymeacoffee.com/pemberj) if you find this tool
    useful and want to support its development!
    """

    bmac()

settings_column, output_column = st.columns(2)

with settings_column:
    with st.expander("Primary Mirror Geometry", expanded=True):
        st.selectbox(
            "Type",
            options=[
                "Circular",
                "Segmented (hexagonal)",
                "Custom",
            ],
            width=300,
            key="aperture_type",
        )

        match st.session_state.aperture_type:
            case "Circular":
                st_quantity_input(
                    "Primary mirror diameter",
                    min_value=0.0,
                    value=1.0,
                    step=0.1,
                    default_unit="m",
                    possible_units=["mm", "cm", "m"],
                    format="%.4g",
                    help="Diameter of the primary mirror aperture",
                    key="primary_diameter",
                )

                primary: Shape = Circle(diameter=st.session_state.primary_diameter)

            case "Segmented (hexagonal)":
                st_quantity_input(
                    "Segment size",
                    min_value=0.0,
                    value=1.0,
                    step=0.1,
                    default_unit="m",
                    possible_units=["mm", "cm", "m"],
                    format="%.4g",
                    help="Distance across flats of the hexagonal segments",
                    key="segment_size",
                )

                st_quantity_input(
                    "Segment gap size",
                    min_value=0.0,
                    value=10.0,
                    step=0.1,
                    default_unit="mm",
                    possible_units=["mm", "cm", "m"],
                    format="%.4g",
                    help="Gap between adjacent segments",
                    key="segment_gap_size",
                )

                st.number_input(
                    "Number of rings",
                    min_value=1,
                    value=2,
                    step=1,
                    width=300,
                    help="Number of rings of hexagonal segments around the central segment",
                    key="num_rings",
                )

                if st.toggle("Limit to a maximum radius?"):
                    st_quantity_input(
                        "Maximum radius",
                        min_value=0.0,
                        value=10.0,
                        step=0.1,
                        default_unit="m",
                        possible_units=["mm", "cm", "m"],
                        format="%.4g",
                        help="Maximum radius of the primary mirror aperture",
                        key="max_radius",
                    )

                st_quantity_input(
                    "Rotation angle",
                    min_value=0.0,
                    max_value=360.0,
                    value=0.0,
                    step=1.0,
                    default_unit="deg",
                    possible_units=["arcsec", "arcmin", "deg", "rad"],
                    help="Rotation angle of the hexagonal segments (degrees)",
                    key="rotation_angle",
                )

                primary: Shape = segmented_telescope(
                    segment_size=st.session_state["segment_size"],
                    segment_gap_size=st.session_state["segment_gap_size"],
                    num_hexagon_rings=st.session_state["num_rings"],
                    clocking_angle=st.session_state["rotation_angle"],
                    max_radius=st.session_state.get("max_radius"),
                )

            case _:
                st.warning(
                    """Custom apertures are not yet supported. Please select a different
                    aperture type."""
                )

                st.stop()

    with st.expander("Secondary Mirror Geometry", expanded=True):
        st.selectbox(
            "Secondary mirror aperture type",
            options=[
                "Circular",
                "Hexagonal",
                "Custom",
            ],
            width=300,
            key="secondary_aperture_type",
        )

        match st.session_state.secondary_aperture_type:
            case "Circular":
                st_quantity_input(
                    "Secondary mirror diameter",
                    min_value=0.0,
                    value=0.2,
                    step=0.1,
                    default_unit="m",
                    possible_units=["mm", "cm", "m"],
                    format="%.4g",
                    help="Diameter of the secondary mirror aperture",
                    key="secondary_diameter",
                )

                secondary: Shape = Circle(diameter=st.session_state.secondary_diameter)

            case "Hexagonal":
                st_quantity_input(
                    "Secondary mirror size",
                    min_value=0.0,
                    value=0.5,
                    step=0.1,
                    default_unit="m",
                    possible_units=["mm", "cm", "m"],
                    format="%.4g",
                    help="Distance across flats of the hexagonal secondary mirror",
                    key="secondary_size",
                )

                secondary: Shape = RegularPolygon(
                    num_sides=6,
                    circumradius=(st.session_state.secondary_size * HEX_FACTOR) / 2,
                )

            case _:
                st.warning(
                    """Custom secondary mirror apertures are not yet supported. Please
                    select a different aperture type."""
                )

                st.stop()

    with st.expander("Support structure", expanded=True):
        ...

    if st.toggle("Sample PSF on detector pixel grid?"):
        with st.expander("Detector", expanded=True):
            ...

    with st.expander("Image settings", expanded=True):
        ...

geometry: Shape = primary - secondary
pupil = Pupil(geometry)
grid = PupilGrid(npix=1024, extent=geometry.max_radius * 2)
sampled_pupil: SampledPupil = pupil.sample(grid)
psf: PSF = sampled_pupil.psf(wavelength=500 * u.nm)

with output_column:
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    ax: plt.Axes = axs[0]

    ax.imshow(
        sampled_pupil.amplitude,
        cmap="gray",
        extent=[
            grid.x[0].to(u.m).value,
            grid.x[-1].to(u.m).value,
            grid.y[0].to(u.m).value,
            grid.y[-1].to(u.m).value,
        ],
    )

    ax.set_xlabel("Pupil position [m]")
    ax.set_ylabel("Pupil position [m]")

    half_width: ScalarQuantity = geometry.max_radius.to(u.m).value
    ax.set_xlim(-half_width, half_width)
    ax.set_ylim(-half_width, half_width)

    ax = axs[1]

    ax.imshow(
        psf.intensity,
        cmap="gray",
        extent=[
            psf.x[0].to(u.arcsec).value,
            psf.x[-1].to(u.arcsec).value,
            psf.y[0].to(u.arcsec).value,
            psf.y[-1].to(u.arcsec).value,
        ],
    )

    ax.set_xlabel("Angular position [arcsec]")
    ax.set_ylabel("Angular position [arcsec]")

    width = (2 * u.arcsec).to(u.arcsec).value
    ax.set_xlim(-width / 2, width / 2)
    ax.set_ylim(-width / 2, width / 2)

    st.pyplot(fig)
