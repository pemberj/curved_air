#! uv run streamlit run

# This calculator allows for generating a spectral format plot (echellogram) for a given
# echelle spectrograph design configuration.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from astropy import units as u
from matplotlib import pyplot as plt

from curved_air.detectors import DETECTORS, Detector
from curved_air.glass import GLASS_CATALOGUES, Glass, GlassCatalogue
from curved_air.instruments.echelle import (
    DoublePassEchelleSpectrograph,
    EchelleSpectrograph,
)
from curved_air.optics.dispersers import (
    CrossDisperser,
    EchelleGrating,
    Prism,
    SimpleGrating,
    TwoPrisms,
)
from curved_air.other import _atomic_lines
from curved_air.plotting import plot_style, wavelength_to_rgb, wl_to_hex
from curved_air.utils import richardson_gratings, st_quantity_input

if TYPE_CHECKING:
    from curved_air.type_hinting import ArrayQuantity, ScalarQuantity

plt.style.use(plot_style)


st.header("Echelle Spectrograph", anchor=False)

with st.expander("About this tool", expanded=False):
    """
    This tool generates a spectral format plot (echellogram) for a given echelle
    spectrograph design. Use the sidebar to adjust the spectrograph parameters, and the
    format plot will update automatically. You can hover over the plot to see the
    wavelength and position of each point on the detector.

    This is meant for quick exploration of the top-level spectrograph design space. The
    results shown here may differ from a more detailed ray-tracing analysis.

    All code is written by Jake Pember :gray-badge[:material/language: [Web](https://www.jakepember.com)] :gray-badge[:material/email: [Email](mailto:jacob.pember@ikmail.com)] :gray-badge[:material/code: [Github](https://www.github.com/pemberj)]

    Please get in contact if you have any questions, suggestions, or find any bugs!

    You can also [Buy me a pint](https://buymeacoffee.com/pemberj) if you find this tool
    useful and want to support its development!
    """

    st.iframe(
        """
        <script type="text/javascript" src="https://cdnjs.buymeacoffee.com/1.0.0/button.prod.min.js" data-name="bmc-button" data-slug="pemberj" data-color="#E5660C" data-emoji="🍺" data-font="Cookie" data-text="Buy me a pint" data-outline-color="#000000" data-font-color="#ffffff"></script>
        """,
    )


@dataclass
class PresetInstrument:
    name: str
    min_wl: ScalarQuantity
    max_wl: ScalarQuantity
    echelle_grating: str
    cross_disperser_type: str
    slit_width: ScalarQuantity
    slit_length: ScalarQuantity
    slit_focal_ratio: float
    beam_diameter: ScalarQuantity
    camera_focal_length: ScalarQuantity


preset_instruments: list[PresetInstrument] = [
    PresetInstrument(
        name="MARVEL",
        min_wl=380 * u.nm,
        max_wl=950 * u.nm,
        echelle_grating="",
        cross_disperser_type="Prism",
        slit_width=40 * u.um,
        slit_length=300 * u.um,
        slit_focal_ratio=3.85,
        beam_diameter=160 * u.mm,
        camera_focal_length=730 * u.mm,
    )
]

ECHELLE_GRATINGS: dict[str, EchelleGrating] = {
    f"{g.catalogue_number}: {g.groove_frequency.value:g}g/mm R{g.R:g}": g
    for g in richardson_gratings()
}


# Interface
with st.sidebar:
    with st.expander(label="Presets", expanded=True):
        preset_instrument: str | None = st.segmented_control(
            "Instrument",
            [
                "HARPS",
                "ESPRESSO",
                "NEID",
                "EXPRES",
                "KPF",
                "MARVEL",
                "2ES",
                "",
                "Custom",
            ],
            default="MARVEL",
            help="""""",
        )

        if preset_instrument in ["Custom", None]:
            ...
        else:
            ...

        # Pre-populate the session_state with parameters for the chosen spectrograph,
        # and then define all the 'default' values of the parameter selections below
        # with st.session_state.get("preset_default_X", "normal_default"), so each of
        # the preset instruments can easily be used as a starting point for a customised
        # version.

    with st.expander(label="Wavelength range", expanded=False):
        min_wl: ScalarQuantity = st_quantity_input(
            "Min λ",
            min_value=0.0,
            value=st.session_state.get("min_wl", 380.0 * u.nm).to(u.nm).value,
            step=1.0,
            possible_units=["nm", "µm"],
            default_unit="nm",
            key="min_wl",
        )
        st.session_state["min_wl"] = min_wl

        max_wl: ScalarQuantity = st_quantity_input(
            "Max λ",
            min_value=0.0,
            value=st.session_state.get("min_wl", 950.0 * u.nm).to(u.nm).value,
            step=1.0,
            possible_units=["nm", "µm"],
            default_unit="nm",
            key="max_wl",
        )
        st.session_state["max_wl"] = max_wl

    with st.expander(label="Echelle Grating", expanded=False):
        grating: str = st.selectbox(
            "Richardson gratings",
            options=list(ECHELLE_GRATINGS.keys()),
            index=6,  # MARVEL grating
            width=300,
            disabled=st.session_state.get("customise_grating", False),
            key="echelle_grating",
            help="""Select from a list of commercially available echelle gratings from
            Richardson Gratings. The grating parameters will be pre-filled based on the
            selected option.""",
        )

        grating: EchelleGrating = ECHELLE_GRATINGS[grating]

        st.toggle(
            "Custom grating parameters",
            value=st.session_state.get("customise_grating", False),
            key="customise_grating",
            help="""Input your own values for the grating groove spacing, blaze angle,
            and ruled width.""",
        )

        if st.session_state["customise_grating"]:
            groove_frequency: ScalarQuantity = st_quantity_input(
                "Groove freqency",
                min_value=0.0,
                value=grating.groove_frequency.to(u.mm**-1).value,
                step=0.1,
                possible_units=["µm", "mm", "cm"],
                default_unit="mm",
                unit_input_text="Per",
                inverse=True,
                key="groove_frequency",
                # help="""Spatial groove frequency""",
            )

            blaze_angle: ScalarQuantity = (
                st.number_input(
                    "Blaze angle (degrees)",
                    min_value=0.0,
                    max_value=89.0,
                    value=grating.blaze_angle.to(u.deg).value,
                    step=0.1,
                    help="""The blaze angle of the echelle grating, in degrees. This is
                    the angle of illumination under which the grating is designed for
                    optimal efficiency.""",
                )
                * u.deg
            )

            ruled_width: ScalarQuantity = st_quantity_input(
                "Ruled width (mm)",
                min_value=0.0,
                # max_value=1200.0,
                value=grating.ruled_width.to(u.mm).value,
                step=0.05,
                possible_units=["mm", "cm", "m"],
                default_unit="mm",
                help="""The ruled width of the echelle grating, in millimetres. This
                    refers to the dimension perpendicular to the grooves themselves.""",
            )

            st.metric(
                "Grating R-number",
                f"R{np.tan(blaze_angle):.1f}",
                help="""The R-number of an echelle grating is a dimensionless parameter
                defined as the tangent of the blaze angle.""",
            )

        else:
            groove_frequency: float = grating.groove_frequency.to(u.mm**-1)
            blaze_angle: float = grating.blaze_angle.to(u.deg)
            ruled_width: float = grating.ruled_width.to(u.mm)

    with st.expander(label="Cross-disperser", expanded=False):
        cross_disperser_type: str = st.segmented_control(
            "Type",
            [
                "Prism",
                "Grating",
                # "Grism",  # TODO
                "Multiple prisms",
                "Double-pass prism",
            ],
            default=st.session_state.get("cross_disperser_type", "Prism"),
            key="cross_disperser_type",
            help="""Select the type of cross-disperser used in the spectrograph.
            The choice of cross-disperser has a significant impact on the layout of the
            spectral format, the achievable wavelength coverage and the order
            separation.
            The "double-pass prism" option refers to a configuration where the beam
            passes through the same prism both before and after the echelle grating.""",
        )  # ty:ignore[invalid-assignment]

        if cross_disperser_type == "Prism":
            default_apex_angle: ScalarQuantity = 60 * u.deg
        elif cross_disperser_type == "Double-pass prism":
            default_apex_angle: ScalarQuantity = 45 * u.deg

        if cross_disperser_type in ["Prism", "Double-pass prism"]:
            catalogue: str = st.selectbox(
                "Glass catalogue",
                GLASS_CATALOGUES.keys(),
                index=list(GLASS_CATALOGUES.keys()).index("Schott")
                if "Schott" in GLASS_CATALOGUES
                else 0,
                width=300,
            )

            try:
                glasses = GlassCatalogue(catalogue)
                glass_names: list[str] = sorted(glasses.keys())
            except Exception:
                st.error(f"Could not load {catalogue} glass catalogue.")

            default_glass: str = "F2"

            glass_name: str = st.selectbox(
                "Glass",
                glass_names,
                index=glass_names.index(default_glass)
                if default_glass in glass_names
                else 0,
                width=300,
            )

            apex_angle: ScalarQuantity = (
                st.number_input(
                    "Apex angle (°)",
                    min_value=5.0,
                    max_value=60.0,
                    value=default_apex_angle.to(u.deg).value,
                    step=1.0,
                    width=300,
                )
                * u.deg
            )

            st.toggle(
                "Plot glass refractive index",
                value=False,
                key="ref_index_plot_toggle",
            )

            if st.session_state["ref_index_plot_toggle"]:
                wls: list[float] = np.linspace(
                    min_wl.to(u.nm).value, max_wl.to(u.nm).value, 50
                )

                fig: go.Figure = go.Figure()  # go figure!
                fig.add_trace(
                    trace=go.Scatter(
                        x=wls,
                        y=glasses[glass_name].refractive_index(wls),
                        mode="lines",
                    )
                )
                fig.update_layout(
                    title=f"{glass_name} Refractive Index",
                    xaxis_title="Wavelength (nm)",
                    yaxis_title="Refractive Index",
                    template="plotly_dark",
                )

                st.plotly_chart(fig)

            _cross_disperser: CrossDisperser = Prism(
                glass=glasses[glass_name],
                apex_angle=apex_angle,
                min_wl=min_wl,
                max_wl=max_wl,
            )

            st.metric(
                "Calculated incidence angle",
                f"{_cross_disperser.incidence_angle.value:.2f} degrees",
                help="""This tool automatically calculates the
                appropriate incidence angle such that the spectral format is symmetrically
                centred on the detector.""",
            )

        elif cross_disperser_type == "Multiple prisms":
            p1_catalogue: str = st.selectbox(
                "Prism 1 glass catalogue",
                GLASS_CATALOGUES.keys(),
                index=list(GLASS_CATALOGUES.keys()).index("Schott")
                if "Schott" in GLASS_CATALOGUES
                else 0,
                width=300,
            )

            try:
                p1_glasses = GlassCatalogue(p1_catalogue)
                p1_glass_names: list[str] = sorted(p1_glasses.keys())
            except Exception:
                st.error(f"Could not load {p1_catalogue} glass catalogue.")

            default_glass: str = "F2"

            p1_glass_name: str = st.selectbox(
                "Prism 1 Glass",
                p1_glass_names,
                index=p1_glass_names.index(default_glass)
                if default_glass in p1_glass_names
                else 0,
                width=300,
            )

            p1_apex_angle: ScalarQuantity = (
                st.number_input(
                    "Apex angle (°)",
                    min_value=5.0,
                    max_value=60.0,
                    value=48.7,
                    step=1.0,
                    key="p1_apex_angle",
                    width=300,
                )
                * u.deg
            )

            st.toggle(
                "Plot glass refractive index",
                value=False,
                key="p1_ref_index_plot_toggle",
            )

            if st.session_state["p1_ref_index_plot_toggle"]:
                wls: list[float] = np.linspace(
                    min_wl.to(u.nm).value, max_wl.to(u.nm).value, 50
                )

                fig = go.Figure()
                fig.add_trace(
                    trace=go.Scatter(
                        x=wls,
                        y=p1_glasses[p1_glass_name].refractive_index(wls),
                        mode="lines",
                    )
                )
                fig.update_layout(
                    title=f"{p1_glass_name} Refractive Index",
                    xaxis_title="Wavelength (nm)",
                    yaxis_title="Refractive Index",
                    template="plotly_dark",
                )

                st.plotly_chart(fig, key="p1_ref_index_plot")

            st.toggle(
                "Use two identical prisms",
                value=True,
                key="identical_prisms_toggle",
            )

            if st.session_state["identical_prisms_toggle"]:
                # p2_catalogue = p1_catalogue
                p2_glasses: GlassCatalogue = p1_glasses
                p2_glass_name: str = p1_glass_name
                p2_apex_angle: ScalarQuantity = p1_apex_angle
            else:
                p2_catalogue: str = st.selectbox(
                    "Prism 2 glass catalogue",
                    GLASS_CATALOGUES.keys(),
                    index=list(GLASS_CATALOGUES.keys()).index("Schott")
                    if "Schott" in GLASS_CATALOGUES
                    else 0,
                    width=300,
                )

                try:
                    p2_glasses = GlassCatalogue(p2_catalogue)
                    p2_glass_names: list[str] = sorted(p2_glasses.keys())
                except Exception:
                    st.error(f"Could not load {p2_catalogue} glass catalogue.")

                p2_glass_name: str = st.selectbox(
                    "Prism 2 Glass",
                    p2_glass_names,
                    index=p2_glass_names.index(default_glass)
                    if default_glass in p2_glass_names
                    else 0,
                    width=300,
                )

                p2_apex_angle: ScalarQuantity = (
                    st.number_input(
                        "Apex angle (°)",
                        min_value=5.0,
                        max_value=60.0,
                        value=48.7,
                        step=1.0,
                        key="p2_apex_angle",
                        width=300,
                    )
                    * u.deg
                )

                st.toggle(
                    "Plot glass refractive index",
                    value=False,
                    key="p2_ref_index_plot_toggle",
                )

                if st.session_state["p2_ref_index_plot_toggle"]:
                    wls: list[float] = np.linspace(
                        min_wl.to(u.nm).value, max_wl.to(u.nm).value, 50
                    )

                    fig = go.Figure()
                    fig.add_trace(
                        trace=go.Scatter(
                            x=wls,
                            y=p2_glasses[p2_glass_name].refractive_index(wls),
                            mode="lines",
                        )
                    )
                    fig.update_layout(
                        title=f"{p2_glass_name} Refractive Index",
                        xaxis_title="Wavelength (nm)",
                        yaxis_title="Refractive Index",
                        template="plotly_dark",
                    )

                    st.plotly_chart(fig, key="p2_ref_index_plot")

            _cross_disperser = TwoPrisms(
                prism1=Prism(
                    glass=p1_glasses[p1_glass_name],
                    apex_angle=p1_apex_angle,
                ),
                prism2=Prism(
                    glass=p2_glasses[p2_glass_name],
                    apex_angle=p2_apex_angle,
                ),
            )

            st.metric(
                "Calculated incidence angle",
                f"{_cross_disperser.incidence_angle.value:.2f} degrees",
                help="""This tool automatically calculates the
                appropriate incidence angle such that the spectral format is symmetrically
                centred on the detector.""",
            )

        elif cross_disperser_type == "Grating":
            cross_disperser_groove_frequency: ScalarQuantity = st_quantity_input(
                "Groove frequency",
                min_value=0.0,
                value=200.0,
                step=10.0,
                possible_units=["µm", "mm", "cm"],
                default_unit="mm",
                unit_input_text="Per",
                inverse=True,
                key="cross_disperser_groove_frequency",
            )

            if st.session_state.get("max_wl", 950 * u.nm) > 2 * st.session_state.get(
                "min_wl", 380 * u.nm
            ):
                st.warning(
                    """Warning: with a grating cross-disperser, the wavelength range
                    should not exceed an octave (max λ should be less than 2x min λ) to
                    avoid significant order overlap issues.""",
                )

            st.info(
                """Note: only the first diffraction order of the cross-disperser is
                displayed in the spectral format plot.""",
            )

        elif cross_disperser_type == "Grism":
            st.warning(
                """The grism cross-disperser option is currently not implemented. Please
                select a different cross-disperser type for now.""",
            )

    with st.expander(label="Slit", expanded=False):
        slit_width: ScalarQuantity = st_quantity_input(
            "Slit width",
            min_value=0.0,
            value=40.0,
            step=1.0,
            possible_units=["µm", "mm"],
            default_unit="µm",
            help="""The dispersion-direction physical width of the slit.""",
            key="slit_width",
        )

        slit_length: ScalarQuantity = st_quantity_input(
            "Slit length",
            min_value=0.0,
            value=360.0,
            step=1.0,
            possible_units=["µm", "mm"],
            default_unit="µm",
            help="""The spatial-direction physical length of the slit.""",
            key="slit_length",
        )

        slit_focal_ratio: float = st.number_input(
            "Slit native f/#",
            min_value=1.0,
            max_value=20.0,
            value=3.85,
            step=0.05,
            width=200,
            help="""The native focal ratio of the slit, i.e. the focal ratio of the beam
            entering the spectrograph, at which the slit appears to be the same size as
            the physical slit dimensions above.""",
        )

    with st.expander(label="Collimator and Camera", expanded=False):
        beam_diameter: ScalarQuantity = st_quantity_input(
            "Collimated beam diameter",
            min_value=0.0,
            value=160.0,
            step=1.0,
            possible_units=["mm", "cm", "m"],
            default_unit="mm",
            help="""The diameter of the collimated beam at the echelle grating. This is
                a critical parameter that affects the spectral resolution and the size
                of the spectral format on the detector.""",
        )

        f_col = None

        if beam_diameter.to(u.mm) > grating.beam_diameter.to(u.mm):
            st.warning(
                "Warning: projected beam diameter is larger than grating ruled width!",
            )

        f_cam: ScalarQuantity = st_quantity_input(
            "Camera focal length",
            min_value=0.0,
            value=730.0,
            step=10.0,
            possible_units=["mm", "cm", "m"],
            default_unit="mm",
            key="f_cam",
        )

        st.metric(
            "Camera focal ratio",
            f"f/{(f_cam / beam_diameter).decompose().value:.1f}",
            help="""The focal ratio of the camera, calculated from the camera focal
            length and the collimated beam diameter. Extreme low f-ratios can have
            important implications for the optical design of the camera in terms of
            complexity, image quality, and cost.""",
        )

    with st.expander(label="Detector", expanded=False):
        detector_choices: list[str] = st.multiselect(
            "Select one or more common detectors",
            [*list(DETECTORS.keys()), "Custom 1", "Custom 2"],
            default="STA1600",
            help="""Select from a list of commercially available detectors.""",
            key="detectors",
        )

        detectors: list[Detector] = [
            Detector(
                name=DETECTORS[d].name,
                pixel_size=DETECTORS[d].pixel_size.to(u.um),
                nx=DETECTORS[d].nx,
                ny=DETECTORS[d].ny,
            )
            for d in detector_choices
            if "Custom" not in d
        ]

        if "Custom 1" in detector_choices:
            detector_name: str = st.text_input(
                "Custom detector name",
                value="Custom Detector 1",
                help="""The name of the detector, for plot labelling only.""",
                key="custom_detector_1_name",
            )

            px_size: ScalarQuantity = (
                st.number_input(
                    "Pixel size (µm)",
                    min_value=0.0,
                    value=10.0,
                    step=0.5,
                    key="custom_detector_1_pixel_size",
                )
                * u.um
            )

            n_x: int = st.number_input(
                "Pixels (x)",
                min_value=0,
                max_value=20_000,
                value=9_000,
                step=256,
                key="custom_detector_1_nx",
            )

            n_y: int = st.number_input(
                "Pixels (y)",
                min_value=0,
                max_value=20_000,
                value=9_000,
                step=256,
                key="custom_detector_1_ny",
            )

            detectors.append(
                Detector(
                    name=detector_name,
                    pixel_size=px_size,
                    nx=int(n_x),
                    ny=int(n_y),
                )
            )

        if "Custom 2" in detector_choices:
            detector_name: str = st.text_input(
                "Custom detector name",
                value="Custom Detector 2",
                help="""The name of the detector, for plot labelling only.""",
                key="custom_detector_2_name",
            )

            px_size: ScalarQuantity = (
                st.number_input(
                    "Pixel size (µm)",
                    min_value=0.0,
                    value=10 * u.um,
                    step=0.5,
                    key="custom_detector_2_pixel_size",
                )
                * u.um
            )

            n_x: int = st.number_input(
                "Pixels (x)",
                min_value=0,
                max_value=20_000,
                value=9_000,
                step=256,
                key="custom_detector_2_nx",
            )

            n_y: int = st.number_input(
                "Pixels (y)",
                min_value=0,
                max_value=20_000,
                value=9_000,
                step=256,
                key="custom_detector_2_ny",
            )

            detectors.append(
                Detector(
                    name=detector_name,
                    pixel_size=px_size,
                    nx=int(n_x),
                    ny=int(n_y),
                )
            )

        st.toggle("Show detector name in plot", value=True, key="detector_name")

    with st.expander(label="Plot Format Options", expanded=False):
        fsr_fraction: float = st.slider(
            "FSR fraction to display",
            min_value=0.05,
            max_value=2.0,
            value=1.0,
            step=0.05,
        )

        st.segmented_control(
            "Dispersion direction in plot",
            options=["Vertical (y)", "Horizontal (x)"],
            default="Vertical (y)",
            key="dispersion_direction",
        )

        if st.session_state["dispersion_direction"] is None:
            st.error(
                "Please select a dispersion direction for the plot.",
                icon=":material/error:",
            )

        st.pills(
            "Coordinate transformations",
            options=["Flip x", "Flip y"],
            selection_mode="multi",
            key="coord_transforms",
        )

        st.toggle(label="Label diffraction orders", value=True, key="label_orders")

        st.toggle(label="Label wavelengths", value=True, key="label_wavelengths")

        st.toggle(label="Label atomic lines", value=False, key="label_atomic_lines")

        st.divider()

        st.pills(
            label="Plot type",
            selection_mode="multi",
            options=["Plotly (interactive)", "Matplotlib (static)"],
            default=["Plotly (interactive)"],
            key="plot_type",
        )

    st.space(10)

    st.toggle("Warn about possible design issues", value=False, key="warn")

    st.space(10)

    with st.container(horizontal_alignment="center"):
        st.button(
            "Plot spectral format",
            type="primary",
            key="recompute_plot",
            help="Click to recompute the spectral format plot.",
        )


# Create Spectrograph

MIN_WL: ScalarQuantity = min_wl.to(u.nm)
MAX_WL: ScalarQuantity = max_wl.to(u.nm)

GRATING: EchelleGrating = EchelleGrating(
    groove_frequency=groove_frequency,
    blaze_angle=blaze_angle,
    gamma_angle=0 * u.deg,
    ruled_width=ruled_width,
)

if cross_disperser_type in ["Prism", "Double-pass prism"]:
    glass: Glass = glasses[glass_name]
    CROSS_DISPERSER: CrossDisperser = Prism(
        glass=glass,
        apex_angle=apex_angle,
    )

elif cross_disperser_type == "Multiple prisms":
    CROSS_DISPERSER = TwoPrisms(
        prism1=Prism(
            glass=p1_glasses[p1_glass_name],
            apex_angle=p1_apex_angle,
        ),
        prism2=Prism(
            glass=p2_glasses[p2_glass_name],
            apex_angle=p2_apex_angle,
        ),
    )

elif cross_disperser_type == "Grating":
    CROSS_DISPERSER: CrossDisperser = SimpleGrating(
        groove_frequency=cross_disperser_groove_frequency,
    )

if cross_disperser_type == "Double-pass prism":
    spectrograph: DoublePassEchelleSpectrograph = DoublePassEchelleSpectrograph(
        grating=GRATING,
        cross_disperser=CROSS_DISPERSER,
        detector=Detector(),
        min_wl=MIN_WL,
        max_wl=MAX_WL,
        beam_diameter=beam_diameter,
        collimator_focal_length=f_col,
        camera_focal_length=f_cam,
        slit_width=slit_width,
        slit_length=slit_length,
        slit_native_focal_ratio=slit_focal_ratio,
    )

else:
    spectrograph: EchelleSpectrograph = EchelleSpectrograph(
        grating=GRATING,
        cross_disperser=CROSS_DISPERSER,
        detector=Detector(),
        min_wl=MIN_WL,
        max_wl=MAX_WL,
        beam_diameter=beam_diameter,
        collimator_focal_length=f_col,
        camera_focal_length=f_cam,
        slit_width=slit_width,
        slit_length=slit_length,
        slit_native_focal_ratio=slit_focal_ratio,
    )

# Summary
c1, c2, c3, c4, c5 = st.columns(
    [1, 1, 1.5, 1.25, 1],
    gap="xxsmall",
    vertical_alignment="top",
    border=True,
    width="stretch",
)

with c1:
    st.metric(
        "Number of orders",
        f"{len(spectrograph.spectral_orders)}",
        help="""The number of diffraction orders spanned by the given wavelength
        range.""",
    )

with c2:
    st.metric(
        "Slit magnification",
        f"{spectrograph.slit_magnification:.3f}x",
        help="""The optical magnification factor between the slit plane and the detector
        focal plane. This affects the slit sampling and the imaged slit length on the
        detector. This value is calculated for the blaze peak only, and does not account
        for the anamorphic magnification from the echelle grating, which varies across
        each order. Anamorphic magnification is accounted for separately in the 'slit
        sampling' metric.""",
    )

with c3:
    sampling: tuple[float, ...] = tuple(
        [
            spectrograph.slit_sampling * a
            for a in spectrograph.anamorphic_magnification()
        ]
    )

    st.metric(
        "Slit sampling",
        f"{min(sampling):.1f}—{max(sampling):.1f} pixels",
        help="""The slit sampling in pixels at the detector. Anamorphic magnification is
        accounted for by the range of values given. Any value under 3 pixels will raise
        an (optional) warning that the slit is undersampled.""",
    )
    if min(sampling) < 3 and st.session_state["warn"]:  # noqa: PLR2004
        st.warning("Minimum slit sampling is less than 3 pixels!")

with c4:
    red_sep: ScalarQuantity = spectrograph.reddest_order_image_plane_separation
    blue_sep: ScalarQuantity = spectrograph.bluest_order_image_plane_separation

    st.metric(
        "Minimum order separation",
        f"{(min(red_sep, blue_sep) / spectrograph.detector.pixel_size).si:.1f} pixels",
        help="""The minimum separation between adjacent diffraction orders on the
        detector, at the reddest or bluest orders. This is compared to the imaged slit
        length to check for potential order overlap issues.""",
    )

    if (
        min(red_sep, blue_sep) < spectrograph.slit_image_length
        and st.session_state["warn"]
    ):
        st.warning("Minimum order separation is smaller than the imaged slit length!")

with c5:
    st.metric(
        f"R = λ/Δλ @ {spectrograph.mid_wavelength_rounded:.0f}",
        f"{spectrograph.spectral_resolution(wl=spectrograph.mid_blaze_wavelength):,.0f}",
        help="""The spectral resolution (R = λ/Δλ) of the spectrograph at the central
        wavelength. This should be considered an upper limit, as it does not account for
        any image quality effects, and assumes a perfectly illuminated slit and ideal
        sampling. The actual achievable resolution may be lower due to a number of
        factors not considered here.
        """,
    )


# if not st.session_state.get("recompute_plot", True):
#     st.stop()


with st.container(horizontal_alignment="center"):
    if "Plotly (interactive)" in st.session_state["plot_type"]:
        if st.session_state.get("recompute_plot", True):
            # Plot Echellogram
            with st.spinner("Computing spectral format..."):
                fig = go.Figure()

                for d in detectors:
                    # Detector outline
                    half_x: float = d.size[0].to(u.mm).value / 2
                    half_y: float = d.size[1].to(u.mm).value / 2

                    b = max(half_x, half_y) * 1.125

                    fig.add_shape(
                        type="rect",
                        x0=-half_x,
                        y0=-half_y,
                        x1=half_x,
                        y1=half_y,
                        line={
                            "color": "rgba(255,255,255,0.5)",
                            "width": 3,
                            "dash": "dash",
                        },
                        fillcolor="rgba(0,0,0,0)",
                    )
                    if st.session_state["detector_name"] and d.name:
                        fig.add_annotation(
                            x=0,
                            y=-half_y,
                            text=d.name,
                            xanchor="center",
                            yanchor="top",
                            showarrow=False,
                            font={"size": 20, "color": "rgba(255,255,255,0.5)"},
                        )

                orders: list[int] = spectrograph.spectral_orders
                for m in orders:
                    wls: ArrayQuantity = spectrograph.order_wavelengths(
                        m=m, num=60, FSR_fraction=fsr_fraction
                    )
                    mean_wl: ScalarQuantity = np.mean(wls.to(u.nm).value)
                    color: str = wl_to_hex(mean_wl)

                    positions = np.array(
                        spectrograph.image_plane_position(m=m, wavelength=wls)
                    )
                    xs = positions[:, 0]
                    ys = positions[:, 1]

                    xshift: float = 5
                    yshift: float = 15
                    xlabel = "Cross-dispersion direction (mm)"
                    ylabel = "Dispersion direction (mm)"

                    if st.session_state["dispersion_direction"] == "Horizontal (x)":
                        xs, ys = -ys, -xs
                        xshift: float = 20
                        yshift: float = 0
                        xlabel = "Dispersion direction (mm)"
                        ylabel = "Cross-dispersion direction (mm)"

                    hover_wls: float = wls.to(u.nm).value

                    fig.add_trace(
                        trace=go.Scatter(
                            x=-xs,
                            y=ys,
                            mode="lines",
                            line={"color": color, "width": 3},
                            name=f"m={m}",
                            hovertemplate=(
                                f"m = {m}<br>"
                                + "λ = %{customdata:.1f} nm<br>"
                                + "x = %{x:.2f} mm<br>"
                                + "y = %{y:.2f} mm<extra></extra>"
                            ),
                            customdata=hover_wls,
                            showlegend=False,
                        )
                    )

                    # Label every 5th order
                    if m % 5 == 0:
                        if st.session_state["label_orders"]:
                            fig.add_annotation(
                                x=-xs[0],
                                y=ys[0],
                                text=f"{m}",
                                showarrow=False,
                                font={"size": 10, "color": color},
                                xshift=-xshift,
                                yshift=-yshift,
                                bgcolor="rgba(0,0,0,0.1)",
                                borderpad=1,
                            )
                        if st.session_state["label_wavelengths"]:
                            fig.add_annotation(
                                x=-xs[-1],
                                y=ys[-1],
                                text=f"{mean_wl:.0f} nm",
                                showarrow=False,
                                font={"size": 10, "color": color},
                                xshift=xshift,
                                yshift=yshift,
                                bgcolor="rgba(0,0,0,0.1)",
                                borderpad=1,
                            )

                    if st.session_state["label_atomic_lines"]:
                        min_FSR_wl, max_FSR_wl = spectrograph.grating.order_FSR_bounds(
                            m=m
                        )
                        for name, wl in _atomic_lines.items():
                            if min_FSR_wl <= wl <= max_FSR_wl:
                                x, y = spectrograph.image_plane_position(
                                    m=m, wavelength=wl
                                )

                                if (
                                    st.session_state["dispersion_direction"]
                                    == "Horizontal (x)"
                                ):
                                    x, y = -y, -x

                                fig.add_annotation(
                                    x=-x.value,
                                    y=y.value,
                                    text=f"{name}",
                                    showarrow=False,
                                    font={"size": 10},
                                    bgcolor="rgba(0,0,0,0.1)",
                                    borderpad=1,
                                )

                fig.update_layout(
                    xaxis_title=xlabel,
                    yaxis_title=ylabel,
                    template="plotly_dark",
                    height=1000,
                    yaxis_scaleanchor="x",
                    yaxis_scaleratio=1,
                    margin={"t": 30},
                )

                # # round b to the nearest 20 for nicer tick spacing
                # b: int = 20 * np.ceil(b / 20)

                # fig.update_xaxes(
                #     tickvals=list(range(-int(b), int(b) + 1, 20)),
                #     showgrid=False,
                #     zeroline=False,
                # )
                # fig.update_yaxes(
                #     tickvals=list(range(-int(b), int(b) + 1, 20)),
                #     showgrid=False,
                #     zeroline=False,
                # )

                if "Flip x" in st.session_state["coord_transforms"]:
                    fig.update_xaxes(autorange="reversed")

                if "Flip y" in st.session_state["coord_transforms"]:
                    fig.update_yaxes(autorange="reversed")

                st.session_state["plotly_fig"] = (
                    fig  # Store the figure in session state to prevent re-computation on every interaction
                )

        if st.session_state.get("plotly_fig", False):
            st.plotly_chart(
                st.session_state["plotly_fig"],
                width=800,
                config={"displayModeBar": True},
            )

    if "Matplotlib (static)" in st.session_state["plot_type"]:
        if st.session_state.get("recompute_plot", True):
            with st.spinner("Computing spectral format..."):
                fig: plt.Figure = plt.figure(figsize=(10, 10))
                ax: plt.Axes = fig.gca()

                for d in detectors:
                    # Detector outline
                    half_x: float = d.size[0].to(u.mm).value / 2
                    half_y: float = d.size[1].to(u.mm).value / 2

                    # spectrograph.plot_detector_edge(
                    #     ax=ax,
                    #     name=st.session_state["detector_name"],
                    # )

                    d.plot_edge(ax=ax, name=st.session_state["detector_name"])

                b: float = max(half_x, half_y) * 1.125

                if "Flip x" in st.session_state["coord_transforms"]:
                    ax.set_xlim(left=-b, right=b)
                else:
                    ax.set_xlim(left=b, right=-b)

                if "Flip y" in st.session_state["coord_transforms"]:
                    ax.set_ylim(bottom=b, top=-b)
                else:
                    ax.set_ylim(bottom=-b, top=b)

                ax.set_aspect(aspect="equal")

                for f in range(1):  # number of fibres
                    fiber_spacing = 130 * u.um  # Example!
                    offset: ScalarQuantity = (
                        f * fiber_spacing * spectrograph.slit_magnification
                    )
                    for m in spectrograph.spectral_orders:
                        wls: ArrayQuantity = spectrograph.order_wavelengths(
                            m=m, num=60, FSR_fraction=fsr_fraction
                        )
                        mean_wl: ScalarQuantity = np.mean(wls.value) * wls.unit

                        xs, ys = np.transpose(
                            spectrograph.image_plane_position(m=m, wavelength=wls)
                        )

                        if st.session_state["dispersion_direction"] == "Horizontal (x)":
                            xs, ys = -ys, -xs

                        ax.plot(
                            xs + offset,
                            ys,
                            color=wavelength_to_rgb(mean_wl, fade_factor=0.8),
                            lw=3.5,
                            # path_effects=stroke(thickness=3.5),
                        )

                        if m % 5 == 0:
                            if m == 90:  # noqa: PLR2004
                                mtext = f"{m = }"
                                wltext = f"$\\lambda = {wls[0].to(u.nm):.0f}$"
                            else:
                                mtext = f"{m}"
                                wltext = f"{wls[0].to(u.nm).value:.0f}"

                            if abs(ys[0].value) + 5 > b or abs(ys[-1].value) + 5 > b:
                                continue

                            # if st.session_state["dispersion_direction"] == "Vertical (y)":
                            x_offset = 0
                            y_offset = 1.5
                            ha_m = "center"
                            va_m = "top"
                            ha_wl = "center"
                            va_wl = "bottom"

                            if (
                                st.session_state["dispersion_direction"]
                                == "Horizontal (x)"
                            ):
                                x_offset = -1.5
                                y_offset = 0
                                ha_m = "right"
                                va_m = "center"
                                ha_wl = "left"
                                va_wl = "center"

                            if st.session_state["label_orders"]:
                                ax.text(
                                    (xs[0] + offset).value - x_offset,
                                    ys[0].value - y_offset,
                                    mtext,
                                    fontsize=14,
                                    ha=ha_m,
                                    va=va_m,
                                )

                            if st.session_state["label_wavelengths"]:
                                ax.text(
                                    (xs[-1] + offset).value + x_offset,
                                    ys[-1].value + y_offset,
                                    wltext,
                                    fontsize=14,
                                    ha=ha_wl,
                                    va=va_wl,
                                )

                        if st.session_state["label_atomic_lines"]:
                            min_FSR_wl, max_FSR_wl = (
                                spectrograph.grating.order_FSR_bounds(m=m)
                            )
                            for name, wl in _atomic_lines.items():
                                if min_FSR_wl <= wl <= max_FSR_wl:
                                    x, y = spectrograph.image_plane_position(
                                        m=m, wavelength=wl
                                    )

                                    if (
                                        st.session_state["dispersion_direction"]
                                        == "Horizontal (x)"
                                    ):
                                        x, y = -y, -x

                                    ax.text(
                                        x.value,
                                        y.value,
                                        f"{name}",
                                        fontsize=14,
                                        ha="center",
                                        va="center",
                                        bbox={
                                            "facecolor": "w",
                                            "edgecolor": "k",
                                            "alpha": 0.75,
                                            "boxstyle": "round,pad=0.25",
                                        },
                                    )

                # ax.patch.set_facecolor("black")
                ax.axis("off")
                st.session_state["pyplot_fig"] = (
                    fig  # Store the figure in session state to prevent re-computation on every interaction
                )
        if st.session_state.get("pyplot_fig", False):
            st.pyplot(st.session_state["pyplot_fig"], width=800)
