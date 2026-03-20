import plotly.graph_objects as go
import streamlit as st
from astropy import units as u

from curved_air.detectors import DETECTORS, Detector
from curved_air.glass import GLASS_CATALOGUES, GlassCatalogue
from curved_air.instruments.LRS import LowResolutionSpectrograph
from curved_air.optics.dispersers import Prism, SimpleGrating
from curved_air.type_hinting import ScalarQuantity

st.title("Low Resolution Spectrograph")

# Sidebar controls
st.sidebar.header("Spectrograph Configuration")

with st.sidebar:
    with st.expander(label="Disperser", expanded=True):
        # Disperser parameters
        st.segmented_control(
            label="Disperser Type",
            options=[
                "Prism",
                "Grating",
            ],
            default="Prism",
            key="disperser_type",
        )

        if st.session_state["disperser_type"] == "Prism":
            catalogue: str = st.selectbox(
                "Glass catalogue",
                GLASS_CATALOGUES.keys(),
                index=list(GLASS_CATALOGUES.keys()).index("Schott")
                if "Schott" in GLASS_CATALOGUES
                else 0,
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
            )

            apex_angle: ScalarQuantity = (
                st.number_input(
                    "Apex angle (°)",
                    min_value=5.0,
                    max_value=60.0,
                    value=60.0,
                    step=1.0,
                )
                * u.deg
            )

            st.toggle(
                "Plot glass refractive index",
                value=False,
                key="ref_index_plot_toggle",
            )

            if st.session_state["ref_index_plot_toggle"]:
                wls: list[float] = glasses[glass_name].wl

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

        elif st.session_state["disperser_type"] == "Grating":
            groove_frequency: ScalarQuantity = (
                st.number_input(
                    "Groove frequency (grooves / mm)",
                    min_value=10.0,
                    max_value=2000.0,
                    value=200.0,
                    step=10.0,
                )
                * 1
                / u.mm
            )

            if st.session_state.get("max_wl", 950) > 2 * st.session_state.get(
                "min_wl", 380
            ):
                st.warning(
                    """Warning: with a grating disperser, the wavelength range should not
                    exceed an octave (max λ should be less than 2x min λ) to avoid
                    significant wavelength overlap issues.""",
                )

            st.info(
                """Note: only the first diffraction order of the grating is displayed in the
                spectral format plot.""",
            )

    with st.expander(label="Wavelength range", expanded=False):
        min_wl: float = (
            st.number_input(
                "Min λ (nm)",
                min_value=100.0,
                max_value=5000.0,
                value=380.0,
                step=5.0,
                key="min_wl",
            )
            * u.nm
        )

        max_wl: float = (
            st.number_input(
                "Max λ (nm)",
                min_value=100.0,
                max_value=5000.0,
                value=950.0,
                step=5.0,
                key="max_wl",
            )
            * u.nm
        )

    with st.expander(label="Slit", expanded=False):
        slit_width: ScalarQuantity = (
            st.number_input(
                "Slit width (µm)",
                min_value=5.0,
                max_value=2000.0,
                value=40.0,
                step=5.0,
                help="""The dispersion-direction width of the spectrograph input (slit
                or fibre).""",
            )
            * u.um
        )

        slit_length: ScalarQuantity = (
            st.number_input(
                "Slit length (µm)",
                min_value=10.0,
                max_value=5000.0,
                value=360.0,
                step=10.0,
                help="""The cross-dispersion-direction length of the spectrograph input.
                For an instrument with multiple fibre inputs, this refers to the total
                length of the input geometry.""",
            )
            * u.um
        )

        slit_focal_ratio: float = st.number_input(
            "Slit native F/#",
            min_value=1.0,
            max_value=20.0,
            value=3.85,
            step=0.05,
            help="""The native focal ratio of the slit, i.e. the focal ratio of the beam
            entering the spectrograph, at which the slit appears to be the same size as
            the physical slit dimensions above.""",
        )

    with st.expander(label="Collimator and Camera", expanded=False):
        if True:  # Temp
            beam_diameter: ScalarQuantity = (
                st.number_input(
                    "Collimated beam diameter (mm)",
                    min_value=10.0,
                    max_value=1200.0,
                    value=160.0,
                    step=1.0,
                    help="""The diameter of the collimated beam at the disperser.
                    This is a critical parameter that affects the spectral resolution
                    and the size of the spectral format on the detector.""",
                )
                * u.mm
            )

        f_cam: ScalarQuantity = (
            st.number_input(
                "Camera focal length (mm)",
                min_value=50.0,
                max_value=3000.0,
                value=730.0,
                step=10.0,
            )
            * u.mm
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
        detector: str = st.selectbox(
            "Common detectors",
            [*list(DETECTORS.keys())],
            index=0,
            help="""Select from a list of commercially available detectors.""",
            disabled=st.session_state.get("customise_detector", False),
        )

        st.toggle(
            "Custom detector parameters",
            value=False,
            help="""Input your own values for the detector parameters.""",
            key="customise_detector",
        )

        if st.session_state["customise_detector"]:
            detector_name: str = st.text_input(
                "Detector name",
                value=DETECTORS[detector].name,
                help="""The name of the detector, for plot labelling only.""",
            )

            pixel_size: ScalarQuantity = (
                st.number_input(
                    "Pixel size (µm)",
                    min_value=1.0,
                    max_value=50.0,
                    value=DETECTORS[detector].pixel_size.to(u.um).value,
                    step=0.5,
                    key="sf_px",
                )
                * u.um
            )

            nx: int = st.number_input(
                "Pixels (x)",
                min_value=256,
                max_value=20000,
                value=DETECTORS[detector].nx,
                step=256,
                key="sf_nx",
            )

            ny: int = st.number_input(
                "Pixels (y)",
                min_value=256,
                max_value=20000,
                value=DETECTORS[detector].ny,
                step=256,
                key="sf_ny",
            )

        else:
            detector_name: str = DETECTORS[detector].name
            pixel_size: ScalarQuantity = DETECTORS[detector].pixel_size.to(u.um)
            nx: int = DETECTORS[detector].nx
            ny: int = DETECTORS[detector].ny

        st.toggle("Show detector name in plot", value=True, key="detector_name")

    with st.expander(label="Plot Format Options", expanded=False):
        fsr_fraction: float = st.slider(
            "FSR fraction to display",
            min_value=0.1,
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
        st.button("Recompute", type="primary")


# Create Spectrograph

MIN_WL: ScalarQuantity = min_wl.to(u.nm)
MAX_WL: ScalarQuantity = max_wl.to(u.nm)

if st.session_state["disperser_type"] == "Prism":
    disperser = Prism(glass=glasses[glass_name], apex_angle=apex_angle)
elif st.session_state["disperser_type"] == "Grating":
    disperser = SimpleGrating(groove_frequency=groove_frequency)

lrs = LowResolutionSpectrograph(
    disperser=disperser,
    detector=Detector(name=detector_name, pixel_size=pixel_size, nx=nx, ny=ny),
    min_wl=MIN_WL,
    max_wl=MAX_WL,
    beam_diameter=100 * u.mm,
    collimator_focal_length=800 * u.mm,
    slit_width=50 * u.um,
    slit_length=2 * u.mm,
    slit_native_focal_ratio=4,
    camera_focal_length=600 * u.mm,
)
