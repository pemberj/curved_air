import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from astropy import units as u
from matplotlib.patches import Rectangle

from curved_air.glass import GLASS_CATALOGUES, GlassCatalogue
from curved_air.optics.dispersers import Prism
from curved_air.type_hinting import ScalarQuantity

st.set_page_config(page_title="Low Resolution Spectrograph", layout="wide")

st.title("Low Resolution Spectrograph - Image Plane")

# Sidebar controls
st.sidebar.header("Spectrograph Configuration")

with st.sidebar.expander(label="Disperser", expanded=True):
    # Disperser parameters
    st.sidebar.segmented_control(
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
        cross_disperser_groove_frequency: ScalarQuantity = (
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
                """Warning: with a grating cross-disperser, the wavelength range
                should not exceed an octave (max λ should be less than 2x min λ) to
                avoid significant order overlap issues.""",
            )

        st.info(
            """Note: only the first diffraction order of the cross-disperser is
            displayed in the spectral format plot.""",
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


# Create Spectrograph

MIN_WL: ScalarQuantity = min_wl.to(u.nm)
MAX_WL: ScalarQuantity = max_wl.to(u.nm)

GRATING: EchelleGrating = EchelleGrating(
    groove_frequency=groove_freq,
    blaze_angle=blaze_angle,
    gamma_angle=0 * u.deg,
    min_wl=MIN_WL,
    max_wl=MAX_WL,
    ruled_width=ruled_width,
)

if cross_disperser_type == "Prism":
    glass: Glass = glasses[glass_name]
    CROSS_DISPERSER: CrossDisperser = Prism(
        glass=glass,
        apex_angle=apex_angle,
        min_wl=MIN_WL,
        max_wl=MAX_WL,
    )
else:
    CROSS_DISPERSER: CrossDisperser = SimpleGrating(
        groove_frequency=cross_disperser_groove_frequency,
        min_wl=MIN_WL,
        max_wl=MAX_WL,
    )

DETECTOR: Detector = Detector(
    name=detector_name,
    pixel_size=px_size,
    nx=int(n_x),
    ny=int(n_y),
)

spectrograph: EchelleSpectrograph = EchelleSpectrograph(
    grating=GRATING,
    cross_disperser=CROSS_DISPERSER,
    detector=DETECTOR,
    min_wl=MIN_WL,
    max_wl=MAX_WL,
    beam_diameter=beam_diameter,
    collimator_focal_length=f_col,
    camera_focal_length=f_cam,
    slit_width=slit_width,
    slit_length=slit_length,
    slit_native_focal_ratio=slit_focal_ratio,
)
