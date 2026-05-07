#! uv run streamlit run
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from astropy import units as u
from matplotlib import pyplot as plt
from streamlit.components.v1 import html

from curved_air.detectors import DETECTORS, Detector
from curved_air.glass import GLASS_CATALOGUES, GlassCatalogue
from curved_air.instruments.LRS import LowResolutionSpectrograph
from curved_air.optics.dispersers import Prism, SimpleGrating
from curved_air.plotting import plot_style, wl_to_hex
from curved_air.type_hinting import ArrayQuantity, ScalarQuantity

plt.style.use(plot_style)

st.set_page_config(
    page_title="Curved Air: Optical Design Tools",
    page_icon="🌈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.header("Low Resolution Spectrograph")

with st.expander("About this tool", expanded=False):
    st.markdown(
        """
        This tool allows for the basic design of a low-resolution spectrograph with a
        single disperser. Given the disperser choice and other input parameters, the
        spectral format on the detector is computed and visualised, and a basic report
        is generated on the slit sampling and spectral resolution.

        All code is written by Jake Pember: [Github](https://github.com/pemberj/) / [Email](mailto:jacob.pember@ikmail.com)

        Please get in contact if you have any questions, suggestions, or find any bugs!

        You can also [Buy me a pint](https://buymeacoffee.com/pemberj) if you find this tool useful and want to support its development!
        """
    )

    html(
        """
        <script type="text/javascript" src="https://cdnjs.buymeacoffee.com/1.0.0/button.prod.min.js" data-name="bmc-button" data-slug="pemberj" data-color="#FF5F5F" data-emoji="🍺" data-font="Cookie" data-text="Buy me a pint" data-outline-color="#000000" data-font-color="#ffffff"></script>
        """,
        height=70,
    )

with st.sidebar:
    st.header("Spectrograph Configuration")

    with st.expander(label="Wavelength range", expanded=False):
        cols = st.columns(2, gap="xsmall")
        with cols[0]:
            min_wl_value: float = st.number_input(
                "Min λ",
                min_value=1.0,
                max_value=2000.0,
                value=380.0,
                step=1.0,
                key="min_wl_value",
            )
        with cols[1]:
            min_wl_unit: u.Unit = u.Unit(
                st.segmented_control(
                    "Unit", ["nm", "um"], default="nm", key="min_wl_unit"
                )
            )
        min_wl: ScalarQuantity = min_wl_value * min_wl_unit
        st.session_state["min_wl"] = min_wl

        cols = st.columns(2, gap="xsmall")
        with cols[0]:
            max_wl_value: float = st.number_input(
                "Max λ",
                min_value=1.0,
                max_value=2000.0,
                value=950.0,
                step=1.0,
                key="max_wl_value",
            )
        with cols[1]:
            max_wl_unit: u.Unit = u.Unit(
                st.segmented_control(
                    "Unit", ["nm", "um"], default="nm", key="max_wl_unit"
                )
            )
        max_wl: ScalarQuantity = max_wl_value * max_wl_unit
        st.session_state["max_wl"] = max_wl

    with st.expander(label="Disperser", expanded=False):
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

                _prism: Prism = Prism(
                    glass=glasses[glass_name],
                    apex_angle=apex_angle,
                    min_wl=min_wl,
                    max_wl=max_wl,
                )

                st.metric(
                    "Calculated incidence angle",
                    f"{_prism.incidence_angle.value:.2f} degrees",
                    help="""This tool automatically calculates the
                    appropriate incidence angle such that the spectral format is symmetrically
                    centred on the detector.""",
                )

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

    with st.expander(label="Slit", expanded=False):
        cols = st.columns(2, gap="xsmall")
        with cols[0]:
            slit_width_value: float = st.number_input(
                "Slit width",
                min_value=1.0,
                max_value=1000.0,
                value=50.0,
                step=5.0,
                help="""The dispersion-direction physical width of the slit.""",
            )

        with cols[1]:
            slit_width_unit: u.Unit = u.Unit(
                st.segmented_control(
                    "Unit",
                    options=["um", "mm"],
                    default="um",
                )
            )

        slit_width: ScalarQuantity = slit_width_value * slit_width_unit

        cols = st.columns(2, gap="xsmall")
        with cols[0]:
            slit_length_value: float = st.number_input(
                "Slit length",
                min_value=1.0,
                max_value=1000.0,
                value=5.0,
                step=5.0,
                help="""The spatial-direction physical length of the slit.""",
            )

        with cols[1]:
            slit_length_unit: u.Unit = u.Unit(
                st.segmented_control(
                    "Unit",
                    options=["um", "mm"],
                    default="mm",
                )
            )

        slit_length: ScalarQuantity = slit_length_value * slit_length_unit

        slit_f_number: float = st.number_input(
            "Slit native f/#",
            min_value=1.0,
            max_value=20.0,
            value=4.0,
            step=0.05,
            help="""The native focal ratio of the slit, i.e. the focal ratio of the beam
            entering the spectrograph, at which the slit appears to be the same size as
            the physical slit dimensions above.""",
        )

    with st.expander(label="Collimator and Camera", expanded=False):
        beam_diameter: ScalarQuantity = (
            st.number_input(
                "Collimated beam diameter (mm)",
                min_value=10.0,
                max_value=1200.0,
                value=100.0,
                step=1.0,
                help="""The diameter of the collimated beam at the disperser.""",
            )
            * u.mm
        )

        f_coll: ScalarQuantity = (
            st.number_input(
                "Collimator focal length (mm)",
                min_value=50.0,
                max_value=3000.0,
                value=800.0,
                step=10.0,
            )
            * u.mm
        )

        st.metric(
            "Collimator focal ratio",
            f"f/{(f_coll / beam_diameter).decompose().value:.1f}",
            help="""The focal ratio (f-number) of the collimator, calculated from its
            focal length and the collimated beam diameter. Very low f-numbers can have
            important implications for the optical design of the camera in terms of
            complexity, image quality, and cost.""",
        )

        f_cam: ScalarQuantity = (
            st.number_input(
                "Camera focal length (mm)",
                min_value=50.0,
                max_value=3000.0,
                value=500.0,
                step=10.0,
            )
            * u.mm
        )

        st.metric(
            "Camera focal ratio",
            f"f/{(f_cam / beam_diameter).decompose().value:.1f}",
            help="""The focal ratio (f-number) of the camera, calculated from the camera
            focal length and the collimated beam diameter. Very low f-numbers can have
            important implications for the optical design of the camera in terms of
            complexity, image quality, and cost.""",
        )

    with st.expander(label="Detector", expanded=False):
        detector: str = st.selectbox(
            "Common detectors",
            [*list(DETECTORS.keys())],
            index=5,  # default to 4k, 15um
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
        st.segmented_control(
            "Dispersion direction in plot",
            options=["Vertical (y)", "Horizontal (x)"],
            default="Horizontal (x)",
            key="dispersion_direction",
        )

        st.pills(
            "Coordinate transformations",
            options=["Flip x", "Flip y"],
            selection_mode="multi",
            key="coord_transforms",
        )

        st.divider()

        st.pills(
            label="Plot type",
            selection_mode="multi",
            options=["Plotly (interactive)", "Matplotlib (static)"],
            default=["Plotly (interactive)"],
            key="plot_type",
        )

    st.space(10)

    # st.toggle("Warn about possible design issues", value=False, key="warn")

    # st.space(10)

    # with st.container(horizontal_alignment="center"):
    # st.button("Recompute", type="primary")


# Create Spectrograph

if st.session_state["disperser_type"] == "Prism":
    disperser = Prism(glass=glasses[glass_name], apex_angle=apex_angle)
elif st.session_state["disperser_type"] == "Grating":
    disperser = SimpleGrating(groove_frequency=groove_frequency)

lrs = LowResolutionSpectrograph(
    disperser=disperser,
    detector=Detector(name=detector_name, pixel_size=pixel_size, nx=nx, ny=ny),
    min_wl=min_wl,
    max_wl=max_wl,
    beam_diameter=beam_diameter,
    slit_width=slit_width,
    slit_length=slit_length,
    slit_native_focal_ratio=slit_f_number,
    collimator_focal_length=f_coll,
    camera_focal_length=f_cam,
)

NUM_WAVELENGTHS: int = 100


# Summary
cols = iter(
    st.columns(5, gap="xxsmall", vertical_alignment="top", border=True, width="stretch")
)

with next(cols):
    st.metric(
        "Slit magnification",
        f"{lrs.slit_magnification:.2f}x",
        help="""The optical magnification factor between the slit plane and the detector
        focal plane. This affects the slit sampling and the imaged slit length on the
        detector.""",
    )

with next(cols):
    st.metric(
        "Slit sampling",
        f"{lrs.slit_sampling:.1f} pixels",
        help="""The sampling in detector pixels of the slit in dispersion direction.
        Any value under 3 pixels will raise an (optional) warning that the slit is
        undersampled.""",
    )

resolution_help: str = """The spectral resolution (R = λ/Δλ) the specified wavelength.
        wavelength. This should be considered an upper limit, as it does not account for
        any image quality effects, and assumes a perfectly illuminated slit and ideal
        sampling. The actual achievable resolution may be lower due to a number of
        factors not considered here.
        """


with next(cols):
    st.metric(
        f"R = λ/Δλ @ {lrs.min_wl:.0f}",
        f"{lrs.spectral_resolution(wl=lrs.min_wl):,.0f}",
        help=resolution_help,
    )

with next(cols):
    # round to the nearest 50nm
    mid_wl: ScalarQuantity = round(lrs.mid_wl.to(u.nm).value / 50) * 50 * u.nm
    st.metric(
        f"R = λ/Δλ @ {mid_wl:.0f}",
        f"{lrs.spectral_resolution(wl=mid_wl):,.0f}",
        help=resolution_help,
    )

with next(cols):
    st.metric(
        f"R = λ/Δλ @ {lrs.max_wl:.0f}",
        f"{lrs.spectral_resolution(wl=lrs.max_wl):,.0f}",
        help=resolution_help,
    )


with st.container(horizontal_alignment="center"):
    if "Plotly (interactive)" in st.session_state["plot_type"]:
        with st.spinner("Computing spectral format..."):
            fig: go.Figure = go.Figure()

            # Detector outline
            half_x: float = lrs.detector.size[0].to(u.mm).value / 2
            half_y: float = lrs.detector.size[1].to(u.mm).value / 2

            b = max(half_x, half_y) * 1.125

            fig.add_shape(
                type="rect",
                x0=-half_x,
                y0=-half_y,
                x1=half_x,
                y1=half_y,
                line={"color": "rgba(255,255,255,0.5)", "width": 3, "dash": "dash"},
                fillcolor="rgba(0,0,0,0)",
            )
            if st.session_state["detector_name"] and detector_name:
                fig.add_annotation(
                    x=0,
                    y=-half_y,
                    text=detector_name,
                    xanchor="center",
                    yanchor="top",
                    showarrow=False,
                    font={"size": 20, "color": "rgba(255,255,255,0.5)"},
                )

            wls: ArrayQuantity = lrs.equidistant_wavelengths(num=NUM_WAVELENGTHS)
            for wl in wls:
                color: str = wl_to_hex(wl.to(u.nm).value)

                slit_image_vertices = lrs.slit_image_vertices(wavelength=wl)

                slit_image_x: list[float] = [
                    float(v.x.value) for v in slit_image_vertices
                ]
                slit_image_y: list[float] = [
                    float(v.y.value) for v in slit_image_vertices
                ]

                xs = [*slit_image_x, slit_image_x[0]]
                ys = [*slit_image_y, slit_image_y[0]]

                if st.session_state["dispersion_direction"] == "Vertical (y)":
                    xs, ys = ys, xs

                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="lines",
                        line={"color": color, "width": 3},
                        name=f"{wl:.0f}",
                        hovertemplate=(
                            f"λ = {wl:.1f}<br>"
                            + "x = %{x:.2f} mm<br>"
                            + "y = %{y:.2f} mm<extra></extra>"
                        ),
                        # customdata=float(wl.to(u.nm).value),
                        showlegend=False,
                    )
                )

            fig.update_layout(
                # xaxis_title=xlabel,
                # yaxis_title=ylabel,
                template="plotly_dark",
                height=1000,
                yaxis_scaleanchor="x",
                yaxis_scaleratio=1,
                margin={"t": 30},
            )

            if "Flip x" in st.session_state["coord_transforms"]:
                fig.update_xaxes(autorange="reversed")

            if "Flip y" in st.session_state["coord_transforms"]:
                fig.update_yaxes(autorange="reversed")

            st.plotly_chart(fig, width=800, config={"displayModeBar": True})

    if "Matplotlib (static)" in st.session_state["plot_type"]:
        with st.spinner("Computing spectral format..."):
            fig: plt.Figure = plt.figure(figsize=(6, 6))
            ax: plt.Axes = fig.gca()

            lrs.plot_detector_edge(ax=ax, name=st.session_state["detector_name"])

            lrs.plot_spectrum(
                ax=ax,
                num_wavelengths=NUM_WAVELENGTHS,
                dispersion_direction=st.session_state["dispersion_direction"][-2:-1],
            )

            if "Flip x" in st.session_state["coord_transforms"]:
                ax.set_xlim(ax.get_xlim()[::-1])
            if "Flip y" in st.session_state["coord_transforms"]:
                ax.set_ylim(ax.get_ylim()[::-1])

            ax.set_aspect(aspect="equal")

            # ax.patch.set_facecolor("black")
            ax.axis("off")
            st.pyplot(fig, width=800)
