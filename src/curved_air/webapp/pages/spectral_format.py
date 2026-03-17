"""
This calculator allows for generating a spectral format plot (echellogram) for a given
echelle spectrograph design configuration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from astropy import units as u
from matplotlib import pyplot as plt

from curved_air.detectors import DETECTORS, Detector
from curved_air.glass import GLASS_CATALOGUES, Glass, GlassCatalogue
from curved_air.instruments.echelle import EchelleSpectrograph
from curved_air.optics.dispersers import (
    CrossDisperser,
    EchelleGrating,
    Prism,
    SimpleGrating,
)
from curved_air.other import _atomic_lines
from curved_air.plotting import plot_style, wavelength_to_rgb, wl_to_hex
from curved_air.utils import richardson_gratings

if TYPE_CHECKING:
    from curved_air.type_hinting import ArrayQuantity, ScalarQuantity

plt.style.use(plot_style)


st.header("Spectral Format")

with st.expander("About this calculator", expanded=False):
    st.markdown(
        """
        This calculator generates a spectral format plot (echellogram) for a given
        echelle spectrograph design. Use the sidebar to adjust the spectrograph
        parameters, and the format plot will update automatically.
        You can hover over the plot to see the wavelength and position of each point on
        the detector.

        This is meant for quick exploration of the top-level spectrograph design space.
        The results shown here may differ from a more detailed ray-tracing analysis.

        All code is written by [Jake Pember](https://github.com/pemberj/)

        Please get in contact if you have any questions, suggestions, or find any bugs!
        """
    )

st.divider()

# Interface
with st.sidebar:
    with st.expander(label="Echelle Grating", expanded=False):
        ECHELLE_GRATINGS: dict[str, EchelleGrating] = {
            f"{g.catalogue_number}: {g.groove_frequency.value:g}g/mm R{g.R:g}": g
            for g in richardson_gratings()
        }

        grating: str = st.selectbox(
            "Richardson gratings",
            options=list(ECHELLE_GRATINGS.keys()),
            index=6,  # MARVEL grating
            help="""Select from a list of commercially available echelle gratings from
            Richardson Gratings. The grating parameters will be pre-filled based on the
            selected option.""",
            disabled=st.session_state.get("customise_grating", False),
        )

        grating: EchelleGrating = ECHELLE_GRATINGS[grating]

        st.toggle(
            "Custom grating parameters",
            value=False,
            key="customise_grating",
            help="""Input your own values for the grating groove spacing, blaze angle,
            and ruled width.""",
        )

        if st.session_state["customise_grating"]:
            groove_freq: ScalarQuantity = (
                st.number_input(
                    "Groove freqency (grooves / mm)",
                    min_value=1.0,
                    max_value=500.0,
                    value=grating.groove_frequency.to(u.mm**-1).value,
                    step=0.1,
                )
                * 1
                / u.mm
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

            ruled_width: ScalarQuantity = (
                st.number_input(
                    "Ruled width (mm)",
                    min_value=0.0,
                    # max_value=1200.0,
                    value=grating.ruled_width.to(u.mm).value,
                    step=0.05,
                    help="""The ruled width of the echelle grating, in millimetres. This
                    refers to the dimension perpendicular to the grooves themselves.""",
                )
                * u.mm
            )

        else:
            groove_freq: float = grating.groove_frequency.to(u.mm**-1)
            blaze_angle: float = grating.blaze_angle.to(u.deg)
            ruled_width: float = grating.ruled_width.to(u.mm)

    with st.expander(label="Cross-disperser", expanded=False):
        cross_disperser_type: str = st.segmented_control(
            "Type",
            [
                "Prism",
                "Grating",
                # "Grism",  # TODO
                # "Multiple prisms",  # TODO
            ],
            default="Prism",
        )  # ty:ignore[invalid-assignment]

        if cross_disperser_type == "Prism":
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

        elif cross_disperser_type == "Grating":
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

    with st.expander(label="Collimator and Camera", expanded=False):
        if True:  # Temp
            beam_diameter: ScalarQuantity = (
                st.number_input(
                    "Collimated beam diameter (mm)",
                    min_value=10.0,
                    max_value=1200.0,
                    value=160.0,
                    step=1.0,
                    help="""The diameter of the collimated beam at the echelle grating.
                    This is a critical parameter that affects the spectral resolution
                    and the size of the spectral format on the detector.""",
                )
                * u.mm
            )

            f_col = None

            if beam_diameter > grating.beam_diameter.to(u.mm):
                st.warning(
                    "Warning: projected beam diameter is larger than grating ruled width!",
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

            px_size: ScalarQuantity = (
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

            n_x: int = st.number_input(
                "Pixels (x)",
                min_value=256,
                max_value=20000,
                value=DETECTORS[detector].nx,
                step=256,
                key="sf_nx",
            )

            n_y: int = st.number_input(
                "Pixels (y)",
                min_value=256,
                max_value=20000,
                value=DETECTORS[detector].ny,
                step=256,
                key="sf_ny",
            )

        else:
            detector_name: str = DETECTORS[detector].name
            px_size: ScalarQuantity = DETECTORS[detector].pixel_size.to(u.um)
            n_x: int = DETECTORS[detector].nx
            n_y: int = DETECTORS[detector].ny

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
    mid_wavelength: ScalarQuantity = (MAX_WL + MIN_WL) / 2

    # round to the nearest 50nm
    mid_wavelength: ScalarQuantity = (
        round(mid_wavelength.to(u.nm).value / 50) * 50 * u.nm
    )

    st.metric(
        f"R = λ/Δλ @ {mid_wavelength:.0f}",
        f"{spectrograph.spectral_resolution(wl=mid_wavelength):,.0f}",
        help="""The spectral resolution (R = λ/Δλ) of the spectrograph at the central
        wavelength. This should be considered an upper limit, as it does not account for
        any image quality effects, and assumes a perfectly illuminated slit and ideal
        sampling. The actual achievable resolution may be lower due to a number of
        factors not considered here.
        """,
    )

with st.container(horizontal_alignment="center"):
    if "Plotly (interactive)" in st.session_state["plot_type"]:
        # Plot Echellogram
        with st.spinner("Computing spectral format..."):
            fig = go.Figure()

            # Detector outline
            half_x: float = spectrograph.detector.size[0].to(u.mm).value / 2
            half_y: float = spectrograph.detector.size[1].to(u.mm).value / 2

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

                if st.session_state["dispersion_direction"] == "Horizontal (x)":
                    xs, ys = -ys, -xs
                    xshift: float = 20
                    yshift: float = 0
                    xlabel = "Dispersion direction (mm)"
                    ylabel = "Cross-dispersion direction (mm)"

                elif st.session_state["dispersion_direction"] == "Vertical (y)":
                    xshift: float = 5
                    yshift: float = 15
                    xlabel = "Cross-dispersion direction (mm)"
                    ylabel = "Dispersion direction (mm)"

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
                    min_FSR_wl, max_FSR_wl = spectrograph.grating.order_FSR_bounds(m=m)
                    for name, wl in _atomic_lines.items():
                        if min_FSR_wl <= wl <= max_FSR_wl:
                            x, y = spectrograph.image_plane_position(m=m, wavelength=wl)

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

            st.plotly_chart(fig, width=800, config={"displayModeBar": True})

    if "Matplotlib (static)" in st.session_state["plot_type"]:
        with st.spinner("Computing spectral format..."):
            fig: plt.Figure = plt.figure(figsize=(10, 10))
            ax: plt.Axes = fig.gca()

            # Detector outline
            half_x: float = spectrograph.detector.size[0].to(u.mm).value / 2
            half_y: float = spectrograph.detector.size[1].to(u.mm).value / 2

            spectrograph.plot_detector_edge(
                ax=ax,
                name=st.session_state["detector_name"],
            )

            b: float = max(half_x, half_y) * 1.125

            if "Flip x" in st.session_state["coord_transforms"]:
                ax.set_xlim(left=b, right=-b)
            else:
                ax.set_xlim(left=-b, right=b)

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

                        if st.session_state["dispersion_direction"] == "Vertical (y)":
                            x_offset = 0
                            y_offset = 1.5
                            ha_m = "center"
                            va_m = "top"
                            ha_wl = "center"
                            va_wl = "bottom"

                        elif (
                            st.session_state["dispersion_direction"] == "Horizontal (x)"
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
            st.pyplot(fig, width=800)
