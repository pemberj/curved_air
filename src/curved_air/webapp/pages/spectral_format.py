"""
Spectral Format page.

Build a full echelle spectrograph and visualise the échellogram on the detector
using interactive Plotly plots.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from astropy import units as u
from matplotlib import pyplot as plt

from curved_air.detectors import DETECTORS, Detector
from curved_air.glass import Glass, GlassCatalogue
from curved_air.instruments.echelle import EchelleSpectrograph
from curved_air.optics.dispersers import (
    CrossDisperser,
    EchelleGrating,
    Prism,
    SimpleGrating,
)
from curved_air.other import _atomic_lines
from curved_air.plotting import plot_style, wavelength_to_rgb
from curved_air.utils import richardson_gratings

if TYPE_CHECKING:
    from curved_air.type_hinting import ArrayQuantity, ScalarQuantity

plt.style.use(plot_style)


st.header("Spectral Format")

with st.expander("About this calculator", expanded=False):
    st.markdown(
        """
        This calculator generates a spectral format plot (echellogram) for a given echelle
        spectrograph design. Use the sidebar to adjust the spectrograph parameters, and the
        format plot will update automatically.
        You can hover over the plot to see the wavelength and position of each point on the
        detector.

        This is meant for quick exploration of the top-level spectrograph design space. The
        results shown here may differ from a more detailed ray-tracing analysis.
        """
    )

st.divider()

# Interface
with st.sidebar:
    st.markdown("Spectrograph configuration")

    with st.expander(label="Echelle Grating", expanded=False):
        ECHELLE_GRATINGS: dict[str, EchelleGrating] = {
            f"{g.catalogue_number}: {g.groove_frequency.value:g}g/mm R{g.R:g}": g
            for g in richardson_gratings()
        }

        grating: str = st.selectbox(
            "Richardson gratings",
            options=list(ECHELLE_GRATINGS.keys()),
            index=6,  # MARVEL grating
            key="sf_grating",
        )

        grating: EchelleGrating = ECHELLE_GRATINGS[grating]

        customise_grating: bool = st.toggle(
            "Customise grating parameters?", key="sf_custom_grating"
        )

        if customise_grating:
            groove_freq = (
                st.number_input(
                    "Grooves / mm",
                    min_value=1.0,
                    max_value=500.0,
                    value=grating.groove_frequency.to(u.mm**-1).value,
                    step=0.1,
                    key="sf_gf",
                )
                * 1
                / u.mm
            )

            blaze_angle = (
                st.number_input(
                    "Blaze angle (degrees)",
                    min_value=10.0,
                    max_value=89.0,
                    value=grating.blaze_angle.to(u.deg).value,
                    step=0.1,
                    key="sf_ba",
                )
                * u.deg
            )

            gamma_angle = (
                st.number_input(
                    "Gamma angle (degrees)",
                    min_value=0.0,
                    max_value=10.0,
                    value=grating.gamma_angle.to(u.deg).value,
                    step=0.05,
                    key="sf_ga",
                )
                * u.deg
            )

        else:
            groove_freq: float = grating.groove_frequency.to(u.mm**-1)
            blaze_angle: float = grating.blaze_angle.to(u.deg)
            gamma_angle: float = grating.gamma_angle.to(u.deg)

    with st.expander(label="Cross-disperser", expanded=False):
        xd_type = st.selectbox(
            "Type",
            ["Prism", "Simple grating"],
            key="sf_xd",
        )

        if xd_type == "Prism":
            catalogue = st.selectbox(
                "Glass catalogue",
                ["Schott", "Ohara", "CDGM"],
                key="sf_xd_catalogue",
            )

            try:
                glasses = GlassCatalogue(catalogue.lower())
                glass_names = sorted(glasses.keys())
            except Exception:
                st.error(f"Could not load {catalogue} glass catalogue.")

            if catalogue.lower() == "schott":
                default_glass = "F2"
            elif catalogue.lower() == "ohara":
                default_glass = "PBM2Y"
            elif catalogue.lower() == "cdgm":
                default_glass = "F2"
            else:
                default_glass = glass_names[0]

            glass_name = st.selectbox(
                "Glass",
                glass_names,
                index=glass_names.index(default_glass)
                if default_glass in glass_names
                else 0,
                key="sf_glass",
            )

            apex_angle = (
                st.number_input(
                    "Apex angle (°)",
                    min_value=5.0,
                    max_value=60.0,
                    value=60.0,
                    step=1.0,
                    key="sf_apex",
                )
                * u.deg
            )

        else:
            xd_groove_freq = (
                st.number_input(
                    "Cross-disp grooves / mm",
                    min_value=10.0,
                    max_value=2000.0,
                    value=200.0,
                    step=10.0,
                    key="sf_xd_gf",
                )
                * 1
                / u.mm
            )

    with st.expander(label="Wavelength range", expanded=False):
        min_wl = (
            st.number_input(
                "Min λ (nm)",
                min_value=100.0,
                max_value=5000.0,
                value=380.0,
                key="sf_minwl",
            )
            * u.nm
        )

        max_wl = (
            st.number_input(
                "Max λ (nm)",
                min_value=100.0,
                max_value=5000.0,
                value=900.0,
                key="sf_maxwl",
            )
            * u.nm
        )

    with st.expander(label="Slit", expanded=False):
        slit_width = (
            st.number_input(
                "Slit width (µm)",
                min_value=5.0,
                max_value=2000.0,
                value=25.0,
                step=5.0,
                key="sf_slit_width",
            )
            * u.um
        )

        slit_length = (
            st.number_input(
                "Slit length (µm)",
                min_value=10.0,
                max_value=5000.0,
                value=360.0,
                step=10.0,
                key="sf_slit_length",
            )
            * u.um
        )

        slit_focal_ratio = st.number_input(
            "Slit native F/#",
            min_value=1.0,
            max_value=20.0,
            value=3.85,
            step=0.05,
            key="sf_focal_ratio",
        )

    with st.expander(label="Collimator and Camera", expanded=False):
        # which_optics_to_specify = st.pills(
        #     "Which optics to specify?",
        #     options=["Collimator focal length", "Collimated beam diameter"],
        #     default="Collimator focal length",
        #     key="sf_optics_spec",
        # )

        # if which_optics_to_specify == "Collimator focal length":
        #     f_col = (
        #         st.number_input(
        #             "Collimator FL (mm)",
        #             min_value=100.0,
        #             max_value=5000.0,
        #             value=1125.0,
        #             step=25.0,
        #             key="sf_fcol",
        #         )
        #         * u.mm
        #     )

        #     beam_diameter = None

        # elif which_optics_to_specify == "Collimated beam diameter":
        if True:  # Temp
            beam_diameter = (
                st.number_input(
                    "Collimated beam diameter (mm)",
                    min_value=10.0,
                    max_value=1200.0,
                    value=160.0,
                    step=1.0,
                    key="sf_ruled_width",
                )
                * u.mm
            )

            f_col = None

            if beam_diameter > grating.beam_diameter.to(u.mm):
                st.warning(
                    "Warning: projected beam diameter is larger than grating ruled width!"
                )

        f_cam = (
            st.number_input(
                "Camera FL (mm)",
                min_value=50.0,
                max_value=3000.0,
                value=730.0,
                step=10.0,
                key="sf_fcam",
            )
            * u.mm
        )

    with st.expander(label="Detector", expanded=False):
        detector: str = st.selectbox(
            "Common detectors",
            [*list(DETECTORS.keys())],
            index=0,
            key="sf_detector",
        )

        customise_detector: bool = st.toggle(
            "Customise detector?", key="sf_custom_detector"
        )

        if customise_detector:
            det_name = st.text_input(
                "Detector name", value=DETECTORS[detector].name, key="sf_det_name"
            )

            px_size = (
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

            n_x = st.number_input(
                "Pixels (x)",
                min_value=256,
                max_value=20000,
                value=DETECTORS[detector].nx,
                step=256,
                key="sf_nx",
            )

            n_y = st.number_input(
                "Pixels (y)",
                min_value=256,
                max_value=20000,
                value=DETECTORS[detector].ny,
                step=256,
                key="sf_ny",
            )

        else:
            det_name = DETECTORS[detector].name
            px_size = DETECTORS[detector].pixel_size.to(u.um)
            n_x = DETECTORS[detector].nx
            n_y = DETECTORS[detector].ny

        det_name_toggle = st.toggle(
            "Show detector name in plot?", key="sf_det_name_toggle"
        )

    with st.expander(label="Plot Format Options", expanded=False):
        fsr_fraction: float = st.slider(
            "FSR fraction to display",
            min_value=0.1,
            max_value=2.0,
            value=1.0,
            step=0.05,
        )

        dispersion_direction: str = st.pills(
            "Dispersion direction in plot",
            options=["Vertical (y)", "Horizontal (x)"],
            default="Vertical (y)",
            key="sf_dispersion_direction",
        )  # type: ignore

        label_orders: bool = st.toggle(
            label="Label diffraction orders?", value=True, key="sf_label_orders"
        )

        label_wavelengths: bool = st.toggle(
            label="Label wavelengths?", value=True, key="sf_label_wavelengths"
        )

        label_atomic_lines: bool = st.toggle(
            label="Label atomic lines?", value=False, key="sf_label_atomic_lines"
        )

        st.divider()

        plot_type: str | list[str] = st.pills(
            label="Plot type",
            selection_mode="multi",
            options=["Plotly (interactive)", "Matplotlib (static)"],
            default=["Plotly (interactive)"],
            key="sf_plot_type",
        )  # type: ignore

    st.button("Recompute", type="primary")


# Create Spectrograph

MIN_WL: ScalarQuantity = min_wl.to(u.nm)
MAX_WL: ScalarQuantity = max_wl.to(u.nm)

grating: EchelleGrating = EchelleGrating(
    groove_frequency=groove_freq,
    blaze_angle=blaze_angle,
    gamma_angle=gamma_angle,
    min_wl=MIN_WL,
    max_wl=MAX_WL,
    ruled_width=grating.ruled_width,
)

if xd_type == "Prism":
    glass: Glass = glasses[glass_name]
    cross_disperser: CrossDisperser = Prism(
        glass=glass,
        apex_angle=apex_angle,
        min_wl=MIN_WL,
        max_wl=MAX_WL,
    )
else:
    cross_disperser: CrossDisperser = SimpleGrating(
        groove_frequency=xd_groove_freq,
        min_wl=MIN_WL,
        max_wl=MAX_WL,
    )

detector: Detector = Detector(
    name=det_name,
    pixel_size=px_size,
    nx=int(n_x),
    ny=int(n_y),
)

spectrograph: EchelleSpectrograph = EchelleSpectrograph(
    beam_diameter=beam_diameter,
    grating=grating,
    cross_disperser=cross_disperser,
    detector=detector,
    min_wl=MIN_WL,
    max_wl=MAX_WL,
    collimator_focal_length=f_col,
    camera_focal_length=f_cam,
    slit_width=slit_width,
    slit_length=slit_length,
    slit_native_focal_ratio=slit_focal_ratio,
)

anamorphism: tuple[float] = spectrograph.anamorphic_magnification()

# Summary
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Slit magnification", f"{spectrograph.slit_magnification:.3f}x")
c2.metric("Slit sampling", f"{spectrograph.slit_sampling * anamorphism:.2f} pixels")
c3.metric("Number of orders", f"{len(spectrograph.spectral_orders)}")
c4.metric(
    "Minimum order separation",
    f"{(min(spectrograph.reddest_order_image_plane_separation, spectrograph.bluest_order_image_plane_separation) / detector.pixel_size).si:.2f} pixels",
)
try:
    R: float = spectrograph.spectral_resolution(wl=0.55 * u.um)
    c5.metric("R = λ/Δλ @ 550 nm", f"{R:,.0f}")
except Exception:
    c5.metric("R = λ/Δλ @ 550 nm", "—")


if "Plotly (interactive)" in plot_type:
    # Plot Echellogram
    def _wl_to_hex(wl_nm: float) -> str:
        r: float
        g: float
        b: float

        r, g, b = wavelength_to_rgb(wl_nm * u.nm)[:3]
        return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"

    with st.spinner("Computing spectral format..."):
        fig = go.Figure()

        # Detector outline
        half_x: float = detector.size[0].to(u.mm).value / 2
        half_y: float = detector.size[1].to(u.mm).value / 2
        fig.add_shape(
            type="rect",
            x0=-half_x,
            y0=-half_y,
            x1=half_x,
            y1=half_y,
            line={"color": "rgba(255,255,255,0.5)", "width": 3, "dash": "dash"},
            fillcolor="rgba(0,0,0,0)",
        )
        if det_name_toggle and det_name:
            fig.add_annotation(
                x=0,
                y=-half_y,
                text=det_name,
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
            color: str = _wl_to_hex(mean_wl)

            positions = np.array(spectrograph.image_plane_position(m=m, wavelength=wls))
            xs = positions[:, 0]
            ys = positions[:, 1]

            if dispersion_direction == "Horizontal (x)":
                xs, ys = -ys, -xs
                xshift: float = 20
                yshift: float = 0
                xlabel = "Dispersion direction (mm)"
                ylabel = "Cross-dispersion direction (mm)"
            elif dispersion_direction == "Vertical (y)":
                xshift: float = 5
                yshift: float = 15
                xlabel = "Cross-dispersion direction (mm)"
                ylabel = "Dispersion direction (mm)"

            hover_wls = wls.to(u.nm).value

            fig.add_trace(
                go.Scatter(
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

            if label_atomic_lines:
                min_FSR_wl, max_FSR_wl = spectrograph.grating.order_FSR_bounds(m=m)
                for name, wl in _atomic_lines.items():
                    if min_FSR_wl <= wl <= max_FSR_wl:
                        x, y = spectrograph.image_plane_position(m=m, wavelength=wl)

                        if dispersion_direction == "Horizontal (x)":
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

        st.plotly_chart(fig, width="content")


if "Matplotlib (static)" in plot_type:
    with st.spinner("Computing spectral format..."):
        fig: plt.Figure = plt.figure(figsize=(10, 10))
        ax: plt.Axes = fig.gca()

        # Detector outline
        half_x: float = detector.size[0].to(u.mm).value / 2
        half_y: float = detector.size[1].to(u.mm).value / 2

        spectrograph.plot_detector_edge(ax=ax, name=True)

        b = max(half_x, half_y) + 2

        ax.set_xlim(b, -b)
        ax.set_ylim(-b, b)

        ax.set_aspect("equal")

        for f in range(1):  # number of fibres
            offset: ScalarQuantity = f * 130 * u.um * spectrograph.slit_magnification
            for m in spectrograph.spectral_orders:
                wls: ArrayQuantity = spectrograph.order_wavelengths(
                    m=m, num=60, FSR_fraction=fsr_fraction
                )
                mean_wl: ScalarQuantity = np.mean(wls.value) * wls.unit

                xs, ys = np.transpose(
                    spectrograph.image_plane_position(m=m, wavelength=wls)
                )

                if dispersion_direction == "Horizontal (x)":
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

                    if dispersion_direction == "Vertical (y)":
                        x_offset = 0
                        y_offset = 1.5
                        ha_m = "center"
                        va_m = "top"
                        ha_wl = "center"
                        va_wl = "bottom"

                    elif dispersion_direction == "Horizontal (x)":
                        x_offset = -1.5
                        y_offset = 0
                        ha_m = "right"
                        va_m = "center"
                        ha_wl = "left"
                        va_wl = "center"

                    ax.text(
                        (xs[0] + offset).value - x_offset,
                        ys[0].value - y_offset,
                        mtext,
                        fontsize=14,
                        ha=ha_m,
                        va=va_m,
                    )

                    ax.text(
                        (xs[-1] + offset).value + x_offset,
                        ys[-1].value + y_offset,
                        wltext,
                        fontsize=14,
                        ha=ha_wl,
                        va=va_wl,
                    )

                if label_atomic_lines:
                    min_FSR_wl, max_FSR_wl = spectrograph.grating.order_FSR_bounds(m=m)
                    for name, wl in _atomic_lines.items():
                        if min_FSR_wl <= wl <= max_FSR_wl:
                            x, y = spectrograph.image_plane_position(m=m, wavelength=wl)

                            if dispersion_direction == "Horizontal (x)":
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
        st.pyplot(fig)
