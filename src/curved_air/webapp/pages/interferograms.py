#! uv run streamlit run

"""
TODO: manual input of actual interferogram data itself in an editable table?
TODO: display as phase from sag import, and vice versa, using wavelength to convert
"""

import hashlib
import io
import logging
from collections.abc import Callable
from itertools import cycle
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from astropy import units as u
from astropy.units import Quantity
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from PIL import Image
from plotly import graph_objects as go
from streamlit.runtime.uploaded_file_manager import UploadedFile

from curved_air.interferograms.convert import (
    from_fisba_csv,
    from_zemax_grid_sag,
    from_zygo,
    to_zemax,
    zemax_units,
)
from curved_air.interferograms.data import (
    GridSagData,
    IntensityData,
    InterferogramData,
    PhaseData,
)
from curved_air.interferograms.fitting import (
    best_fit_cylinder,
    best_fit_plane,
    best_fit_saddle,
    best_fit_sphere,
    best_fit_trefoil,
)
from curved_air.interferograms.streamlit_components import profile_selector
from curved_air.interferograms.utils import (
    flag_edge_zeroes,
    line_profile,
    native_endian_array,
    sag2phase,
    transform,
    x_offset,
    y_offset,
)
from curved_air.interferograms.zernike import ZernikeFitResult, fit_zernike
from curved_air.utils import st_quantity_input
from curved_air.webapp import badges, bmac

st.set_page_config(
    page_title="Interferograms",
    layout="wide",
)


st.title("Interferograms", anchor=False)

with st.expander("About this app"):
    f"""
    In this app you can o load and visualise interferogram data files from Fisba and
    Zygo interferometers, as well as grid files compatible with OpticStudio Zemax.
    The data can be processed in various ways (re-sampling, removal of zero data,
    correction by beft-fit forms, etc.) and converted to a format suitable for import
    into Zemax as either phase or surface sag grid data.

    Use the sidebar to upload a data file and control the scaling, transformation and
    resampling options.

    All code is written by Jake Pember {badges}

    Please get in contact if you have any questions, suggestions, or find any bugs!

    You can also [Buy me a pint](https://buymeacoffee.com/pemberj) if you find this tool
    useful and want to support its development!
    """

    bmac()


if st.session_state.get("file_uploader", None) is None:
    st.info("Please upload a data file in the sidebar to continue.")

with st.sidebar:
    input_mode: str = st.radio(
        "Input data format",
        options=[
            "Auto-detect",
            "Fisba (CSV, .txt)",
            "Zygo (.dat)",
            "Zemax Grid Sag (.dat)",
        ],
        index=0,
        horizontal=True,
    )

    file: UploadedFile | None = st.file_uploader(
        "Upload interferogram data file",
        # type=["txt", "dat"],
        accept_multiple_files=False,
        key="file_uploader",
        help="""Select and upload a file for analysis. Only one file at a time is
        supported""",
    )

    if file is None:
        st.info("Please upload a data file to continue.")
        st.stop()

    file_bytes: bytes = file.getvalue()
    suffix: str = f".{file.name.split('.')[-1].lower()}"
    file_hash: str = hashlib.sha256(file_bytes).hexdigest()
    file_cache_key: tuple[str, int, str, str] = (
        file.name,
        file.size,
        file_hash,
        input_mode,
    )

    if st.session_state.get("input_file_cache_key") != file_cache_key:
        with st.spinner("Processing file..."):
            with NamedTemporaryFile(mode="wb", suffix=suffix, delete=False) as tmp:
                tmp.write(file_bytes)
                temp_file_path: str = tmp.name

            if input_mode == "Auto-detect":
                if suffix == ".txt":
                    loader: Callable = from_fisba_csv
                    loader_name = "fisba"
                    loader_unit = "Å"
                elif suffix == ".dat":
                    loader: Callable = from_zygo
                    loader_name = "zygo"
                    loader_unit = "nm"
                else:
                    st.warning(
                        "Could not auto-detect file type. Please select the correct input format."
                    )
                    st.stop()

            elif input_mode == "Fisba (CSV, .txt)":
                loader: Callable = from_fisba_csv
                loader_name = "fisba"
                loader_unit = "Å"
            elif input_mode == "Zygo (.dat)":
                loader: Callable = from_zygo
                loader_name = "zygo"
                loader_unit = "nm"
            elif input_mode == "Zemax Grid Sag (.dat)":
                loader: Callable = from_zemax_grid_sag
                loader_name = "zemax"
                loader_unit = "mm"

            log_stream = io.StringIO()
            log_handler: logging.StreamHandler = logging.StreamHandler(log_stream)
            log_handler.setLevel(logging.INFO)
            log_handler.setFormatter(logging.Formatter("%(message)s"))
            convert_logger: logging.Logger = logging.getLogger("interferograms.convert")
            previous_level: int = convert_logger.level
            convert_logger.setLevel(logging.INFO)
            convert_logger.addHandler(log_handler)

            try:
                datalist: InterferogramData | list[InterferogramData] | GridSagData = (
                    loader(temp_file_path)
                )
            except Exception as e:
                st.error(f"Error processing file: {e}")
                st.stop()
            finally:
                convert_logger.removeHandler(log_handler)
                log_handler.flush()
                convert_logger.setLevel(previous_level)

            processing_log: str = log_stream.getvalue().strip()
            log_stream.close()

            st.session_state["input_file_cache_key"] = file_cache_key
            st.session_state["cached_datalist"] = datalist
            st.session_state["cached_loader_name"] = loader_name
            st.session_state["cached_loader_unit"] = loader_unit
            st.session_state["cached_processing_log"] = processing_log
    else:
        datalist: InterferogramData | list[InterferogramData] | GridSagData = (
            st.session_state["cached_datalist"]
        )
        loader_name: str = st.session_state["cached_loader_name"]
        loader_unit: str = st.session_state["cached_loader_unit"]
        processing_log: str = st.session_state.get("cached_processing_log", "")

    with st.expander("Import log"):
        if processing_log:
            st.text(processing_log)
        else:
            st.caption("No import logs captured.")

    if not isinstance(datalist, list):
        datalist: list[InterferogramData | GridSagData] = [
            datalist
        ]  # Wrap in a list for uniform processing

    with st.expander("Parsed metadata"):
        assert isinstance(datalist, list)
        for i, d in enumerate(datalist):
            if d.metadata is not None:
                st.subheader(f"Metadata for array {i}", anchor=False)
                st.write(d.metadata.__dict__())


# Jump back out to main canvas to select and initialise the data array
if len(datalist) > 1:
    options: list[str] = []

    for i, d in enumerate(datalist):
        if isinstance(d, IntensityData):
            options.append(f"{i} - Intensity")
        elif isinstance(d, PhaseData):
            options.append(f"{i} - Phase")
        else:
            options.append(f"{i} - Unknown")

    with st.container(horizontal_alignment="center"):  # noqa: SIM117
        with st.container(width=600, horizontal_alignment="center"):
            data_choice: str | None = st.segmented_control(
                "Select data array:",
                options=options,
                default=options[0] if options else None,
            )

            if data_choice is None:
                st.warning("No interferogram selected. Displaying the first one.")
                data_index = 0

            else:
                data_index = int(data_choice.split(" - ")[0])

else:
    data_index = 0

d: InterferogramData | GridSagData = datalist[data_index]  # ty:ignore[invalid-assignment]

data: NDArray = d.data.copy()

with st.sidebar:
    with st.expander("Unit selection", expanded=False):
        wavelength_unit_options: list[str] = ["Å", "pm", "nm", "µm", "mm", "cm", "m"]

        phase_unit_options: list[str] = [
            "phase (waves, cycles or 2π)",
            "phase (radians)",
            "phase (degrees)",
        ]

        unit_options: list[str] = wavelength_unit_options + phase_unit_options

        cols = st.columns([4, 1, 4])
        with cols[0]:
            input_unit: str = st.selectbox(
                "Select input data unit",
                options=unit_options,
                index=unit_options.index(loader_unit)
                if loader_unit in unit_options
                else 0,
                help="What unit are the input data values in?",
            )

        if input_unit in wavelength_unit_options:
            input_unit: u.Unit = u.Unit(input_unit)
        elif input_unit in phase_unit_options:
            st.error("Phase units are not yet implemented")
            st.stop()
        assert isinstance(input_unit, u.Unit), "Input unit is not a valid astropy unit"

        with cols[1]:
            st.container(height=8, border=False)
            st.subheader("→", anchor=False)

        with cols[2]:
            output_unit: str = st.selectbox(
                "Select display data unit",
                options=unit_options,
                index=unit_options.index(loader_unit)
                if loader_unit in unit_options
                else 0,
                help="The unit to convert the data into for display",
            )

        if output_unit in wavelength_unit_options:
            output_unit: u.Unit = u.Unit(output_unit)
        elif output_unit in phase_unit_options:
            st.error("Phase units are not yet implemented")
            st.stop()

        if input_unit == output_unit:
            st.info("No unit conversion will be applied.")
        else:
            st.write(
                "Data will be scaled by a factor of",
                f"{input_unit.to(output_unit):.5g}",
                f"to convert from {input_unit} to {output_unit}.",
            )

    with st.expander("Wavelength settings", expanded=False):
        st.toggle(
            "Force wavelength for phase conversion?",
            key="force_wavelength_toggle",
            value=False,
        )

        st_quantity_input(
            "Input wavelength",
            min_value=0.0,
            value=632.8,
            step=0.1,
            default_unit="nm",
            possible_units=["Å", "nm", "µm", "mm"],
            format="%0.1f",
            disabled=not st.session_state.get("force_wavelength_toggle", False),
            key="input_wavelength",
        )

    with st.expander("Transformations", expanded=True):
        st.toggle(
            "Force pixel size?",
            key="force_pixel_size_toggle",
            value=False,
        )

        if st.session_state.get("force_pixel_size_toggle", False):
            st_quantity_input(
                "Input pixel size",
                min_value=0.0,
                value=1.0,
                step=0.01,
                default_unit="mm",
                possible_units=["µm", "mm"],
                format="%0.3g",
                key="input_pixel_size",
            )

        st.segmented_control(
            "Flip",
            options=["Horizontal", "Vertical", "Both"],
            key="flip_direction",
        )

        cols = st.columns(2)
        with cols[0]:
            rotation_input: str | None = st.segmented_control(
                "Rotation",
                options=["90°", "180°", "270°", "Other"],
                default="Other",
                key="rotation_angle",
            )
        with cols[1]:
            rotation_input_custom: str | None = st.text_input(
                "Enter rotation angle in degrees (0-360)",
                value="0",
                key="rotation_angle_other",
                disabled=st.session_state.get("rotation_angle", None) != "Other",
            )
            if rotation_input_custom is not None:
                rotation_input_custom += "°"

            if rotation_input == "Other":
                rotation_input = rotation_input_custom
            rotation_angle: float | None = (
                float(rotation_input[:-1])
                if rotation_input not in (None, "Other")
                else None
            )

        if rotation_angle is not None and rotation_angle not in (0, 90, 180, 270):
            st.warning(
                f"""You have selected a rotation angle ({rotation_angle:g}°) that is not
                a multiple of 90°. This requires resampling and interpolation of the
                data, which may introduce artifacts. Use with caution."""
            )

        st.container(height=1, border=False)
        st.toggle("Crop to valid data?", key="crop_toggle", value=False)

        cols = st.columns(2)
        with cols[0]:
            st.container(height=10, border=False)
            st.toggle("Resample?", key="resample_toggle", value=False)
        with cols[1]:
            st.number_input(
                "Resample size on shortest axis (pixels)",
                min_value=1,
                value=200,
                step=1,
                key="resample_size",
                disabled=not st.session_state.get("resample_toggle", False),
            )

        cols = st.columns([1, 3])
        with cols[0]:
            st.container(height=10, border=False)
            st.toggle("X offset", key="x_offset_toggle", value=False)
            with cols[1]:
                st_quantity_input(
                    "X Offset",
                    value=0.0,
                    step=0.1,
                    possible_units=["µm", "mm"],
                    default_unit="mm",
                    format="%0.1f",
                    disabled=not st.session_state.get("x_offset_toggle", False),
                    key="x_offset",
                )

        cols = st.columns([1, 3])
        with cols[0]:
            st.container(height=10, border=False)
            st.toggle("Y offset", key="y_offset_toggle", value=False)
        with cols[1]:
            st_quantity_input(
                "Y Offset",
                value=0.0,
                step=0.1,
                possible_units=["µm", "mm"],
                default_unit="mm",
                format="%0.1f",
                disabled=not st.session_state.get("y_offset_toggle", False),
                key="y_offset",
            )

    with st.expander("Corrections", expanded=True):
        st.pills(
            "Corrections to apply",
            options=[
                "Flag zeroes",
                "Subtract piston",
                "Subtract piston (q99)",
                "Subtract tilt",
                "Subtract best-fit sphere",
                "Subtract best-fit cylinder",
                "Subtract best-fit saddle",
                "Subtract best-fit trefoil",
                "Reverse phase sign",
            ],
            selection_mode="multi",
            # default=["Flag zeroes"],
            key="corrections",
            help="""Note: All of these corrections are applied before any Zernike
            fitting! You can also subtract individual Zernike terms in the Zernike
            table at the right-hand side of the interface.""",
        )

        cols = st.columns(2)
        with cols[0]:
            st.container(height=10, border=False)
            st.toggle("Multiply by custom factor", key="multiply_toggle", value=False)
        if st.session_state.get("multiply_toggle", False):
            with cols[1]:
                st.number_input(
                    "Factor to multiply",
                    min_value=0.0,
                    value=1.0,
                    step=0.1,
                    format="%0.3g",
                    key="multiply_custom_factor",
                )
            st.session_state["corrections"].append(
                f"Multiply by {st.session_state.get('multiply_custom_factor', 1.0)}"
            )

        cols = st.columns(2)
        with cols[0]:
            st.container(height=10, border=False)
            st.toggle("Divide by custom factor", key="divide_toggle", value=False)
        if st.session_state.get("divide_toggle", False):
            with cols[1]:
                st.number_input(
                    "Factor to divide",
                    min_value=0.0,
                    value=1.0,
                    step=0.1,
                    format="%0.3g",
                    key="divide_custom_factor",
                )
            st.session_state["corrections"].append(
                f"Divide by {st.session_state.get('divide_custom_factor', 1.0)}"
            )

# Apply corrections if requested
if "Flag zeroes" in st.session_state.get("corrections", []):
    data: NDArray = flag_edge_zeroes(data)

if "Subtract piston" in st.session_state.get("corrections", []):
    data -= np.nanmedian(data)
if "Subtract piston (q99)" in st.session_state.get("corrections", []):
    q01: float = np.nanquantile(data, 0.01)
    q99: float = np.nanquantile(data, 0.99)
    temp_data: NDArray = data.copy()
    temp_data[data < q01] = np.nan
    temp_data[data > q99] = np.nan
    data -= np.nanmedian(temp_data)
    del temp_data

if "Subtract tilt" in st.session_state.get("corrections", []):
    _plane, residual, _fit_params = best_fit_plane(data)
    data: NDArray = residual
if "Subtract best-fit sphere" in st.session_state.get("corrections", []):
    _sphere, residual, _fit_params = best_fit_sphere(data)
    data: NDArray = residual
if "Subtract best-fit cylinder" in st.session_state.get("corrections", []):
    _cylinder, residual, _fit_params = best_fit_cylinder(data)
    data: NDArray = residual
if "Subtract best-fit saddle" in st.session_state.get("corrections", []):
    _saddle, residual, _fit_params = best_fit_saddle(data)
    data: NDArray = residual
if "Subtract best-fit trefoil" in st.session_state.get("corrections", []):
    _trefoil, residual, _fit_params = best_fit_trefoil(data)
    data: NDArray = residual

if "Subtract piston" in st.session_state.get("corrections", []):
    data -= np.nanmedian(data)
if "Subtract piston (q99)" in st.session_state.get("corrections", []):
    q01: float = np.nanquantile(data, 0.01)
    q99: float = np.nanquantile(data, 0.99)
    temp_data: NDArray = data.copy()
    temp_data[data < q01] = np.nan
    temp_data[data > q99] = np.nan
    data -= np.nanmedian(temp_data)
    del temp_data

if "Reverse phase sign" in st.session_state.get("corrections", []):
    data *= -1

for correction in st.session_state.get("corrections", []):
    if correction.startswith("Multiply by"):
        factor = float(correction.split()[-1])
        data *= factor
    elif correction.startswith("Divide by"):
        factor = float(correction.split()[-1])
        data /= factor

# First, apply any flips, rotations, and cropping - this is so the resampling results in
# the final data having the correct size
data: NDArray = transform(
    data,
    flip_direction=st.session_state.get("flip_direction", None),
    rotation_angle=rotation_angle,
    crop_to_valid=st.session_state.get("crop_toggle", False),
)

# Next, resample the data
if st.session_state.get("resample_toggle", False):
    resample_size: int | None = st.session_state.get("resample_size", None)
    total_size: tuple[int, int] = (data.shape[1], data.shape[0])
    min_size: int = min(total_size)

    if resample_size is not None:
        resample_factor: float = resample_size / min_size
        resample_shape: tuple[int, int] = (
            max(1, round(data.shape[1] * resample_factor)),
            max(1, round(data.shape[0] * resample_factor)),
        )
    else:
        resample_factor = 1
        resample_shape: tuple[float, float] = total_size

    if resample_size is not None and resample_size > min_size:
        st.warning(
            f"""Resample size ({resample_size}) is larger than the shortest axis
            ({min_size}). Data will be upsampled."""
        )
else:
    resample_shape = None
    resample_factor: float = 1

# Now apply the resampling if requested
data: NDArray = transform(data, resample_size=resample_shape)

data: NDArray = native_endian_array(data)

if st.session_state.get("force_pixel_size_toggle", False):
    pixel_size: Quantity = st.session_state.get("input_pixel_size", None)
else:
    pixel_size: float | None = d.metadata.pixel_size

if isinstance(pixel_size, int | float):
    pixel_size: Quantity = pixel_size * u.mm

if pixel_size is None:
    st.warning(
        """Pixel size is not available in the metadata. Please provide a pixel size
        using the 'Force pixel size' option in the sidebar."""
    )
    st.stop()

assert pixel_size is not None

if st.session_state.get("x_offset", 0.0 * u.mm).value:
    data: NDArray = x_offset(
        data,
        offset=int(
            (
                st.session_state.get("x_offset", 0.0 * u.mm).value
                * pixel_size.to(u.mm)
                / resample_factor
            ).value
        ),
    )

if st.session_state.get("y_offset", 0.0 * u.mm).value:
    data: NDArray = y_offset(
        data,
        offset=int(
            (
                st.session_state.get("y_offset", 0.0 * u.mm).value
                * pixel_size.to(u.mm)
                / resample_factor
            ).value
        ),
    )

image_col, zernike_col = st.columns([2, 1])

with zernike_col:
    st.subheader("Zernike decomposition", anchor=False)

    zernike_tab_max_term: int = st.number_input(
        "Number of Zernike terms to fit",
        min_value=0,
        max_value=231,
        value=0,
        step=1,
        key="zernike_tab_max_term",
    )

    if zernike_tab_max_term == 0:
        st.info(
            "To enable Zernike decomposition, please select a number of terms to fit."
        )

    else:
        try:
            zernike_result: ZernikeFitResult = fit_zernike(
                data, max_term=zernike_tab_max_term
            )
        except (AssertionError, ValueError) as e:
            st.warning(f"Could not fit Zernike terms: {e}")
        else:
            metric_cols = cycle(st.columns(4))
            with next(metric_cols):
                st.metric("Terms fit", value=f"{zernike_tab_max_term}", border=True)
            with next(metric_cols):
                st.metric(
                    "Residual RMS",
                    value=f"{(zernike_result.rms_residual * input_unit).to(output_unit):.4g}",
                    border=True,
                )
            with next(metric_cols):
                st.metric(
                    "Residual PTV",
                    value=f"{(zernike_result.ptv_residual * input_unit).to(output_unit):.4g}",
                    border=True,
                )
            with next(metric_cols):
                st.metric(
                    "Fit aperture radius",
                    value=f"{zernike_result.radius:.4g} px",
                    border=True,
                )

            # st.pyplot(plot_zernike_weights(zernike_result))

            zernike_table: pd.DataFrame = pd.DataFrame(zernike_result.table())
            zernike_table["coefficient"] = (
                (zernike_table["coefficient"].to_numpy() * input_unit)
                .to(output_unit)
                .value
            )
            zernike_table["weight"] *= 100
            zernike_table: pd.DataFrame = zernike_table.rename(
                columns={
                    "term": "Term (Noll)",
                    "name": "Name",
                    "coefficient": f"Coefficient [{output_unit}]",
                    "weight": "Weight [%]",
                }
            )

            # Add a column with a checkbox for each term to allow the user to select
            # which terms to subtract
            zernike_table["Subtract"] = False

            # Colour-code the Weight column on a smooth green (low) to red (high) scale.
            # Note that pandas.Styler formatting is only applied to non-editable columns
            # in st.data_editor, which is why "Weight [%]" needs to stay in the
            # `disabled` list below.
            zernike_styler = zernike_table.style.background_gradient(
                cmap="RdYlGn_r",
                subset=["Weight [%]"],
                low=0.35,
                high=0.35,
            )

            # st.dataframe(zernike_table, width="stretch", hide_index=True)
            zernike_table: pd.DataFrame = st.data_editor(
                zernike_styler,
                disabled=[
                    "Term (Noll)",
                    "Name",
                    f"Coefficient [{output_unit}]",
                    "Weight [%]",
                ],
                width="stretch",
                height=600,
                hide_index=True,
            )

            zernike_terms_to_subtract: list[int] = zernike_table.loc[
                zernike_table["Subtract"] == True, "Term (Noll)"  # noqa: E712
            ].tolist()

            _, zernike_residual = zernike_result.remove(zernike_terms_to_subtract)

            data: NDArray = zernike_residual

with image_col:
    tab3d, tab2d, tab1d, tabdata = st.tabs(
        [
            "3D Surface Plot",
            "2D Surface Plot",
            "1D Profile Plot",
            "Convert & Download",
        ],
        default=st.session_state.get("tabs", "2D Surface Plot"),
        on_change="rerun",
        key="tabs",
    )

    with tab3d, st.container(horizontal_alignment="center"):
        if tab3d.open:
            x: NDArray = (np.arange(data.shape[1]) * pixel_size / resample_factor).to(
                u.mm
            )
            y: NDArray = (np.arange(data.shape[0]) * pixel_size / resample_factor).to(
                u.mm
            )
            z: NDArray = data * input_unit.to(output_unit)

            fig = go.Figure(
                go.Surface(
                    x=x,
                    y=y,
                    z=z,
                    colorscale=st.session_state.get("colormap_3d", "jet"),
                ),
            )

            x_span: float = (x.max() - x.min()).to(u.mm).value
            y_span: float = (y.max() - y.min()).to(u.mm).value
            half_range: float = max(x_span, y_span) / 2

            x_center: float = ((x.max() + x.min()) / 2).to(u.mm).value
            y_center: float = ((y.max() + y.min()) / 2).to(u.mm).value

            if zernike_tab_max_term > 0 and zernike_terms_to_subtract:
                fig_title: str = (
                    "Interferogram data with Zernike terms subtracted: "
                    + ", ".join([f"Z{t}" for t in zernike_terms_to_subtract])
                )
            else:
                fig_title: str = "Interferogram data"

            fig.update_layout(
                title=fig_title,
                coloraxis_showscale=True,
                xaxis_title="X [mm]",
                yaxis_title="Y [mm]",
                scene={
                    "xaxis": {
                        "title": "X [mm]",
                        "range": [x_center - half_range, x_center + half_range],
                    },
                    "yaxis": {
                        "title": "Y [mm]",
                        "range": [y_center - half_range, y_center + half_range],
                    },
                    "zaxis": {
                        "title": f"Z [{output_unit}]",
                        "range": [np.min(z), np.max(z)],
                    },
                },
            )

            with st.skeleton(height=800):
                st.plotly_chart(fig, height=800)

            colormaps: list[str] = px.colors.named_colorscales()

            colormap: str = st.selectbox(
                "Select colormap",
                options=colormaps,
                index=colormaps.index("jet") if "jet" in colormaps else 0,
                width=300,
                key="colormap_3d",
            )

    with tab2d, st.container(horizontal_alignment="center"):
        if tab2d.open:
            fig = px.imshow(
                data * input_unit.to(output_unit),
                x=(np.arange(data.shape[1]) * pixel_size / resample_factor).to(u.mm),
                y=(np.arange(data.shape[0]) * pixel_size / resample_factor).to(u.mm),
                color_continuous_scale=st.session_state.get("colormap_2d", "jet"),
                # color_continuous_midpoint=0,
            )

            if zernike_tab_max_term > 0 and zernike_terms_to_subtract:
                fig_title = (
                    "Interferogram data with Zernike terms subtracted: "
                    + ", ".join([f"Z{t}" for t in zernike_terms_to_subtract])
                )
            else:
                fig_title = "Interferogram data"

            fig.update_layout(
                title=fig_title,
                coloraxis_showscale=True,
                xaxis_title="X [mm]",
                yaxis_title="Y [mm]",
            )

            with st.skeleton(height=800):
                st.plotly_chart(fig, height=800)

            colormaps: list[str] = px.colors.named_colorscales()

            colormap: str = st.selectbox(
                "Select colormap",
                options=colormaps,
                index=colormaps.index("jet") if "jet" in colormaps else 0,
                width=300,
                key="colormap_2d",
            )

    with tab1d:
        if tab1d.open:
            st.subheader("Profile", anchor=False)

            profile_selector_key: str = "profile_selector"
            component_state: dict = st.session_state.get(profile_selector_key, {})
            current_line: dict | None = component_state.get("line")

            status_col, clear_col = st.columns([4, 1])
            with status_col:
                if current_line is None:
                    st.caption(
                        """Click and drag on the image below to draw a line between two
                        points, and optionally scroll over it to set the profile width.
                        The data along that line will be plotted below, with a median
                        filter over the width."""
                    )
                else:
                    st.caption(
                        """Line selected below. Drag the endpoints to adjust it, drag
                        elsewhere to start a new line, or scroll over it to change its
                        width."""
                    )
            with clear_col:
                if st.button("Clear selection", key="profile_clear"):
                    st.session_state.setdefault(profile_selector_key, {})["line"] = None
                    current_line = None

            step_mm: float = (pixel_size / resample_factor).to(u.mm).value

            # Make image for the interface
            vmin: float = float(np.nanmin(data))
            vmax: float = float(np.nanmax(data))
            value_range: float = vmax - vmin if vmax > vmin else 1.0
            normalized: NDArray = (data - vmin) / value_range

            # Grab the same colormap as the 2d display
            cmap = plt.get_cmap(st.session_state.get("colormap_2d", "jet"))

            rgba: NDArray = (cmap(np.ma.masked_invalid(normalized)) * 255).astype(
                np.uint8
            )
            image: Image.Image = Image.fromarray(rgba, mode="RGBA")

            # Reset a stale selection if the underlying data shape changed.
            dims_key: str = f"{profile_selector_key}_dims"
            current_dims = (image.width, image.height)
            if st.session_state.get(dims_key) != current_dims:
                st.session_state.setdefault(profile_selector_key, {})["line"] = None
                current_line = None
            st.session_state[dims_key] = current_dims

            with st.container(horizontal_alignment="center"):  # noqa: SIM117
                with st.container(width=600):
                    result: dict[str, Any] = profile_selector(
                        image,
                        key=profile_selector_key,
                    )
            line: dict[str, float] = result.line
            profile_half_width: int = result.half_width or 0

            if line is not None:
                col_a, row_a = line["x1"], line["y1"]
                col_b, row_b = line["x2"], line["y2"]

                distance_px, values = line_profile(
                    data,
                    start=(row_a, col_a),
                    end=(row_b, col_b),
                    half_width=profile_half_width,
                )
                distance_mm: NDArray = distance_px * step_mm
                values_converted: NDArray = values * input_unit.to(output_unit)

                x_a_mm, y_a_mm = col_a * step_mm, row_a * step_mm
                x_b_mm, y_b_mm = col_b * step_mm, row_b * step_mm
                A: str = f"({x_a_mm:.3g}, {y_a_mm:.3g}) mm"
                B: str = f"({x_b_mm:.3g}, {y_b_mm:.3g}) mm"

                fig_profile = go.Figure()
                fig_profile.add_trace(
                    go.Scatter(
                        x=distance_mm,
                        y=values_converted,
                        mode="lines",
                        name="Profile",
                    )
                )

                fig_profile.update_layout(
                    title=f"Profile from {A} to {B} with width {profile_half_width * 2 + 1} px",
                    xaxis_title="Distance along line [mm]",
                    yaxis_title=f"Value [{output_unit}]",
                )

                with st.skeleton(height=400):
                    st.plotly_chart(fig_profile, height=400)

            else:  # line is None
                st.info("Draw a line on the image above to see its cross section.")

    with tabdata:
        ny, nx = np.shape(data)
        pixel_size: float = pixel_size / resample_factor
        data_size_x: float = pixel_size * nx
        data_size_y: float = pixel_size * ny

        if tabdata.open:
            st.subheader("Zemax", anchor=False)

            st.segmented_control(
                "Which unit should be used for Zemax export?",
                options=list(zemax_units.keys()),
                default="mm",
                key="zemax_unit",
            )

            scale_factor: float = input_unit.to(u.mm)

            st.write(
                f"""Data will be scaled by a factor of {scale_factor:.5g} to convert
                from {input_unit} to mm for Zemax export."""
            )

            st.download_button(
                "Export to Zemax grid sag",
                data=to_zemax(
                    data=data * scale_factor,
                    file=None,
                    size=(data_size_x, data_size_y),
                    passed_unit="mm",
                    requested_unit=st.session_state.get("zemax_unit", "mm"),
                ),
                file_name=f"{file.name}_sag.dat",
                mime="text/plain",
                type="primary",
                key=f"export_{file.name}_sag",
            )

            if st.session_state.get("force_wavelength_toggle", False):
                wavelength_to_use: Quantity | None = st.session_state.get(
                    "input_wavelength"
                )
                assert wavelength_to_use is not None, "Input wavelength is None"
            elif d.metadata.wavelength is not None:
                wavelength_to_use: float = d.metadata.wavelength
            else:
                wavelength_to_use: Quantity = 632.8 * u.nm

            st.download_button(
                "Export to Zemax grid phase",
                data=to_zemax(
                    data=sag2phase(
                        sag_data=data,
                        sag_unit=input_unit,
                        wavelength=wavelength_to_use,
                    ),
                    file=None,
                    size=(data_size_x, data_size_y),
                ),
                file_name=f"{file.name}_phase.dat",
                mime="text/plain",
                type="primary",
                key=f"export_{file.name}_phase",
            )

            with st.expander("Raw Data", expanded=False):
                st.dataframe(pd.DataFrame(data), width="stretch", hide_index=True)

            """Do you have a file format that is not yet supported?
            Please [get in contact](mailto:jacob.pember@ikmail.com); I would love to add
            support for it. So far this tool is only for my own use, and only supports
            files that I have needed to work with so far."""

st.divider()

heading_col, unit_select_col, *_ = st.columns(4)

with heading_col:
    st.subheader("Data statistics", anchor=False)

with unit_select_col:  # noqa: SIM117
    with st.container():
        stats_unit = st.selectbox(
            "Select unit for statistics",
            options=unit_options,
            index=unit_options.index(loader_unit) if loader_unit in unit_options else 0,
            key="stats_unit",
        )

cols = st.columns(4)
column = cycle(cols)

min_data: float = np.nanmin(data)
max_data: float = np.nanmax(data)

with next(column):
    st.metric(
        label="Maximum",
        value=f"{(max_data * input_unit).to(stats_unit):.4g}",
        border=True,
    )

with next(column):
    st.metric(
        label="Minimum",
        value=f"{(min_data * input_unit).to(stats_unit):.4g}",
        border=True,
    )

with next(column):
    st.metric(
        "Peak-to-valley",
        value=f"{((max_data - min_data) * input_unit).to(stats_unit):.4g}",
        border=True,
    )

with next(column):
    st.metric(
        label="RMS",
        value=f"{(np.nanstd(data) * input_unit).to(stats_unit):.4g}",
        border=True,
    )

with next(column):
    st.metric(
        label="Mean",
        value=f"{(np.nanmean(data) * input_unit).to(stats_unit):.4g}",
        border=True,
    )

with next(column):
    st.metric(
        label="Median",
        value=f"{(np.nanmedian(data) * input_unit).to(stats_unit):.4g}",
        border=True,
    )

q99: NDArray = np.nanpercentile(data, 99)
q01: NDArray = np.nanpercentile(data, 1)

with next(column):
    st.metric(
        "PTVq (99%)",
        value=f"{((q99 - q01) * input_unit).to(stats_unit):.4g}",
        border=True,
        help="Peak-to-valley value after removing the top and bottom 1% of data points.",
    )

with next(column):
    st.metric(
        "RMSq (99%)",
        value=f"{
            (np.nanstd(data[(data >= q01) & (data <= q99)]) * input_unit).to(
                stats_unit
            ):.4g}",
        border=True,
        help="RMS value after removing the top and bottom 1% of data points.",
    )

valid_mask: NDArray = ~np.isnan(data)
valid_cols: int = int(np.count_nonzero(np.any(valid_mask, axis=0)))
valid_rows: int = int(np.count_nonzero(np.any(valid_mask, axis=1)))

aperture_size_x_valid: Quantity = data_size_x * (valid_cols / data.shape[1])
aperture_size_y_valid: Quantity = data_size_y * (valid_rows / data.shape[0])

with next(column):
    st.metric(
        "Aperture size (x)",
        value=f"{aperture_size_x_valid.to(u.mm):.4g}",
        border=True,
    )

with next(column):
    st.metric(
        "Aperture size (y)",
        value=f"{aperture_size_y_valid.to(u.mm):.4g}",
        border=True,
    )
