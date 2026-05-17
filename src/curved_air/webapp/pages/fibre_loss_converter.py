#! uv run streamlit run
import io
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from astropy import units as u
from astropy.units import Quantity
from numpy.typing import NDArray
from streamlit.components.v1 import html

from curved_air.data import (
    example_fibre_attenuation_dBkm,
    example_fibre_transmission_relative,
)
from curved_air.type_hinting import ArrayQuantity


def relative_to_dBkm(
    transmission: float | list[float] | NDArray,
    length: Quantity,
) -> Quantity:
    total_loss: Quantity = -10.0 * np.log10(transmission) * u.dB
    return (total_loss / length).to(u.dB / u.km)


def percent_to_dBkm(
    transmission_percent: float | list[float] | NDArray,
    length: Quantity,
) -> Quantity:

    if isinstance(transmission_percent, list):
        transmission_percent: NDArray = np.array(transmission_percent)

    transmission_fraction: float | NDArray = transmission_percent / 100.0
    return relative_to_dBkm(transmission_fraction, length)


def dBkm_to_relative(attenuation: Quantity, length: Quantity) -> float:
    total_loss = (attenuation * length.to(u.km)).to(u.dB)
    return 10 ** (-total_loss.value / 10.0)


def dBkm_to_percent(attenuation: Quantity, length: Quantity) -> float:
    transmission_fraction = dBkm_to_relative(attenuation, length)
    return 100 * transmission_fraction


st.set_page_config(
    page_title="Curved Air: Optical Design Tools",
    page_icon="🌈",
    layout="centered",  # This app doesn't want to be wide
    initial_sidebar_state="expanded",
)

st.header("Fibre Loss Converter")

with st.expander("About this tool", expanded=False):
    st.markdown(
        """
        A simple calculator to convert between fibre attenuation (typically specified in
        dB/km) and fraction transmission over a chosen fibre length.

        The conversion uses the following relationships:
        - Total loss in dB: $L_{dB} = \\alpha_{dB/km} \\cdot \\ell_{km}$
        - Throughput fraction: $T = 10^{-L_{dB}/10}$

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

st.divider()

with st.container(horizontal_alignment="center"):
    mode: str | None = st.segmented_control(
        "Mode",
        ["dB/km -> Relative Transmission", "Relative Transmission -> dB/km"],
        default="dB/km -> Relative Transmission",
    )

if mode is None:
    st.warning("Please select a calculation mode.")
    st.stop()

# Common inputs
cols = st.columns([1, 2])
with cols[0]:
    length_value: float = st.number_input(
        "Fibre length",
        min_value=0.0,
        value=50.0,
        step=1.0,
        width=200,
    )
with cols[1]:
    length_unit: str | None = st.segmented_control(
        "Unit",
        ["cm", "m", "km"],
        default="m",
    )
if length_unit is None:
    st.warning("Please select a unit for fibre length.")
    st.stop()

length: Quantity = length_value * u.Unit(length_unit)

st.divider()

# Create tabs for single-value and multi-value modes
tab1, tab2 = st.tabs(["Single Value", "Multi-Value"])

# ============================================================================
# SINGLE VALUE MODE
# ============================================================================
with tab1:
    if mode == "Relative Transmission -> dB/km":
        cols = st.columns([1, 2])

        with cols[0]:
            transmission_mode: str | None = st.segmented_control(
                "Interpret transmission value as",
                options=["Fraction", "Percent"],
                default="Fraction",
            )

        with cols[1]:
            if transmission_mode == "Percent":
                transmission: float = st.number_input(
                    "Relative Transmission (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=96.6,
                    step=0.1,
                    width=200,
                )
                transmission: float = transmission / 100.0
            else:
                transmission: float = st.number_input(
                    "Relative Transmission",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.966,
                    step=0.0001,
                    width=200,
                )

        total_loss: Quantity = -10.0 * math.log10(transmission) * u.dB
        attenuation: Quantity = total_loss / length.to(u.km)

        st.divider()

        cols = st.columns(2)

        with cols[0]:
            st.metric("Total attenuation over length", f"{total_loss:.3f}")
        with cols[1]:
            st.metric("Equivalent attenuation", f"{attenuation:.3f}")

    else:
        attenuation = (
            st.number_input(
                "Attenuation (dB/km)",
                min_value=0.0,
                value=3.0,
                step=0.1,
                width=200,
            )
            * u.dB
            / u.km
        )

        total_loss: Quantity = (attenuation * length.to(u.km)).to(u.dB)
        transmission_fraction: float = 10 ** (-total_loss.value / 10.0)
        transmission_percent: float = 100 * transmission_fraction

        st.divider()

        cols = st.columns(3)

        with cols[0]:
            st.metric("Total attenuation over length", f"{total_loss:.3f}")
        with cols[1]:
            st.metric("Relative Transmission", f"{transmission_fraction:.6f}")
        with cols[2]:
            st.metric("Relative Transmission", f"{transmission_percent:.3f}%")

# ============================================================================
# MULTI-VALUE MODE
# ============================================================================
with tab2:
    st.subheader("Paste CSV Data")
    with st.expander("Data examples", expanded=False):
        st.markdown(
            """
            Paste CSV data with columns:
            - `wavelength` (optional): Only used for plotting
            - `attenuation` or `transmission`: Values to convert

            **Example with wavelength values:**
            ```
            500,2.5
            600,3.0
            700,3.2
            ```

            **Example without wavelength values:**
            ```
            2.5
            3.0
            3.2
            ```
            """
        )

    if mode == "dB/km -> Relative Transmission":
        default_data = example_fibre_attenuation_dBkm

    elif mode == "Relative Transmission -> dB/km":
        default_data = example_fibre_transmission_relative

    csv_data = st.text_area(
        "Paste CSV data",
        height=200,
        value=default_data,
    )

    if csv_data.strip():
        try:
            # Parse CSV data
            df = pd.read_csv(io.StringIO(csv_data), header=None)
        except Exception as e:
            st.error(f"Error parsing CSV data: {e}")
            st.info(
                "Please ensure your CSV is properly formatted with headers and numeric values."
            )

        # Determine if dataframe has one or two columns
        if len(df.columns) == 1:
            wavelength = None
            attenuation = pd.to_numeric(df.iloc[:, 0]).values

        elif len(df.columns) == 2:  # noqa: PLR2004
            wavelength = pd.to_numeric(df.iloc[:, 0]).values
            attenuation = pd.to_numeric(df.iloc[:, 1]).values

        else:
            st.error(
                """Invalid CSV format. Please provide either one column (attenuation
                or transmission) or two columns (wavelength and attenuation/
                transmission)."""
            )
            st.stop()

        fig: go.Figure = go.Figure()

        if mode == "dB/km -> Relative Transmission":
            attenuation: ArrayQuantity = attenuation * u.dB / u.km
            y_axis = dBkm_to_relative(attenuation, length=length)
            y_label = "Relative Transmission"
            y_label_csv = "relative_transmission"

            if wavelength is None:
                x_axis = range(len(df))
                x_label = ""
            else:
                x_axis = wavelength
                x_label = "Wavelength (nm)"

        else:
            assert mode == "Relative Transmission -> dB/km"
            y_axis = relative_to_dBkm(attenuation, length=length)
            y_label = "Attenuation (dB/km)"
            y_label_csv = "attenutation"

            if wavelength is None:
                x_axis = range(len(df))
                x_label = ""
            else:
                x_axis = wavelength
                x_label = "Wavelength (nm)"

        fig.add_trace(
            trace=go.Scatter(
                x=x_axis,
                y=y_axis,
                mode="lines",
            )
        )
        fig.update_layout(
            title="",
            xaxis_title=x_label,
            yaxis_title=y_label,
            template="plotly_dark",
        )
        st.plotly_chart(fig, width="stretch")

        # Display converted values in text format
        st.subheader("Converted Values")
        if wavelength is not None:
            result_df = pd.DataFrame(
                {
                    "wavelength": wavelength,
                    y_label_csv: y_axis,
                }
            )
        else:
            result_df = pd.DataFrame({y_label: y_axis})

        st.dataframe(result_df)
