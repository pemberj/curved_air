#! uv run streamlit run
import math

import streamlit as st
from astropy import units as u
from astropy.units import Quantity

st.set_page_config(
    page_title="Fibre Loss Converter",
    page_icon="📉",
    layout="centered",
)

st.header("Fibre Loss Converter")

with st.expander("About this calculator", expanded=True):
    st.markdown(
        """
        A simple calculator to convert between fibre attenuation (typically specified in
        dB/km) and fraction transmission over a chosen fibre length.

        The conversion uses the following relationships:
        - Total loss in dB: $L_{dB} = \\alpha_{dB/km} \\cdot \\ell_{km}$
        - Throughput fraction: $T = 10^{-L_{dB}/10}$
        """
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
            disabled=st.session_state.get("upload_db_per_km", False),
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
