#! uv run streamlit run

"""
curved_air

webapp

coatings


"""

from __future__ import annotations

import streamlit as st
from astropy import units as u
from plotly import graph_objects as go

from curved_air.coatings import EDMUND_OPTICS_AR, THORLABS_AR
from curved_air.utils import st_quantity_input
from curved_air.webapp import badges, bmac

st.set_page_config(
    page_title="Coatings",
    layout="wide",
)

st.header("Coatings", anchor=False)

with st.expander("About this tool", expanded=False):
    f"""
    Use this tool to visualise and compare various optical coatings. A number of
    standard commercial coatings available from Edmund Optics and Thorlabs are included,
    or you can input your own data.

    All code is written by Jake Pember {badges}

    Please get in contact if you have any questions, suggestions, or find any bugs!

    You can also [Buy me a pint](https://buymeacoffee.com/pemberj) if you find this tool
    useful and want to support its development!
    """

    bmac()

ar_tab, hr_tab = st.tabs(["AR Coatings", "HR Coatings"])

with ar_tab:
    st.subheader("Anti-Reflection (AR) Coatings")

    select_col, plot_col = st.columns([2, 3])

    with select_col:
        # TODO: Use pills here instead of multiselect
        st.multiselect(
            "Select AR Coatings to Compare",
            options=EDMUND_OPTICS_AR + THORLABS_AR,
            format_func=lambda coating: f"{coating.name}",
            wrap=True,
            key="selected_ar_coatings",
        )

        with st.container(horizontal_alignment="center"):  # noqa: SIM117
            with st.container(width=500):
                st_quantity_input(
                    "Minimum Wavelength",
                    value=380.0,
                    min_value=0.0,
                    step=10.0,
                    possible_units=["Å", "nm", "µm"],
                    default_unit="nm",
                    key="min_wavelength",
                )

                st_quantity_input(
                    "Maximum Wavelength",
                    value=900.0,
                    min_value=0.0,
                    step=10.0,
                    possible_units=["Å", "nm", "µm"],
                    default_unit="nm",
                    key="max_wavelength",
                )

    with plot_col:
        if st.session_state["selected_ar_coatings"] is not None:
            if not isinstance(st.session_state["selected_ar_coatings"], list):
                st.session_state["selected_ar_coatings"] = [
                    st.session_state["selected_ar_coatings"]
                ]

            # Make a plotly plot
            fig = go.Figure()

            max_y_in_range: float = 0.0

            for coating in st.session_state["selected_ar_coatings"]:
                range_mask = (
                    coating.wavelengths >= st.session_state["min_wavelength"]
                ) & (coating.wavelengths <= st.session_state["max_wavelength"])

                try:
                    candidate_max_y: float = max(coating.reflectivity[range_mask])
                except ValueError:
                    continue

                if candidate_max_y > max_y_in_range:
                    max_y_in_range: float = candidate_max_y

                fig.add_trace(
                    trace=go.Scatter(
                        x=coating.wavelengths.to(u.nm),
                        y=coating.reflectivity,
                        mode="lines",
                        hovertemplate=f"{coating.name.split('(')[0]}<br />"
                        + "%{y:.4g}<extra></extra>",
                        name=f"{coating.name}",
                    )
                )

            fig.update_layout(
                title="",
                xaxis_title="Wavelength (nm)",
                xaxis_range=[
                    st.session_state["min_wavelength"].to(u.nm).value,
                    st.session_state["max_wavelength"].to(u.nm).value,
                ],
                yaxis_title="Reflectivity",
                yaxis_range=[0, max_y_in_range * 1.1],
                template="plotly_dark",
                height=800,
                hovermode="x",
            )

            fig.update_xaxes(
                showspikes=True,
                spikecolor="white",
                spikesnap="cursor",
                spikemode="across",
            )

            st.plotly_chart(
                fig,
                width="stretch",
                on_select="rerun",
                selection_mode="points",
                key="ar_coating_plot",
            )


with hr_tab:
    st.subheader("High-Reflection (HR) Coatings")
