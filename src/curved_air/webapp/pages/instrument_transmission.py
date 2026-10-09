#! uv run streamlit run
from itertools import cycle
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import streamlit as st
from astropy import units as u
from matplotlib import pyplot as plt

from curved_air.optics.dispersers import EchelleGrating
from curved_air.plotting import plot_style
from curved_air.transmission import (
    MARVEL_TRANSMISSION,
    FileTransmission,
    FlatTransmission,
    Transmission,
    ZemaxMultiTransmission,
    ZemaxTransmission,
    total_transmission,
    transmission_plot,
)
from curved_air.webapp import badges, bmac

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray

plt.style.use(plot_style)

DATA_PATH: Path = Path(__file__).parent / ".." / ".." / "transmission_data"
MIRROR_REFLECTIVITY_FILES: list[Path] = list(DATA_PATH.glob("ASA*.csv"))
ZEMAX_SINGLE_CONFIG_FILE: Path = DATA_PATH / "example_zemax" / "frontend.txt"
ZEMAX_MULTI_CONFIG_FILES: list[Path] = list(
    DATA_PATH.glob("example_zemax_multi/c[123456789].txt")
)
FIBRE_TRANSMISSION_FILES: list[Path] = list(DATA_PATH.glob("*[Ff]iber*.csv"))
DETECTOR_QE_FILES: list[Path] = list(DATA_PATH.glob("STA1600LN*.txt")) + list(
    DATA_PATH.glob("*[Dd]etector*.csv")
)

ALL_FILES: list[Path] = [
    *MIRROR_REFLECTIVITY_FILES,
    *FIBRE_TRANSMISSION_FILES,
    *DETECTOR_QE_FILES,
]

if not st.session_state.get("transmission_list", None):
    st.session_state.transmission_list: list[Transmission | EchelleGrating | None] = (
        MARVEL_TRANSMISSION
    )


def show_item_controls(index: int) -> None:
    """Show the relevant controls for a Transmision item"""

    match t := st.session_state.transmission_list[index]:
        case FlatTransmission():
            st.text_input(
                "Name",
                value=t.name,
                key=f"edit_flat_name_{index}",
            )

            st.number_input(
                "Flat fractional transmission value",
                min_value=0.0,
                max_value=1.0,
                value=float(t.transmission),
                step=0.01,
                key=f"edit_flat_value_{index}",
            )

            if st.button("Apply changes", key=f"apply_flat_{index}"):
                t.name: str = st.session_state[f"edit_flat_name_{index}"]
                t.transmission: float = st.session_state[f"edit_flat_value_{index}"]
                t._transmissions: NDArray[np.floating] = np.array([t.transmission])
                st.rerun()

        case None:
            st.info("No editable parameters for `None` entries.")

        case EchelleGrating():
            st.number_input(
                "Groove frequency (g/mm)",
                min_value=0.01,
                value=float(t.groove_frequency.to(u.mm**-1).value),
                step=0.1,
                key=f"edit_echelle_gf_{index}",
            )

            st.number_input(
                "Blaze angle (degrees)",
                min_value=0.0,
                max_value=89.0,
                value=float(t.blaze_angle.to(u.deg).value),
                step=0.1,
                key=f"edit_echelle_blaze_{index}",
            )

            st.number_input(
                "Peak efficiency",
                min_value=0.0,
                max_value=1.0,
                value=float(t.peak_efficiency),
                step=0.01,
                key=f"edit_echelle_peak_efficiency_{index}",
            )

            if st.button("Apply changes", key=f"apply_echelle_{index}"):
                st.session_state.transmission_list[index] = EchelleGrating(
                    groove_frequency=st.session_state[f"edit_echelle_gf_{index}"]
                    / u.mm,
                    blaze_angle=st.session_state[f"edit_echelle_blaze_{index}"] * u.deg,
                    peak_efficiency=st.session_state[
                        f"edit_echelle_peak_efficiency_{index}"
                    ],
                )
                st.rerun()

        case FileTransmission():
            st.text_input(
                "Name",
                value=st.session_state.transmission_list[index].name,
                key=f"edit_name_{index}",
            )

            try:
                current_index = ALL_FILES.index(
                    st.session_state.transmission_list[index]._file.stem
                )
            except Exception:
                st.warning(
                    f"Could not find the file {st.session_state.transmission_list[index]._file.stem} in the list of available files. Please select a new file."
                )
                current_index = 0

            st.selectbox(
                "Select a file:",
                options=ALL_FILES,
                index=current_index,
                format_func=lambda x: x.name,
                key=f"edit_reflectivity_file_{index}",
            )

            st.number_input(
                "Number of reflections",
                value=st.session_state.transmission_list[index].num_passes,
                min_value=1,
                width=300,
                help="""The number of reflections in the optical path. For example, a
                telescope with a primary and secondary mirror using the same coating can
                be modelled as 2 reflections from the same mirror.""",
                key=f"edit_num_reflections_{index}",
            )

            st.number_input(
                "Scale reflectivity by factor",
                value=float(st.session_state.transmission_list[index].scale_factor),
                min_value=0.0,
                step=0.01,
                width=300,
                key=f"edit_scale_factor_{index}",
            )

            if st.button("Apply changes", key=f"apply_echelle_{index}"):
                st.session_state.transmission_list[index] = FileTransmission(
                    name=st.session_state[f"edit_name_{index}"],
                    num_passes=st.session_state[f"edit_num_reflections_{index}"],
                    scale_factor=st.session_state[f"edit_scale_factor_{index}"],
                )

                try:
                    st.session_state.transmission_list[index].load_file(
                        file=st.session_state[f"edit_reflectivity_file_{index}"],
                        wavelength_unit=u.nm,
                        delimiter="\t",
                    )
                except Exception as e:
                    st.error(f"Error loading file: {e}")

                try:
                    st.session_state.transmission_list[index].load_file(
                        file=st.session_state[f"edit_reflectivity_file_{index}"],
                        wavelength_unit=u.nm,
                        delimiter=",",
                    )
                except Exception as e:
                    st.error(f"Error loading file: {e}")

                st.rerun()

        case FlatTransmission():
            st.text_input("Name", value="Flat transmission", key=f"edit_name_{index}")

            st.number_input(
                "Flat fractional transmission value",
                value=0.9,
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                width=300,
                key=f"edit_flat_transmission_{index}",
            )

            if st.button("Apply changes", key=f"apply_flat_{index}"):
                st.session_state.transmission_list[index] = FlatTransmission(
                    name=st.session_state[f"edit_name_{index}"],
                    transmission=st.session_state[f"edit_flat_transmission_{index}"],
                )
                st.rerun()

        case ZemaxTransmission():
            ZEMAX_SINGLE_CONFIG_FILE: Path = (
                DATA_PATH / "example_zemax" / "frontend.txt"
            )

            st.text_input(
                "Name",
                value="Frontend optics (incl. 5% split for guiding)",
                key=f"edit_name_{index}",
            )

            st.selectbox(
                "Select a Zemax single configuration file:",
                options=[ZEMAX_SINGLE_CONFIG_FILE],
                format_func=lambda x: x.name,
                key=f"edit_zemax_single_config_file_{index}",
            )

            st.number_input(
                "Scale transmission by factor",
                value=1.0,
                min_value=0.0,
                step=0.01,
                width=300,
                key=f"edit_zemax_single_config_scale_factor_{index}",
            )

            if st.button("Apply changes", key=f"apply_zemax_single_{index}"):
                st.session_state.transmission_list[index] = ZemaxTransmission(
                    name=st.session_state[f"edit_name_{index}"],
                    scale_factor=st.session_state[
                        f"edit_zemax_single_config_scale_factor_{index}"
                    ],
                ).load_file(
                    file=st.session_state[f"edit_zemax_single_config_file_{index}"],
                )
                st.rerun()

        case ZemaxMultiTransmission():
            ZEMAX_MULTI_CONFIG_FILES: list[Path] = sorted(
                DATA_PATH.glob("example_zemax_multi/c[123456789].txt")
            )

            st.text_input(
                "Name",
                value="Spectrograph, excluding echelle",
                key=f"edit_zemax_multi_config_name_{index}",
            )

            # TODO: File upload!
            files: list[Path] = [
                Path(p)
                for p in st.multiselect(
                    "Select a Zemax multi-configuration file:",
                    options=ZEMAX_MULTI_CONFIG_FILES,
                    format_func=lambda x: x.name,
                    key=f"edit_zemax_multi_config_file_{index}",
                )
            ]

            if not files:
                st.warning("Please select at least one Zemax multi-configuration file.")
                files = [ZEMAX_MULTI_CONFIG_FILES[0]]

            st.number_input(
                "Scale transmission by factor",
                value=1.0,
                min_value=0.0,
                step=0.01,
                width=300,
                key=f"edit_zemax_multi_config_scale_factor_{index}",
            )

            if st.button("Apply changes", key=f"apply_zemax_multi_{index}"):
                st.session_state.transmission_list[index] = ZemaxMultiTransmission(
                    name=st.session_state[f"edit_zemax_multi_config_name_{index}"],
                    scale_factor=st.session_state[
                        f"edit_zemax_multi_config_scale_factor_{index}"
                    ],
                ).load_file(
                    files=files,
                )
                st.rerun()

        case Transmission():
            st.text_input(
                "Name",
                value=t.name,
                key=f"edit_name_{index}",
            )

            if st.toggle(
                label="Smoothed values",
                value=False,
                key=f"edit_smooth_{index}",
            ):
                edited_data = st.data_editor(
                    data={
                        "Wavelength (nm)": t.wavelengths.to(u.nm).value,
                        "Transmission": t.transmissions,
                    },
                    hide_index=True,
                )

            else:
                edited_data = st.data_editor(
                    data={
                        "Wavelength (nm)": t._wavelengths.to(u.nm).value,
                        "Transmission": t._transmissions,
                    },
                    hide_index=True,
                )

            st.number_input(
                "Multiply transmission by factor",
                min_value=0.0,
                value=1.0,
                step=0.01,
                key=f"edit_scale_{index}",
                help=(
                    "Scales the current transmission values. Use 1.0 to keep unchanged."
                ),
            )

            if st.button("Apply changes", key=f"apply_generic_{index}"):
                st.session_state.transmission_list[index] = Transmission(
                    name=st.session_state[f"edit_name_{index}"],
                    _wavelengths=edited_data["Wavelength (nm)"].to_numpy() * u.nm,
                    _transmissions=edited_data["Transmission"].to_numpy(),
                    scale_factor=st.session_state[f"edit_scale_{index}"],
                )
                st.rerun()

        case _:
            st.warning(
                f"Unknown transmission type: {type(st.session_state.transmission_list[index])}"
            )


def display_list() -> None:
    """Show and manage the list of transmission curves"""

    def _move_item(index: int, direction: int) -> None:
        new_index: int = index + direction
        if new_index < 0 or new_index >= len(st.session_state.transmission_list):
            return
        (
            st.session_state.transmission_list[index],
            st.session_state.transmission_list[new_index],
        ) = (
            st.session_state.transmission_list[new_index],
            st.session_state.transmission_list[index],
        )

    def _delete_item(index: int) -> None:
        del st.session_state.transmission_list[index]

    if len(st.session_state.transmission_list) == 0:
        st.write("No transmission curves added yet.")
        return

    with st.container(horizontal_alignment="center"):
        st.write("Transmission curves")

    for i, t in enumerate(st.session_state.transmission_list):
        title: str

        if t is None:
            title = f"{i + 1}. None"

        elif isinstance(t, EchelleGrating):
            title = f"{i + 1}. Echelle Grating"

        else:
            title = f"{i + 1}. {t.name}"

        main_column, controls_column = st.columns([3, 1])

        with controls_column:
            controls = st.columns([1, 1, 2])
            with controls[0]:
                if st.button("↑", key=f"move_up_{i}", disabled=(i == 0)):
                    _move_item(i, -1)
                    st.rerun()

            with controls[1]:
                if st.button(
                    "↓",
                    key=f"move_down_{i}",
                    disabled=(i == len(st.session_state.transmission_list) - 1),
                ):
                    _move_item(i, 1)
                    st.rerun()

            with controls[2]:
                if st.button("Delete", key=f"delete_{i}", type="tertiary"):
                    _delete_item(i)
                    st.rerun()

        with main_column, st.popover(title, width="stretch"):
            show_item_controls(i)


def transmission_selector() -> None:
    """
    A dropdown selector for choosing which transmission curves to add to the list.
    """

    cols: Iterator = cycle(st.columns(3))

    with next(cols):
        if st.button(
            label="Tabulated",
            width="stretch",
            type="primary",
        ):
            st.session_state.transmission_list.append(
                Transmission(
                    name="Tabulated",
                    _wavelengths=np.array([400, 500, 600, 700, 800, 900]) * u.nm,
                    _transmissions=np.array([0.8, 0.9, 0.85, 0.9, 0.95, 0.99]),
                )
            )

    with next(cols):
        if st.button(
            label="Mirror reflectivity",
            width="stretch",
            type="primary",
        ):
            st.session_state.transmission_list.append(
                FileTransmission(
                    name="Mirror reflectivity",
                    num_passes=2,
                    scale_factor=1,
                ).load_file(file=MIRROR_REFLECTIVITY_FILES[0])
            )

    with next(cols):
        if st.button(
            label="Zemax (Single Configuration)",
            width="stretch",
            type="primary",
        ):
            st.session_state.transmission_list.append(
                ZemaxTransmission(
                    name="Zemax (Single Configuration)",
                    scale_factor=1,
                ).load_file(
                    file=ZEMAX_SINGLE_CONFIG_FILE,
                )
            )

    with next(cols):
        if st.button(
            label="Zemax (Multi-Configuration)",
            width="stretch",
            type="primary",
        ):
            st.session_state.transmission_list.append(
                ZemaxMultiTransmission(
                    name="Zemax (Multi-Configuration)",
                    scale_factor=1,
                ).load_file(
                    files=ZEMAX_MULTI_CONFIG_FILES,
                )
            )

    with next(cols):
        if st.button(
            label="Fibre Transmission (CSV File)",
            width="stretch",
            type="primary",
        ):
            st.session_state.transmission_list.append(
                FileTransmission(
                    name="Fibre Transmission (CSV File)",
                ).load_file(
                    file=FIBRE_TRANSMISSION_FILES[0],
                    wavelength_unit=u.nm,
                )
            )

    with next(cols):
        if st.button(
            label="Flat Transmission",
            width="stretch",
            type="primary",
        ):
            st.session_state.transmission_list.append(
                FlatTransmission(
                    name="Flat Transmission",
                    transmission=1,
                )
            )

    with next(cols):
        if st.button(
            label="Detector QE",
            width="stretch",
            type="primary",
        ):
            st.session_state.transmission_list.append(
                FileTransmission(
                    name="Detector QE",
                ).load_file(
                    file=DETECTOR_QE_FILES[0],
                    wavelength_unit=u.nm,
                )
            )

    with next(cols):
        if st.button(
            label="Echelle Grating",
            width="stretch",
            type="primary",
        ):
            if any(
                isinstance(t, EchelleGrating)
                for t in st.session_state.transmission_list
            ):
                st.warning(
                    """
                    Only one Echelle Grating can be added to the list.
                    Please remove the existing one first.
                    """,
                )
            else:
                st.session_state.transmission_list.append(EchelleGrating())

    with next(cols):
        if st.button(
            label="Reset",
            width="stretch",
            type="secondary",
        ):
            st.session_state.transmission_list = [MARVEL_TRANSMISSION]
            st.rerun()


def main() -> None:
    """
    Main function to run the Streamlit app.
    """
    st.header("Instrument Transmission", anchor=False)

    with st.expander("About this tool", expanded=False):
        f"""
    This tool can be used to plot a flowdown diagram of the transmission of an
    optical system. Any number of unitary transmission curves can be added to the list,
    which will be plotted in order from top to bottom, as a function of wavelength.
    The total transmission, the product of all transmission curves, can be downloaded as
    a CSV file for further analysis or use in other tools.

    All code is written by Jake Pember {badges}

    Please get in contact if you have any questions, suggestions, or find any bugs!

    You can also [Buy me a pint](https://buymeacoffee.com/pemberj) if you find this tool
    useful and want to support its development!
    """

        bmac()

    left, right = st.columns(2)

    with left:
        transmission_selector()

        st.divider()

        display_list()

        with st.container(horizontal_alignment="center"):
            st.caption(
                "Click on a transmission curve to expand and edit its parameters."
            )

            if st.button(label="Remove all", width=300, type="tertiary"):
                st.session_state.transmission_list = [None]
                st.rerun()

    with right:
        if len(st.session_state.transmission_list) > 0:
            fig: plt.Figure = transmission_plot(
                Ts=st.session_state.transmission_list,
                min_wl=0.37 * u.um,
                max_wl=0.92 * u.um,
                num_wls=500,
            )

            st.pyplot(fig)

        total: tuple[np.ndarray, np.ndarray] = total_transmission(
            T=st.session_state.transmission_list,
            min_wl=0.37 * u.um,
            max_wl=0.92 * u.um,
            num_wls=500,
            save=False,
        )

        with st.container(horizontal_alignment="center"):
            st.download_button(
                label="Download CSV",
                data="\n".join(
                    [
                        "Wavelength (nm), Transmission",
                        *[
                            f"{wl:.5f}, {trans:.5f}"
                            for wl, trans in zip(total[0], total[1], strict=True)
                        ],
                    ]
                ),
                file_name="total_transmission.csv",
                mime="text/csv",
            )

            with st.container(width=300):
                st.caption(
                    """Exported CSV files have two columns: `Wavelength (nm)` and
`Transmission`.

Note: Echelle grating blaze functions are not included in the exported data."""
                )


if __name__ == "__main__":
    main()
