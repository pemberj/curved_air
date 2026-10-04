#! uv run streamlit run
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
    CSVTransmission,
    FileTransmission,
    FlatTransmission,
    Transmission,
    ZemaxMultiTransmission,
    ZemaxTransmission,
    total_transmission,
    transmission_plot,
    # total_transmission,
)
from curved_air.utils import richardson_gratings, st_quantity_input

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from curved_air.type_hinting import ScalarQuantity

plt.style.use(plot_style)

DATA_PATH: Path = Path(__file__).parent / ".." / ".." / "transmission_data"


if not st.session_state.get("transmission_list", None):
    st.session_state.transmission_list: list[Transmission | EchelleGrating | None] = (
        MARVEL_TRANSMISSION
    )


def show_item_controls(index: int) -> None:
    """Show the relevant controls for a Transmision item"""

    match st.session_state.transmission_list[index]:
        case FlatTransmission():
            t = st.session_state.transmission_list[index]
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
            g = st.session_state.transmission_list[index]
            st.number_input(
                "Groove frequency (g/mm)",
                min_value=0.01,
                value=float(g.groove_frequency.to(u.mm**-1).value),
                step=0.1,
                key=f"edit_echelle_gf_{index}",
            )
            st.number_input(
                "Blaze angle (degrees)",
                min_value=0.0,
                max_value=89.0,
                value=float(g.blaze_angle.to(u.deg).value),
                step=0.1,
                key=f"edit_echelle_blaze_{index}",
            )
            st.number_input(
                "Ruled width (mm)",
                min_value=0.01,
                value=float(g.ruled_width.to(u.mm).value),
                step=0.1,
                key=f"edit_echelle_width_{index}",
            )
            st.number_input(
                "Gamma angle (degrees)",
                min_value=0.0,
                value=float(g.gamma_angle.to(u.deg).value),
                step=0.1,
                key=f"edit_echelle_gamma_{index}",
            )
            st.number_input(
                "Peak efficiency",
                min_value=0.0,
                max_value=1.0,
                value=float(g.peak_efficiency),
                step=0.01,
                key=f"edit_echelle_peak_efficiency_{index}",
            )

            if st.button("Apply changes", key=f"apply_echelle_{index}"):
                st.session_state.transmission_list[index] = EchelleGrating(
                    groove_frequency=st.session_state[f"edit_echelle_gf_{index}"]
                    / u.mm,
                    blaze_angle=st.session_state[f"edit_echelle_blaze_{index}"] * u.deg,
                    ruled_width=st.session_state[f"edit_echelle_width_{index}"] * u.mm,
                    gamma_angle=st.session_state[f"edit_echelle_gamma_{index}"] * u.deg,
                    peak_efficiency=st.session_state[
                        f"edit_echelle_peak_efficiency_{index}"
                    ],
                )
                st.rerun()

        case Transmission():
            t = st.session_state.transmission_list[index]
            st.text_input(
                "Name",
                value=t.name,
                key=f"edit_name_{index}",
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
                t.name = st.session_state[f"edit_name_{index}"]
                scale: float = st.session_state[f"edit_scale_{index}"]
                if not np.isclose(scale, 1.0):
                    t.transmissions = np.clip(
                        np.asarray(t.transmissions) * scale,
                        0.0,
                        1.0,
                    )
                st.rerun()


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

    st.write("Transmission curves:")

    for i, t in enumerate(st.session_state.transmission_list):
        title: str

        if t is None:
            title = f"{i + 1}. None"

        elif isinstance(t, EchelleGrating):
            title = f"{i + 1}. Echelle Grating"

        else:
            title = f"{i + 1}. {t.name}"

        main_column, controls_column = st.columns([4, 1])

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


def add_flat_transmission() -> None:

    st.text_input("Name", value="Flat transmission", key="flat_transmission_name")

    st.number_input(
        "Flat fractional transmission value",
        value=0.9,
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        width=300,
        key="flat_transmission_value",
    )

    st.button(
        "Add to list",
        on_click=lambda: st.session_state.transmission_list.append(
            FlatTransmission(
                name=st.session_state.flat_transmission_name,
                transmission=st.session_state.flat_transmission_value,
            )
        ),
        type="primary",
    )


def add_mirror_reflectivity() -> None:

    MIRROR_REFLECTIVITY_FILES: list[Path] = list(DATA_PATH.glob("ASA*.csv"))

    st.text_input("Name", value="Telescope mirrors", key="mirror_reflectivity_name")

    st.selectbox(
        "Select a reflectivity file:",
        options=MIRROR_REFLECTIVITY_FILES,
        format_func=lambda x: x.name,
        key="reflectivity_file",
    )

    st.number_input(
        "Number of reflections",
        value=2,
        min_value=1,
        width=300,
        help="""The number of reflections in the optical path. For example, a telescope
        with a primary and secondary mirror has 2 reflections.""",
        key="number_of_reflections",
    )

    st.number_input(
        "Scale reflectivity by factor",
        value=1.0,
        min_value=0.0,
        step=0.01,
        width=300,
        key="mirror_reflectivity_scale_factor",
    )

    st.button(
        "Add to list",
        on_click=lambda: st.session_state.transmission_list.append(
            CSVTransmission(
                name=st.session_state.mirror_reflectivity_name,
                num_passes=st.session_state.number_of_reflections,
                scale_factor=st.session_state.mirror_reflectivity_scale_factor,
            ).load_file(file=st.session_state.reflectivity_file)
        ),
        type="primary",
    )


def add_zemax_single_config() -> None:

    ZEMAX_SINGLE_CONFIG_FILE: Path = DATA_PATH / "example_zemax" / "frontend.txt"

    st.text_input(
        "Name",
        value="Frontend optics (incl. 5% split for guiding)",
        key="zemax_single_config_name",
    )

    st.selectbox(
        "Select a Zemax single configuration file:",
        options=[ZEMAX_SINGLE_CONFIG_FILE],
        format_func=lambda x: x.name,
        key="zemax_single_config_file",
    )

    st.number_input(
        "Scale transmission by factor",
        value=1.0,
        min_value=0.0,
        step=0.01,
        width=300,
        key="zemax_single_config_scale_factor",
    )

    st.button(
        "Add to list",
        on_click=lambda: st.session_state.transmission_list.append(
            ZemaxTransmission(
                name=st.session_state.zemax_single_config_name,
                scale_factor=st.session_state.zemax_single_config_scale_factor,
            ).load_file(
                file=st.session_state.zemax_single_config_file,
            )
        ),
        type="primary",
    )


def add_zemax_multi_config() -> None:

    ZEMAX_MULTI_CONFIG_FILES: list[Path] = list(
        DATA_PATH.glob("example_zemax_multi/c[123456789].txt")
    )

    st.text_input(
        "Name",
        value="Spectrograph, excluding echelle",
        key="zemax_multi_config_name",
    )

    st.selectbox(
        "Select a Zemax multi-configuration file:",
        options=ZEMAX_MULTI_CONFIG_FILES,
        format_func=lambda x: x.name,
        key="zemax_multi_config_file",
    )

    st.number_input(
        "Scale transmission by factor",
        value=1.0,
        min_value=0.0,
        step=0.01,
        width=300,
        key="zemax_multi_config_scale_factor",
    )

    st.button(
        "Add to list",
        on_click=lambda: st.session_state.transmission_list.append(
            ZemaxMultiTransmission(
                name=st.session_state.zemax_multi_config_name,
                scale_factor=st.session_state.zemax_multi_config_scale_factor,
            ).load_file(
                files=[st.session_state.zemax_multi_config_file],
            )
        ),
    )


def add_fibre_transmission() -> None:

    FIBRE_TRANSMISSION_FILES: list[Path] = list(DATA_PATH.glob("*[Ff]iber*.csv"))

    st.text_input(
        "Name",
        value="Fibre transmission",
        key="fibre_transmission_name",
    )

    st.selectbox(
        "Select a fibre transmission file:",
        options=FIBRE_TRANSMISSION_FILES,
        format_func=lambda x: x.name,
        key="fibre_transmission_file",
    )

    st.number_input(
        "Scale transmission by factor",
        value=1.0,
        min_value=0.0,
        step=0.01,
        width=300,
        key="fibre_transmission_scale_factor",
    )

    st.button(
        "Add to list",
        on_click=lambda: st.session_state.transmission_list.append(
            FileTransmission(
                name=st.session_state.fibre_transmission_name,
            ).load_file(
                file=st.session_state.fibre_transmission_file,
                wavelength_unit=u.nm,
            )
        ),
    )


def add_detector_qe() -> None:

    DETECTOR_QE_FILES: list[Path] = list(DATA_PATH.glob("STA1600LN*.txt")) + list(
        DATA_PATH.glob("*[Dd]etector*.csv")
    )

    st.text_input(
        "Name",
        value="Detector QE",
        key="detector_qe_name",
    )

    st.selectbox(
        "Select a detector QE file:",
        options=DETECTOR_QE_FILES,
        format_func=lambda x: x.name,
        key="detector_qe_file",
    )

    st.button(
        "Add to list",
        on_click=lambda: st.session_state.transmission_list.append(
            FileTransmission(
                name=st.session_state.detector_qe_name,
            ).load_file(
                file=st.session_state.detector_qe_file,
                wavelength_unit=u.nm,
            )
        ),
    )


def add_echelle_grating() -> None:

    if any(isinstance(t, EchelleGrating) for t in st.session_state.transmission_list):
        st.warning(
            "There is already an echelle grating in the list. Only one is allowed."
        )
        return

    ECHELLE_GRATINGS: dict[str, EchelleGrating] = {
        f"{g.catalogue_number}: {g.groove_frequency.value:g}g/mm R{g.R:g}": g
        for g in richardson_gratings(selection_query=None)
    }

    grating: str = st.selectbox(
        "Richardson gratings",
        options=list(ECHELLE_GRATINGS.keys()),
        index=16,  # MARVEL grating
        width=300,
        disabled=st.session_state.get("customise_grating", False),
        help="""Select from a list of commercially available echelle gratings from
        Richardson Gratings. The grating parameters will be pre-filled based on the
        selected option.""",
    )

    grating: EchelleGrating = ECHELLE_GRATINGS[grating]
    grating.peak_efficiency = 0.75

    if not st.session_state.get("customise_grating", False):
        if grating.groove_length is not None:
            st.metric(
                "Maximum Groove Length",
                f"{grating.groove_length.to(u.mm):.0f}",
            )
        else:
            st.warning(
                "Warning: maximum groove length is not specified for this grating."
            )

        if grating.ruled_width is not None:
            st.metric(
                "Maximum Ruled Width",
                f"{grating.ruled_width.to(u.mm):.0f}",
            )
        else:
            st.warning(
                "Warning: maximum ruled width is not specified for this grating."
            )

    st.toggle(
        "Custom grating parameters",
        value=False,
        key="customise_grating",
        help="""Input your own values for the grating groove spacing, blaze angle,
        and ruled width.""",
    )

    if st.session_state["customise_grating"]:
        groove_frequency: ScalarQuantity = st_quantity_input(
            "Groove freqency",
            min_value=0.0,
            value=grating.groove_frequency.to(u.mm**-1).value,
            step=0.1,
            possible_units=["µm", "mm", "cm"],
            default_unit="mm",
            unit_input_text="Per",
            inverse=True,
            key="grating_groove_frequency",
            # help="""Spatial groove frequency""",
        )

        blaze_angle: ScalarQuantity = (
            st.number_input(
                "Blaze angle (degrees)",
                min_value=0.0,
                max_value=89.0,
                value=grating.blaze_angle.to(u.deg).value,
                step=0.1,
                width=300,
                key="grating_blaze_angle",
                help="""The blaze angle of the echelle grating, in degrees. This is the
                angle of illumination under which the grating is designed for optimal
                efficiency.""",
            )
            * u.deg
        )

        ruled_width: ScalarQuantity = st_quantity_input(
            "Ruled width (mm)",
            min_value=0.0,
            # max_value=1200.0,
            value=grating.ruled_width.to(u.mm).value,
            step=0.05,
            possible_units=["mm", "cm", "m"],
            default_unit="mm",
            key="grating_ruled_width",
            help="""The ruled width of the echelle grating, in millimetres. This refers
            to the dimension perpendicular to the grooves themselves.""",
        )

        st.metric(
            "Grating R-number",
            f"R{np.tan(blaze_angle):.1f}",
            help="""The R-number of an echelle grating is a dimensionless parameter
            defined as the tangent of the blaze angle.""",
        )

    else:
        groove_frequency: float = grating.groove_frequency.to(u.mm**-1)
        blaze_angle: float = grating.blaze_angle.to(u.deg)
        ruled_width: float = grating.ruled_width.to(u.mm)

    gamma_angle: ScalarQuantity = (
        st.number_input(
            "Gamma angle (degrees)",
            min_value=0.0,
            value=0.3,
            step=0.1,
            width=300,
            key="grating_gamma_angle",
            help="""The gamma (out-of-plane) angle under which the echelle grating is
            used, in degrees.""",
        )
        * u.deg
    )

    peak_efficiency: float = st.number_input(
        "Peak efficiency",
        value=grating.peak_efficiency,
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        width=300,
        key="grating_peak_efficiency",
        help="""The peak efficiency of the echelle grating, as a fraction between 0 and
        1. This is the maximum efficiency of the grating at the blaze wavelength.""",
    )

    echelle_grating: EchelleGrating = EchelleGrating(
        groove_frequency=groove_frequency,
        blaze_angle=blaze_angle,
        ruled_width=ruled_width,
        gamma_angle=gamma_angle,
        peak_efficiency=peak_efficiency,
    )

    st.button(
        "Add to list",
        on_click=lambda: st.session_state.transmission_list.append(echelle_grating),
    )


def transmission_selector() -> None:
    """
    A dropdown selector for choosing which transmission curves to add to the list.
    """

    options: list[str] = [
        "Mirror Reflectivity and Number",
        "Zemax, Single Configuration",
        "Zemax, Multi-Configuration",
        "Fibre Transmission & Coupling",
        "Flat Transmission",
        "Detector Quantum Efficiency",
        "Echelle Grating",
    ]

    st.selectbox(
        "Select a transmission curve to add to the list:",
        options=options,
        key="transmission_selector",
    )

    if st.session_state.transmission_selector == "Mirror Reflectivity and Number":
        add_mirror_reflectivity()
    elif st.session_state.transmission_selector == "Zemax, Single Configuration":
        add_zemax_single_config()
    elif st.session_state.transmission_selector == "Zemax, Multi-Configuration":
        add_zemax_multi_config()
    elif st.session_state.transmission_selector == "Fibre Transmission & Coupling":
        add_fibre_transmission()
    elif st.session_state.transmission_selector == "Flat Transmission":
        add_flat_transmission()
    elif st.session_state.transmission_selector == "Detector Quantum Efficiency":
        add_detector_qe()
    elif st.session_state.transmission_selector == "Echelle Grating":
        add_echelle_grating()


def main() -> None:
    """
    Main function to run the Streamlit app.
    """
    st.header("Instrument Transmission Curves", anchor=False)

    with st.expander("About this tool", expanded=False):
        """
        This tool can be used to plot a flowdown diagram of the transmission of an
        optical system. Any number of unitary transmission curves can be added to the
        list, which will be plotted in order from top to bottom, as a function of
        wavelength. The total transmission, the product of all transmission curves,
        can be downloaded as a CSV file for further analysis or use in other tools.

        All code is written by Jake Pember
        :gray-badge[:material/language: [Web](https://www.jakepember.com)]
        :gray-badge[:material/email: [Email](mailto:jacob.pember@ikmail.com)]
        :gray-badge[:material/code: [Github](https://www.github.com/pemberj)]

        Please get in contact if you have any questions, suggestions, or find any bugs!

        You can also [Buy me a pint](https://buymeacoffee.com/pemberj) if you find this
        tool useful and want to support its development!
        """

        st.iframe(
            """
            <script type="text/javascript" src="https://cdnjs.buymeacoffee.com/1.0.0/button.prod.min.js" data-name="bmc-button" data-slug="pemberj" data-color="#E5660C" data-emoji="🍺" data-font="Cookie" data-text="Buy me a pint" data-outline-color="#000000" data-font-color="#ffffff"></script>
            """,
        )

    left, right = st.columns(2)

    with left:
        transmission_selector()

        st.divider()

        display_list()

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


def test() -> None:
    Telescope: Transmission = CSVTransmission(
        name="Telescope mirrors (ASA)",
        num_passes=3,
    ).load_file(file=next(iter(DATA_PATH.glob("ASA_Protected_Enhanced_Aluminium.csv"))))

    Frontend: Transmission = ZemaxTransmission(
        name="Frontend optics (incl. 5% split for guiding)",
    ).load_file(
        file=DATA_PATH / "example_zemax" / "frontend.txt",
    )

    FibreAperture: Transmission = FlatTransmission(
        name="Fibre aperture loss", transmission=0.80
    )

    Fibre: Transmission = Transmission(
        name="Fibre transmission & coupling from\ncircular to octagonal fibre",
        _wavelengths=np.array([390, 400, 500, 700, 920]) * u.nm,
        _transmissions=np.array(
            [
                0.91 ** (15 / 10) * 0.92,
                0.920 ** (15 / 10) * 0.92,
                0.963 ** (15 / 10) * 0.92,
                0.987 ** (15 / 10) * 0.92,
                0.991 ** (15 / 10) * 0.92,
            ]
        ),
    )

    Slicer = FlatTransmission(name="Slicer/double scrambler", transmission=0.9)

    Spectrograph: Transmission = ZemaxMultiTransmission(
        name="Spectrograph, excluding echelle",
    ).load_file(
        files=list((DATA_PATH / "example_zemax_multi").glob("c[123456789].txt")),
    )

    Detector: Transmission = FileTransmission(
        name="STA1600LN Detector QE",
    ).load_file(
        file=next(iter(DATA_PATH.glob("STA1600LN*.txt"))),
        wavelength_unit=u.nm,
    )

    Ts: list[Transmission | EchelleGrating | None] = [
        # atmosphere(), # Not yet implemented
        Telescope,
        Frontend,
        FibreAperture,
        Fibre,
        Slicer,
        Spectrograph,
        Detector,
        EchelleGrating(
            groove_frequency=31.6 / u.mm,
            blaze_angle=63.5 * u.deg,
            ruled_width=100 * u.mm,
            gamma_angle=0.3 * u.deg,
            min_wl=370 * u.nm,
            max_wl=950 * u.nm,
        ),
        None,
    ]

    fig: plt.Figure = transmission_plot(
        Ts=Ts,
        min_wl=0.37 * u.um,
        max_wl=0.92 * u.um,
        num_wls=500,
    )

    st.pyplot(fig)


if __name__ == "__main__":
    main()
