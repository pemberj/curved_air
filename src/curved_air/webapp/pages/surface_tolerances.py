#! uv run streamlit run

import pandas as pd
import streamlit as st

from curved_air.type_hinting import ScalarQuantity
from curved_air.utils import (
    SurfaceTolerance,
    radius2sagitta,
    sagitta2radius,
    st_quantity_input,
)

st.header("Surface Tolerances", anchor=False)

with st.expander("About this tool", expanded=False):
    """
    A simple calculator to convert between optical fringes and radius-of-curvature
    tolerance, via calculation of the nominal, flatter, and steeper sagitta of the
    surface.

    After entering the surface parameters and tolerances, click "Add surface" to add the
    surface to the table. You can add as many surfaces as you like, and the table can be
    downloaded as a CSV file.

    All code is written by Jake Pember :gray-badge[:material/language: [Web](https://www.jakepember.com)] :gray-badge[:material/email: [Email](mailto:jacob.pember@ikmail.com)] :gray-badge[:material/code: [Github](https://www.github.com/pemberj)]

    Please get in contact if you have any questions, suggestions, or find any bugs!

    You can also [Buy me a pint](https://buymeacoffee.com/pemberj) if you find this tool
    useful and want to support its development!
    """

    st.iframe(
        """
        <script type="text/javascript" src="https://cdnjs.buymeacoffee.com/1.0.0/button.prod.min.js" data-name="bmc-button" data-slug="pemberj" data-color="#E5660C" data-emoji="🍺" data-font="Cookie" data-text="Buy me a pint" data-outline-color="#000000" data-font-color="#ffffff"></script>
        """,
    )

if "surfaces" not in st.session_state:
    st.session_state["surfaces"] = []

cols = st.columns([1, 1.25])

with cols[0]:
    num_surfaces: int = len(st.session_state.get("surfaces", []))

    name: str = st.text_input(
        "Surface name",
        value="",
        width=500,
        placeholder=f"Surface {num_surfaces + 1}",
        help="""An optional name for the surface. This is used to identify the
        surface in the table to the side.""",
        key="surface_name",
    )

    diameter: ScalarQuantity = st_quantity_input(
        name="Diameter",
        possible_units=["mm", "cm", "m"],
        min_value=0.0,
        value=200.0,
        step=1.0,
        default_unit="mm",
        help="""The surface diameter. This is used to calculate the sagitta, which is
        needed to convert between radius tolerance and fringe tolerance.""",
        key="diameter",
    )

    wavelength: ScalarQuantity = st_quantity_input(
        name="Test wavelength",
        possible_units=["nm", "um"],
        min_value=0.0,
        max_value=None,
        value=632.8,
        step=0.1,
        default_unit="nm",
        help="""The wavelength at which the surface will be tested. This is used to
        convert between radius tolerance and fringe tolerance.""",
        key="wavelength",
    )

    radius: ScalarQuantity = st_quantity_input(
        name="Nominal radius of curvature",
        possible_units=["mm", "cm", "m"],
        min_value=0.0,
        value=1000.0,
        step=0.01,
        default_unit="mm",
        help="""The nominal radius of curvature of the surface.""",
        key="radius",
    )
    if radius < diameter / 2:
        st.warning(
            "The radius of curvature is less than half the diameter of the surface!"
        )

    mode: str | None = st.segmented_control(
        label="Specify tolerances in:",
        options=["Radius", "Fringes", "Sagitta"],
        selection_mode="single",
        default="Radius",
        help="""Choose whether to specify the surface tolerances in terms of radius of
        curvature, the number of interference fringes, or the allowable deviation from
        nominal sagitta""",
        key="tolerance_specification_method",
    )

    match mode:
        case "Radius":
            neg_toggle: bool = st.toggle(
                label="Different positive and negative radius tolerances?",
                value=False,
                help="""By default, the tolerance is specified symmetrically, with the
                positive and negative radius tolerances are the same. Toggle this on if you
                want to specify different positive and negative radius tolerances.""",
                key="neg_toggle",
            )

            positive_radius_tolerance: ScalarQuantity = st_quantity_input(
                name="Positive radius tolerance (+)"
                if neg_toggle
                else "Radius tolerance",
                possible_units=["um", "mm", "cm", "m"],
                min_value=0.0,
                value=1.0,
                step=0.01,
                default_unit="mm",
                key="positive_radius_tolerance",
            )

            if neg_toggle:
                negative_radius_tolerance: ScalarQuantity = st_quantity_input(
                    name="Negative radius tolerance (-)",
                    possible_units=["um", "mm", "cm", "m"],
                    min_value=0.0,
                    value=1.0,
                    step=0.01,
                    default_unit="mm",
                    key="negative_radius_tolerance",
                )
            else:
                negative_radius_tolerance: ScalarQuantity = positive_radius_tolerance

        case "Fringes":
            fringes: float = st.number_input(
                "Fringe tolerance",
                min_value=0.0,
                value=5.0,
                step=0.1,
                width=200,
                help="""The surface tolerance in number of interference fringes.
                1 fringe = 1/2 test wavelength of sagitta difference between the nominal and
                the tolerated curvature of the surface.""",
                key="fringes",
            )

        case "Sagitta":
            neg_toggle: bool = st.toggle(
                label="Different positive and negative sagitta tolerances?",
                value=False,
                help="""By default, the tolerance is specified symmetrically, with the
                positive and negative sagitta tolerances are the same. Toggle this on if
                you want to specify different positive and negative allowed sagitta
                deviations.""",
                key="neg_toggle",
            )

            positive_sagitta_tolerance: ScalarQuantity = st_quantity_input(
                name="Positive sagitta tolerance (+)"
                if neg_toggle
                else "Sagitta tolerance",
                possible_units=["nm", "um", "mm", "cm", "m"],
                min_value=0.0,
                value=5.0,
                step=0.01,
                default_unit="um",
                key="positive_sagitta_tolerance",
            )

            if neg_toggle:
                negative_sagitta_tolerance: ScalarQuantity = st_quantity_input(
                    name="Negative sagitta tolerance (-)",
                    possible_units=["nm", "um", "mm", "cm", "m"],
                    min_value=0.0,
                    value=5,
                    step=0.01,
                    default_unit="um",
                    key="negative_sagitta_tolerance",
                )
            else:
                negative_sagitta_tolerance: ScalarQuantity = (
                    -1 * positive_sagitta_tolerance
                )

            nominal_sagitta: ScalarQuantity = radius2sagitta(radius, diameter)

            positive_radius_tolerance: ScalarQuantity = radius - 1 * sagitta2radius(
                nominal_sagitta + 1 * positive_sagitta_tolerance,
                diameter=diameter,
            )
            negative_radius_tolerance: ScalarQuantity = radius - 1 * sagitta2radius(
                nominal_sagitta - 1 * negative_sagitta_tolerance,
                diameter=diameter,
            )

        case _:
            st.warning("Please select a tolerance specification method.")

    st.divider()

    surface = SurfaceTolerance(
        name=name if name else f"Surface {num_surfaces + 1}",
        wavelength=wavelength,
        nominal_radius=radius,
        diameter=diameter,
        positive_tolerance=positive_radius_tolerance
        if mode in ["Radius", "Sagitta"]
        else None,
        negative_tolerance=negative_radius_tolerance
        if mode in ["Radius", "Sagitta"]
        else None,
        fringes=fringes if mode == "Fringes" else None,
    )

    with st.container(horizontal_alignment="center"):
        st.button(
            "Add surface",
            on_click=lambda: st.session_state["surfaces"].append(surface),
            type="primary",
        )

    df = pd.DataFrame(
        data=[
            {
                "Name": surface.name,
                "Diameter": f"{surface.diameter:.6g}",
                "Radius": f"{surface.nominal_radius:.7g} {surface.positive_tolerance.value:+.4g}/{-surface.negative_tolerance:+.4g}",
                "Fringes": f"{surface.fringes:.0f} fringes @ λ = {surface.wavelength:.4g}",
                "Sagitta": f"{surface.nominal_sagitta:.4g} {surface.sagitta_tolerance[0].value:+.4g}/{surface.sagitta_tolerance[1]:+.4g}",
            }
            for surface in st.session_state.get("surfaces", [])
        ]
    )

with cols[1]:
    st.dataframe(df, hide_index=True, width="stretch", height=600)

    subcols = st.columns(2)

    with subcols[0], st.container(horizontal_alignment="center"):
        st.button(
            "Clear table",
            on_click=lambda: st.session_state.pop("surfaces", None),
            type="secondary",
            disabled=df.empty,
            help="Clear all surface data. This cannot be undone.",
        )

    with subcols[1], st.container(horizontal_alignment="center"):
        st.download_button(
            "Download table as CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="surface_tolerances.csv",
            disabled=df.empty,
            mime="text/csv",
        )
