#! uv run streamlit run

import pandas as pd
import streamlit as st

from curved_air.type_hinting import ScalarQuantity
from curved_air.utils import SurfaceTolerance, st_quantity_input

st.header("Surface Tolerances", anchor=False)

with st.expander("About this tool", expanded=False):
    st.markdown(
        """
        A simple calculator to convert between optical fringes and radius-of-curvature tolerance, via calculation of the nominal, flatter, and steeper sagitta of the surface.

        After entering the surface parameters and tolerances, click "Add surface" to add the surface to the table. You can add as many surfaces as you like, and the table can be downloaded as a CSV file.

        All code is written by Jake Pember: [Github](https://github.com/pemberj/) / [Email](mailto:jacob.pember@ikmail.com)

        Please get in contact if you have any questions, suggestions, or find any bugs!

        You can also [Buy me a pint](https://buymeacoffee.com/pemberj) if you find this tool useful and want to support its development!
        """
    )

    st.iframe(
        """
        <script type="text/javascript" src="https://cdnjs.buymeacoffee.com/1.0.0/button.prod.min.js" data-name="bmc-button" data-slug="pemberj" data-color="#FF5F5F" data-emoji="🍺" data-font="Cookie" data-text="Buy me a pint" data-outline-color="#000000" data-font-color="#ffffff"></script>
        """,
    )

cols = st.columns([1, 1.25])

with cols[0]:
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
        options=["Radius", "Fringes"],
        selection_mode="single",
        default="Radius",
        help="""Choose whether to specify the surface tolerances in terms of radius of
        curvature or in number of interference fringes.""",
        key="tolerance_specification_method",
    )

    if mode is None:
        st.warning("Please select a tolerance specification method.")

    if mode == "Radius":
        neg_toggle: bool = st.toggle(
            label="Different positive and negative radius tolerances?",
            value=False,
            help="""By default, the tolerance is specified symmetrically, with the
            positive and negative radius tolerances are the same. Toggle this on if you
            want to specify different positive and negative radius tolerances.""",
            key="neg_toggle",
        )

        positive_radius_tolerance: ScalarQuantity = st_quantity_input(
            name="Positive radius tolerance (+)" if neg_toggle else "Radius tolerance",
            possible_units=["mm", "cm", "m"],
            min_value=0.0,
            value=1.0,
            step=0.01,
            default_unit="mm",
            key="positive_radius_tolerance",
        )

        if neg_toggle:
            negative_radius_tolerance: ScalarQuantity = st_quantity_input(
                name="Negative radius tolerance (-)",
                possible_units=["mm", "cm", "m"],
                min_value=0.0,
                value=1.0,
                step=0.01,
                default_unit="mm",
                key="negative_radius_tolerance",
            )
        else:
            negative_radius_tolerance: ScalarQuantity = positive_radius_tolerance

    elif mode == "Fringes":
        fringes: float = st.number_input(
            "Fringe tolerance",
            min_value=0.0,
            value=5.0,
            step=0.1,
            width=400,
            help="""The surface tolerance in number of interference fringes.
            1 fringe = 1/2 test wavelength of sagitta difference between the nominal and
            the tolerated curvature of the surface.""",
            key="fringes",
        )

    num_surfaces: int = len(st.session_state.get("surfaces", []))

    if mode is not None:
        name: str = st.text_input(
            "Surface name",
            value="",
            width=400,
            placeholder=f"Surface {num_surfaces + 1}",
            help="""An optional name for the surface. This is used to identify the
            surface in the table to the side.""",
            key="surface_name",
        )

        st.divider()

        surface = SurfaceTolerance(
            name=name if name else f"Surface {num_surfaces + 1}",
            wavelength=wavelength,
            nominal_radius=radius,
            diameter=diameter,
            positive_tolerance=positive_radius_tolerance if mode == "Radius" else None,
            negative_tolerance=negative_radius_tolerance if mode == "Radius" else None,
            fringes=fringes if mode == "Fringes" else None,
        )

        if "surfaces" not in st.session_state:
            st.session_state["surfaces"] = []

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
                "Radius": f"{surface.nominal_radius.value:.7g} {surface.positive_tolerance.value:+.4g}/{-surface.negative_tolerance.value:+.4g} {surface.nominal_radius.unit}",
                "Fringes": f"{surface.fringes:.0f} fringes @ λ = {surface.wavelength:.4g}",
                "Nominal sagitta": f"{surface.nominal_sagitta:.4g}",
            }
            for surface in st.session_state.get("surfaces", [])
        ]
    )

with cols[1]:
    if not df.empty:
        st.dataframe(df, hide_index=True, width="stretch", height=600)

        subcols = st.columns(2)

        with subcols[0], st.container(horizontal_alignment="center"):
            st.button(
                "Clear table",
                on_click=lambda: st.session_state.pop("surfaces", None),
                type="secondary",
                help="Clear all surface data. This cannot be undone.",
            )

        with subcols[1], st.container(horizontal_alignment="center"):
            st.download_button(
                "Download table as CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="surface_tolerances.csv",
                mime="text/csv",
            )
