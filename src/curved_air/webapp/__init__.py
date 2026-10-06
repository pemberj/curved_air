import streamlit as st

badges = """:gray-badge[:material/language: [Web](https://www.jakepember.com)]
    :gray-badge[:material/email: [Email](mailto:jacob.pember@ikmail.com)]
    :gray-badge[:material/code: [Github](https://www.github.com/pemberj)]
    """


def bmac():  # noqa
    return st.iframe(
        """
    <script type="text/javascript"
    src="https://cdnjs.buymeacoffee.com/1.0.0/button.prod.min.js"
    data-name="bmc-button" data-slug="pemberj" data-color="#E5660C"
    data-emoji="🍺" data-font="Cookie" data-text="Buy me a pint"
    data-outline-color="#000000" data-font-color="#ffffff"></script>
    """,
    )
