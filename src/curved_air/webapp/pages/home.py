import streamlit as st
from streamlit.components.v1 import html

st.title("Curved Air: Optical Design Tools")

st.markdown(
    "Welcome to [**Curved Air**](https://curvedair.jakepember.com/). Use the sidebar to navigate between the available tools."
)

st.markdown(
    """
---
All code is written by Jake Pember: [Github](https://github.com/pemberj/) / [Email](mailto:jacob.pember@ikmail.com)

Please get in contact if you have any questions, suggestions, or find any bugs!

The web interface to these tools was built with [Streamlit](https://streamlit.io)

You can also [Buy me a pint](https://buymeacoffee.com/pemberj) if you find any of these tools useful and want to support their development!
"""
)

html(
    """
    <script type="text/javascript" src="https://cdnjs.buymeacoffee.com/1.0.0/button.prod.min.js" data-name="bmc-button" data-slug="pemberj" data-color="#FF5F5F" data-emoji="🍺" data-font="Cookie" data-text="Buy me a pint" data-outline-color="#000000" data-font-color="#ffffff"></script>
    """,
    height=70,
)
