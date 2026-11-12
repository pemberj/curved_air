import streamlit as st

from curved_air.webapp import badges, bmac

st.set_page_config(
    page_title="About Curved Air",
    layout="wide",
)

st.title("Curved Air: Optical Design Tools", anchor=False)

"""
Welcome to [**Curved Air**](https://curvedair.jakepember.com/). Use the sidebar to
navigate between the available tools.
"""

st.divider()

f"""
All code is written by Jake Pember {badges}

Please get in contact if you have any questions, suggestions, or find any bugs!

The web interface to these tools was built with [Streamlit](https://streamlit.io)

You can also [Buy me a pint](https://buymeacoffee.com/pemberj) if you find any of these
tools useful and want to support their development!
"""

bmac()
