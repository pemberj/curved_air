"""
Calculate the couping efficiency of various surfaces or beams into optical fibres.

Ideas:
- Gaussian beam into a single-mode fibre
- Blackbody into a multi-mode fibre of a particular core size (want to get the spectral
  flux)
- In general, a source with particular spatial, spectral, and angular properties into a
  fibre with particular core size and NA, get the spectral flux after coupling
- Coupling between fibres of different core sizes (and shapes?) and different NAs, with
  a particular separation and angular offset (including zero separation for
  butt-coupling)
-

"""

import numpy as np
import streamlit as st

from curved_air.type_hinting import ScalarQuantity

_ = np.inf
st.set_page_config(
    page_title="Curved Air: Coupling Efficiency",
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="expanded",
)


def geometric_coupling_efficiency(
    _fibre_shape: str,
    _fibre_size: ScalarQuantity | tuple[ScalarQuantity, ScalarQuantity],
    _fibre_NA: float,
    _beam_shape: str,
    _beam_size: ScalarQuantity | tuple[ScalarQuantity, ScalarQuantity],
    _beam_NA: float,
) -> float:
    """
    Calculate the geometric coupling efficiency of a beam into a fibre.
    """
    return 0
