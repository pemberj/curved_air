# /// script
# dependencies = [
#     "curved_air>=0.1.0",
#     "pandas",
# ]
# [tool.uv.sources]
# curved_air = { path = "..", editable = true }
# ///
import re
from pathlib import Path
from typing import TYPE_CHECKING

import astropy.units as u
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from curved_air.glass import GlassCatalogue
from curved_air.optimisation import constant_dispersion
from curved_air.plotting import plot_style

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from curved_air.glass import Glass
    from curved_air.type_hinting import ArrayQuantity

plt.style.use(plot_style)


def main() -> None:
    script_path: Path = Path(__file__).parent
    agf_dir: Path = script_path / "../data/AGFs"

    agf_files: list[Path] = sorted(
        [f for f in agf_dir.glob("*.[Aa][Gg][Ff]") if re.search("ohara|schott", f.name)]
    )

    if not agf_files:
        raise FileNotFoundError(f"No AGF files found in {agf_dir}")

    print(agf_files)

    catalogue: GlassCatalogue = GlassCatalogue().read_AGF(agf_filename=agf_files)

    catalogue: GlassCatalogue = catalogue.filter(
        min_wl=380 * u.nm,
        max_wl=950 * u.nm,
        min_transmission=0.97,
        min_average_transmission=0.99,
        # status_description="preferred",
    )

    results: pd.DataFrame = pd.DataFrame(constant_dispersion(catalogue))

    print(results.query("residuals <= 1e-8").sort_values("residuals").head(10))

    # return

    fig: plt.Figure = plt.figure()
    ax: plt.Axes = fig.gca()

    wavelengths: ArrayQuantity = np.linspace(380, 950, 30) * u.nm

    for _idx, r in (
        results.query("residuals <= 1e-8").sort_values("residuals").iterrows()
    ):
        g1: Glass = catalogue[r.loc["g1"]]
        g2: Glass = catalogue[r.loc["g2"]]

        curve: NDArray = np.power(
            (g1.n(wavelengths) - r.loc["k"] * g2.n(wavelengths))
            - (r.loc["m"] * wavelengths.value + r.loc["b"]),
            2,
        )
        ax.plot(wavelengths, curve)

    plt.show()


if __name__ == "__main__":
    main()
