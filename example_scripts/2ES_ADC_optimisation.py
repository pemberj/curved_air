# /// script
# dependencies = [
#     "curved_air>=0.1.0",
#     "pandas",
# ]
# [tool.uv.sources]
# curved_air = { path = "..", editable = true }
# ///

"""
- Create an input catalogue of glasses from AGF files
- Filter the catalogue for glasses with a minimum transmission of 98% and an average
  transmission of at least 99% over the visible range
- Iterate over all unique glass combinations in the catalogue, optimise an ADC for each,
  and list any that have angular residuals less than 20 milliarcseconds
"""

from pathlib import Path

import astropy.units as u
from matplotlib import pyplot as plt
from pandas import DataFrame

from curved_air.glass import GlassCatalogue
from curved_air.plotting import plot_style

plt.style.use(plot_style)


def main() -> None:
    script_path: Path = Path(__file__).parent

    catalogue: GlassCatalogue = GlassCatalogue(agf=["Schott", "Ohara"])

    catalogue: GlassCatalogue = catalogue.filter(
        min_wl=370 * u.nm,
        max_wl=900 * u.nm,
        min_transmission=0.98,
        min_average_transmission=0.99,
        # status_description="preferred",
    )

    n: int = len(catalogue)

    print(f"Number of glasses in catalogue: {n}")
    print(f"Number of unique glass combinations: {int((n * (n - 1)) / 2)}")

    # catalogue.plot_Abbe_diagram()

    # return

    _ADCs, results = catalogue.compute_ADCs(
        D_telescope=2.2 * u.m,
        max_residuals=20 * u.mas,
        num_samples=50,
    )

    results_df = DataFrame(results)
    results_df.to_csv(script_path / "2ES_ADCs.csv")


if __name__ == "__main__":
    main()
