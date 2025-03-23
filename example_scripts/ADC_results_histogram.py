# /// script
# dependencies = [
#     "curved_air>=0.1.0",
#     "pandas",
# ]
# [tool.uv.sources]
# curved_air = { path = "..", editable = true }
# ///

from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from curved_air.plotting import plot_style

plt.style.use(plot_style)

SCRIPT_PATH = Path(__file__).parent


def main() -> None:
    results = pd.read_csv(
        SCRIPT_PATH / "ADCs_from_catalogue.csv",
        dtype={
            "Glass 1": str,
            "Glass 2": str,
            "Apex angle": float,
            "Compound angle": float,
            "Zero-deviation wavelength": float,
            "PTV Residuals": float,
        },
        index_col=0,
    )

    # sorted_results = results.sort_values("PTV Residuals")
    # sorted_results.to_csv(SCRIPT_PATH / "ADCs_from_catalogue_sorted.csv")

    residuals_cutoff = 40  # noqa: F841
    apex_cutoff = 10  # noqa: F841
    compound_cutoff = 1  # noqa: F841

    selected = results.query(
        "`PTV Residuals` <= @residuals_cutoff and abs(`Apex angle`) <= @apex_cutoff and abs(`Compound angle`) <= @compound_cutoff"
    )

    selected_sorted = selected.sort_values("PTV Residuals")
    selected_sorted.to_csv(SCRIPT_PATH / "ADCs_from_catalogue_selected.csv")

    rejected = results.query(
        "`PTV Residuals` > @residuals_cutoff or abs(`Apex angle`) > @apex_cutoff or abs(`Compound angle`) > @compound_cutoff"
    )

    fig = plt.figure("Angular residuals histogram")
    ax = fig.gca()
    ax.hist(
        [selected["PTV Residuals"], rejected["PTV Residuals"]],
        bins=np.linspace(0, 100, 51),
        edgecolor="k",
        alpha=0.75,
        label=[
            f"PTV residuals$\\leq${residuals_cutoff} mas",
            f"Rejected, PTV$>${residuals_cutoff} mas",
        ],
        histtype="barstacked",
    )
    ax.set_xlabel("Angular residuals [milliarcseconds]")
    ax.set_ylabel("Frequency")
    ax.set_xlim(0, 110)
    ax.legend()

    fig, axs = plt.subplots(3, 1, figsize=(8, 8))

    ax = axs[0]
    ax.hist(
        [abs(selected["Apex angle"]), abs(rejected["Apex angle"])],
        bins=np.linspace(0, 25, 26),
        edgecolor="k",
        alpha=0.75,
        label=[
            f"PTV residuals$\\leq${residuals_cutoff} mas",
            f"Rejected, PTV$>${residuals_cutoff} mas",
        ],
        histtype="barstacked",
    )
    ax.set_xlabel("Apex angle [degrees]")

    ax = axs[1]
    ax.hist(
        [abs(selected["Compound angle"]), abs(rejected["Compound angle"])],
        bins=np.linspace(0, 3, 31),
        edgecolor="k",
        alpha=0.75,
        label=[
            f"PTV residuals$\\leq${residuals_cutoff} mas",
            f"Rejected, PTV$>${residuals_cutoff} mas",
        ],
        histtype="barstacked",
    )
    ax.set_xlabel("Compound angle [degrees]")

    ax = axs[2]
    ax.hist(
        [selected["Zero-deviation wavelength"], rejected["Zero-deviation wavelength"]],
        bins=np.linspace(0.39, 1.6, 26),
        edgecolor="k",
        alpha=0.75,
        label=[
            f"PTV residuals$\\leq${residuals_cutoff} mas",
            f"Rejected, PTV$>${residuals_cutoff} mas",
        ],
        histtype="barstacked",
    )
    ax.set_xlabel("Zero-deviation wavelength [microns]")
    ax.set_yscale("log")

    for ax in axs:
        ax.set_ylabel("Frequency")
        # ax.legend()

    plt.show()


if __name__ == "__main__":
    main()
