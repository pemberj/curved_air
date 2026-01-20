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

SCRIPT_PATH: Path = Path(__file__).parent


def main() -> None:
    results: pd.DataFrame = pd.read_csv(
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

    selected: pd.DataFrame = results.query(
        "`PTV Residuals` <= @residuals_cutoff and abs(`Apex angle`) <= @apex_cutoff and abs(`Compound angle`) <= @compound_cutoff and `Zero-deviation wavelength` >= 0.38 and `Zero-deviation wavelength` <= 0.95"
    )

    selected_sorted: pd.DataFrame = selected.sort_values("PTV Residuals")
    selected_sorted.to_csv(SCRIPT_PATH / "ADCs_from_catalogue_selected.csv")

    rejected: pd.DataFrame = results.query(
        "`PTV Residuals` > @residuals_cutoff or abs(`Apex angle`) > @apex_cutoff or abs(`Compound angle`) > @compound_cutoff"
    )

    fig: plt.Figure = plt.figure("Angular residuals histograms", figsize=(12, 8))

    axs: dict[str, plt.Axes] = fig.subplot_mosaic(
        [
            ["main", "apex"],
            ["main", "compound"],
            ["main", "lambda_0"],
        ],
        width_ratios=[3, 2],
    )

    ax: plt.Axes = axs["main"]
    ax.hist(
        [selected["PTV Residuals"], rejected["PTV Residuals"]],
        bins=np.linspace(0, 100, 31).tolist(),
        edgecolor="k",
        alpha=0.75,
        label=[
            f"Candidates meeting criteria (n={len(selected)})",
            f"Rejected candidates (n={len(rejected)})",
        ],
        histtype="barstacked",
        align="mid",
    )
    ax.axvline(
        residuals_cutoff,
        color="r",
        linestyle="--",
        lw=3,
        label="Selection cutoffs",
    )
    ax.set_xlabel(
        "PTV angular residuals (milliarcseconds projected on sky)", fontsize=16
    )
    ax.set_ylabel("Number of candidate designs", fontsize=16)
    ax.set_xlim(0, 110)
    ax.legend(fontsize=14, loc="upper right", frameon=True)

    ax: plt.Axes = axs["apex"]
    ax.hist(
        [abs(selected["Apex angle"]), abs(rejected["Apex angle"])],
        bins=np.linspace(0, 20, 31).tolist(),
        edgecolor="k",
        alpha=0.75,
        histtype="barstacked",
        align="mid",
    )
    ax.axvline(
        apex_cutoff,
        color="r",
        linestyle="--",
        lw=3,
    )
    ax.set_xlim(0, 20)
    ax.set_xlabel("Apex angle (degrees)", fontsize=16)
    # ax.set_ylabel("Frequency")

    ax: plt.Axes = axs["compound"]
    ax.hist(
        [abs(selected["Compound angle"]), abs(rejected["Compound angle"])],
        bins=np.linspace(0, 1.5, 31).tolist(),
        edgecolor="k",
        alpha=0.75,
        histtype="barstacked",
        align="mid",
    )
    ax.axvline(
        compound_cutoff,
        color="r",
        linestyle="--",
        lw=3,
    )
    ax.set_xlim(0, 1.5)
    ax.set_xlabel("Compound angle (degrees)", fontsize=16)
    # ax.set_ylabel("Frequency")

    ax: plt.Axes = axs["lambda_0"]
    ax.hist(
        [selected["Zero-deviation wavelength"], rejected["Zero-deviation wavelength"]],
        bins=np.linspace(0.39, 1.5, 31).tolist(),
        edgecolor="k",
        alpha=0.75,
        histtype="barstacked",
        align="mid",
    )
    ax.axvline(
        0.38,
        color="r",
        linestyle="--",
        lw=3,
    )
    ax.axvline(
        0.95,
        color="r",
        linestyle="--",
        lw=3,
    )
    ax.set_xlim(0.35, 1.55)
    ax.set_yscale("log")
    ax.set_xlabel("Zero-deviation wavelength (µm)", fontsize=16)
    # ax.set_ylabel("Frequency")

    # for ext in ["svg", "png"]:
    #     fig.savefig(
    #         Path("/home/jake/Syncthing/thesis/chapters/06-marvel-telescope-units/image")
    #         / f"adc-histograms.{ext}"
    #     )

    plt.show()


if __name__ == "__main__":
    main()
