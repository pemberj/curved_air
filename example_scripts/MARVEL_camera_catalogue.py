# /// script
# dependencies = [
#     "curved_air>=0.1.0",
#     "rich>=10.0.0",
# ]
# [tool.uv.sources]
# curved_air = { path = "..", editable = true }
# ///

"""
A script used to filter input glass catalogues and output the filtered catalogue to a
new AGF file for use in optimising the MARVEL spectrograph camera design.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
from astropy import units as u
from rich import print  # noqa: A004

# from glass_select.glass import GlassCatalogue, _optimise_ADC
from curved_air.glass import GlassCatalogue
from curved_air.plotting import plot_style

plt.style.use(plot_style)


def main() -> None:
    script_path = Path(__file__).parent
    agf_dir = script_path / "../data/AGFs"

    MARVEL_gc = GlassCatalogue().read_AGF(agf_filename=agf_dir / "MARVEL_CAMERA.AGF")

    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    colours = {
        "schott": colours[0],
        "ohara": colours[1],
        "corning": colours[2],
        "nikon": colours[3],
    }

    fig = plt.figure()
    ax = fig.gca()

    print("original_file,name,nd,Vd")
    for glass in sorted(MARVEL_gc.glasses, key=lambda g: g.nd):
        print(f"{glass.original_file.split('.')[0]},{glass.name},{glass.nd},{glass.Vd}")
        ax.plot(
            glass.Vd, glass.nd, "o", color=colours[glass.manufacturer], label="MARVEL"
        )

    return

    print(agf_dir)
    print(agf_dir.glob("*.[Aa][Gg][Ff]"))

    agf_files: list[Path] = sorted(
        [
            f
            for f in agf_dir.glob("*.[Aa][Gg][Ff]")
            if re.search(pattern="ohara|schott", string=f.name)
        ]
    )

    if not agf_files:
        raise FileNotFoundError("No AGF files found in the data directory.")

    gc: GlassCatalogue = GlassCatalogue().read_AGF(agf_filename=agf_files)

    gc = gc.filter(
        # status_description=["standard", "preferred"],
        min_wl=390 * u.nm,
        max_wl=950 * u.nm,
        min_transmission=0.985,
        min_average_transmission=0.99,
    )

    # gc.plot_Abbe_diagram()
    # gc.plot_transmission(wls=np.linspace(0.39, 0.45, 100) * u.um)

    # print(gc)

    print("original_file,name,nd,Vd")
    for glass in sorted(gc.glasses, key=lambda g: g.nd):
        print(f"{glass.original_file.split('.')[0]},{glass.name},{glass.nd},{glass.Vd}")

    # gc.write_AGF("./OUTPUT_CATALOGUE.AGF")


if __name__ == "__main__":
    main()
