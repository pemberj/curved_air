# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "curved-air>=0.1.0",
#     "rich>=10.0.0",
# ]
# [tool.uv.sources]
# curved_air = { path = "..", editable = true }
# ///

from curved_air.optics import Grating  # noqa: TC001, F401
from curved_air.utils import richardson_gratings
from rich import print  # noqa: A004


def test() -> None:
    gratings: list[Grating] = richardson_gratings(
        plot=True,
        selection_query="Ruled_width>=400 and Grooves_per_mm<=110",
    )

    for g in gratings:
        g.sign_of_blaze = -1
        print(f"{g}")
        # print(f"Aspect ratio: {g.aspect_ratio()}")
        # grid = order_grid(grating=g)
        # print(grid)


if __name__ == "__main__":
    test()
