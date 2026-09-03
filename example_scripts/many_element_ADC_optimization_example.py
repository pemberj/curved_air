"""
Example script demonstrating three-element ADC optimization.

This script shows how to use the ADC3 class to optimize apex angles
for a three-element compound prism to match air dispersion.
"""

from astropy import units as u
from matplotlib import pyplot as plt
from rich import print  # noqa: A004

from curved_air.glass import GlassCatalogue
from curved_air.instruments.ADC import ADC, PrismStack
from curved_air.optics.dispersers import Prism


def example_three_element_adc() -> tuple[ADC, dict] | None:
    """
    Example: Optimize a three-element prism ADC using three glasses from a catalogue.
    """
    # Load glass catalogue
    gc = GlassCatalogue(["Schott", "Ohara"])

    # Create ADC3 optimizer
    adc = ADC(
        prism_stack=PrismStack(
            prism_list=[
                Prism(glass=gc["K10"], apex_angle=1 * u.deg),
                Prism(glass=gc["S-FPL51"], apex_angle=-1 * u.deg),
                Prism(glass=gc["N-BK7"], apex_angle=1 * u.deg),
                Prism(glass=gc["F2"], apex_angle=-1 * u.deg),
            ]
        ),
        min_wl=390 * u.nm,
        max_wl=950 * u.nm,
        D_telescope=0.8 * u.m,
        D_ADC=15 * u.mm,
        z_max=65 * u.degree,
        criterion="PTV",  # Can also use "RMS", "RSS", "max_abs"
        debug_level=0,
    )

    # Run optimization
    print("Starting three-element ADC optimization...")
    results = adc.optimise(plot=True, num_samples=12)

    assert adc.apex_angles is not None, "Optimization failed to find apex angles."

    if results is None:
        print("Optimization failed!")
        return None

    plt.show()

    # Display results
    print("\n" + "=" * 60)
    print("OPTIMISATION RESULTS")
    print("=" * 60)
    print()
    print(f"Zero-deviation wavelength: {adc.wl_0:.3f}")
    print()
    print("Optimized apex angles:")
    for i, angle in enumerate(adc.apex_angles, 1):
        print(f"  Prism {i}: {angle:.4f}")
    print()
    print("Residual metrics:")
    print(f"  RMS: {adc.residuals_RMS:.3f}")
    print(f"  RSS: {adc.residuals_RSS:.3f}")
    print(f"  PTV: {adc.residuals_PTV:.3f}")

    return adc, results


def example_third_material_search() -> None:
    """
    Example: Staged materials search for a three-element ADC.
    """
    # Load glass catalogue
    gc: GlassCatalogue = GlassCatalogue(["Schott", "Ohara"]).filter(
        min_wl=0.39 * u.um,
        max_wl=0.95 * u.um,
        min_transmission=0.98,
    )

    # Try all glasses in the catalogue as the third prism material
    for g in gc:
        adc = ADC(
            prism_stack=PrismStack(
                prism_list=[
                    Prism(glass=gc["K10"], apex_angle=2.8 * u.deg),
                    Prism(glass=g, apex_angle=0 * u.deg),
                    Prism(glass=gc["S-FPL51"], apex_angle=-2.8 * u.deg),
                ]
            ),
            min_wl=390 * u.nm,
            max_wl=950 * u.nm,
            D_telescope=0.8 * u.m,
            D_ADC=15 * u.mm,
            z_max=65 * u.degree,
            criterion="max_abs",
            debug_level=0,
        )

        results = adc.optimise(plot=False, num_samples=30)

        if results is None:
            print(f"Optimisation failed for *{g}*.")
            print()

        else:
            if adc.residuals_max_abs < 15 * u.mas:
                print(f"Optimisation successful for *{g}*.")
                print(f"  Zero-deviation wavelength: {adc.wl_0:.3f}")
                print("  Optimized apex angles:")
                for i, angle in enumerate(adc.apex_angles, 1):  # ty:ignore[invalid-argument-type]
                    print(f"    Prism {i}: {angle:.4f}")
                print("  Residuals:")
                print(f"    RMS: {adc.residuals_RMS:.3f}")
                # print(f"    RSS: {adc.residuals_RSS:.3f}")
                # print(f"    PTV: {adc.residuals_PTV:.3f}")
                print(f"    Max. Abs.: {adc.residuals_max_abs:.3f}")
                print()


if __name__ == "__main__":
    # example_three_element_adc()
    example_third_material_search()
