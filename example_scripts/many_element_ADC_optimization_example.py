"""
Example script demonstrating three-element ADC optimization.

This script shows how to use the ADC3 class to optimize apex angles
for a three-element compound prism to match air dispersion.
"""

from itertools import permutations

import tqdm
from astropy import units as u
from matplotlib import pyplot as plt
from rich import print  # noqa: A004

from curved_air.glass import Glass, GlassCatalogue
from curved_air.instruments.ADC import ADC, PrismStack
from curved_air.optics.dispersers import Prism
from curved_air.other import Air

ZEMAX_AIR = Air(air_model="zemax")


def example_three_element_adc() -> tuple[ADC, dict] | None:
    """
    Example: Optimize a three-element prism ADC using three glasses from a catalogue.
    """

    gc: GlassCatalogue = GlassCatalogue(["Schott", "Ohara"])

    adc = ADC(
        prism_stack=PrismStack(
            # prism_list=[
            #     Prism(glass=gc["K10"]),
            #     Prism(glass=gc["S-FPM4"]),
            #     Prism(glass=gc["S-FPL51"]),
            # ]
            prism_list=[
                Prism(glass=gc["K10"]),
                Prism(glass=gc["S-FPL51"]),
                Prism(glass=gc["N-BAK1"]),
            ]
        ),
        air_model=ZEMAX_AIR,
        min_wl=370 * u.nm,
        max_wl=900 * u.nm,
        D_telescope=2.2 * u.m,
        D_ADC=15 * u.mm,
        z_max=65 * u.degree,
        criterion="max_abs",  # Can use "RMS", "RSS", "PTV", or "max_abs"
        debug_level=0,
    )

    # Run optimization
    print("Starting three-element ADC optimization...")
    results = adc.optimise(plot=True, num_samples=12)

    assert adc.apex_angles is not None, "Optimization failed to find apex angles."

    if results is None:
        print("Optimization failed!")
        return None

    # Display results
    print("\n" + "=" * 60)
    print("OPTIMISATION RESULTS")
    print("=" * 60)
    print()
    print(f"Zero-deviation wavelength: {adc.wl_0:.3f}")
    for i, prism in enumerate(adc.prism_stack.prism_list, 1):
        print(f"  Prism {i}: {prism.glass}, {prism.apex_angle:.4f}")
    print()
    print("Residual metrics:")
    print(f"  RMS: {adc.residuals_RMS:.3f}")
    print(f"  PTV: {adc.residuals_PTV:.3f}")

    plt.show()

    return adc, results


def example_third_material_search() -> ADC | None:
    """
    Example: Staged materials search for a three-element ADC.
    """
    # Load glass catalogue
    gc: GlassCatalogue = GlassCatalogue(["Schott", "Ohara"]).filter(
        min_wl=370 * u.nm,
        max_wl=900 * u.nm,
        min_transmission=0.98,
    )

    baseline_adc = ADC(
        prism_stack=PrismStack(
            prism_list=[
                Prism(glass=gc["K10"], apex_angle=2.8 * u.deg),
                Prism(glass=gc["S-FPL51"], apex_angle=-2.8 * u.deg),
            ]
        ),
        air_model=ZEMAX_AIR,
        min_wl=370 * u.nm,
        max_wl=900 * u.nm,
        D_telescope=2.2 * u.m,
        D_ADC=15 * u.mm,
        z_max=65 * u.degree,
        criterion="max_abs",
        debug_level=0,
    )

    results = baseline_adc.optimise(plot=False, num_samples=30)

    if results is None:
        raise RuntimeError("Baseline ADC optimization failed.")

    print("=== Baseline ADC ===")
    print(f"  Zero-deviation wavelength: {baseline_adc.wl_0:.3f}")
    for i, p in enumerate(baseline_adc.prism_stack.prism_list, start=1):
        print(f"    Prism {i}: {p.glass}, {p.apex_angle:.4f}")
    print("  Residuals:")
    print(f"    RMS: {baseline_adc.residuals_RMS:.3f}")
    print(f"    PTV: {baseline_adc.residuals_PTV:.3f}")
    print()

    current_best: ADC = baseline_adc

    # Try all glasses in the catalogue as the third prism material
    for g in tqdm.tqdm(gc, unit="glass", ncols=120):
        adc = ADC(
            prism_stack=PrismStack(
                prism_list=[
                    Prism(glass=gc["K10"], apex_angle=2.8 * u.deg),
                    Prism(glass=g, apex_angle=0 * u.deg),
                    Prism(glass=gc["S-FPL51"], apex_angle=-2.8 * u.deg),
                ]
            ),
            air_model=ZEMAX_AIR,
            min_wl=370 * u.nm,
            max_wl=900 * u.nm,
            D_telescope=2.2 * u.m,
            D_ADC=15 * u.mm,
            z_max=65 * u.degree,
            criterion="max_abs",
            debug_level=0,
        )

        results = adc.optimise(plot=False, num_samples=12)

        if results is None:
            continue
            # print(f"Optimisation failed for *{g}*.")
            # print()

        if adc.residuals_max_abs < baseline_adc.residuals_max_abs:
            print()
            print(f"=== {g} ===")
            print(f"  Zero-deviation wavelength: {adc.wl_0:.3f}")
            for i, p in enumerate(adc.prism_stack.prism_list, start=1):
                print(f"    Prism {i}: {p.glass}, {p.apex_angle:.4f}")
            print("  Residuals:")
            print(f"    RMS: {adc.residuals_RMS:.3f}")
            print(f"    PTV: {adc.residuals_PTV:.3f}")
            print()

        if adc.residuals_max_abs < current_best.residuals_max_abs:
            current_best: ADC = adc

    if current_best is baseline_adc:
        print("No better ADC found than the baseline.")
        return current_best

    print("=== Best ADC ===")
    print(f"  Zero-deviation wavelength: {current_best.wl_0:.3f}")
    for i, p in enumerate(current_best.prism_stack.prism_list, start=1):
        print(f"    Prism {i}: {p.glass}, {p.apex_angle:.4f}")
    print("  Residuals:")
    print(f"    RMS: {current_best.residuals_RMS:.3f}")
    print(f"    PTV: {current_best.residuals_PTV:.3f}")
    print()

    current_best.plot()
    plt.show()

    return current_best


def example_exhaustive_search() -> ADC | None:
    """
    Example: Exhaustive search for a three-element ADC using all possible combinations
    (and orders) of glasses in a (small) input catalogue.
    """
    # Load glass catalogue
    gc: GlassCatalogue = GlassCatalogue(["Schott", "Ohara"])

    glasses_to_use: list[Glass] = [
        gc["K10"],
        gc["N-BK7"],
        gc["F2"],
        gc["LLF1"],
        gc["N-BAK1"],
        gc["N-BAK2"],
        gc["N-FK5"],
        gc["PBL6Y"],
        # gc["PBL35Y"],
        gc["S-FSL5"],
        gc["S-FPL51"],
        # gc["S-FPL53"],
        # gc["S-FPL55"],
        gc["S-FPM2"],
        gc["S-FPM4"],
        gc["S-LAL12"],
        gc["S-BAL12"],
    ]

    baseline_adc = ADC(
        prism_stack=PrismStack(
            prism_list=[Prism(glass=gc["K10"]), Prism(glass=gc["S-FPL51"])]
        ),
        air_model=ZEMAX_AIR,
        min_wl=370 * u.nm,
        max_wl=900 * u.nm,
        D_telescope=2.2 * u.m,
        D_ADC=15 * u.mm,
        z_max=65 * u.degree,
        criterion="max_abs",
        debug_level=0,
    )

    results: dict | None = baseline_adc.optimise(plot=False, num_samples=30)

    if results is None:
        raise RuntimeError("Baseline ADC optimization failed.")

    print("  ==== Baseline ADC ====")
    print(f"  Zero-deviation wavelength: {baseline_adc.wl_0:.3f}")
    for i, p in enumerate(baseline_adc.prism_stack.prism_list, start=1):
        print(f"    Prism {i}: {p.glass}, {p.apex_angle:.4f}")
    print("  Residuals:")
    print(f"    RMS: {baseline_adc.residuals_RMS:.3f}")
    print(f"    PTV: {baseline_adc.residuals_PTV:.3f}")
    print()

    current_best: ADC = baseline_adc

    permutations_of_glasses: list[tuple[Glass, Glass, Glass]] = list(
        permutations(glasses_to_use, 3)
    )

    # Try all glasses in the catalogue as the third prism material
    for p in tqdm.tqdm(permutations_of_glasses, unit="permutation", ncols=120):
        glasses = f"{p[0].name} / {p[1].name} / {p[2].name}"

        # print(glasses)
        # continue

        adc = ADC(
            prism_stack=PrismStack(
                prism_list=[
                    Prism(glass=p[0]),
                    Prism(glass=p[1]),
                    Prism(glass=p[2]),
                ]
            ),
            air_model=ZEMAX_AIR,
            min_wl=370 * u.nm,
            max_wl=900 * u.nm,
            D_telescope=2.2 * u.m,
            D_ADC=15 * u.mm,
            z_max=65 * u.degree,
            criterion="max_abs",
            debug_level=0,
        )

        results = adc.optimise(plot=False, num_samples=12)

        if results is None:
            continue
            # print(f"Optimisation failed for *{g}*.")
            # print()

        if adc.residuals_PTV < baseline_adc.residuals_PTV:
            print()
            print(f"  ==== {glasses} ====")
            print(f"  Zero-deviation wavelength: {adc.wl_0:.3f}")
            for i, prism in enumerate(adc.prism_stack.prism_list, start=1):
                print(f"    Prism {i}: {prism.glass}, {prism.apex_angle:.4f}")
            print("  Residuals:")
            print(f"    RMS: {adc.residuals_RMS:.3f}")
            print(f"    PTV: {adc.residuals_PTV:.3f}")
            print()

        if adc.residuals_PTV < current_best.residuals_PTV:
            current_best: ADC = adc

    print("  ==== Best ADC ====")
    print(f"  Zero-deviation wavelength: {current_best.wl_0:.3f}")
    for i, p in enumerate(current_best.prism_stack.prism_list, start=1):
        print(f"    Prism {i}: {p.glass}, {p.apex_angle:.4f}")
    print("  Residuals:")
    print(f"    RMS: {current_best.residuals_RMS:.3f}")
    print(f"    PTV: {current_best.residuals_PTV:.3f}")
    print()

    current_best.plot(layout=False)
    plt.show()

    return current_best


if __name__ == "__main__":
    example_three_element_adc()
    # example_third_material_search()
    # example_exhaustive_search()
