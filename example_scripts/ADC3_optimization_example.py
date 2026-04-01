"""
Example script demonstrating three-element ADC optimization.

This script shows how to use the ADC3 class to optimize apex angles
for a three-element compound prism to match air dispersion.
"""

from astropy import units as u

from curved_air.glass import Glass, GlassCatalogue
from curved_air.instruments.ADC import PrismStack
from curved_air.instruments.ADC3 import ADC3
from curved_air.optics.dispersers import Prism


def example_three_element_adc() -> tuple[ADC3, dict] | None:
    """
    Example: Optimize a three-element prism ADC using three glasses from a catalogue.
    """
    # Load glass catalogue
    catalogue = GlassCatalogue("Schott")

    # Select three different glasses for the prism stack
    # You can choose any glasses from the catalogue
    glass1: Glass = catalogue["K10"]  # First prism glass
    glass2: Glass = catalogue["F2"]  # Second prism glass
    glass3: Glass = catalogue["N-PSK3"]  # Third prism glass

    # Create initial prism stack with dummy apex angles
    # These will be optimized later
    prism_stack = PrismStack(
        prism_list=[
            Prism(glass=glass1, apex_angle=10 * u.deg),
            Prism(glass=glass2, apex_angle=-5 * u.deg),
            Prism(glass=glass3, apex_angle=2 * u.deg),
        ]
    )

    # Create ADC3 optimizer
    adc = ADC3(
        prism_stack=prism_stack,
        min_wl=390 * u.nm,
        max_wl=950 * u.nm,
        D_telescope=0.8 * u.m,
        D_ADC=15 * u.mm,
        z_max=65 * u.degree,
        criterion="PTV",  # Can also use "RMS", "RSS", "max_abs"
    )

    print(ADC3.prism_stack)
    return None

    # Run optimization
    print("Starting three-element ADC optimization...")
    results = adc.optimise(plot=True, num_samples=30)

    assert adc.apex_angles is not None, "Optimization failed to find apex angles."

    if results is None:
        print("Optimization failed!")
        return None

    # Display results
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)
    print(f"Zero-deviation wavelength: {adc.wl_0:.3f}")
    print(f"Incidence angle: {adc.incidence_angle:.3f}")
    print("\nOptimized apex angles:")
    for i, angle in enumerate(adc.apex_angles, 1):
        print(f"  Prism {i}: {angle:.4f}")

    print("\nResidual metrics:")
    print(f"  RMS: {adc.residuals_RMS:.3f}")
    print(f"  RSS: {adc.residuals_RSS:.3f}")
    print(f"  PTV: {adc.residuals_PTV:.3f}")

    print(f"\nOptimization status: {results['optimization_success']}")
    if not results["optimization_success"]:
        print(f"  Message: {results['optimization_message']}")

    return adc, results


def understand_parameters() -> None:
    """
    Explanation of key parameters in ADC3 optimization.
    """
    print("""
    ADC3 Optimization Parameters:

    1. wl_0 (zero-deviation wavelength):
       - The wavelength at which the ADC produces zero deviation
       - Typically set to a reference wavelength (e.g., H-alpha or middle range)
       - Optimized range: 390-950 nm (visible spectrum)

    2. apex_angles (list of 3 angles):
       - The prism angles that determine dispersion
       - Positive angles: increase dispersion
       - Negative angles: decrease dispersion
       - Optimization range: -20° to +20°

    3. incidence_angle:
       - The angle at which light enters the first prism
       - Affects the coupling efficiency and dispersion relationship
       - Optimization range: -10° to +10°

    4. Criterion (optimization objective):
       - PTV: Peak-to-Valley (minimax) error
       - RMS: Root Mean Square error
       - RSS: Root Sum Square error
       - max_abs: Maximum absolute deviation

    The optimizer adjusts these parameters to minimize the difference
    between the ADC's dispersion curve and the air dispersion curve
    across the wavelength range 390-950 nm.
    """)


if __name__ == "__main__":
    example_three_element_adc()
