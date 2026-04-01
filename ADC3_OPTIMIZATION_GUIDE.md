# Three-Element ADC (ADC3) Optimization

## Overview

The `ADC3` class in `curved_air.instruments.ADC3` provides optimization of apex angles for three-element compound prisms to match the dispersion of air. This extends the two-element optimization implemented in the `ADC` class.

## Physics Background

### Air Dispersion Compensation

Differential atmospheric refraction (DAR) causes point sources to elongate as a function of wavelength due to air's wavelength-dependent refractive index. An Atmospheric Dispersion Corrector (ADC) uses prisms to introduce compensating dispersion.

For a three-element system:
- **Element 1**: Primary disperser (typically positive apex angle)
- **Element 2**: Secondary disperser (typically negative apex angle)
- **Element 3**: Fine-tuning element (can be positive or negative)

### Optimization Objective

The optimizer adjusts three apex angles to minimize the residual between:
- **ADC dispersion**: computed from `PrismStack.trace()`
- **Air dispersion**: from atmospheric refraction model

At a specified zero-deviation wavelength (wl_0), both should be equal.

## Implementation Details

### Parameters Optimized (5 parameters)

```
x = [wl_0, alpha1, alpha2, alpha3, incidence_angle]
```

1. **wl_0**: Zero-deviation wavelength (μm)
   - Range: 0.39 - 0.95 μm (visible spectrum)
   - Reference wavelength where ADC has zero deviation

2. **alpha1**: First prism apex angle (degrees)
   - Range: -20° to +20°
   - Typically positive for primary focusing

3. **alpha2**: Second prism apex angle (degrees)
   - Range: -20° to +20°
   - Typically negative to oppose first prism

4. **alpha3**: Third prism apex angle (degrees)
   - Range: -20° to +20°
   - Fine-tuning element

5. **incidence_angle**: Angle of incidence on first prism (degrees)
   - Range: -10° to +10°
   - Affects effective dispersion relationship

### Optimization Algorithm

- **Method**: L-BFGS-B (Quasi-Newton with bounds)
- **Residual Function**: User-selectable criterion:
  - `PTV` (default): Peak-to-valley error
  - `RMS`: Root mean square error
  - `RSS`: Root sum square error
  - `max_abs`: Maximum absolute deviation

### Ray Tracing

The effective deviation angle is computed using `PrismStack.trace()`:

```python
delta_ADC = prism_stack.trace(
    wls=wls,
    incidence_angle=incidence_angle,
)
```

This traces rays through all three prisms and accounts for refraction at each interface.

### Air Dispersion Model

```python
delta_air = (D_telescope / D_ADC) * z_max * (
    n_air(wls) - n_air(wl_0)
)
```

Where:
- `D_telescope / D_ADC`: Beam compression ratio
- `z_max`: Maximum zenith angle (typically 65°)
- `n_air(wls)`: Refractive index of air at wavelengths

## Usage

### Basic Usage

```python
from curved_air.instruments.ADC3 import ADC3
from curved_air.instruments.ADC import PrismStack
from curved_air.optics.dispersers import Prism
from curved_air.glass import GlassCatalogue
from astropy import units as u

# Load glasses
catalogue = GlassCatalogue.populate_from_schott()
glass1 = catalogue["SF67"]
glass2 = catalogue["NSF6"]
glass3 = catalogue["PSK3"]

# Create prism stack
prism_stack = PrismStack(
    prism_list=[
        Prism(glass=glass1, apex_angle=10 * u.deg),
        Prism(glass=glass2, apex_angle=-5 * u.deg),
        Prism(glass=glass3, apex_angle=2 * u.deg),
    ]
)

# Create optimizer
adc = ADC3(
    prism_stack=prism_stack,
    D_telescope=0.8 * u.m,
    D_ADC=15 * u.mm,
    criterion="PTV",
)

# Run optimization
results = adc.optimise(plot=True, num_samples=30)

# Access results
print(f"Zero-deviation wavelength: {adc.wl_0}")
print(f"Apex angles: {adc.apex_angles}")
print(f"Residuals (PTV): {adc.residuals_PTV}")
```

### Optimization Configuration

```python
adc = ADC3(
    prism_stack=prism_stack,

    # Wavelength range
    min_wl=390 * u.nm,  # Blue edge (visible spectrum)
    max_wl=950 * u.nm,  # Red edge (visible spectrum)

    # Telescope parameters
    D_telescope=0.8 * u.m,      # Primary mirror diameter
    D_ADC=15 * u.mm,            # ADC aperture diameter
    z_max=65 * u.degree,        # Maximum zenith angle

    # Optimization criterion
    criterion="PTV",  # "PTV", "RMS", "RSS", "max_abs"
)
```

### Running Optimization

```python
results = adc.optimise(
    plot=True,          # Generate visualization
    num_samples=20,     # Wavelength samples for optimization
)
```

### Results

The `optimise()` method returns a dictionary with:

```python
{
    "g1", "g2", "g3": Glass objects used,
    "apex_angles": [α1, α2, α3]  # Optimized angles
    "incidence_angle": Incidence angle,
    "wl_0": Zero-deviation wavelength,
    "residuals_rms": RMS residual (mas),
    "residuals_rss": RSS residual (mas),
    "residuals_ptv": PTV residual (mas),
    "optimization_success": bool,
    "optimization_message": str,
}
```

### Accessing Results

```python
# Optimized parameters
adc.wl_0                # Zero-deviation wavelength
adc.apex_angles         # List of 3 apex angles
adc.incidence_angle     # Incidence angle

# Residual metrics
adc.residuals_RMS       # Root mean square (mas)
adc.residuals_RSS       # Root sum square (mas)
adc.residuals_PTV       # Peak-to-valley (mas)

# Residuals at specific wavelengths
res = adc.residuals_function(wls)  # Callable
```

## Visualization

The `plot()` method generates three subplots:

```python
adc.plot(
    num_samples=100,        # Wavelength samples for plotting
    save_filename=None,     # Save to file if specified
    show=True,              # Display plot
)
```

**Plot 1**: Refractive indices of the three glasses across the wavelength range

**Plot 2**: Effective ADC dispersion vs. air dispersion
- Shows how well the optimized ADC matches air refraction
- Curves should be close for good compensation

**Plot 3**: Residuals (ADC - air) as function of wavelength
- Ideally flat and near zero
- Shows remaining errors after optimization

## Comparison to Two-Element ADC

### Two-Element (ADC class)

**Parameters optimized**: 2
- `alpha`: Combined apex angle for both prisms
- `wl_0`: Zero-deviation wavelength

**Constraint**: Analytical relationship between prism angles via epsilon ratio
```python
epsilon = (n1 - 1) / (n2 - 1)
alpha2 = epsilon * alpha1
```

**Advantage**: Simpler, fewer parameters, converges faster

**Limitation**: Less flexible, may not achieve good compensation for some glass pairs

### Three-Element (ADC3 class)

**Parameters optimized**: 5
- `alpha1`, `alpha2`, `alpha3`: Individual apex angles
- `wl_0`: Zero-deviation wavelength
- `incidence_angle`: Ray incidence angle

**Constraint**: None - full freedom for optimization

**Advantage**: More flexible, can achieve better residuals with appropriate glass selection

**Limitation**: More parameters to optimize, longer convergence time, more likely to find local minima

## Advanced Usage

### Custom Optimization Criteria

```python
# Try different criteria to find best for your application
for criterion in ["PTV", "RMS", "RSS", "max_abs"]:
    adc = ADC3(prism_stack=prism_stack, criterion=criterion)
    results = adc.optimise()
    print(f"{criterion}: PTV={adc.residuals_PTV}")
```

### Fine-Tuning Initial Guesses

If optimization doesn't converge well, you can help by providing better initial apex angles:

```python
# Edit optimise() method to modify x0:
x0 = [
    0.55,      # wl_0 closer to H-alpha
    15.0,      # higher alpha1
    -8.0,      # more negative alpha2
    4.0,       # stronger alpha3
    0.5,       # small positive incidence angle
]
```

### Batch Optimization Over Glass Combinations

```python
# Try multiple glass combinations
best_adc = None
best_residual = float('inf')

for g1_name in catalogue:
    for g2_name in catalogue:
        for g3_name in catalogue:
            if len(set([g1_name, g2_name, g3_name])) < 3:
                continue  # Skip if same glass used twice

            ps = PrismStack([
                Prism(glass=catalogue[g1_name], apex_angle=10*u.deg),
                Prism(glass=catalogue[g2_name], apex_angle=-5*u.deg),
                Prism(glass=catalogue[g3_name], apex_angle=2*u.deg),
            ])

            adc = ADC3(prism_stack=ps)
            results = adc.optimise()

            if adc.residuals_PTV < best_residual:
                best_residual = adc.residuals_PTV
                best_adc = adc
```

## References

- **Two-element ADC**: [ADC.py](src/curved_air/instruments/ADC.py)
- **Multi-element optimization**: [multi_optimisation.py](src/curved_air/multi_optimisation.py)
- **Prism optics**: [dispersers.py](src/curved_air/optics/dispersers.py)
- **Example script**: [ADC3_optimization_example.py](example_scripts/ADC3_optimization_example.py)

## Troubleshooting

### Optimization doesn't converge

- Try different initial guesses (modify `x0` in `optimise()`)
- Increase `num_samples` for smoother residual surface
- Check that glass combination has sufficient dispersion diversity
- Consider using 2-element ADC instead if simpler

### Residuals remain large

- Check that glasses have appropriate refractive index curves
- Verify that `z_max` and aperture sizes are realistic
- Consider that some glass combinations cannot achieve very small residuals
- Try different optimization criteria (`PTV` vs `RMS`)

### Physically unrealistic apex angles

- Add tighter bounds in `bounds` parameter
- Use glasses with more extreme refractive indices
- Accept that some wavelength ranges may have higher residuals
