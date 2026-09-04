from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# =============================================================================
# File names
# =============================================================================

base_folder = Path(r"Y:\KiranPhalke\NMOR_sensor_characterization\sweep_test_05_after_offset_correction\biasfield_adjustment")
cell1_csv = base_folder / "lorentzian_fit_results_cell1.csv"
cell2_csv = base_folder / "lorentzian_fit_results_cell2.csv"

cols_to_include = ["absolute_y_slope", "amplitude_over_fwhm", "fwhm_hz", "y_r_squared", "amplitude"]

output_dir = base_folder / "plots_ratio_slope"
output_dir.mkdir(exist_ok=True)

# =============================================================================
# Load data
# =============================================================================

cell1 = pd.read_csv(cell1_csv)
cell2 = pd.read_csv(cell2_csv)

# =============================================================================
# Sort by bias field correction
# =============================================================================

x_col = "adjusted_parameter"  # bias field correction in uA

cell1 = cell1.sort_values(x_col)
cell2 = cell2.sort_values(x_col)

# multiple bias fields of cell1 by -1
cell1["adjusted_parameter"] = -cell1["adjusted_parameter"]

# =============================================================================
# Remove entries whose fit was not good ("y_r_squared" > 0.95)
# =============================================================================
cell1 = cell1[cell1["y_r_squared"] > 0.95]
cell2 = cell2[cell2["y_r_squared"] > 0.95]

# =============================================================================
# Generate plots
# =============================================================================

for col in cell1.columns:
    if col not in cols_to_include:
        continue

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        cell1[x_col],
        cell1[col],
        "o-",
        color="tab:blue",
        linewidth=1.5,
        markersize=5,
        label="Cell 1",
    )

    ax.plot(
        cell2[x_col],
        cell2[col],
        "s-",
        color="tab:red",
        linewidth=1.5,
        markersize=5,
        label="Cell 2",
    )

    ax.set_xlabel("Bias Field Correction (uA)")
    ax.set_ylabel(col)
    ax.set_title(f"{col} vs Bias Field Correction")

    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    filename = output_dir / f"{col}_vs_bias_field_correction.png"
    plt.savefig(filename, dpi=300)
    plt.close()

    print(f"Saved: {filename}")

print("\nFinished generating bias field correction plots.")
