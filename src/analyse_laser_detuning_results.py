from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# =============================================================================
# File names
# =============================================================================

base_folder = Path(r"Y:\KiranPhalke\NMOR_sensor_characterization\sweep_test_with_power_cells_separated\laser_detuning")
cell1_csv = base_folder / "lorentzian_fit_results_cell1.csv"
cell2_csv = base_folder / "lorentzian_fit_results_cell2.csv"

cols_to_exclude = ["chunk_label", "adjusted_parameter"]

output_dir = base_folder / "plots"
output_dir.mkdir(exist_ok=True)

# =============================================================================
# Load data
# =============================================================================

cell1 = pd.read_csv(cell1_csv)
cell2 = pd.read_csv(cell2_csv)

# =============================================================================
# Sort by laser detuning
# =============================================================================

x_col = "adjusted_parameter"  # laser detuning in V

cell1 = cell1.sort_values(x_col)
cell2 = cell2.sort_values(x_col)

# =============================================================================
# Remove entries whose fit was not good ("y_r_squared" > 0.95)
# =============================================================================
cell1 = cell1[cell1["y_r_squared"] > 0.95]
cell2 = cell2[cell2["y_r_squared"] > 0.95]

# =============================================================================
# Generate plots
# =============================================================================

for col in cell1.columns:
    if col in cols_to_exclude:
        continue

    fig, ax = plt.subplots(figsize=(8, 5))

    # -------------------------------------------------------------------------
    # Cell 1
    # -------------------------------------------------------------------------

    ax.plot(
        cell1[x_col],
        cell1[col],
        "o-",
        color="tab:blue",
        linewidth=1.5,
        markersize=5,
        label="Cell 1",
    )

    # -------------------------------------------------------------------------
    # Cell 2
    # -------------------------------------------------------------------------

    ax.plot(
        cell2[x_col],
        cell2[col],
        "s-",
        color="tab:red",
        linewidth=1.5,
        markersize=5,
        label="Cell 2",
    )

    ax.set_xlabel("Laser Detuning (V)")
    ax.set_ylabel(col)
    ax.set_title(f"{col} vs Laser Detuning")

    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    filename = output_dir / f"{col}_vs_detuning.png"
    plt.savefig(filename, dpi=300)
    plt.close()

    print(f"Saved: {filename}")

print("\nFinished generating detuning plots.")
