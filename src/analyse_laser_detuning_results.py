import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# File names
# =============================================================================

base_folder = Path(r"Y:\KiranPhalke\NMOR_sensor_characterization\sweep_test_with_power_measurements\laser_detuning")
cell1_csv = base_folder / "lorentzian_fit_results_cell1.csv"
cell2_csv = base_folder / "lorentzian_fit_results_cell2.csv"

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

x_col = "l"

cell1 = cell1.sort_values(x_col)
cell2 = cell2.sort_values(x_col)

# =============================================================================
# Find mean/std parameter pairs automatically
# =============================================================================

mean_columns = [
    col for col in cell1.columns
    if col.endswith("_mean")
]

# =============================================================================
# Generate plots
# =============================================================================

for mean_col in mean_columns:

    std_col = mean_col.replace("_mean", "_std")

    if std_col not in cell1.columns:
        print(f"Skipping {mean_col}: no matching std column")
        continue

    fig, ax = plt.subplots(figsize=(8, 5))

    # -------------------------------------------------------------------------
    # Cell 1
    # -------------------------------------------------------------------------

    ax.errorbar(
        cell1[x_col],
        cell1[mean_col],
        yerr=cell1[std_col],
        fmt="o-",
        color="tab:blue",
        capsize=3,
        linewidth=1.5,
        markersize=5,
        label="Cell 1"
    )

    # -------------------------------------------------------------------------
    # Cell 2
    # -------------------------------------------------------------------------

    ax.errorbar(
        cell2[x_col],
        cell2[mean_col],
        yerr=cell2[std_col],
        fmt="s-",
        color="tab:red",
        capsize=3,
        linewidth=1.5,
        markersize=5,
        label="Cell 2"
    )

    ax.set_xlabel("Laser Detuning")
    ax.set_ylabel(mean_col.replace("_mean", ""))
    ax.set_title(
        f"{mean_col.replace('_mean', '')} vs Laser Detuning"
    )

    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    filename = output_dir / f"{mean_col}_vs_detuning.png"
    plt.savefig(filename, dpi=300)
    plt.close()

    print(f"Saved: {filename}")

print("\nFinished generating detuning plots.")
