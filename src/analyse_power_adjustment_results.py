from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# =============================================================================
# File names
# =============================================================================

base_folder = Path(r"Y:\KiranPhalke\NMOR_sensor_characterization\sweep_test_with_power_measurements\power_adjustment")
power_csv = base_folder / "laser_power_measurements.csv"
cell1_csv = base_folder / "lorentzian_fit_results_cell1.csv"
cell2_csv = base_folder / "lorentzian_fit_results_cell2.csv"

output_dir = base_folder / "plots"
output_dir.mkdir(exist_ok=True)

# =============================================================================
# Load data
# =============================================================================

power_df = pd.read_csv(power_csv)
cell1_df = pd.read_csv(cell1_csv)
cell2_df = pd.read_csv(cell2_csv)

# =============================================================================
# Merge power measurements with results
# =============================================================================

cell1 = pd.merge(
    cell1_df,
    power_df,
    on=["p1", "p2"],
    how="inner"
)

cell2 = pd.merge(
    cell2_df,
    power_df,
    on=["p1", "p2"],
    how="inner"
)

# =============================================================================
# Sort by fibre entry power
# =============================================================================

x_col = "power_fibre_entry_mW"

cell1 = cell1.sort_values(x_col)
cell2 = cell2.sort_values(x_col)

# =============================================================================
# Identify mean/std parameter pairs automatically
# =============================================================================

mean_columns = [
    col for col in cell1.columns
    if col.endswith("_mean")
]

# =============================================================================
# Plot all mean parameters with std error bars
# =============================================================================

for mean_col in mean_columns:

    std_col = mean_col.replace("_mean", "_std")

    # Skip if matching std column does not exist
    if std_col not in cell1.columns:
        print(f"Skipping {mean_col}: no matching std column.")
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
        capsize=3,
        color="tab:blue",
        label="Cell 1",
        alpha=0.9
    )

    # -------------------------------------------------------------------------
    # Cell 2
    # -------------------------------------------------------------------------

    ax.errorbar(
        cell2[x_col],
        cell2[mean_col],
        yerr=cell2[std_col],
        fmt="s-",
        capsize=3,
        color="tab:red",
        label="Cell 2",
        alpha=0.9
    )

    ax.set_xlabel("Power at Fibre Entry (mW)")
    ax.set_ylabel(mean_col.replace("_mean", ""))
    ax.set_title(
        f"{mean_col.replace('_mean','')} vs Fibre Entry Power"
    )

    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    filename = output_dir / f"{mean_col}_vs_fibre_entry_power.png"
    plt.savefig(filename, dpi=300)
    plt.close()

    print(f"Saved: {filename}")

print("\nFinished generating plots.")

# =============================================================================
# Power transfer plot
# =============================================================================

power_plot_df = power_df.sort_values("power_fibre_entry_mW")

fig, ax1 = plt.subplots(figsize=(8, 5))

# -------------------------------------------------------------------------
# Fibre exit power
# -------------------------------------------------------------------------

color1 = "tab:blue"

ax1.plot(
    power_plot_df["power_fibre_entry_mW"],
    power_plot_df["power_fibre_exit_mW"],
    "o-",
    color=color1,
    linewidth=2,
    label="Fibre Exit Power"
)

ax1.set_xlabel("Power at Fibre Entry (mW)")
ax1.set_ylabel("Power at Fibre Exit (mW)", color=color1)
ax1.tick_params(axis="y", labelcolor=color1)

# -------------------------------------------------------------------------
# Cell entry power
# -------------------------------------------------------------------------

ax2 = ax1.twinx()

color2 = "tab:red"

ax2.plot(
    power_plot_df["power_fibre_entry_mW"],
    power_plot_df["power_cell1_entry_uW"],
    "s-",
    color=color2,
    linewidth=2,
    label="Cell 1 Entry Power"
)

ax2.set_ylabel("Power at Cell 1 Entry (µW)", color=color2)
ax2.tick_params(axis="y", labelcolor=color2)

# -------------------------------------------------------------------------
# Combined legend
# -------------------------------------------------------------------------

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc="best"
)

ax1.grid(True, alpha=0.3)

plt.title("Power Transmission Through Fibre System")
plt.tight_layout()

plt.savefig(
    output_dir / "power_transfer_overview.png",
    dpi=300
)

plt.close()

print("Saved: power_transfer_overview.png")