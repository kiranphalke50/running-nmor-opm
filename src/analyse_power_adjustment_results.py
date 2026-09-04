from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PLOT_POWER_TRANSFER = False  # set to True to plot power transfer overview

# =============================================================================
# File names
# =============================================================================

base_folder = Path(r"Y:\KiranPhalke\NMOR_sensor_characterization\sweep_test_05_after_offset_correction\power_adjustment")
cell1_power_csv = base_folder / "laser_power_measurements_cell1.csv"
cell2_power_csv = base_folder / "laser_power_measurements_cell2.csv"
cell1_csv = base_folder / "lorentzian_fit_results_cell1.csv"
cell2_csv = base_folder / "lorentzian_fit_results_cell2.csv"
# cols_to_exclude = ["chunk_label", "adjusted_parameter", "p1", "p2"]
cols_to_include = ["absolute_y_slope", "amplitude_over_fwhm", "fwhm_hz", "y_r_squared", "amplitude"]

output_dir = base_folder / "plots_slope_ratio"
output_dir.mkdir(exist_ok=True)

# =============================================================================
# Load data
# =============================================================================

cell1_power_df = pd.read_csv(cell1_power_csv)
cell2_power_df = pd.read_csv(cell2_power_csv)

cell1_df = pd.read_csv(cell1_csv)
cell2_df = pd.read_csv(cell2_csv)

# =============================================================================
# Merge power measurements with results
# =============================================================================

# merge two dataframes such that "adjusted_parameter" column from cell1_df and "power_fibre_entry_uW" from the "cell1_power_df" are matched
cell1 = pd.merge(cell1_df, cell1_power_df, left_on="adjusted_parameter", right_on="power_fibre_entry_uW", how="inner")

# merge two dataframes such that "adjusted_parameter" column from cell2_df and "power_fibre_entry_uW" from the "cell2_power_df" are matched
cell2 = pd.merge(cell2_df, cell2_power_df, left_on="adjusted_parameter", right_on="power_fibre_entry_uW", how="inner")

# =============================================================================
# Sort by fibre entry power
# =============================================================================

x_col_cell1 = "power_cell1_entry_uW"
x_col_cell2 = "power_cell2_entry_uW"

cell1 = cell1.sort_values(x_col_cell1)
cell2 = cell2.sort_values(x_col_cell2)

# =============================================================================
# Remove entries whose fit was not good ("y_r_squared" > 0.95)
# =============================================================================
cell1 = cell1[cell1["y_r_squared"] > 0.97]
cell2 = cell2[cell2["y_r_squared"] > 0.95]

# =============================================================================
# Merge 2 dataframes such that 2 cells can be plotted together with respect to the same x-axis (fibre entry power)
# =============================================================================

# =============================================================================
# Plot all mean parameters with std error bars
# =============================================================================
for col in cell1.columns:
    if col not in cols_to_include:
        continue

    fig, ax = plt.subplots(figsize=(8, 5))

    # -------------------------------------------------------------------------
    # Cell 1
    # -------------------------------------------------------------------------
    ax.plot(
        cell1[x_col_cell1],
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
        cell2[x_col_cell2],
        cell2[col],
        "s-",
        color="tab:red",
        linewidth=1.5,
        markersize=5,
        label="Cell 2",
    )

    ax.set_xlabel("Power at the entry of respective cell (µW)")
    ax.set_ylabel(col)
    ax.set_title(f"{col} vs Power at Entry of the Respective Cell")

    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    filename = output_dir / f"{col}_vs_power_in_front_of_cells.png"
    plt.savefig(filename, dpi=300)
    plt.close()

    print(f"Saved: {filename}")

print("\nFinished generating plots.")

if PLOT_POWER_TRANSFER:
    # =============================================================================
    # Power transfer plot
    # =============================================================================

    power_plot_df = cell1_df.sort_values("power_fibre_entry_mW")

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
        label="Fibre Exit Power",
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
        label="Cell 1 Entry Power",
    )

    ax2.set_ylabel("Power at Cell 1 Entry (µW)", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    # -------------------------------------------------------------------------
    # Combined legend
    # -------------------------------------------------------------------------

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    ax1.grid(True, alpha=0.3)

    plt.title("Power Transmission Through Fibre System")
    plt.tight_layout()

    plt.savefig(output_dir / "power_transfer_overview.png", dpi=300)

    plt.close()

    print("Saved: power_transfer_overview.png")
