import csv
import os

import matplotlib.pyplot as plt
import numpy as np

base_folder = r"Y:\KiranPhalke\NMOR_sensor_characterization\sweep_test_04_with_power_with_ania\power_adjustment"
cell1_laser_power_csv = os.path.join(base_folder, "laser_power_measurements_cell1.csv")
cell2_laser_power_csv = os.path.join(base_folder, "laser_power_measurements_cell2.csv")

# read csv files (skip header row)
with open(cell1_laser_power_csv, newline="") as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip header row

    # read values and write NaN for empty values
    cell1_laser_power = np.array([[float(value) if value else np.nan for value in row] for row in reader], dtype=float)

with open(cell2_laser_power_csv, newline="") as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip header row
    cell2_laser_power = np.array([[float(value) if value else np.nan for value in row] for row in reader], dtype=float)


############################################
# fitting based on measured power for cell 2
############################################
# take the cell2 power and fit a line between column1 ("power_fibre_entry_uW") and column2 ("power_cell1_entry_uW")
cell2_measured_power_fibre_entry_uW = cell2_laser_power[:, 0]
cell2_measured_power_cell1_entry_uW = cell2_laser_power[:, 1]

# Fit a line to the data
fit_params = np.polyfit(cell2_measured_power_fibre_entry_uW, cell2_measured_power_cell1_entry_uW, 1)
fit_line = np.polyval(fit_params, cell2_measured_power_fibre_entry_uW)

# fit a line between column1 ("power_fibre_entry_uW") and column3 ("power_cell2_entry_uW")
cell2_measured_power_cell2_entry_uW = cell2_laser_power[:, 2]
fit_params_cell2 = np.polyfit(cell2_measured_power_fibre_entry_uW, cell2_measured_power_cell2_entry_uW, 1)
fit_line_cell2 = np.polyval(fit_params_cell2, cell2_measured_power_fibre_entry_uW)

# plot all the fits and measured data
# Plot measured data
plt.figure(figsize=(10, 6))
plt.scatter(cell2_measured_power_fibre_entry_uW, cell2_measured_power_cell1_entry_uW, color='blue', label='Cell 1 Measured')
plt.scatter(cell2_measured_power_fibre_entry_uW, cell2_measured_power_cell2_entry_uW, color='red', label='Cell 2 Measured')

# Plot fit lines
plt.plot(cell2_measured_power_fibre_entry_uW, fit_line, color='blue', linestyle='--', label='Cell 1 Fit')
plt.plot(cell2_measured_power_fibre_entry_uW, fit_line_cell2, color='red', linestyle='--', label='Cell 2 Fit')

plt.xlabel('Power at Fibre Entry (µW)')
plt.ylabel('Power at Cell Entry (µW)')
plt.title('Power Transfer Characteristics')
plt.legend()
plt.grid()
plt.savefig(os.path.join(base_folder, "power_transfer_characteristics_measured_during_cell1_characterization.png"), dpi=300)
plt.show()

# ############################################
# # use fitting from cell2 to estimate cell1 powers and compare with measured cell1 powers
# ############################################
# cell1_measured_power_fibre_entry_uW = cell1_laser_power[:, 0]
# cell1_measured_power_cell1_entry_uW = cell1_laser_power[:, 1]
# cell1_measured_power_cell2_entry_uW = cell1_laser_power[:, 2]

# cell1_predicted_power_cell1_entry_uW = np.polyval(fit_params, cell1_measured_power_fibre_entry_uW)
# cell1_predicted_power_cell2_entry_uW = np.polyval(fit_params_cell2, cell1_measured_power_fibre_entry_uW)

# # plot measured vs predicted for cell1
# plt.figure(figsize=(10, 6))
# plt.scatter(cell1_measured_power_fibre_entry_uW, cell1_measured_power_cell1_entry_uW, color='blue', label='Cell 1 Measured')
# plt.scatter(cell1_measured_power_fibre_entry_uW, cell1_predicted_power_cell1_entry_uW, color='cyan', label='Cell 1 Predicted')
# plt.scatter(cell1_measured_power_fibre_entry_uW, cell1_measured_power_cell2_entry_uW, color='red', label='Cell 2 Measured')
# plt.scatter(cell1_measured_power_fibre_entry_uW, cell1_predicted_power_cell2_entry_uW, color='magenta', label='Cell 2 Predicted')
# plt.xlabel('Power at Fibre Entry (µW)')
# plt.ylabel('Power at Cell Entry (µW)')
# plt.title('Measured vs Predicted Power at Cell Entry for Cell 1')
# plt.legend()
# plt.grid()
# plt.show()

############################################
# fitting based on measured power for cell 1
############################################
# take the cell1 power and fit a line between column1 ("power_fibre_entry_uW") and column2 ("power_cell1_entry_uW")
cell1_measured_power_fibre_entry_uW = cell1_laser_power[:, 0]
cell1_measured_power_cell1_entry_uW = cell1_laser_power[:, 1]

# Fit a line to the data
fit_params = np.polyfit(cell1_measured_power_fibre_entry_uW, cell1_measured_power_cell1_entry_uW, 1)
fit_line = np.polyval(fit_params, cell1_measured_power_fibre_entry_uW)

# fit a line between column1 ("power_fibre_entry_uW") and column3 ("power_cell2_entry_uW")
cell1_measured_power_cell2_entry_uW = cell1_laser_power[:, 2]
fit_params_cell2 = np.polyfit(cell1_measured_power_fibre_entry_uW, cell1_measured_power_cell2_entry_uW, 1)
fit_line_cell2 = np.polyval(fit_params_cell2, cell1_measured_power_fibre_entry_uW)

# plot all the fits and measured data
# Plot measured data
plt.figure(figsize=(10, 6))
plt.scatter(cell1_measured_power_fibre_entry_uW, cell1_measured_power_cell1_entry_uW, color='blue', label='Cell 1 Measured')
plt.scatter(cell1_measured_power_fibre_entry_uW, cell1_measured_power_cell2_entry_uW, color='red', label='Cell 2 Measured')

# Plot fit lines
plt.plot(cell1_measured_power_fibre_entry_uW, fit_line, color='blue', linestyle='--', label='Cell 1 Fit')
plt.plot(cell1_measured_power_fibre_entry_uW, fit_line_cell2, color='red', linestyle='--', label='Cell 2 Fit')

plt.xlabel('Power at Fibre Entry (µW)')
plt.ylabel('Power at Cell Entry (µW)')
plt.title('Power Transfer Characteristics')
plt.legend()
plt.grid()
plt.savefig(os.path.join(base_folder, "power_transfer_characteristics_measured_during_cell2_characterization.png"), dpi=300)
plt.show()