# Fit multiple runs of the same sensor, and save the results to a CSV file. 
# This is useful for analyzing the performance of the sensor over the input space 
# This script will make a list of csv files to read and call fit_lorentzian_single_run.py
# Finally it will compile results into a single CSV file for analysis
# A separate csv file will relate power adjustments to measured power values
# Power will be measured in front of the fibre, after the fibre, and in front of the cell1

import os
import re

import pandas as pd

from fit_lorentzian_single_run import fit_single_sensor_run

strings_in_foldernames = ["b1", # bias field value for cell 1 (uA)
                          "b2", # bias field value for cell 2 (uA)
                          "p1", # power adjustment 1 (degrees)
                          "p2", # power adjustment 2 (degrees)
                          "l"   # laser detuning value (mV)
                          ]
csv_filename = "dev3994_demods_0_sample_00000.csv" # the name of the csv file to read in each folder
base_folder = r"Y:\KiranPhalke\NMOR_sensor_characterization\sweep_test_with_power_measurements\power_adjustment"
cell_str = ["cell1", "cell2"]
freq_limits = [(800, 2400), (2800, 4500)]  # frequency limits for cell1 and cell2


def is_valid_foldername(foldername):
    for string in strings_in_foldernames:
        if string not in foldername:
            return False
    return True


def get_relevant_folders(base_folder):
    import os
    relevant_folders = []
    for root, dirs, _ in os.walk(base_folder):
        for dir in dirs:
            if is_valid_foldername(dir):
                relevant_folders.append(os.path.join(root, dir))
    return relevant_folders


def get_csv_files(base_folder):
    import os
    csv_files = []
    relevant_folders = get_relevant_folders(base_folder)
    for folder in relevant_folders:
        csv_file_path = os.path.join(folder, csv_filename)
        if os.path.isfile(csv_file_path):
            csv_files.append(csv_file_path)
    return csv_files


def get_parameters_from_filenames(csv_files):
    """Extract b1, b2, p1, p2, l values from the folder names of the csv files.
    And store them in a dictionary with the csv file path as the key.
    """
    parameters_dict = {}
    for csv_file in csv_files:
        foldername = os.path.basename(os.path.dirname(csv_file))
        parameters = {}
        for string in strings_in_foldernames:
            match = re.search(rf"{string}_(\d+)", foldername)
            if match:
                parameters[string] = int(match.group(1))
            else:
                parameters[string] = None
        parameters_dict[csv_file] = parameters
    return parameters_dict


def fit_multiple_runs(base_folder, freq_limits=None):
    csv_files = get_csv_files(base_folder)
    parameters_dict = get_parameters_from_filenames(csv_files)
    results_dict = {}
    for csv_file in csv_files:
        result = fit_single_sensor_run(csv_file, freq_limits=freq_limits)
        results_dict[csv_file] = result
    
    # combine results_dict and parameters_dict into a single dictionary
    results_with_parameters = {}
    for csv_file in csv_files:
        results_with_parameters[csv_file] = {**parameters_dict[csv_file], **results_dict[csv_file]}

    return results_with_parameters


if __name__ == "__main__":
    for cell, freq_limits_cell in zip(cell_str, freq_limits, strict=True):
        print(f"Fitting Lorentzian for {cell} with frequency limits {freq_limits_cell}")
        results_filename = f"lorentzian_fit_results_{cell}.csv"
        results_filepath = os.path.join(base_folder, results_filename)
        results = fit_multiple_runs(base_folder, freq_limits=freq_limits_cell)

        # save results to a CSV file
        df = pd.DataFrame.from_dict(results, orient="index")
        df.to_csv(results_filepath, index=False)
