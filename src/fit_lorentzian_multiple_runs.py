# Fit multiple runs of the same sensor, and save the results to a CSV file.
# This is useful for analyzing the performance of the sensor over the input space
# This script will make a list of csv files to read and call fit_lorentzian_single_run.py
# Finally it will compile results into a single CSV file for analysis
# A separate csv file will relate power adjustments to measured power values
# Power will be measured in front of the fibre, after the fibre, and in front of the cell1

import csv
import json
import os
import re

import pandas as pd

from fit_lorentzian_single_run import fit_multiple_chunks_without_averaging

strings_in_foldernames = [
    "b1",  # bias field value for cell 1 (uA)
    "b2",  # bias field value for cell 2 (uA)
    "p1",  # power adjustment 1 (degrees)
    "p2",  # power adjustment 2 (degrees)
    "l",  # laser detuning value (mV)
]  # parameters that change between runs, and are used to identify the relevant folders to read in
csv_filename = "dev3994_demods_0_sample_00000.csv"  # the name of the csv file to read in each folder
csv_header_filename = "dev3994_demods_0_sample_header_00000.csv"  # headerfile for chunk label to adjusted parameter mapping
base_folder = r"Y:\KiranPhalke\NMOR_sensor_characterization\sweep_test_04_with_power_with_ania\power_adjustment"
cell_list = ["cell1", "cell2", "cell12"]  # cell12 is a gradiometer configuration
freq_limits = (1000, 1800)  # frequency limits


def is_valid_foldername(foldername, cell_str=None):
    for string in strings_in_foldernames:
        if string not in foldername:
            return False
    if cell_str is not None and cell_str not in foldername:
        return False
    return True


def get_relevant_folders(base_folder, cell_str=None):
    relevant_folders = []
    for root, dirs, _ in os.walk(base_folder):
        for dir in dirs:
            if is_valid_foldername(dir, cell_str=cell_str):
                relevant_folders.append(os.path.join(root, dir))
    return relevant_folders


def get_csv_files(base_folder, cell_str=None):
    csv_files = []
    header_csv_files = []
    relevant_folders = get_relevant_folders(base_folder, cell_str=cell_str)
    for folder in relevant_folders:
        csv_file_path = os.path.join(folder, csv_filename)
        if os.path.isfile(csv_file_path):
            csv_files.append(csv_file_path)
        header_csv_file_path = os.path.join(folder, csv_header_filename)
        if os.path.isfile(header_csv_file_path):
            header_csv_files.append(header_csv_file_path)
    return csv_files, header_csv_files


def get_constant_parameters_from_filename(csv_file):
    """Extract b1, b2, p1, p2, l values from the folder names of the csv file.
    And store them in a dictionary with the csv file path as the key.

    negative values are stored as _m, e.g. b1_m320 for b1 = -320
    """
    parameters_dict = {}
    foldername = os.path.basename(os.path.dirname(csv_file))
    parameters = {}
    for string in strings_in_foldernames:
        match = re.search(rf"{string}_(m?\d+)", foldername)
        if match:
            value_str = match.group(1)
            if value_str.startswith("m"):
                value = -int(value_str[1:])
            else:
                value = int(value_str)
            parameters[string] = value
    return parameters_dict


def get_adjusted_parameter_from_chunk_labels(header_csv_file):
    """Extract chunk labels from the header CSV file."""
    adjusted_parameters = {}
    with open(header_csv_file, newline="") as file:
        reader = csv.reader(file, delimiter=";")
        _ = next(reader)  # Read the header row
        for row in reader:
            if len(row) < 2:
                continue
            chunk_number = int(row[0])
            adjusted_parameter = float(
                row[16]
            )  # Assuming the adjusted parameter is in the 17th column (index 16)
            adjusted_parameters[chunk_number] = adjusted_parameter
    return adjusted_parameters


def fit_single_param_adjustment(base_folder, cell_str=None, freq_limits=None):
    # get the names of the csv files to read in each folder, and the header csv file for adjusted parameters
    csv_file, header_csv_file = get_csv_files(base_folder, cell_str=cell_str)
    csv_file = csv_file[0] if csv_file else None # get the first csv file if it exists, otherwise None
    header_csv_file = header_csv_file[0] if header_csv_file else None # get the first header csv file if it exists, otherwise None

    if not csv_file or not header_csv_file:
        print(f"No CSV files found for {cell_str}.")
        return

    # extract unadjusted parameters from the the csv filename and save as a json metadata
    parameters_dict = get_constant_parameters_from_filename(csv_file)
    with open(os.path.join(base_folder, f"all_params_metadata_{cell_str}.json"), "w") as json_file:
        json.dump(parameters_dict, json_file)

    # get adjusted parameters dict and save them as a csv file in the base folder
    # csv structure - columns named chunk_label and adjusted_parameter, with chunk_label as the index
    adjusted_parameters = get_adjusted_parameter_from_chunk_labels(header_csv_file)

    # Fit the model to each chunk of data
    results_all_chunks = fit_multiple_chunks_without_averaging(base_folder, csv_file, cell_str, freq_limits=freq_limits)

    # adjusted parameters : dict with chunk number as key and adjusted parameter as value
    # results_all_chunks : dict with chunk number as key and fit results as value
    for chunk in results_all_chunks:
        if chunk in adjusted_parameters:
            results_all_chunks[chunk]["adjusted_parameter"] = adjusted_parameters[chunk]
        else:
            print(f"Warning: Chunk mismatch for {cell_str}. Chunk {chunk} not found in adjusted parameters.")

    return results_all_chunks


if __name__ == "__main__":
    for cell_str in cell_list:
        results_all_chunks = fit_single_param_adjustment(base_folder, cell_str=cell_str, freq_limits=freq_limits)

        if not results_all_chunks:
            print(f"No results found for {cell_str}.")
            continue

        results_filename = f"lorentzian_fit_results_{cell_str}.csv"
        results_filepath = os.path.join(base_folder, results_filename)

        # save results to a CSV file
        df = pd.DataFrame.from_dict(results_all_chunks, orient="index")
        df.to_csv(results_filepath, index=False)
