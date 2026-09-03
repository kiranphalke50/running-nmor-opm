"""Minimal NMOR resonance fit for a Zurich Instruments sweeper CSV.

Edit CHUNK, FIT_MIN_HZ and FIT_MAX_HZ below, then run:
    python nmor_manual_fit.py

X is fitted with the old lab single-Lorentzian model.
Y is fitted with a straight line around the fitted centre frequency.
The Y slope is a responsivity/sensitivity proxy, not absolute sensitivity.
"""

import csv
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, differential_evolution

# --------------------------- user settings ---------------------------
LINEAR_HALF_WINDOW_FWHM = 0.25  # fit Y over centre +/- 0.25 * FWHM
TOGGLE_PLOT = True  # set to False to skip plotting
# --------------------------------------------------------------------


def read_zi_csv(filename, chunk=0):
    """Read frequency, x, y and phase from the selected ZI CSV chunk."""
    wanted = {"frequency", "x", "y", "phase"}
    data = {}

    with open(filename, newline="") as file:
        reader = csv.reader(file, delimiter=";")
        next(reader)  # header

        for row in reader:
            if len(row) < 5:
                continue
            if int(row[0]) == chunk and row[3] in wanted:
                values = [float(value) for value in row[4:] if value != ""]
                data[row[3]] = np.asarray(values, dtype=float)

    missing = wanted.difference(data)
    if missing:
        raise ValueError(f"Missing fields in chunk {chunk}: {sorted(missing)}")

    lengths = {name: len(values) for name, values in data.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Field lengths do not match: {lengths}")

    order = np.argsort(data["frequency"])
    return tuple(data[name][order] for name in ("frequency", "x", "y", "phase"))


def read_zi_csv_all_chunks(filename):
    """Read frequency, x, y and phase from all ZI CSV chunks."""
    chunks = {}
    chunk_index = 0

    while True:
        try:
            chunk_data = read_zi_csv(filename, chunk=chunk_index)
            chunks[chunk_index] = chunk_data
            chunk_index += 1
        except ValueError:
            break

    if not chunks:
        raise ValueError("No valid chunks found in the CSV file.")

    return chunks


# Same positive single-Lorentzian model as the old script.
# wid0 is HWHM, so FWHM = 2 * abs(wid0).
def lorentzian_single(x, amp0, cen0, wid0, slope, offset):
    return amp0 * wid0**2 / ((x - cen0) ** 2 + wid0**2) + slope * x + offset


def sum_of_squared_error(parameters, x_data, y_data):
    warnings.filterwarnings("ignore")
    return np.sum((y_data - lorentzian_single(x_data, *parameters)) ** 2)


def generate_initial_parameters(x_data, y_data):
    """Old lab differential-evolution initialisation and magic numbers."""
    max_x, min_x = max(x_data), min(x_data)
    max_y = max(y_data)

    parameter_bounds = [
        [max_y / 1.5, max_y * 2],  # positive amplitude
        [min_x, max_x],  # centre frequency
        [10, 150],  # HWHM in Hz
        [-0.005, 0.005],  # baseline slope
        [max_y / -0.005, max_y / 0.005],  # baseline offset
    ]

    result = differential_evolution(
        lambda p: sum_of_squared_error(p, x_data, y_data),
        parameter_bounds,
        seed=3,
    )
    return result.x


def plot_fitted_single_chunk(
    base_folder,
    cell_str,
    frequency,
    x,
    y,
    phase,
    f_fit,
    parameters,
    centre_hz,
    fwhm_hz,
    half_window_hz,
    linear_mask,
    y_line,
    y_slope,
    chunk,
    freq_limits,
):
    output_fit_plots_dir = os.path.join(base_folder, "fit_plots_single_chunks")
    os.makedirs(output_fit_plots_dir, exist_ok=True)

    dense_frequency = np.linspace(f_fit.min(), f_fit.max(), 1000)

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

    # X and Lorentzian fit.
    axes[0].plot(frequency, x, "o", markersize=3, label="X data")
    axes[0].plot(
        dense_frequency,
        lorentzian_single(dense_frequency, *parameters),
        "r-",
        label="Lorentzian fit",
    )
    axes[0].axvspan(freq_limits[0], freq_limits[1], color="grey", alpha=0.12)
    axes[0].axvline(centre_hz, color="black", linestyle=":")
    axes[0].set_ylabel("X (nA)")
    axes[0].legend()
    axes[0].grid()

    # Y and central line fit.
    axes[1].plot(frequency, y, "o", markersize=3, label="Y data")
    axes[1].plot(frequency[linear_mask], y_line, "r-", label="Central line fit")
    axes[1].axvspan(centre_hz - half_window_hz, centre_hz + half_window_hz, color="orange", alpha=0.18)
    axes[1].axvline(centre_hz, color="black", linestyle=":")
    axes[1].set_ylabel("Y (nA)")
    axes[1].legend()
    axes[1].grid()

    # Saved phase.
    axes[2].plot(frequency, phase, "o", markersize=3)
    axes[2].axvline(centre_hz, color="black", linestyle=":")
    axes[2].set_xlabel("Frequency (Hz)")
    axes[2].set_ylabel("Phase (rad)")
    axes[2].grid()

    fig.suptitle(f"Chunk {chunk}: centre = {centre_hz:.1f} Hz, FWHM = {fwhm_hz:.1f} Hz, |Y slope| = {abs(y_slope):.3e} nA/Hz")
    fig.tight_layout()
    # plt.show()

    # Save the plot to a file.
    plot_filename = f"{cell_str}_chunk_{chunk}_fit_plot.png"
    fig.savefig(os.path.join(output_fit_plots_dir, plot_filename), dpi=300)
    print(f"Saved plot for chunk {chunk} to {plot_filename}")


def plot_all_chunks_fits(base_folder, chunks, results_all_chunks, cell_str):
    """Plot all fitted chunks in a single figure."""
    # check if chunk size is same with chunks and results_all_chunks
    if len(chunks) != len(results_all_chunks):
        print(f"Warning: Number of chunks ({len(chunks)}) does not match number of results ({len(results_all_chunks)}).")
    # temporarily remove any chunks that are not in results_all_chunks
    chunks = {k: v for k, v in chunks.items() if k in results_all_chunks}
    num_chunks = len(chunks)

    # create output directory
    output_fit_plots_dir = os.path.join(base_folder, "fit_plots_overlay_all_chunks")
    os.makedirs(output_fit_plots_dir, exist_ok=True)

    # create 3 sub-plots : subplot1 has overlaid x of all chunks with their lorentzian fits
    # subplot2 has overlaid y of all chunks with their linear fits
    # subplot3 has overlaid phase of all chunks
    fig, axes = plt.subplots(3, 1, figsize=(9, 3 * num_chunks), sharex=True)
    for idx, (result, chunk) in enumerate(zip(results_all_chunks.values(), chunks.values(), strict=True)):
        frequency = chunk[0]  # index 0 : frequency, 1 : x, 2 : y, 3 : phase
        x = chunk[1]
        y = chunk[2]
        phase = chunk[3]
        parameters = result["parameters"]
        dense_frequency = np.linspace(frequency.min(), frequency.max(), 1000)

        axes[0].plot(frequency, x, "o", markersize=3, label=f"Chunk {result['chunk_label']} X data")
        axes[0].plot(
            dense_frequency,
            lorentzian_single(dense_frequency, *parameters),
            "r-",
            label=f"Chunk {result['chunk_label']} Lorentzian fit",
        )
        axes[0].axvline(result["centre_hz"], color="black", linestyle=":")
        axes[0].set_ylabel("X (nA)")
        axes[0].legend()
        axes[0].grid()

        centre_hz = result["centre_hz"]
        fwhm_hz = result["fwhm_hz"]
        half_window_hz = LINEAR_HALF_WINDOW_FWHM * fwhm_hz
        linear_mask = np.abs(frequency - centre_hz) <= half_window_hz
        y_line = result["y_slope"] * frequency[linear_mask] + result["y_intercept"]
        axes[1].plot(frequency[linear_mask], y_line, "r-", label=f"Chunk {result['chunk_label']} fit")
        axes[1].axvspan(centre_hz - half_window_hz, centre_hz + half_window_hz, color="orange", alpha=0.18)
        axes[1].axvline(centre_hz, color="black", linestyle=":")
        axes[1].set_ylabel("Y (nA)")
        axes[1].legend()
        axes[1].grid()

        axes[2].plot(frequency, phase, "o", markersize=3, label=f"Chunk {result['chunk_label']} Phase data")
        axes[2].axvline(centre_hz, color="black", linestyle=":")
        axes[2].set_xlabel("Frequency (Hz)")
        axes[2].set_ylabel("Phase (rad)")
        axes[2].legend()
        axes[2].grid()

    fig.suptitle(f"Overlay of all fitted chunks for {cell_str}")
    fig.tight_layout()
    plt.savefig(os.path.join(output_fit_plots_dir, f"{cell_str}_all_chunks_fit_plot.png"), dpi=300)


def fit_single_chunk(base_folder, chunk, cell_str, frequency, x, y, phase, freq_limits):
    result_single_chunk = {}
    try:
        # Select the frequency interval containing the resonance.
        resonance_mask = (frequency >= freq_limits[0]) & (frequency <= freq_limits[1])

        f_fit = frequency[resonance_mask]
        x_fit = x[resonance_mask]

        if len(f_fit) < 8:
            raise ValueError("Too few points in the selected Lorentzian fit window.")

        # Fit the Lorentzian X channel.
        initial = generate_initial_parameters(f_fit, x_fit)

        parameters, covariance = curve_fit(
            lorentzian_single,
            f_fit,
            x_fit,
            p0=initial,
            maxfev=5000,
        )

        (
            amplitude,
            centre_hz,
            hwhm_hz,
            background_slope,
            offset,
        ) = parameters

        hwhm_hz = abs(hwhm_hz)
        fwhm_hz = 2.0 * hwhm_hz

        parameter_errors = np.sqrt(np.diag(covariance))
        fwhm_error_hz = 2.0 * parameter_errors[2]

        # Fit Y close to the fitted resonance centre.
        half_window_hz = LINEAR_HALF_WINDOW_FWHM * fwhm_hz

        linear_mask = np.abs(frequency - centre_hz) <= half_window_hz

        if np.count_nonzero(linear_mask) < 3:
            raise ValueError("Too few points for the Y line fit.")

        y_slope, y_intercept = np.polyfit(
            frequency[linear_mask],
            y[linear_mask],
            deg=1,
        )

        y_line = y_slope * frequency[linear_mask] + y_intercept

        # Calculate goodness of the Y line fit.
        y_residuals = y[linear_mask] - y_line
        ss_res = np.sum(y_residuals**2)

        ss_tot = np.sum((y[linear_mask] - np.mean(y[linear_mask])) ** 2)

        y_r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        # Old A/W proxy, now using FWHM consistently.
        amplitude_over_fwhm = amplitude / fwhm_hz

        # Save results for this chunk.
        result_single_chunk = {
            "chunk_label": chunk,
            "parameters": parameters,
            "amplitude": amplitude,
            "centre_hz": centre_hz,
            "hwhm_hz": hwhm_hz,
            "fwhm_hz": fwhm_hz,
            "fwhm_error_hz": fwhm_error_hz,
            "background_slope": background_slope,
            "offset": offset,
            "amplitude_over_fwhm": amplitude_over_fwhm,
            "y_slope": y_slope,
            "absolute_y_slope": abs(y_slope),
            "y_intercept": y_intercept,
            "y_r_squared": y_r_squared,
        }

        # plot fitted chunk
        if TOGGLE_PLOT:
            plot_fitted_single_chunk(
                base_folder,
                cell_str,
                frequency,
                x,
                y,
                phase,
                f_fit,
                parameters,
                centre_hz,
                fwhm_hz,
                half_window_hz,
                linear_mask,
                y_line,
                y_slope,
                chunk,
                freq_limits,
            )

        print(
            f"Chunk {chunk:2d}: "
            f"centre = {centre_hz:9.3f} Hz, "
            f"FWHM = {fwhm_hz:8.3f} Hz, "
            f"|Y slope| = {abs(y_slope):.6e}/Hz, "
            f"R^2 = {y_r_squared:.5f}"
        )

        return result_single_chunk

    except (
        ValueError,
        RuntimeError,
        FloatingPointError,
    ) as error:
        print(f"Chunk {chunk:2d}: Fit failed with error: {error}")
        return None


def fit_multiple_chunks_without_averaging(base_folder, filepath, cell_str, freq_limits):
    results_all_chunks = {}

    if not isinstance(freq_limits, (list, tuple)) or len(freq_limits) != 2:
        raise ValueError("freq_limits must be a list or tuple of two values (min, max) with single entry for each cell.")
    if freq_limits[0] >= freq_limits[1]:
        raise ValueError("freq_limits[0] must be less than freq_limits[1].")
    if filepath is None or not isinstance(filepath, str):
        raise ValueError(f"filepath must be a valid string : {filepath}")

    # read all the chunks available in the csv file and fit each chunk separately
    chunks = read_zi_csv_all_chunks(filepath)

    for chunk_id, chunk in chunks.items():
        # extract frequency, x, y and phase from the chunk
        frequency, x, y, phase = chunk

        result = fit_single_chunk(base_folder, chunk_id, cell_str, frequency, x, y, phase, freq_limits)
        if result is not None:
            results_all_chunks[chunk_id] = result

    if not results_all_chunks:
        raise RuntimeError("No chunks were fitted successfully.")

    if TOGGLE_PLOT:
        plot_all_chunks_fits(base_folder, chunks, results_all_chunks, cell_str)

    return results_all_chunks


if __name__ == "__main__":
    print("This script is intended to be imported as a module, not run directly.")
    print("Use fit_lorentzian_multiple_runs.py to fit multiple runs and cells.")
    print("Make appropriate changes if you want to run this script directly for a single run.")
    print("Ensure that the CSV_FILE, CHUNK, FIT_MIN_HZ, and FIT_MAX_HZ variables are set correctly.")
