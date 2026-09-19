"""Plot the OU distributions in Figure 2 of Paper_H200.pdf, pages 24-25.

Read saved paths at times 0, 1, ..., 6; no simulation is performed here.
Show a histogram, a Gaussian kernel density estimate, and the exact OU
transition density at times 1 and 6 for each of the two parameter settings.
Matplotlib is used only for drawing and saving the figure.
"""

import csv
import math
from pathlib import Path

from Python.functions.OU_model import OU_model


# These settings must match the simulations used to produce the CSV files.
x0 = 1.5
times = [1, 6]
grid_size = 800

cases = [
    {
        "label": "Standard",
        "theta1": 2.0,
        "theta2": 1.0,
        "filename": "Figure2_standard.csv",
    },
    {
        "label": "Strong reversion",
        "theta1": 6.0,
        "theta2": 3.0,
        "filename": "Figure2_strong_reversion.csv",
    },
]


def read_paths(csv_file):
    """Read all seven values of every saved path."""
    all_paths = []

    with open(csv_file, "r", newline="") as input_file:
        reader = csv.reader(input_file)
        column_names = next(reader, None)
        expected_columns = ["X0", "X1", "X2", "X3", "X4", "X5", "X6"]

        if column_names != expected_columns:
            raise ValueError(f"{csv_file.name}: expected columns X0 through X6")

        for row_number, row in enumerate(reader, start=2):
            if len(row) != 7:
                raise ValueError(f"{csv_file.name}, row {row_number}: expected 7 values")

            path_values = []
            for value in row:
                number = float(value)
                if not math.isfinite(number):
                    raise ValueError(f"{csv_file.name}, row {row_number}: nonfinite value")
                path_values.append(number)

            if path_values[0] != x0:
                raise ValueError(f"{csv_file.name}, row {row_number}: X0 must equal {x0}")

            all_paths.append(path_values)

    if len(all_paths) == 0:
        raise ValueError(f"{csv_file.name}: no paths found")

    return all_paths


def quantile(sorted_values, probability):
    """Find a quantile by interpolating between two sorted observations."""
    position = (len(sorted_values) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    fraction = position - lower_index

    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    return lower_value + fraction * (upper_value - lower_value)


def kernel_density(values, grid):
    """Average Gaussian bumps centered at the observations."""
    n = len(values)

    # A single observation is not enough to select a KDE bandwidth.
    if n < 2:
        return []

    sample_mean = sum(values) / n
    squared_distances = 0.0
    for value in values:
        squared_distances += (value - sample_mean) ** 2
    sample_sd = math.sqrt(squared_distances / (n - 1))

    sorted_values = sorted(values)
    q25 = quantile(sorted_values, 0.25)
    q75 = quantile(sorted_values, 0.75)
    interquartile_range = q75 - q25

    # This is the bandwidth rule used by R's default density(), bw.nrd0.
    scale = min(sample_sd, interquartile_range / 1.34)
    if scale == 0:
        scale = sample_sd
    if scale == 0:
        scale = abs(values[0])
    if scale == 0:
        scale = 1.0

    bandwidth = 0.9 * scale * n ** (-0.2)
    normalizing_constant = n * bandwidth * math.sqrt(2 * math.pi)

    # Evaluate the KDE directly, with one loop over grid points and another
    # over observations. R uses an FFT approximation, so tiny differences
    # from its plotted curve are expected even with identical observations.
    density_values = []
    for y in grid:
        total = 0.0
        for value in values:
            standardized_distance = (y - value) / bandwidth
            total += math.exp(-0.5 * standardized_distance ** 2)
        density_values.append(total / normalizing_constant)

    return density_values


def main():
    """Read the two cases, prepare four panels, and save PNG and PDF files."""
    import matplotlib.pyplot as plt

    folder = Path(__file__).resolve().parent
    panels = []
    left = math.inf
    right = -math.inf
    has_data = False
    is_preview = False

    # Keep every path, but select columns X1 and X6 for the displayed panels.
    for case in cases:
        csv_file = folder / case["filename"]
        all_paths = []
        if csv_file.exists():
            all_paths = read_paths(csv_file)
            has_data = True
            print(case["label"], ": loaded", len(all_paths), "paths")
        else:
            print("Missing:", csv_file)

        # The paper uses 10,000 paths per case. Small test files still plot.
        if len(all_paths) < 10000:
            is_preview = True

        model = OU_model(case["theta1"], case["theta2"])
        for time in times:
            values = []
            for path_values in all_paths:
                values.append(path_values[time])

            # These are the marginals from the ORIGINAL initial value x0,
            # at cumulative times 1 and 6, not one-step proposal densities.
            mean = model["transition_mean"](x0, time)
            sd = math.sqrt(model["transition_var"](time))
            left = min(left, mean - 4 * sd)
            right = max(right, mean + 4 * sd)
            if len(values) > 0:
                left = min(left, min(values))
                right = max(right, max(values))

            panels.append({
                "label": case["label"],
                "filename": case["filename"],
                "time": time,
                "model": model,
                "values": values,
            })

    if not has_data:
        raise FileNotFoundError("Neither Figure 2 CSV file was found beside this script")

    # Use the same horizontal scale in all four panels.
    padding = 0.04 * (right - left)
    left -= padding
    right += padding
    grid = []
    for i in range(grid_size):
        grid.append(left + i * (right - left) / (grid_size - 1))

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 10
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))
    legend_added = False

    for panel_number, panel in enumerate(panels):
        row = panel_number // 2
        column = panel_number % 2
        axis = axes[row, column]
        values = panel["values"]
        time = panel["time"]
        axis.set_title(f'{panel["label"]}, T = {time}', fontsize=10)

        # Leave missing panels explicitly unavailable; never reuse data from
        # the other parameter setting or generate replacement samples here.
        if len(values) == 0:
            axis.text(
                0.5, 0.5, "Data file not found:\n" + panel["filename"],
                transform=axis.transAxes, ha="center", va="center", fontsize=9,
            )
            axis.set_axis_off()
            continue

        exact = []
        for y in grid:
            exact.append(panel["model"]["transition_density"](y, x=x0, t=time))
        kde = kernel_density(values, grid)

        # density=True makes the total histogram area equal to one.
        # "fd" selects Freedman-Diaconis bins. R uses the same rule but rounds
        # its bin boundaries differently, so the bars need not be identical.
        heights, bin_edges, bars = axis.hist(
            values, bins="fd", density=True, color="0.88",
            edgecolor="white", label="Histogram",
        )
        y_max = max(max(heights), max(exact))
        if len(kde) > 0:
            axis.plot(grid, kde, color="#1f5a9d", linewidth=1.6, label="KDE")
            y_max = max(y_max, max(kde))
        else:
            print(panel["label"], "T =", time, ": KDE omitted (only one path)")
        axis.plot(grid, exact, color="#b2182b", linewidth=1.6, label="Exact transition")

        axis.set_xlim(left, right)
        axis.set_ylim(0, 1.12 * y_max)
        if column == 0:
            axis.set_ylabel("Density")
        if row == 1:
            axis.set_xlabel(r"$X_T$")
        if not legend_added:
            axis.legend(loc="upper right", frameon=False, fontsize=7)
            legend_added = True

    if is_preview:
        figure.text(
            0.5, 0.015, "Preview: missing data or fewer than 10,000 paths per case.",
            ha="center", fontsize=9,
        )
        print("Preview only: the paper uses 10,000 paths in each parameter setting.")
    figure.tight_layout(rect=(0, 0.05, 1, 1), h_pad=2.0, w_pad=2.0)

    # Rerunning this file replaces only these plots, never the CSV files.
    png_file = folder / "Figure02.png"
    pdf_file = folder / "Figure02.pdf"
    figure.savefig(png_file, dpi=320)
    figure.savefig(pdf_file)
    plt.close(figure)
    print("Saved:", png_file)
    print("Saved:", pdf_file)


if __name__ == "__main__":
    main()
