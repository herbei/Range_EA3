"""Plot the double-well terminal distributions in Figure 3 of Paper_H200.pdf.

Read completed exact and Euler studies; this file does not simulate paths.
Each panel uses the exact sample's bandwidth for every Gaussian KDE.
At T=4 only, the two coarsest Euler curves are conditional on finite
endpoints inside that panel's exact sample range, as in the paper.
All other Euler KDEs keep the original sample count in their denominator.
Matplotlib is used only for drawing and saving the figure.

Run from the project root:
    python3 -m Python.examples.DoubleWell.Plot_Figure3
"""

import csv
import math
from pathlib import Path

from Python.examples.DoubleWell import study_io


horizons = [0.5, 1.0, 2.0, 4.0]
euler_steps = [2 ** (-level) for level in range(1, 6)]
grid_size = 1024
exact_color = "#0b5cad"
euler_colors = ["#b2182b", "#ef8a17", "#1b9e77", "#6a3d9a", "#4d4d4d"]
euler_styles = ["--", ":", "-.", (0, (6, 2, 1, 2)), (0, (3, 2, 1, 2, 1, 2))]


def positive_integer(value, name, allow_zero=False):
    """Read an integer count from either JSON metadata or a CSV cell."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer count")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer count") from None
    minimum = 0 if allow_zero else 1
    if not math.isfinite(number) or number < minimum or number != math.floor(number):
        raise ValueError(f"{name} must be an integer count of at least {minimum}")
    return int(number)


def read_rows(filename, required_columns):
    """Read raw CSV rows without changing the saved file."""
    with open(filename, "r", newline="") as source:
        reader = csv.DictReader(source)
        columns = reader.fieldnames
        if columns is None or len(columns) != len(set(columns)):
            raise ValueError(f"{filename.name}: missing or duplicate column names")
        if not set(required_columns).issubset(columns):
            raise ValueError(f"{filename.name}: required study columns are missing")
        rows = []
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{filename.name}: a row has the wrong number of cells")
            rows.append(row)
    return rows


def validate_statuses(exact_status, euler_status):
    """Require compatible completed studies, allowing different sample sizes."""
    reference = None
    counts = []
    for phase, status in [("exact", exact_status), ("euler", euler_status)]:
        if status.get("complete") is not True:
            raise ValueError(f"The {phase} study is not complete")
        target = positive_integer(status["target"], phase + " target")
        completed = positive_integer(status["completed"], phase + " completed")
        if completed != target:
            raise ValueError(f"The {phase} study has not reached its requested target")
        counts.append(target)

        config = status["config"]
        if config.get("gamma") != 1 or isinstance(config.get("gamma"), bool) or config.get("x0") != -1:
            raise ValueError("Figure 3 requires gamma=1 and x0=-1")
        if config.get("horizons") != horizons:
            raise ValueError("Figure 3 requires horizons 0.5, 1, 2, and 4")
        if config.get("implementation") != study_io.IMPLEMENTATION:
            raise ValueError("The study implementation does not match this workflow")
        metadata = {}
        for name in ["implementation", "sampler_fingerprint", "python_version", "platform"]:
            value = config.get(name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Study metadata {name} is missing")
            metadata[name] = value
        fingerprint = metadata["sampler_fingerprint"]
        if len(fingerprint) != 64 or any(letter not in "0123456789abcdefABCDEF" for letter in fingerprint):
            raise ValueError("The sampler fingerprint must be a SHA256 string")
        if reference is None:
            reference = metadata
        if metadata != reference:
            raise ValueError("Exact and Euler implementation, fingerprint, or environment differ")
        if phase == "exact" and config.get("step_delta") != 0.2:
            raise ValueError("The stepwise exact study must use increments of length 0.2")
        if phase == "euler" and config.get("euler_steps") != euler_steps:
            raise ValueError("The Euler study must use step sizes 2^-1 through 2^-5")
        seed_names = ["direct_seed", "stepwise_seed"] if phase == "exact" else ["euler_seed"]
        for name in seed_names:
            positive_integer(config.get(name), name, allow_zero=True)
        if phase == "euler":
            positive_integer(config.get("seed_spacing"), "seed_spacing")
    return counts[0], counts[1]


def read_study(folder):
    """Return validated full rows and groups; never truncate Euler observations."""
    exact_status = study_io.read_complete_status(folder, "exact")
    euler_status = study_io.read_complete_status(folder, "euler")
    n_paths, n_euler = validate_statuses(exact_status, euler_status)
    common_columns = ["gamma", "x0", "path", "seed", "T", "delta", "endpoint", "implementation"]
    exact_counters = [
        "proposals", "accepted_poisson_points", "total_poisson_points",
        "bridge_max_calls", "rbmmax_calls", "icb_calls",
    ]
    exact_rows = read_rows(
        folder / "Figure3_exact_draws.csv",
        common_columns + ["kind", "step", "start_value"] + exact_counters,
    )
    euler_rows = read_rows(
        folder / "Figure3_euler_draws.csv",
        common_columns + ["level", "euler_exploded", "euler_explosion_step"],
    )
    exact_by_time = {time: [] for time in horizons}
    euler_by_time_level = {(time, level): [] for time in horizons for level in range(1, 6)}

    for phase, rows in [("exact", exact_rows), ("euler", euler_rows)]:
        for row in rows:
            for name in ["gamma", "x0", "T", "delta", "endpoint"]:
                row[name] = float(row[name])
            if row["gamma"] != 1 or row["x0"] != -1 or row["T"] not in horizons:
                raise ValueError(f"Unexpected {phase} row parameters or horizon")
            if row["implementation"] != study_io.IMPLEMENTATION:
                raise ValueError(f"Unexpected {phase} row implementation")
            row["path"] = positive_integer(row["path"], phase + " path")
            row["seed"] = positive_integer(row["seed"], phase + " seed", allow_zero=True)
            time = row["T"]

            if phase == "exact":
                row["step"] = positive_integer(row["step"], "exact step")
                row["start_value"] = float(row["start_value"])
                if not math.isfinite(row["endpoint"]) or not math.isfinite(row["start_value"]):
                    raise ValueError("Exact endpoints and starting values must be finite")
                delta = 0.5 if time == 0.5 else 0.2
                kind = "direct" if time == 0.5 else "stepwise"
                step = 1 if time == 0.5 else round(time / delta)
                if row["delta"] != delta or row["kind"] != kind or row["step"] != step:
                    raise ValueError("Exact terminal-increment metadata does not match its horizon")
                seed_name = "direct_seed" if kind == "direct" else "stepwise_seed"
                seed = exact_status["config"][seed_name] + row["path"] - 1
                if row["seed"] != seed:
                    raise ValueError("Exact row seed does not match its completed study")
                if time == 0.5 and row["start_value"] != -1:
                    raise ValueError("The direct exact paths must start at -1")
                for name in exact_counters:
                    row[name] = positive_integer(row[name], name, allow_zero=(name != "proposals"))
                if row["accepted_poisson_points"] > row["total_poisson_points"]:
                    raise ValueError("Accepted Poisson points exceed the total count")
                exact_by_time[time].append(row)
            else:
                row["level"] = positive_integer(row["level"], "Euler level")
                level = row["level"]
                if level not in range(1, 6) or row["delta"] != 2 ** (-level):
                    raise ValueError("Unexpected Euler level or step size")
                config = euler_status["config"]
                seed = config["euler_seed"] + (level - 1) * config["seed_spacing"] + row["path"] - 1
                if row["seed"] != seed:
                    raise ValueError("Euler row seed does not match its completed study")
                if math.isnan(row["endpoint"]):
                    raise ValueError("Euler failures must use signed infinity, not NaN")
                if row["euler_exploded"] not in ["True", "False"]:
                    raise ValueError("Euler explosion flags must be True or False")
                row["euler_exploded"] = row["euler_exploded"] == "True"
                if row["euler_exploded"]:
                    row["euler_explosion_step"] = positive_integer(
                        row["euler_explosion_step"], "Euler explosion step",
                    )
                    if row["euler_explosion_step"] > round(time / row["delta"]):
                        raise ValueError("Euler explosion step exceeds its horizon")
                    if math.isfinite(row["endpoint"]):
                        raise ValueError("An exploded Euler endpoint must be nonfinite")
                else:
                    if row["euler_explosion_step"] != "" or not math.isfinite(row["endpoint"]):
                        raise ValueError("Euler endpoint and explosion metadata disagree")
                    row["euler_explosion_step"] = None
                euler_by_time_level[(time, level)].append(row)

    for name, groups, count in [
        ("exact", exact_by_time, n_paths), ("Euler", euler_by_time_level, n_euler),
    ]:
        expected_paths = list(range(1, count + 1))
        for key, rows in groups.items():
            rows.sort(key=lambda row: row["path"])
            if [row["path"] for row in rows] != expected_paths:
                raise ValueError(f"The {name} group {key} has missing or duplicate path IDs")

    # These are four observations of the same Euler path. An overflow cannot
    # disappear, change sign, or acquire a different first failure step later.
    for level in range(1, 6):
        for index in range(n_euler):
            previous_time = 0.0
            first_failure = None
            failed_endpoint = None
            for time in horizons:
                row = euler_by_time_level[(time, level)][index]
                if first_failure is not None:
                    if (not row["euler_exploded"] or row["euler_explosion_step"] != first_failure
                            or row["endpoint"] != failed_endpoint):
                        raise ValueError("Euler overflow must persist with the same sign and first step")
                elif row["euler_exploded"]:
                    first_failure = row["euler_explosion_step"]
                    failed_endpoint = row["endpoint"]
                    if first_failure * row["delta"] <= previous_time:
                        raise ValueError("Euler overflow predates an already finite horizon")
                previous_time = time

    return {
        "exact_status": exact_status, "euler_status": euler_status,
        "exact_rows": exact_rows, "euler_rows": euler_rows,
        "exact_by_time": exact_by_time, "euler_by_time_level": euler_by_time_level,
        "n_paths": n_paths, "n_euler": n_euler,
    }


def quantile(sorted_values, probability):
    """Interpolate between the two sorted observations around a quantile."""
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return sorted_values[lower] + fraction * (sorted_values[upper] - sorted_values[lower])


def select_bandwidth(values):
    """Use R's bw.nrd0 rule, selected from the exact sample only."""
    n = len(values)
    if n < 2:
        return None
    mean = sum(values) / n
    squared_distances = 0.0
    for value in values:
        squared_distances += (value - mean) ** 2
    sd = math.sqrt(squared_distances / (n - 1))
    ordered = sorted(values)
    iqr = quantile(ordered, 0.75) - quantile(ordered, 0.25)
    scale = min(sd, iqr / 1.34)
    if scale == 0:
        scale = sd
    if scale == 0:
        scale = abs(values[0])
    if scale == 0:
        scale = 1.0
    return 0.9 * scale * n ** (-0.2)


def gaussian_kde(values, grid, bandwidth, normalizing_count):
    """Sum finite Gaussian bumps, with an explicitly chosen denominator."""
    if bandwidth is None or normalizing_count == 0:
        return []
    if not math.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("The KDE bandwidth must be finite and positive")
    if normalizing_count < len(values) or normalizing_count < 1:
        raise ValueError("The KDE denominator cannot be smaller than the retained sample")
    normalization = normalizing_count * bandwidth * math.sqrt(2 * math.pi)
    density = []
    for y in grid:
        total = 0.0
        for value in values:
            z = (y - value) / bandwidth
            # Multiplication can safely become infinity for huge finite Euler
            # outliers; exp(-infinity) then contributes zero on this grid.
            total += math.exp(-0.5 * z * z)
        density.append(total / normalization)
    return density


def euler_plot_sample(values, exact_values, time, level):
    """Apply the paper's conditional plotting exception, never to saved data."""
    finite_values = []
    for value in values:
        if math.isfinite(value):
            finite_values.append(value)
    conditional = time == 4 and level in [1, 2]
    retained = finite_values
    if conditional:
        lower = min(exact_values)
        upper = max(exact_values)
        retained = []
        for value in finite_values:
            if lower <= value <= upper:
                retained.append(value)
    normalizing_count = len(retained) if conditional else len(values)
    return {
        "values": retained, "normalizing_count": normalizing_count,
        "original_count": len(values), "finite_count": len(finite_values),
        "retained_count": len(retained), "conditional": conditional,
    }


def prepare_panels(study, points=grid_size, show_progress=False):
    """Compute plot curves without importing any drawing library."""
    if type(points) is not int or points < 2:
        raise ValueError("The plotting grid needs at least two points")
    all_exact = [row["endpoint"] for row in study["exact_rows"]]
    left, right = min(all_exact), max(all_exact)
    padding = 0.04 * (right - left) if right > left else 0.5
    left -= padding
    right += padding
    grid = [left + i * (right - left) / (points - 1) for i in range(points)]
    panels = []
    for time in horizons:
        exact = [row["endpoint"] for row in study["exact_by_time"][time]]
        bandwidth = select_bandwidth(exact)
        if show_progress:
            print(f"T={time:g}: preparing the exact KDE ({len(exact)} endpoints).", flush=True)
        exact_kde = gaussian_kde(exact, grid, bandwidth, len(exact))
        euler_curves = []
        for level in range(1, 6):
            values = [row["endpoint"] for row in study["euler_by_time_level"][(time, level)]]
            sample = euler_plot_sample(values, exact, time, level)
            sample["level"] = level
            if show_progress:
                print(f"T={time:g}: preparing Euler KDE {level}/5.", flush=True)
            sample["density"] = gaussian_kde(
                sample["values"], grid, bandwidth, sample["normalizing_count"],
            )
            euler_curves.append(sample)
        panels.append({
            "T": time, "exact": exact, "exact_kde": exact_kde,
            "bandwidth": bandwidth, "euler_curves": euler_curves,
        })
    return grid, panels


def caption_text(study, panels):
    """State the sample sizes and precisely which Euler curves are conditional."""
    exact_n = f'{study["n_paths"]:,}'.replace(",", "{,}")
    euler_n = f'{study["n_euler"]:,}'.replace(",", "{,}")
    text = (
        r"\caption{Terminal distributions for the double-well diffusion with "
        r"$\gamma=1$ and $X_0=-1$ at $T\in\{0.5,1,2,4\}$. Gray histograms "
        r"and blue KDEs use $n=" + exact_n + r"$ exact endpoints per horizon. "
        r"For $T>0.5$, exact paths use Markov increments of length $\Delta=0.2$. "
        r"Euler--Maruyama curves use $n=" + euler_n + r"$ paths per horizon and "
        r"step size $\delta=2^{-i}$, $i=1,\ldots,5$. All KDEs in a panel use "
        r"the bandwidth selected from its exact sample. At $T=4$ only, the "
        r"$\delta=2^{-1}$ and $2^{-2}$ curves are conditional KDEs of finite "
        r"Euler endpoints inside that panel's exact sample range, normalized "
        r"by the number retained. "
    )
    for panel in panels:
        for sample in panel["euler_curves"]:
            if sample["conditional"]:
                fraction = sample["retained_count"] / sample["original_count"]
                text += (
                    r"For $\delta=2^{-" + str(sample["level"]) + r"}$, "
                    + str(sample["retained_count"]) + " of " + str(sample["original_count"])
                    + f" endpoints are retained ({100 * fraction:.2f}\\%). "
                )
                if sample["retained_count"] == 0:
                    text += "The empty conditional curve is omitted. "
    text += (
        "All other Euler curves retain their original sample count in the "
        "normalization, so nonfinite endpoints have no finite-density mass; "
        "finite endpoints outside the displayed range are not discarded. "
        "These plotting conventions do not alter the saved samples or Table 3."
    )
    for panel in panels:
        for sample in panel["euler_curves"]:
            missing = sample["original_count"] - sample["finite_count"]
            if missing and not sample["conditional"]:
                text += (
                    f' At $T={panel["T"]:g}$, $\\delta=2^{{-{sample["level"]}}}$, '
                    + f"{missing} of {sample['original_count']} Euler endpoints are nonfinite."
                )
    if study["n_paths"] < 10000 or study["n_euler"] < 10000:
        text += " Preview only: the paper uses 10,000 paths for each sample."
    if any(panel["bandwidth"] is None for panel in panels):
        text += " KDEs are omitted where the exact sample has fewer than two observations."
    return text + "}\n" + r"\label{fig:double-well-algorithm1-study-panel}" + "\n"


def draw_figure(study, grid, panels):
    """Draw prepared panels and return the figure without saving any files."""
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 10
    figure, axes = plt.subplots(2, 2, figsize=(7.6, 5.4))
    y_max = 0.0
    for index, panel in enumerate(panels):
        axis = axes[index // 2, index % 2]
        heights, edges, bars = axis.hist(
            panel["exact"], bins="fd", density=True, color="0.86",
            edgecolor="white", label="Exact histogram",
        )
        y_max = max(y_max, max(heights))
        if panel["exact_kde"]:
            axis.plot(grid, panel["exact_kde"], color=exact_color, linewidth=1.8, label="Exact KDE")
            y_max = max(y_max, max(panel["exact_kde"]))
        else:
            print(f'T={panel["T"]:g}: KDEs omitted (fewer than two exact paths).')
        for sample in panel["euler_curves"]:
            level = sample["level"]
            if sample["density"]:
                axis.plot(
                    grid, sample["density"], color=euler_colors[level - 1],
                    linestyle=euler_styles[level - 1], linewidth=1.4,
                    label=rf"Euler KDE, $\delta=2^{{-{level}}}$",
                )
                y_max = max(y_max, max(sample["density"]))
            if sample["conditional"]:
                fraction = sample["retained_count"] / sample["original_count"]
                print(
                    f'T=4, Euler level {level}: conditional plot retains '
                    f'{sample["retained_count"]}/{sample["original_count"]} '
                    f'({100 * fraction:.2f}%) finite endpoints inside the exact range.'
                )
            elif sample["finite_count"] != sample["original_count"]:
                print(
                    f'Warning: T={panel["T"]:g}, Euler level {level}: '
                    f'{sample["original_count"] - sample["finite_count"]} nonfinite endpoints; '
                    "KDE still divides by the original sample count."
                )
        axis.set_xlim(grid[0], grid[-1])
        axis.set_title(f'T = {panel["T"]:g}', fontsize=10)
        if index % 2 == 0:
            axis.set_ylabel("Density")
        if index // 2 == 1:
            axis.set_xlabel(r"$X_T$")
        if index == 0:
            axis.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="white", fontsize=7)
    for row in axes:
        for axis in row:
            axis.set_ylim(0, 1.06 * y_max)

    preview = study["n_paths"] < 10000 or study["n_euler"] < 10000

    #note = "T=4: the two coarsest Euler curves are conditional; retained counts are in the caption."
    note=""
    
    if preview:
        note = "Preview: fewer than 10,000 paths per sample.\n" + note
        print("Preview only: the paper uses 10,000 paths per sample.")
    figure.text(0.5, 0.015, note, ha="center", fontsize=8)
    figure.tight_layout(rect=(0, 0.09, 1, 1), h_pad=1.7, w_pad=1.5)
    return figure


def main():
    """Draw completed saved studies; no paths are generated by this function."""
    folder = Path(__file__).resolve().parent
    study = read_study(folder / study_io.DATA_DIRECTORY)
    grid, panels = prepare_panels(study, show_progress=True)
    figure = draw_figure(study, grid, panels)
    import matplotlib.pyplot as plt

    png_file = folder / "Figure03.png"
    pdf_file = folder / "Figure03.pdf"
    caption_file = folder / "Figure03_caption.tex"
    figure.savefig(png_file, dpi=320)
    figure.savefig(pdf_file)
    plt.close(figure)
    with open(caption_file, "w", encoding="utf-8") as output:
        output.write(caption_text(study, panels))
    print("Saved:", png_file)
    print("Saved:", pdf_file)
    print("Saved:", caption_file)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, KeyError) as error:
        print("Figure 3 was not generated:", error)
        print("Finish the exact and Euler studies before plotting. No simulations were started.")
        raise SystemExit(1) from None
