"""Generate OU data for Figure 2 and Tables 1 and 2 of Paper_H200.pdf.

Figure 2 and Table 1 use paths started at X0 = 1.5, with six increments
of length one. Table 2 uses a separate stationary-start timing experiment
and reports the mean and sample variance of replicate-average timings.
Save the individual measurements as well as the table summaries, so the
tables can be formatted later without repeating the simulations.

Timing replicates are saved separately in Table2_ICB_TimeReversal. By default,
each run clears the selected scenario's previous checkpoints and timing outputs
before starting again. The other scenario's results are preserved. Set
reset_timing_experiment = False to resume after the last completed replicate
instead; resuming requires the same sampler code and experiment settings.
The new timing files never replace the older files in this directory.

Reference: Section 4.1, pages 24-26. Run from the project root with:
    python3 -m Python.examples.OU.Simulate_Figure2
"""

import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random

from Python.functions.OU_model import OU_model
from Python.functions.Algorithm01_Vanilla import Algorithm01_Vanilla


# Choose one case per run. This also selects its parameters and filenames.
scenario = "standard"                 # or "standard"

if scenario == "standard":
    theta1 = 2.0
    theta2 = 1.0
    seed = 20260909
    timing_seed = 20261909
elif scenario == "strong_reversion":
    theta1 = 6.0
    theta2 = 3.0
    seed = 20260910
    timing_seed = 20262909
else:
    raise ValueError("scenario must be 'standard' or 'strong_reversion'")

# Either experiment can be switched off without rerunning the other one.
run_path_experiment = True
run_timing_experiment = False

# Start fresh for the selected scenario, including after sampler updates.
# Set False to resume an interrupted run with unchanged code and settings.
reset_timing_experiment = True

# The original paper used 10000 paths and 10 timing replicates of 500 draws
# per scenario. The larger timing study below uses 100 replicates.
n_paths = 10000 
n_timing_draws = 500
n_timing_replicates = 100

# Keep measurements of the revised ICB implementation separate from the old
# timing data. The RBMMax sampler is still the vanilla implementation.
timing_output_directory = "Table2_ICB_TimeReversal"
timing_implementation = "icb_iv_time_reversal_v1"

# Also report the first and last completed path/draw, including small tests.
progress_every = 100

x0 = 1.5
delta = 1.0
n_steps = 6

# The timing seed bases above keep the replicate streams separate
# between the two scenarios, as well as separate from the path experiment.

timing_buckets = [
    ("endpoint_proposal", "endpoint proposal"),
    ("bridge_max", "BridgeMax"),
    ("rbmmax", "RBMMax"),
    ("phi_bound_poisson", "phi bound + Poisson"),
    ("icb_phi_accept", "ICB + phi + accept"),
    ("other", "other"),
    ("total", "Total"),
]


def write_csv(filename, rows):
    """Write a list of dictionaries, using their keys as column names."""
    with open(filename, "w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print("Saved:", filename, flush=True)


def draw_statistics(draw):
    """Keep the endpoint and workload counters for one accepted increment."""
    stats = draw["stats"]
    return {
        "endpoint": draw["endpoint"],
        "proposals": stats["proposals"],
        "accepted_poisson_points": stats["accepted_poisson_points"],
        "total_poisson_points": stats["total_poisson_points"],
        "bridge_max_calls": stats["bridge_max_calls"],
        "rbmmax_calls": stats["rbmmax_calls"],
        "icb_calls": stats["icb_calls"],
    }


def summarize_table1(model, records):
    """Calculate one Table 1 row for each terminal time T = 1, ..., 6."""
    summary = []

    for step in range(1, n_steps + 1):
        values = []
        proposal_total = 0
        poisson_total = 0
        inverse_acceptance_total = 0.0

        # Keep ONLY this step of each path, not all steps up to this time.
        for record in records:
            if record["step"] == step:
                values.append(record["endpoint"])
                proposal_total += record["proposals"]
                poisson_total += record["total_poisson_points"]
                inverse_acceptance_total += 1 / record["acceptance_probability"]

        n = len(values)
        time = step * delta
        mean_empirical = sum(values) / n
        mean_exact = model["transition_mean"](x0, time)
        var_exact = model["transition_var"](time)

        # Sample variance uses n - 1. For a one-path test it is undefined;
        # None is written as an empty CSV cell, not as a misleading zero.
        var_empirical = None
        variance_ratio = None
        if n > 1:
            squared_distances = 0.0
            for value in values:
                squared_distances += (value - mean_empirical) ** 2
            var_empirical = squared_distances / (n - 1)
            variance_ratio = var_empirical / var_exact

        # D_T is the standard score of the sample mean, not a KS statistic.
        D_T = (mean_empirical - mean_exact) / math.sqrt(var_exact / n)

        # The predicted acceptance is a HARMONIC mean of the one-step
        # probabilities evaluated at the actual starting values X_(T-delta).
        acceptance_predicted = n / inverse_acceptance_total
        acceptance_empirical = n / proposal_total

        summary.append({
            "scenario": scenario,
            "theta1": theta1,
            "theta2": theta2,
            "x0": x0,
            "T": time,
            "delta": delta,
            "n": n,
            "proposals_mean": proposal_total / n,
            "poisson_total_mean": poisson_total / n,
            "mean_empirical": mean_empirical,
            "mean_exact": mean_exact,
            "var_empirical": var_empirical,
            "var_exact": var_exact,
            "D_T": D_T,
            "variance_ratio": variance_ratio,
            "acceptance_predicted": acceptance_predicted,
            "acceptance_empirical": acceptance_empirical,
        })

    return summary


def run_paths(model, folder):
    """Generate the Figure 2 paths and collect the Table 1 diagnostics."""
    random.seed(seed)
    all_paths = []
    records = []
    print(
        f"[{scenario}] Figure 2 / Table 1: starting {n_paths} paths "
        f"({n_steps} increments per path).",
        flush=True,
    )

    for path_number in range(1, n_paths + 1):
        current_value = x0
        path_values = [x0]

        for step in range(1, n_steps + 1):
            start_value = current_value
            draw = Algorithm01_Vanilla(model, x=start_value, T=delta)
            current_value = draw["endpoint"]
            path_values.append(current_value)

            record = {
                "scenario": scenario,
                "theta1": theta1,
                "theta2": theta2,
                "seed": seed,
                "path": path_number,
                "step": step,
                "T": step * delta,
                "delta": delta,
                "start_value": start_value,
                "acceptance_probability": model["acceptance_probability"](start_value, delta),
            }
            record.update(draw_statistics(draw))
            records.append(record)

        all_paths.append(path_values)
        if path_number == 1 or path_number % progress_every == 0 or path_number == n_paths:
            # This path has completed all six increments. flush=True shows
            # the message immediately instead of waiting in an output buffer.
            print(
                f"[{scenario}] Figure 2 / Table 1: "
                f"completed path {path_number}/{n_paths}.",
                flush=True,
            )

    # Preserve the wide X0, ..., X6 file used by Plot_Figure2.py.
    print(f"[{scenario}] Figure 2 / Table 1: saving data and summaries.", flush=True)
    output_file = folder / f"Figure2_{scenario}.csv"
    with open(output_file, "w", newline="") as output:
        writer = csv.writer(output)
        column_names = []
        for step in range(n_steps + 1):
            column_names.append(f"X{step}")
        writer.writerow(column_names)
        for one_path in all_paths:
            writer.writerow(one_path)
    print("Saved:", output_file, flush=True)

    write_csv(folder / f"Table1_{scenario}_draws.csv", records)
    summary = summarize_table1(model, records)
    write_csv(folder / f"Table1_{scenario}_summary.csv", summary)
    print(f"[{scenario}] Figure 2 / Table 1: finished.", flush=True)


def summarize_table2(replicates):
    """Summarize the mean and sample variance of replicate-average times.

    Each input time is one replicate's mean milliseconds per accepted draw.
    Variance uses R - 1, where R is the number of replicates, and has units
    (ms/draw)^2. It is not the variance of individual draw times or the
    estimated variance of the reported overall mean.
    """
    n_replicates = len(replicates)
    total_mean_ms = 0.0
    for replicate in replicates:
        total_mean_ms += replicate["total_ms"]
    total_mean_ms /= n_replicates

    summary = []
    for name, label in timing_buckets:
        replicate_means = []
        for replicate in replicates:
            replicate_means.append(replicate[name + "_ms"])

        mean_ms = sum(replicate_means) / n_replicates
        variance_ms2 = None
        if n_replicates > 1:
            squared_distances = 0.0
            for value in replicate_means:
                squared_distances += (value - mean_ms) ** 2
            variance_ms2 = squared_distances / (n_replicates - 1)
        percent_of_mean = 100 * mean_ms / total_mean_ms
        summary.append({
            "scenario": scenario,
            "theta1": theta1,
            "theta2": theta2,
            "n_replicates": n_replicates,
            "n_draws_per_replicate": replicates[0]["n"],
            "delta": delta,
            "bucket": label,
            "mean_ms": mean_ms,
            "variance_ms2": variance_ms2,
            "percent_of_mean": percent_of_mean,
        })

    # Component means sum to the total mean. Component variances need not
    # sum to the total variance because the bucket timings can covary.
    # Calculate the total variance from the replicate totals themselves.
    return summary


def timing_configuration():
    """Identify the experiment and sampler code that may share checkpoints."""
    # A source fingerprint prevents mixing timings after a sampler is edited.
    # Only the functions are hashed: switching scenarios or increasing the
    # requested replicate count in this script must not invalidate a run.
    source_folder = Path(__file__).resolve().parents[2] / "functions"
    fingerprint = hashlib.sha256()
    for filename in sorted(source_folder.glob("*.py")):
        fingerprint.update(filename.name.encode("utf-8"))
        fingerprint.update(filename.read_bytes())

    return {
        "scenario": scenario,
        "theta1": theta1,
        "theta2": theta2,
        "delta": delta,
        "n_draws": n_timing_draws,
        "timing_seed": timing_seed,
        "implementation": timing_implementation,
        "sampler_fingerprint": fingerprint.hexdigest(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }


def save_timing_checkpoint(folder, config, replicate, records):
    """Commit one complete replicate, including its individual draw timings."""
    number = replicate["replicate"]
    filename = folder / f"replicate_{number:03d}.json"
    temporary = filename.with_suffix(".json.tmp")
    checkpoint = {"config": config, "replicate": replicate, "records": records}
    # First finish a temporary file, then replace its name in one operation.
    # An interruption before replace leaves no half-written checkpoint.
    with open(temporary, "w", encoding="utf-8") as output:
        json.dump(checkpoint, output, allow_nan=False)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, filename)
    print(f"[{scenario}] Table 2: checkpoint saved for replicate {number}.", flush=True)


def load_timing_checkpoints(folder, config):
    """Read consecutive completed replicates; refuse incompatible data."""
    records = []
    replicates = []
    # Sorting by the number also works when there are more than 999 replicates.
    filenames = list(folder.glob("replicate_*.json"))
    try:
        filenames.sort(key=lambda path: int(path.stem.split("_")[1]))
    except (ValueError, IndexError):
        raise ValueError("Unexpected timing checkpoint filename") from None
    for number, filename in enumerate(filenames, start=1):
        if filename.name != f"replicate_{number:03d}.json":
            raise ValueError("Timing checkpoints must be consecutive, starting at 1")
        if number > n_timing_replicates:
            raise ValueError("More checkpoints than requested replicates; do not shrink this run")
        with open(filename, encoding="utf-8") as source:
            checkpoint = json.load(source)
        if not isinstance(checkpoint, dict):
            raise ValueError(f"{filename}: expected a checkpoint dictionary")
        if checkpoint.get("config") != config:
            raise ValueError(
                f"{filename}: checkpoint configuration does not match this run. "
                "Set reset_timing_experiment = True to start fresh, restore the "
                "original settings/code, or choose a new timing output directory."
            )
        replicate = checkpoint["replicate"]
        draws = checkpoint["records"]
        expected_seed = timing_seed + number - 1
        if (replicate["replicate"] != number or replicate["seed"] != expected_seed
                or replicate["n"] != n_timing_draws or len(draws) != n_timing_draws):
            raise ValueError(f"{filename}: incomplete or inconsistent replicate")
        for draw_number, draw in enumerate(draws, start=1):
            if (draw["replicate"] != number or draw["seed"] != expected_seed
                    or draw["draw"] != draw_number):
                raise ValueError(f"{filename}: inconsistent draw IDs or seeds")
        for row in [replicate] + draws:
            for name in ["scenario", "theta1", "theta2", "delta", "implementation"]:
                if row[name] != config[name]:
                    raise ValueError(f"{filename}: inconsistent {name}")
        for name in ["sampler_fingerprint", "python_version", "platform"]:
            if replicate[name] != config[name]:
                raise ValueError(f"{filename}: inconsistent {name}")
        for name, label in timing_buckets:
            values = [draw[name + "_seconds"] for draw in draws]
            mean_ms = replicate[name + "_ms"]
            if (any(not math.isfinite(value) or value < 0 for value in values)
                    or not math.isfinite(mean_ms) or mean_ms < 0
                    or not math.isclose(mean_ms, 1000 * sum(values) / n_timing_draws,
                                        rel_tol=1e-12, abs_tol=1e-12)):
                raise ValueError(f"{filename}: inconsistent or invalid timing values")
        for draw in draws:
            component_total = sum(draw[name + "_seconds"] for name, label in timing_buckets
                                  if name != "total")
            if not math.isclose(draw["total_seconds"], component_total,
                                rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"{filename}: total time does not match its components")
        replicates.append(replicate)
        records.extend(draws)
    return records, replicates


def save_timing_status(folder, config, completed, complete=False):
    """Tell the table formatter whether the requested timing run is complete."""
    filename = folder / f"Table2_{scenario}_status.json"
    temporary = filename.with_suffix(".json.tmp")
    status = {
        "config": config,
        "target_replicates": n_timing_replicates,
        "completed_replicates": completed,
        "complete": complete,
    }
    with open(temporary, "w", encoding="utf-8") as output:
        json.dump(status, output, allow_nan=False)
    os.replace(temporary, filename)


def export_timing_csvs(folder, config, records, replicates):
    """Export completed checkpoints; replacing CSV files does not resimulate."""
    summary = summarize_table2(replicates)
    for row in summary:
        for name in ["implementation", "sampler_fingerprint", "python_version", "platform"]:
            row[name] = config[name]
        row["target_replicates"] = n_timing_replicates
    for suffix, rows in [("draws", records), ("replicates", replicates), ("summary", summary)]:
        filename = folder / f"Table2_{scenario}_{suffix}.csv"
        temporary = filename.with_suffix(".csv.tmp")
        with open(temporary, "w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, filename)
        print("Saved:", filename, flush=True)


def clear_timing_outputs(folder, checkpoint_folder):
    """Remove this scenario's old timing data while its run lock is held."""
    # Invalidate the old completed status before deleting any measurements.
    output_names = [f"Table2_{scenario}_status.json"]
    output_names.extend(f"Table2_{scenario}_{suffix}.csv"
                        for suffix in ("draws", "replicates", "summary"))
    for name in output_names:
        for filename in (folder / name, folder / (name + ".tmp")):
            filename.unlink(missing_ok=True)

    # Keep run.lock in place: removing it would allow another process to
    # acquire a different lock while this run still holds the original one.
    for pattern in ("replicate_*.json", "replicate_*.json.tmp"):
        for filename in checkpoint_folder.glob(pattern):
            filename.unlink()
    print(f"[{scenario}] Table 2: cleared previous timing checkpoints and outputs.",
          flush=True)


def run_timing(model, folder):
    """Start fresh or resume timing replicates, as selected in the settings."""
    folder = folder / timing_output_directory
    checkpoint_folder = folder / "checkpoints" / scenario
    checkpoint_folder.mkdir(parents=True, exist_ok=True)
    # On this Mac, this standard-library file lock is released automatically
    # when Python exits, even if the run is interrupted. Never delete the lock
    # file while a run is active; its presence alone does not mean it is locked.
    with open(checkpoint_folder / "run.lock", "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"A timing run for {scenario} is already using {folder}") from None
        if reset_timing_experiment:
            clear_timing_outputs(folder, checkpoint_folder)
        run_timing_replicates(model, folder, checkpoint_folder)


def run_timing_replicates(model, folder, checkpoint_folder):
    """Run only missing replicates while holding this scenario's file lock."""
    stationary_mean = theta1 / theta2
    stationary_sd = math.sqrt(1 / (2 * theta2))
    config = timing_configuration()
    try:
        records, replicates = load_timing_checkpoints(checkpoint_folder, config)
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("Malformed timing checkpoint; no simulations were started") from error
    save_timing_status(folder, config, len(replicates))
    print(
        f"[{scenario}] Table 2: starting {n_timing_replicates} replicates "
        f"with {n_timing_draws} stationary-start draws each.",
        flush=True,
    )
    print("Timing output:", folder, flush=True)
    if replicates:
        print(f"[{scenario}] Table 2: reusing {len(replicates)} completed replicates.", flush=True)

    for replicate_number in range(len(replicates) + 1, n_timing_replicates + 1):
        replicate_records = []
        replicate_seed = timing_seed + replicate_number - 1
        random.seed(replicate_seed)
        print(
            f"[{scenario}] Table 2: starting replicate "
            f"{replicate_number}/{n_timing_replicates} (seed {replicate_seed}).",
            flush=True,
        )

        # Generate all fresh starts BEFORE timing. These are not endpoints
        # of the Figure 2 chains, nor endpoints of previous timing draws.
        start_values = []
        for draw_number in range(n_timing_draws):
            start_values.append(random.gauss(stationary_mean, stationary_sd))

        seconds = {}
        for name, label in timing_buckets:
            seconds[name] = 0.0
        proposal_total = 0
        poisson_total = 0

        for draw_number in range(1, n_timing_draws + 1):
            start_value = start_values[draw_number - 1]
            draw = Algorithm01_Vanilla(
                model, x=start_value, T=delta, record_timing=True,
            )

            # Each timing bucket includes rejected proposals as well as the
            # accepted proposal. CSV work here is OUTSIDE the timed region.
            record = {
                "scenario": scenario,
                "theta1": theta1,
                "theta2": theta2,
                "replicate": replicate_number,
                "seed": replicate_seed,
                "draw": draw_number,
                "delta": delta,
                "start_value": start_value,
                "implementation": timing_implementation,
            }
            record.update(draw_statistics(draw))
            for name, label in timing_buckets:
                elapsed = draw["timing"][name]
                record[name + "_seconds"] = elapsed
                seconds[name] += elapsed
            replicate_records.append(record)
            proposal_total += record["proposals"]
            poisson_total += record["total_poisson_points"]

            if draw_number == 1 or draw_number % progress_every == 0 or draw_number == n_timing_draws:
                # The sampler has returned, so printing is outside its timers.
                print(
                    f"[{scenario}] Table 2: replicate "
                    f"{replicate_number}/{n_timing_replicates}, "
                    f"completed draw {draw_number}/{n_timing_draws}.",
                    flush=True,
                )

        replicate = {
            "scenario": scenario,
            "theta1": theta1,
            "theta2": theta2,
            "replicate": replicate_number,
            "seed": replicate_seed,
            "n": n_timing_draws,
            "delta": delta,
            "proposals_mean": proposal_total / n_timing_draws,
            "poisson_total_mean": poisson_total / n_timing_draws,
            "acceptance_empirical": n_timing_draws / proposal_total,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "implementation": timing_implementation,
            "sampler_fingerprint": config["sampler_fingerprint"],
        }
        for name, label in timing_buckets:
            replicate[name + "_ms"] = 1000 * seconds[name] / n_timing_draws
        save_timing_checkpoint(checkpoint_folder, config, replicate, replicate_records)
        replicates.append(replicate)
        records.extend(replicate_records)
        save_timing_status(folder, config, len(replicates))
        print(
            f"[{scenario}] Table 2: replicate "
            f"{replicate_number}/{n_timing_replicates} finished.",
            flush=True,
        )

    print(f"[{scenario}] Table 2: saving data and summaries.", flush=True)
    export_timing_csvs(folder, config, records, replicates)
    save_timing_status(folder, config, len(replicates), complete=True)
    print(f"[{scenario}] Table 2: finished.", flush=True)


def main():
    """Run the selected experiments; importing the file runs neither."""
    for name, count in [("n_paths", n_paths), ("n_steps", n_steps),
                        ("n_timing_draws", n_timing_draws),
                        ("n_timing_replicates", n_timing_replicates),
                        ("progress_every", progress_every)]:
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError(f"{name} must be a positive integer")

    folder = Path(__file__).resolve().parent
    model = OU_model(theta1, theta2)
    print("Scenario:", scenario, "theta1 =", theta1, "theta2 =", theta2, flush=True)

    if run_path_experiment:
        if n_paths < 10000:
            print("Small-run check: the paper uses 10,000 paths per scenario.", flush=True)
        run_paths(model, folder)

    if run_timing_experiment:
        if n_timing_draws != 500 or n_timing_replicates != 10:
            print("Reference: the original Table 2 used 10 replicates of 500 draws per scenario.", flush=True)
        run_timing(model, folder)

    print(f"[{scenario}] All enabled experiments finished.", flush=True)


if __name__ == "__main__":
    main()
