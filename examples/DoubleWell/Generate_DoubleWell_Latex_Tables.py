"""Write Tables 3 and 4 from the completed double-well experiments.

Table 3 compares exact endpoints with Euler endpoints and reports work
for the terminal exact increment. Table 4 reports means and sample
variances of replicate-average stationary-start timings. No samples are
generated here; theoretical acceptance probabilities use quadrature.

Reference: Paper_H200.pdf, double-well example, Tables 3 and 4.
Run from the project root:
    python3 -m Python.examples.DoubleWell.Generate_DoubleWell_Latex_Tables
"""

import csv
import math
import os
from pathlib import Path

from Python.functions.double_well_model import double_well_model
from . import study_io
from .Plot_Figure3 import read_study


HORIZONS = [0.5, 1.0, 2.0, 4.0]
TIMING_BUCKETS = [
    ("endpoint_proposal", "endpoint proposal"),
    ("bridge_max", r"\textsc{BridgeMax}"),
    ("rbmmax", r"\textsc{RBMMax}"),
    ("phi_bound_poisson", r"$\phi$ bound + Poisson"),
    ("icb_phi_accept", r"\textsc{ICB} + $\phi$ + accept"),
    ("other", "other"),
    ("total", "Total"),
]
METADATA = ["implementation", "sampler_fingerprint", "python_version", "platform"]


def log_sample_variance(values):
    """Compute log(sample variance) without squaring enormous endpoints.

    None means fewer than two observations or a nonfinite observation.
    Negative infinity means a sample variance of exactly zero.
    """
    n = len(values)
    if n < 2 or any(not math.isfinite(value) for value in values):
        return None
    scale = max(abs(value) for value in values)
    if scale == 0:
        return -math.inf
    scaled_values = [value / scale for value in values]
    mean = math.fsum(scaled_values) / n
    squared_distances = math.fsum((value - mean) ** 2 for value in scaled_values)
    if squared_distances == 0:
        return -math.inf
    return 2 * math.log(scale) + math.log(squared_distances) - math.log(n - 1)


def format_log_number(log_value, digits=3):
    """Display very small ratios without first rounding them to float zero."""
    if log_value is None:
        return "--"
    if log_value == -math.inf:
        return f"{0:.{digits}f}"
    if not math.isfinite(log_value):
        raise ValueError("A logarithmic table value must be finite or negative infinity")
    exponent = math.floor(log_value / math.log(10))
    if -digits <= exponent < 4:
        return f"{math.exp(log_value):.{digits}f}"
    mantissa = math.exp(log_value - exponent * math.log(10))
    mantissa = round(mantissa, 2)
    if mantissa >= 10:
        mantissa /= 10
        exponent += 1
    return rf"${mantissa:.2f}\times 10^{{{exponent}}}$"


def format_number(value, digits=3):
    """Format a finite table value, preserving tiny positive values."""
    if value is None:
        return "--"
    if not math.isfinite(value):
        raise ValueError("A table value is not finite")
    if value > 0:
        return format_log_number(math.log(value), digits)
    return f"{value:.{digits}f}"


def summarize_table3(exact_rows, euler_rows, model):
    """Use complete endpoint samples, never Euler samples trimmed for plots."""
    summary = []
    acceptance_cache = {}
    checked = 0

    for horizon in HORIZONS:
        exact = [row for row in exact_rows if row["T"] == horizon]
        n = len(exact)
        if n == 0:
            raise ValueError(f"Table 3: no exact endpoints for T={horizon}")
        exact_log_variance = log_sample_variance([row["endpoint"] for row in exact])
        log_ratios = []
        euler_counts = []
        for level in range(1, 6):
            euler = [row for row in euler_rows if row["T"] == horizon and row["level"] == level]
            if not euler:
                raise ValueError(f"Table 3: no Euler endpoints for T={horizon}, level={level}")
            euler_counts.append(len(euler))
            log_ratio = None
            # One explosion invalidates the full level's sample variance.
            # Keeping only finite endpoints would change the experiment.
            if not any(row["euler_exploded"] for row in euler):
                euler_log_variance = log_sample_variance([row["endpoint"] for row in euler])
                if (exact_log_variance is not None and euler_log_variance is not None
                        and euler_log_variance != -math.inf):
                    log_ratio = exact_log_variance - euler_log_variance
            log_ratios.append(log_ratio)

        inverse_logs = []
        for row in exact:
            key = (row["start_value"], row["delta"])
            if key not in acceptance_cache:
                if "log_acceptance_probability" in model:
                    log_probability = model["log_acceptance_probability"](*key)
                else:
                    probability = model["acceptance_probability"](*key)
                    if not 0 < probability <= 1:
                        raise ValueError("The theoretical acceptance probability must lie in (0, 1]")
                    log_probability = math.log(probability)
                if not math.isfinite(log_probability) or log_probability > 0:
                    raise ValueError("The theoretical log acceptance probability is invalid")
                acceptance_cache[key] = log_probability
            inverse_logs.append(-acceptance_cache[key])
            checked += 1
            if checked == 1 or checked % 100 == 0 or checked == len(exact_rows):
                print("Table 3: checked theoretical acceptance for", checked,
                      "of", len(exact_rows), "terminal increments.", flush=True)

        # Log-sum-exp gives log(sum(1/p_i)) without overflowing for tiny p_i.
        largest = max(inverse_logs)
        log_inverse_total = largest + math.log(math.fsum(
            math.exp(value - largest) for value in inverse_logs))
        log_predicted = math.log(n) - log_inverse_total
        proposals = sum(row["proposals"] for row in exact)
        summary.append({
            "T": horizon,
            "delta": exact[0]["delta"],
            "n_exact": n,
            "n_euler": euler_counts[0],
            "proposals_mean": proposals / n,
            "poisson_total_mean": sum(row["total_poisson_points"] for row in exact) / n,
            "log_variance_ratios": log_ratios,
            "log_acceptance_predicted": log_predicted,
            "acceptance_empirical": n / proposals,
        })
    return summary


def positive_integer(value, name):
    """Read an integer CSV count without silently rounding it."""
    number = float(value)
    if isinstance(value, bool) or not math.isfinite(number) or number < 1 or number != int(number):
        raise ValueError(f"{name} must be a positive integer")
    return int(number)


def read_timing_replicates(folder, reference_config):
    """Validate the completed timing export against its status and study."""
    status = study_io.read_complete_status(folder, "timing")
    config = status["config"]
    for name in METADATA + ["gamma"]:
        if config.get(name) != reference_config.get(name):
            raise ValueError(f"Table 4: {name} differs from the endpoint experiment")
    if config.get("gamma") != 1.0 or config.get("delta") != 0.2:
        raise ValueError("Table 4 requires gamma=1 and stationary increments of length 0.2")
    n = positive_integer(config["n_draws"], "timing n_draws")
    target = positive_integer(status["target"], "timing target")
    base_seed = positive_integer(config["timing_seed"], "timing seed")
    with open(folder / "Table4_replicates.csv", newline="") as source:
        rows = list(csv.DictReader(source))
    if len(rows) != target:
        raise ValueError("Table 4: timing export does not contain all completed replicates")
    parsed = []
    seen = set()
    for row in rows:
        replicate = positive_integer(row["replicate"], "replicate")
        if replicate in seen or replicate > target:
            raise ValueError("Table 4: replicate IDs must be exactly 1 through R")
        seen.add(replicate)
        if positive_integer(row["n"], "draw count") != n:
            raise ValueError("Table 4: draw counts differ from the timing configuration")
        if positive_integer(row["seed"], "replicate seed") != base_seed + replicate - 1:
            raise ValueError("Table 4: unexpected replicate seed")
        for name in METADATA:
            if row[name] != config[name]:
                raise ValueError(f"Table 4: replicate {name} differs from its status")
        if float(row["gamma"]) != 1.0 or float(row["delta"]) != 0.2:
            raise ValueError("Table 4: unexpected gamma or delta")
        result = {"replicate": replicate, "n": n}
        for key, label in TIMING_BUCKETS:
            value = float(row[key + "_ms"])
            if not math.isfinite(value) or value < 0:
                raise ValueError("Table 4: timings must be finite and nonnegative")
            result[key + "_ms"] = value
        component_total = math.fsum(result[key + "_ms"] for key, label in TIMING_BUCKETS[:-1])
        if result["total_ms"] <= 0 or not math.isclose(
                component_total, result["total_ms"], rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("Table 4: component times do not sum to a positive total")
        proposals = float(row["proposals_mean"])
        poisson = float(row["poisson_total_mean"])
        acceptance = float(row["acceptance_empirical"])
        if (not math.isfinite(proposals) or proposals < 1 or not math.isfinite(poisson)
                or poisson < 0 or not math.isfinite(acceptance)
                or not math.isclose(acceptance, 1 / proposals, rel_tol=1e-10, abs_tol=1e-12)):
            raise ValueError("Table 4: invalid workload or empirical acceptance")
        parsed.append(result)
    parsed.sort(key=lambda row: row["replicate"])
    return parsed, status


def summarize_table4(replicates):
    """Compute variance across replicate means, not across individual draws."""
    if not replicates:
        raise ValueError("Table 4 requires at least one timing replicate")
    count = len(replicates)
    total_mean = math.fsum(row["total_ms"] / count for row in replicates)
    rows = []
    for key, label in TIMING_BUCKETS:
        values = [row[key + "_ms"] for row in replicates]
        mean = math.fsum(value / count for value in values)
        rows.append({
            "bucket": label,
            "n_replicates": count,
            "n_draws": replicates[0]["n"],
            "mean_ms": mean,
            "log_variance_ms2": log_sample_variance(values),
            "percent_of_mean": 100 * mean / total_mean,
        })
    return rows


def table3_lines(rows):
    """Format the four horizons and all five untrimmed Euler comparisons."""
    n_exact = rows[0]["n_exact"]
    n_euler = rows[0]["n_euler"]
    caption = (
        "Double-well diagnostics for $\\gamma=1$ and $X_0=-1$, using "
        f"{n_exact:,} exact endpoints per horizon and {n_euler:,} Euler endpoints "
        "per horizon and step size. The $T=0.5$ experiment is one exact "
        "increment; the other horizons use exact increments of length $0.2$. "
        r"$\overline P$ and $\overline K_{\rm tot}$ include rejected proposals "
        "for the terminal increment only. Each ratio "
        r"$S_{\rm exact}^2/S_{{\rm EM},\delta}^2$ divides the exact sample variance "
        r"by the Euler--Maruyama sample variance with $\delta=2^{-i}$, $i=1,\ldots,5$. "
        "All Euler endpoints are used, including very large finite values; "
        "a dash means an undefined ratio (fewer than two observations, zero "
        "Euler variance, or any nonfinite/exploded Euler path in that level). "
        "Predicted acceptance is the harmonic mean of conditional one-step "
        "acceptance probabilities; empirical acceptance pools proposal counts."
    )
    lines = [
        r"\begin{table}[!ht]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{3pt}", r"\caption{" + caption + "}",
        r"\label{tab:double-well:3}", r"\begin{tabular}{@{}rrrrrrrrrrr@{}}",
        r"\toprule",
        r"$T$ & $\Delta$ & $\overline P$ & $\overline K_{\rm tot}$ & "
        r"\multicolumn{5}{c}{$S_{\rm exact}^2/S_{{\rm EM},\delta}^2$} & "
        r"$\widehat p_{\rm pred}$ & $\widehat p_{\rm emp}$ \\",
        r"\cmidrule(lr){5-9}",
        r" & & & & $\delta=2^{-1}$ & $\delta=2^{-2}$ & $\delta=2^{-3}$ & "
        r"$\delta=2^{-4}$ & $\delta=2^{-5}$ & & \\",
        r"\midrule",
    ]
    for row in rows:
        cells = [f"{row['T']:.1f}", f"{row['delta']:.1f}",
                 format_number(row["proposals_mean"]), format_number(row["poisson_total_mean"])]
        cells.extend(format_log_number(value) for value in row["log_variance_ratios"])
        cells.extend([format_log_number(row["log_acceptance_predicted"], 4),
                      format_number(row["acceptance_empirical"], 4)])
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return lines


def table4_lines(rows):
    """Display the mean and sample variance for each timing bucket."""
    count = rows[0]["n_replicates"]
    n = rows[0]["n_draws"]
    caption = (
        "Stationary-start double-well timing for $\\gamma=1$ and $\\Delta=0.2$, "
        f"using $R={count}$ independent replicates of {n:,} accepted increments. "
        r"Mean ms/draw and sample variance in $(\mathrm{ms}/\mathrm{draw})^2$ "
        r"are calculated across replicate-average times, with denominator $R-1$. "
        "The variance is not a variance of individual draw times or a standard "
        "error of the overall mean. Percentages divide the mean bucket time "
        "by the mean total time. Rejected proposals are included. "
        r"\textsc{RBMMax} remains vanilla; the ICB boundary-to-boundary sampler "
        "uses time reversal. A dash denotes an undefined one-replicate variance."
    )
    lines = [r"\begin{table}[!ht]", r"\centering", r"\small",
             r"\caption{" + caption + "}", r"\label{tab:double-well:4}",
             r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
             r"Bucket & Mean ms/draw & Variance $(\mathrm{ms}/\mathrm{draw})^2$ & \% of mean \\",
             r"\midrule"]
    for row in rows:
        if row["bucket"] == "Total":
            lines.append(r"\midrule")
        cells = [row["bucket"], format_number(row["mean_ms"]),
                 format_log_number(row["log_variance_ms2"]),
                 "" if row["bucket"] == "Total" else format_number(row["percent_of_mean"], 1)]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return lines


def main():
    """Validate every input, then atomically replace the two-table fragment."""
    folder = Path(__file__).resolve().parent
    data_folder = folder / study_io.DATA_DIRECTORY
    study = read_study(data_folder)
    replicates, timing_status = read_timing_replicates(data_folder, study["exact_status"]["config"])
    model = double_well_model(1.0)
    table3 = summarize_table3(study["exact_rows"], study["euler_rows"], model)
    table4 = summarize_table4(replicates)
    lines = ["% Double-well diagnostics and timing tables.",
             r"% Requires \usepackage{booktabs} in the main document.", ""]
    lines.extend(table3_lines(table3))
    lines.append("")
    lines.extend(table4_lines(table4))
    output_file = folder / "DoubleWell_Latex_Tables.tex"
    temporary = output_file.with_suffix(".tex.tmp")
    with open(temporary, "w", encoding="utf-8") as output:
        output.write("\n".join(lines) + "\n")
    os.replace(temporary, output_file)
    print("Saved Tables 3 and 4 to:", output_file, flush=True)


def cli():
    """Report a numerical diagnostic failure without suggesting new sampling."""
    try:
        main()
    except ArithmeticError as error:
        print("Numerical acceptance/table calculation failed:", error, flush=True)
        print("The saved simulation data and any existing LaTeX file are unchanged. "
              "Resolve the numerical diagnostic before running the table formatter again; "
              "this error does not require new simulations.", flush=True)
        raise SystemExit(1) from None
    except (FileNotFoundError, ValueError, KeyError, TypeError) as error:
        print("Tables were not written:", error, flush=True)
        print("Check the reported input or completion-status problem in", study_io.DATA_DIRECTORY,
              ". Saved simulation data and any existing LaTeX file are unchanged.", flush=True)
        raise SystemExit(1) from None


if __name__ == "__main__":
    cli()
