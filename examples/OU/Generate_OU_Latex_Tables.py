"""Write both OU tables to OU_Latex_Tables.tex from the saved CSV summaries.

Table 1 contains the stepwise diagnostics. Table 2 contains the mean and
sample variance of replicate-average timings. Both tables include the
standard and strong-reversion scenarios.
The output is a LaTeX fragment for a document that loads booktabs.
Table 1 is read from this folder. Table 2 is read only from the completed
ICB time-reversal benchmark in Table2_ICB_TimeReversal; older timings in
this folder are not substituted. No simulations are repeated and no CSV
files are changed.

Run from the project root:
    python3 -m Python.examples.OU.Generate_OU_Latex_Tables
"""

import csv
import json
import math
from pathlib import Path


scenarios = [
    ("standard", "standard", 2.0, 1.0),
    ("strong_reversion", "strong reversion", 6.0, 3.0),
]

timing_buckets = [
    ("endpoint proposal", "endpoint proposal"),
    ("BridgeMax", r"\textsc{BridgeMax}"),
    ("RBMMax", r"\textsc{RBMMax}"),
    ("phi bound + Poisson", r"$\phi$ bound + Poisson"),
    ("ICB + phi + accept", r"\textsc{ICB} + $\phi$ + accept"),
    ("other", "other"),
    ("Total", "Total"),
]

timing_columns = {
    "endpoint proposal": "endpoint_proposal_ms",
    "BridgeMax": "bridge_max_ms",
    "RBMMax": "rbmmax_ms",
    "phi bound + Poisson": "phi_bound_poisson_ms",
    "ICB + phi + accept": "icb_phi_accept_ms",
    "other": "other_ms",
    "Total": "total_ms",
}

table2_implementation = "icb_iv_time_reversal_v1"


def read_summary(filename, scenario, theta1, theta2):
    """Read a summary and check its scenario, OU parameters, and increment."""
    rows = []
    with open(filename, "r", newline="") as source:
        reader = csv.DictReader(source)
        for row in reader:
            if row["scenario"] != scenario:
                raise ValueError(f"{filename.name}: unexpected scenario")
            if float(row["theta1"]) != theta1 or float(row["theta2"]) != theta2:
                raise ValueError(f"{filename.name}: unexpected OU parameters")
            if float(row["delta"]) != 1.0:
                raise ValueError(f"{filename.name}: expected increments of length 1")
            rows.append(row)

    if len(rows) == 0:
        raise ValueError(f"{filename.name}: no summary rows found")
    return rows


def common_count(groups, column):
    """Check that every row in both scenarios reports the same sample count."""
    expected = None
    for label, rows in groups:
        for row in rows:
            value = float(row[column])
            if not math.isfinite(value) or value < 1 or value != math.floor(value):
                raise ValueError(f"{column} must be a positive integer")
            if expected is None:
                expected = int(value)
            elif value != expected:
                raise ValueError(f"{column} differs between summary rows or scenarios")
    return expected


def validate_table2_run(groups):
    """Require completed timing runs from one implementation and environment."""
    if len(groups) != 2 or any(len(rows) == 0 for label, rows in groups):
        raise ValueError("Table 2: both completed scenario summaries are required")

    reference = None
    for label, rows in groups:
        for row in rows:
            if row.get("implementation") != table2_implementation:
                raise ValueError(
                    "Table 2: every row must use implementation " + table2_implementation
                )

            metadata = {}
            for name in ["sampler_fingerprint", "python_version", "platform"]:
                value = row.get(name)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"Table 2: {name} is required in every row")
                metadata[name] = value

            fingerprint = metadata["sampler_fingerprint"]
            if len(fingerprint) != 64 or any(
                character not in "0123456789abcdefABCDEF" for character in fingerprint
            ):
                raise ValueError("Table 2: sampler_fingerprint must be a SHA256 string")

            if reference is None:
                reference = metadata
            for name in metadata:
                if metadata[name] != reference[name]:
                    raise ValueError(f"Table 2: {name} differs between rows or scenarios")

            for name in ["n_replicates", "target_replicates", "n_draws_per_replicate"]:
                if row.get(name) is None or row[name] == "":
                    raise ValueError(f"Table 2: {name} is required in every row")

    n_replicates = common_count(groups, "n_replicates")
    target_replicates = common_count(groups, "target_replicates")
    common_count(groups, "n_draws_per_replicate")
    if n_replicates != target_replicates:
        raise ValueError(
            "Table 2: incomplete benchmark; completed replicates must equal "
            "target_replicates. Finish both timing runs before generating the tables."
        )


def validate_timing_status(filename, rows):
    """Reject stale summaries when a timing run or its final export is unfinished."""
    with open(filename, "r", encoding="utf-8") as source:
        status = json.load(source)
    if not isinstance(status, dict) or status.get("complete") is not True:
        raise ValueError(
            f"Table 2: {filename.name}: timing run is not complete. "
            "Resume the benchmark and finish its exports before generating the tables."
        )
    if not rows:
        raise ValueError("Table 2: a completed timing summary is required")

    config = status.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"Table 2: {filename.name}: run config is missing")
    reference = rows[0]
    for name in ["scenario", "implementation", "sampler_fingerprint", "python_version", "platform"]:
        if config.get(name) is None or config[name] != reference.get(name):
            raise ValueError(f"Table 2: status {name} does not match the summary")
    for name in ["theta1", "theta2", "delta"]:
        value = config.get(name)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"Table 2: status {name} must be a number")
        if not math.isfinite(value) or value != float(reference[name]):
            raise ValueError(f"Table 2: status {name} does not match the summary")

    expected_counts = {
        "completed_replicates": common_count([("summary", rows)], "n_replicates"),
        "target_replicates": common_count([("summary", rows)], "target_replicates"),
    }
    for name, expected in expected_counts.items():
        value = status.get(name)
        if type(value) is not int or value < 1 or value != expected:
            raise ValueError(f"Table 2: status {name} does not match the summary")
    if status["completed_replicates"] != status["target_replicates"]:
        raise ValueError("Table 2: status reports incomplete timing replicates")

    n_draws = config.get("n_draws")
    expected_draws = common_count([("summary", rows)], "n_draws_per_replicate")
    if type(n_draws) is not int or n_draws != expected_draws:
        raise ValueError("Table 2: status n_draws does not match the summary")


def add_legacy_variances(summary, replicates):
    """Fill historical variances explicitly; the default main does not use this."""
    n_replicates = common_count([("summary", summary)], "n_replicates")
    n_draws = common_count([("summary", summary)], "n_draws_per_replicate")
    if len(replicates) != n_replicates:
        raise ValueError("Table 2: saved replicate count does not match the summary")
    if common_count([("replicates", replicates)], "n") != n_draws:
        raise ValueError("Table 2: draws per replicate do not match the summary")

    names = [row["bucket"] for row in summary]
    if len(names) != len(timing_columns) or set(names) != set(timing_columns):
        raise ValueError("Table 2: unexpected, missing, or duplicate summary buckets")

    reference = summary[0]
    replicate_ids = []
    for row in summary + replicates:
        if row["scenario"] != reference["scenario"]:
            raise ValueError("Table 2: saved timing scenarios do not match")
        for name in ["theta1", "theta2", "delta"]:
            value = float(row[name])
            if not math.isfinite(value) or value != float(reference[name]):
                raise ValueError(f"Table 2: saved timing {name} values do not match")
    for row in replicates:
        replicate_id = float(row["replicate"])
        if not math.isfinite(replicate_id) or replicate_id != math.floor(replicate_id):
            raise ValueError("Table 2: saved replicate IDs must be integers")
        replicate_ids.append(int(replicate_id))
    if sorted(replicate_ids) != list(range(1, n_replicates + 1)):
        raise ValueError("Table 2: saved replicate IDs must be unique and run from 1 to R")

    updated = []
    for row in summary:
        values = []
        column = timing_columns[row["bucket"]]
        for replicate in replicates:
            value = float(replicate[column])
            if not math.isfinite(value) or value < 0:
                raise ValueError("Table 2: saved replicate times must be finite and nonnegative")
            values.append(value)
        mean = sum(values) / n_replicates
        summary_mean = float(row["mean_ms"])
        if not math.isfinite(summary_mean) or not math.isclose(
            mean, summary_mean, rel_tol=1e-10, abs_tol=1e-12,
        ):
            raise ValueError("Table 2: saved replicate means do not match the summary")

        variance = None
        if n_replicates > 1:
            squared_distances = 0.0
            for value in values:
                squared_distances += (value - mean) ** 2
            variance = squared_distances / (n_replicates - 1)
        updated_row = row.copy()
        updated_row["variance_ms2"] = variance
        updated.append(updated_row)

    return updated


def format_number(value, digits, allow_missing=False, show_small=False):
    """Round one displayed value, leaving the source CSV unchanged."""
    if value is None or value == "":
        if allow_missing:
            return "--"
        raise ValueError("A required table value is missing")

    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Table values must be finite")
    text = f"{number:.{digits}f}"

    if float(text) == 0:
        # Do not make a small positive timing or percentage look exactly zero.
        if show_small and number > 0:
            smallest_displayed = 10 ** (-digits)
            return f"$<{smallest_displayed:.{digits}f}$"
        # Avoid displaying a rounded negative value as, for example, -0.00.
        return f"{0:.{digits}f}"
    return text


def format_variance(value, n_replicates):
    """Format a nonnegative variance, retaining precision for small values."""
    if value is None or value == "":
        if n_replicates == 1:
            return "--"
        raise ValueError("Table 2: variance_ms2 is required for multiple replicates")

    variance = float(value)
    if not math.isfinite(variance) or variance < 0:
        raise ValueError("Table 2: variance_ms2 must be finite and nonnegative")
    if 0 < variance < 0.001:
        # Two decimal places in the mantissa give three significant digits.
        mantissa, exponent = f"{variance:.2e}".split("e")
        return rf"${mantissa}\times 10^{{{int(exponent)}}}$"
    return format_number(variance, 3)


def table1_lines(groups):
    """Build the nine-column diagnostics table, with six rows per scenario."""
    n = common_count(groups, "n")
    n_latex = f"{n:,}".replace(",", "{,}")
    caption = (
        r"Diagnostics for stepwise exact simulation of the Ornstein--Uhlenbeck "
        r"diffusion with $X_0=1.5$. Each row summarizes the terminal transition "
        r"$X_{T-\Delta}\to X_T$ using $n=" + n_latex + r"$ accepted increments. "
        r"Here $\overline P$ is the mean number of proposals and "
        r"$\overline K_{\rm tot}$ is the mean total Poisson count, including "
        r"accepted and rejected proposals, per terminal increment. "
        r"$D_T$ is the standardized discrepancy of the sample mean, "
        r"$S_T^2/q(T)$ is the empirical-to-exact variance ratio, and "
        r"$\widehat p_{\rm acc}^{\mathrm{pred}}(\Delta,T)$ and "
        r"$\widehat p_{\rm acc}^{\mathrm{emp}}(\Delta,T)$ are the predicted "
        r"and empirical acceptance probabilities."
    )
    if n == 1:
        caption += " A dash denotes the undefined one-sample variance ratio."

    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{" + caption + "}",
        r"\label{tab:ou:1}",
        r"\begin{tabular}{@{}lrrrrrrrr@{}}",
        r"\toprule",
        r"Scenario & $T$ & $\Delta$ & $\overline P$ & "
        r"$\overline K_{\rm tot}$ & $D_T$ & $S_T^2/q(T)$ & "
        r"$\widehat p_{\rm acc}^{\mathrm{pred}}(\Delta,T)$ & "
        r"$\widehat p_{\rm acc}^{\mathrm{emp}}(\Delta,T)$ \\",
        r"\midrule",
    ]

    for group_number, (label, rows) in enumerate(groups):
        rows_by_time = {}
        for row in rows:
            time = float(row["T"])
            if time in rows_by_time:
                raise ValueError(f"Table 1: duplicate time in {label}")
            if float(row["x0"]) != 1.5:
                raise ValueError(f"Table 1: expected X0 = 1.5 in {label}")
            rows_by_time[time] = row
        if sorted(rows_by_time) != [1, 2, 3, 4, 5, 6]:
            raise ValueError(f"Table 1: expected times 1 through 6 in {label}")

        if group_number > 0:
            lines.append(r"\midrule")
        for time in range(1, 7):
            row = rows_by_time[time]
            cells = [
                label,
                format_number(row["T"], 1),
                format_number(row["delta"], 1),
                format_number(row["proposals_mean"], 3),
                format_number(row["poisson_total_mean"], 3),
                format_number(row["D_T"], 2),
                format_number(row["variance_ratio"], 3, allow_missing=(n == 1)),
                format_number(row["acceptance_predicted"], 4),
                format_number(row["acceptance_empirical"], 4),
            ]
            lines.append(" & ".join(cells) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return lines


def table2_lines(groups, time_reversal=False):
    """Format saved means and variances of replicate-average timing values."""
    if time_reversal:
        validate_table2_run(groups)
    n_replicates = common_count(groups, "n_replicates")
    n_draws = common_count(groups, "n_draws_per_replicate")
    n_latex = f"{n_draws:,}".replace(",", "{,}")
    caption = (
        "Coarse timing breakdown for the stationary-start one-step "
        "Ornstein--Uhlenbeck benchmark, using " + str(n_replicates)
        + r" independent replicates of $n=" + n_latex
        + r"$ accepted increments of length $\Delta=1$ per scenario. "
        r"The mean in ms/draw and sample variance in $(\mathrm{ms}/\mathrm{draw})^2$ "
        r"are computed across the $R=" + str(n_replicates)
        + r"$ replicate-average timings, with denominator $R-1$ for the variance. "
        r"This variance describes replicate-average times, not individual draw "
        r"times or the standard error of the overall mean. Component times include accepted and "
        r"rejected proposals. Percentages divide the unrounded mean component "
        r"time by the unrounded mean total time for the same scenario. "
        r"Positive means and percentages that would round to zero are shown "
        r"as upper bounds; positive variances below $0.001$ use scientific notation."
    )
    if n_replicates == 1:
        caption += " A dash denotes the undefined variance for one replicate."
    if time_reversal:
        caption += (
            r" The ICB boundary-to-boundary sampler uses time reversal; "
            r"\textsc{RBMMax} remains vanilla."
        )
    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\small",
        r"\caption{" + caption + "}",
        r"\label{tab:ou-algorithm01-vanilla-stationary-timing-10rep-buckets}",
        r"\begin{tabular}{@{}llrrr@{}}",
        r"\toprule",
        r"Scenario & Bucket & Mean ms/draw & Variance $(\mathrm{ms}/\mathrm{draw})^2$ & \% of mean \\",
        r"\midrule",
    ]

    expected_buckets = []
    for name, latex_name in timing_buckets:
        expected_buckets.append(name)

    for group_number, (label, rows) in enumerate(groups):
        rows_by_bucket = {}
        for row in rows:
            if "variance_ms2" not in row:
                raise ValueError(
                    "Table 2: variance_ms2 is missing. Regenerate the timing "
                    "summaries from the saved Table2_<scenario>_replicates.csv "
                    "files before formatting; no new simulations are needed."
                )
            name = row["bucket"]
            if name in rows_by_bucket:
                raise ValueError(f"Table 2: duplicate bucket in {label}")
            rows_by_bucket[name] = row
        if sorted(rows_by_bucket) != sorted(expected_buckets):
            raise ValueError(f"Table 2: unexpected or missing buckets in {label}")

        if group_number > 0:
            lines.append(r"\midrule")
        for name, latex_name in timing_buckets:
            row = rows_by_bucket[name]
            mean = format_number(row["mean_ms"], 3, show_small=True)
            variance = format_variance(row["variance_ms2"], n_replicates)

            if name == "Total":
                # Total variance includes covariances; do not sum component variances.
                lines.append(r"\midrule")
                cells = ["Total", "", mean, variance, ""]
            else:
                percent = format_number(row["percent_of_mean"], 1, show_small=True)
                cells = [label, latex_name, mean, variance, percent]
            lines.append(" & ".join(cells) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return lines


def main():
    """Read completed summaries, then write one file containing both tables."""
    folder = Path(__file__).resolve().parent
    timing_folder = folder / "Table2_ICB_TimeReversal"
    table1_groups = []
    table2_groups = []
    for scenario, label, theta1, theta2 in scenarios:
        table1 = read_summary(
            folder / f"Table1_{scenario}_summary.csv", scenario, theta1, theta2,
        )
        table2 = read_summary(
            timing_folder / f"Table2_{scenario}_summary.csv", scenario, theta1, theta2,
        )
        validate_timing_status(timing_folder / f"Table2_{scenario}_status.json", table2)
        table1_groups.append((label, table1))
        table2_groups.append((label, table2))

    # Build and validate both tables before replacing any existing output.
    lines = [
        "% OU diagnostics and timing tables.",
        "% Table 2 implementation: " + table2_implementation + ".",
        r"% Requires \usepackage{booktabs} in the main document.",
        "",
    ]
    lines.extend(table1_lines(table1_groups))
    lines.append("")
    lines.extend(table2_lines(table2_groups, time_reversal=True))

    output_file = folder / "OU_Latex_Tables.tex"
    with open(output_file, "w", encoding="utf-8") as output:
        output.write("\n".join(lines) + "\n")
    print("Saved both tables to:", output_file)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, KeyError) as error:
        print("Tables were not written:", error)
        print(
            "Keep the Table 1 summaries beside this script and finish both "
            "Table 2 runs in Table2_ICB_TimeReversal. Existing OU_Latex_Tables.tex "
            "has not been replaced."
        )
        raise SystemExit(1) from None
