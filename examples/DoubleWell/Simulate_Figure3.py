"""Generate double-well data for Figure 3 and Tables 3 and 4.

The model is dX = gamma * (X - X**3) dt + dW, starting from X0 = -1.

Each phase can be switched off and has its own reset flag. An enabled phase
starts fresh when its reset flag is True, clearing only that phase's saved
checkpoints and outputs. Set its reset flag to False to reuse completed
batches or timing replicates with unchanged code and settings. An interrupted
batch is repeated from its original seeds. Plotting and numerical acceptance
diagnostics are separate scripts.

Run from the project root:
    python3 -m Python.examples.DoubleWell.Simulate_Figure3
"""

import fcntl
import math
from pathlib import Path
import random

from Python.functions.Algorithm01_Vanilla import Algorithm01_Vanilla
from Python.functions.double_well_model import double_well_model
from Python.examples.DoubleWell import study_io


gamma = 1.0
x0 = -1.0
horizons = [0.5, 1.0, 2.0, 4.0]
step_delta = 0.2
euler_steps = [2.0 ** (-level) for level in range(1, 6)]

run_exact_experiment = False
run_euler_experiment = False
run_timing_experiment = True

# Start each enabled phase fresh, including after sampler updates.
# Set a phase's reset flag False to resume with unchanged code and settings.
reset_exact_experiment = True
reset_euler_experiment = True
reset_timing_experiment = True

n_paths = 10000
n_euler_paths = 10000
n_timing_draws = 500
n_timing_replicates = 100  # 
timing_delta = 0.2

progress_every = 100
checkpoint_every = 100  # Save complete batches of exact/Euler paths.

# Non-overlapping integer seed ranges, one seed per path or timing replicate.
# A separate seed for every path makes checkpoint recovery straightforward.
seed_spacing = 10000000
direct_seed = 31000001
stepwise_seed = 41000001
euler_seed = 51000001
timing_seed = 101000001

timing_buckets = [
    ("endpoint_proposal", "endpoint proposal"),
    ("bridge_max", "BridgeMax"),
    ("rbmmax", "RBMMax"),
    ("phi_bound_poisson", "phi bound + Poisson"),
    ("icb_phi_accept", "ICB + phi + accept"),
    ("other", "other"),
    ("total", "Total"),
]


def show_progress(number, total):
    """Report the first, every hundredth, and final completed item."""
    return number == 1 or number % progress_every == 0 or number == total


def draw_statistics(draw):
    """Keep the accepted endpoint and counters including rejected proposals."""
    record = {"endpoint": draw["endpoint"]}
    for name in ["proposals", "accepted_poisson_points", "total_poisson_points",
                 "bridge_max_calls", "rbmmax_calls", "icb_calls"]:
        record[name] = draw["stats"][name]
    return record


def validate_draw(record):
    """Reject incomplete or non-finite exact-sampler records."""
    for name in ["start_value", "endpoint"]:
        if not math.isfinite(record[name]):
            raise ValueError(f"Exact draw has non-finite {name}")
    for name in ["proposals", "accepted_poisson_points", "total_poisson_points",
                 "bridge_max_calls", "rbmmax_calls", "icb_calls"]:
        value = record[name]
        if type(value) is not int or value < 0:
            raise ValueError(f"Invalid sampler counter {name}")
    if record["proposals"] < 1:
        raise ValueError("An accepted draw must have at least one proposal")


def exact_configuration():
    config = study_io.implementation_metadata()
    config.update({"gamma": gamma, "x0": x0, "horizons": horizons,
                   "step_delta": step_delta, "direct_seed": direct_seed,
                   "stepwise_seed": stepwise_seed})
    return config


def one_exact_path(model, kind, number):
    """Return every increment, not just those shown in Figure 3."""
    if kind == "direct":
        path_seed = direct_seed + number - 1
        delta = 0.5
        steps = 1
    else:
        path_seed = stepwise_seed + number - 1
        delta = step_delta
        steps = round(max(horizons) / delta)
    random.seed(path_seed)
    value = x0
    records = []
    for step in range(1, steps + 1):
        start = value
        draw = Algorithm01_Vanilla(model, x=start, T=delta)
        value = draw["endpoint"]
        record = {
            "gamma": gamma, "x0": x0, "kind": kind, "path": number,
            "seed": path_seed, "step": step, "T": step * delta,
            "delta": delta, "start_value": start,
        }
        record.update(draw_statistics(draw))
        record["implementation"] = study_io.IMPLEMENTATION
        records.append(record)
    if show_progress(number, n_paths):
        print(f"[exact {kind}] Completed path {number}/{n_paths}, "
              f"through T={steps * delta:g}.", flush=True)
    return records


def validate_exact_path(kind, number, records):
    delta = 0.5 if kind == "direct" else step_delta
    steps = 1 if kind == "direct" else round(max(horizons) / delta)
    seed = (direct_seed if kind == "direct" else stepwise_seed) + number - 1
    if len(records) != steps:
        raise ValueError("Incomplete exact path checkpoint")
    previous = x0
    for step, record in enumerate(records, start=1):
        expected = {"gamma": gamma, "x0": x0, "kind": kind, "path": number,
                    "seed": seed, "step": step, "delta": delta,
                    "start_value": previous, "implementation": study_io.IMPLEMENTATION}
        if any(record[name] != value for name, value in expected.items()):
            raise ValueError("Exact path checkpoint metadata or chaining does not match")
        if not math.isclose(record["T"], step * delta, rel_tol=0, abs_tol=1e-12):
            raise ValueError("Exact path checkpoint time does not match")
        validate_draw(record)
        previous = record["endpoint"]


def run_exact(model, folder):
    """Save all exact increments and the four terminal samples separately."""
    if reset_exact_experiment:
        clear_phase_outputs(folder, "exact")
    config = exact_configuration()
    study_io.begin_phase(folder, "exact", config, n_paths)
    all_records = []
    for kind in ["direct", "stepwise"]:
        checkpoint_config = dict(config, kind=kind)
        paths = study_io.run_checkpointed(
            folder, "exact_" + kind, checkpoint_config, n_paths, checkpoint_every,
            lambda number: one_exact_path(model, kind, number),
            lambda number, records: validate_exact_path(kind, number, records),
        )
        for records in paths:
            all_records.extend(records)

    terminal_records = []
    for record in all_records:
        wanted = [0.5] if record["kind"] == "direct" else horizons[1:]
        if record["T"] in wanted:
            terminal_records.append(record)
    study_io.write_csv(folder / "Figure3_exact_steps.csv", all_records)
    study_io.write_csv(folder / "Figure3_exact_draws.csv", terminal_records)
    study_io.save_status(folder, "exact", config, n_paths, n_paths, complete=True)


def euler_configuration():
    config = study_io.implementation_metadata()
    config.update({"gamma": gamma, "x0": x0, "horizons": horizons,
                   "euler_steps": euler_steps, "euler_seed": euler_seed,
                   "seed_spacing": seed_spacing})
    return config


def one_euler_path(model, level, number):
    """Advance one Euler path to T=4 and retain all four target horizons.

    No finite value is clipped. If floating-point arithmetic overflows, keep
    a signed infinity and the first failed step. This failure is part of the
    comparison data, not grounds for replacing or discarding the path.
    """
    delta = euler_steps[level - 1]
    path_seed = euler_seed + (level - 1) * seed_spacing + number - 1
    random.seed(path_seed)
    value = x0
    explosion_step = None
    records = []
    for step in range(1, round(max(horizons) / delta) + 1):
        if explosion_step is None:
            noise = math.sqrt(delta) * random.gauss(0.0, 1.0)
            previous = value
            try:
                value = previous + model["alpha"](previous) * delta + noise
            except ArithmeticError:
                # Cubic drift points opposite to a sufficiently large value.
                value = math.copysign(math.inf, -previous)
            if not math.isfinite(value):
                if math.isnan(value):
                    value = math.copysign(math.inf, -previous)
                explosion_step = step
        time = step * delta
        if time in horizons:
            # JSON forbids infinity: store its explicit string spelling.
            endpoint = value if math.isfinite(value) else str(value)
            records.append({
                "gamma": gamma, "x0": x0, "path": number, "seed": path_seed,
                "T": time, "delta": delta, "level": level, "endpoint": endpoint,
                "euler_exploded": explosion_step is not None,
                "euler_explosion_step": explosion_step,
                "implementation": study_io.IMPLEMENTATION,
            })
    if show_progress(number, n_euler_paths):
        print(f"[Euler level {level}/5, delta={delta:g}] "
              f"Completed path {number}/{n_euler_paths} through T=4.", flush=True)
    return records


def validate_euler_path(level, number, records):
    if len(records) != len(horizons):
        raise ValueError("Incomplete Euler path checkpoint")
    delta = euler_steps[level - 1]
    seed = euler_seed + (level - 1) * seed_spacing + number - 1
    first_failure = None
    failed_endpoint = None
    last_finite_step = 0
    for time, record in zip(horizons, records):
        expected = {"gamma": gamma, "x0": x0, "path": number, "seed": seed,
                    "T": time, "delta": delta, "level": level,
                    "implementation": study_io.IMPLEMENTATION}
        if any(record[name] != value for name, value in expected.items()):
            raise ValueError("Euler path checkpoint metadata does not match")
        endpoint = float(record["endpoint"])
        exploded = record["euler_exploded"]
        failed_step = record["euler_explosion_step"]
        if math.isnan(endpoint) or type(exploded) is not bool:
            raise ValueError("Invalid Euler endpoint or explosion flag")
        if exploded != (not math.isfinite(endpoint)):
            raise ValueError("Euler endpoint and explosion flag disagree")
        if exploded:
            if type(failed_step) is not int or not 1 <= failed_step <= round(time / delta):
                raise ValueError("Invalid Euler explosion step")
            if failed_step <= last_finite_step:
                raise ValueError("Euler failure precedes a saved finite endpoint")
            if first_failure is None:
                first_failure = failed_step
                failed_endpoint = endpoint
            elif failed_step != first_failure or endpoint != failed_endpoint:
                raise ValueError("Euler overflow must retain its first step and sign")
        elif failed_step is not None:
            raise ValueError("A finite Euler endpoint cannot have an explosion step")
        elif first_failure is not None:
            raise ValueError("An overflowed Euler path cannot become finite again")
        else:
            last_finite_step = round(time / delta)


def run_euler(model, folder):
    """Save each grid independently so completed grids survive interruption."""
    if reset_euler_experiment:
        clear_phase_outputs(folder, "euler")
    config = euler_configuration()
    study_io.begin_phase(folder, "euler", config, n_euler_paths)
    all_records = []
    for level in range(1, len(euler_steps) + 1):
        paths = study_io.run_checkpointed(
            folder, f"euler_{level}", dict(config, level=level),
            n_euler_paths, checkpoint_every,
            lambda number: one_euler_path(model, level, number),
            lambda number, records: validate_euler_path(level, number, records),
        )
        for records in paths:
            all_records.extend(records)
    study_io.write_csv(folder / "Figure3_euler_draws.csv", all_records)
    study_io.save_status(folder, "euler", config, n_euler_paths, n_euler_paths, complete=True)


def timing_configuration():
    config = study_io.implementation_metadata()
    config.update({"gamma": gamma, "delta": timing_delta,
                   "n_draws": n_timing_draws, "timing_seed": timing_seed})
    return config


def one_timing_replicate(model, config, number):
    """Time accepted increments from independent stationary starting values."""
    seed = timing_seed + number - 1
    random.seed(seed)
    print(f"[Table 4] Starting replicate {number}/{n_timing_replicates} "
          f"(seed {seed}, delta={timing_delta:g}).", flush=True)
    # Draw ALL stationary starts before any of the measured increments.
    starts = model["sample_stationary"](n=n_timing_draws)
    records = []
    for number_in_replicate, start in enumerate(starts, start=1):
        draw = Algorithm01_Vanilla(model, x=start, T=timing_delta, record_timing=True)
        record = {"gamma": gamma, "delta": timing_delta, "replicate": number,
                  "draw": number_in_replicate, "seed": seed, "start_value": start}
        record.update(draw_statistics(draw))
        for name, label in timing_buckets:
            record[name + "_seconds"] = draw["timing"][name]
        record["implementation"] = study_io.IMPLEMENTATION
        records.append(record)
        if show_progress(number_in_replicate, n_timing_draws):
            print(f"[Table 4] Replicate {number}/{n_timing_replicates}, "
                  f"completed draw {number_in_replicate}/{n_timing_draws}.", flush=True)

    proposals = sum(record["proposals"] for record in records)
    poisson = sum(record["total_poisson_points"] for record in records)
    replicate = {"gamma": gamma, "delta": timing_delta, "replicate": number,
                 "seed": seed, "n": n_timing_draws,
                 "proposals_mean": proposals / n_timing_draws,
                 "poisson_total_mean": poisson / n_timing_draws,
                 "acceptance_empirical": n_timing_draws / proposals}
    for name in ["implementation", "sampler_fingerprint", "python_version", "platform"]:
        replicate[name] = config[name]
    for name, label in timing_buckets:
        replicate[name + "_ms"] = 1000 * sum(
            record[name + "_seconds"] for record in records) / n_timing_draws
    return {"replicate": replicate, "draws": records}


def validate_timing_replicate(config, number, result):
    replicate = result["replicate"]
    records = result["draws"]
    seed = timing_seed + number - 1
    if len(records) != n_timing_draws or replicate["n"] != n_timing_draws:
        raise ValueError("Incomplete timing replicate checkpoint")
    for record in [replicate] + records:
        for name, expected in [("gamma", gamma), ("delta", timing_delta),
                               ("replicate", number), ("seed", seed),
                               ("implementation", study_io.IMPLEMENTATION)]:
            if record[name] != expected:
                raise ValueError(f"Timing checkpoint {name} does not match")
    for name in ["sampler_fingerprint", "python_version", "platform"]:
        if replicate[name] != config[name]:
            raise ValueError(f"Timing checkpoint {name} does not match")
    for index, record in enumerate(records, start=1):
        if record["draw"] != index:
            raise ValueError("Timing draw numbers must be consecutive")
        validate_draw(record)
        components = 0.0
        for name, label in timing_buckets:
            value = record[name + "_seconds"]
            if not math.isfinite(value) or value < 0:
                raise ValueError("Timing buckets must be finite and nonnegative")
            if name != "total":
                components += value
        if not math.isclose(components, record["total_seconds"], rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("Timing components do not sum to the total")
    for name, label in timing_buckets:
        expected = 1000 * sum(record[name + "_seconds"] for record in records) / n_timing_draws
        if not math.isclose(replicate[name + "_ms"], expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("Replicate mean does not match its raw timings")


def clear_phase_outputs(folder, phase):
    """Remove one phase's saved data while main holds the study lock."""
    exports, checkpoint_names = {
        "exact": (("Figure3_exact_steps.csv", "Figure3_exact_draws.csv"),
                  ("exact_direct", "exact_stepwise")),
        "euler": (("Figure3_euler_draws.csv",),
                  tuple(f"euler_{level}" for level in range(1, 6))),
        "timing": (("Table4_draws.csv", "Table4_replicates.csv"), ("timing",)),
    }[phase]
    # Invalidate the completed status before deleting any measurements.
    for name in (f"{phase}_status.json",) + exports:
        for filename in (folder / name, folder / (name + ".tmp")):
            filename.unlink(missing_ok=True)
    for name in checkpoint_names:
        checkpoint_folder = folder / "checkpoints" / name
        for pattern in ("batch_*.json", "batch_*.json.tmp"):
            for filename in checkpoint_folder.glob(pattern):
                filename.unlink()
    print(f"[{phase}] Cleared previous checkpoints and outputs.", flush=True)


def clear_timing_outputs(folder):
    """Remove timing checkpoints and exports without changing the other phases."""
    clear_phase_outputs(folder, "timing")


def run_timing(model, folder):
    """Start fresh or resume timing replicates, as selected in the settings."""
    if reset_timing_experiment:
        clear_timing_outputs(folder)
    config = timing_configuration()
    study_io.begin_phase(folder, "timing", config, n_timing_replicates)
    results = study_io.run_checkpointed(
        folder, "timing", config, n_timing_replicates, 1,
        lambda number: one_timing_replicate(model, config, number),
        lambda number, result: validate_timing_replicate(config, number, result),
        lambda completed: study_io.save_status(folder, "timing", config,
                                               n_timing_replicates, completed),
    )
    records = []
    replicates = []
    for result in results:
        records.extend(result["draws"])
        replicates.append(result["replicate"])
    study_io.write_csv(folder / "Table4_draws.csv", records)
    study_io.write_csv(folder / "Table4_replicates.csv", replicates)
    study_io.save_status(folder, "timing", config, n_timing_replicates,
                         n_timing_replicates, complete=True)


def validate_settings():
    for name, value in [("run_exact_experiment", run_exact_experiment),
                        ("run_euler_experiment", run_euler_experiment),
                        ("run_timing_experiment", run_timing_experiment),
                        ("reset_exact_experiment", reset_exact_experiment),
                        ("reset_euler_experiment", reset_euler_experiment),
                        ("reset_timing_experiment", reset_timing_experiment)]:
        if type(value) is not bool:
            raise ValueError(f"{name} must be True or False")
    for name, value in [("n_paths", n_paths), ("n_euler_paths", n_euler_paths),
                        ("n_timing_draws", n_timing_draws),
                        ("n_timing_replicates", n_timing_replicates),
                        ("progress_every", progress_every),
                        ("checkpoint_every", checkpoint_every)]:
        study_io.positive_integer(value, name)
    if max(n_paths, n_euler_paths, n_timing_replicates) >= seed_spacing:
        raise ValueError("Increase the seed spacing before using ten million paths/replicates")
    if horizons != [0.5, 1.0, 2.0, 4.0] or step_delta != 0.2:
        raise ValueError("This Figure 3 driver requires horizons .5,1,2,4 and step_delta=.2")
    if euler_steps != [2.0 ** (-level) for level in range(1, 6)]:
        raise ValueError("Figure 3 requires the five dyadic Euler step sizes")
    if gamma != 1.0 or x0 != -1.0 or timing_delta != 0.2:
        raise ValueError("This paper example requires gamma=1, x0=-1, timing_delta=.2")


def main():
    """Run selected phases; importing the module starts no simulations."""
    validate_settings()
    model = double_well_model(gamma)
    folder = Path(__file__).resolve().parent / study_io.DATA_DIRECTORY
    folder.mkdir(parents=True, exist_ok=True)
    # This Mac's standard-library lock prevents simultaneous copies of this
    # benchmark. The OS releases the lock even after an interrupted process.
    with open(folder / "study.lock", "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another double-well simulation is already running") from None
        print(f"Double-well study: gamma={gamma:g}, X0={x0:g}.", flush=True)
        print("Data directory:", folder, flush=True)
        if run_exact_experiment:
            run_exact(model, folder)
        if run_euler_experiment:
            run_euler(model, folder)
        if run_timing_experiment:
            run_timing(model, folder)
    print("All enabled double-well experiments are complete.", flush=True)


if __name__ == "__main__":
    main()
