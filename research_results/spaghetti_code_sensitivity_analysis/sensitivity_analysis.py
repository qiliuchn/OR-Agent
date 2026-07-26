"""
Activity-aware comparative sensitivity analysis for:
1. spaghetti_code_solution.py
2. driving_final_solution.py

The two controller sources are never edited. Temporary parameter variants are
evaluated in isolated subprocesses and deleted after each run.

The primary local sensitivity summaries are robust to appended spurious
parameters:

* active-parameter mean: mean effect only among parameters with a non-zero
  local score response;
* total L1 effect: sum of parameter effects, which is unchanged by adding
  zero-effect parameters;
* effective dimension: participation ratio of the non-negative parameter
  effects, also unchanged by appending zeros.

Declared-parameter means are retained only as a dilution-prone diagnostic.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import random
import re
import runpy
import shutil
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ANALYSIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = ANALYSIS_DIR.parents[1]
EVAL_PATH = ANALYSIS_DIR / "eval.py"
WORK_DIR = ANALYSIS_DIR / "_sensitivity_work"
PERTURBATION = 0.10
JOINT_TRIALS = 24
RANDOM_SEED = 20260729
MAX_WORKERS = 4
ACTIVITY_TOLERANCE = 1e-9

METRIC_NAMES = (
    "critical_ttc_count",
    "collisions",
    "emergencyStops",
    "emergencyBraking",
    "teleports",
    "avg_speed",
    "speed_variance",
)


CONTROLLERS: dict[str, dict[str, Any]] = {
    "spaghetti": {
        "source": ANALYSIS_DIR / "spaghetti_code_solution.py",
        "parameters": {
            "SAFE_TTC": 5.0,
            "MIN_GAP": 1.6,
            "LANE_CHANGE_GAP_BASE": 10.0,
            "OFFSET_BUFFER": 1.6,
            "ENTRY_DISTANCE": 15.0,
            "ENTRY_TIME": 3.0,
            "LANE_CHANGE_COOLDOWN": 2.5,
            "TARGET_LANE_PATIENCE": 100.0,
            "URGENCY_START_DIST": 400.0,
            "CRITICAL_DIST": 80.0,
            "MIN_ACCEL": 0.3,
            "MAX_ACCEL_BOOST": 1.2,
            "COOPERATION_RANGE": 40.0,
            "COOPERATION_WINDOW": 6.0,
            "SPEED_MATCH_FACTOR": 1.6,
            "GAP_SEARCH_ACCEL": 4.5,
            "GAP_SEARCH_DECEL": 1.8,
            "SWAP_DETECT_RANGE": 50.0,
            "SWAP_COORDINATION_ACCEL": 1.0,
            "SWAP_COORDINATION_DECEL": 2.0,
            "PLATOON_DETECT_RANGE": 18.0,
            "PLATOON_ACCEL": 1.0,
            "PLATOON_DECEL": 1.5,
        },
    },
    "final": {
        "source": ANALYSIS_DIR / "driving_final_solution.py",
        "parameters": {
            "FOLLOW_GAP": 2.0,
            "REACTION_TIME": 0.50,
            "MERGE_BASE_GAP": 6.0,
            "MERGE_TTC": 3.5,
            "MERGE_COOLDOWN": 3.0,
            "ENTRY_DELAY": 2.0,
            "ENTRY_MERGE_DISTANCE": 30.0,
            "RESERVATION_DISTANCE": 16.0,
            "COOP_BRAKE": 2.0,
        },
    },
}


EVAL_RUNNER = """
import importlib.util
import runpy
import sys

variant_path, evaluator_path, root_dir, output_prefix = sys.argv[1:5]
spec = importlib.util.spec_from_file_location(
    "spaghetti_code_solution",
    variant_path,
)
module = importlib.util.module_from_spec(spec)
sys.modules["spaghetti_code_solution"] = module
spec.loader.exec_module(module)
sys.argv = [
    evaluator_path,
    "--root_dir",
    root_dir,
    "--file_output_prefix",
    output_prefix,
]
runpy.run_path(evaluator_path, run_name="__main__")
"""


def replace_parameters(source: str, overrides: dict[str, float]) -> str:
    """Replace the first assignment of each named local scalar constant."""
    result = source
    for parameter, value in overrides.items():
        pattern = re.compile(
            rf"^(\s*{re.escape(parameter)}\s*=\s*)([^#\n]+?)(\s*(?:#.*)?)$",
            flags=re.MULTILINE,
        )
        result, count = pattern.subn(
            lambda match: f"{match.group(1)}{value!r}{match.group(3)}",
            result,
            count=1,
        )
        if count != 1:
            raise RuntimeError(
                f"Expected one assignment for {parameter}, found {count}"
            )
    return result


def parse_evaluation_output(stdout: str) -> dict[str, Any]:
    """Extract the aggregate score and per-case metrics from eval.py output."""
    score_match = re.search(
        r"__SCORE_START__\s*(.*?)\s*__SCORE_END__",
        stdout,
        flags=re.DOTALL,
    )
    if not score_match:
        raise RuntimeError("Evaluator output did not contain a final score")

    metrics_match = re.search(
        r"Metrics for all tests:\s*(.*?)\s*Scores for all tests:",
        stdout,
        flags=re.DOTALL,
    )
    if not metrics_match:
        raise RuntimeError("Evaluator output did not contain per-test metrics")

    metrics_per_test: dict[str, dict[str, Any]] = {}
    for line in metrics_match.group(1).splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        test_name, payload = line.split(":", 1)
        metrics_per_test[test_name.strip()] = ast.literal_eval(payload.strip())

    return {
        "score": float(ast.literal_eval(score_match.group(1).strip())),
        "metrics": metrics_per_test,
    }


def source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def usage_count(source: str, parameter: str) -> int:
    """Count executable reads of a parameter inside driving_actions."""
    tree = ast.parse(source)
    driving_function = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "driving_actions"
    )
    return sum(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id == parameter
        for node in ast.walk(driving_function)
    )


def evaluate_variant(
    controller: str,
    overrides: dict[str, float],
    run_id: str,
) -> dict[str, Any]:
    """Evaluate one isolated source variant without editing a controller."""
    source_path: Path = CONTROLLERS[controller]["source"]
    variant_source = replace_parameters(
        source_path.read_text(encoding="utf-8"),
        overrides,
    )

    run_dir = WORK_DIR / run_id
    run_dir.mkdir(parents=True)
    module_path = run_dir / "spaghetti_code_solution.py"
    module_path.write_text(variant_source, encoding="utf-8")
    output_prefix = str(run_dir.relative_to(ANALYSIS_DIR) / "output_")
    command = [
        sys.executable,
        "-c",
        EVAL_RUNNER,
        str(module_path),
        str(EVAL_PATH),
        str(ROOT_DIR),
        output_prefix,
    ]

    try:
        completed = subprocess.run(
            command,
            cwd=ANALYSIS_DIR,
            text=True,
            capture_output=True,
            timeout=180,
            check=False,
        )
        if (
            completed.returncode != 0
            or "__SANDBOX_SUCCESS__" not in completed.stdout
        ):
            raise RuntimeError(
                f"Evaluation failed for {run_id}\n"
                f"stdout:\n{completed.stdout[-4000:]}\n"
                f"stderr:\n{completed.stderr[-4000:]}"
            )
        result = parse_evaluation_output(completed.stdout)
        result.update({
            "controller": controller,
            "run_id": run_id,
            "overrides": overrides,
        })
        return result
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def metric(result: dict[str, Any], test: str, name: str) -> float:
    return float(result["metrics"].get(test, {}).get(name, 0.0))


def feasible(result: dict[str, Any]) -> bool:
    return (
        metric(result, "case_1", "collisions") == 0.0
        and metric(result, "case_1", "teleports") == 0.0
        and metric(result, "case_1", "avg_speed") >= 8.0
    )


def compact_result(result: dict[str, Any]) -> dict[str, Any]:
    compact = {
        "controller": result["controller"],
        "run_id": result["run_id"],
        "overrides": result["overrides"],
        "score": result["score"],
        "case_0": result["metrics"].get("case_0", {}),
        "case_1": result["metrics"].get("case_1", {}),
        "feasible": feasible(result),
    }
    for key in (
        "parameter",
        "direction",
        "multiplier",
        "parameter_value",
        "multipliers",
        "perturbed_parameters",
    ):
        if key in result:
            compact[key] = result[key]
    return compact


def run_batch(
    jobs: list[dict[str, Any]],
    completed_offset: int,
    total_runs: int,
) -> list[dict[str, Any]]:
    results = []
    completed_count = completed_offset
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                evaluate_variant,
                job["controller"],
                job["overrides"],
                job["run_id"],
            ): job
            for job in jobs
        }
        for future in as_completed(futures):
            job = futures[future]
            result = future.result()
            result.update(job["metadata"])
            results.append(result)
            completed_count += 1
            print(
                f"[{completed_count}/{total_runs}] {job['run_id']}: "
                f"score={result['score']:.6f}",
                flush=True,
            )
    return results


def run_experiments() -> dict[str, Any]:
    """Run baselines, central OAT trials, and structural joint trials."""
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    WORK_DIR.mkdir()

    hashes_before = {
        controller: source_hash(spec["source"])
        for controller, spec in CONTROLLERS.items()
    }
    baseline_jobs = []
    oat_jobs = []
    joint_jobs = []

    for controller, spec in CONTROLLERS.items():
        base_parameters = dict(spec["parameters"])
        baseline_jobs.append({
            "controller": controller,
            "overrides": base_parameters,
            "run_id": f"{controller}_baseline",
            "metadata": {},
        })

        for parameter, base_value in base_parameters.items():
            for direction, multiplier in (
                ("low", 1.0 - PERTURBATION),
                ("high", 1.0 + PERTURBATION),
            ):
                overrides = dict(base_parameters)
                overrides[parameter] = base_value * multiplier
                oat_jobs.append({
                    "controller": controller,
                    "overrides": overrides,
                    "run_id": (
                        f"{controller}_oat_{parameter.lower()}_{direction}"
                    ),
                    "metadata": {
                        "parameter": parameter,
                        "direction": direction,
                        "multiplier": multiplier,
                        "parameter_value": overrides[parameter],
                    },
                })

        source = spec["source"].read_text(encoding="utf-8")
        structural_parameters = [
            parameter
            for parameter in base_parameters
            if usage_count(source, parameter) > 0
        ]
        controller_index = list(CONTROLLERS).index(controller)
        rng = random.Random(RANDOM_SEED + controller_index * 10_000)
        for trial_index in range(JOINT_TRIALS):
            multipliers = {
                parameter: rng.uniform(
                    1.0 - PERTURBATION,
                    1.0 + PERTURBATION,
                )
                for parameter in structural_parameters
            }
            overrides = dict(base_parameters)
            for parameter, multiplier in multipliers.items():
                overrides[parameter] = base_parameters[parameter] * multiplier
            joint_jobs.append({
                "controller": controller,
                "overrides": overrides,
                "run_id": f"{controller}_joint_{trial_index:02d}",
                "metadata": {
                    "multipliers": multipliers,
                    "perturbed_parameters": structural_parameters,
                },
            })

    total_runs = len(baseline_jobs) + len(oat_jobs) + len(joint_jobs)
    try:
        baseline_results = run_batch(baseline_jobs, 0, total_runs)
        oat_results = run_batch(
            oat_jobs,
            len(baseline_results),
            total_runs,
        )
        joint_results = run_batch(
            joint_jobs,
            len(baseline_results) + len(oat_results),
            total_runs,
        )
    finally:
        shutil.rmtree(WORK_DIR, ignore_errors=True)

    hashes_after = {
        controller: source_hash(spec["source"])
        for controller, spec in CONTROLLERS.items()
    }
    return {
        "configuration": {
            "perturbation_fraction": PERTURBATION,
            "joint_trials_per_controller": JOINT_TRIALS,
            "random_seed": RANDOM_SEED,
            "activity_tolerance": ACTIVITY_TOLERANCE,
            "max_workers": MAX_WORKERS,
            "joint_parameter_rule": (
                "perturb only parameters referenced after assignment"
            ),
        },
        "source_integrity": {
            controller: {
                "sha256_before": hashes_before[controller],
                "sha256_after": hashes_after[controller],
                "unchanged": (
                    hashes_before[controller] == hashes_after[controller]
                ),
            }
            for controller in CONTROLLERS
        },
        "baselines": {
            result["controller"]: result
            for result in baseline_results
        },
        "oat_results": oat_results,
        "joint_results": joint_results,
    }


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def permutation_test(
    spaghetti_values: list[float],
    final_values: list[float],
    iterations: int = 20_000,
) -> tuple[float, float]:
    """Exploratory one-sided test for mean(spaghetti) > mean(final)."""
    if not spaghetti_values or not final_values:
        return math.nan, math.nan
    observed = statistics.mean(spaghetti_values) - statistics.mean(final_values)
    pooled = spaghetti_values + final_values
    first_size = len(spaghetti_values)
    rng = random.Random(RANDOM_SEED + 1)
    exceedances = 0
    for _ in range(iterations):
        shuffled = pooled[:]
        rng.shuffle(shuffled)
        difference = (
            statistics.mean(shuffled[:first_size])
            - statistics.mean(shuffled[first_size:])
        )
        if difference >= observed:
            exceedances += 1
    return observed, (exceedances + 1) / (iterations + 1)


def bootstrap_difference_ci(
    spaghetti_values: list[float],
    final_values: list[float],
    iterations: int = 10_000,
) -> tuple[float, float]:
    if not spaghetti_values or not final_values:
        return math.nan, math.nan
    rng = random.Random(RANDOM_SEED + 2)
    differences = []
    for _ in range(iterations):
        spaghetti_sample = [
            rng.choice(spaghetti_values) for _ in spaghetti_values
        ]
        final_sample = [rng.choice(final_values) for _ in final_values]
        differences.append(
            statistics.mean(spaghetti_sample)
            - statistics.mean(final_sample)
        )
    return percentile(differences, 0.025), percentile(differences, 0.975)


def max_output_response(
    baseline: dict[str, Any],
    variants: list[dict[str, Any]],
) -> float:
    responses = []
    for variant in variants:
        responses.append(abs(variant["score"] - baseline["score"]))
        for test_name in ("case_0", "case_1"):
            for metric_name in METRIC_NAMES:
                responses.append(
                    abs(
                        metric(variant, test_name, metric_name)
                        - metric(baseline, test_name, metric_name)
                    )
                )
    return max(responses, default=0.0)


def safe_mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def safe_median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def safe_stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) >= 2 else 0.0


def analyze(raw: dict[str, Any]) -> dict[str, Any]:
    baselines = raw["baselines"]
    oat_results = raw["oat_results"]
    joint_results = raw["joint_results"]
    parameter_sensitivity: dict[str, list[dict[str, Any]]] = {}
    aggregate: dict[str, dict[str, Any]] = {}

    for controller, spec in CONTROLLERS.items():
        baseline = baselines[controller]
        baseline_score = baseline["score"]
        score_scale = max(abs(baseline_score), 1.0)
        source = spec["source"].read_text(encoding="utf-8")
        controller_oat = [
            result
            for result in oat_results
            if result["controller"] == controller
        ]
        rows = []

        for parameter, base_value in spec["parameters"].items():
            low = next(
                result
                for result in controller_oat
                if result["parameter"] == parameter
                and result["direction"] == "low"
            )
            high = next(
                result
                for result in controller_oat
                if result["parameter"] == parameter
                and result["direction"] == "high"
            )
            low_change = abs(low["score"] - baseline_score)
            high_change = abs(high["score"] - baseline_score)
            mean_abs_change = (low_change + high_change) / 2.0
            structural_uses = usage_count(source, parameter)
            score_active = mean_abs_change > ACTIVITY_TOLERANCE
            output_response = max_output_response(baseline, [low, high])
            output_active = output_response > ACTIVITY_TOLERANCE

            if structural_uses == 0:
                activity_status = "structurally_unused"
            elif score_active:
                activity_status = "score_active"
            elif output_active:
                activity_status = "metric_only_active"
            else:
                activity_status = "locally_dormant"

            rows.append({
                "parameter": parameter,
                "base_value": base_value,
                "low_score": low["score"],
                "high_score": high["score"],
                "mean_abs_score_change": mean_abs_change,
                "central_half_range": abs(
                    high["score"] - low["score"]
                ) / 2.0,
                "score_points_per_fractional_unit": (
                    (high["score"] - low["score"])
                    / (2.0 * PERTURBATION)
                ),
                "normalized_mean_abs_score_change": (
                    mean_abs_change / score_scale
                ),
                "normalized_central_elasticity": (
                    abs(high["score"] - low["score"])
                    / (2.0 * PERTURBATION * score_scale)
                ),
                "max_output_response": output_response,
                "score_active": score_active,
                "output_active": output_active,
                "activity_status": activity_status,
                "usage_count_after_assignment": structural_uses,
                "case1_collision_low": metric(low, "case_1", "collisions"),
                "case1_collision_high": metric(high, "case_1", "collisions"),
                "case1_speed_low": metric(low, "case_1", "avg_speed"),
                "case1_speed_high": metric(high, "case_1", "avg_speed"),
                "feasible_low": feasible(low),
                "feasible_high": feasible(high),
            })

        rows.sort(
            key=lambda row: row["mean_abs_score_change"],
            reverse=True,
        )
        parameter_sensitivity[controller] = rows

        effects = [row["mean_abs_score_change"] for row in rows]
        active_effects = [
            row["mean_abs_score_change"]
            for row in rows
            if row["score_active"]
        ]
        l1_effect = sum(effects)
        squared_effect = sum(value * value for value in effects)
        effective_dimension = (
            l1_effect * l1_effect / squared_effect
            if squared_effect > 0.0
            else 0.0
        )
        active_parameters = {
            row["parameter"]
            for row in rows
            if row["score_active"]
        }
        active_oat_results = [
            result
            for result in controller_oat
            if result["parameter"] in active_parameters
        ]

        controller_joint = [
            result
            for result in joint_results
            if result["controller"] == controller
        ]
        joint_scores = [result["score"] for result in controller_joint]
        joint_deviations = [
            abs(score - baseline_score)
            for score in joint_scores
        ]

        aggregate[controller] = {
            "baseline_score": baseline_score,
            "baseline_case0_collisions": metric(
                baseline, "case_0", "collisions"
            ),
            "baseline_case0_speed": metric(
                baseline, "case_0", "avg_speed"
            ),
            "baseline_case1_collisions": metric(
                baseline, "case_1", "collisions"
            ),
            "baseline_case1_speed": metric(
                baseline, "case_1", "avg_speed"
            ),
            "baseline_feasible": feasible(baseline),
            "declared_parameter_count": len(rows),
            "structurally_referenced_count": sum(
                row["usage_count_after_assignment"] > 0
                for row in rows
            ),
            "score_active_count": sum(
                row["activity_status"] == "score_active"
                for row in rows
            ),
            "metric_only_active_count": sum(
                row["activity_status"] == "metric_only_active"
                for row in rows
            ),
            "locally_dormant_count": sum(
                row["activity_status"] == "locally_dormant"
                for row in rows
            ),
            "structurally_unused_count": sum(
                row["activity_status"] == "structurally_unused"
                for row in rows
            ),
            "oat_declared_mean_abs_score_deviation": safe_mean(effects),
            "oat_active_mean_abs_score_deviation": safe_mean(active_effects),
            "oat_active_median_abs_score_deviation": safe_median(
                active_effects
            ),
            "oat_total_l1_score_deviation": l1_effect,
            "oat_effective_dimension": effective_dimension,
            "oat_max_abs_score_deviation": max(effects, default=0.0),
            "oat_active_mean_normalized_deviation": (
                safe_mean(active_effects) / score_scale
            ),
            "oat_all_variant_feasibility_rate": (
                sum(feasible(result) for result in controller_oat)
                / len(controller_oat)
            ),
            "oat_active_variant_feasibility_rate": (
                sum(feasible(result) for result in active_oat_results)
                / len(active_oat_results)
                if active_oat_results
                else 0.0
            ),
            "joint_perturbed_parameter_count": len(
                controller_joint[0]["perturbed_parameters"]
            ),
            "joint_mean_score": safe_mean(joint_scores),
            "joint_score_std": safe_stdev(joint_scores),
            "joint_normalized_score_std": (
                safe_stdev(joint_scores) / score_scale
            ),
            "joint_score_min": min(joint_scores),
            "joint_score_max": max(joint_scores),
            "joint_mean_abs_score_deviation": safe_mean(joint_deviations),
            "joint_feasibility_rate": (
                sum(feasible(result) for result in controller_joint)
                / len(controller_joint)
            ),
            "joint_case1_collision_mean": safe_mean([
                metric(result, "case_1", "collisions")
                for result in controller_joint
            ]),
            "joint_case1_speed_mean": safe_mean([
                metric(result, "case_1", "avg_speed")
                for result in controller_joint
            ]),
        }

    spaghetti_active = [
        row["mean_abs_score_change"]
        for row in parameter_sensitivity["spaghetti"]
        if row["score_active"]
    ]
    final_active = [
        row["mean_abs_score_change"]
        for row in parameter_sensitivity["final"]
        if row["score_active"]
    ]
    observed_difference, permutation_p = permutation_test(
        spaghetti_active,
        final_active,
    )
    ci_low, ci_high = bootstrap_difference_ci(
        spaghetti_active,
        final_active,
    )

    spaghetti_aggregate = aggregate["spaghetti"]
    final_aggregate = aggregate["final"]
    return {
        "parameter_sensitivity": parameter_sensitivity,
        "aggregate": aggregate,
        "comparison": {
            "active_mean_difference_spaghetti_minus_final": (
                observed_difference
            ),
            "active_mean_bootstrap_95_percent_ci": [ci_low, ci_high],
            "active_mean_one_sided_permutation_p": permutation_p,
            "active_mean_ratio_spaghetti_to_final": (
                spaghetti_aggregate[
                    "oat_active_mean_abs_score_deviation"
                ]
                / max(
                    final_aggregate[
                        "oat_active_mean_abs_score_deviation"
                    ],
                    1e-12,
                )
            ),
            "total_l1_ratio_spaghetti_to_final": (
                spaghetti_aggregate["oat_total_l1_score_deviation"]
                / max(
                    final_aggregate["oat_total_l1_score_deviation"],
                    1e-12,
                )
            ),
            "joint_std_ratio_spaghetti_to_final": (
                spaghetti_aggregate["joint_score_std"]
                / max(final_aggregate["joint_score_std"], 1e-12)
            ),
            "joint_normalized_std_ratio_spaghetti_to_final": (
                spaghetti_aggregate["joint_normalized_score_std"]
                / max(
                    final_aggregate["joint_normalized_score_std"],
                    1e-12,
                )
            ),
        },
    }


def write_csv_files(raw: dict[str, Any], analysis: dict[str, Any]) -> None:
    oat_path = ANALYSIS_DIR / "sensitivity_oat_results.csv"
    summary_lookup = {
        (controller, row["parameter"]): row
        for controller, rows in analysis["parameter_sensitivity"].items()
        for row in rows
    }
    with oat_path.open("w", newline="", encoding="utf-8") as output:
        fieldnames = [
            "controller",
            "parameter",
            "direction",
            "multiplier",
            "parameter_value",
            "score",
            "score_change_from_baseline",
            "activity_status",
            "case0_collisions",
            "case0_avg_speed",
            "case1_collisions",
            "case1_critical_ttc",
            "case1_avg_speed",
            "case1_speed_variance",
            "case1_teleports",
            "feasible",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for result in raw["oat_results"]:
            baseline = raw["baselines"][result["controller"]]
            summary = summary_lookup[
                (result["controller"], result["parameter"])
            ]
            writer.writerow({
                "controller": result["controller"],
                "parameter": result["parameter"],
                "direction": result["direction"],
                "multiplier": result["multiplier"],
                "parameter_value": result["parameter_value"],
                "score": result["score"],
                "score_change_from_baseline": (
                    result["score"] - baseline["score"]
                ),
                "activity_status": summary["activity_status"],
                "case0_collisions": metric(
                    result, "case_0", "collisions"
                ),
                "case0_avg_speed": metric(
                    result, "case_0", "avg_speed"
                ),
                "case1_collisions": metric(
                    result, "case_1", "collisions"
                ),
                "case1_critical_ttc": metric(
                    result, "case_1", "critical_ttc_count"
                ),
                "case1_avg_speed": metric(
                    result, "case_1", "avg_speed"
                ),
                "case1_speed_variance": metric(
                    result, "case_1", "speed_variance"
                ),
                "case1_teleports": metric(
                    result, "case_1", "teleports"
                ),
                "feasible": feasible(result),
            })

    summary_path = ANALYSIS_DIR / "sensitivity_parameter_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as output:
        rows = [
            {"controller": controller, **row}
            for controller, controller_rows
            in analysis["parameter_sensitivity"].items()
            for row in controller_rows
        ]
        writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    joint_path = ANALYSIS_DIR / "sensitivity_joint_results.csv"
    with joint_path.open("w", newline="", encoding="utf-8") as output:
        fieldnames = [
            "controller",
            "run_id",
            "score",
            "case1_collisions",
            "case1_critical_ttc",
            "case1_avg_speed",
            "case1_speed_variance",
            "case1_teleports",
            "feasible",
            "perturbed_parameters_json",
            "multipliers_json",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for result in raw["joint_results"]:
            writer.writerow({
                "controller": result["controller"],
                "run_id": result["run_id"],
                "score": result["score"],
                "case1_collisions": metric(
                    result, "case_1", "collisions"
                ),
                "case1_critical_ttc": metric(
                    result, "case_1", "critical_ttc_count"
                ),
                "case1_avg_speed": metric(
                    result, "case_1", "avg_speed"
                ),
                "case1_speed_variance": metric(
                    result, "case_1", "speed_variance"
                ),
                "case1_teleports": metric(
                    result, "case_1", "teleports"
                ),
                "feasible": feasible(result),
                "perturbed_parameters_json": json.dumps(
                    result["perturbed_parameters"]
                ),
                "multipliers_json": json.dumps(
                    result["multipliers"],
                    sort_keys=True,
                ),
            })


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def format_number(value: float, digits: int = 3) -> str:
    if math.isnan(value):
        return "n/a"
    if math.isinf(value):
        return "∞"
    return f"{value:.{digits}f}"


def relation(first: float, second: float) -> str:
    if math.isclose(first, second, rel_tol=1e-9, abs_tol=1e-9):
        return "approximately equal to"
    return "higher than" if first > second else "lower than"


def write_report(raw: dict[str, Any], analysis: dict[str, Any]) -> None:
    aggregate = analysis["aggregate"]
    comparison = analysis["comparison"]
    spaghetti = aggregate["spaghetti"]
    final = aggregate["final"]
    ci_low, ci_high = comparison["active_mean_bootstrap_95_percent_ci"]
    sensitivity_conclusion_supported = (
        ci_low > 0.0
        and comparison["active_mean_ratio_spaghetti_to_final"] > 1.0
        and comparison["total_l1_ratio_spaghetti_to_final"] > 1.0
        and comparison["joint_std_ratio_spaghetti_to_final"] > 1.0
        and comparison[
            "joint_normalized_std_ratio_spaghetti_to_final"
        ] > 1.0
    )

    report_lines = [
        "# Activity-Aware Comparative Hyperparameter Sensitivity Analysis",
        "",
        "## Executive result",
        "",
        (
            "The comparison explicitly avoids dilution by inactive or "
            "spurious parameters. The primary local sensitivity is the mean "
            "absolute score response conditional on a parameter having a "
            "non-zero local score effect. It is accompanied by the total L1 "
            "effect and an effective sensitivity dimension; appending any "
            "number of exact zero-effect parameters changes none of these "
            "three quantities."
        ),
        "",
        (
            f"The spaghetti controller's active-parameter mean response is "
            f"{format_number(spaghetti['oat_active_mean_abs_score_deviation'])} "
            "score points, which is "
            f"{relation(spaghetti['oat_active_mean_abs_score_deviation'], final['oat_active_mean_abs_score_deviation'])} "
            f"the final controller's {format_number(final['oat_active_mean_abs_score_deviation'])}. "
            f"The corresponding spaghetti/final ratio is "
            f"{format_number(comparison['active_mean_ratio_spaghetti_to_final'])}."
        ),
        "",
        (
            f"Across joint ±{PERTURBATION * 100:.0f}% perturbations of all "
            "structurally referenced parameters, the spaghetti score standard "
            f"deviation is {format_number(spaghetti['joint_score_std'])}, "
            f"compared with {format_number(final['joint_score_std'])} for the "
            "final controller. These results distinguish local main-effect "
            "sensitivity from interaction-driven robustness."
        ),
        "",
        (
            "Taken together, the activity-aware evidence "
            + (
                "supports"
                if sensitivity_conclusion_supported
                else "does not consistently support"
            )
            + " the conclusion that the spaghetti controller is more "
            "hyperparameter-sensitive than the final controller in the tested "
            "±10% neighborhood."
        ),
        "",
        "## Methods",
        "",
        (
            f"Each declared scalar behavioral parameter was perturbed one at "
            f"a time by ±{PERTURBATION * 100:.0f}% around its current value. "
            "Both low- and high-demand simulations were run for every variant. "
            f"In addition, {JOINT_TRIALS} deterministic random joint trials "
            "per controller perturbed every parameter that has at least one "
            "source-code reference after its assignment."
        ),
        "",
        (
            "A parameter is **score-active** when either local perturbation "
            "changes the aggregate score beyond numerical tolerance. A "
            "**metric-only active** parameter changes at least one reported "
            "metric but not the rounded aggregate score. A **locally dormant** "
            "parameter is referenced by the controller but produces no "
            "observed response within ±10%. A **structurally unused** parameter "
            "has no reference after assignment and is treated as spurious."
        ),
        "",
        (
            "The declared-parameter mean is reported only as a diagnostic "
            "because it can be made arbitrarily small by appending unused "
            "parameters. The primary activity-aware measures are: (i) mean "
            "absolute score change over score-active parameters, (ii) the sum "
            "of all parameter effects (L1), and (iii) effective dimension, "
            "defined as the participation ratio of the non-negative effects. "
            "The joint trials exclude structurally unused parameters but retain "
            "locally dormant referenced parameters so that interactions are "
            "not discarded."
        ),
        "",
        (
            "A trial is classified as feasible when the high-demand case has "
            "zero collisions, zero teleports, and average speed of at least "
            "8 m/s. The controller files were hashed before and after all "
            "experiments."
        ),
        "",
        "## Source-integrity check",
        "",
        markdown_table(
            ["Controller", "Unchanged", "SHA-256"],
            [
                [
                    controller,
                    values["unchanged"],
                    values["sha256_after"],
                ]
                for controller, values in raw["source_integrity"].items()
            ],
        ),
        "",
        "## Baseline performance",
        "",
        markdown_table(
            [
                "Controller",
                "Score",
                "Case-0 collisions",
                "Case-0 speed",
                "Case-1 collisions",
                "Case-1 speed",
                "Feasible",
            ],
            [
                [
                    controller,
                    format_number(values["baseline_score"]),
                    f"{values['baseline_case0_collisions']:.0f}",
                    f"{values['baseline_case0_speed']:.2f}",
                    f"{values['baseline_case1_collisions']:.0f}",
                    f"{values['baseline_case1_speed']:.2f}",
                    values["baseline_feasible"],
                ]
                for controller, values in aggregate.items()
            ],
        ),
        "",
        "## Parameter activity accounting",
        "",
        markdown_table(
            [
                "Controller",
                "Declared",
                "Referenced",
                "Score-active",
                "Metric-only",
                "Locally dormant",
                "Structurally unused",
            ],
            [
                [
                    controller,
                    values["declared_parameter_count"],
                    values["structurally_referenced_count"],
                    values["score_active_count"],
                    values["metric_only_active_count"],
                    values["locally_dormant_count"],
                    values["structurally_unused_count"],
                ]
                for controller, values in aggregate.items()
            ],
        ),
        "",
        "## Activity-aware aggregate sensitivity",
        "",
        markdown_table(
            [
                "Controller",
                "Active mean abs(Δscore)",
                "Total L1 effect",
                "Effective dimension",
                "Max abs(Δscore)",
                "Active normalized mean",
                "Joint score SD",
                "Joint normalized SD",
            ],
            [
                [
                    controller,
                    format_number(
                        values["oat_active_mean_abs_score_deviation"]
                    ),
                    format_number(values["oat_total_l1_score_deviation"]),
                    format_number(values["oat_effective_dimension"]),
                    format_number(values["oat_max_abs_score_deviation"]),
                    format_number(
                        values["oat_active_mean_normalized_deviation"]
                    ),
                    format_number(values["joint_score_std"]),
                    format_number(values["joint_normalized_score_std"]),
                ]
                for controller, values in aggregate.items()
            ],
        ),
        "",
        (
            "For reference, the dilution-prone declared-parameter means are "
            f"{format_number(spaghetti['oat_declared_mean_abs_score_deviation'])} "
            "for spaghetti and "
            f"{format_number(final['oat_declared_mean_abs_score_deviation'])} "
            "for final. These values are not used for the primary comparison."
        ),
        "",
        "## Most sensitive score-active parameters",
        "",
    ]

    for controller in ("spaghetti", "final"):
        active_rows = [
            row
            for row in analysis["parameter_sensitivity"][controller]
            if row["score_active"]
        ]
        report_lines.extend([
            f"### {controller.capitalize()} controller",
            "",
            markdown_table(
                [
                    "Rank",
                    "Parameter",
                    "Mean abs(Δscore)",
                    "Low score",
                    "High score",
                    "Case-1 collisions low/high",
                    "Case-1 speed low/high",
                ],
                [
                    [
                        rank,
                        row["parameter"],
                        format_number(row["mean_abs_score_change"]),
                        format_number(row["low_score"]),
                        format_number(row["high_score"]),
                        (
                            f"{row['case1_collision_low']:.0f}/"
                            f"{row['case1_collision_high']:.0f}"
                        ),
                        (
                            f"{row['case1_speed_low']:.2f}/"
                            f"{row['case1_speed_high']:.2f}"
                        ),
                    ]
                    for rank, row in enumerate(active_rows, start=1)
                ],
            )
            if active_rows
            else "No score-active parameters were detected.",
            "",
        ])

    report_lines.extend([
        "## Inactive and dormant parameters",
        "",
        (
            "These parameters are shown explicitly rather than being allowed "
            "to dilute the primary active-set mean."
        ),
        "",
        markdown_table(
            ["Controller", "Parameter", "Classification", "Usage count"],
            [
                [
                    controller,
                    row["parameter"],
                    row["activity_status"],
                    row["usage_count_after_assignment"],
                ]
                for controller in ("spaghetti", "final")
                for row in analysis["parameter_sensitivity"][controller]
                if not row["score_active"]
            ],
        ),
        "",
        "## Comparative inference",
        "",
        (
            "Using only score-active parameters, the estimated difference in "
            "mean absolute local response (spaghetti minus final) is "
            f"{format_number(comparison['active_mean_difference_spaghetti_minus_final'])} "
            "score points. The parameter-level bootstrap 95% interval is "
            f"[{format_number(ci_low)}, {format_number(ci_high)}], and the "
            "exploratory one-sided permutation p-value is "
            f"{format_number(comparison['active_mean_one_sided_permutation_p'], 4)}."
        ),
        "",
        (
            "The spaghetti/final ratios are "
            f"{format_number(comparison['active_mean_ratio_spaghetti_to_final'])} "
            "for active local mean, "
            f"{format_number(comparison['total_l1_ratio_spaghetti_to_final'])} "
            "for total L1 effect, "
            f"{format_number(comparison['joint_std_ratio_spaghetti_to_final'])} "
            "for absolute joint score SD, and "
            f"{format_number(comparison['joint_normalized_std_ratio_spaghetti_to_final'])} "
            "for baseline-normalized joint SD."
        ),
        "",
        (
            "The active-set comparison answers how strongly a parameter matters "
            "when it matters locally. The L1 comparison answers how much total "
            "local sensitivity is distributed across the controller. The joint "
            "comparison answers how much the score varies when the referenced "
            "parameter vector moves simultaneously. These are distinct "
            "properties and should not be collapsed into a single parameter-"
            "count-dependent average."
        ),
        "",
        (
            "The permutation and bootstrap calculations treat named parameters "
            "as exchangeable observations, which is only an exploratory "
            "approximation. The effect sizes, activity classifications, and "
            "raw simulation responses should carry more interpretive weight "
            "than the p-value."
        ),
        "",
        "## Robustness and feasibility",
        "",
        markdown_table(
            [
                "Controller",
                "OAT active-variant feasible rate",
                "Joint feasible rate",
                "Joint mean case-1 collisions",
                "Joint mean case-1 speed",
                "Joint score range",
            ],
            [
                [
                    controller,
                    (
                        f"{100 * values['oat_active_variant_feasibility_rate']:.1f}%"
                    ),
                    f"{100 * values['joint_feasibility_rate']:.1f}%",
                    format_number(values["joint_case1_collision_mean"]),
                    format_number(values["joint_case1_speed_mean"]),
                    (
                        f"[{format_number(values['joint_score_min'])}, "
                        f"{format_number(values['joint_score_max'])}]"
                    ),
                ]
                for controller, values in aggregate.items()
            ],
        ),
        "",
        (
            "Feasibility rates must be interpreted relative to the baseline. "
            "If a baseline is already infeasible, its feasibility rate measures "
            "whether perturbations repair that state, not the probability of "
            "losing a feasible operating regime."
        ),
        "",
        "## Limitations",
        "",
        (
            "OAT responses are local to ±10% and can classify thresholded "
            "parameters as locally dormant even when larger changes would "
            "activate them. Joint trials partially address interactions but "
            "are finite empirical samples rather than exhaustive bounds. "
            "Results are specific to the two supplied demand scenarios, "
            "simulation seed, evaluator, and current tuned baselines."
        ),
        "",
        "## Reproducibility",
        "",
        (
            f"Perturbation magnitude: ±{PERTURBATION * 100:.0f}%; "
            f"joint trials per controller: {JOINT_TRIALS}; "
            f"random seed: {RANDOM_SEED}; workers: {MAX_WORKERS}. "
            "Raw OAT and joint trials are supplied as CSV files; the complete "
            "raw and analyzed results are supplied as JSON."
        ),
        "",
    ])

    (ANALYSIS_DIR / "sensitivity_report.md").write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )


def main() -> None:
    raw = run_experiments()
    analysis = analyze(raw)
    write_csv_files(raw, analysis)
    write_report(raw, analysis)

    output = {
        "configuration": raw["configuration"],
        "source_integrity": raw["source_integrity"],
        "raw": {
            "baselines": {
                controller: compact_result(result)
                for controller, result in raw["baselines"].items()
            },
            "oat_results": [
                compact_result(result)
                for result in raw["oat_results"]
            ],
            "joint_results": [
                compact_result(result)
                for result in raw["joint_results"]
            ],
        },
        "analysis": analysis,
    }
    (ANALYSIS_DIR / "sensitivity_results.json").write_text(
        json.dumps(output, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({
        "source_integrity": raw["source_integrity"],
        "aggregate": analysis["aggregate"],
        "comparison": analysis["comparison"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
