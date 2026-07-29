#!/usr/bin/env python3
"""Reproduce the incumbent-level OR-Agent BPP-ONLINE evolution analysis."""

from __future__ import annotations

import ast
import csv
import json
import re
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DETAILS = ROOT / "details"
RESULTS = ROOT / "results.json"
OUTPUT = ROOT / "output.txt"
PROGRESS = ROOT / "progress.txt"
TIMELINE_CSV = ROOT / "incumbent_evolution.csv"
TIMELINE_FIGURE = ROOT / "incumbent_evolution.png"
SUMMARY_JSON = ROOT / "evolution_metrics.json"
PRINCIPAL_ROUNDS = (1, 12, 23, 34, 45)

FILENAME_PATTERN = re.compile(
    r"lead1_round(?P<round>\d+)_count(?P<count>\d+)_id(?P<id>\d+)_solution\.py"
)


def creation_time(path: Path) -> datetime:
    stat = path.stat()
    timestamp = getattr(stat, "st_birthtime", stat.st_ctime)
    return datetime.fromtimestamp(timestamp).astimezone()


def generated_tree(path: Path) -> ast.Module:
    """Exclude the shared module import scaffold and function docstrings."""
    module = ast.parse(path.read_text(encoding="utf-8"))
    functions = [
        node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    for function in functions:
        if (
            function.body
            and isinstance(function.body[0], ast.Expr)
            and isinstance(function.body[0].value, ast.Constant)
            and isinstance(function.body[0].value.value, str)
        ):
            function.body = function.body[1:]
    return ast.Module(body=functions, type_ignores=[])


def subtree_fingerprint(node: ast.AST, depth: int = 2) -> tuple[Any, ...]:
    parts: list[Any] = [type(node).__name__]
    if depth == 0:
        return tuple(parts)

    for field, value in ast.iter_fields(node):
        if isinstance(value, ast.AST):
            parts.append(subtree_fingerprint(value, depth - 1))
        elif isinstance(value, list):
            children = tuple(
                subtree_fingerprint(child, depth - 1)
                for child in value
                if isinstance(child, ast.AST)
            )
            if children:
                parts.append(children)
        elif isinstance(node, ast.Constant) and field == "value":
            parts.append(("CONST", type(value).__name__))
    return tuple(parts)


def subtree_counts(tree: ast.Module) -> Counter[tuple[Any, ...]]:
    return Counter(subtree_fingerprint(node) for node in ast.walk(tree))


def ast_similarity(
    left: Counter[tuple[Any, ...]], right: Counter[tuple[Any, ...]]
) -> float:
    return sum((left & right).values()) / sum((left | right).values())


def branch_complexity(tree: ast.Module) -> int:
    complexity = 1
    branch_nodes = (
        ast.If,
        ast.For,
        ast.While,
        ast.IfExp,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
        ast.comprehension,
        ast.ExceptHandler,
        ast.Assert,
    )
    for node in ast.walk(tree):
        if isinstance(node, branch_nodes):
            complexity += 1
        elif isinstance(node, ast.BoolOp):
            complexity += max(1, len(node.values) - 1)
    return complexity


def mechanisms(source: str) -> list[str]:
    detected: list[str] = []
    checks = (
        ("post-placement", "post_placement" in source),
        (
            "best-fit",
            "best_fit" in source.lower()
            or "-feasible_post_caps" in source
            or "-post_placement_caps" in source,
        ),
        (
            "multiple-fit",
            "multiple" in source.lower()
            and ("common_size" in source or "estimated_common" in source),
        ),
        ("percentile", "percentile" in source or "searchsorted" in source),
        (
            "dynamic-remainders",
            "quantile_values" in source
            or "dynamic_common_sizes" in source
            or "unique_caps" in source,
        ),
        ("Farey", "farey" in source.lower()),
        (
            "geometric",
            "phi" in source
            or "golden ratio" in source.lower()
            or "np.sqrt(2)" in source,
        ),
        ("exponential", "np.exp" in source),
        (
            "stateful",
            "hasattr(priority" in source
            or "recent_items" in source
            or "item_history" in source,
        ),
    )
    for label, present in checks:
        if present:
            detected.append(label)
    return detected


def build_timeline() -> list[dict[str, Any]]:
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    previous_counts: Counter[tuple[Any, ...]] | None = None
    start_time: datetime | None = None

    for index, result in enumerate(results):
        path = ROOT / result["code_filepath"]
        match = FILENAME_PATTERN.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Unexpected solution filename: {path.name}")

        tree = generated_tree(path)
        counts = subtree_counts(tree)
        timestamp = creation_time(path)
        if start_time is None:
            start_time = timestamp

        functions = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        generated_loc = sum(
            getattr(node, "end_lineno", node.lineno) - node.lineno + 1
            for node in tree.body
        )
        score = float(result["best_obj_overall"])
        previous_score = (
            float(results[index - 1]["best_obj_overall"]) if index else score
        )
        source = path.read_text(encoding="utf-8")

        rows.append(
            {
                "incumbent_index": index,
                "node_id": int(match["id"]),
                "round": int(match["round"]),
                "count": int(match["count"]),
                "creation_time": timestamp.isoformat(),
                "elapsed_hours": (timestamp - start_time).total_seconds() / 3600,
                "total_responses": result["total_responses"],
                "total_function_evals": result["total_function_evals"],
                "score": score,
                "improvement": previous_score - score,
                "generated_loc": generated_loc,
                "ast_nodes": sum(1 for _ in ast.walk(tree)),
                "branch_complexity": branch_complexity(tree),
                "function_count": len(functions),
                "previous_incumbent_ast_similarity": (
                    "" if previous_counts is None else ast_similarity(previous_counts, counts)
                ),
                "mechanisms": ";".join(mechanisms(source)),
                "code_filepath": result["code_filepath"],
            }
        )
        previous_counts = counts
    return rows


def save_csv(rows: list[dict[str, Any]]) -> None:
    with TIMELINE_CSV.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(rows: list[dict[str, Any]]) -> None:
    elapsed = [row["elapsed_hours"] for row in rows]
    scores = [row["score"] for row in rows]
    indices = [row["incumbent_index"] for row in rows]
    loc = [row["generated_loc"] for row in rows]
    complexity = [row["branch_complexity"] for row in rows]

    figure, (objective_axis, code_axis) = plt.subplots(
        2, 1, figsize=(9, 7), constrained_layout=True
    )
    objective_axis.step(elapsed, scores, where="post", linewidth=2)
    objective_axis.scatter(elapsed, scores, s=30, zorder=3)
    objective_axis.set_ylabel("Best objective (bins; lower is better)")
    objective_axis.set_xlabel("Elapsed wall-clock time (hours)")
    objective_axis.grid(alpha=0.3)

    code_axis.plot(indices, loc, marker="o", label="Generated LOC")
    complexity_axis = code_axis.twinx()
    complexity_axis.plot(
        indices,
        complexity,
        marker="s",
        color="tab:red",
        label="Branch complexity",
    )
    code_axis.set_xlabel("Incumbent update index")
    code_axis.set_ylabel("Generated LOC")
    complexity_axis.set_ylabel("Branch complexity")
    code_axis.grid(alpha=0.3)

    handles_left, labels_left = code_axis.get_legend_handles_labels()
    handles_right, labels_right = complexity_axis.get_legend_handles_labels()
    code_axis.legend(
        handles_left + handles_right,
        labels_left + labels_right,
        loc="upper left",
    )

    figure.savefig(TIMELINE_FIGURE, dpi=300, bbox_inches="tight")
    plt.close(figure)


def parse_run_totals() -> dict[str, int]:
    text = PROGRESS.read_text(encoding="utf-8")
    labels = {
        "research_rounds": r"Number of research rounds:\s*(\d+)",
        "solution_count": r"Solution count:\s*(\d+)",
        "llm_calls": r"Total number of LLM calls:\s*(\d+)",
        "valid_responses": r"Total number of valid responses:\s*(\d+)",
        "function_evaluations": r"Total number of function evaluations:\s*(\d+)",
    }
    totals: dict[str, int] = {}
    for key, pattern in labels.items():
        match = re.search(pattern, text)
        if match is None:
            raise ValueError(f"Could not find {key} in {PROGRESS}")
        totals[key] = int(match.group(1))
    return totals


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_paths = sorted(DETAILS.glob("*_solution.py"))
    parseable: dict[int, tuple[Path, ast.Module]] = {}
    invalid: list[str] = []

    for path in candidate_paths:
        match = FILENAME_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        try:
            parseable[int(match["id"])] = (path, generated_tree(path))
        except SyntaxError:
            invalid.append(path.name)

    whole_ast_counts = Counter(
        ast.dump(tree, include_attributes=False)
        for _, tree in parseable.values()
    )
    duplicate_groups = [
        count for count in whole_ast_counts.values() if count > 1
    ]
    subtree_by_node = {
        node_id: subtree_counts(tree)
        for node_id, (_, tree) in parseable.items()
    }

    edge_similarities: list[float] = []
    parsed_edges = 0
    skipped_edges = 0
    for round_number in PRINCIPAL_ROUNDS:
        graph = DETAILS / f"flow_graph_lead1_round{round_number}_done.txt"
        stack: dict[int, int] = {}
        for line in graph.read_text(encoding="utf-8").splitlines():
            match = re.search(r"Node \[?(\d+)", line)
            if match is None or line.startswith(("Format:", "PS:")):
                continue
            depth = (line.index("Node") - 2) // 8
            node_id = int(match.group(1))
            stack[depth] = node_id
            for stale_depth in [key for key in stack if key > depth]:
                del stack[stale_depth]
            if depth == 0:
                continue
            parsed_edges += 1
            parent_id = stack[depth - 1]
            if parent_id not in subtree_by_node or node_id not in subtree_by_node:
                skipped_edges += 1
                continue
            edge_similarities.append(
                ast_similarity(subtree_by_node[parent_id], subtree_by_node[node_id])
            )

    totals = parse_run_totals()
    output_text = OUTPUT.read_text(encoding="utf-8")
    lower_bound_match = re.search(
        r"Lower bound on optimum:\s*([0-9.]+)", output_text
    )
    if lower_bound_match is None:
        raise ValueError(f"Could not find lower bound in {OUTPUT}")
    lower_bound = float(lower_bound_match.group(1))

    start_time = datetime.fromisoformat(rows[0]["creation_time"])
    end_time = max(
        creation_time(path) for path in DETAILS.iterdir() if path.is_file()
    )
    initial = float(rows[0]["score"])
    final = float(rows[-1]["score"])
    initial_gap = initial - lower_bound
    final_gap = final - lower_bound

    return {
        "run_totals": totals,
        "objective": {
            "lower_bound": lower_bound,
            "initial_incumbent": initial,
            "final_incumbent": final,
            "absolute_improvement": initial - final,
            "initial_excess_percent": 100 * initial_gap / lower_bound,
            "final_excess_percent": 100 * final_gap / lower_bound,
            "fraction_of_initial_gap_closed_percent": (
                100 * (initial_gap - final_gap) / initial_gap
            ),
            "largest_single_improvement": max(
                float(row["improvement"]) for row in rows
            ),
        },
        "timing": {
            "first_solution_creation_time": start_time.isoformat(),
            "final_best_creation_time": rows[-1]["creation_time"],
            "last_detail_artifact_creation_time": end_time.isoformat(),
            "run_artifact_span_hours": (
                end_time - start_time
            ).total_seconds() / 3600,
            "final_best_elapsed_hours": rows[-1]["elapsed_hours"],
            "post_final_best_hours": (
                end_time - datetime.fromisoformat(rows[-1]["creation_time"])
            ).total_seconds() / 3600,
            "post_final_best_llm_calls": (
                totals["llm_calls"] - int(rows[-1]["total_responses"])
            ),
            "post_final_best_function_evaluations": (
                totals["function_evaluations"]
                - int(rows[-1]["total_function_evals"])
            ),
        },
        "code_artifacts": {
            "saved_solution_files": len(candidate_paths),
            "parseable_solution_files": len(parseable),
            "syntax_invalid_files": invalid,
            "unique_whole_ast_structures": len(whole_ast_counts),
            "duplicate_ast_groups": len(duplicate_groups),
            "files_in_duplicate_ast_groups": sum(duplicate_groups),
        },
        "principal_tree_edges": {
            "rounds": list(PRINCIPAL_ROUNDS),
            "edges_in_graphs": parsed_edges,
            "parseable_parent_child_edges": len(edge_similarities),
            "skipped_edges": skipped_edges,
            "mean_ast_similarity": statistics.mean(edge_similarities),
            "median_ast_similarity": statistics.median(edge_similarities),
            "minimum_ast_similarity": min(edge_similarities),
            "maximum_ast_similarity": max(edge_similarities),
        },
    }


def save_summary(metrics: dict[str, Any]) -> None:
    SUMMARY_JSON.write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    rows = build_timeline()
    save_csv(rows)
    save_figure(rows)
    save_summary(aggregate_metrics(rows))
    print(f"Wrote {TIMELINE_CSV}")
    print(f"Wrote {TIMELINE_FIGURE}")
    print(f"Wrote {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
