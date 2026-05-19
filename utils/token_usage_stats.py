"""
token_usage_stats.py
--------------------
Calculates token usage (input, output, total) from LLM annotation results
stored under the output/ directory.

Directory layout expected:
    output/
        r<N>/
            <model_dir>/
                <commit_hash>.json   <- contains usage_metadata

Usage:
    # All models, all rounds
    python utils/token_usage_stats.py

    # Filter to a specific model (substring match on the folder name)
    python utils/token_usage_stats.py --model ollama_gpt-oss_120b

    # Custom output dir
    python utils/token_usage_stats.py --output-dir /path/to/output
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate LLM token usage statistics.")
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Path to the output directory (default: output)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Filter by model folder name substring (e.g. 'ollama_gpt-oss_120b'). "
             "If omitted, all models are included.",
    )
    parser.add_argument(
        "--cost-input",
        type=float,
        default=None,
        metavar="USD_PER_1M",
        help="Cost in USD per 1M input tokens (e.g. 0.15).",
    )
    parser.add_argument(
        "--cost-output",
        type=float,
        default=None,
        metavar="USD_PER_1M",
        help="Cost in USD per 1M output tokens (e.g. 0.60).",
    )
    return parser.parse_args()


def collect_stats(output_dir: Path, model_filter: str | None):
    """
    Returns a nested dict:
        stats[model][round] = {"input": int, "output": int, "total": int, "files": int, "missing": int}
    """
    stats: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(
        lambda: {"input": 0, "output": 0, "total": 0, "files": 0, "missing": 0}
    ))

    round_dirs = sorted(output_dir.glob("r[0-9]*"), key=lambda p: p.name)
    if not round_dirs:
        print(f"No round directories (r1, r2, …) found in '{output_dir}'.", file=sys.stderr)
        sys.exit(1)

    for round_dir in round_dirs:
        round_name = round_dir.name
        model_dirs = [d for d in round_dir.iterdir() if d.is_dir()]

        for model_dir in model_dirs:
            model_name = model_dir.name
            if model_filter and model_filter.lower() not in model_name.lower():
                continue

            for json_file in model_dir.glob("*.json"):
                entry = stats[model_name][round_name]
                entry["files"] += 1
                try:
                    with json_file.open(encoding="utf-8") as fh:
                        data = json.load(fh)
                    meta = data.get("usage_metadata")
                    if meta is None:
                        entry["missing"] += 1
                        continue
                    entry["input"] += meta.get("input_tokens", 0)
                    entry["output"] += meta.get("output_tokens", 0)
                    entry["total"] += meta.get("total_tokens", 0)
                except (json.JSONDecodeError, OSError) as exc:
                    print(f"Warning: could not read {json_file}: {exc}", file=sys.stderr)
                    entry["missing"] += 1

    return stats


def compute_cost(input_tokens: int, output_tokens: int,
                 cost_input: float | None, cost_output: float | None) -> float | None:
    if cost_input is None or cost_output is None:
        return None
    return (input_tokens * cost_input + output_tokens * cost_output) / 1_000_000


def print_report(stats: dict, all_rounds: list[str],
                 cost_input: float | None, cost_output: float | None) -> None:
    col_w = 12  # width for numeric columns
    show_cost = cost_input is not None and cost_output is not None

    for model_name, rounds in sorted(stats.items()):
        print(f"\n{'=' * 70}")
        print(f"  Model: {model_name}")
        print(f"{'=' * 70}")

        header = f"{'Round':<8} {'Commits':>{col_w}} "
        header += f"{'Input':>{col_w}} {'Output':>{col_w}} {'Total':>{col_w}}"
        if show_cost:
            header += f" {'Cost (USD)':>{col_w}} {'$/commit':>{col_w}}"
        print(header)
        print("-" * len(header))

        totals = {"files": 0, "missing": 0, "input": 0, "output": 0, "total": 0}

        for round_name in sorted(rounds.keys()):
            e = rounds[round_name]
            line = (
                f"{round_name:<8} "
                f"{e['files']:>{col_w},} "
                f"{e['input']:>{col_w},} "
                f"{e['output']:>{col_w},} "
                f"{e['total']:>{col_w},}"
            )
            if show_cost:
                cost = compute_cost(e['input'], e['output'], cost_input, cost_output)
                cpc = cost / e['files'] if e['files'] else 0.0
                line += f" {cost:>{col_w}.4f} {cpc:>{col_w}.6f}"
            print(line)
            for key in totals:
                totals[key] += e[key]

        print("-" * len(header))
        total_line = (
            f"{'TOTAL':<8} "
            f"{totals['files']:>{col_w},} "
            f"{totals['input']:>{col_w},} "
            f"{totals['output']:>{col_w},} "
            f"{totals['total']:>{col_w},}"
        )
        if show_cost:
            total_cost = compute_cost(totals['input'], totals['output'], cost_input, cost_output)
            commits = totals['files']
            cost_per_commit = total_cost / commits if commits else 0.0
            total_line += f" {total_cost:>{col_w}.4f} {cost_per_commit:>{col_w}.6f}"
        print(total_line)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    if not output_dir.is_dir():
        print(f"Error: output directory '{output_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    stats = collect_stats(output_dir, args.model)

    if not stats:
        print("No results found. Check --output-dir and --model arguments.", file=sys.stderr)
        sys.exit(1)

    all_rounds = sorted({r for rounds in stats.values() for r in rounds})
    print_report(stats, all_rounds, args.cost_input, args.cost_output)

    # Grand total across all models
    if len(stats) > 1:
        grand = {"files": 0, "missing": 0, "input": 0, "output": 0, "total": 0}
        for rounds in stats.values():
            for e in rounds.values():
                for key in grand:
                    grand[key] += e[key]

        col_w = 12
        print(f"\n{'=' * 70}")
        print("  GRAND TOTAL (all models)")
        print(f"{'=' * 70}")
        print(
            f"{'Commits':>{col_w}}: {grand['files']:,}\n"
            f"{'Input tokens':>{col_w}}: {grand['input']:,}\n"
            f"{'Output tokens':>{col_w}}: {grand['output']:,}\n"
            f"{'Total tokens':>{col_w}}: {grand['total']:,}"
        )
        if args.cost_input is not None and args.cost_output is not None:
            grand_cost = compute_cost(grand['input'], grand['output'], args.cost_input, args.cost_output)
            commits = grand['files']
            cost_per_commit = grand_cost / commits if commits else 0.0
            print(
                f"{'Total cost':>{col_w}}: ${grand_cost:.4f}\n"
                f"{'Cost/commit':>{col_w}}: ${cost_per_commit:.6f}"
            )


if __name__ == "__main__":
    main()
