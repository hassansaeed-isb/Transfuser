#!/usr/bin/env python3
"""Print TransFuser / CARLA Leaderboard metrics from a results JSON file."""

from __future__ import annotations

import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORK_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(WORK_DIR, "leaderboard", "scripts"))

from pretty_print_json import prettify_json  # noqa: E402


def summarize(json_path: str) -> str:
    with open(json_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    checkpoint = data.get("_checkpoint", {})
    global_record = checkpoint.get("global_record", {})
    scores = global_record.get("scores", {})
    labels = data.get("labels", [])
    values = data.get("values", [])

    lines = [
        "=" * 60,
        "LONGEST6 EVALUATION SUMMARY",
        "=" * 60,
        f"Results file: {json_path}",
        f"Entry status: {data.get('entry_status', 'unknown')}",
        "",
        "Global metrics:",
        f"  Driving Score (score_composed): {scores.get('score_composed', 'N/A')}",
        f"  Route Completion (score_route): {scores.get('score_route', 'N/A')}",
        f"  Penalty Multiplier (score_penalty): {scores.get('score_penalty', 'N/A')}",
        "",
        "Infraction rates (per km driven; lower is better):",
    ]

    if labels and values and len(labels) > 3:
        for label, value in zip(labels[3:], values[3:]):
            lines.append(f"  {label}: {value}")
    else:
        infractions = global_record.get("infractions", {})
        if isinstance(infractions, dict):
            for name, value in sorted(infractions.items()):
                lines.append(f"  {name}: {value}")

    progress = checkpoint.get("progress", [0, 0])
    if progress and len(progress) == 2 and progress[1]:
        lines.extend(
            [
                "",
                f"Progress: {progress[0]}/{progress[1]} routes",
            ]
        )

    lines.append("=" * 60)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Print Longest6 evaluation metrics")
    parser.add_argument("--file", required=True, help="Path to leaderboard JSON results")
    parser.add_argument("--save", default="", help="Optional text file for detailed table output")
    args = parser.parse_args()

    if not os.path.isfile(args.file):
        print(f"ERROR: file not found: {args.file}")
        return 1

    print(summarize(args.file))
    print()

    class Args:
        file = args.file
        format = "fancy_grid"
        output = args.save or None

    return prettify_json(Args())


if __name__ == "__main__":
    raise SystemExit(main())
