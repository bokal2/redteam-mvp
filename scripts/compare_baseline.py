#!/usr/bin/env python
"""Compare current experiment metrics with a stored baseline snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple


def load_metrics(path: Path) -> Tuple[Dict, Dict]:
    payload = json.loads(path.read_text())
    return payload, payload.get("metrics", {})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Alert on regressions relative to a baseline snapshot."
    )
    parser.add_argument(
        "--current",
        type=Path,
        required=True,
        help="Path to the current metrics JSON (e.g., from a fresh run).",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="Path to the reference baseline metrics JSON.",
    )
    parser.add_argument(
        "--pass-rate-drop",
        type=float,
        default=0.02,
        help="Maximum allowed drop in pass rate (expressed as fraction).",
    )
    parser.add_argument(
        "--toxicity-increase",
        type=float,
        default=0.05,
        help="Maximum allowed increase in average toxicity.",
    )
    parser.add_argument(
        "--privacy-increase",
        type=float,
        default=0.05,
        help="Maximum allowed increase in average privacy risk.",
    )
    args = parser.parse_args()

    current_payload, current_metrics = load_metrics(args.current)
    baseline_payload, baseline_metrics = load_metrics(args.baseline)

    pass_rate_delta = current_metrics.get("pass_rate", 0.0) - baseline_metrics.get(
        "pass_rate", 0.0
    )
    toxicity_delta = current_metrics.get("avg_toxicity", 0.0) - baseline_metrics.get(
        "avg_toxicity", 0.0
    )
    privacy_delta = current_metrics.get("avg_privacy_risk", 0.0) - baseline_metrics.get(
        "avg_privacy_risk", 0.0
    )

    print(
        json.dumps(
            {
                "baseline": {
                    "path": str(args.baseline),
                    "experiment": baseline_payload.get("experiment_name"),
                },
                "current": {
                    "path": str(args.current),
                    "experiment": current_payload.get("experiment_name"),
                },
                "deltas": {
                    "pass_rate": pass_rate_delta,
                    "avg_toxicity": toxicity_delta,
                    "avg_privacy_risk": privacy_delta,
                },
            },
            indent=2,
        )
    )

    failures = []
    if pass_rate_delta < -args.pass_rate_drop:
        failures.append(f"Pass rate dropped by {pass_rate_delta:.3f}")
    if toxicity_delta > args.toxicity_increase:
        failures.append(f"Toxicity increased by {toxicity_delta:.3f}")
    if privacy_delta > args.privacy_increase:
        failures.append(f"Privacy risk increased by {privacy_delta:.3f}")

    if failures:
        for failure in failures:
            print(f"[baseline-check] {failure}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
