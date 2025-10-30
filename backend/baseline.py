"""
Utilities for seeding and replaying manual baseline experiments.

Usage
-----
python -m backend.baseline --config baseline/baseline_config.json --out baseline/results

The script will:
1. Ensure the configured experiment exists (creating it if needed)
2. Optionally launch a fresh run with the stored prompts
3. Export run-level data and aggregate metrics to the output directory
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import unicodedata

from sqlmodel import select

from .db import AsyncSessionLocal, init_db
from .models import Experiment, Run
from .orchestrator import run_full_experiment


@dataclass
class JudgeConfig:
    use_openai: bool
    model: Optional[str]


@dataclass
class BaselineConfig:
    experiment_name: str
    model: str
    instruction: str
    prompts: List[str]
    description: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 1024
    parallelism: int = 3
    model_timeout: int = 30
    judge: JudgeConfig = field(
        default_factory=lambda: JudgeConfig(use_openai=False, model=None)
    )

    @staticmethod
    def load(path: Path) -> "BaselineConfig":
        payload = json.loads(path.read_text())

        try:
            judge_section = payload.get("judge") or {}
            judge_conf = JudgeConfig(
                use_openai=bool(judge_section.get("use_openai", False)),
                model=judge_section.get("model"),
            )
            prompts = payload["prompts"]
            if not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts):
                raise ValueError("`prompts` must be a list of strings")
            return BaselineConfig(
                experiment_name=payload["experiment_name"],
                model=payload["model"],
                instruction=payload["instruction"],
                prompts=prompts,
                description=payload.get("description"),
                temperature=float(payload.get("temperature", 0.0)),
                max_tokens=int(payload.get("max_tokens", 1024)),
                parallelism=int(payload.get("parallelism", 3)),
                model_timeout=int(payload.get("model_timeout", 30)),
                judge=judge_conf,
            )
        except KeyError as exc:  # pragma: no cover - defensive validation
            raise ValueError(f"Missing required config field: {exc.args[0]}") from exc


async def _ensure_experiment(
    *, name: str, model: str, description: Optional[str]
) -> Experiment:
    async with AsyncSessionLocal() as session:
        query = await session.exec(select(Experiment).where(Experiment.name == name))
        existing = query.one_or_none()
        if existing:
            # Update description/model if they drift so metadata stays current.
            has_update = False
            if description and existing.description != description:
                existing.description = description
                has_update = True
            if model and existing.model != model:
                existing.model = model
                has_update = True
            if has_update:
                session.add(existing)
                await session.commit()
                await session.refresh(existing)
            return existing

        experiment = Experiment(name=name, description=description, model=model)
        session.add(experiment)
        await session.commit()
        await session.refresh(experiment)
        return experiment


def _normalize_text(value: Optional[str]) -> str:
    if value is None:
        return ""
    normalized = unicodedata.normalize("NFKC", value)
    normalized = (
        normalized.replace("’", "'")
        .replace("‘", "'")
        .replace("–", "-")
        .replace("—", "-")
    )
    return normalized.strip().casefold()


async def _load_runs(experiment_id: str, *, instruction: Optional[str] = None) -> List[Run]:
    async with AsyncSessionLocal() as session:
        stmt = (
            select(Run)
            .where(Run.experiment_id == experiment_id)
            .order_by(Run.created_at.asc())
        )
        query = await session.exec(stmt)
        runs = query.all()

    if instruction:
        target = _normalize_text(instruction)
        runs = [
            run for run in runs if _normalize_text(run.base_instruction) == target
        ]
    return runs


def _aggregate_metrics(runs: Iterable[Run]) -> Dict[str, Any]:
    runs_list = list(runs)
    total = len(runs_list)
    passed = sum(1 for run in runs_list if (run.judge_scores or {}).get("overall_pass"))
    blocked = sum(1 for run in runs_list if run.blocked)
    errored = sum(1 for run in runs_list if run.error)
    pass_rate = passed / total if total else 0.0

    tox_vals: List[float] = []
    privacy_vals: List[float] = []
    for run in runs_list:
        det = run.detector_scores or {}
        toxicity = det.get("toxicity")
        if isinstance(toxicity, dict):
            tox_score = (
                toxicity.get("toxicity")
                or toxicity.get("toxicity_score")
                or toxicity.get("severe_toxicity")
            )
        else:
            tox_score = toxicity
        if isinstance(tox_score, (int, float)):
            tox_vals.append(float(tox_score))

        pii = det.get("pii")
        privacy_val = pii.get("privacy_risk") if isinstance(pii, dict) else None
        if isinstance(privacy_val, (int, float)):
            privacy_vals.append(float(privacy_val))

    avg_toxicity = sum(tox_vals) / len(tox_vals) if tox_vals else 0.0
    avg_privacy = sum(privacy_vals) / len(privacy_vals) if privacy_vals else 0.0

    return {
        "total_runs": total,
        "passed": passed,
        "pass_rate": pass_rate,
        "blocked_runs": blocked,
        "errored_runs": errored,
        "avg_toxicity": avg_toxicity,
        "avg_privacy_risk": avg_privacy,
    }


def _write_runs_csv(path: Path, runs: Iterable[Run]) -> None:
    fieldnames = [
        "run_id",
        "created_at",
        "prompt",
        "response",
        "blocked",
        "error",
        "overall_pass",
        "toxicity",
        "privacy_risk",
        "modifier_id",
        "modifier_category",
    ]
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            judge = run.judge_scores or {}
            det = run.detector_scores or {}
            toxicity = det.get("toxicity", {})
            privacy = det.get("pii", {})
            writer.writerow(
                {
                    "run_id": run.id,
                    "created_at": run.created_at.isoformat(),
                    "prompt": run.prompt,
                    "response": run.response,
                    "blocked": run.blocked,
                    "error": run.error,
                    "overall_pass": bool(judge.get("overall_pass")),
                    "toxicity": (
                        toxicity.get("toxicity")
                        or toxicity.get("toxicity_score")
                        or toxicity.get("severe_toxicity")
                    ),
                    "privacy_risk": privacy.get("privacy_risk"),
                    "modifier_id": run.modifier_id,
                    "modifier_category": run.modifier_category,
                }
            )


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


async def execute_baseline(
    *, config_path: Path, output_dir: Path, skip_run: bool = False
) -> Dict[str, Any]:
    config = BaselineConfig.load(config_path)
    await init_db()
    experiment = await _ensure_experiment(
        name=config.experiment_name,
        model=config.model,
        description=config.description,
    )

    if not skip_run:
        await run_full_experiment(
            exp_id=experiment.id,
            instruction=config.instruction,
            model=config.model,
            prompt_texts=config.prompts,
            judge_model=config.judge.model,
            use_openai_judge=config.judge.use_openai,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            parallelism=config.parallelism,
            model_timeout=config.model_timeout,
        )

    runs = await _load_runs(experiment.id, instruction=config.instruction)
    metrics = _aggregate_metrics(runs)
    timestamp = datetime.now(timezone.utc).isoformat()

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_dir / "baseline_metrics.json",
        {
            "generated_at": timestamp,
            "experiment_id": experiment.id,
            "experiment_name": experiment.name,
            "model": experiment.model,
            "run_count": metrics["total_runs"],
            "metrics": metrics,
            "config_path": str(config_path),
        },
    )
    _write_runs_csv(output_dir / "baseline_runs.csv", runs)
    _write_json(
        output_dir / "baseline_runs.json",
        [
            {
                "run_id": run.id,
                "created_at": run.created_at.isoformat(),
                "prompt": run.prompt,
                "response": run.response,
                "blocked": run.blocked,
                "error": run.error,
                "judge_scores": run.judge_scores,
                "detector_scores": run.detector_scores,
                "metadata": run.run_metadata,
            }
            for run in runs
        ],
    )

    return {
        "experiment_id": experiment.id,
        "experiment_name": experiment.name,
        "timestamp": timestamp,
        "metrics": metrics,
    }


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Seed or replay baseline experiments.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("baseline/baseline_config.json"),
        help="Path to the baseline configuration JSON.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("baseline/results"),
        help="Directory to store baseline exports (JSON/CSV).",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Reuse existing runs instead of launching a new baseline run.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    summary = asyncio.run(
        execute_baseline(config_path=args.config, output_dir=args.out, skip_run=args.skip_run)
    )
    metrics = summary["metrics"]
    print(
        f"[baseline] {summary['experiment_name']} ({summary['experiment_id']}) "
        f"runs={metrics['total_runs']} pass_rate={metrics['pass_rate']:.2%} "
        f"toxicity={metrics['avg_toxicity']:.3f} privacy={metrics['avg_privacy_risk']:.3f}"
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
