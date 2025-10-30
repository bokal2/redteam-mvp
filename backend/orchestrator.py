# backend/orchestrator.py
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional

from sqlmodel import select

from .db import AsyncSessionLocal
from .judge import (
    fallback_scores,
    openai_client_available,
    run_judge_local_async,
    run_judge_openai_async,
)
from .models import Experiment, Run
from .ola_client import generate as ola_generate
from .prompts import REFUSAL_TOKEN, generate_prompts, is_potentially_unsafe

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Explicit opt-in to allow unsafe prompts to flow to the model.
REDTEAM_ALLOW_UNSAFE = os.getenv("REDTEAM_ALLOW_UNSAFE", "0") == "1"


class RedTeamAgent:
    """Coordinates prompt generation, model calls, detectors, and judging."""

    def __init__(
        self,
        *,
        session_factory=AsyncSessionLocal,
        concurrency: int = 3,
        model_timeout: int = 30,
    ) -> None:
        self._session_factory = session_factory
        self._concurrency = max(1, concurrency)
        self._model_timeout = model_timeout

    async def run_experiment(
        self,
        *,
        experiment_id: str,
        instruction: str,
        model: str,
        use_openai_judge: bool,
        judge_model: Optional[str],
        temperature: float,
        max_tokens: int,
        prompt_texts: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        experiment = await self._get_experiment(experiment_id)
        if not prompt_texts:
            raise ValueError("No prompts provided for run.")
        prompts = generate_prompts(
            instruction=instruction,
            user_prompts=prompt_texts,
        )

        semaphore = asyncio.Semaphore(self._concurrency)
        results: List[Optional[Dict[str, Any]]] = [None] * len(prompts)

        async def worker(idx: int, prompt_obj: Dict[str, Any]) -> None:
            async with semaphore:
                try:
                    results[idx] = await self._execute_prompt(
                        experiment_id=experiment.id,
                        instruction=instruction,
                        model=model,
                        prompt_obj=prompt_obj,
                        use_openai_judge=use_openai_judge,
                        judge_model=judge_model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    logger.exception(
                        "RedTeamAgent worker failed for modifier=%s",
                        prompt_obj.get("modifier_id"),
                    )
                    results[idx] = {
                        "run_id": None,
                        "prompt": prompt_obj.get("prompt_text"),
                        "response": None,
                        "detector_scores": None,
                        "judge_scores": fallback_scores(
                            f"worker_error:{exc.__class__.__name__}"
                        ),
                        "blocked": False,
                        "error": str(exc),
                        "modifier_id": prompt_obj.get("modifier_id"),
                        "modifier_category": prompt_obj.get("modifier_category"),
                    }

        await asyncio.gather(
            *(asyncio.create_task(worker(idx, prompt)) for idx, prompt in enumerate(prompts))
        )

        completed = [r for r in results if r is not None]
        summary = self._summarize_runs(completed)

        return {
            "experiment_id": experiment.id,
            "model": model,
            "instruction": instruction,
            "summary": summary,
            "runs": completed,
        }

    async def _get_experiment(self, experiment_id: str) -> Experiment:
        async with self._session_factory() as session:
            query = await session.exec(
                select(Experiment).where(Experiment.id == experiment_id)
            )
            experiment = query.one_or_none()
            if experiment is None:
                raise ValueError("Experiment not found")
            return experiment

    async def _execute_prompt(
        self,
        *,
        experiment_id: str,
        instruction: str,
        model: str,
        prompt_obj: Dict[str, Any],
        use_openai_judge: bool,
        judge_model: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        prompt_text = prompt_obj["prompt_text"]
        modifier_id = prompt_obj.get("modifier_id")
        modifier_category = prompt_obj.get("modifier_category")

        metadata_payload = {
            "modifier_text": prompt_obj.get("modifier_text"),
            "judge_model": judge_model,
            "use_openai_judge": use_openai_judge,
            "model": model,
            "timestamp": int(time.time()),
        }

        if self._should_block_prompt(instruction, prompt_text):
            logger.info(
                "Blocking prompt for modifier=%s due to safety gate", modifier_id
            )
            detector_scores = {
                "toxicity": {"toxicity": 0.0},
                "pii": {"privacy_risk": 0.0},
            }
            judge_scores = {
                "safety": 1.0,
                "privacy_risk": 0.0,
                "bias": 0.0,
                "instruction_following": 0.0,
                "overall_pass": False,
                "notes": "Blocked by safety gate (keyword match)",
            }
            run_record = await self._persist_run(
                experiment_id=experiment_id,
                instruction=instruction,
                prompt_obj=prompt_obj,
                response=REFUSAL_TOKEN,
                blocked=True,
                detector_scores=detector_scores,
                judge_scores=judge_scores,
                error=None,
                metadata=metadata_payload | {"blocked_reason": "keyword_match"},
            )
            return self._serialize_run(run_record)

        response_text = await self._call_model_with_timeout(
            model=model,
            prompt=prompt_text,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        judge_scores = await self._run_judges(
            instruction=instruction,
            response=response_text,
            model=model,
            use_openai_judge=use_openai_judge,
            judge_model=judge_model,
        )
        detector_scores = self._derive_metrics_from_judge(judge_scores)

        run_record = await self._persist_run(
            experiment_id=experiment_id,
            instruction=instruction,
            prompt_obj=prompt_obj,
            response=response_text,
            blocked=False,
            detector_scores=detector_scores,
            judge_scores=judge_scores,
            error=None,
            metadata=metadata_payload,
        )
        return self._serialize_run(run_record)

    @staticmethod
    def _should_block_prompt(instruction: str, prompt_text: str) -> bool:
        if REDTEAM_ALLOW_UNSAFE:
            return False
        return is_potentially_unsafe(instruction) or is_potentially_unsafe(prompt_text)

    async def _call_model_with_timeout(
        self, *, model: str, prompt: str, temperature: float, max_tokens: int
    ) -> str:
        try:
            generate_coro = ola_generate(
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            result = await asyncio.wait_for(generate_coro, timeout=self._model_timeout)
        except asyncio.TimeoutError:
            logger.error("Model call timed out for model=%s", model)
            return "[ERROR:MODEL_TIMEOUT]"
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Model call error for model=%s", model)
            return f"[ERROR:MODEL_EXCEPTION:{exc.__class__.__name__}]"

        if isinstance(result, dict):
            return (
                result.get("text")
                or result.get("completion")
                or result.get("response")
                or str(result)
            )
        return str(result)

    @staticmethod
    def _derive_metrics_from_judge(judge_scores: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        scores = judge_scores or {}
        toxicity_entry: Dict[str, Any] = {"source": "llm_judge"}
        safety = scores.get("safety")
        if isinstance(safety, (int, float)):
            toxicity_entry["safety"] = float(safety)
            toxicity_entry["toxicity"] = max(0.0, min(1.0, 1.0 - float(safety)))
        bias = scores.get("bias")
        if isinstance(bias, (int, float)):
            toxicity_entry["bias"] = max(0.0, min(1.0, float(bias)))

        privacy_entry: Dict[str, Any] = {"source": "llm_judge"}
        privacy = scores.get("privacy_risk")
        if isinstance(privacy, (int, float)):
            privacy_entry["privacy_risk"] = max(0.0, min(1.0, float(privacy)))

        return {"toxicity": toxicity_entry, "pii": privacy_entry}

    async def _run_judges(
        self,
        *,
        instruction: str,
        response: str,
        model: str,
        use_openai_judge: bool,
        judge_model: Optional[str],
    ) -> Dict[str, Any]:
        openai_model = judge_model or "gpt-4o-mini"
        local_model = judge_model if (judge_model and not use_openai_judge) else model
        openai_error: Optional[Exception] = None
        judge_result: Optional[Dict[str, Any]] = None

        if use_openai_judge and openai_client_available():
            try:
                judge_result = await run_judge_openai_async(
                    instruction=instruction,
                    response=response,
                    model=openai_model,
                    temperature=0.0,
                    max_tokens=400,
                    retries=2,
                )
            except Exception as exc:
                openai_error = exc
                logger.warning(
                    "OpenAI judge failed (%s); falling back to local judge", exc
                )

        if judge_result is None:
            try:
                judge_result = await run_judge_local_async(
                    instruction=instruction,
                    response=response,
                    model=local_model,
                    temperature=0.0,
                    max_tokens=400,
                    retries=2,
                )
                if openai_error:
                    note = judge_result.get("notes", "")
                    suffix = f"OpenAI fallback: {openai_error.__class__.__name__}"
                    judge_result["notes"] = f"{note}; {suffix}" if note else suffix
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("Local judge failed: %s", exc)
                judge_result = fallback_scores(
                    f"local_judge_error:{exc.__class__.__name__}", local_model
                )

        return judge_result or fallback_scores("judge_unavailable", local_model)

    async def _persist_run(
        self,
        *,
        experiment_id: str,
        instruction: str,
        prompt_obj: Dict[str, Any],
        response: Optional[str],
        blocked: bool,
        detector_scores: Optional[Dict[str, Any]],
        judge_scores: Optional[Dict[str, Any]],
        error: Optional[str],
        metadata: Optional[Dict[str, Any]],
    ) -> Run:
        async with self._session_factory() as session:
            run_record = Run(
                experiment_id=experiment_id,
                base_instruction=instruction,
                prompt=prompt_obj.get("prompt_text", ""),
                response=response,
                blocked=blocked,
                error=error,
                modifier_id=prompt_obj.get("modifier_id"),
                modifier_category=prompt_obj.get("modifier_category"),
                detector_scores=detector_scores,
                judge_scores=judge_scores,
                run_metadata=metadata,
            )
            session.add(run_record)
            await session.commit()
            await session.refresh(run_record)
            return run_record

    @staticmethod
    def _serialize_run(run_record: Run) -> Dict[str, Any]:
        return {
            "run_id": run_record.id,
            "prompt": run_record.prompt,
            "response": run_record.response,
            "detector_scores": run_record.detector_scores,
            "judge_scores": run_record.judge_scores,
            "blocked": run_record.blocked,
            "error": run_record.error,
            "modifier_id": run_record.modifier_id,
            "modifier_category": run_record.modifier_category,
            "metadata": run_record.run_metadata,
        }

    @staticmethod
    def _summarize_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
        total = len(runs)
        blocked = sum(1 for r in runs if r.get("blocked"))
        errored = sum(1 for r in runs if r.get("error"))
        passed = sum(
            1 for r in runs if (r.get("judge_scores") or {}).get("overall_pass")
        )
        pass_rate = passed / total if total else 0.0

        tox_vals: List[float] = []
        pii_vals: List[float] = []
        for r in runs:
            det = r.get("detector_scores") or {}
            tox = det.get("toxicity") or {}
            val: Optional[float] = None
            if isinstance(tox, dict):
                for key in ("toxicity", "risk", "derived_toxicity"):
                    candidate = tox.get(key)
                    if isinstance(candidate, (int, float)):
                        val = float(candidate)
                        break
                if val is None and isinstance(tox.get("safety"), (int, float)):
                    val = max(0.0, min(1.0, 1.0 - float(tox["safety"])))
            elif isinstance(tox, (int, float)):
                val = float(tox)

            if val is None:
                judge = r.get("judge_scores") or {}
                safety = judge.get("safety")
                if isinstance(safety, (int, float)):
                    val = max(0.0, min(1.0, 1.0 - float(safety)))
            if val is not None:
                tox_vals.append(val)

            pii = det.get("pii") or {}
            pv: Optional[float] = None
            if isinstance(pii, dict):
                for key in ("privacy_risk", "risk"):
                    candidate = pii.get(key)
                    if isinstance(candidate, (int, float)):
                        pv = float(candidate)
                        break
            elif isinstance(pii, (int, float)):
                pv = float(pii)

            if pv is None:
                judge = r.get("judge_scores") or {}
                privacy = judge.get("privacy_risk")
                if isinstance(privacy, (int, float)):
                    pv = max(0.0, min(1.0, float(privacy)))
            if pv is not None:
                pii_vals.append(pv)

        avg_toxicity = (sum(tox_vals) / len(tox_vals)) if tox_vals else 0.0
        avg_privacy = (sum(pii_vals) / len(pii_vals)) if pii_vals else 0.0

        return {
            "total_runs": total,
            "blocked_runs": blocked,
            "errored_runs": errored,
            "passed": passed,
            "pass_rate": pass_rate,
            "avg_toxicity": avg_toxicity,
            "avg_privacy_risk": avg_privacy,
        }


async def run_full_experiment(
    *,
    exp_id: str,
    instruction: str,
    model: str,
    prompt_texts: Optional[List[str]] = None,
    judge_model: Optional[str] = None,
    use_openai_judge: bool = True,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    parallelism: int = 3,
    model_timeout: int = 30,
) -> Dict[str, Any]:
    agent = RedTeamAgent(
        concurrency=parallelism,
        model_timeout=model_timeout,
    )
    return await agent.run_experiment(
        experiment_id=exp_id,
        instruction=instruction,
        model=model,
        use_openai_judge=use_openai_judge,
        judge_model=judge_model,
        temperature=temperature,
        max_tokens=max_tokens,
        prompt_texts=prompt_texts,
    )
