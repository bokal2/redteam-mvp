# backend/orchestrator.py
from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional, Any

from sqlmodel import select

from .db import AsyncSessionLocal
from .models import Experiment, Run
from .ola_client import generate as ola_generate
from .prompts import generate_prompts
from .detectors import run_toxicity, run_pii_checks
from .judge import (
    run_judge_openai_async,
    run_judge_local_async,
    openai_client_available,
    fallback_scores,
)

logger = logging.getLogger(__name__)


async def _run_single_prompt(
    exp_id: str,
    instruction: str,
    model: str,
    prompt_text: str,
    use_openai_judge: bool = True,
    judge_model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> Dict[str, Any]:
    """
    Executes the full flow for a single adversarial prompt:
      1) Call LLM (Ollama)
      2) Run detectors (toxicity + PII)
      3) Run judge (OpenAI or local Ollama fallback)
      4) Persist Run
    Returns a dict with persisted run data.
    """
    # 1) model generation
    gen = await ola_generate(
        model=model,
        prompt=prompt_text,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    response_text = gen.get("text", "") if isinstance(gen, dict) else str(gen)

    # 2) detectors (CPU-bound but fast; run in threadpool)
    loop = asyncio.get_event_loop()

    tox_future = loop.run_in_executor(None, run_toxicity, response_text)
    pii_future = loop.run_in_executor(None, run_pii_checks, response_text)
    tox_scores, pii_scores = await asyncio.gather(tox_future, pii_future)

    # 3) judge, optionally using OpenAI
    selected_openai_model = judge_model or "gpt-4o-mini"
    # If we're primarily using OpenAI, fall back to the experiment's base model for local judging.
    local_judge_model = (
        judge_model if not use_openai_judge else model
    )
    judge_result: Optional[Dict[str, Any]] = None
    openai_error: Optional[Exception] = None

    if use_openai_judge and openai_client_available():
        try:
            judge_result = await run_judge_openai_async(
                instruction=instruction,
                response=response_text,
                model=selected_openai_model,
                temperature=0.0,
                max_tokens=400,
                retries=2,
            )
        except Exception as exc:
            openai_error = exc
            logger.warning(
                "OpenAI judge failed (%s); falling back to local judge.",
                exc,
            )
    elif use_openai_judge and not openai_client_available():
        logger.warning("OpenAI judge requested but client unavailable; using local judge instead.")

    if judge_result is None:
        try:
            judge_result = await run_judge_local_async(
                instruction=instruction,
                response=response_text,
                model=local_judge_model,
                temperature=0.0,
                max_tokens=400,
                retries=2,
            )
            if openai_error:
                existing_note = judge_result.get("notes", "")
                suffix = f"OpenAI fallback due to {openai_error.__class__.__name__}"
                judge_result["notes"] = (
                    f"{existing_note}; {suffix}" if existing_note else suffix
                )
        except Exception as exc:
            logger.error("Local judge failed (%s). Returning fallback scores.", exc)
            judge_result = fallback_scores(
                f"local_judge_error:{exc.__class__.__name__}", local_judge_model
            )

    # 4) persist using a fresh session per run so concurrent workers do not share state
    async with AsyncSessionLocal() as session:
        run_record = Run(
            experiment_id=exp_id,
            prompt=prompt_text,
            response=response_text,
            detector_scores={"toxicity": tox_scores, "pii": pii_scores},
            judge_scores=judge_result,
        )
        session.add(run_record)
        await session.commit()
        await session.refresh(run_record)

    return {
        "run_id": run_record.id,
        "prompt": prompt_text,
        "response": response_text,
        "detector_scores": run_record.detector_scores,
        "judge_scores": run_record.judge_scores,
    }


async def run_full_experiment(
    exp_id: str,
    instruction: str,
    model: str,
    *,
    extra_modifiers: Optional[List[str]] = None,
    judge_model: Optional[str] = None,
    use_openai_judge: bool = True,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    parallelism: int = 3,
) -> Dict[str, Any]:
    """
    Orchestrates a full experiment:
      - Generate adversarial prompts
      - For each prompt: call model -> detectors -> judge -> store
      - Runs with limited parallelism for speed while keeping system stable

    Args:
        exp_id: Experiment ID
        instruction: Base instruction to test (user task)
        model: Ollama model name for generation (e.g., "llama3.2")
        extra_modifiers: Optional list of extra adversarial modifiers
        judge_model: Optional judge model override
        use_openai_judge: Toggle whether to call OpenAI judge (saves cost if disabled)
        temperature: Generation temperature (0.0 recommended for eval)
        max_tokens: Max tokens for generation
        parallelism: Number of concurrent prompt runs

    Returns:
        Dict with summary and list of run results.
    """
    prompts = generate_prompts(instruction, extra_modifiers)
    results: List[Optional[Dict[str, Any]]] = [None] * len(prompts)

    async with AsyncSessionLocal() as session:
        # Validate experiment exists
        q = await session.exec(select(Experiment).where(Experiment.id == exp_id))
        experiment = q.one_or_none()
        if not experiment:
            raise ValueError("Experiment not found")

    # Worker semaphore for bounded concurrency
    sem = asyncio.Semaphore(parallelism)

    async def worker(idx: int, prompt_text: str):
        async with sem:
            res = await _run_single_prompt(
                exp_id=exp_id,
                instruction=instruction,
                model=model,
                prompt_text=prompt_text,
                use_openai_judge=use_openai_judge,
                judge_model=judge_model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            results[idx] = res

    tasks = [
        asyncio.create_task(worker(idx, prompt_text))
        for idx, prompt_text in enumerate(prompts)
    ]
    await asyncio.gather(*tasks)

    # Build summary (basic)
    completed_results = [r for r in results if r is not None]
    total = len(completed_results)
    passed = sum(
        1 for r in completed_results if (r.get("judge_scores") or {}).get("overall_pass")
    )
    pass_rate = (passed / total) if total else 0.0

    # Average toxicity (using Detoxify's 'toxicity' if present)
    tox_vals: List[float] = []
    privacy_vals: List[float] = []
    for r in completed_results:
        det = r.get("detector_scores") or {}
        tox = det.get("toxicity") or {}
        if isinstance(tox, dict):
            val = tox.get("toxicity") or tox.get("toxicity_score") or 0.0
            if isinstance(val, (int, float)):
                tox_vals.append(float(val))
        pii = det.get("pii") or {}
        pv = pii.get("privacy_risk", 0.0) if isinstance(pii, dict) else 0.0
        privacy_vals.append(float(pv))

    avg_toxicity = (sum(tox_vals) / len(tox_vals)) if tox_vals else 0.0
    avg_privacy_risk = (sum(privacy_vals) / len(privacy_vals)) if privacy_vals else 0.0

    return {
        "experiment_id": exp_id,
        "summary": {
            "total_runs": total,
            "passed": passed,
            "pass_rate": pass_rate,
            "avg_toxicity": avg_toxicity,
            "avg_privacy_risk": avg_privacy_risk,
        },
        "runs": completed_results,
    }
