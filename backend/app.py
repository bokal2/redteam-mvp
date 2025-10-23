# backend/app.py
import asyncio
from fastapi import FastAPI, HTTPException
from .db import init_db, AsyncSessionLocal
from .models import Experiment, Run, JudgePrompt
from .schemas import ExperimentCreate, RunCreate
from sqlmodel import select
from .orchestrator import run_full_experiment

app = FastAPI(title="RedTeam Masters Capstone")


@app.on_event("startup")
async def on_startup():
    await init_db()


@app.post("/experiments")
async def create_experiment(payload: ExperimentCreate):
    async with AsyncSessionLocal() as session:
        exp = Experiment(
            name=payload.name, description=payload.description, model=payload.model
        )
        session.add(exp)
        await session.commit()
        await session.refresh(exp)
        return {"id": exp.id, "name": exp.name, "model": exp.model}


@app.get("/experiments")
async def list_experiments():
    async with AsyncSessionLocal() as session:
        q = await session.exec(select(Experiment))
        rows = q.all()
        return rows


@app.post("/experiments/{exp_id}/run")
async def run_experiment(exp_id: str, payload: RunCreate):
    # fetch experiment to get model
    async with AsyncSessionLocal() as session:
        q = await session.exec(select(Experiment).where(Experiment.id == exp_id))
        exp = q.one_or_none()
        if not exp:
            raise HTTPException(status_code=404, detail="Experiment not found")
    # orchestrate
    try:
        result = await run_full_experiment(
            exp_id=exp_id,
            instruction=payload.instruction,
            model=exp.model,
            judge_model=payload.judge_model,
            use_openai_judge=payload.use_openai_judge,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/experiments/{exp_id}/runs")
async def get_runs(exp_id: str):
    async with AsyncSessionLocal() as session:
        q = await session.exec(
            select(Run)
            .where(Run.experiment_id == exp_id)
            .order_by(Run.created_at.desc())
        )
        rows = q.all()
        return rows


@app.get("/experiments/{exp_id}/metrics")
async def metrics(exp_id: str):
    # compute aggregated metrics: pass_rate, avg toxicity (per run averaged), avg privacy risk
    async with AsyncSessionLocal() as session:
        q = await session.exec(select(Run).where(Run.experiment_id == exp_id))
        runs = q.all()
    if not runs:
        raise HTTPException(status_code=404, detail="No runs found")
    total = len(runs)
    passed = sum(1 for r in runs if (r.judge_scores or {}).get("overall_pass"))
    # tox average across runs: use detox keys if present
    tox_vals = []
    privacy_vals = []
    for r in runs:
        det = r.detector_scores or {}
        tox = det.get("toxicity", {})
        # detox returns dict; pick 'toxicity' key if present
        if isinstance(tox, dict):
            val = tox.get("toxicity") or tox.get("toxicity_score") or 0.0
        else:
            val = 0.0
        if isinstance(val, (int, float)):
            tox_vals.append(val)
        pii = det.get("pii", {})
        pv = pii.get("privacy_risk", 0.0) if isinstance(pii, dict) else 0.0
        privacy_vals.append(pv)
    avg_tox = (sum(tox_vals) / len(tox_vals)) if tox_vals else 0.0
    avg_priv = (sum(privacy_vals) / len(privacy_vals)) if privacy_vals else 0.0
    return {
        "total_runs": total,
        "passed": passed,
        "pass_rate": passed / total,
        "avg_toxicity": avg_tox,
        "avg_privacy_risk": avg_priv,
    }
