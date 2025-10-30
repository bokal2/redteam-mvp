from fastapi import FastAPI, HTTPException
from sqlmodel import select

from .agent import generate_prompts_async
from .db import AsyncSessionLocal, init_db
from .models import AgentRun, Experiment, Run
from .orchestrator import RedTeamAgent, run_full_experiment
from .schemas import AgentRunCreate, ExperimentCreate, RunCreate

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

    async with AsyncSessionLocal() as session:
        q = await session.exec(select(Experiment).where(Experiment.id == exp_id))
        exp = q.one_or_none()
        if not exp:
            raise HTTPException(status_code=404, detail="Experiment not found")

    try:
        result = await run_full_experiment(
            exp_id=exp_id,
            instruction=payload.instruction,
            model=exp.model,
            judge_model=payload.judge_model,
            use_openai_judge=payload.use_openai_judge,
            prompt_texts=list(payload.prompts),
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

    async with AsyncSessionLocal() as session:
        q = await session.exec(select(Run).where(Run.experiment_id == exp_id))
        runs = q.all()
    if not runs:
        raise HTTPException(status_code=404, detail="No runs found")
    serialized = [RedTeamAgent._serialize_run(r) for r in runs]
    summary = RedTeamAgent._summarize_runs(serialized)
    return summary


@app.post("/experiments/{exp_id}/agent-run")
async def run_agent_experiment(exp_id: str, payload: AgentRunCreate):
    async with AsyncSessionLocal() as session:
        query = await session.exec(select(Experiment).where(Experiment.id == exp_id))
        experiment = query.one_or_none()
        if not experiment:
            raise HTTPException(status_code=404, detail="Experiment not found")

    metrics_focus = list(payload.metrics)
    prompts = await generate_prompts_async(
        experiment_name=experiment.name,
        instruction=payload.instruction,
        metrics_focus=metrics_focus,
        prompt_count=payload.prompt_count,
        target_model=payload.target_model or experiment.model,
        guidance=payload.guidance,
        generator_model=payload.generator_model or "gpt-4.1-mini",
    )
    prompt_texts = [item.prompt for item in prompts]

    try:
        execution = await run_full_experiment(
            exp_id=experiment.id,
            instruction=payload.instruction,
            model=payload.target_model or experiment.model,
            judge_model=payload.judge_model,
            use_openai_judge=payload.use_openai_judge,
            prompt_texts=prompt_texts,
            temperature=payload.temperature,
            max_tokens=payload.max_tokens,
            parallelism=payload.parallelism,
            model_timeout=payload.model_timeout,
        )
    except Exception as exc:  # pragma: no cover - runtime failure
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    run_ids = [
        run.get("run_id") for run in execution.get("runs", []) if run.get("run_id")
    ]

    async with AsyncSessionLocal() as session:
        record = AgentRun(
            experiment_id=experiment.id,
            generator_model=payload.generator_model or "gpt-4.1-mini",
            prompt_count=len(prompt_texts),
            metrics_focus=metrics_focus,
            target_temperature=payload.temperature,
            guidance=payload.guidance,
            request_payload=payload.model_dump(),
            generated_prompts=[item.as_dict() for item in prompts],
            execution_summary=execution.get("summary"),
            run_ids=run_ids,
        )
        session.add(record)
        await session.commit()
        await session.refresh(record)

    return {
        "agent_run_id": record.id,
        "experiment_id": experiment.id,
        "generated_prompts": record.generated_prompts,
        "execution": execution,
    }


@app.get("/experiments/{exp_id}/agent-runs")
async def list_agent_runs(exp_id: str):
    async with AsyncSessionLocal() as session:
        query = await session.exec(
            select(AgentRun)
            .where(AgentRun.experiment_id == exp_id)
            .order_by(AgentRun.created_at.desc())
        )
        return query.all()


@app.get("/experiments/{exp_id}/agent-metrics")
async def agent_metrics(exp_id: str):
    async with AsyncSessionLocal() as session:
        query = await session.exec(
            select(AgentRun).where(AgentRun.experiment_id == exp_id)
        )
        records = query.all()

    if not records:
        raise HTTPException(status_code=404, detail="No agent runs found.")

    total_prompts = 0
    weighted_passes = 0.0
    weighted_toxicity = 0.0
    weighted_privacy = 0.0
    history = []

    for record in records:
        summary = record.execution_summary or {}
        run_total = summary.get("total_runs", record.prompt_count or 0)
        total_prompts += run_total
        pass_rate = summary.get("pass_rate", 0.0)
        avg_toxicity = summary.get("avg_toxicity", 0.0)
        avg_privacy = summary.get("avg_privacy_risk", 0.0)
        weighted_passes += pass_rate * run_total
        weighted_toxicity += avg_toxicity * run_total
        weighted_privacy += avg_privacy * run_total
        history.append(
            {
                "agent_run_id": record.id,
                "created_at": record.created_at.isoformat(),
                "metrics": summary,
                "prompt_count": run_total,
                "metrics_focus": record.metrics_focus,
                "target_temperature": record.target_temperature,
            }
        )

    aggregated = {
        "total_agent_runs": len(records),
        "total_prompts": total_prompts,
        "avg_pass_rate": (weighted_passes / total_prompts) if total_prompts else 0.0,
        "avg_toxicity": (weighted_toxicity / total_prompts) if total_prompts else 0.0,
        "avg_privacy_risk": (weighted_privacy / total_prompts) if total_prompts else 0.0,
    }

    return {"aggregate": aggregated, "history": history}
