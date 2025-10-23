# RedTeam MVP

An evaluation harness for red-teaming large language models. The project couples a FastAPI backend, SQLModel storage, adversarial prompt generation, automated detectors, and a Streamlit dashboard so you can repeatedly probe an inference model (Ollama and/or OpenAI) and track safety metrics over time.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Repository Layout](#repository-layout)
- [Prerequisites](#prerequisites)
- [Backend Setup](#backend-setup)
- [Frontend Setup](#frontend-setup)
- [Running the System](#running-the-system)
- [Core Concepts](#core-concepts)
- [REST API Reference](#rest-api-reference)
- [Testing & Tooling](#testing--tooling)
- [Troubleshooting](#troubleshooting)
- [Roadmap Ideas](#roadmap-ideas)
- [Contributing](#contributing)

---

## Architecture Overview

```
┌──────────────────┐      ┌──────────────────────────────┐
│  Streamlit UI    │──────│ FastAPI Backend (uvicorn)    │──────┐
└──────────────────┘      │  • Experiment CRUD           │      │
                          │  • Run orchestration         │      │
                          │  • Metrics aggregation       │      │
                          └──────────────┬───────────────┘      │
                                         │                      │
                                         ▼                      │
                               ┌──────────────────┐             │
                               │ SQLModel / SQLite│             │
                               │ redteam_master.db│             │
                               └──────────────────┘             │
                                                                 │
     ┌────────────────┐    ┌────────────────┐     ┌──────────────┘
     │  Ollama Model  │    │  OpenAI Judge  │     │  Detectors
     │ (localhost)    │    │ (optional)     │     │  • Detoxify toxicity
     └────────────────┘    └────────────────┘     │  • spaCy PII heuristics
                                                  └───────────────
```

1. **Streamlit frontend**: Users create experiments, trigger runs, and inspect results.
2. **FastAPI backend**: Persists experiments, orchestrates prompt generation, calls inference + detectors, and aggregates metrics.
3. **Judging**: Can use OpenAI (when `OPENAI_API_KEY` is set) or fall back to an Ollama-hosted model for cost-free evaluation.
4. **Detectors**: Toxicity (Detoxify) and privacy (regex + spaCy + phonenumbers) score each model response.
5. **Storage**: SQLModel on top of SQLite (`redteam_master.db`). Async sessions allow streaming workflows.

---

## Key Features

- **Adversarial prompt generation**: Fixed and user-supplied modifiers expand a base instruction into multiple red-team scenarios.
- **Pluggable inference**: Primary model served via Ollama (`/api/generate`). Configure per experiment.
- **Dual judge pipeline**:
  - OpenAI GPT-4o family (async API) for high-quality scoring.
  - Automatic fallback to an Ollama local model when OpenAI is disabled or fails.
- **Detectors**:
  - Detoxify for toxicity scores (gracefully skips if checkpoints missing).
  - Privacy heuristics combining regex, phone parsing, and spaCy NER.
- **Streamlit dashboard**: Launch experiments, inspect run-by-run details, and review aggregate metrics.
- **Async orchestration**: Concurrent prompt processing keeps throughput high without overwhelming local resources.

---

## Tech Stack

- **Python 3.12**
- **FastAPI**, **SQLModel**, **SQLAlchemy Async**, **Uvicorn**
- **Streamlit** (frontend)
- **httpx** (async HTTP client)
- **Detoxify**, **spaCy**, **phonenumbers** (detectors)
- **OpenAI Python SDK** (>=1.0) for GPT judging
- **Ollama** (local LLM serving)
- **Pandas** (dataframes in UI)
- **LangChain / LangGraph** (optional future integrations; currently only prompt formatting remains)

---

## Repository Layout

```
backend/
  app.py              FastAPI application entrypoint
  db.py               Async SQLModel engine/session setup
  detectors.py        Toxicity + privacy detectors
  judge.py            OpenAI + Ollama judging helpers
  models.py           SQLModel table definitions
  ola_client.py       Thin async client for Ollama's REST API
  orchestrator.py     Core experiment workflow
  prompts.py          Base + adversarial prompt generation
  schemas.py          Pydantic request models
frontend/
  app.py              Streamlit dashboard
  requirements.txt    Frontend-specific dependencies
requirements.txt      Backend / shared Python deps
test_api.py           Minimal async judge smoke test
README.md             This document
redteam_master.db     SQLite database (created at runtime)
local.env/.env        Example environment variable storage
```

---

## Prerequisites

1. **Python 3.12** (recommend `pyenv` or system Python).
2. **Ollama** installed and running locally (`ollama serve`) with at least one model pulled, e.g. `ollama pull llama3.2`.
3. **OpenAI API key** (optional but recommended) with compatible access to `gpt-4o-mini` or an alternative of your choice.
4. **System dependencies** for spaCy (`libffi`, `gcc`, etc.) depending on OS.
5. **pip** for package management.

> The repo ships with a `venv/` folder. You can reuse it, but the recommended approach is to create a fresh virtual environment so dependency management stays predictable.

---

## Backend Setup

```bash
# 1. Create & activate a virtual environment
python3.12 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install backend dependencies
pip install -r requirements.txt

# 3. spaCy English model (required for privacy detector)
python -m spacy download en_core_web_sm

# 4. (Optional) Detoxify runs best with PyTorch; install if missing
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Configure environment variables (create a `.env` file or export them directly):

```bash
export OPENAI_API_KEY="sk-..."        # optional
export OLLAMA_BASE="http://localhost:11434"  # default matches code
```

Initialize the database on first run (tables auto-create during FastAPI startup).

Run the backend:

```bash
uvicorn backend.app:app --reload
# FastAPI docs available at http://localhost:8000/docs
```

---

## Frontend Setup

The Streamlit dashboard lives in `frontend/app.py`.

```bash
# Activate the same virtual environment used for the backend
source .venv/bin/activate

pip install -r frontend/requirements.txt

streamlit run frontend/app.py
# Streamlit launches on http://localhost:8501 by default
```

The UI expects the FastAPI backend to be running on `http://localhost:8000` (adjust `API` constant in `frontend/app.py` if needed).

---

## Running the System

1. **Start supporting services**
   - `ollama serve` (ensure the model referenced in your experiment exists, e.g. `ollama pull llama3.2`).
   - Export `OPENAI_API_KEY` if you want OpenAI-based judging.
2. **Launch backend** (`uvicorn backend.app:app --reload`).
3. **Launch frontend** (`streamlit run frontend/app.py`).
4. **Streamlit workflow**:
   - Create an experiment (name, description, primary Ollama model).
   - Select the experiment from the dropdown.
   - Enter an instruction to red-team.
   - Choose whether to **Use OpenAI judge**. When disabled, the backend automatically switches to the local Ollama judge (using the provided `judge_model` hint).
   - Trigger “Run experiment” to evaluate across all adversarial modifiers.
   - Inspect run details, detector scores, judge verdicts, and aggregate metrics on the same page.

Each execution produces multiple `Run` records (one per modifier), persists them to SQLite, and surfaces aggregated pass rates and risk scores.

---

## Core Concepts

### Experiments
Stored in `backend/models.py`. Each experiment represents a target model + configuration. Fields:
- `id`, `name`, `description`, `model`, `created_at`.

### Runs
One per adversarial prompt. Captures prompt, response, detector outputs, judge scores, and timestamps. Automatically persisted during orchestration.

### Judge Options
- **OpenAI judge** (`run_judge_openai_async`): Uses GPT-4o series to evaluate safety, privacy, bias, instruction-following, and pass/fail. Retries on parse errors and returns normalized scores.
- **Local judge** (`run_judge_local_async`): Reuses the same rubric but calls an Ollama-served model. Useful when you need to conserve API credits or operate offline.
- The frontend exposes a toggle and model selector/hint so users can choose per run.

### Detectors
- `run_toxicity`: Wraps Detoxify. If the package or checkpoints are unavailable, the system logs a warning and returns `{"warning": "detoxify_unavailable"}` instead of failing hard.
- `run_pii_checks`: Uses regex, `phonenumbers`, and spaCy NER to estimate privacy risk.

### Prompt Generation
`backend/prompts.py` composes a base system prompt and merges curated adversarial modifiers. You can extend `ADVERSARIAL_MODIFIERS` or inject additional modifiers per run (`extra_modifiers` parameter on `run_full_experiment`).

---

## REST API Reference

| Method | Endpoint                           | Description                                           |
|--------|------------------------------------|-------------------------------------------------------|
| POST   | `/experiments`                     | Create a new experiment (`name`, `description`, `model`). |
| GET    | `/experiments`                     | List all experiments.                                 |
| POST   | `/experiments/{exp_id}/run`        | Trigger a full experiment run. Payload includes `instruction`, optional `judge_model`, and `use_openai_judge` toggle. |
| GET    | `/experiments/{exp_id}/runs`       | Retrieve all runs for an experiment (most recent first). |
| GET    | `/experiments/{exp_id}/metrics`    | Summary stats (pass rate, average toxicity/privacy).  |

Interactive docs are available at `http://localhost:8000/docs` when the backend is running.

---

## Testing & Tooling

- **Async judge smoke test**: `python test_api.py` (requires `OPENAI_API_KEY`). Demonstrates direct usage of the OpenAI judge helper.
- **Static checks / formatting**: not yet configured; consider adding `ruff`/`black` for consistency.
- **Manual QA**: Use the Streamlit UI or `curl`/`HTTPie` to exercise endpoints.

---

## Troubleshooting

| Issue | Likely Cause | Fix |
|-------|--------------|-----|
| `OPENAI_API_KEY environment variable not set` | Judge toggle is ON but key missing | Export `OPENAI_API_KEY` or disable OpenAI judge in UI. |
| `OpenAIError` with fallback note in judge output | OpenAI call failed | The system automatically retries and falls back to the local judge; inspect `notes` for details. |
| `detoxify_unavailable` warning in detector scores | Detoxify not installed or model download failed | `pip install detoxify` and ensure internet access; restarting after dependencies are satisfied will re-enable toxicity scoring. |
| spaCy error about `en_core_web_sm` missing | Model not downloaded | Run `python -m spacy download en_core_web_sm`. |
| Ollama request errors / empty responses | Ollama daemon not running or model missing | Start `ollama serve` and `ollama pull <model>` referenced in experiments. |
| SQLite locking issues | Running multiple writers simultaneously | Reduce `parallelism` in `run_full_experiment` or move to a managed SQL database. |

---

## Roadmap Ideas

- Add LangGraph flows to orchestrate more complex evaluation pipelines.
- Enrich prompt catalog with auto-generated modifiers or uploadable templates.
- Introduce additional detectors (e.g., jailbreak classifiers, bias detection).
- Support alternative backends (OpenRouter, Azure OpenAI, Anthropic, etc.).
- Provide scheduling / automation hooks to run experiments on a cadence.
- Add CI linting and pytest-based suites for regression coverage.

---

## Contributing

1. Fork the repository and clone your fork.
2. Create a virtual environment and install dependencies.
3. Make your changes with clear commit messages.
4. Run `python -m compileall backend frontend` or your preferred lint/test commands to ensure nothing obvious breaks.
5. Submit a pull request describing your changes, testing performed, and any follow-up work you recommend.

Bug reports and feature requests are welcome—open an issue with as much detail as possible (logs, reproduction steps, environment info).  

Happy red-teaming! 🎯
