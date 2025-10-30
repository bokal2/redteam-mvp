# Baseline Reference

This folder captures the canonical manual baseline for the `test_emails` experiment.  
It provides everything needed to replay the run or compare future experiments against the stored metrics.

## Files
- `baseline_config.json` – experiment definition (model, instruction, judge toggle, prompt list).
- `results/baseline_metrics.json` – aggregate metrics exported from the harness (pass rate, toxicity, privacy).
- `results/baseline_runs.csv` / `results/baseline_runs.json` – per-prompt data for audits or reloads.

## Replaying the Baseline
1. Ensure the backend dependencies are installed (`pip install -r requirements.txt`) and the database is initialized.
2. Run the helper script:
   ```bash
   ./venv/bin/python -m backend.baseline --config baseline/baseline_config.json --out baseline/results
   ```
   Use `--skip-run` to regenerate exports without contacting the model.
3. The script will update the existing `test_emails` experiment (or create it if missing), run the prompts, and refresh the JSON/CSV artifacts.

## Using the Snapshot
- The CSV can be imported into notebooks or dashboards for analysis.
- The JSON files feed the Streamlit comparison view (see `frontend/app.py`) to compute deltas between live metrics and the baseline.
- Because the export includes the `experiment_id` and timestamp, you can archive multiple baseline versions by copying the `results` folder with a new suffix. 

## Reloading Into SQLite
If you need to seed a fresh database, convert `baseline_runs.csv` into SQL `INSERT` statements (e.g., with a short Python script) and apply them via:
```bash
sqlite3 redteam_master.db ".read your_seed_file.sql"
```
We do not ship a canned SQL dump so each team can enforce its own retention policies, but the CSV/JSON files contain all of the information required to reconstruct the baseline.
