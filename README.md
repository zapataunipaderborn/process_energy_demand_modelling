# Process-Aware Energy Demand Modelling

This repository contains the full modelling pipeline for predicting industrial energy load profiles from process event logs: process models are mined from event logs (pm4py), process behaviour is simulated, and per-activity energy curves are predicted by a set of competing approaches — naive medians, DTW/DBA-based regression (`ml_*`), the segment-based step-DTW method (`ml_step_dtw_smooth`), and seq2seq neural baselines — which are then evaluated on held-out data and assembled into complete case-level and schedule-level load profiles.

## Repository layout

| Path | What it is |
|---|---|
| `src/00_pipeline_preprocessing.py` | Raw → gold data preparation (column selection, event-log construction, external factors). Only needed when rebuilding the gold datasets from raw data; not needed for replication. |
| `src/01_pipeline.py` | Experiment orchestrator. Each entry in its `EXPERIMENTS` list defines one run (which processes, which curve approaches, split, seed, evaluation stages) and is passed to `02_modelling.py` via `PIPELINE_*` environment variables. The docstring at the top documents every field. |
| `src/02_modelling.py` | The modelling pipeline itself: train/test split, process mining, simulation, curve-model training, curve-only evaluation, complete-curve evaluation, schedule-profile evaluation, result export. |
| `src/utils/` | Library code: `sim_extractor.py` (feature/curve extraction, all curve-model builders and predictors, evaluation), `simulation.py` (discrete-event process simulation), `sim_modeller.py` (duration/transition ML models). |
| `src/simulation_process/` | The synthetic process_1 generator (`generate_process_1.py` + physical autoclave/distillation models). |
| `src/03…08_*.ipynb` | Results and figure notebooks; they read a finished run from `results/`. |
| `data/gold/experiment_1/<process>/datasets/` | Model-ready inputs per process: `df_event_log.parquet`, `df_expanded.parquet`, `df_production_plan.parquet`. |
| `results/<run_name>_<timestamp>/` | One folder per pipeline run: metrics tables (parquet/csv), evaluation results per stage, mined Petri nets, run log, `info.json`. |
| `requirements.txt` | Pinned Python dependencies (Python 3.10). |

## Data availability

- **process_1 — fully replicable.** It is a synthetic sterilization process produced by a physical simulation (`src/simulation_process/generate_process_1.py`): autoclave heat/hold/cool power curves come from a thermodynamic model, batch volume drives durations and energy levels, and deterministic weather series provide the exogenous features. Anyone can regenerate its complete dataset and reproduce every process_1 number end to end.
- **process_2 … process_5 — confidential.** These are real industrial processes; the underlying measurements and event logs are proprietary and are **not** included. What this repository publishes for them is anonymized evaluation output only: process labels are anonymized (`process_2` … `process_5`), and the reported artifacts are aggregate error metrics (sMAE, WAPE, Wasserstein distances, process-model quality scores). No raw sensor data, timestamps, or company-identifying information is part of the published pipeline.

## Setup

```bash
# Python 3.10
python -m venv .venv && source .venv/bin/activate   # or: conda create -n pedm python=3.10
pip install -r requirements.txt
```

## Replicating the experiment on process_1

1. **Generate the synthetic data** (writes `data/gold/experiment_1/process_1/datasets/`):

   ```bash
   python src/simulation_process/generate_process_1.py
   ```

2. **Configure the run.** In `src/01_pipeline.py`, use (or adapt) the active experiment entry and restrict it to the synthetic process:

   ```python
   'processes_to_run': ['process_1'],
   ```

   The other fields control scope and cost: `curve_approaches` selects which prediction approaches are trained (e.g. `['baseline', 'median_activity_sensor', 'ml_external', 'ml_only', 'ml_step_dtw_smooth', 'seq2seq_external']`), `curve_optimize_hyperparams` / `curve_n_optuna_trials` control the Optuna search, `split_type: 'temporal'` with `train_ratio: 0.70` and `random_seed: 42` reproduce the published split exactly.

3. **Run the pipeline:**

   ```bash
   python src/01_pipeline.py
   ```

   Everything is CPU-only by design (GPUs are hidden from the child process). A full tuned run on process_1 takes on the order of hours; a throttled sanity run (no Optuna, one sensor via `PIPELINE_CURVE_MAX_SENSORS=1`) finishes in minutes.

4. **Inspect the results.** Outputs land in `results/<run_name>_<timestamp>/`:
   - `summary_train_test.parquet`, `curve_eval_results.parquet` — per-curve and aggregated curve-prediction metrics per approach,
   - `process_results/` — process-model (Petri net) quality per mining algorithm and simulation mode,
   - `complete_curve_eval_results/` — complete case-profile evaluation per (process × simulation mode × approach),
   - `schedule_profile_eval_results/` — schedule-level load-profile comparison (best model vs. schedule-direct vs. stochastic generators),
   - `runtime_profile.parquet`/`.csv`, `runtime_summary.csv`, `runtime_by_method.csv` — wall-clock cost of every stage (mining, duration models, each simulation, each evaluation) and the training cost of each curve method,
   - `pipeline_execution.log`, `info.json` — full run log and configuration record.

   The notebooks `src/03…08` reproduce the paper's tables and figures from such a run folder (set the experiment folder name at the top of each notebook).

5. *(Optional)* Reproducibility is seeded end to end: the same `random_seed` yields the same train/test partition, the same per-combo model seeds, and a pinned `PYTHONHASHSEED` in the child process.

## Replicating on your own data

To apply the pipeline to a new process, provide the three parquet files under `data/gold/experiment_<n>/<process_name>/datasets/`:

- `df_event_log.parquet` — one row per activity instance: `case_id`, `activity`, `object`, `timestamp_start`, `timestamp_end`, object/case attributes,
- `df_expanded.parquet` — the minute-resolution sensor time series joined with the event log (one column per sensor, suffixed `_to_model` for modelling targets) plus `ef_*` exogenous columns,
- `df_production_plan.parquet` — one row per planned case (schedule attributes only, no measurements),

then add the process name to `processes_to_run` in `src/01_pipeline.py`. `src/00_pipeline_preprocessing.py` shows how these frames were constructed for the reported datasets.

## License

See [LICENSE](LICENSE).
