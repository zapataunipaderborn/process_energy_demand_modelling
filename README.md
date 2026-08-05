# Process and Energy Digital Twin — modelling pipeline

This repository contains the code for the paper *"Process and Energy Digital Twin: modeling and simulating industrial processes and their dynamic energy profiles with Process Mining and Machine Learning"*. The pipeline extracts a process model from event logs with Process Mining (pm4py) and per-activity energy profile models with Machine Learning, couples the two for simulation, and evaluates how faithfully the simulated process and energy profiles reproduce the real system. The energy profile approaches compared in the paper — the proposed segment-based Step DTW method, its ablations, sequence-to-sequence baselines, and median baselines — are all implemented here.

## Repository layout

| Path | What it is |
|---|---|
| `src/01_pipeline.py` | Experiment orchestrator. Each entry in its `EXPERIMENTS` list defines one run (which processes, which curve approaches, split, seed, evaluation stages) and is passed to `02_modelling.py` via `PIPELINE_*` environment variables. The docstring at the top documents every field. |
| `src/02_modelling.py` | The modelling pipeline itself: train/test split, process mining, simulation, curve-model training, curve-only evaluation, complete-curve evaluation, schedule-profile evaluation, result export. |
| `src/utils/` | Library code: `sim_extractor.py` (feature/curve extraction, all curve-model builders and predictors, evaluation), `simulation.py` (discrete-event process simulation). |
| `src/simulation_process/` | The synthetic process_1 generator (`generate_process_1.py` + physical autoclave/distillation models). |
| `src/03_results.ipynb` | Results notebook: builds the paper's evaluation tables (process models, individual profiles, complete profiles). Ships with executed outputs, so the paper's results can be inspected directly. |
| `src/04_visuals.ipynb` | Figure notebook: builds the paper's figures. |
| `src/05_computational_time_analysis.ipynb` | Computational-cost notebook: builds the runtime tables of the appendix. |
| `data/gold/experiment_<n>/<process>/datasets/` | Model-ready inputs per process: `df_event_log.parquet`, `df_expanded.parquet`, `df_production_plan.parquet`. Only the synthetic process_1 data can be regenerated here (see below). |
| `text/` | Manuscript sources. |
| `requirements.txt` | Pinned Python dependencies (Python 3.10). |

There is no shipped results folder: each pipeline run creates its own `results/<run_name>_<timestamp>/` directory locally (metrics tables, evaluation results per stage, mined Petri nets, run log, `info.json`), and the notebooks read the latest run of their experiment from there. The evaluation results of the paper's full run are visible in the executed notebook outputs.

## Data availability

- **process_1 — fully replicable.** It is a synthetic sterilization process produced by a physical simulation (`src/simulation_process/generate_process_1.py`): autoclave heat/hold/cool power curves come from a thermodynamic model, batch volume drives durations and energy levels, and deterministic weather series provide the exogenous features. Anyone can regenerate its complete dataset and reproduce every process_1 number end to end.
- **process_2 … process_5 — confidential.** These are real industrial processes; the underlying measurements and event logs are proprietary and are **not** included. What this repository publishes for them is anonymized evaluation output only: process labels are anonymized (`process_2` … `process_5`), and the reported artifacts are aggregate error metrics (WAPE, per-curve realism errors on Sum/Max/Mean/Std/Roughness, process-model quality scores). No raw sensor data, timestamps, or company-identifying information is part of the published pipeline.

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

4. **Inspect the results.** The run writes its outputs to `results/<run_name>_<timestamp>/`. The notebooks `src/03_results.ipynb`, `src/04_visuals.ipynb`, and `src/05_computational_time_analysis.ipynb` then rebuild the paper's tables and figures from that folder (set the experiment name at the top of each notebook).

5. *(Optional)* Reproducibility is seeded end to end: the same `random_seed` yields the same train/test partition, the same per-combo model seeds, and a pinned `PYTHONHASHSEED` in the child process.

## Replicating on your own data

To apply the pipeline to a new process, provide the three parquet files under `data/gold/experiment_<n>/<process_name>/datasets/`:

- `df_event_log.parquet` — one row per activity instance: `case_id`, `activity`, `object`, `timestamp_start`, `timestamp_end`, object/case attributes,
- `df_expanded.parquet` — the minute-resolution sensor time series joined with the event log (one column per sensor, suffixed `_to_model` for modelling targets) plus `ef_*` exogenous columns,
- `df_production_plan.parquet` — one row per planned case (schedule attributes only, no measurements),

then add the process name to `processes_to_run` in `src/01_pipeline.py`.

## License

See [LICENSE](LICENSE).
