"""
01_pipeline.py — Orchestrate multiple 02_modelling.py runs with different configurations.

Each entry in EXPERIMENTS defines one run. Fields:
  data_experiment       str        which gold data folder to load  (e.g. '1' → data/gold/experiment_1)
  run_name              str        label stored in info.json and used as the results sub-folder suffix
  processes_to_run      list       process folders to include
  temporal_resolution   str        'original' | '1min' | '5min' | '15min'
  run_process_modelling bool       whether to run the process modelling loop (default: False)
  mining_algorithms     list|None  which pm4py process-discovery algorithms to test/compare,
                                   e.g. ['heuristic', 'inductive']. Subset of
                                   'alpha' | 'heuristic' | 'inductive' | 'ilp'.
                                   None (default) → use 02_modelling.py's own default list.
  run_energy_modelling  bool       whether to run energy profile/curve modelling
                                   (sensor curve fitting + curve-quality benchmark).
                                   Set False to run process modelling only. Default: True.
  run_joint_duration_eval bool     whether to run the (slow) per-instance-matched
                                   "Joint Duration + Profile Evaluation" heatmaps.
                                   Superseded by the energy-distribution metrics
                                   (energy_distribution_results/), which don't rely
                                   on instance matching. Default: False.
  run_schedule_profile_eval bool   whether to run the "Schedule Profile Evaluation"
                                   (Best/mine vs. Schedule-direct vs. Stochastic
                                   generator, per process) — writes to
                                   schedule_profile_eval_results/. Default: False.
  run_autoregressive_eval bool     whether to add the autoregressive test-time
                                   rollout rows ("… (autoreg)") to the Curve-Only
                                   Evaluation. Only adds those extra comparison
                                   rows; affects nothing else. Default: False.
                                   Inert unless the lagged-energy prev-activity
                                   features are switched on explicitly
                                   (split_curves_with_prev_activity
                                   include_prev_energy=True), which reported runs
                                   do not do.
  save_predicted_curves bool       whether to also persist the actual real/predicted
                                   curve arrays (not just the aggregated Wasserstein
                                   distances) behind complete-curve eval and schedule
                                   profile eval, for recomputing other metrics or
                                   plotting later. Default: False.
  complete_curve_approaches list|None restrict the complete-curve eval to these
                                   approaches only. None (default) → every
                                   trained approach is assembled into complete
                                   case profiles and scored, per simulation
                                   mode. e.g. ['ml_step_dtw'] evaluates just
                                   that one — the complete-curve stage is
                                   (process × mode × approach) inference over
                                   every simulated case, so each dropped
                                   approach cuts that stage proportionally.
                                   The schedule-profile "Best, mine" comparator
                                   follows: it uses 'ml_external' when listed,
                                   otherwise the first listed approach.
  train_ratio           float      fraction of cases used for training in the
                                   train/test split (the rest go to test).
                                   e.g. 0.8 for an 80/20 split. Default: 0.70
                                   (02_modelling.py's own default, used when omitted).
  random_seed           int        master seed for the whole run — python, numpy,
                                   torch, the train/test split and every
                                   per-combo training seed derive from it, and
                                   PYTHONHASHSEED is pinned to it in the child
                                   process. Default: 42.
  split_type            str        how cases are assigned to train vs. test:
                                   'temporal' (default) — earliest train_ratio
                                   fraction of cases by start time → train, the
                                   rest (the "future") → test, no leakage.
                                   'random' — fixed-seed shuffle of case IDs at
                                   the same ratio, no time ordering. Same
                                   case-id partition is applied to every
                                   evaluation (process, energy/profile,
                                   schedule-profile) either way.
  curve_approaches      list|None  which curve-fitting models to train, e.g.
                                   ['ml_external']. Subset of the
                                   APPROACHES list in 02_modelling.py ('baseline'
                                   — "Baseline", ONE median curve per sensor
                                   (pooled over all activities, the naive
                                   floor); 'median_activity_sensor' — "Median
                                   per Activity & Sensor", median curve per
                                   sensor+activity+object, no model;
                                   'ml_dtw' — DBA + DTW +
                                   regression; 'ml_external',
                                   'ml_only', 'seq2seq', 'seq2seq_only',
                                   'seq2seq_external', 'seq2seq_iom' (the
                                   competing paper's own selection rule:
                                   generate + score against a softDTW reference
                                   every 10 epochs); 'ml_step_dtw' and
                                   'ml_step_dtw_smooth' (the method); plus the
                                   row-weighted train/eval-gap variants of
                                   ml_external: 'ml_external_wcounts',
                                   'ml_external_wmetric'. None
                                   (default) → use 02_modelling.py's own list.
                                   Fewer models = much faster runs. The
                                   schedule-profile "Best, mine" comparator uses
                                   'ml_external' when it is in
                                   complete_curve_approaches, otherwise the
                                   first listed approach — 'ml_external' is NOT
                                   required.
  curve_models          list|None  which regressors compete for every curve
                                   approach, per (sensor, activity, object) —
                                   the best is kept by validation MAE. Subset of
                                   'Linear Regression' | 'Ridge' |
                                   'Random Forest' | 'XGBoost' |
                                   'Hist Gradient Boosting' | 'MLP' (short
                                   aliases also accepted: linear, ridge, rf,
                                   xgb, hgb, mlp). None (default) → all six
                                   compete. Fewer = faster: the curve stage
                                   trains one model per name per combo, times
                                   curve_n_optuna_trials.
  seq2seq_cells         list|None  which cells compete inside every seq2seq
                                   approach: 'lstm' and/or 'transformer'. Both
                                   are trained per combo and the lower
                                   validation loss wins. None (default) → both.
                                   Set ['lstm'] to reproduce a pre-transformer
                                   run or to halve the seq2seq cost.
  curve_optimize_hyperparams bool  whether curve models run the (slow) Optuna
                                   hyperparameter search. None → 02_modelling.py
                                   default (True). Set False for fast test runs.
  curve_n_optuna_trials int|None   Optuna trials per (sensor, activity, object)
                                   when the search is on. None → default (50).
  curve_loss str|None              Training loss for the sklearn curve models:
                                   'absolute_error' (default) fits the conditional
                                   MEDIAN, matching the L1 metrics (sMAE, WAPE)
                                   and the val-MAE model selection.
                                   'squared_error' restores the old conditional-
                                   MEAN behaviour for ablation.
  curve_select_by_realism bool|None which candidate regressor wins each
                                   (sensor, activity, object) leaf. None/False =
                                   validation MAE on the CANONICAL rows (the
                                   historical behaviour). True = the realism
                                   'Overall' the results notebooks report, scored
                                   by running each candidate through its REAL
                                   predictor (decode included) on the held-out
                                   curves and comparing per-curve features
                                   against the REAL curve values. Fixes the
                                   space/unit/metric mismatch between what is
                                   selected on and what is reported.
  curve_selection_metrics list|None which per-curve features are averaged into
                                   that 'Overall'. Subset of 'sum' | 'max' |
                                   'mean' | 'std' | 'roughness' | 'acf1'.
                                   None -> ['sum','max','mean','std']. Changing
                                   this changes the selection objective.
                                   Only used when curve_select_by_realism.
  dtw_shape_blind_decode bool|None decode canonical predictions back to the test
                                   timeline using ONLY the curve's length
                                   (linear resample), instead of DTW-warping
                                   against the curve's own values. Puts the DTW
                                   and non-DTW approaches on the same test-time
                                   information, so DTW is measured purely as a
                                   training-time representation. None → True
                                   (blind decode is the default; set False only
                                   to reproduce the pre-970 shape-aware decode,
                                   which leaks the test curve's values).
  curve_median_floor    bool|None  fall back to the "Median per Activity &
                                   Sensor" curve whenever a trained approach
                                   fails to beat that leaf's own median on
                                   held-out curves. None -> False (OFF).
                                   Acceptance is on POINTWISE error, so turning
                                   it on deletes any method that trades
                                   pointwise accuracy for curve realism: it fired
                                   on 33/40 leaves for 'exemplar', replacing its
                                   real measured curve with the smooth median.
                                   Leave it off to see what each approach really
                                   predicts; turn it on only for a
                                   "never worse than the naive floor" run, and
                                   not alongside exemplar*/realism reporting.
  curve_median_floor_ratio float|None
                                   margin the model must win by: kept only when
                                   its held-out MAE < ratio x the floor's.
                                   None -> 1.0 (strictly better). 0.95 would
                                   demand a 5% improvement.
  step_dtw_fallback_ratio  float|str|None
                                   the ml_step_dtw* do-no-harm gate: a leaf is
                                   routed to its ml_external pipeline iff the
                                   step model's held-out MAE > ratio x
                                   ml_external's. None -> 1.25 (sim_extractor
                                   default). 'inf' DISABLES the gate — the
                                   step/smooth model is kept on every leaf,
                                   even where it clearly loses. Note the gate
                                   is silently inert anyway when ml_external
                                   is not in curve_approaches (nothing to
                                   route to) — that, not retraining noise, is
                                   why the smooth rows differed between
                                   experiment_999 and experiment_1000.
  save_curve_values     bool|None  persist the full predicted / real curve on
                                   every scored curve in curve_eval_results.parquet,
                                   so any metric can be computed offline later
                                   without re-running. None → True (ON). The per-curve
                                   SHAPE STATISTICS (std / lag-1 acf / zero
                                   fraction / peak, for both the real and the
                                   predicted curve) are written either way — those
                                   are what the realism measures need, and they
                                   cost ~10 floats a row. Turn this on only when
                                   you want arbitrary future metrics: it inflates
                                   the file by roughly the curve length per row.
"""

import os
import subprocess
import sys
from pathlib import Path

setting = True

# ── Experiment definitions ────────────────────────────────────────────────────
EXPERIMENTS = [

    {
        'data_experiment':       '1',
        'run_name':              'experiment_1',
        'processes_to_run':      ['process_1', 'process_2', 'process_3',
                                  'process_4_1', 'process_4_2', 'process_5'],
        'temporal_resolution':   '1min',
        'run_process_modelling': True,
        'mining_algorithms':     ['heuristic', 'alpha', 'inductive'],
        'run_energy_modelling':  True,
        'run_joint_duration_eval':   False,
        'run_schedule_profile_eval': True,
        'run_autoregressive_eval':   False,
        'save_predicted_curves': True,
        'train_ratio':           0.70,
        'split_type':            'temporal',
        'random_seed':           42,     
                                         
        'curve_approaches': [
            'baseline',
            'median_activity_sensor',
            'ml_external',
            'ml_only',
            'ml_step_dtw_smooth',
            'seq2seq_external',
            'seq2seq_iom',
        ],
        'curve_optimize_hyperparams': True,
        'curve_n_optuna_trials': 10,
        'complete_curve_approaches': ['ml_step_dtw_smooth', 'baseline'],
        'curve_median_floor':    False,
        'save_curve_values':     True,
        'step_dtw_fallback_ratio': 'inf',  
    },

]

# ── Runner ────────────────────────────────────────────────────────────────────
modelling_script = Path(__file__).parent / '02_modelling.py'

for i, exp in enumerate(EXPERIMENTS, start=1):
    mining_algorithms = exp.get('mining_algorithms')
    run_energy_modelling = exp.get('run_energy_modelling', True)
    run_joint_duration_eval = exp.get('run_joint_duration_eval', False)
    run_schedule_profile_eval = exp.get('run_schedule_profile_eval', False)
    run_autoregressive_eval = exp.get('run_autoregressive_eval', False)
    save_predicted_curves = exp.get('save_predicted_curves', False)
    curve_approaches = exp.get('curve_approaches')
    complete_curve_approaches = exp.get('complete_curve_approaches')
    curve_models = exp.get('curve_models')
    seq2seq_cells = exp.get('seq2seq_cells')
    curve_optimize_hyperparams = exp.get('curve_optimize_hyperparams')
    curve_n_optuna_trials = exp.get('curve_n_optuna_trials')
    curve_loss = exp.get('curve_loss')
    curve_select_by_realism = exp.get('curve_select_by_realism')
    curve_selection_metrics = exp.get('curve_selection_metrics')
    dtw_shape_blind_decode = exp.get('dtw_shape_blind_decode')
    save_curve_values = exp.get('save_curve_values')
    curve_median_floor = exp.get('curve_median_floor')
    curve_median_floor_ratio = exp.get('curve_median_floor_ratio')
    step_dtw_fallback_ratio = exp.get('step_dtw_fallback_ratio')
    train_ratio = exp.get('train_ratio')
    split_type = exp.get('split_type', 'temporal')
    random_seed = int(exp.get('random_seed', 42))
    if split_type not in ('temporal', 'random'):
        print(f"[pipeline] ERROR: {exp['run_name']} has split_type={split_type!r}, "
              f"must be 'temporal' or 'random'. Stopping.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Running experiment {i}/{len(EXPERIMENTS)}: {exp['run_name']}")
    print(f"  data_experiment={exp['data_experiment']}  "
          f"temporal_resolution={exp['temporal_resolution']}  "
          f"processes={exp['processes_to_run']}  "
          f"run_process_modelling={exp.get('run_process_modelling', False)}  "
          f"mining_algorithms={mining_algorithms or '(default)'}  "
          f"run_energy_modelling={run_energy_modelling}  "
          f"run_joint_duration_eval={run_joint_duration_eval}  "
          f"run_schedule_profile_eval={run_schedule_profile_eval}  "
          f"run_autoregressive_eval={run_autoregressive_eval}  "
          f"save_predicted_curves={save_predicted_curves}  "
          f"curve_approaches={curve_approaches or '(default)'}  "
          f"curve_models={curve_models or '(all)'}  "
          f"seq2seq_cells={seq2seq_cells or '(all)'}  "
          f"curve_optimize_hyperparams={curve_optimize_hyperparams if curve_optimize_hyperparams is not None else '(default)'}  "
          f"curve_loss={curve_loss or '(default absolute_error)'}  "
          f"curve_select_by_realism={curve_select_by_realism if curve_select_by_realism is not None else '(default False)'}  "
          f"curve_selection_metrics={curve_selection_metrics or '(default sum,max,mean,std)'}  "
          f"dtw_shape_blind_decode={dtw_shape_blind_decode if dtw_shape_blind_decode is not None else '(default True)'}  "
          f"save_curve_values={save_curve_values if save_curve_values is not None else '(default True)'}  "
          f"curve_median_floor={curve_median_floor if curve_median_floor is not None else '(default False = OFF)'}  "
          f"curve_median_floor_ratio={curve_median_floor_ratio if curve_median_floor_ratio is not None else '(default 1.0)'}  "
          f"train_ratio={train_ratio if train_ratio is not None else '(default 0.70)'}  "
          f"split_type={split_type}  "
          f"random_seed={random_seed}")
    print(f"{'='*60}\n")

    env = os.environ.copy()
    # ── CPU-only, fork-safe child ────────────────────────────────────────────
    # Hide the GPUs from the modelling child BEFORE torch is ever imported.
    # Everything in the run is CPU-only by design, but a parent that has merely
    # probed CUDA (set_global_seeds' torch.cuda.is_available at startup) forks
    # poisoned children: experiment_986 lost 7 hours with all 16 seq2seq
    # workers and the parent stuck in futex_wait at 4s CPU each — the
    # in-worker guards (_train_seq2seq_worker) run only AFTER the fork, which
    # is too late for locks inherited FROM the fork. With no visible GPU the
    # parent never initialises CUDA and there is nothing to inherit.
    env['CUDA_VISIBLE_DEVICES']           = ''
    env['PYTORCH_NVML_BASED_CUDA_CHECK']  = '1'
    # ── Reproducibility ──────────────────────────────────────────────────────
    # One master seed for the child run: 02_modelling.py seeds python/numpy/torch
    # from it and derives every per-combo training seed from it.
    env['PIPELINE_RANDOM_SEED']       = str(random_seed)
    env['PIPELINE_RANDOM_SPLIT_SEED'] = str(random_seed)
    # PYTHONHASHSEED only takes effect at interpreter start, so it has to be set
    # HERE, on the child's environment — str hashing is randomised per process
    # otherwise, and any set-of-strings iterated without sorting would order
    # differently between two runs that are identical in every other respect.
    env['PYTHONHASHSEED']             = str(random_seed)
    env['PIPELINE_DATA_EXPERIMENT']       = exp['data_experiment']
    env['PIPELINE_RUN_NAME']              = exp['run_name']
    env['PIPELINE_PROCESSES_TO_RUN']      = ','.join(exp['processes_to_run'])
    env['PIPELINE_TEMPORAL_RESOLUTION']   = exp['temporal_resolution']
    env['PIPELINE_RUN_PROCESS_MODELLING'] = 'true' if exp.get('run_process_modelling', False) else 'false'
    env['PIPELINE_RUN_ENERGY_MODELLING']  = 'true' if run_energy_modelling else 'false'
    env['PIPELINE_RUN_JOINT_DURATION_EVAL'] = 'true' if run_joint_duration_eval else 'false'
    env['PIPELINE_RUN_SCHEDULE_PROFILE_EVAL'] = 'true' if run_schedule_profile_eval else 'false'
    env['PIPELINE_RUN_AUTOREGRESSIVE_EVAL'] = 'true' if run_autoregressive_eval else 'false'
    env['PIPELINE_SAVE_PREDICTED_CURVES'] = 'true' if save_predicted_curves else 'false'
    if mining_algorithms:
        env['PIPELINE_MINING_ALGORITHMS'] = ','.join(mining_algorithms)
    if curve_approaches:
        env['PIPELINE_CURVE_APPROACHES'] = ','.join(curve_approaches)
    if complete_curve_approaches:
        env['PIPELINE_COMPLETE_CURVE_APPROACHES'] = ','.join(complete_curve_approaches)
    if curve_models:
        env['PIPELINE_CURVE_MODELS'] = ','.join(curve_models)
    if seq2seq_cells:
        env['PIPELINE_SEQ2SEQ_CELLS'] = ','.join(seq2seq_cells)
    if curve_optimize_hyperparams is not None:
        env['PIPELINE_CURVE_OPTIMIZE_HYPERPARAMS'] = 'true' if curve_optimize_hyperparams else 'false'
    if curve_n_optuna_trials is not None:
        env['PIPELINE_CURVE_N_OPTUNA_TRIALS'] = str(curve_n_optuna_trials)
    if curve_loss is not None:
        env['PIPELINE_CURVE_LOSS'] = str(curve_loss)
    if curve_select_by_realism is not None:
        env['PIPELINE_CURVE_SELECT_BY_REALISM'] = 'true' if curve_select_by_realism else 'false'
    if curve_selection_metrics:
        env['PIPELINE_CURVE_SELECTION_METRICS'] = ','.join(curve_selection_metrics)
    if dtw_shape_blind_decode is not None:
        env['PIPELINE_DTW_SHAPE_BLIND_DECODE'] = 'true' if dtw_shape_blind_decode else 'false'
    if save_curve_values is not None:
        env['PIPELINE_SAVE_CURVE_VALUES'] = 'true' if save_curve_values else 'false'
    if curve_median_floor is not None:
        env['PIPELINE_CURVE_MEDIAN_FLOOR'] = 'true' if curve_median_floor else 'false'
    if curve_median_floor_ratio is not None:
        env['PIPELINE_CURVE_MEDIAN_FLOOR_RATIO'] = str(curve_median_floor_ratio)
    if step_dtw_fallback_ratio is not None:
        # 'inf' disables the step-DTW do-no-harm gate: no leaf is ever routed
        # to ml_external, the step/smooth model is kept everywhere.
        env['PIPELINE_STEP_DTW_FALLBACK_RATIO'] = str(step_dtw_fallback_ratio)
    if train_ratio is not None:
        env['PIPELINE_TRAIN_RATIO'] = str(train_ratio)
    env['PIPELINE_SPLIT_TYPE'] = split_type

    result = subprocess.run(
        [sys.executable, str(modelling_script)],
        env=env,
    )

    if result.returncode != 0:
        print(f"\n[pipeline] ERROR: {exp['run_name']} exited with code {result.returncode}. Stopping.")
        sys.exit(result.returncode)

    print(f"\n[pipeline] Finished: {exp['run_name']}")

print(f"\n{'='*60}")
print(f"  All {len(EXPERIMENTS)} experiment(s) completed.")
print(f"{'='*60}\n")
