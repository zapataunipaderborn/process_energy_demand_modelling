"""
pipeline.py — Orchestrate multiple modelling.py runs with different configurations.

Each entry in EXPERIMENTS defines one run. Fields:
  data_experiment       str        which gold data folder to load  (e.g. '1' → data/gold/experiment_1)
  run_name              str        label stored in info.json and used as the results sub-folder suffix
  processes_to_run      list       process folders to include
  temporal_resolution   str        'original' | '1min' | '5min' | '15min'
  run_process_modelling bool       whether to run the process modelling loop (default: False)
  mining_algorithms     list|None  which pm4py process-discovery algorithms to test/compare,
                                   e.g. ['heuristic', 'inductive']. Subset of
                                   'alpha' | 'heuristic' | 'inductive' | 'ilp'.
                                   None (default) → use modelling.py's own default list.
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
  train_ratio           float      fraction of cases used for training in the
                                   train/test split (the rest go to test).
                                   e.g. 0.8 for an 80/20 split. Default: 0.70
                                   (modelling.py's own default, used when omitted).
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
                                   APPROACHES list in modelling.py ('baseline'
                                   — "Baseline", ONE median curve per sensor
                                   (pooled over all activities, the naive
                                   floor); 'median_activity_sensor' — "Median
                                   per Activity & Sensor", median curve per
                                   sensor+activity+object, no model;
                                   'ml_dtw' — DBA + DTW +
                                   regression; 'ml_external',
                                   'ml_only', 'seq2seq', 'seq2seq_only',
                                   'seq2seq_external', ...). None
                                   (default) → use modelling.py's own list.
                                   Fewer models = much faster runs. The
                                   schedule-profile / complete-curve eval needs
                                   'ml_external' present.
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
                                   hyperparameter search. None → modelling.py
                                   default (True). Set False for fast test runs.
  curve_n_optuna_trials int|None   Optuna trials per (sensor, activity, object)
                                   when the search is on. None → default (50).
  curve_loss str|None              Training loss for the sklearn curve models:
                                   'absolute_error' (default) fits the conditional
                                   MEDIAN, matching the L1 metrics (sMAE, WAPE)
                                   and the val-MAE model selection.
                                   'squared_error' restores the old conditional-
                                   MEAN behaviour for ablation.
  dtw_shape_blind_decode bool|None decode canonical predictions back to the test
                                   timeline using ONLY the curve's length
                                   (linear resample), instead of DTW-warping
                                   against the curve's own values. Puts the DTW
                                   and non-DTW approaches on the same test-time
                                   information, so DTW is measured purely as a
                                   training-time representation. None → False
                                   (original behaviour).
  save_curve_values     bool|None  persist the full y_true / y_pred array on
                                   every scored curve in curve_eval_results.parquet,
                                   so any metric can be computed offline later
                                   without re-running. None → False. The per-curve
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
        'run_name':              'experiment_971',
        # FULL REPORTABLE RUN: all 6 processes, heuristic + alpha miners, Optuna
        # hyperparameter search ON. Tests the budget over-generation fix
        # (simulation.py BUDGET_EXIT_DISCOUNT=0.02 while under budget,
        # BUDGET_EXIT_BOOST=25.0 once spent, TIME backstop BUDGET_MAX_TIME_RATIO=1.5;
        # the activity-count cap BUDGET_MAX_LENGTH_RATIO is OFF, see
        # BUDGET_USE_LENGTH_CAP) and the DBA zero-calibration
        # (sim_extractor._zero_calibrate_barycenter) end-to-end, and is directly
        # comparable to experiment_944 (same split, same miners) for the
        # before/after leakage check.
        'processes_to_run':      ['process_1', 'process_2', 'process_3', 'process_4_1', 'process_4_2', 'process_5'],
        'temporal_resolution':   '1min',
        'run_process_modelling': True,
        'mining_algorithms':     ['heuristic', 'alpha'],#, 'alpha'],#, 'alpha'],   # matches experiment_944 for apples-to-apples comparison
        'run_energy_modelling':  setting,    # ON: needed so the curve pipelines exist for the schedule-profile "Best, mine" comparison
        'run_joint_duration_eval': False, # slow, per-instance-matched heatmaps; superseded by energy_distribution_results
        'run_schedule_profile_eval':setting, # ON: Best/mine vs. Schedule-direct vs. Stochastic generator, per process
        'run_autoregressive_eval': False, # OFF: skip the "… (autoreg)" curve-eval rows
        'save_predicted_curves': setting, # ON: persists predicted_curves.parquet the schedule-profile comparison reads
        'train_ratio':           0.70, # fraction of cases used for training (e.g. 0.8 for 80/20)
        'split_type':            'temporal',#'temporal', # 'temporal' (default, no leakage) or 'random' (fixed-seed shuffle)
        # All standard curve approaches (the uncommented defaults in
        # modelling.py's APPROACHES list — same set experiment_944 produced),
        # renamed to a consistent 'ml_*' prefix for every sklearn/DTW-regression
        # approach, with each paired against its seq2seq counterpart (see the
        # full reference table below). NOTE: this trains ~10 curve families per
        # (sensor, activity, object) instead of one, so with Optuna ON this is a
        # big runtime multiplier — the seq2seq_* families especially are slow.
        # The schedule-profile / complete-curve eval still keys off
        # 'ml_external'.
        'curve_approaches':      [
            'baseline',              # "Baseline": ONE median curve per SENSOR, pooled over all activities/objects (naive floor)
            'median_activity_sensor', # "Median per Activity & Sensor": median curve per (sensor, activity, object), no model
            'ml_dtw',                # DBA barycenter + DTW alignment + regression (formerly just 'baseline')
            'ml_external',  # DBA + DTW + regression + external factors + prev-activity NAME (no lagged energy)
            'ml_only',                # no DTW at all (formerly 'ml_linear')
            'exemplar',               # real training curve (DTW medoid of a shape cluster),
                                      # picked by a classifier over attributes + ef_*, scaled
                                      # by an L1 level model — the only approach that never
                                      # averages, so the only one whose output keeps realistic
                                      # texture. Expect WORSE sMAE and a far better std ratio.
            # 'seq2seq',
            # 'seq2seq_only',
            'seq2seq_external',
        ],
        # Full reference — every valid approach name (uncomment to enable the
        # experimental ones, which are commented out in modelling.py by
        # default). Kept here so the whole universe is togglable in future.
        # Naming: every sklearn/DTW-regression approach now starts with 'ml_'.
        # One-to-one pairing with the seq2seq family (same conditioning, same
        # DTW/no-DTW alignment choice, different regressor):
        #   ml_dtw                 <-> seq2seq
        #   ml_only                <-> seq2seq_only
        #   ml_external  <-> seq2seq_external
        # The two median baselines sit outside that chain as the naive floors
        # everything else should beat: 'baseline' = "Baseline" (ONE median per
        # sensor, pooled over all activities/objects — the coarsest floor),
        # 'median_activity_sensor' = "Median per Activity & Sensor" (median per
        # sensor+activity+object):
        #   'baseline', 'median_activity_sensor', 'ml_dtw', 'ml_external',
        #   'ml_only', 'exemplar', 'seq2seq', 'seq2seq_only', 'seq2seq_external'
        # 'exemplar' sits outside the ml_*/seq2seq pairing: it is the only
        # approach that predicts a REAL measured curve rather than an average of
        # them, so it is the one to look at when the question is whether the
        # output resembles a real load profile rather than whether it minimises
        # pointwise error.
        # Regressors competed for every curve approach, per (sensor, activity,
        # object); best kept by validation MAE. Comment a line out to turn that
        # model off. None / omitted = all six.
        'curve_models': [
            'Linear Regression',   # unpenalised OLS reference (nothing to tune —
                                   # its Optuna trials are all identical)
            'Ridge',               # penalised counterpart; the one-hot design
                                   # matrix is high-dimensional and collinear
            #'Random Forest',
            'XGBoost',             # the gradient-boosting member
            'Hist Gradient Boosting',  # native-categorical, histogram-binned GB
            #'MLP',                 # feed-forward net (sklearn MLPRegressor)
        ],
        # Cells competed inside every seq2seq approach; lower validation loss
        # wins. ['lstm'] reproduces the pre-transformer behaviour.
        'seq2seq_cells': ['lstm'],
        'curve_optimize_hyperparams': False,  # ON: proper tuned run (slow, publication-grade)
        'curve_n_optuna_trials': 10,         # trials per (sensor, activity, object)
    },

    # ── Exemplar comparison ──────────────────────────────────────────────────
    # Uncomment this dict to run it. Trains ONLY the cheap approaches plus the
    # new 'exemplar' one, so it finishes in a fraction of the full run and is a
    # direct answer to "does the exemplar method produce a curve that looks
    # real, and what does that cost in pointwise error?".
    #
    # What to expect, measured over 90 (sensor, activity, object) leaves and
    # 4,187 test curves on data_experiment 1 with this same 70/30 temporal split:
    #
    #     approach                 sMAE    WAPE %   std ratio (want 1.0)
    #     median_activity_sensor   0.845   7.70     0.110
    #     ml_dtw                   0.848   7.70     0.133
    #     ml_external              0.852   7.71     0.135
    #     ml_only                  0.844   7.62     0.086
    #     exemplar                 1.018   8.92     0.738
    #
    # i.e. the exemplar curve carries ~74% of the real curves' standard
    # deviation against ~9-14% for every averaging approach, at a cost of ~20%
    # on sMAE. That trade is the whole point: sMAE rewards a smooth line through
    # the middle of a spiky signal, so it cannot see the difference between a
    # usable load profile and a flat one. Report the two together.
    #
    # dtw_shape_blind_decode is left at its default (True) — no approach may use
    # the test curve's own values to decode, which is the only setting under
    # which these columns are comparable to each other or reachable in simulation.
    #
    # {
    #     'data_experiment':       '1',
    #     'run_name':              'experiment_970_exemplar',
    #     'processes_to_run':      ['process_1', 'process_2', 'process_3',
    #                               'process_4_1', 'process_4_2', 'process_5'],
    #     'temporal_resolution':   '1min',
    #     'run_process_modelling': False,   # curve comparison only — no mining needed
    #     'run_energy_modelling':  True,
    #     'run_schedule_profile_eval': True,
    #     'save_predicted_curves': True,    # needed to plot the curves and see the texture
    #     'train_ratio':           0.70,
    #     'split_type':            'temporal',
    #     'curve_approaches':      [
    #         'median_activity_sensor',   # the floor everything is measured against
    #         'ml_dtw',                   # the incumbent DTW + regression approach
    #         'ml_external',              # + external factors, the strongest incumbent
    #         'exemplar',                 # the new one
    #     ],
    #     'curve_models': ['Ridge', 'Hist Gradient Boosting'],
    #     'curve_optimize_hyperparams': False,
    #     'save_curve_values': True,   # keep every y_true/y_pred so new metrics
    #                                  # can be computed from the parquet later
    # },

]

# ── Runner ────────────────────────────────────────────────────────────────────
modelling_script = Path(__file__).parent / 'modelling.py'

for i, exp in enumerate(EXPERIMENTS, start=1):
    mining_algorithms = exp.get('mining_algorithms')
    run_energy_modelling = exp.get('run_energy_modelling', True)
    run_joint_duration_eval = exp.get('run_joint_duration_eval', False)
    run_schedule_profile_eval = exp.get('run_schedule_profile_eval', False)
    run_autoregressive_eval = exp.get('run_autoregressive_eval', False)
    save_predicted_curves = exp.get('save_predicted_curves', False)
    curve_approaches = exp.get('curve_approaches')
    curve_models = exp.get('curve_models')
    seq2seq_cells = exp.get('seq2seq_cells')
    curve_optimize_hyperparams = exp.get('curve_optimize_hyperparams')
    curve_n_optuna_trials = exp.get('curve_n_optuna_trials')
    curve_loss = exp.get('curve_loss')
    dtw_shape_blind_decode = exp.get('dtw_shape_blind_decode')
    save_curve_values = exp.get('save_curve_values')
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
          f"dtw_shape_blind_decode={dtw_shape_blind_decode if dtw_shape_blind_decode is not None else '(default True)'}  "
          f"save_curve_values={save_curve_values if save_curve_values is not None else '(default False)'}  "
          f"train_ratio={train_ratio if train_ratio is not None else '(default 0.70)'}  "
          f"split_type={split_type}  "
          f"random_seed={random_seed}")
    print(f"{'='*60}\n")

    env = os.environ.copy()
    # ── Reproducibility ──────────────────────────────────────────────────────
    # One master seed for the child run: modelling.py seeds python/numpy/torch
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
    if dtw_shape_blind_decode is not None:
        env['PIPELINE_DTW_SHAPE_BLIND_DECODE'] = 'true' if dtw_shape_blind_decode else 'false'
    if save_curve_values is not None:
        env['PIPELINE_SAVE_CURVE_VALUES'] = 'true' if save_curve_values else 'false'
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
