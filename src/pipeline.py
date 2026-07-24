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
  save_predicted_curves bool       whether to also persist the actual real/predicted
                                   curve arrays (not just the aggregated Wasserstein
                                   distances) behind complete-curve eval and schedule
                                   profile eval, for recomputing other metrics or
                                   plotting later. Default: False.
  train_ratio           float      fraction of cases used for training in the
                                   train/test split (the rest go to test).
                                   e.g. 0.8 for an 80/20 split. Default: 0.70
                                   (modelling.py's own default, used when omitted).
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
  curve_optimize_hyperparams bool  whether curve models run the (slow) Optuna
                                   hyperparameter search. None → modelling.py
                                   default (True). Set False for fast test runs.
  curve_n_optuna_trials int|None   Optuna trials per (sensor, activity, object)
                                   when the search is on. None → default (50).
"""

import os
import subprocess
import sys
from pathlib import Path

setting = True

# ── Experiment definitions ────────────────────────────────────────────────────
EXPERIMENTS = [
    # {
    #     'data_experiment':       '1',
    #     'run_name':              'experiment_512',
    #     'processes_to_run':      ['process_1', 'process_2', 'process_3', 'process_4', 'process_5'],
    #     'temporal_resolution':   '15min',
    #     'run_process_modelling': True,
    #     'mining_algorithms':     ['heuristic'],#None,   # e.g. ['heuristic', 'inductive'] to test only those
    #     'run_energy_modelling':  False,   # set False to skip energy profile/curve modelling
    # },
    {
        'data_experiment':       '1',
        'run_name':              'experiment_964',
        # FULL REPORTABLE RUN: all 6 processes, heuristic + alpha miners, Optuna
        # hyperparameter search ON. Tests the budget over-generation fix
        # (simulation.py BUDGET_EXIT_DISCOUNT=0.35 + BUDGET_MAX_LENGTH_RATIO) and
        # the DBA zero-calibration (sim_extractor._zero_calibrate_barycenter)
        # end-to-end, and is directly comparable to experiment_944 (same split,
        # same miners) for the before/after leakage check.
        'processes_to_run':      ['process_1', 'process_2', 'process_3', 'process_4_1', 'process_4_2', 'process_5'],
        'temporal_resolution':   '1min',
        'run_process_modelling': True,
        'mining_algorithms':     ['heuristic', 'alpha'],#, 'alpha'],   # matches experiment_944 for apples-to-apples comparison
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
            'ml_external',  # DBA + DTW + regression + external factors + prev-activity
            'ml_only',                # no DTW at all (formerly 'ml_linear')
            'seq2seq',
            'seq2seq_only',
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
        #   'baseline', 'median_activity_sensor', 'ml_dtw', 'instance_stats', 'istats_leakfree',
        #   'dtw_phase', 'basis', 'ml_external', 'amplitude_shape',
        #   'ml_only', 'seq2seq', 'seq2seq_only', 'seq2seq_external'
        'curve_optimize_hyperparams': True,  # ON: proper tuned run (slow, publication-grade)
        'curve_n_optuna_trials': 10,         # trials per (sensor, activity, object)
    },



    #{
    #    'data_experiment':       '1',
    #    'run_name':              'experiment_936',
    #    'processes_to_run':      ['process_1', 'process_2', 'process_3', 'process_4_1', 'process_4_2', 'process_5'],
    #    'temporal_resolution':   '1min',
    #    'run_process_modelling': True,
    #    'mining_algorithms':     ['heuristic', 'alpha'],#, 'inductive'],#None,   # e.g. ['heuristic', 'inductive'] to test only those
    #    'run_energy_modelling':  True,    # set False to skip energy profile/curve modelling
    #    'run_joint_duration_eval': False, # slow, per-instance-matched heatmaps; superseded by energy_distribution_results
    #    'run_schedule_profile_eval':True, # Best/mine vs. Schedule-direct vs. Stochastic generator, per process
    #    'save_predicted_curves': True, # persist real/predicted curve arrays for later metrics/plots
    #    'train_ratio':           0.70, # fraction of cases used for training (e.g. 0.8 for 80/20)
    #    'split_type':            'temporal',#'temporal', # 'temporal' (default, no leakage) or 'random' (fixed-seed shuffle)
    #},


    # {
    #     'data_experiment':       '1',
    #     'run_name':              'experiment_701',
    #     'processes_to_run':      ['process_1', 'process_2', 'process_3', 'process_4', 'process_5'],
    #     'temporal_resolution':   '5min',ok
    #     'run_process_modelling': True,
    #     'mining_algorithms':     ['heuristic'],#None,   # e.g. ['heuristic', 'inductive'] to test only those
    #     'run_energy_modelling':  True,    # set False to skip energy profile/curve modelling
    #     'run_joint_duration_eval': False, # slow, per-instance-matched heatmaps; superseded by energy_distribution_results
    # },
    
    # {
    #     'data_experiment':       '1',
    #     'run_name':              'experiment_401',
    #     'processes_to_run':      ['process_1', 'process_2', 'process_3', 'process_4'],
    #     'temporal_resolution':   '1min',
    #     'run_process_modelling': True,
    # },
    # {
    #     'data_experiment':       '1',
    #     'run_name':              'experiment_301',
    #     'processes_to_run':      ['process_1', 'process_2', 'process_3', 'process_4'],
    #     'temporal_resolution':   '1min',
    #     'run_process_modelling': True,
    # },
    # {
    #     'data_experiment':       '1',
    #     'run_name':              'experiment_300',
    #     'processes_to_run':      ['process_1', 'process_2', 'process_3', 'process_4'],
    #     'temporal_resolution':   'original',
    #     'run_process_modelling': True,
    # },
    # {
    #     'data_experiment':      '1',
    #     'run_name':             'experiment_89',
    #     'processes_to_run':     ['process_3', 'process_2', 'process_3', 'process_4'],
    #     'temporal_resolution':  '15min',
    #     'run_process_modelling': True,
    # },

    # {
    #     'data_experiment':      '1',
    #     'run_name':             'experiment_81',
    #     'processes_to_run':     ['process_2', 'process_3', 'process_4'],
    #     'temporal_resolution':  '1min',
    #     'run_process_modelling': True,
    # },
    # {
    #     'data_experiment':      '1',
    #     'run_name':             'experiment_66',
    #     'processes_to_run':     ['process_1', 'process_2', 'process_3', 'process_4'],
    #     'temporal_resolution':  '1min',
    #     'run_process_modelling': True,
    # },
    # {
    #     'data_experiment':      '1',
    #     'run_name':             'experiment_24',
    #     'processes_to_run':     ['process_2', 'process_3', 'process_4'],
    #     'temporal_resolution':  'original',
    #     'run_process_modelling': True,
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
    curve_optimize_hyperparams = exp.get('curve_optimize_hyperparams')
    curve_n_optuna_trials = exp.get('curve_n_optuna_trials')
    train_ratio = exp.get('train_ratio')
    split_type = exp.get('split_type', 'temporal')
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
          f"curve_optimize_hyperparams={curve_optimize_hyperparams if curve_optimize_hyperparams is not None else '(default)'}  "
          f"train_ratio={train_ratio if train_ratio is not None else '(default 0.70)'}  "
          f"split_type={split_type}")
    print(f"{'='*60}\n")

    env = os.environ.copy()
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
    if curve_optimize_hyperparams is not None:
        env['PIPELINE_CURVE_OPTIMIZE_HYPERPARAMS'] = 'true' if curve_optimize_hyperparams else 'false'
    if curve_n_optuna_trials is not None:
        env['PIPELINE_CURVE_N_OPTUNA_TRIALS'] = str(curve_n_optuna_trials)
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
