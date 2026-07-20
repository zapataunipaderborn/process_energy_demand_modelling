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
        'run_name':              'experiment_944',
        'processes_to_run':      ['process_1', 'process_2', 'process_3', 'process_4_1', 'process_4_2', 'process_5'],
        'temporal_resolution':   '1min',
        'run_process_modelling': True,
        'mining_algorithms':     ['heuristic', 'alpha'],#, 'inductive'],#None,   # e.g. ['heuristic', 'inductive'] to test only those
        'run_energy_modelling':  setting,    # ON: needed so the curve pipelines exist for the schedule-profile "Best, mine" comparison
        'run_joint_duration_eval': False, # slow, per-instance-matched heatmaps; superseded by energy_distribution_results
        'run_schedule_profile_eval':setting, # ON: Best/mine vs. Schedule-direct vs. Stochastic generator, per process
        'save_predicted_curves': setting, # ON: persists predicted_curves.parquet the schedule-profile comparison reads
        'train_ratio':           0.70, # fraction of cases used for training (e.g. 0.8 for 80/20)
        'split_type':            'temporal',#'temporal', # 'temporal' (default, no leakage) or 'random' (fixed-seed shuffle)
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
    save_predicted_curves = exp.get('save_predicted_curves', False)
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
          f"save_predicted_curves={save_predicted_curves}  "
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
    env['PIPELINE_SAVE_PREDICTED_CURVES'] = 'true' if save_predicted_curves else 'false'
    if mining_algorithms:
        env['PIPELINE_MINING_ALGORITHMS'] = ','.join(mining_algorithms)
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
