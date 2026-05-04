"""
pipeline.py — Orchestrate multiple modelling.py runs with different configurations.

Each entry in EXPERIMENTS defines one run. Fields:
  data_experiment      str   which gold data folder to load  (e.g. '1' → data/gold/experiment_1)
  run_name             str   label stored in info.json and used as the results sub-folder suffix
  processes_to_run     list  process folders to include
  temporal_resolution  str   'original' | '1min' | '5min' | '15min'
  run_process_modelling bool  whether to run the process modelling loop (default: False)
"""

import os
import subprocess
import sys
from pathlib import Path

# ── Experiment definitions ────────────────────────────────────────────────────
EXPERIMENTS = [
    {
    'data_experiment':      '1',
        'run_name':             'experiment_13',
        'processes_to_run':     ['process_2', 'process_3', 'process_4'],
        'temporal_resolution':  '15min',
        'run_process_modelling': True,
    },

    # {
    #     'data_experiment':      '1',
    #     'run_name':             'experiment_11',
    #     'processes_to_run':     ['process_2', 'process_3', 'process_4'],
    #     'temporal_resolution':  '5min',
    #     'run_process_modelling': True,
    # },
    # {
    #     'data_experiment':      '1',
    #     'run_name':             'experiment_12',
    #     'processes_to_run':     ['process_2', 'process_3', 'process_4'],
    #     'temporal_resolution':  '15min',
    #     'run_process_modelling': True,
    # },
    # {
    #     'data_experiment':      '1',
    #     'run_name':             'experiment_10',
    #     'processes_to_run':     ['process_2', 'process_3', 'process_4'],
    #     'temporal_resolution':  'original',
    #     'run_process_modelling': True,
    # },
    # Add more experiments here...
]

# ── Runner ────────────────────────────────────────────────────────────────────
modelling_script = Path(__file__).parent / 'modelling.py'

for i, exp in enumerate(EXPERIMENTS, start=1):
    print(f"\n{'='*60}")
    print(f"  Running experiment {i}/{len(EXPERIMENTS)}: {exp['run_name']}")
    print(f"  data_experiment={exp['data_experiment']}  "
          f"temporal_resolution={exp['temporal_resolution']}  "
          f"processes={exp['processes_to_run']}  "
          f"run_process_modelling={exp.get('run_process_modelling', False)}")
    print(f"{'='*60}\n")

    env = os.environ.copy()
    env['PIPELINE_DATA_EXPERIMENT']       = exp['data_experiment']
    env['PIPELINE_RUN_NAME']              = exp['run_name']
    env['PIPELINE_PROCESSES_TO_RUN']      = ','.join(exp['processes_to_run'])
    env['PIPELINE_TEMPORAL_RESOLUTION']   = exp['temporal_resolution']
    env['PIPELINE_RUN_PROCESS_MODELLING'] = 'true' if exp.get('run_process_modelling', False) else 'false'

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
