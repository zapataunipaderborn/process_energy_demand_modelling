# %% [markdown]
# <div style="text-align: center; font-size: 50px;">
#     <b>Energy and Sensor mapping on Process Data</b>
# </div>

# %%

import os
# Disable ALL progress bars (tqdm) to silence pm4py replay noise
os.environ["TQDM_DISABLE"] = "1"

import logging
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import jensenshannon
from collections import Counter
import pm4py
import tempfile
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pm4py
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay

# --- LOGGING AND REDIRECTION SETUP ---
# Silence pm4py internal logs and TBR replay messages
logging.getLogger("pm4py").setLevel(logging.ERROR)
# Silence other potential noise
logging.getLogger("fsspec").setLevel(logging.ERROR)

# Log file is set up after the config block so it goes straight into the run folder.
# Placeholder so references to LOG_FILE later don't break before setup runs.
LOG_FILE = None
_original_stdout = sys.stdout

def report(*args, **kwargs):
    print(*args, file=_original_stdout, flush=True, **kwargs)
    logging.info("[REPORT] " + " ".join(map(str, args)))

class StreamToLogger:
    """Redirects stdout/stderr to logging."""
    def __init__(self, logger_func):
        self.logger_func = logger_func
    def write(self, data):
        for line in data.splitlines():
            if line.strip():
                self.logger_func(line.strip())
    def flush(self):
        pass
RANDOM_SEED = 42

import random
import itertools
import os
from datetime import datetime, timedelta
random.seed(RANDOM_SEED)
import simpy
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
np.random.seed(RANDOM_SEED)
import chardet
import plotly.io as pio
from pathlib import Path
import pm4py
import plotly.express as px
import matplotlib.image as mpimg
import tempfile

pio.renderers.default='notebook'
pd.options.mode.chained_assignment = None
from IPython.display import display, Markdown

# Force pm4py to hide progress bars
from pm4py.util import constants
constants.SHOW_PROGRESS_BAR = False

# %%
import pandas as pd
from sim_extractor import extract_process
from simulation import ProcessSimulation
from sim_modeller import SimModeller

from sklearn.linear_model import Lasso, LogisticRegression
from sim_extractor import extract_energy_modifiers, extract_energy_direct_models
from xgboost import XGBRegressor

# %%

from pathlib import Path
import pandas as pd

experiment          = os.environ.get('PIPELINE_DATA_EXPERIMENT', '5')
# ── Temporal resolution for df_expanded aggregation ──────────────────────────
#   'original' → no aggregation (keep raw rows)
#   '1min'     → resample to 1-minute bins
#   '5min'     → resample to 5-minute bins
#   '15min'    → resample to 15-minute bins
TEMPORAL_RESOLUTION = os.environ.get('PIPELINE_TEMPORAL_RESOLUTION', '15min')

# Parse to minutes so the simulation can convert duration → expected timestep count
def _parse_resolution_minutes(res_str):
    import re
    if res_str == 'original':
        return 1.0   # raw rows — assume 1-minute equivalent as fallback
    m = re.match(r'^(\d+(?:\.\d+)?)(min|h)$', res_str.strip())
    if m:
        val = float(m.group(1))
        return val * 60.0 if m.group(2) == 'h' else val
    return 15.0  # safe default
TEMPORAL_RESOLUTION_MINUTES = _parse_resolution_minutes(TEMPORAL_RESOLUTION)

current_path = Path(__file__).resolve().parent if '__file__' in globals() else Path().resolve()
folder_gold_base = current_path.parent / 'data' / 'gold' / f'experiment_{experiment}'

process_datasets = {}

for process_dir in sorted(folder_gold_base.glob('process_*')):
    datasets_dir = process_dir / 'datasets'
    if not datasets_dir.exists():
        continue

    df_event_log_path = datasets_dir / 'df_event_log.parquet'
    df_expanded_path = datasets_dir / 'df_expanded.parquet'
    df_production_plan_path = datasets_dir / 'df_production_plan.parquet'

    if not (df_event_log_path.exists() and df_expanded_path.exists() and df_production_plan_path.exists()):
        continue

    process_datasets[process_dir.name] = {
        'expanded': pd.read_parquet(df_expanded_path),
        'event_log': pd.read_parquet(df_event_log_path),
        'production_plan': pd.read_parquet(df_production_plan_path),
    }

print(process_datasets.keys())

# ── Aggregate df_expanded to the requested temporal resolution ────────────────
def _aggregate_expanded(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    df = df.copy()
    df['datetime_energy'] = pd.to_datetime(df['datetime_energy'])
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    non_numeric_cols = [c for c in df.columns if c != 'datetime_energy' and c not in numeric_cols]
    agg_dict = {c: 'mean' for c in numeric_cols}
    agg_dict.update({c: 'first' for c in non_numeric_cols})
    return df.resample(freq, on='datetime_energy').agg(agg_dict).reset_index()

if TEMPORAL_RESOLUTION != 'original':
    for _pname, _pdata in process_datasets.items():
        _pdata['expanded'] = _aggregate_expanded(_pdata['expanded'], TEMPORAL_RESOLUTION)
    print(f"df_expanded aggregated to {TEMPORAL_RESOLUTION} resolution.")

# %%
_env_processes   = os.environ.get('PIPELINE_PROCESSES_TO_RUN')
processes_to_run = _env_processes.split(',') if _env_processes else ['process_1', 'process_2', 'process_3', 'process_4']

print(f"Processes to run: {processes_to_run}")

process_datasets_to_model = process_datasets

process_datasets_to_model_sensors = process_datasets_to_model.copy()
process_datasets_to_model_sensors['process_1'] = process_datasets_to_model_sensors.get('process_1', {})
# process_datasets_to_model_sensors['process_1']['objects_to_model'] = ['autoclaving_1']
# process_datasets_to_model_sensors['process_1']['activities_to_model'] = ['heat', 'hold', 'cool']
# process_datasets_to_model_sensors['process_1']['sensors_to_model'] = ['autoclave_steam_demand_kW_energy', 'autoclave_cooling_water_demand_kW_energy', 'destillation_steam_demand_kW_energy']

process_datasets_to_model_sensors['process_4'] = process_datasets_to_model_sensors.get('process_4', {})
# process_datasets_to_model_sensors['process_4']['objects_to_model'] = ['Erhitzer']
# process_datasets_to_model_sensors['process_4']['activities_to_model'] = ['Step-032 = Umlauf', 'Step-030 = Produktion']
# process_datasets_to_model_sensors['process_4']['sensors_to_model'] = ['temp_nach_WR2_(WT2)_5s_energy']

process_datasets_to_model_sensors['process_2'] = process_datasets_to_model_sensors.get('process_2', {})
# process_datasets_to_model_sensors['process_2']['objects_to_model'] = ['l01']
# process_datasets_to_model_sensors['process_2']['activities_to_model'] = ['Produktion']
# process_datasets_to_model_sensors['process_2']['sensors_to_model'] = ['pro_volstrom_l/h_energy']

process_datasets_to_model_sensors['process_3'] = process_datasets_to_model_sensors.get('process_3', {})
# process_datasets_to_model_sensors['process_3']['objects_to_model'] = ['tower_1']
# process_datasets_to_model_sensors['process_3']['activities_to_model'] = ['Produktion']
# process_datasets_to_model_sensors['process_3']['sensors_to_model'] = ['(8)_abluft_mas_kg/h_energy']

print(process_datasets_to_model_sensors)
# display(process_datasets_to_model_sensors['process_3']['expanded'])

# Filter the original dictionary
process_datasets_to_model = {
    k: v for k, v in process_datasets.items() 
    if k in processes_to_run
}
# The rest of your configuration will now only see process_3
process_datasets_to_model_sensors = process_datasets_to_model.copy()


# ─────────────────────────────────────────────────────────────────────────────
# MODES TO COMPARE
#
#   What each mode uses:
#
#   Mode                          Miner                  Duration source
#   ─────────────────────────────────────────────────────────────────────
#   statistical                   MINING_ALGORITHM        fitted distributions
#   petri_net_alpha/heuristic/    that specific miner     fitted distributions
#     inductive/ilp
#   petri_net_energy_*            best of the above       per-activity auto-selected:
#                                 (auto-selected by         ML if val score beats
#                                 train score)              statistical baseline,
#                                                           else statistical
#   ml / ml_duration_only         MINING_ALGORITHM        ML models (ML_MODEL_TYPES)
#
#   → MINING_ALGORITHM:  only affects 'statistical' and the ml* modes.
#                        energy-aware modes ignore it — they auto-pick the
#                        best miner from the petri_net_* results.
#   → SIMULATION_MODE:   only affects ml* modes ('ml' or 'ml_duration_only').
#                        statistical and petri_net_* modes do not use it.
#
#   Energy-aware auto-selection (per activity, per modifier type):
#     Duration  : ML kept only if val R² > 0  (> statistical baseline of predicting mean)
#     Transition: ML kept only if val Acc > majority-class baseline accuracy
#     petri_net_energy_aware = best duration choice + best transition choice independently
#
#   Energy-direct modes (new):
#     energy_state → ML → duration_minutes directly   (not a correction factor)
#     energy_state → ML → sample next activity directly from predict_proba
#     Same ML-vs-statistical auto-selection applies per activity
# ─────────────────────────────────────────────────────────────────────────────
MODES_TO_COMPARE = [
    'statistical',
    'petri_net_alpha',
    'petri_net_heuristic',
    'petri_net_inductive',
    'petri_net_combined',
    'petri_net_ilp',
    # ── energy-aware Petri-net variants ──────────────────────────────
    # Modifier approach: ML corrects a statistical base (ML only if it beats baseline)
    'petri_net_energy_duration_aware',    # best duration (ML or stat) per activity; base PN transitions
    'petri_net_energy_transition_aware',  # best transitions (ML or stat) per activity; base PN durations
    'petri_net_energy_aware',             # best duration + best transition independently per activity
    # Direct approach: ML IS the prediction (energy_state → duration or next_activity directly)
    'petri_net_energy_direct_duration_only',    # ML predicts duration directly; base PN transitions
    'petri_net_energy_direct_transition_only',  # ML predicts next activity directly; stat durations
    'petri_net_energy_direct',                  # ML predicts both directly per activity
    #'petri_net_statistical',
    #'petri_net_statistical_memory',
    #'ml_duration_only',
    #'ml_duration_only_with_activity_past',
    #'ml_duration_only_with_activity_past_point_estimate',
    #'ml_global_model',
]

# Keep requested modes, but drop Petri-net variants that are not enabled.
_ENERGY_AWARE_MODES = {
    # Modifier approach: ML corrects a statistical base duration/PN weights
    'petri_net_energy_aware',
    'petri_net_energy_duration_aware',
    'petri_net_energy_transition_aware',
    # Direct approach: ML is the full prediction (no statistical base)
    'petri_net_energy_direct',
    'petri_net_energy_direct_duration_only',
    'petri_net_energy_direct_transition_only',
}
_ENERGY_DIRECT_MODES = {
    'petri_net_energy_direct',
    'petri_net_energy_direct_duration_only',
    'petri_net_energy_direct_transition_only',
}

# ── Duration modifier models ───────────────────────────────────────────────
# List of sklearn-compatible regressor types to compete per activity
ENERGY_DURATION_MODELS    = ['xgboost', 'linear', 'lasso', 'mlp', 'statistical']

# ── Transition modifier models ─────────────────────────────────────────────
# List of classifier types to compete per activity
ENERGY_TRANSITION_MODELS  = ['logistic', 'random_forest', 'gradient_boosting']

ENERGY_DURATION_SCALE_CLIP = (0.7, 1.3)   # max ±30% shift per activity
ENERGY_LOGIT_BIAS_CLIP     = (-1.0, 1.0)  # max ~2.7× odds-ratio shift per competing activity
ENERGY_MIN_SAMPLES         = 5           # skip ML (use statistical) if n_samples < this

# Will be populated per process after energy modelling:
energy_modifiers_by_process = {}

# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION MODE TOGGLE
#   'statistical'      – sample durations/transitions from best-fit distributions
#   'ml'               – use ML models (falls back to statistical when needed)
#   'ml_duration_only' – use ML only for the duration median; std and
#                        transition probabilities still come from data extraction
# ─────────────────────────────────────────────────────────────────────────────
SIMULATION_MODE = 'ml_duration_only'   # ← change to 'ml' or 'ml_duration_only'

# ─────────────────────────────────────────────────────────────────────────────
# PROCESS MINING ALGORITHM
#   'inductive'  – pm4py Inductive Miner → guarantees a sound Petri net
#   'heuristic'  – pm4py Heuristics Miner → better noise filtering
#   'alpha'      – pm4py Alpha Miner → classic algorithm
#   'ilp'        – pm4py ILP Miner → precise/sound, can be strict
#   'manual'     – original manual extraction (no process mining)
# ─────────────────────────────────────────────────────────────────────────────
#MINING_ALGORITHM = 'inductive'   # ← change to 'manual' for old behavior
MINING_ALGORITHM = 'heuristic'
#MINING_ALGORITHM = 'alpha'
#MINING_ALGORITHM = 'ilp'

# Petri-net miner variants to compare when mode names include the algorithm.
PETRI_NET_ALGORITHMS = ['alpha', 'heuristic', 'inductive']#, 'ilp']

# ─────────────────────────────────────────────────────────────────────────────
# MINER HYPERPARAMETER OPTIMIZATION (for inductive + heuristic)
#   Runs local per-group search during extraction and keeps best model.
# ─────────────────────────────────────────────────────────────────────────────
OPTIMIZE_MINING_HYPERPARAMS = False
MINING_SEARCH_SPACE = {
    'inductive_noise_thresholds': [0.05, 0.10, 0.20, 0.30, 0.40],
    'heuristic_params_grid': [
        {'dependency_threshold': 0.30, 'and_threshold': 0.65, 'loop_two_threshold': 0.50},
        {'dependency_threshold': 0.50, 'and_threshold': 0.65, 'loop_two_threshold': 0.50},
        {'dependency_threshold': 0.70, 'and_threshold': 0.65, 'loop_two_threshold': 0.50},
        {'dependency_threshold': 0.50, 'and_threshold': 0.50, 'loop_two_threshold': 0.50},
        {'dependency_threshold': 0.50, 'and_threshold': 0.80, 'loop_two_threshold': 0.50},
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# ML MODEL CONFIGURATION (only used when SIMULATION_MODE is 'ml' or 'ml_duration_only')
#   model_types: list of models to train — best is selected per activity key
#                Supported: 'xgboost', 'linear', 'lasso', 'mlp'
#   optimize_hyperparams: True  → Optuna hyper-parameter search
#                         False → use default model parameters
#   n_optuna_trials: number of Optuna trials per model (ignored if optimize=False)
# ─────────────────────────────────────────────────────────────────────────────
ML_MODEL_TYPES          = ['xgboost', 'linear', 'lasso', 'mlp']  # ← train all, pick best
ML_MODEL_TYPES          = ['xgboost', 'mean', 'median']
ML_OPTIMIZE_HYPERPARAMS = True    # ← set True to enable Optuna tuning
ML_OPTUNA_TRIALS        = 20

# ─────────────────────────────────────────────────────────────────────────────
# CURVE MODELLING HYPERPARAMETER OPTIMIZATION
#   When True, each build_and_train_pipeline_* call runs an Optuna search over
#   regressor hyperparameters (learning rate, max_depth, n_estimators, etc.)
#   instead of using defaults.  Significantly slower but often improves fit.
# ─────────────────────────────────────────────────────────────────────────────
CURVE_OPTIMIZE_HYPERPARAMS = True   # ← Optuna search for sklearn curve models
CURVE_N_OPTUNA_TRIALS      = 50     # ← trials per (sensor, activity, object) combo

# %%
# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION  —  edit everything here, nothing else needs to change
# ══════════════════════════════════════════════════════════════════════════════

# ── Train / test split ────────────────────────────────────────────────────────
TEMPORAL_SPLIT      = True    # True → split by case start time; False → use all data
TRAIN_RATIO         = 0.50    # fraction of cases used for training

# ── Pipeline execution flags ──────────────────────────────────────────────────
RUN_TEST_EVALUATION       = True   # evaluate on held-out test set
RUN_CURVE_ONLY_EVALUATION = True   # run curve-quality benchmark (MAE/RMSE/R²)
RUN_PROCESS_MODELLING     = True#os.environ.get('PIPELINE_RUN_PROCESS_MODELLING', 'false').lower() == 'true'

# ── Approaches to train — comment out any you want to skip ───────────────────
#    'baseline'          DTW + position index (sklearn regressor)
#    'instance_stats'    DTW + per-curve stats  (leaky — known invalid)
#    'istats_leakfree'   DTW + two-stage leak-free stats
#    'dtw_phase'         DTW + phase features
#    'basis'             DTW + B-spline basis expansion
#    'exog'              DTW + external factors (ef_* columns)
#    'amplitude_shape'   Separate amplitude (curve_mean) from shape (z-score);
#                        stage A predicts amplitude from metadata, stage B shape
#    'seq2seq'           DTW + LSTM encoder-decoder
#    'seq2seq_only'      LSTM encoder-decoder, no DTW
#    'seq2seq_exog'      DTW + LSTM encoder-decoder + external factors
APPROACHES = [
    'baseline',
    'instance_stats',
    'istats_leakfree',
    # 'dtw_phase',
    # 'basis',
    'exog',
    'exog_prev_activity',
    #'amplitude_shape',

    'seq2seq',
    'seq2seq_only',
    # 'seq2seq_exog',
    'seq2seq_prev_activity',
]

# ── Seq2Seq hyperparameters ───────────────────────────────────────────────────
SEQ2SEQ_HIDDEN_SIZE          = 128
SEQ2SEQ_NUM_LAYERS           = 2
SEQ2SEQ_DROPOUT              = 0.1
SEQ2SEQ_EPOCHS               = 80
SEQ2SEQ_BATCH_SIZE           = 32
SEQ2SEQ_LR                   = 1e-3
SEQ2SEQ_TEACHER_FORCING      = 0.5
SEQ2SEQ_PATIENCE             = 10

# ── Results export ────────────────────────────────────────────────────────────
EXPORT_RESULTS = True   # save parquet + HTML to results/<timestamp>/

# ── Run folder + live log (created immediately so log captures everything) ───
import datetime as _dt
_run_ts       = _dt.datetime.now().strftime('%Y%m%d_%H%M%S')
_run_name     = os.environ.get('PIPELINE_RUN_NAME', f'experiment_{experiment}')
_results_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
_run_dir = os.path.join(_results_root, f"{_run_name}_{_run_ts}")
_plots_dir           = os.path.join(_run_dir, 'plots')
_process_results_dir = os.path.join(_run_dir, 'process_results')
_energy_results_dir  = os.path.join(_run_dir, 'energy_results')
os.makedirs(_plots_dir, exist_ok=True)
os.makedirs(_process_results_dir, exist_ok=True)
os.makedirs(_energy_results_dir, exist_ok=True)

LOG_FILE = os.path.join(_run_dir, 'pipeline_execution.log')

# Wire logging directly to the run folder log file, flushing after every record
_log_handler = logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')
_log_handler.setLevel(logging.INFO)
_log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
_log_handler.stream.reconfigure(line_buffering=True)   # flush every line

_root_logger = logging.getLogger()
_root_logger.handlers.clear()   # drop any handlers basicConfig may have added
_root_logger.addHandler(_log_handler)
_root_logger.setLevel(logging.INFO)

logging.getLogger("pm4py").setLevel(logging.ERROR)
logging.getLogger("fsspec").setLevel(logging.ERROR)
logging.captureWarnings(True)

# Redirect all prints to the live log (original stdout still used by report())
sys.stdout = StreamToLogger(logging.info)
sys.stderr = StreamToLogger(logging.error)

import time as _time
_pipeline_start = _time.perf_counter()

logging.info(f"Run started — output folder: {_run_dir}")
logging.info(f"Approaches: {APPROACHES}")


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION UTILITIES — HEATMAP ENGINE
# ══════════════════════════════════════════════════════════════════════════════

# Metrics where LOWER is better (prefixes like train_ or test_ are stripped before checking)
METRICS_LOWER_IS_BETTER = {
    'basic_metrics_event_count_error',
    'basic_metrics_case_count_error',
    'activity_metrics_js_divergence',
    'activity_metrics_frequency_mae',
    'duration_metrics_ks_statistic',
    'duration_metrics_mean_duration_error',
    'duration_metrics_median_duration_error',
    'duration_metrics_std_duration_error',
    'case_metrics_events_per_case_ks',
    'case_metrics_median_events_per_case_error',
}

# The core set of metrics the user wants to see in the heatmaps
CORE_METRIC_BASES = [
    # Overall
    'overall_score',
    # Direct sim-output vs real-log comparison
    'basic_metrics_event_count_ratio',
    'duration_metrics_mean_duration_error',
    'duration_metrics_median_duration_error',
    'activity_metrics_js_divergence',
    'control_flow_metrics_edge_precision',
    'control_flow_metrics_edge_recall',
    'control_flow_metrics_edge_f1_score',
    # Process model quality (Petri net vs real log)
    'conformance_metrics_fitness',
    'conformance_metrics_precision',
    'conformance_metrics_generalization',
    'conformance_metrics_simplicity',
]

# Default definitions to avoid NameError when testing is skipped
process_test_cols = []
energy_test_cols = []
enabled_test_metrics = set()
lower_is_better = set()
higher_is_better = set()

# Metrics where HIGHER is better (prefixes like train_ or test_ are stripped before checking)
METRICS_HIGHER_IS_BETTER = {

    'overall_score',
    'basic_metrics_event_count_ratio',
    'basic_metrics_case_count_ratio',
    'activity_metrics_activity_coverage_ratio',
    'duration_metrics_ks_pvalue',
    'case_metrics_events_per_case_pvalue',
    'conformance_metrics_fitness',
    'conformance_metrics_precision',
    'conformance_metrics_generalization',
    'conformance_metrics_simplicity',
    'control_flow_metrics_edge_precision',
    'control_flow_metrics_edge_recall',
    'control_flow_metrics_edge_f1_score',
    'control_flow_metrics_start_activities_jaccard',
    'control_flow_metrics_end_activities_jaccard',
}

def _normalise_metrics(df, cols, lower_set_test_prefixed=None):
    """Min-max normalise. Inverts lower-is-better metrics so 1 = best."""
    norm_df = df.copy()
    
    # We strip prefixes for the 'lower is better' check to be phase-agnostic
    base_lower_names = set()
    if lower_set_test_prefixed:
        base_lower_names = {s.replace('test_', '').replace('train_', '') for s in lower_set_test_prefixed}
    
    # Add our static base names
    base_lower_names.update(METRICS_LOWER_IS_BETTER)

    for c in cols:
        vals = df[c].dropna()
        if len(vals) == 0:
            continue
        vmin, vmax = vals.min(), vals.max()

        rng = vmax - vmin if vmax != vmin else 1.0
        clean_c = c.replace('test_', '').replace('train_', '')

        is_lower = clean_c in base_lower_names or any(m in clean_c for m in ['MAE', 'RMSE', 'WAPE'])
        if 'R2' in clean_c:
            is_lower = False

        if is_lower:
            norm_df[c] = (vmax - df[c]) / rng
        else:
            norm_df[c] = (df[c] - vmin) / rng
    return norm_df


def _plot_results_heatmap(cols, title, metric_type='process', local_df=None, save_path=None, agg='mean'):
    """Displays a normalized heatmap for comparing different simulation modes."""
    # Use global evaluation_results_df if no local_df is provided
    target_df = local_df if local_df is not None else globals().get('evaluation_results_df')

    if not cols or target_df is None or target_df.empty or 'mode' not in target_df.columns:
        return

    # Filter columns that actually exist in the dataframe AND are numeric
    valid_cols = [
        c for c in cols
        if c in target_df.columns and target_df[c].dtype in ('float64', 'float32', 'int64', 'int32')
    ]
    if not valid_cols:
        return

    # Phase-agnostic normalization
    if agg == 'median':
        mode_avg = target_df.groupby('mode')[valid_cols].median()
    else:
        mode_avg = target_df.groupby('mode')[valid_cols].mean()
    # We pass higher_is_better if it exists globally, otherwise empty set for base check
    global_lower = globals().get('lower_is_better', set())
    mode_avg_norm = _normalise_metrics(mode_avg, valid_cols, global_lower)
    
    # Sort by overall score if available
    sort_opts = ['test_overall_score', 'train_overall_score', valid_cols[0]]
    sort_key = next((k for k in sort_opts if k in mode_avg_norm.columns), valid_cols[0])
    mode_avg_norm = mode_avg_norm.sort_values(sort_key, ascending=False)
    
    # Generate labels dynamically
    current_labels = {
        'train_overall_score': 'Overall',
        'test_overall_score':  'Overall',
        # Direct sim-vs-real
        'train_basic_metrics_event_count_ratio':      '1/EvtRatio',
        'test_basic_metrics_event_count_ratio':       '1/EvtRatio',
        'train_duration_metrics_mean_duration_error': '1-MeanDurErr',
        'test_duration_metrics_mean_duration_error':  '1-MeanDurErr',
        'train_duration_metrics_median_duration_error': '1-MedDurErr',
        'test_duration_metrics_median_duration_error':  '1-MedDurErr',
        'train_activity_metrics_js_divergence':       '1-JS div',
        'test_activity_metrics_js_divergence':        '1-JS div',
        'train_control_flow_metrics_edge_precision':  'EdgePrec',
        'test_control_flow_metrics_edge_precision':   'EdgePrec',
        'train_control_flow_metrics_edge_recall':     'EdgeRec',
        'test_control_flow_metrics_edge_recall':      'EdgeRec',
        'train_control_flow_metrics_edge_f1_score':   'EdgeF1',
        'test_control_flow_metrics_edge_f1_score':    'EdgeF1',
        # Process model quality
        'train_conformance_metrics_fitness':          'Fitness',
        'test_conformance_metrics_fitness':           'Fitness',
        'train_conformance_metrics_precision':        'Precision',
        'test_conformance_metrics_precision':         'Precision',
        'train_conformance_metrics_generalization':   'Generaliz',
        'test_conformance_metrics_generalization':    'Generaliz',
        'train_conformance_metrics_simplicity':       'Simplicity',
        'test_conformance_metrics_simplicity':        'Simplicity',
    }
    
    for c in valid_cols:
        if c in current_labels:
            continue
        if '_energy_' in c:

            parts = c.split('_')
            sensor = parts[2]
            metric = parts[-1]
            current_labels[c] = f"{sensor[:3]}: {metric}"
        else:
            current_labels[c] = c.replace('test_', '').replace('train_', '').replace('_metrics_', ': ').replace('_', ' ').title()
    
    disp_cols = [current_labels[c] for c in valid_cols]
    plot_df = mode_avg_norm[valid_cols].copy()
    plot_df.columns = disp_cols

    # Annotation values: transform so 1 = best for every metric
    annot_df = mode_avg[valid_cols].reindex(mode_avg_norm.index).copy()
    for c in valid_cols:
        clean_c = c.replace('test_', '').replace('train_', '')
        is_lower = clean_c in METRICS_LOWER_IS_BETTER or any(m in clean_c for m in ['MAE', 'RMSE', 'WAPE'])
        if is_lower:
            annot_df[c] = (1.0 - annot_df[c]).clip(0, 1)
        elif clean_c == 'basic_metrics_event_count_ratio':
            # 1.0 = perfect; above 1 or below 1 both worse
            annot_df[c] = annot_df[c].apply(
                lambda r: (1.0 / r if r > 1 else r) if pd.notna(r) and r > 0 else np.nan
            ).clip(0, 1)
    annot_df = annot_df.round(3)
    annot_df.columns = disp_cols

    fig, ax = plt.subplots(figsize=(max(10, len(valid_cols) * 1.2), max(3, len(mode_avg_norm) * 0.8)))
    sns.heatmap(plot_df, annot=annot_df, fmt='', cmap='RdYlGn', vmin=0, vmax=1, linewidths=0.5, ax=ax,
                cbar_kws={'label': 'Normalised score (1 = best)'})
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel('')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    # Log/Table summary
    metrics_table = mode_avg[valid_cols].reindex(mode_avg_norm.index).round(4)
    metrics_table.columns = disp_cols
    print(f"\n{title.upper()}")
    print("-" * len(title))
    with pd.option_context('display.max_columns', None, 'display.width', 200):
        print(metrics_table.to_string())

# %% [markdown]
# # Define Paths and Load Datasets

# %%
process_datasets = {}



# %% [markdown]
# # Model the process

try:
    from pm4py.algo.evaluation.generalization import algorithm as pm4py_generalization
except Exception:
    pm4py_generalization = None

try:
    from pm4py.algo.evaluation.simplicity import algorithm as pm4py_simplicity
except Exception:
    pm4py_simplicity = None


def _mean_trace_fitness(log_df, net, im, fm):
    """Compute mean trace fitness from token replay; returns np.nan on failure."""
    try:
        replay = token_replay.apply(
            log_df, net, im, fm,
            parameters={'consider_remaining_in_fitness': True}
        )
        scores = []
        for item in replay:
            if 'trace_fitness' in item and item['trace_fitness'] is not None:
                scores.append(float(item['trace_fitness']))
            elif item.get('trace_is_fit') is True:
                scores.append(1.0)
            else:
                scores.append(0.0)
        return float(np.mean(scores)) if scores else np.nan
    except Exception:
        return np.nan


def _safe_precision(log_df, net, im, fm):
    """Compute token-based precision; returns np.nan on failure."""
    try:
        return float(pm4py.precision_token_based_replay(log_df, net, im, fm))
    except Exception:
        return np.nan


def _safe_generalization(log_df, net, im, fm):
    """Compute pm4py generalization when available; returns np.nan otherwise."""
    if pm4py_generalization is None:
        return np.nan
    try:
        return float(pm4py_generalization.apply(log_df, net, im, fm))
    except Exception:
        return np.nan


def _safe_simplicity(net):
    """Compute simplicity; prefer pm4py metric and fall back to complexity proxy."""
    if pm4py_simplicity is not None:
        try:
            return float(pm4py_simplicity.apply(net))
        except Exception:
            pass

    complexity = len(net.places) + len(net.transitions) + len(net.arcs)
    return float(1.0 / (1.0 + 0.005 * float(complexity)))

def comprehensive_simulation_evaluation(simulated_df, real_df, real_expanded_df=None,
                                       case_col='case_id', activity_col='activity',
                                       start_col='timestamp_start', end_col='timestamp_end',
                                       process_models=None, station_col='higher_level_activity'):
    """
    Comprehensive evaluation of simulation quality based on process mining literature
    
    Metrics based on:
    - Rozinat et al. (2008): Conformance checking of processes based on monitoring real behavior
    - van der Aalst et al. (2010): Process mining manifesto
    - Burattin & Sperduti (2010): Automatic determination of parameters' values for heuristics miner++
    """
    
    results = {}

    # Work on copies and guarantee required columns exist to avoid KeyError
    simulated_df = simulated_df.copy()
    real_df = real_df.copy()

    required_cols = [case_col, activity_col, start_col, end_col]
    for col in required_cols:
        if col not in simulated_df.columns:
            simulated_df[col] = pd.Series(dtype='object')
        if col not in real_df.columns:
            real_df[col] = pd.Series(dtype='object')

    simulated_df[start_col] = pd.to_datetime(simulated_df[start_col], errors='coerce')
    simulated_df[end_col] = pd.to_datetime(simulated_df[end_col], errors='coerce')
    real_df[start_col] = pd.to_datetime(real_df[start_col], errors='coerce')
    real_df[end_col] = pd.to_datetime(real_df[end_col], errors='coerce')
    
    report("="*80)
    report("COMPREHENSIVE SIMULATION EVALUATION")
    report("="*80)
    
    # ========== 1. BASIC PROCESS METRICS ==========
    report("\n1. BASIC PROCESS METRICS")
    report("-" * 40)
    
    # Event counts
    sim_events = len(simulated_df)
    real_events = len(real_df)
    event_ratio = sim_events / real_events if real_events > 0 else 0
    
    # Case counts  
    sim_cases = simulated_df[case_col].nunique() if case_col in simulated_df.columns else 0
    real_cases = real_df[case_col].nunique() if case_col in real_df.columns else 0
    case_ratio = sim_cases / real_cases if real_cases > 0 else 0
    
    report(f"Events - Real: {real_events}, Sim: {sim_events}, Ratio: {event_ratio:.3f}")
    report(f"Cases - Real: {real_cases}, Sim: {sim_cases}, Ratio: {case_ratio:.3f}")
    
    results['basic_metrics'] = {
        'event_count_ratio': event_ratio,
        'case_count_ratio': case_ratio,
        'event_count_error': abs(1 - event_ratio),
        'case_count_error': abs(1 - case_ratio)
    }
    
    # ========== 2. ACTIVITY FREQUENCY ANALYSIS ==========
    report("\n2. ACTIVITY FREQUENCY ANALYSIS")
    report("-" * 40)
    
    # Activity frequencies
    sim_activity_freq = simulated_df[activity_col].dropna().value_counts(normalize=True).sort_index()
    real_activity_freq = real_df[activity_col].dropna().value_counts(normalize=True).sort_index()
    
    # Align activities (handle missing activities)
    all_activities = sorted(set(sim_activity_freq.index) | set(real_activity_freq.index))
    if all_activities:
        sim_freq_aligned = pd.Series([sim_activity_freq.get(act, 0) for act in all_activities], index=all_activities)
        real_freq_aligned = pd.Series([real_activity_freq.get(act, 0) for act in all_activities], index=all_activities)
        # Jensen-Shannon divergence for activity distributions
        js_divergence = jensenshannon(sim_freq_aligned.values, real_freq_aligned.values)
        # Mean Absolute Error of frequencies
        freq_mae = np.mean(np.abs(sim_freq_aligned.values - real_freq_aligned.values))
    else:
        # No activities to compare -> treat as worst similarity.
        js_divergence = 1.0
        freq_mae = 1.0
    
    report(f"Activity Coverage - Real: {len(real_activity_freq)}, Sim: {len(sim_activity_freq)}")
    report(f"Jensen-Shannon Divergence (activities): {js_divergence:.4f} (0=perfect, 1=worst)")
    report(f"Mean Absolute Error (frequencies): {freq_mae:.4f}")
    
    results['activity_metrics'] = {
        'js_divergence': js_divergence,
        'frequency_mae': freq_mae,
        'activity_coverage_ratio': len(sim_activity_freq) / len(real_activity_freq) if len(real_activity_freq) > 0 else 0
    }
    
    # ========== 3. DURATION ANALYSIS ==========
    report("\n3. DURATION ANALYSIS")
    report("-" * 40)
    
    # Calculate durations
    sim_durations = (simulated_df[end_col] - simulated_df[start_col]).dt.total_seconds() / 60
    real_durations = (real_df[end_col] - real_df[start_col]).dt.total_seconds() / 60
    sim_durations = sim_durations.replace([np.inf, -np.inf], np.nan).dropna()
    real_durations = real_durations.replace([np.inf, -np.inf], np.nan).dropna()

    if len(sim_durations) > 0 and len(real_durations) > 0:
        # Statistical comparison
        duration_ks_stat, duration_ks_pvalue = stats.ks_2samp(sim_durations, real_durations)

        # Duration statistics comparison
        duration_stats = pd.DataFrame({
            'Real': [real_durations.mean(), real_durations.median(), real_durations.std()],
            'Simulated': [sim_durations.mean(), sim_durations.median(), sim_durations.std()],
        }, index=['Mean', 'Median', 'Std'])

        duration_stats['Error'] = np.where(
            duration_stats['Real'] != 0,
            np.abs(duration_stats['Simulated'] - duration_stats['Real']) / np.abs(duration_stats['Real']),
            1.0
        )
    else:
        duration_ks_stat, duration_ks_pvalue = 1.0, 0.0
        duration_stats = pd.DataFrame(
            {'Real': [np.nan, np.nan, np.nan],
             'Simulated': [np.nan, np.nan, np.nan],
             'Error': [1.0, 1.0, 1.0]},
            index=['Mean', 'Median', 'Std']
        )
    
    report("Duration Statistics:")
    report(duration_stats.round(3))
    report(f"\nKolmogorov-Smirnov Test: KS={duration_ks_stat:.4f}, p-value={duration_ks_pvalue:.4f}")
    report(f"(p > 0.05 suggests distributions are similar)")
    
    results['duration_metrics'] = {
        'ks_statistic': duration_ks_stat,
        'ks_pvalue': duration_ks_pvalue,
        'mean_duration_error': duration_stats.loc['Mean', 'Error'],
        'median_duration_error': duration_stats.loc['Median', 'Error'],
        'std_duration_error': duration_stats.loc['Std', 'Error']
    }
    
    # ========== 4. CASE-LEVEL ANALYSIS ==========
    report("\n4. CASE-LEVEL ANALYSIS")
    report("-" * 40)
    
    # Events per case
    sim_events_per_case = simulated_df.groupby(case_col).size() if len(simulated_df) > 0 else pd.Series(dtype=float)
    real_events_per_case = real_df.groupby(case_col).size() if len(real_df) > 0 else pd.Series(dtype=float)

    if len(sim_events_per_case) > 0 and len(real_events_per_case) > 0:
        # Statistical test for events per case
        events_per_case_ks, events_per_case_pvalue = stats.ks_2samp(sim_events_per_case, real_events_per_case)

        case_stats = pd.DataFrame({
            'Real': [real_events_per_case.mean(), real_events_per_case.median(), real_events_per_case.std()],
            'Simulated': [sim_events_per_case.mean(), sim_events_per_case.median(), sim_events_per_case.std()],
        }, index=['Mean', 'Median', 'Std'])

        case_stats['Error'] = np.where(
            case_stats['Real'] != 0,
            np.abs(case_stats['Simulated'] - case_stats['Real']) / np.abs(case_stats['Real']),
            1.0
        )
    else:
        events_per_case_ks, events_per_case_pvalue = 1.0, 0.0
        case_stats = pd.DataFrame(
            {'Real': [np.nan, np.nan, np.nan],
             'Simulated': [np.nan, np.nan, np.nan],
             'Error': [1.0, 1.0, 1.0]},
            index=['Mean', 'Median', 'Std']
        )
    
    report("Events per Case Statistics:")
    report(case_stats.round(3))
    report(f"\nKS Test (events per case): KS={events_per_case_ks:.4f}, p-value={events_per_case_pvalue:.4f}")
    
    results['case_metrics'] = {
        'events_per_case_ks': events_per_case_ks,
        'events_per_case_pvalue': events_per_case_pvalue,
        'mean_events_per_case_error': case_stats.loc['Mean', 'Error'],
        'median_events_per_case_error': case_stats.loc['Median', 'Error']
    }
    
    # ========== 5. CONTROL-FLOW ANALYSIS ==========
    report("\n5. CONTROL-FLOW ANALYSIS (Directly-Follows Graph)")
    report("-" * 40)
    
    # Create event logs for pm4py
    sim_for_dfg = simulated_df.dropna(subset=[case_col, activity_col, start_col])
    real_for_dfg = real_df.dropna(subset=[case_col, activity_col, start_col])

    if len(sim_for_dfg) > 0 and len(real_for_dfg) > 0:
        sim_log = pm4py.format_dataframe(sim_for_dfg, case_id=case_col, activity_key=activity_col, timestamp_key=start_col)
        real_log = pm4py.format_dataframe(real_for_dfg, case_id=case_col, activity_key=activity_col, timestamp_key=start_col)

        # Discover DFGs
        sim_dfg, sim_start, sim_end = pm4py.discover_dfg(sim_log)
        real_dfg, real_start, real_end = pm4py.discover_dfg(real_log)

        # Compare DFG edges
        sim_edges = set(sim_dfg.keys())
        real_edges = set(real_dfg.keys())
    else:
        sim_edges, real_edges = set(), set()
        sim_start, real_start = {}, {}
        sim_end, real_end = {}, {}
    
    edge_precision = len(sim_edges & real_edges) / len(sim_edges) if len(sim_edges) > 0 else 0
    edge_recall = len(sim_edges & real_edges) / len(real_edges) if len(real_edges) > 0 else 0
    edge_f1 = 2 * (edge_precision * edge_recall) / (edge_precision + edge_recall) if (edge_precision + edge_recall) > 0 else 0
    
    report(f"DFG Edges - Real: {len(real_edges)}, Sim: {len(sim_edges)}, Common: {len(sim_edges & real_edges)}")
    report(f"Edge Precision: {edge_precision:.4f}")
    report(f"Edge Recall: {edge_recall:.4f}")
    report(f"Edge F1-Score: {edge_f1:.4f}")
    
    # Compare start/end activities (convert dict keys to sets)
    sim_start_set = set(sim_start.keys())
    real_start_set = set(real_start.keys())
    sim_end_set = set(sim_end.keys())
    real_end_set = set(real_end.keys())
    
    start_jaccard = len(sim_start_set & real_start_set) / len(sim_start_set | real_start_set) if len(sim_start_set | real_start_set) > 0 else 0
    end_jaccard = len(sim_end_set & real_end_set) / len(sim_end_set | real_end_set) if len(sim_end_set | real_end_set) > 0 else 0
    
    report(f"Start Activities Jaccard: {start_jaccard:.4f}")
    report(f"End Activities Jaccard: {end_jaccard:.4f}")
    
    results['control_flow_metrics'] = {
        'edge_precision': edge_precision,
        'edge_recall': edge_recall,
        'edge_f1_score': edge_f1,
        'start_activities_jaccard': start_jaccard,
        'end_activities_jaccard': end_jaccard
    }
    
    # ========== 6. CLASSIC PROCESS-MINING DIMENSIONS (ADDITIVE) ==========
    report("\n6. CLASSIC PROCESS-MINING DIMENSIONS")
    report("-" * 40)

    conformance_metrics = {
        'fitness': np.nan,
        'precision': np.nan,
        'generalization': np.nan,
        'simplicity': np.nan,
    }

    if process_models is not None:
        # Per-station evaluation: filter the REAL log by station and replay against
        # each station's own net.  This avoids the cross-station precision artifact
        # and is not confounded by the simulation's inflated event count.
        fit_list, prec_list, gen_list, sim_list = [], [], [], []
        seen_stations: set = set()
        for key, pm_entry in process_models.items():
            _obj, obj_type, station = key
            if (obj_type, station) in seen_stations:
                continue
            seen_stations.add((obj_type, station))
            net_s, im_s, fm_s = pm_entry["net"], pm_entry["im"], pm_entry["fm"]
            if station_col in real_df.columns:
                df_st = real_df[real_df[station_col] == station].dropna(
                    subset=[case_col, activity_col, start_col])
            else:
                df_st = real_df.dropna(subset=[case_col, activity_col, start_col])
            if len(df_st) == 0:
                continue
            try:
                log_st = pm4py.format_dataframe(
                    df_st, case_id=case_col,
                    activity_key=activity_col, timestamp_key=start_col)
                f  = _mean_trace_fitness(log_st, net_s, im_s, fm_s)
                p  = _safe_precision(log_st, net_s, im_s, fm_s)
                g  = _safe_generalization(log_st, net_s, im_s, fm_s)
                s  = _safe_simplicity(net_s)
                if not np.isnan(f):  fit_list.append(f)
                if not np.isnan(p):  prec_list.append(p)
                if not np.isnan(g):  gen_list.append(g)
                if not np.isnan(s):  sim_list.append(s)
            except Exception as e:
                report(f"  Conformance skipped for station '{station}': {e}")
        if fit_list:  conformance_metrics['fitness']        = float(np.mean(fit_list))
        if prec_list: conformance_metrics['precision']      = float(np.mean(prec_list))
        if gen_list:  conformance_metrics['generalization'] = float(np.mean(gen_list))
        if sim_list:  conformance_metrics['simplicity']     = float(np.mean(sim_list))
    # For non-Petri modes (statistical, ML) there is no process model, so
    # conformance metrics stay NaN.  Direct sim-vs-real comparison is
    # already captured by the DFG edge metrics and the basic metrics above.

    for m_name, m_val in conformance_metrics.items():
        if pd.isna(m_val):
            report(f"{m_name:35}: n/a")
        else:
            report(f"{m_name:35}: {m_val:.4f}")

    results['conformance_metrics'] = conformance_metrics

    # ========== 7. ENERGY PROFILE METRICS ==========
    energy_results = {}
    if real_expanded_df is not None and len(simulated_df) > 0 and 'simulated_energy_curves' in simulated_df.columns:
        report("\n7. ENERGY PROFILE METRICS (Simulation vs Reality)")
        report("-" * 40)
        
        # We need to compare curves for matching (Case, Activity) pairs
        sensor_metrics = {}
        
        # Flatten all simulated curves from the log
        for idx, row in simulated_df.iterrows():
            sim_curves = row.get('simulated_energy_curves', {})
            if not isinstance(sim_curves, dict) or not sim_curves:
                continue
            
            cid = row[case_col]
            act = row[activity_col]
            
            # Find the REAL curve for this case/activity in the test expanded df
            # We filter for the specific case and activity.
            # Note: There might be multiple instances, we take the one closest in sequence if needed,
            # but for now we look for any match in that case.
            real_case_act = real_expanded_df[
                (real_expanded_df['case_id_log'] == cid) & 
                (real_expanded_df['activity_log'] == act)
            ]
            
            if real_case_act.empty:
                continue
                
            for sensor, y_sim in sim_curves.items():
                if sensor in real_case_act.columns:
                    y_real = real_case_act[sensor].dropna().values
                    if len(y_real) > 1:
                        # Resample either sim or real to match lengths for direct comparison
                        # using simple linear interpolation (like in sim_extractor)
                        if len(y_sim) != len(y_real):
                            from scipy.interpolate import interp1d
                            x_old = np.linspace(0, 1, len(y_sim))
                            f = interp1d(x_old, y_sim, kind='linear', fill_value='extrapolate')
                            x_new = np.linspace(0, 1, len(y_real))
                            y_sim_resampled = f(x_new)
                        else:
                            y_sim_resampled = y_sim
                        
                        mae = np.mean(np.abs(y_real - y_sim_resampled))
                        rmse = np.sqrt(np.mean((y_real - y_sim_resampled)**2))
                        denom = np.sum(np.abs(y_real))
                        wape = np.sum(np.abs(y_real - y_sim_resampled)) / denom * 100 if denom > 0 else 0
                        ss_tot = np.sum((y_real - np.mean(y_real))**2)
                        ss_res = np.sum((y_real - y_sim_resampled)**2)
                        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

                        if sensor not in sensor_metrics:
                            sensor_metrics[sensor] = {'mae': [], 'rmse': [], 'wape': [], 'r2': [], 'activity': []}
                        sensor_metrics[sensor]['mae'].append(mae)
                        sensor_metrics[sensor]['rmse'].append(rmse)
                        sensor_metrics[sensor]['wape'].append(wape)
                        sensor_metrics[sensor]['r2'].append(r2)
                        sensor_metrics[sensor]['activity'].append(act)

        # Aggregate per sensor
        agg_energy = {}
        act_energy = {}
        for sensor, m in sensor_metrics.items():
            s_mae  = np.mean(m['mae'])
            s_rmse = np.mean(m['rmse'])
            s_wape = np.mean(m['wape'])
            s_r2   = np.mean(m['r2'])
            agg_energy[sensor] = {'MAE': s_mae, 'RMSE': s_rmse, 'WAPE': s_wape, 'R2': s_r2}
            report(f"  {sensor:40}: MAE={s_mae:.2f}, RMSE={s_rmse:.2f}, WAPE={s_wape:.4f}  R²={s_r2:.4f}")
            # Per-activity breakdown
            for _a in set(m['activity']):
                _idxs = [i for i, x in enumerate(m['activity']) if x == _a]
                act_energy[(sensor, _a)] = {
                    'MAE':  float(np.mean([m['mae'][i]  for i in _idxs])),
                    'RMSE': float(np.mean([m['rmse'][i] for i in _idxs])),
                    'WAPE': float(np.mean([m['wape'][i] for i in _idxs])),
                    'R2':   float(np.mean([m['r2'][i]   for i in _idxs])),
                }

        results['energy_metrics'] = agg_energy
        results['activity_energy_metrics'] = act_energy
        energy_results = agg_energy

    # ========== 8. OVERALL QUALITY SCORE ==========
    report("\n8. OVERALL QUALITY ASSESSMENT")
    report("-" * 40)

    # All components transformed so that 1 = perfect, 0 = worst.
    _evt_ratio = results['basic_metrics']['event_count_ratio']
    _js_div    = results['activity_metrics'].get('js_divergence', np.nan)
    _edge_f1   = results['control_flow_metrics'].get('edge_f1_score', np.nan)
    active_components = {
        # 1/ratio when ratio>1, ratio when ratio<1 → 1.0 when ratio=1.0
        'event_count_ratio':        (1.0 / _evt_ratio if _evt_ratio > 1 else _evt_ratio) if _evt_ratio and not np.isnan(_evt_ratio) else np.nan,
        'mean_duration_similarity': (1.0 - results['duration_metrics']['mean_duration_error']),
        'js_similarity':            (1.0 - _js_div) if not np.isnan(_js_div) else np.nan,
        'edge_f1':                  _edge_f1,
        'fitness':                  results['conformance_metrics'].get('fitness'),
        'precision':                results['conformance_metrics'].get('precision'),
        'generalization':           results['conformance_metrics'].get('generalization'),
        'simplicity':               results['conformance_metrics'].get('simplicity'),
    }

    active_values = []
    report("\nActive Score Components:")
    for comp_name, comp_val in active_components.items():
        if pd.isna(comp_val):
            report(f"  {comp_name:30}: n/a")
            continue
        comp_val = float(comp_val)
        comp_val = max(0.0, min(1.0, comp_val))
        active_values.append(comp_val)
        report(f"  {comp_name:30}: {comp_val:.4f}")

    overall_score = float(np.mean(active_values)) if active_values else np.nan
    report(f"\nOVERALL QUALITY SCORE (active): {overall_score:.4f} (0=worst, 1=perfect)")
    
    if pd.notna(overall_score) and overall_score >= 0.8:
        quality_assessment = "EXCELLENT"
    elif pd.notna(overall_score) and overall_score >= 0.6:
        quality_assessment = "GOOD"
    elif pd.notna(overall_score) and overall_score >= 0.4:
        quality_assessment = "FAIR"
    else:
        quality_assessment = "POOR"
    
    report(f"QUALITY ASSESSMENT: {quality_assessment}")
    
    results['overall_score'] = overall_score
    results['quality_assessment'] = quality_assessment
    
    return results


def plot_simulation_comparison(simulated_df, real_df, case_col='case_id', 
                             activity_col='activity', start_col='timestamp_start', 
                             end_col='timestamp_end'):
    """Create detailed comparison plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Simulation vs Real Data Comparison', fontsize=16, fontweight='bold')
    
    # 1. Activity frequencies
    sim_freq = simulated_df[activity_col].value_counts()
    real_freq = real_df[activity_col].value_counts()
    
    all_activities = list(set(sim_freq.index) | set(real_freq.index))
    sim_aligned = [sim_freq.get(act, 0) for act in all_activities]
    real_aligned = [real_freq.get(act, 0) for act in all_activities]
    
    x = np.arange(len(all_activities))
    axes[0,0].bar(x - 0.2, real_aligned, 0.4, label='Real', alpha=0.7, color='skyblue')
    axes[0,0].bar(x + 0.2, sim_aligned, 0.4, label='Simulated', alpha=0.7, color='orange')
    axes[0,0].set_title('Activity Frequencies')
    axes[0,0].set_xlabel('Activities')
    axes[0,0].set_ylabel('Count')
    axes[0,0].legend()
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # 2. Duration distributions
    sim_durations = (simulated_df[end_col] - simulated_df[start_col]).dt.total_seconds() / 60
    real_durations = (real_df[end_col] - real_df[start_col]).dt.total_seconds() / 60
    
    axes[0,1].hist(real_durations, alpha=0.7, label='Real', bins=30, density=True, color='skyblue')
    axes[0,1].hist(sim_durations, alpha=0.7, label='Simulated', bins=30, density=True, color='orange')
    axes[0,1].set_title('Activity Duration Distributions')
    axes[0,1].set_xlabel('Duration (minutes)')
    axes[0,1].set_ylabel('Density')
    axes[0,1].legend()
    
    # 3. Events per case
    sim_events_per_case = simulated_df.groupby(case_col).size()
    real_events_per_case = real_df.groupby(case_col).size()
    
    axes[1,0].hist(real_events_per_case, alpha=0.7, label='Real', bins=20, density=True, color='skyblue')
    axes[1,0].hist(sim_events_per_case, alpha=0.7, label='Simulated', bins=20, density=True, color='orange')
    axes[1,0].set_title('Events per Case Distribution')
    axes[1,0].set_xlabel('Events per Case')
    axes[1,0].set_ylabel('Density')
    axes[1,0].legend()
    
    # 4. Cumulative case duration
    sim_case_durations = simulated_df.groupby(case_col).apply(
        lambda x: (x[end_col].max() - x[start_col].min()).total_seconds() / 3600
    )
    real_case_durations = real_df.groupby(case_col).apply(
        lambda x: (x[end_col].max() - x[start_col].min()).total_seconds() / 3600
    )
    
    axes[1,1].hist(real_case_durations, alpha=0.7, label='Real', bins=20, density=True, color='skyblue')
    axes[1,1].hist(sim_case_durations, alpha=0.7, label='Simulated', bins=20, density=True, color='orange')
    axes[1,1].set_title('Case Duration Distribution')
    axes[1,1].set_xlabel('Case Duration (hours)')
    axes[1,1].set_ylabel('Density')
    axes[1,1].legend()
    
    plt.tight_layout()
    pass


def visualize_heuristic_nets(df_compare, simulated_log):
    """
    Generate and visualize heuristic nets for real and simulated data side by side.

    Parameters:
    -----------
    df_compare : pd.DataFrame
        DataFrame containing the real event log data.
    simulated_log : pd.DataFrame
        DataFrame containing the simulated event log data.
    """
    required_cols = {'case_id', 'activity', 'timestamp_start'}
    if simulated_log is None or simulated_log.empty:
        print("Skipping heuristic-net visualization: simulated log is empty.")
        return
    if not required_cols.issubset(set(simulated_log.columns)):
        print("Skipping heuristic-net visualization: simulated log is missing required columns.")
        return

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_real, \
         tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_sim:
        
        temp_real_path = temp_real.name
        temp_sim_path = temp_sim.name

    try:
        # Generate and save real data heuristic net
        event_log_real = pm4py.format_dataframe(df_compare, case_id='case_id', activity_key='activity', timestamp_key='timestamp_start')
        net_real = pm4py.discover_heuristics_net(event_log_real)
        pm4py.save_vis_heuristics_net(net_real, temp_real_path, bgcolor='white', dpi=500)
        
        # Generate and save simulated data heuristic net
        event_log_sim = pm4py.format_dataframe(simulated_log, case_id='case_id', activity_key='activity', timestamp_key='timestamp_start')
        net_sim = pm4py.discover_heuristics_net(event_log_sim)
        pm4py.save_vis_heuristics_net(net_sim, temp_sim_path, bgcolor='white', dpi=500)
        
        # Load and print side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Real data on the left
        img_real = mpimg.imread(temp_real_path)
        ax1.imshow(img_real)
        ax1.set_title('Real Data Process Model', fontsize=16, fontweight='bold')
        ax1.axis('off')
        
        # Simulated data on the right
        img_sim = mpimg.imread(temp_sim_path)
        ax2.imshow(img_sim)
        ax2.set_title('Simulated Data Process Model', fontsize=16, fontweight='bold')
        ax2.axis('off')
        
        plt.tight_layout()
        pass
        
    finally:
        # Clean up temporary files
        if os.path.exists(temp_real_path):
            os.unlink(temp_real_path)
        if os.path.exists(temp_sim_path):
            os.unlink(temp_sim_path)

##ä sim code

def _split_process_datasets(datasets, train_ratio=0.80):
    """
    Split *every* process inside ``datasets`` into two copies:
    ``train_datasets`` and ``test_datasets``.

    For each process the event_log is used to determine the temporal
    cutoff by case start time.  The same case partition is then applied
    to **production_plan** and **expanded** (using ``case_id_log``).

    Returns (train_datasets, test_datasets).
    """
    train_datasets = {}
    test_datasets  = {}

    for proc_name, proc_data in datasets.items():
        event_log        = proc_data['event_log']
        production_plan  = proc_data['production_plan']
        expanded         = proc_data.get('expanded')   # may or may not exist

        # ── determine train / test case IDs from the event log ────────────
        el = event_log.dropna(subset=['case_id']).copy()
        case_start = el.groupby('case_id')['timestamp_start'].min().sort_values()
        n_train = max(1, int(len(case_start) * train_ratio))

        train_cases = set(case_start.index[:n_train])
        test_cases  = set(case_start.index[n_train:])

        # ── event log split ───────────────────────────────────────────────
        el_train = el[el['case_id'].isin(train_cases)].copy()
        el_test  = el[el['case_id'].isin(test_cases)].copy()

        # ── production plan split ─────────────────────────────────────────
        pp_train = production_plan[production_plan['case_id'].isin(train_cases)].copy()
        pp_test  = production_plan[production_plan['case_id'].isin(test_cases)].copy()

        # If PP case_ids don't match EL case_ids (different ID schemes), split
        # the PP independently by its own temporal order.
        if len(pp_train) == 0 and len(pp_test) == 0 and len(production_plan) > 0:
            pp_case_start = (
                production_plan
                .groupby('case_id')['timestamp_start'].min()
                .sort_values()
            )
            n_pp_train = max(1, int(len(pp_case_start) * train_ratio))
            pp_train_ids = set(pp_case_start.index[:n_pp_train])
            pp_test_ids  = set(pp_case_start.index[n_pp_train:])
            pp_train = production_plan[production_plan['case_id'].isin(pp_train_ids)].copy()
            pp_test  = production_plan[production_plan['case_id'].isin(pp_test_ids)].copy()

        # ── expanded df split (uses case_id_log) ─────────────────────────
        exp_train, exp_test = None, None
        if expanded is not None:
            if 'case_id_log' in expanded.columns:
                exp_train = expanded[expanded['case_id_log'].isin(train_cases)].copy()
                exp_test  = expanded[expanded['case_id_log'].isin(test_cases)].copy()
            else:
                # fallback: use datetime_energy and the cutoff date
                cutoff = case_start.iloc[n_train - 1]
                if 'datetime_energy' in expanded.columns:
                    exp_train = expanded[expanded['datetime_energy'] <= cutoff].copy()
                    exp_test  = expanded[expanded['datetime_energy'] > cutoff].copy()
                else:
                    exp_train = expanded.copy()
                    exp_test  = expanded.iloc[0:0].copy()   # empty

        train_datasets[proc_name] = {
            'event_log': el_train,
            'production_plan': pp_train,
            'expanded': exp_train,
        }
        test_datasets[proc_name] = {
            'event_log': el_test,
            'production_plan': pp_test,
            'expanded': exp_test,
        }

        print(f"  {proc_name}: {len(train_cases)} train cases, "
              f"{len(test_cases)} test cases  |  "
              f"event_log {len(el_train)}/{len(el_test)}  |  "
              f"production_plan {len(pp_train)}/{len(pp_test)}"
              + (f"  |  expanded {len(exp_train)}/{len(exp_test)}"
                 if exp_train is not None else ""))

    return train_datasets, test_datasets


# ── Build the two dictionaries ───────────────────────────────────────────────
if TEMPORAL_SPLIT:
    print(f"\n📌 TEMPORAL SPLIT ({TRAIN_RATIO:.0%} train / "
          f"{1-TRAIN_RATIO:.0%} test) — splitting all processes:")
    train_datasets, test_datasets = _split_process_datasets(
        process_datasets_to_model, TRAIN_RATIO
    )
else:
    print("\n📌 NO SPLIT – using all data for extraction & training")
    train_datasets = process_datasets_to_model
    test_datasets  = process_datasets_to_model

all_energy_pipelines = {}

### models to compare

_filtered_modes = []
for _mode_name in MODES_TO_COMPARE:
    if _mode_name in _ENERGY_AWARE_MODES:
        _filtered_modes.append(_mode_name)   # always kept; validated at runtime
        continue
    if _mode_name.startswith('petri_net_'):
        _mode_alg = _mode_name.replace('petri_net_', '', 1).strip().lower()
        if _mode_alg == 'combined':
            _filtered_modes.append(_mode_name)
            continue
        if _mode_alg not in PETRI_NET_ALGORITHMS:
            print(
                f"⚠️ Skipping unsupported mode '{_mode_name}' "
                f"(enabled algorithms: {PETRI_NET_ALGORITHMS})"
            )
            continue
    _filtered_modes.append(_mode_name)
MODES_TO_COMPARE = _filtered_modes

# Initialize a list to store results for each process × mode
evaluation_results_list = []
_combined_sim_store = []  # stores (process, mode, sim_df, exp_df, sensors) for curve plotting

if not RUN_PROCESS_MODELLING:
    print("RUN_PROCESS_MODELLING=False — skipping process modelling loop.")

for process in process_datasets_to_model.keys() if RUN_PROCESS_MODELLING else []:
    print("\n" + "="*80)
    print(f"ANALYZING {process.upper()}")
    print("="*80)
    display(Markdown(f"# 🔍 Process: {process.upper()}"))


    # Use the pre-split dictionaries
    df_train        = train_datasets[process]['event_log']
    production_plan = train_datasets[process]['production_plan']

    # ── Extract process statistics for non-Petri modes (shared baseline) ─────
    activity_stats_df, raw_df, process_models = extract_process(
        df_train,
        mining_algorithm=MINING_ALGORITHM,
        optimize_mining_hyperparams=OPTIMIZE_MINING_HYPERPARAMS,
        mining_search_space=MINING_SEARCH_SPACE,
    )

    # ── Extract process models for each requested Petri-net algorithm ─────────
    petri_mode_algorithms = []
    for mode_name in MODES_TO_COMPARE:
        if mode_name.startswith('petri_net_'):
            mode_alg = mode_name.replace('petri_net_', '', 1).strip().lower()
            if mode_alg in PETRI_NET_ALGORITHMS:
                petri_mode_algorithms.append(mode_alg)

    # Preserve order while removing duplicates
    petri_mode_algorithms = list(dict.fromkeys(petri_mode_algorithms))

    extraction_by_algorithm = {
        MINING_ALGORITHM: {
            'activity_stats_df': activity_stats_df,
            'raw_df': raw_df,
            'process_models': process_models,
        }
    }

    for mode_alg in petri_mode_algorithms:
        if mode_alg in extraction_by_algorithm:
            continue
        pm_activity_stats_df, pm_raw_df, pm_process_models = extract_process(
            df_train,
            mining_algorithm=mode_alg,
            optimize_mining_hyperparams=OPTIMIZE_MINING_HYPERPARAMS,
            mining_search_space=MINING_SEARCH_SPACE,
        )
        extraction_by_algorithm[mode_alg] = {
            'activity_stats_df': pm_activity_stats_df,
            'raw_df': pm_raw_df,
            'process_models': pm_process_models,
        }

    # ── Visualize mined Petri nets ────────────────────────────────────────
    for alg_name, extracted in extraction_by_algorithm.items():
        models_for_alg = extracted['process_models']
        if not models_for_alg:
            continue
        print("\n" + "="*50)
        print(f"MINED PETRI NETS ({alg_name})")
        print("="*50)
        for key, model in models_for_alg.items():
            obj_name, obj_type, hla = key
            print(f"\n  Petri net: {obj_name} ({obj_type}) — {hla}")
            print(f"    Places: {len(model['net'].places)}, "
                  f"Transitions: {len(model['net'].transitions)}, "
                  f"Arcs: {len(model['net'].arcs)}")
            # Show visible transitions
            visible = [t.label for t in model['net'].transitions if t.label]
            silent = [t for t in model['net'].transitions if t.label is None]
            print(f"    Visible transitions: {visible}")
            print(f"    Silent (tau) transitions: {len(silent)}")
            if model.get('label_stochastic'):
                print(f"    Stochastic weights: {model['label_stochastic']}")
            try:
                # Add a very prominent title above the Petri net view
                title_str = f"PETRI NET VIEW: {obj_name} ({obj_type}) | Process: {process} | Alg: {alg_name}"
                report("\n" + "="*len(title_str))
                report(title_str)
                report("="*len(title_str) + "\n")
                display(Markdown(f"### 🌐 Petri Net: {obj_name} ({obj_type})"))
                display(Markdown(f"*Process: {process} | Mining Algorithm: {alg_name}*"))

                # In Jupyter, this will display the Graphviz object
                view_obj = pm4py.view_petri_net(model['net'], model['im'], model['fm'], format='png')
                if view_obj:
                    display(view_obj)

                if EXPORT_RESULTS and '_run_dir' in dir():
                    _pn_dir = os.path.join(_run_dir, 'petri_nets')
                    os.makedirs(_pn_dir, exist_ok=True)
                    _safe = lambda s: str(s).replace('/', '_').replace(' ', '_')
                    _stem = f"{process}_{alg_name}_{_safe(obj_name)}_{_safe(obj_type)}"
                    try:
                        pm4py.save_vis_petri_net(
                            model['net'], model['im'], model['fm'],
                            os.path.join(_pn_dir, f"{_stem}.png"),
                        )
                    except Exception as _ve:
                        print(f"    (PNG save failed: {_ve})")
                    try:
                        pm4py.write_pnml(
                            model['net'], model['im'], model['fm'],
                            os.path.join(_pn_dir, f"{_stem}.pnml"),
                        )
                    except Exception as _ve:
                        print(f"    (PNML save failed: {_ve})")
            except Exception as e:
                print(f"    (Could not render Petri net: {e})")

    # ── Save DFGs from training event log ────────────────────────────────────
    if EXPORT_RESULTS and '_run_dir' in dir():
        try:
            _dfg_dir = os.path.join(_run_dir, 'dfgs')
            os.makedirs(_dfg_dir, exist_ok=True)
            _train_log_pm = pm4py.format_dataframe(
                df_train.dropna(subset=['case_id', 'activity', 'timestamp_start']),
                case_id='case_id', activity_key='activity', timestamp_key='timestamp_start',
            )
            _dfg, _dfg_start, _dfg_end = pm4py.discover_dfg(_train_log_pm)
            _dfg_path = os.path.join(_dfg_dir, f"{process}_train_dfg.png")
            pm4py.save_vis_dfg(_dfg, _dfg_start, _dfg_end, _dfg_path)
            print(f"  Saved DFG → {_dfg_path}")
        except Exception as _de:
            print(f"  (DFG save failed for {process}: {_de})")

    print("\n" + "="*50)
    print("PROBABILISTIC END ACTIVITY STATS (from train set):")
    print("="*50)
    print(activity_stats_df)

    # ── Train ML models once (shared by ml-based modes) ───────────────────
    ml_models = None
    if any(m != 'statistical' for m in MODES_TO_COMPARE):
        print("\n" + "="*50)
        print(f"TRAINING ML MODELS  (types={ML_MODEL_TYPES}, "
              f"optuna={ML_OPTIMIZE_HYPERPARAMS})")
        print("="*50)
        ml_models = SimModeller(
            model_types=ML_MODEL_TYPES,
            optimize_hyperparams=ML_OPTIMIZE_HYPERPARAMS,
            n_optuna_trials=ML_OPTUNA_TRIALS,
            train_transitions=False,   # only duration for ml_duration_only
        )
        ml_models.train(raw_df, activity_stats_df)
        print(ml_models.summary())

    # ── Loop over modes ───────────────────────────────────────────────────
    process_mode_results = []

    for sim_mode in MODES_TO_COMPARE:
        if sim_mode == 'petri_net_combined':
            # Combined mode is derived after all explicit modes are evaluated.
            continue

        if sim_mode in _ENERGY_AWARE_MODES:
            # Energy-aware modes run in their own dedicated block below,
            # after the best base PN has been selected.
            continue

        print("\n" + "─"*80)
        print(f"  ▶ SIMULATION MODE: {sim_mode.upper()}")
        print("─"*80)

        mode_algorithm = None
        simulation_mode = sim_mode
        mode_activity_stats_df = activity_stats_df

        if sim_mode.startswith('petri_net_'):
            mode_algorithm = sim_mode.replace('petri_net_', '', 1).strip().lower()
            if mode_algorithm not in extraction_by_algorithm:
                print(
                    f"⚠️ Skipping unsupported Petri-net mode '{sim_mode}'. "
                    f"Expected one of: {[f'petri_net_{a}' for a in PETRI_NET_ALGORITHMS]}"
                )
                continue
            simulation_mode = 'petri_net'
            mode_activity_stats_df = extraction_by_algorithm[mode_algorithm]['activity_stats_df']

        mode_ml = ml_models if simulation_mode not in ('statistical', 'petri_net') else None
        mode_pm = extraction_by_algorithm.get(mode_algorithm, {}).get('process_models') if simulation_mode in ('petri_net', 'petri_net_statistical', 'petri_net_statistical_memory') else None

        simulated_log_train = ProcessSimulation(
            mode_activity_stats_df, production_plan,
            mode=simulation_mode, ml_models=mode_ml,
            process_models=mode_pm,
        ).run()

        print(f"\n  Simulated log TRAIN ({sim_mode}): {len(simulated_log_train)} events")

        # ══════════════════════════════════════════════════════════════════
        # EVALUATION — TRAIN SET
        # ══════════════════════════════════════════════════════════════════
        split_label = "TRAIN" if TEMPORAL_SPLIT else "ALL DATA"
        print(f"\n  🔍 EVALUATION ON {split_label}  [{sim_mode}]")
        print("  " + "="*76)

        eval_train = comprehensive_simulation_evaluation(simulated_log_train, df_train,
                                                          process_models=mode_pm)

        print(f"\n  📊 COMPARISON PLOTS ({split_label})  [{sim_mode}]")
        #plot_simulation_comparison(simulated_log_train, df_train)
        df_compare_train = df_train.dropna(subset=['case_id'])
        try:
            visualize_heuristic_nets(df_compare_train, simulated_log_train)
        except Exception as e:
            print(f"  ⚠️ Skipping Graphviz visualization (Train): {e}")

        # Flatten train results
        flattened = {
            'process': process,
            'mode': sim_mode,
            'simulation_mode': simulation_mode,
            'mining_algorithm': mode_algorithm if mode_algorithm else MINING_ALGORITHM,
            'split': split_label
        }
        for category, metrics in eval_train.items():
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    flattened[f"train_{category}_{metric_name}"] = value
            else:
                flattened[f"train_{category}"] = metrics

        # ══════════════════════════════════════════════════════════════════
        # EVALUATION — TEST SET
        # ══════════════════════════════════════════════════════════════════
        df_test = test_datasets[process]['event_log'] if test_datasets else None
        if TEMPORAL_SPLIT and df_test is not None and len(df_test) > 0:
            production_plan_test = test_datasets[process]['production_plan']
            simulated_log_test = ProcessSimulation(
                mode_activity_stats_df,
                production_plan_test,
                mode=simulation_mode,
                ml_models=mode_ml,
                process_models=mode_pm,
            ).run()

            print(f"\n  Simulated log TEST  ({sim_mode}): {len(simulated_log_test)} events")

            print(f"\n  🔍 EVALUATION ON TEST SET  [{sim_mode}]")
            print("  " + "="*76)

            eval_test = comprehensive_simulation_evaluation(simulated_log_test, df_test,
                                                              process_models=mode_pm)

            print(f"\n  📊 COMPARISON PLOTS (TEST)  [{sim_mode}]")
            #plot_simulation_comparison(simulated_log_test, df_test)
            df_compare_test = df_test.dropna(subset=['case_id'])
            try:
                visualize_heuristic_nets(df_compare_test, simulated_log_test)
            except Exception as e:
                print(f"  ⚠️ Skipping Graphviz visualization (Test): {e}")

            for category, metrics in eval_test.items():
                if isinstance(metrics, dict):
                    for metric_name, value in metrics.items():
                        flattened[f"test_{category}_{metric_name}"] = value
                else:
                    flattened[f"test_{category}"] = metrics

        process_mode_results.append(flattened)
        evaluation_results_list.append(flattened)

    # Build process-level combined Petri-net mode based on best TRAIN score.
    combined_candidate_modes = {
        'petri_net_alpha',
        'petri_net_heuristic',
        'petri_net_inductive',
    }
    combined_candidates = [
        row for row in process_mode_results
        if row.get('mode') in combined_candidate_modes
    ]

    if combined_candidates:
        # Use -inf when score is missing so valid scores are preferred.
        best_row = max(
            combined_candidates,
            key=lambda r: (
                float(r.get('train_overall_score'))
                if pd.notna(r.get('train_overall_score')) else -np.inf
            )
        )

        combined_row = dict(best_row)
        combined_row['mode'] = 'petri_net_combined'
        combined_row['selected_mode'] = best_row.get('mode')
        combined_row['selected_mining_algorithm'] = best_row.get('mining_algorithm')

        print("\n" + "─"*80)
        print("  ▶ SIMULATION MODE: PETRI_NET_COMBINED")
        print("─"*80)
        print(
            "  Selected mode for this process based on TRAIN overall score: "
            f"{combined_row['selected_mode']} "
            f"(train_overall_score={best_row.get('train_overall_score')})"
        )

        process_mode_results.append(combined_row)
        evaluation_results_list.append(combined_row)

    # ── Energy-aware Petri-net modes ─────────────────────────────────────────
    # These run AFTER all base modes (including petri_net_combined) so the
    # best base PN can be identified from the already-computed train scores.
    _energy_modes_requested = [
        m for m in MODES_TO_COMPARE if m in _ENERGY_AWARE_MODES
    ]
    all_energy_pipelines = {}  # Global store for evaluation
    VERBOSE_EVAL = False        # Set to True for detailed logs

    if _energy_modes_requested:
        # Identify best non-energy Petri-net mode by TRAIN overall score
        _base_candidate_modes = {
            'petri_net_alpha', 'petri_net_heuristic', 'petri_net_inductive',
        }
        _base_candidates = [
            row for row in process_mode_results
            if row.get('mode') in _base_candidate_modes
        ]
        if not _base_candidates:
            print(
                "⚠️  No base Petri-net modes evaluated — cannot run energy-aware modes. "
                "Add at least one of petri_net_alpha / petri_net_heuristic / "
                "petri_net_inductive to MODES_TO_COMPARE."
            )
        else:
            _best_base_row = max(
                _base_candidates,
                key=lambda r: (
                    float(r.get('train_overall_score'))
                    if pd.notna(r.get('train_overall_score')) else -np.inf
                )
            )
            _best_base_alg = _best_base_row.get('mining_algorithm')
            _best_base_pm  = extraction_by_algorithm[_best_base_alg]['process_models']
            _best_base_stats = extraction_by_algorithm[_best_base_alg]['activity_stats_df']

            print("\n" + "="*80)
            print(f"ENERGY-AWARE MODES: using '{_best_base_row['mode']}' as base PN "
                  f"(train_overall_score={_best_base_row.get('train_overall_score'):.4f})")
            print("="*80)

            # ── Extract energy modifiers once per process ─────────────────
            _df_expanded_train = train_datasets[process].get('expanded')
            if _df_expanded_train is None or _df_expanded_train.empty:
                print("⚠️  No expanded training df available — skipping energy modifiers.")
            else:
                # Detect external-factor columns (ef_*) from expanded training df
                _ef_ep_cols = [
                    c for c in _df_expanded_train.columns
                    if c.startswith('ef_')
                    and _df_expanded_train[c].dtype in ('float64', 'float32', 'int64', 'int32')
                ]
                if _ef_ep_cols:
                    print(f"  ℹ️  Energy models will use {len(_ef_ep_cols)} external factor(s): {_ef_ep_cols}")
                # Auto-detect sensor columns: any *_energy column that is not a log column.
                # If process_datasets_to_model_sensors is defined and has sensors_to_model,
                # use that curated list instead (it is defined later in the energy-modelling section).
                _sensors_from_config = (
                    process_datasets_to_model_sensors  # noqa: F821
                    .get(process, {})
                    .get('sensors_to_model', [])
                ) if 'process_datasets_to_model_sensors' in dir() else []

                if _sensors_from_config:
                    _sensors = _sensors_from_config
                else:
                    # Fall back: all *_energy columns that are numeric, not log columns,
                    # and not external-factor columns (ef_* are inputs, not prediction targets)
                    _sensors = [
                        c for c in _df_expanded_train.columns
                        if c.endswith('_energy')
                        and not c.endswith('_log')
                        and not c.startswith('ef_')
                        and _df_expanded_train[c].dtype in ('float64', 'float32', 'int64', 'int32')
                    ]
                    if _sensors:
                        print(f"  ℹ️  Auto-detected {len(_sensors)} sensor column(s): {_sensors}")

                if not _sensors:
                    print("⚠️  No sensors found for this process — skipping energy modifiers.")
                    _activity_exog_means = {}
                else:
                    # Initialise direct-model dicts so they're always defined in scope
                    _energy_direct_dur_mods = {}
                    _energy_direct_tr_mods  = {}
                    try:
                        _energy_dur_mods, _energy_tr_mods, _energy_state_cols, _model_choices_report = \
                            extract_energy_modifiers(
                                df_expanded=_df_expanded_train,
                                sensors=_sensors,
                                duration_models=ENERGY_DURATION_MODELS,
                                transition_models=ENERGY_TRANSITION_MODELS,
                                min_samples=ENERGY_MIN_SAMPLES,
                                ef_cols=_ef_ep_cols,
                            )
                        energy_modifiers_by_process[process] = {
                            'duration':    _energy_dur_mods,
                            'transition':  _energy_tr_mods,
                            'columns':     _energy_state_cols,
                            'report':      _model_choices_report
                        }

                        # Print Model Choices Tracking Report
                        if _model_choices_report:
                            report("\n" + "="*80)
                            report(f"ENERGY MODIFIER APPROACH TRACKING | Process: {process}")
                            report("="*80)
                            _choices_df = pd.DataFrame.from_dict(_model_choices_report, orient='index').reset_index()
                            _choices_df.rename(columns={'index': 'Subprocess (Activity)'}, inplace=True)
                            _choices_df.insert(0, 'Dataset/Process', process)
                            report(_choices_df.to_string(index=False))
                            display(_choices_df)

                        # ── Train direct ML models (if any direct modes requested) ──
                        _direct_modes_requested = [
                            m for m in _energy_modes_requested if m in _ENERGY_DIRECT_MODES
                        ]
                        if _direct_modes_requested:
                            _energy_direct_dur_mods, _energy_direct_tr_mods, _, _direct_report = \
                                extract_energy_direct_models(
                                    df_expanded=_df_expanded_train,
                                    sensors=_sensors,
                                    duration_models=ENERGY_DURATION_MODELS,
                                    transition_models=ENERGY_TRANSITION_MODELS,
                                    min_samples=ENERGY_MIN_SAMPLES,
                                    ef_cols=_ef_ep_cols,
                                )
                            if _direct_report:
                                report("\n" + "="*80)
                                report(f"ENERGY DIRECT MODEL APPROACH TRACKING | Process: {process}")
                                report("="*80)
                                _direct_df = pd.DataFrame.from_dict(_direct_report, orient='index').reset_index()
                                _direct_df.rename(columns={'index': 'Subprocess (Activity)'}, inplace=True)
                                _direct_df.insert(0, 'Dataset/Process', process)
                                report(_direct_df.to_string(index=False))
                                display(_direct_df)

                        # ── Train Dynamic ML Curve Predictors ──────────────────
                        # One pipeline per (sensor, activity, object) so each barycenter
                        # and model is fit on a homogeneous set of curves.
                        # Only run this if we actually want to evaluate on Test results
                        # as this DTW-based training is the slowest part of the pipeline.
                        if RUN_TEST_EVALUATION:
                            from sim_extractor import (predict_raw_curve, predict_raw_curve_exog,
                                                       _train_energy_pipeline_worker)
                            import concurrent.futures, os

                            _energy_pipelines = {}

                            _config = process_datasets_to_model_sensors.get(process, {}) if 'process_datasets_to_model_sensors' in dir() else {}
                            _activities_list = _config.get('activities_to_model', _df_expanded_train['activity_log'].dropna().unique().tolist())
                            _objects_list    = _config.get('objects_to_model',    _df_expanded_train['object_log'].dropna().unique().tolist())

                            def _make_predict_fn(ep_bound):
                                _ep_exog = ep_bound.get('exog_cols', [])
                                if _ep_exog:
                                    return (lambda ep: lambda raw_values, activity, object_attributes, exog=None:
                                        predict_raw_curve_exog(raw_values, activity, object_attributes,
                                                               pipeline=ep, exog_values=exog or {})
                                    )(ep_bound)
                                return (lambda ep: lambda raw_values, activity, object_attributes, exog=None:
                                    predict_raw_curve(raw_values, activity, object_attributes, pipeline=ep)
                                )(ep_bound)

                            # Pre-compute per-activity exog stats from training data.
                            # Raw key (ef_col) retained for curve-prediction pipelines that
                            # still use exog_cols with raw names.  Expanded keys (_mean/_end/_std)
                            # are required by energy_state_columns after the ef_* expansion.
                            _activity_exog_means = {}
                            if _ef_ep_cols and 'activity_log' in _df_expanded_train.columns:
                                for _act in _activities_list:
                                    _act_rows = _df_expanded_train[_df_expanded_train['activity_log'] == _act]
                                    if len(_act_rows) > 0:
                                        _ef_entry = {}
                                        for col in _ef_ep_cols:
                                            if col not in _act_rows.columns or _act_rows[col].isna().all():
                                                continue
                                            _vals = _act_rows[col].dropna()
                                            _mean = float(_vals.mean())
                                            _std  = float(_vals.std()) if len(_vals) > 1 else 0.0
                                            _ef_entry[col]              = _mean  # raw key for curve pipelines
                                            _ef_entry[f'{col}_mean']    = _mean
                                            _ef_entry[f'{col}_end']     = _mean  # best proxy at sim time
                                            _ef_entry[f'{col}_std']     = _std
                                        _activity_exog_means[_act] = _ef_entry

                            _combos = [
                                (s, a, o)
                                for s in _sensors
                                for a in _activities_list
                                for o in _objects_list
                            ]
                            _n_workers = min(len(_combos), os.cpu_count() or 4)
                            print(f"\n  ℹ️ Training {len(_combos)} pipelines across {_n_workers} workers"
                                  f" ({'with' if _ef_ep_cols else 'without'} external factors)...")
                            print(f"     sensors={_sensors}")
                            print(f"     activities={_activities_list}")
                            print(f"     objects={_objects_list}")

                            # ProcessPoolExecutor works from Jupyter notebooks; joblib/loky does not
                            # reliably spawn from an interactive kernel on Linux.
                            _results = []
                            with concurrent.futures.ProcessPoolExecutor(max_workers=_n_workers) as _ctx:
                                _futures = {
                                    _ctx.submit(_train_energy_pipeline_worker, s, a, o,
                                                _df_expanded_train, 1, _ef_ep_cols): (s, a, o)
                                    for s, a, o in _combos
                                }
                                for _fut in concurrent.futures.as_completed(_futures):
                                    try:
                                        _results.append(_fut.result())
                                    except Exception as _e:
                                        s, a, o = _futures[_fut]
                                        print(f"    ⚠️  Worker failed {s}|{a}|{o}: {_e}")

                            for _sensor, _activity, _object, _ep_pipeline in _results:
                                if _ep_pipeline is None:
                                    print(f"    ⚠️  Skipped {_sensor} | {_activity} | {_object} (too few curves).")
                                    continue
                                _energy_pipelines.setdefault(_sensor, {}).setdefault(_activity, {})[_object] = {
                                    'reference_curve': _ep_pipeline['reference_curve'],
                                    'predict_fn':      _make_predict_fn(_ep_pipeline),
                                    'exog_cols':       _ep_pipeline.get('exog_cols', []),
                                    'full_pipeline':   _ep_pipeline,
                                }

                            all_energy_pipelines[process] = _energy_pipelines
                        else:
                            _energy_pipelines = {}
                            _activity_exog_means = {}

                    except Exception as _exc:
                        print(f"⚠️  extract_energy_modifiers or ML curve modeling failed: {_exc}")
                        _energy_dur_mods, _energy_tr_mods, _energy_state_cols = {}, {}, []
                        _energy_pipelines = {}
                        _activity_exog_means = {}
                        _config = process_datasets_to_model_sensors.get(process, {}) if 'process_datasets_to_model_sensors' in dir() else {}
                        _activities_list = _config.get(
                            'activities_to_model',
                            _df_expanded_train['activity_log'].dropna().unique().tolist()
                            if _df_expanded_train is not None and 'activity_log' in _df_expanded_train.columns
                            else []
                        )

                # ── Simulate energy-aware modes ───────────────────────────
                for _energy_mode in _energy_modes_requested:
                    print("\n" + "─"*80)
                    print(f"  ▶ SIMULATION MODE: {_energy_mode.upper()}")
                    print("─"*80)

                    def _build_activity_exog_means(df_expanded, activities, ef_cols):
                        """Compute per-activity ef_* stats from any expanded dataframe."""
                        means = {}
                        if not ef_cols or 'activity_log' not in df_expanded.columns:
                            return means
                        for act in activities:
                            act_rows = df_expanded[df_expanded['activity_log'] == act]
                            if len(act_rows) == 0:
                                continue
                            entry = {}
                            for col in ef_cols:
                                if col not in act_rows.columns or act_rows[col].isna().all():
                                    continue
                                vals = act_rows[col].dropna()
                                mean = float(vals.mean())
                                std  = float(vals.std()) if len(vals) > 1 else 0.0
                                entry[col]           = mean   # raw key for curve pipelines
                                entry[f'{col}_mean'] = mean
                                entry[f'{col}_end']  = mean   # best proxy at sim time
                                entry[f'{col}_std']  = std
                            means[act] = entry
                        return means

                    def _run_energy_sim(plan, stats_df, pm, exog_means=None):
                        # Direct modes use the direct ML models; modifier modes use the modifier models
                        _is_direct = _energy_mode in _ENERGY_DIRECT_MODES
                        _dur_mods  = _energy_direct_dur_mods if _is_direct else _energy_dur_mods
                        _tr_mods   = _energy_direct_tr_mods  if _is_direct else _energy_tr_mods
                        return ProcessSimulation(
                            stats_df, plan,
                            mode=_energy_mode,
                            base_simulation_mode=SIMULATION_MODE,
                            ml_models=ml_models,
                            process_models=pm,
                            energy_duration_modifiers=_dur_mods,
                            energy_transition_modifiers=_tr_mods,
                            energy_state_columns=_energy_state_cols,
                            energy_pipelines=_energy_pipelines,
                            activity_exog_means=exog_means if exog_means is not None else _activity_exog_means,
                            duration_scale_clip=ENERGY_DURATION_SCALE_CLIP,
                            logit_bias_clip=ENERGY_LOGIT_BIAS_CLIP,
                            temporal_resolution_minutes=TEMPORAL_RESOLUTION_MINUTES,
                            verbose=VERBOSE_EVAL,
                        ).run()

                    _energy_sim_train = _run_energy_sim(
                        production_plan, _best_base_stats, _best_base_pm
                    )
                    if VERBOSE_EVAL:
                        print(f"\n  Simulated log TRAIN ({_energy_mode}): "
                              f"{len(_energy_sim_train)} events")

                    # Use _df_expanded_train for energy comparison in comprehensive_simulation_evaluation
                    _eval_train = comprehensive_simulation_evaluation(
                        _energy_sim_train, df_train, real_expanded_df=_df_expanded_train,
                        process_models=_best_base_pm)

                    _energy_flattened = {
                        'process':           process,
                        'mode':              _energy_mode,
                        'simulation_mode':   _energy_mode,
                        'mining_algorithm':  _best_base_alg,
                        'split':             split_label,
                        'selected_mode':     _best_base_row['mode'],
                    }
                    for _cat, _met in _eval_train.items():
                        if _cat == 'energy_metrics' and isinstance(_met, dict):
                            # Special handling to flatten nested energy metrics
                            for _sensor, _vals in _met.items():
                                for _mn, _mv in _vals.items():
                                    _energy_flattened[f"train_energy_{_sensor}_{_mn}"] = _mv
                        elif isinstance(_met, dict):
                            for _mn, _mv in _met.items():
                                _energy_flattened[f"train_{_cat}_{_mn}"] = _mv
                        else:
                            _energy_flattened[f"train_{_cat}"] = _met

                    # ── Test evaluation (Guarded for speed) ────────────────
                    if RUN_TEST_EVALUATION:
                        _df_test = test_datasets[process]['event_log'] if test_datasets else None
                        if TEMPORAL_SPLIT and _df_test is not None and len(_df_test) > 0:
                            _pp_test  = test_datasets[process]['production_plan']
                            _exp_test = test_datasets[process]['expanded']
                            # ef_* are real observable external factors — use actual test-set
                            # values instead of training-time means so the model sees real
                            # conditions, not a constant proxy.
                            _exog_means_test = _build_activity_exog_means(
                                _exp_test, _activities_list, _ef_ep_cols
                            ) if _ef_ep_cols else _activity_exog_means
                            _energy_sim_test = _run_energy_sim(
                                _pp_test, _best_base_stats, _best_base_pm,
                                exog_means=_exog_means_test,
                            )
                            if VERBOSE_EVAL:
                                print(f"\n  Simulated log TEST  ({_energy_mode}): "
                                      f"{len(_energy_sim_test)} events")

                            _eval_test = comprehensive_simulation_evaluation(
                                _energy_sim_test, _df_test, real_expanded_df=_exp_test,
                                process_models=_best_base_pm)
                            for _cat, _met in _eval_test.items():
                                if _cat == 'energy_metrics' and isinstance(_met, dict):
                                    for _sensor, _vals in _met.items():
                                        for _mn, _mv in _vals.items():
                                            _energy_flattened[f"test_energy_{_sensor}_{_mn}"] = _mv
                                elif _cat == 'activity_energy_metrics':
                                    pass  # kept in _eval_test for plotting; not flattened into wide df
                                elif isinstance(_met, dict):
                                    for _mn, _mv in _met.items():
                                        _energy_flattened[f"test_{_cat}_{_mn}"] = _mv
                                else:
                                    _energy_flattened[f"test_{_cat}"] = _met

                            # Store sim df for later curve plotting
                            _combined_sim_store.append({
                                'process':  process,
                                'mode':     _energy_mode,
                                'sim_df':   _energy_sim_test,
                                'exp_df':   _exp_test,
                                'sensors':  _sensors,
                                'act_metrics': _eval_test.get('activity_energy_metrics', {}),
                            })

                    process_mode_results.append(_energy_flattened)
                    evaluation_results_list.append(_energy_flattened)


        # ── Intermediate Per-Process Training Heatmap ─────────────────────
        if process_mode_results:
            _proc_df = pd.DataFrame(process_mode_results)
            
            # Specifically filter for the CORE metrics the user wants to see
            _train_cols = []
            for _base in CORE_METRIC_BASES:
                _full = f"train_{_base}"
                if _full in _proc_df.columns:
                    _train_cols.append(_full)
            
            # If no core metrics found, fall back to any training metric (fast fallback)
            if not _train_cols:
                _train_cols = [c for c in _proc_df.columns if c.startswith('train_') and not c.startswith('train_energy_')]
            
            display(Markdown(f"## 📊 Training Verification: {process.upper()}"))
            display(Markdown(f"*Evaluation on training data using real energy curves (verification of modifier fitting)*"))
            _hm_train_mean_path = os.path.join(_process_results_dir, f'process_train_heatmap_{process}_mean.png') if EXPORT_RESULTS and '_process_results_dir' in dir() else None
            _hm_train_median_path = os.path.join(_process_results_dir, f'process_train_heatmap_{process}_median.png') if EXPORT_RESULTS and '_process_results_dir' in dir() else None
            _plot_results_heatmap(_train_cols, f"Training Quality (Mean): {process}", local_df=_proc_df, save_path=_hm_train_mean_path, agg='mean')
            _plot_results_heatmap(_train_cols, f"Training Quality (Median): {process}", local_df=_proc_df, save_path=_hm_train_median_path, agg='median')


if RUN_PROCESS_MODELLING:
    # Convert the results list into a DataFrame
    evaluation_results_df = pd.DataFrame(evaluation_results_list)

    # Reorder columns to place key columns first
    priority_cols = ['process', 'mode', 'split']
    for prefix in ['train', 'test']:
        for col_name in ['overall_score', 'quality_assessment']:
            full = f"{prefix}_{col_name}"
            if full in evaluation_results_df.columns:
                priority_cols.append(full)
    remaining_cols = [c for c in evaluation_results_df.columns if c not in priority_cols]
    evaluation_results_df = evaluation_results_df[priority_cols + remaining_cols]

    # print the DataFrame
    print("\n" + "="*80)
    print("AGGREGATED EVALUATION RESULTS — MODE COMPARISON")
    print("="*80)

    if EXPORT_RESULTS and '_run_dir' in dir():
        _pm_full_path = os.path.join(_run_dir, 'process_eval_results.parquet')
        evaluation_results_df.to_parquet(_pm_full_path, index=False)
        print(f"Saved process eval results → {_pm_full_path}")

        _core_prefixes = ['process', 'mode', 'split']
        for _pfx in ['train', 'test']:
            for _base in CORE_METRIC_BASES:
                _core_prefixes.append(f'{_pfx}_{_base}')
        _core_cols = [c for c in _core_prefixes if c in evaluation_results_df.columns]
        _pm_core_path = os.path.join(_run_dir, 'process_eval_core_metrics.parquet')
        evaluation_results_df[_core_cols].to_parquet(_pm_core_path, index=False)
        print(f"Saved process core metrics  → {_pm_core_path}")
    evaluation_results_df

    # ── Combined Training Heatmap — ALL processes together ────────────────────
    _all_train_cols = [f"train_{b}" for b in CORE_METRIC_BASES if f"train_{b}" in evaluation_results_df.columns]
    if not _all_train_cols:
        _all_train_cols = [c for c in evaluation_results_df.columns if c.startswith('train_') and not c.startswith('train_energy_')]
    if _all_train_cols:
        display(Markdown("---"))
        display(Markdown("## 📊 Training Quality: ALL Processes Combined"))
        display(Markdown("*Aggregated across all processes on training data.*"))
        _hm_train_all_mean_path = os.path.join(_process_results_dir, 'process_train_heatmap_all_mean.png') if EXPORT_RESULTS and '_process_results_dir' in dir() else None
        _hm_train_all_median_path = os.path.join(_process_results_dir, 'process_train_heatmap_all_median.png') if EXPORT_RESULTS and '_process_results_dir' in dir() else None
        _plot_results_heatmap(_all_train_cols, "Training Quality (Mean): All Processes", save_path=_hm_train_all_mean_path, agg='mean')
        _plot_results_heatmap(_all_train_cols, "Training Quality (Median): All Processes", save_path=_hm_train_all_median_path, agg='median')

    # ── Per-process breakdown: show modes sorted by test_overall_score ────────
    report("\n" + "="*80)
    report("PER-PROCESS RESULTS — MODES SORTED BY test_overall_score")
    report("="*80)

    sort_col = 'test_overall_score'
    if sort_col in evaluation_results_df.columns:
        # Columns to display (key columns only for readability)
        display_cols = ['process', 'mode', 'split']
        for prefix in ['test', 'train']:
            for col_name in ['overall_score', 'quality_assessment']:
                full = f"{prefix}_{col_name}"
                if full in evaluation_results_df.columns:
                    display_cols.append(full)
        # Add all test_ metric columns for full visibility
        test_metric_cols = [c for c in evaluation_results_df.columns
                            if c.startswith('test_') and c not in display_cols]
        display_cols.extend(test_metric_cols)
        display_cols = [c for c in display_cols if c in evaluation_results_df.columns]

        for process_name, grp in evaluation_results_df.groupby('process'):
            report(f"\n{'─'*80}")
            report(f"  PROCESS: {process_name}")
            report(f"{'─'*80}")
            sorted_grp = grp.sort_values(sort_col, ascending=False)
            # Pretty-print with pandas
            with pd.option_context('display.max_columns', None,
                                   'display.width', 200,
                                   'display.max_colwidth', 30):
                report(sorted_grp[display_cols].to_string(index=False))
    else:
        report(f"  Column '{sort_col}' not found — skipping per-process ranking.")
        print(f"  Available columns: {list(evaluation_results_df.columns)}")

# %% 

# %% [markdown]
# # Data fusion

# %% 
# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────
# All modeling and per-process evaluations are complete.

if RUN_TEST_EVALUATION and RUN_PROCESS_MODELLING:
    # Final consolidated summary of TEST set performance across ALL processes
    # (Focuses strictly on the core metrics to maintain clarity)
    _final_test_cols = [f"test_{b}" for b in CORE_METRIC_BASES if f"test_{b}" in evaluation_results_df.columns]
    if _final_test_cols:
        display(Markdown("---"))
        display(Markdown("# 📊 FINAL CONSOLIDATED PERFORMANCE: TEST SET ENSEMBLE"))
        display(Markdown("*Consolidated simulation quality across all processes on unseen data.*"))
        _hm_path_mean = os.path.join(_process_results_dir, 'process_test_heatmap_mean.png') if EXPORT_RESULTS and '_process_results_dir' in dir() else None
        _hm_path_median = os.path.join(_process_results_dir, 'process_test_heatmap_median.png') if EXPORT_RESULTS and '_process_results_dir' in dir() else None
        _plot_results_heatmap(_final_test_cols, "Generalization Performance: Test Set Ensemble (Mean)", save_path=_hm_path_mean, agg='mean')
        _plot_results_heatmap(_final_test_cols, "Generalization Performance: Test Set Ensemble (Median)", save_path=_hm_path_median, agg='median')

        # ── Per-process test heatmaps ──────────────────────────────────────────
        display(Markdown("## 📊 Test Performance: Per Process"))
        for _proc_name, _proc_grp in evaluation_results_df.groupby('process'):
            _proc_test_cols = [c for c in _final_test_cols if c in _proc_grp.columns]
            if not _proc_test_cols:
                continue
            display(Markdown(f"### {_proc_name.upper()}"))
            _hm_proc_mean = os.path.join(_process_results_dir, f'process_test_heatmap_{_proc_name}_mean.png') if EXPORT_RESULTS and '_process_results_dir' in dir() else None
            _hm_proc_median = os.path.join(_process_results_dir, f'process_test_heatmap_{_proc_name}_median.png') if EXPORT_RESULTS and '_process_results_dir' in dir() else None
            _plot_results_heatmap(_proc_test_cols, f"Test Performance (Mean): {_proc_name}", local_df=_proc_grp, save_path=_hm_proc_mean, agg='mean')
            _plot_results_heatmap(_proc_test_cols, f"Test Performance (Median): {_proc_name}", local_df=_proc_grp, save_path=_hm_proc_median, agg='median')



# %%
# ── CURVE-ONLY: train ALL approaches without process modelling ────────────────
# Trains baseline, Approach 2 (B-spline basis), and Approach 3 (DTW-phase)
# side by side so results can be compared in the evaluation cell below.
if RUN_CURVE_ONLY_EVALUATION:
    from sim_extractor import (
        split_curves,
        split_curves_with_prev_activity,
        build_and_train_pipeline,            predict_raw_curve,
        build_and_train_pipeline_instance_stats, predict_raw_curve_instance_stats,
        build_and_train_pipeline_istats_leakfree, predict_raw_curve_istats_leakfree,
        build_and_train_pipeline_dtw_phase,  predict_raw_curve_dtw_phase,
        build_and_train_pipeline_basis,      predict_raw_curve_basis,
        build_and_train_pipeline_exog,            predict_raw_curve_exog,
        build_and_train_pipeline_exog_prev_activity, predict_raw_curve_exog_prev_activity,
        build_and_train_pipeline_amplitude_shape, predict_raw_curve_amplitude_shape,
        # amplitude_shape_exog removed
        build_and_train_pipeline_seq2seq,         predict_raw_curve_seq2seq,
        build_and_train_pipeline_seq2seq_only,    predict_raw_curve_seq2seq_only,
        build_and_train_pipeline_seq2seq_exog,    predict_raw_curve_seq2seq_exog,
        build_and_train_pipeline_seq2seq_prev_activity, predict_raw_curve_seq2seq_prev_activity,
    )
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import GradientBoostingRegressor

    all_energy_pipelines                  = {}   # baseline
    all_energy_pipelines_mean            = {}   # mean baseline
    all_energy_pipelines_instance_stats  = {}   # Instance Stats (leaky)
    all_energy_pipelines_istats_leakfree = {}   # Instance Stats (leak-free)
    all_energy_pipelines_dtw_phase       = {}   # Approach 3
    all_energy_pipelines_basis           = {}   # Approach 2
    all_energy_pipelines_exog            = {}   # DTW + External Factors
    all_energy_pipelines_exog_prev_activity   = {}   # DTW + Ext. Factors + Prev Activity
    all_energy_pipelines_amplitude_shape      = {}   # Amplitude + Shape
    all_energy_pipelines_amplitude_shape_exog = {}   # removed — kept as empty for safety
    all_energy_pipelines_seq2seq              = {}   # DTW + Seq2Seq
    all_energy_pipelines_seq2seq_only         = {}   # Seq2Seq only (no DTW)
    all_energy_pipelines_seq2seq_exog         = {}   # DTW + Seq2Seq + Ext. Factors
    all_energy_pipelines_seq2seq_prev_activity = {}  # DTW + Seq2Seq + Ext. Factors + Prev Act

    _CURVE_MODELS = {
        'Linear Regression': LinearRegression,
        'Gradient Boosting': GradientBoostingRegressor,
    }

    for _proc in process_datasets_to_model.keys():
        _proc_cfg   = process_datasets_to_model_sensors.get(_proc, {}) \
                      if 'process_datasets_to_model_sensors' in dir() else {}
        _sensors    = _proc_cfg.get('sensors_to_model', [])
        _activities = _proc_cfg.get('activities_to_model', [])
        _objects    = _proc_cfg.get('objects_to_model', [])

        _df_train_exp = train_datasets[_proc].get('expanded')
        if _df_train_exp is None or _df_train_exp.empty:
            print(f"  Skipping {_proc}: no expanded data.")
            continue

        if not _sensors:
            _sensors = [
                c for c in _df_train_exp.columns
                if c.endswith('_energy')
                and not c.endswith('_log')
                and _df_train_exp[c].dtype in ('float64', 'float32', 'int64', 'int32')
            ]
        if not _sensors:
            print(f"  Skipping {_proc}: no sensor columns found.")
            continue

        if not _activities and 'activity_log' in _df_train_exp.columns:
            _activities = _df_train_exp['activity_log'].dropna().unique().tolist()
        if not _objects and 'object_log' in _df_train_exp.columns:
            _objects = _df_train_exp['object_log'].dropna().unique().tolist()

        # Detect ef_* columns — external factors are always predictors of every curve
        _ef_cols = [
            c for c in _df_train_exp.columns
            if c.startswith('ef_')
            and _df_train_exp[c].dtype in ('float64', 'float32', 'int64', 'int32')
        ]

        print(f"\n{'='*60}")
        print(f"CURVE-ONLY TRAINING — {_proc.upper()}")
        print(f"  Sensors          : {_sensors}")
        print(f"  Activities       : {_activities}")
        print(f"  External factors : {_ef_cols}")
        print(f"{'='*60}")

        _pipelines_baseline            = {}
        _pipelines_instance_stats      = {}
        _pipelines_istats_leakfree     = {}
        _pipelines_dtw_phase           = {}
        _pipelines_basis               = {}
        _pipelines_exog                = {}
        _pipelines_exog_prev_activity   = {}
        _pipelines_amplitude_shape      = {}
        _pipelines_amplitude_shape_exog = {}  # removed
        _pipelines_seq2seq                  = {}
        _pipelines_seq2seq_only             = {}
        _pipelines_seq2seq_exog             = {}
        _pipelines_seq2seq_prev_activity    = {}

        # ── Parallel training for all sklearn-based approaches ───────────────
        # One worker per (sensor, activity, object) combo — each trains its own
        # barycenter and model on a homogeneous set of curves.
        # Seq2seq approaches use PyTorch and run sequentially afterwards.
        from sim_extractor import _train_curve_only_worker
        import concurrent.futures, os as _os

        _sklearn_approaches = [a for a in APPROACHES
                               if a in {'baseline','instance_stats','istats_leakfree','dtw_phase','basis','exog','exog_prev_activity','amplitude_shape'}]
        _seq2seq_approaches = [a for a in APPROACHES
                               if a in {'seq2seq','seq2seq_only','seq2seq_exog','seq2seq_prev_activity'}]

        _combos    = [(s, a, o) for s in _sensors for a in _activities for o in _objects]
        _n_workers = min(len(_combos), _os.cpu_count() or 4)

        if _sklearn_approaches and _combos:
            print(f"\n  Parallel sklearn training: {len(_combos)} combos × {len(_sklearn_approaches)} approaches "
                  f"across {_n_workers} workers...")
            _sklearn_t0 = _time.perf_counter()
            with concurrent.futures.ProcessPoolExecutor(max_workers=_n_workers) as _pool:
                _futs = {
                    _pool.submit(_train_curve_only_worker,
                                 s, a, o, _df_train_exp, _sklearn_approaches, _ef_cols,
                                 100, 0.2, CURVE_OPTIMIZE_HYPERPARAMS, CURVE_N_OPTUNA_TRIALS): (s, a, o)
                    for s, a, o in _combos
                }
                _worker_results = []
                for _fut in concurrent.futures.as_completed(_futs):
                    try:
                        _worker_results.append(_fut.result())
                    except Exception as _we:
                        _ws, _wa, _wo = _futs[_fut]
                        print(f"  ⚠️  Worker failed {_ws}|{_wa}|{_wo}: {_we}")
            print(f"  sklearn training done in {_time.perf_counter() - _sklearn_t0:.1f}s")

            # Reassemble into per-sensor dicts keyed [sensor][activity][object]
            for _r in _worker_results:
                _s, _a, _o = _r['sensor'], _r['activity'], _r['object']
                if _r.get('skipped'):
                    print(f"  ⚠️  Skipped {_s}|{_a}|{_o} (too few curves).")
                    continue
                if 'baseline' in _r:
                    _pipelines_baseline.setdefault(_s, {}).setdefault(_a, {})[_o] = {
                        'reference_curve': _r['baseline']['reference_curve'],
                        'predict_fn':      lambda rv, act, attrs, ep=_r['baseline']: predict_raw_curve(rv, act, attrs, pipeline=ep),
                        'full_pipeline':   _r['baseline'],
                    }
                if 'instance_stats' in _r:
                    _pipelines_instance_stats.setdefault(_s, {}).setdefault(_a, {})[_o] = {
                        'reference_curve': _r['instance_stats']['reference_curve'],
                        'predict_fn':      lambda rv, act, attrs, ep=_r['instance_stats']: predict_raw_curve_instance_stats(rv, act, attrs, pipeline=ep),
                        'full_pipeline':   _r['instance_stats'],
                    }
                if 'istats_leakfree' in _r:
                    _pipelines_istats_leakfree.setdefault(_s, {}).setdefault(_a, {})[_o] = {
                        'reference_curve': _r['istats_leakfree']['reference_curve'],
                        'predict_fn':      lambda rv, act, attrs, ep=_r['istats_leakfree']: predict_raw_curve_istats_leakfree(rv, act, attrs, pipeline=ep),
                        'full_pipeline':   _r['istats_leakfree'],
                    }
                if 'dtw_phase' in _r:
                    _pipelines_dtw_phase.setdefault(_s, {}).setdefault(_a, {})[_o] = {
                        'reference_curve': _r['dtw_phase']['reference_curve'],
                        'predict_fn':      lambda rv, act, attrs, ep=_r['dtw_phase']: predict_raw_curve_dtw_phase(rv, act, attrs, pipeline=ep),
                        'full_pipeline':   _r['dtw_phase'],
                    }
                if 'basis' in _r:
                    _pipelines_basis.setdefault(_s, {}).setdefault(_a, {})[_o] = {
                        'reference_curve': _r['basis']['reference_curve'],
                        'predict_fn':      lambda rv, act, attrs, ep=_r['basis']: predict_raw_curve_basis(rv, act, attrs, pipeline=ep),
                        'full_pipeline':   _r['basis'],
                    }
                if 'exog' in _r:
                    _pipelines_exog.setdefault(_s, {}).setdefault(_a, {})[_o] = {
                        'reference_curve': _r['exog']['reference_curve'],
                        'predict_fn':      lambda rv, act, attrs, exog=None, ep=_r['exog']: predict_raw_curve_exog(rv, act, attrs, pipeline=ep, exog_values=exog or {}),
                        'full_pipeline':   _r['exog'],
                    }
                if 'amplitude_shape' in _r:
                    _pipelines_amplitude_shape.setdefault(_s, {}).setdefault(_a, {})[_o] = {
                        'reference_curve': _r['amplitude_shape']['reference_curve'],
                        'predict_fn':      (lambda ep: lambda rv, act, attrs: predict_raw_curve_amplitude_shape(rv, act, attrs, pipeline=ep))(_r['amplitude_shape']),
                        'full_pipeline':   _r['amplitude_shape'],
                    }
                if 'exog_prev_activity' in _r:
                    _pipelines_exog_prev_activity.setdefault(_s, {}).setdefault(_a, {})[_o] = {
                        'reference_curve': _r['exog_prev_activity']['reference_curve'],
                        'predict_fn':      (lambda ep: lambda rv, act, attrs, exog=None: predict_raw_curve_exog_prev_activity(rv, act, attrs, pipeline=ep, exog_values=exog or {}))(_r['exog_prev_activity']),
                        'full_pipeline':   _r['exog_prev_activity'],
                    }

        # ── Seq2seq approaches — one worker per (sensor, activity, object) combo ──
        if _seq2seq_approaches and _combos:
            from sim_extractor import _train_seq2seq_worker
            _s2s_n_workers = min(len(_combos), _os.cpu_count() or 4)
            print(f"\n  Parallel seq2seq training: {len(_combos)} combos × "
                  f"{len(_seq2seq_approaches)} approaches across {_s2s_n_workers} workers...")
            _s2s_t0 = _time.perf_counter()
            with concurrent.futures.ProcessPoolExecutor(max_workers=_s2s_n_workers) as _s2s_pool:
                _s2s_futs = {
                    _s2s_pool.submit(
                        _train_seq2seq_worker,
                        _s, _a, _o, _df_train_exp, _seq2seq_approaches, _ef_cols,
                        SEQ2SEQ_HIDDEN_SIZE, SEQ2SEQ_NUM_LAYERS, SEQ2SEQ_DROPOUT,
                        SEQ2SEQ_EPOCHS, SEQ2SEQ_BATCH_SIZE, SEQ2SEQ_LR,
                        SEQ2SEQ_TEACHER_FORCING, SEQ2SEQ_PATIENCE,
                    ): (_s, _a, _o, _time.perf_counter())
                    for _s, _a, _o in _combos
                }
                _s2s_results = []
                for _s2s_fut in concurrent.futures.as_completed(_s2s_futs):
                    _s, _a, _o, _s_t0 = _s2s_futs[_s2s_fut]
                    try:
                        _res = _s2s_fut.result()
                        _res['_elapsed'] = _time.perf_counter() - _s_t0
                        _s2s_results.append(_res)
                    except Exception as _s2s_e:
                        print(f"  ⚠️  Seq2seq worker failed [{_s}|{_a}|{_o}]: {_s2s_e}")
            print(f"  seq2seq training done in {_time.perf_counter() - _s2s_t0:.1f}s")

            for _r2 in _s2s_results:
                _s, _a, _o = _r2['sensor'], _r2['activity'], _r2['object']
                if _r2.get('skipped'):
                    print(f"  ⚠️  Skipped {_s}|{_a}|{_o} — seq2seq (too few curves).")
                    continue
                _s_elapsed = _r2.get('_elapsed', 0)
                if 'seq2seq' in _r2:
                    _ep = _r2['seq2seq']
                    _ep['variable_name'] = _s
                    _pipelines_seq2seq.setdefault(_s, {}).setdefault(_a, {})[_o] = {
                        'reference_curve': _ep['reference_curve'],
                        'predict_fn':      (lambda ep: lambda rv, act, attrs: predict_raw_curve_seq2seq(rv, act, attrs, pipeline=ep))(_ep),
                        'full_pipeline':   _ep,
                    }
                    print(f"  [{_s}|{_a}|{_o}] seq2seq       val_loss={_ep['val_loss']:.5f}  ({_s_elapsed:.1f}s)")
                if 'seq2seq_only' in _r2:
                    _ep = _r2['seq2seq_only']
                    _ep['variable_name'] = _s
                    _pipelines_seq2seq_only.setdefault(_s, {}).setdefault(_a, {})[_o] = {
                        'reference_curve': None,
                        'predict_fn':      (lambda ep: lambda rv, act, attrs: predict_raw_curve_seq2seq_only(rv, act, attrs, pipeline=ep))(_ep),
                        'full_pipeline':   _ep,
                    }
                    print(f"  [{_s}|{_a}|{_o}] seq2seq_only  val_loss={_ep['val_loss']:.5f}  ({_s_elapsed:.1f}s)")
                if 'seq2seq_exog' in _r2:
                    _ep = _r2['seq2seq_exog']
                    _ep['variable_name'] = _s
                    _pipelines_seq2seq_exog.setdefault(_s, {}).setdefault(_a, {})[_o] = {
                        'reference_curve': _ep['reference_curve'],
                        'predict_fn':      (lambda ep: lambda rv, act, attrs, exog=None: predict_raw_curve_seq2seq_exog(rv, act, attrs, pipeline=ep, exog_values=exog or {}))(_ep),
                        'full_pipeline':   _ep,
                    }
                    print(f"  [{_s}|{_a}|{_o}] seq2seq_exog  val_loss={_ep['val_loss']:.5f}  ({_s_elapsed:.1f}s)")
                if 'seq2seq_prev_activity' in _r2:
                    _ep = _r2['seq2seq_prev_activity']
                    _ep['variable_name'] = _s
                    _pipelines_seq2seq_prev_activity.setdefault(_s, {}).setdefault(_a, {})[_o] = {
                        'reference_curve': _ep['reference_curve'],
                        'predict_fn':      (lambda ep: lambda rv, act, attrs, exog=None: predict_raw_curve_seq2seq_prev_activity(rv, act, attrs, pipeline=ep, exog_values=exog or {}))(_ep),
                        'full_pipeline':   _ep,
                    }
                    print(f"  [{_s}|{_a}|{_o}] seq2seq_prev_activity  val_loss={_ep['val_loss']:.5f}  ({_s_elapsed:.1f}s)")

        # ── Mean baseline: predict training mean at every timestep ──────────
        _pipelines_mean = {}
        for _s in _sensors:
            for _a in _activities:
                for _o in _objects:
                    _mask_m = (
                        (_df_train_exp['activity_log'] == _a) &
                        (_df_train_exp['object_log']   == _o) &
                        _df_train_exp[_s].notna()
                    )
                    _vals_m = _df_train_exp.loc[_mask_m, _s].values
                    if len(_vals_m) == 0:
                        continue
                    _mean_val = float(np.mean(_vals_m))
                    _pip_m = {
                        'reference_curve': None,
                        'full_pipeline': {
                            'approach':      'mean_baseline',
                            'train_mean':    _mean_val,
                            'variable_name': _s,
                        },
                        'predict_fn': (lambda m: lambda rv, act, attrs: np.full(len(rv), m))(_mean_val),
                    }
                    _pipelines_mean.setdefault(_s, {}).setdefault(_a, {})[_o] = _pip_m

        all_energy_pipelines[_proc]                   = _pipelines_baseline
        all_energy_pipelines_mean[_proc]              = _pipelines_mean
        all_energy_pipelines_instance_stats[_proc]   = _pipelines_instance_stats
        all_energy_pipelines_istats_leakfree[_proc]  = _pipelines_istats_leakfree
        all_energy_pipelines_dtw_phase[_proc]        = _pipelines_dtw_phase
        all_energy_pipelines_basis[_proc]            = _pipelines_basis
        all_energy_pipelines_exog[_proc]             = _pipelines_exog
        all_energy_pipelines_exog_prev_activity[_proc]   = _pipelines_exog_prev_activity
        all_energy_pipelines_amplitude_shape[_proc]      = _pipelines_amplitude_shape

        all_energy_pipelines_seq2seq[_proc]               = _pipelines_seq2seq
        all_energy_pipelines_seq2seq_only[_proc]          = _pipelines_seq2seq_only
        all_energy_pipelines_seq2seq_exog[_proc]          = _pipelines_seq2seq_exog
        all_energy_pipelines_seq2seq_prev_activity[_proc] = _pipelines_seq2seq_prev_activity

# %%
# ══════════════════════════════════════════════════════════════════════════════
# CURVE-ONLY EVALUATION — BASELINE vs APPROACH 2 (B-spline) vs APPROACH 3 (DTW-phase)
# ══════════════════════════════════════════════════════════════════════════════

def _run_curve_eval(pipelines_dict, approach_label, split_label,
                    df_lookup, activities, objects, save_dir=None):
    """
    Evaluate every (process, sensor) in pipelines_dict against curves from
    df_lookup.  Returns a list of per-curve metric dicts.
    """
    import importlib, sim_extractor as _se
    importlib.reload(_se)
    from sim_extractor import (
        evaluate_pipeline_on_test,
        split_curves,
        split_curves_with_prev_activity,
    )
    records = []
    for _proc, _sensors in pipelines_dict.items():
        _df = df_lookup.get(_proc, {}).get('expanded')
        if _df is None or _df.empty:
            continue
        _acts = activities.get(_proc, [])
        _objs = objects.get(_proc, [])
        if not _acts and 'activity_log' in _df.columns:
            _acts = _df['activity_log'].dropna().unique().tolist()
        if not _objs and 'object_log' in _df.columns:
            _objs = _df['object_log'].dropna().unique().tolist()

        for _sensor, _sensor_val in _sensors.items():
            # nested: {activity: {object: ep}}
            _leaf_eps = [
                ([_a], [_o], _ep)
                for _a, _obj_map in _sensor_val.items()
                for _o, _ep in _obj_map.items()
            ]

            if not _leaf_eps:
                print(f"  [WARN] {approach_label} | {_proc} | {_sensor}: no leaf pipelines found, sensor_val keys={list(_sensor_val.keys())[:5]}")
                continue

            for _leaf_acts, _leaf_objs, _ep in _leaf_eps:
                _fp = _ep.get('full_pipeline', {})
                _approach_eval = _fp.get('approach', 'baseline')
                _exog_cols_eval = _fp.get('exog_cols', []) if _approach_eval in ('exog', 'seq2seq_exog', 'exog_prev_activity', 'seq2seq_prev_activity') else None
                if _approach_eval in ('exog_prev_activity', 'seq2seq_prev_activity'):
                    _curves, _ = split_curves_with_prev_activity(
                        _df, _sensor, _leaf_acts, _leaf_objs,
                        test_size=0.0, verbose=0,
                        exog_columns=_exog_cols_eval,
                    )
                else:
                    _curves, _ = split_curves(_df, _sensor, _leaf_acts, _leaf_objs,
                                              test_size=0.0, verbose=0,
                                              exog_columns=_exog_cols_eval)
                if not _curves:
                    print(f"  [WARN] {approach_label} | {_proc} | {_sensor} | act={_leaf_acts} obj={_leaf_objs}: split_curves returned 0 curves")
                    continue
                show_plots = (split_label == 'TEST')
                _curve_save = None
                if save_dir and show_plots:
                    _curve_save = os.path.join(
                        save_dir,
                        approach_label.replace(' ', '_').replace('/', '-'),
                    )
                try:
                    _metrics_df, _agg = evaluate_pipeline_on_test(
                        _curves, _ep['full_pipeline'],
                        max_plot_curves=6 if show_plots else 0,
                        verbose=1 if show_plots else 0,
                        save_dir=_curve_save,
                    )
                    for _, _r in _metrics_df.iterrows():
                        records.append({
                            'Approach': approach_label,
                            'Process':  _proc,
                            'Sensor':   _sensor,
                            'Split':    split_label,
                            'Activity': _r['activity'],
                            'N':        _r['n_points'],
                            'MAE':      _r['MAE'],
                            'RMSE':     _r['RMSE'],
                            'WAPE':     _r['WAPE (%)'],
                            'R2':       _r['R2'],
                            'sMAE':     _r.get('sMAE'),
                            'sRMSE':    _r.get('sRMSE'),
                        })
                except Exception as _eval_e:
                    print(f"  [ERROR] {approach_label} | {_proc} | {_sensor} | act={_leaf_acts}: {_eval_e}")
    return records


def _run_curve_eval_autoregressive_prev_act(pipelines_dict, approach_label, split_label,
                                            df_lookup, activities, objects, save_dir=None):
    """
    Autoregressive evaluator for the exog_prev_activity approach.

    For each test case in chronological order:
      • Activity 1 of the case has no real predecessor → uses the pipeline's
        per-activity training-median defaults (first_of_case_defaults).
      • Activity k (k≥2) uses prev_act_* features computed from the PREDICTED
        curve of activity k-1, never from real test values.
      • If an intermediate activity has no trained pipeline (or its curve is
        too short), the next activity is treated as first-of-case (prev reset).
    """
    import importlib, sim_extractor as _se
    importlib.reload(_se)
    from sim_extractor import _dispatch_predict
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    records = []
    for _proc, _sensor_pipelines in pipelines_dict.items():
        _df = df_lookup.get(_proc, {}).get('expanded')
        if _df is None or _df.empty:
            continue

        _proc_acts = activities.get(_proc, [])
        _proc_objs = objects.get(_proc, [])
        if not _proc_acts and 'activity_log' in _df.columns:
            _proc_acts = _df['activity_log'].dropna().unique().tolist()
        if not _proc_objs and 'object_log' in _df.columns:
            _proc_objs = _df['object_log'].dropna().unique().tolist()

        for _sensor, _act_pipes in _sensor_pipelines.items():
            if _sensor not in _df.columns:
                continue

            # Discover ef_cols once from any pipeline (all share the same set)
            _ef_cols = []
            for _a, _obj_pips in _act_pipes.items():
                for _o, _ep in _obj_pips.items():
                    _fp_any = _ep.get('full_pipeline', {})
                    if _fp_any.get('exog_cols'):
                        _ef_cols = list(_fp_any['exog_cols'])
                        break
                if _ef_cols:
                    break

            _sub = _df[_df['object_log'].isin(_proc_objs)
                       & _df['activity_log'].isin(_proc_acts)].copy()
            if _sub.empty:
                continue
            _sub['timestamp_start_log'] = pd.to_datetime(_sub['timestamp_start_log'])
            _sub['datetime_energy']     = pd.to_datetime(_sub['datetime_energy'])

            for _case_id, _case_df in _sub.groupby('case_id_log'):
                _inst_groups = list(_case_df.groupby(
                    ['object_log', 'activity_log', 'timestamp_start_log']
                ))
                _inst_groups.sort(key=lambda x: pd.to_datetime(x[0][2]))

                _prev = None   # None ⇒ next activity is first-of-case
                _inst_counter = 0

                for (_obj, _act, _ts_start), _inst_df in _inst_groups:
                    _inst_df = _inst_df.sort_values('datetime_energy').reset_index(drop=True)
                    _raw_values = np.asarray(_inst_df[_sensor].dropna().values).squeeze()
                    if _raw_values.ndim != 1 or len(_raw_values) < 5:
                        _prev = None
                        continue

                    _ep = _act_pipes.get(_act, {}).get(_obj)
                    if _ep is None:
                        _prev = None
                        continue

                    _fp = _ep['full_pipeline']

                    _raw_attrs = (_inst_df['object_attributes_log'].iloc[0]
                                  if 'object_attributes_log' in _inst_df.columns
                                  and not _inst_df['object_attributes_log'].empty else {})
                    _attrs = dict(_raw_attrs) if isinstance(_raw_attrs, dict) else {}
                    try:
                        _ts_obj = pd.to_datetime(_ts_start)
                        _attrs['hour_of_day'] = float(_ts_obj.hour)
                        _attrs['day_of_week']  = float(_ts_obj.dayofweek)
                    except Exception:
                        pass

                    if _prev is None:
                        _defaults = _fp.get('first_of_case_defaults', {})
                        _prev = _defaults.get(_act, {
                            'prev_act_name':   'none',
                            'prev_act_mean':   0.0,
                            'prev_act_std':    0.0,
                            'prev_act_max':    0.0,
                            'prev_act_end':    0.0,
                            'prev_act_length': 0.0,
                        })
                    _attrs.update(_prev)

                    _exog_values = {
                        _col: _inst_df[_col].values
                        for _col in _ef_cols if _col in _inst_df.columns
                    }
                    _inst_counter += 1
                    _curve = {
                        'activity':    _act,
                        'attributes':  _attrs,
                        'exog_values': _exog_values,
                        'instance_id': _inst_counter,
                    }

                    try:
                        _yp = _dispatch_predict(_raw_values, _curve, _fp)
                    except Exception as _ar_e:
                        print(f"  [WARN] autoreg predict failed: {_proc}|{_sensor}|{_act}: {_ar_e}")
                        _prev = None
                        continue

                    _mae  = float(mean_absolute_error(_raw_values, _yp))
                    _rmse = float(np.sqrt(mean_squared_error(_raw_values, _yp)))
                    _r2   = float(r2_score(_raw_values, _yp))
                    _den  = float(np.sum(np.abs(_raw_values)))
                    _wape = float(np.sum(np.abs(_raw_values - _yp))) / _den * 100 if _den != 0 else np.nan

                    _mu_ar  = _raw_values.mean()
                    _sig_ar = _raw_values.std()
                    if _sig_ar > 1e-10:
                        _zt_ar = (_raw_values - _mu_ar) / _sig_ar
                        _zp_ar = (_yp         - _mu_ar) / _sig_ar
                        _smae_ar  = float(np.mean(np.abs(_zt_ar - _zp_ar)))
                        _srmse_ar = float(np.sqrt(np.mean((_zt_ar - _zp_ar) ** 2)))
                    else:
                        _smae_ar = _srmse_ar = np.nan

                    records.append({
                        'Approach': approach_label,
                        'Process':  _proc,
                        'Sensor':   _sensor,
                        'Split':    split_label,
                        'Activity': _act,
                        'N':        int(len(_raw_values)),
                        'MAE':      _mae,
                        'RMSE':     _rmse,
                        'WAPE':     _wape,
                        'R2':       _r2,
                        'sMAE':     _smae_ar,
                        'sRMSE':    _srmse_ar,
                    })

                    _yp_arr = np.asarray(_yp, dtype=float)
                    _prev = {
                        'prev_act_name':   _act,
                        'prev_act_mean':   float(np.mean(_yp_arr)),
                        'prev_act_std':    float(np.std(_yp_arr))  if len(_yp_arr) >= 2 else 0.0,
                        'prev_act_max':    float(np.max(_yp_arr)),
                        'prev_act_end':    float(_yp_arr[-1]),
                        'prev_act_length': float(len(_yp_arr)),
                    }
    return records


if RUN_CURVE_ONLY_EVALUATION and 'all_energy_pipelines' in dir() and all_energy_pipelines:
    import importlib, sim_extractor as _se
    importlib.reload(_se)
    from sim_extractor import evaluate_pipeline_on_test, split_curves, split_curves_with_prev_activity

    # _run_dir and _plots_dir are already created at startup
    _plot_counter = [0]   # mutable counter usable inside nested scopes

    def _savefig(name):
        _plot_counter[0] += 1
        _safe = name.replace('/', '_per_').replace('\\', '_')
        _p = os.path.join(_plots_dir, f"{_plot_counter[0]:02d}_{_safe}.png")
        plt.savefig(_p, dpi=150, bbox_inches='tight')

    display(Markdown("---"))
    display(Markdown("# Curve-Only Evaluation — Baseline vs Approach 2 (B-spline) vs Approach 3 (DTW-phase)"))

    _proc_cfg_lookup = {
        _p: process_datasets_to_model_sensors.get(_p, {})
        for _p in process_datasets_to_model.keys()
    } if 'process_datasets_to_model_sensors' in dir() else {}
    _activities_map = {p: c.get('activities_to_model', []) for p, c in _proc_cfg_lookup.items()}
    _objects_map    = {p: c.get('objects_to_model',    []) for p, c in _proc_cfg_lookup.items()}

    _all_records = []

    # ── Evaluate both approaches on TRAIN and TEST ───────────────────────────
    for _split_label, _df_src in [('TRAIN', train_datasets),
                                   ('TEST',  test_datasets if TEMPORAL_SPLIT else train_datasets)]:

        display(Markdown(f"## Split: {_split_label}"))

        for _approach_label, _pipelines in [
            ('Mean Baseline',               all_energy_pipelines_mean),
            ('DTW + pos',        all_energy_pipelines),
            ('Instance Stats (leaky)',      all_energy_pipelines_instance_stats),
            ('Instance Stats (leak-free)',  all_energy_pipelines_istats_leakfree),
            ('Approach 2 (B-spline)',       all_energy_pipelines_basis),
            ('Approach 3 (DTW-phase)',      all_energy_pipelines_dtw_phase),
            ('DTW + Ext. Factors',          all_energy_pipelines_exog),
            ('DTW + Ext. Factors + Prev Act', all_energy_pipelines_exog_prev_activity),
            ('Amplitude + Shape',           all_energy_pipelines_amplitude_shape),

            ('DTW + Seq2Seq',               all_energy_pipelines_seq2seq),
            ('Seq2Seq only',                all_energy_pipelines_seq2seq_only),
            ('DTW + Seq2Seq + Ext. Factors', all_energy_pipelines_seq2seq_exog),
            ('DTW + Seq2Seq + Ext. Factors + Prev Act', all_energy_pipelines_seq2seq_prev_activity),
        ]:
            if not _pipelines:
                continue
            display(Markdown(f"### {_approach_label}"))
            _recs = _run_curve_eval(
                _pipelines, _approach_label, _split_label,
                _df_src, _activities_map, _objects_map,
                save_dir=None,
            )
            _all_records.extend(_recs)

        # ── Autoregressive rollout for exog_prev_activity ───────────────────
        if all_energy_pipelines_exog_prev_activity:
            display(Markdown("### DTW + Ext. Factors + Prev Act (autoreg)"))
            _ar_recs = _run_curve_eval_autoregressive_prev_act(
                all_energy_pipelines_exog_prev_activity,
                'DTW + Ext. Factors + Prev Act (autoreg)',
                _split_label,
                _df_src, _activities_map, _objects_map,
                save_dir=None,
            )
            _all_records.extend(_ar_recs)

        # ── Autoregressive rollout for seq2seq_prev_activity ─────────────────
        if all_energy_pipelines_seq2seq_prev_activity:
            display(Markdown("### DTW + Seq2Seq + Ext. Factors + Prev Act (autoreg)"))
            _ar_s2s_recs = _run_curve_eval_autoregressive_prev_act(
                all_energy_pipelines_seq2seq_prev_activity,
                'DTW + Seq2Seq + Ext. Factors + Prev Act (autoreg)',
                _split_label,
                _df_src, _activities_map, _objects_map,
                save_dir=None,
            )
            _all_records.extend(_ar_s2s_recs)

    # ── Side-by-side comparison table ───────────────────────────────────────
    if not _all_records:
        print("[ERROR] _all_records is empty — no curves were evaluated. Check WARN messages above.")
    if _all_records:
        _all_df = pd.DataFrame(_all_records)

        # ── Top-level table: Approach × Split (all sensors/processes aggregated) ─
        display(Markdown("---"))
        display(Markdown("## Approach Comparison — Train & Test (median over ALL sensors, processes, curves)"))
        _appr_summary = (
            _all_df
            .groupby(['Approach', 'Split'])[['MAE', 'RMSE', 'WAPE', 'R2']]
            .median()
            .round(4)
            .unstack('Split')
        )
        _appr_summary.columns = [f'{m}_{s}' for m, s in _appr_summary.columns]
        _a_train = [c for c in _appr_summary.columns if c.endswith('_TRAIN')]
        _a_test  = [c for c in _appr_summary.columns if c.endswith('_TEST')]
        _appr_summary = _appr_summary[_a_train + _a_test]
        display(_appr_summary)
        report("\nAPPROACH COMPARISON (TRAIN & TEST)")
        report(_appr_summary.to_string())

        # ── Master summary table: Process × Sensor × Approach, TRAIN and TEST ─
        display(Markdown("---"))
        display(Markdown("## Model Summary — Train & Test metrics per Process / Sensor / Approach (median over curves)"))
        _summary = (
            _all_df
            .groupby(['Process', 'Sensor', 'Approach', 'Split'])[['MAE', 'RMSE', 'WAPE', 'R2']]
            .median()
            .round(4)
        )
        # Unstack Split so TRAIN / TEST appear as column groups side by side
        _summary_wide = _summary.unstack('Split')
        # Flatten multi-level column names: MAE_TRAIN, MAE_TEST, …
        _summary_wide.columns = [f'{m}_{s}' for m, s in _summary_wide.columns]
        # Reorder: all TRAIN cols first, then TEST
        _train_cols = [c for c in _summary_wide.columns if c.endswith('_TRAIN')]
        _test_cols  = [c for c in _summary_wide.columns if c.endswith('_TEST')]
        _summary_wide = _summary_wide[_train_cols + _test_cols]
        display(_summary_wide)
        report("\nMODEL SUMMARY (TRAIN & TEST)")
        report(_summary_wide.to_string())

        display(Markdown("---"))
        display(Markdown("## Detailed comparison — TEST set (median over curves)"))

        _test_df = _all_df[_all_df['Split'] == 'TEST']
        if not _test_df.empty:
            _compare = (
                _test_df
                .groupby(['Approach', 'Process', 'Sensor'])[['MAE', 'RMSE', 'WAPE', 'R2']]
                .median()
                .round(4)
            )
            display(_compare)
            report("\nCURVE MODEL COMPARISON (TEST SET)")
            report(_compare.to_string())

            # ── Per-activity table ───────────────────────────────────────────
            display(Markdown("### Per-activity breakdown (TEST) — median over curves"))
            _act_compare = (
                _test_df
                .groupby(['Approach', 'Process', 'Sensor', 'Activity'])[['MAE', 'RMSE', 'WAPE', 'R2']]
                .median()
                .round(4)
            )
            display(_act_compare)

            # ── WAPE distribution per approach (boxplots) ───────────────────
            try:
                display(Markdown("---"))
                display(Markdown("### WAPE Distribution — TEST set (per-curve boxplots)"))
                _wape_cap = _test_df['WAPE'].quantile(0.95)
                _appr_order = (
                    _test_df.groupby('Approach')['WAPE'].median()
                    .sort_values().index.tolist()
                )
                fig_wd, ax_wd = plt.subplots(figsize=(max(8, len(_appr_order) * 1.6), 6))
                sns.boxplot(
                    data=_test_df[_test_df['WAPE'] <= _wape_cap],
                    x='Approach', y='WAPE',
                    order=_appr_order,
                    palette='tab10',
                    width=0.5,
                    flierprops=dict(marker='o', markersize=3, alpha=0.4),
                    ax=ax_wd,
                )
                ax_wd.set_xticklabels(ax_wd.get_xticklabels(), rotation=25, ha='right')
                ax_wd.set_ylabel('WAPE (%)')
                ax_wd.set_xlabel('')
                ax_wd.set_title(
                    f'WAPE Distribution per Approach — TEST set  '
                    f'(capped at {_wape_cap:.0f}% = 95th pct)',
                    fontweight='bold'
                )
                ax_wd.grid(True, axis='y', alpha=0.3)
                plt.tight_layout()
                if EXPORT_RESULTS and '_run_dir' in dir():
                    _savefig('wape_distribution')
                plt.show()
            except Exception as _e:
                print(f"[WARN] WAPE distribution plot failed: {_e}")

            # ── WAPE distribution — interactive Plotly version ───────────────
            try:
                import plotly.graph_objects as _go
                _wape_cap_px = _test_df['WAPE'].quantile(0.95)
                _df_px = _test_df[_test_df['WAPE'] <= _wape_cap_px].copy()
                _appr_order_px = (
                    _df_px.groupby('Approach')['WAPE'].median()
                    .sort_values().index.tolist()
                )
                _fig_px = _go.Figure()
                for _appr_px in _appr_order_px:
                    _vals_px = _df_px[_df_px['Approach'] == _appr_px]['WAPE'].dropna()
                    _fig_px.add_trace(_go.Box(
                        y=_vals_px,
                        name=_appr_px,
                        boxpoints='outliers',
                        marker=dict(size=4, opacity=0.5),
                        boxmean='sd',
                    ))
                _fig_px.update_layout(
                    title=dict(
                        text=f'WAPE Distribution per Approach — TEST set'
                             f'<br><sup>capped at {_wape_cap_px:.0f}% (95th pct) · mean±sd shown as dashed line</sup>',
                        font=dict(size=14),
                    ),
                    yaxis_title='WAPE (%)',
                    xaxis_title='',
                    showlegend=False,
                    height=520,
                    template='plotly_white',
                )
                _fig_px.show()
                if EXPORT_RESULTS and '_plots_dir' in dir():
                    _px_path = os.path.join(_plots_dir, 'wape_distribution_interactive.html')
                    _fig_px.write_html(_px_path, include_plotlyjs='cdn')
                    print(f"Saved interactive plot → {_px_path}")
            except Exception as _e:
                print(f"[WARN] Plotly WAPE distribution failed: {_e}")

            # ── R² heatmap — one subplot per approach ────────────────────────
            try:
                _approaches = _test_df['Approach'].unique()
                fig_h, axes_h = plt.subplots(
                    1, len(_approaches),
                    figsize=(max(8, _test_df['Activity'].nunique() * 1.4) * len(_approaches),
                             max(3, _test_df[['Process', 'Sensor']].drop_duplicates().shape[0] * 1.2))
                )
                axes_h = np.array(axes_h).reshape(-1)

                for ax_h, _appr in zip(axes_h, _approaches):
                    _sub = _test_df[_test_df['Approach'] == _appr]
                    _ph = _sub.pivot_table(
                        index=['Process', 'Sensor'], columns='Activity',
                        values='R2', aggfunc='median'
                    )
                    sns.heatmap(
                        _ph, annot=True, fmt='.3f', cmap='RdYlGn',
                        vmin=0, vmax=1, linewidths=0.5, ax=ax_h,
                        cbar_kws={'label': 'R²'}
                    )
                    ax_h.set_title(f'R² — {_appr}', fontsize=11, fontweight='bold')
                    ax_h.set_xticklabels(ax_h.get_xticklabels(), rotation=30, ha='right', fontsize=8)

                plt.suptitle('Curve R² per Activity — TEST set', fontsize=13, fontweight='bold', y=1.02)
                plt.tight_layout()
                if EXPORT_RESULTS and '_run_dir' in dir():
                    _savefig('r2_heatmap_all_approaches')
                plt.show()
            except Exception as _e:
                print(f"[WARN] R² heatmap failed: {_e}")

            # ── Delta heatmaps: each new approach minus baseline ─────────────
            try:
                _base_pivot = _test_df[_test_df['Approach'] == 'DTW + pos'].pivot_table(
                    index=['Process', 'Sensor'], columns='Activity', values='R2', aggfunc='median'
                )
                for _delta_label, _delta_appr in [
                    ('Instance Stats',         'Instance Stats'),
                    ('Approach 2 (B-spline)', 'Approach 2 (B-spline)'),
                    ('Approach 3 (DTW-phase)', 'Approach 3 (DTW-phase)'),
                    ('DTW + Ext. Factors',           'DTW + Ext. Factors'),
                    ('DTW + Seq2Seq',                'DTW + Seq2Seq'),
                    ('Seq2Seq only',                 'Seq2Seq only'),
                    ('DTW + Seq2Seq + Ext. Factors', 'DTW + Seq2Seq + Ext. Factors'),
                ]:
                    _new_pivot = _test_df[_test_df['Approach'] == _delta_appr].pivot_table(
                        index=['Process', 'Sensor'], columns='Activity', values='R2', aggfunc='median'
                    )
                    if _base_pivot.empty or _new_pivot.empty:
                        continue
                    _delta = (_new_pivot - _base_pivot).reindex_like(_base_pivot)
                    fig_d, ax_d = plt.subplots(figsize=(max(8, len(_base_pivot.columns) * 1.4),
                                                         max(3, len(_base_pivot) * 1.2)))
                    sns.heatmap(
                        _delta, annot=True, fmt='.3f', cmap='RdYlGn',
                        center=0, linewidths=0.5, ax=ax_d,
                        cbar_kws={'label': f'ΔR² ({_delta_label} − Baseline)'}
                    )
                    ax_d.set_title(
                        f'ΔR² {_delta_label} vs Baseline — green = {_delta_label} better',
                        fontsize=11, fontweight='bold'
                    )
                    ax_d.set_xticklabels(ax_d.get_xticklabels(), rotation=30, ha='right', fontsize=8)
                    plt.tight_layout()
                    if EXPORT_RESULTS and '_run_dir' in dir():
                        _savefig(f'delta_r2_{_delta_label.replace(" ", "_").replace("/", "-")}')
                    plt.show()
            except Exception as _e:
                print(f"[WARN] Delta heatmaps failed: {_e}")

            # ── 5 BEST / 5 WORST curves per approach (TEST R²) ──────────────────
            try:
                from sklearn.metrics import r2_score, mean_absolute_error
                display(Markdown("---"))
                display(Markdown("## 5 Best & 5 Worst Curve Fits per Approach — TEST set"))

                _APPROACH_PIPELINES = {
                    'Mean Baseline':                  all_energy_pipelines_mean
                        if 'all_energy_pipelines_mean' in dir() else {},
                    'DTW + pos':           all_energy_pipelines
                        if 'all_energy_pipelines' in dir() else {},
                    'DTW + Ext. Factors + Prev Act':  all_energy_pipelines_exog_prev_activity
                        if 'all_energy_pipelines_exog_prev_activity' in dir() else {},
                    'Amplitude + Shape':              all_energy_pipelines_amplitude_shape
                        if 'all_energy_pipelines_amplitude_shape' in dir() else {},

                    'DTW + Seq2Seq':                  all_energy_pipelines_seq2seq
                        if 'all_energy_pipelines_seq2seq' in dir() else {},
                    'Seq2Seq only':                   all_energy_pipelines_seq2seq_only
                        if 'all_energy_pipelines_seq2seq_only' in dir() else {},
                    'DTW + Seq2Seq + Ext. Factors':   all_energy_pipelines_seq2seq_exog
                        if 'all_energy_pipelines_seq2seq_exog' in dir() else {},
                    'DTW + Seq2Seq + Ext. Factors + Prev Act': all_energy_pipelines_seq2seq_prev_activity
                        if 'all_energy_pipelines_seq2seq_prev_activity' in dir() else {},
                }

                _ranked_bw = (
                    _test_df
                    .groupby(['Approach', 'Process', 'Sensor', 'Activity'])['WAPE']
                    .median()
                    .reset_index()
                    .rename(columns={'WAPE': 'Median_WAPE'})
                )

                for _appr, _appr_pips in _APPROACH_PIPELINES.items():
                    if not _appr_pips:
                        continue
                    _sub_bw = _ranked_bw[_ranked_bw['Approach'] == _appr].sort_values(
                        'Median_WAPE', ascending=True
                    )
                    if _sub_bw.empty:
                        continue

                    for _group_label, _group_rows in [
                        ('5 BEST',  _sub_bw.head(5)),
                        ('5 WORST', _sub_bw.tail(5)),
                    ]:
                        _row_list = list(_group_rows.iterrows())
                        _n_rows   = len(_row_list)
                        if not _n_rows:
                            continue

                        display(Markdown(f"### {_appr} — {_group_label} (by median TEST WAPE)"))
                        fig_bw, axes_bw = plt.subplots(
                            _n_rows, 3, figsize=(15, 4 * _n_rows), squeeze=False
                        )

                        for _ri, (_, _row) in enumerate(_row_list):
                            _proc_bw   = _row['Process']
                            _sensor_bw = _row['Sensor']
                            _act_bw    = _row['Activity']
                            _wape_bw   = _row['Median_WAPE']

                            _pip_act = (
                                _appr_pips
                                .get(_proc_bw, {})
                                .get(_sensor_bw, {})
                                .get(_act_bw, {})
                            )
                            if not _pip_act:
                                for _ax in axes_bw[_ri]:
                                    _ax.set_visible(False)
                                continue

                            _obj_bw = next(iter(_pip_act))
                            _ep_bw  = _pip_act[_obj_bw]
                            _df_test_bw = (test_datasets if TEMPORAL_SPLIT else train_datasets).get(
                                _proc_bw, {}
                            ).get('expanded')
                            if _df_test_bw is None or _df_test_bw.empty:
                                for _ax in axes_bw[_ri]:
                                    _ax.set_visible(False)
                                continue

                            _bw_approach = _ep_bw.get('full_pipeline', {}).get('approach', 'baseline')
                            _bw_exog = _ep_bw.get('full_pipeline', {}).get('exog_cols', []) \
                                if _bw_approach in ('exog', 'seq2seq_exog', 'exog_prev_activity') else None
                            if _bw_approach == 'exog_prev_activity':
                                _tc_bw, _ = split_curves_with_prev_activity(
                                    _df_test_bw, _sensor_bw, [_act_bw], [_obj_bw],
                                    test_size=0.0, verbose=0,
                                    exog_columns=_bw_exog,
                                )
                            else:
                                _tc_bw, _ = split_curves(
                                    _df_test_bw, _sensor_bw, [_act_bw], [_obj_bw],
                                    test_size=0.0, verbose=0,
                                    exog_columns=_bw_exog,
                                )
                            if not _tc_bw:
                                for _ax in axes_bw[_ri]:
                                    _ax.set_visible(False)
                                continue

                            for _ci, _ax in enumerate(axes_bw[_ri]):
                                if _ci >= len(_tc_bw):
                                    _ax.set_visible(False)
                                    continue
                                _curve_bw = _tc_bw[_ci]
                                _rv_bw    = _curve_bw['original_values']
                                _yp_bw    = _ep_bw['predict_fn'](
                                    _rv_bw, _curve_bw['activity'],
                                    _curve_bw.get('attributes', {})
                                )
                                _denom_c = np.sum(np.abs(_rv_bw))
                                _wape_c  = np.sum(np.abs(_rv_bw - _yp_bw)) / _denom_c * 100 if _denom_c != 0 else np.nan
                                _r2_c    = r2_score(_rv_bw, _yp_bw)
                                _ax.plot(_rv_bw, label='Actual', color='steelblue', linewidth=2)
                                _ax.plot(_yp_bw, label='Predicted', color='tomato',
                                         linewidth=2, linestyle='--')
                                _ax.set_title(
                                    f"{_act_bw[:28]} | {_sensor_bw[:20]}\n"
                                    f"WAPE={_wape_c:.1f}%  R²={_r2_c:.3f}  "
                                    f"(bucket median WAPE={_wape_bw:.1f}%)",
                                    fontsize=8
                                )
                                _ax.set_xlabel("Time step")
                                _ax.set_ylabel("Energy")
                                _ax.grid(True, alpha=0.3)
                                _ax.legend(fontsize=7)

                        fig_bw.suptitle(
                            f"{_appr}  —  {_group_label} by TEST WAPE",
                            fontsize=12, fontweight='bold'
                        )
                        plt.tight_layout()
                        if EXPORT_RESULTS and '_run_dir' in dir():
                            _appr_slug = _appr.replace(" ", "_").replace("+", "p").replace("/", "-")
                            _gl_slug   = _group_label.replace(" ", "_")
                            _savefig(f'best_worst_{_appr_slug}_{_gl_slug}')
                        plt.show()
            except Exception as _e:
                print(f"[WARN] Best/worst plot failed: {_e}")
                import traceback; traceback.print_exc()

elif RUN_CURVE_ONLY_EVALUATION:
    display(Markdown(
        "> **Curve-Only Evaluation skipped** — `all_energy_pipelines` is empty. "
        "Make sure the energy pipeline training block ran successfully."
    ))

# %%
# ── STANDALONE PROFILE EVALUATION (TRAIN & TEST) ──────────────────────────────
import importlib, sim_extractor as _se
importlib.reload(_se)
from sim_extractor import evaluate_pipeline_on_test, split_curves
profile_summary_records = []

if 'process_datasets_to_model_sensors' in dir():
    for process, config in process_datasets_to_model_sensors.items():
        sensors_to_model = config.get('sensors_to_model', [])

        for sensor in sensors_to_model:
            if process not in all_energy_pipelines or sensor not in all_energy_pipelines[process]:
                continue

            # Iterate every (activity, object) sub-pipeline
            for _activity, _obj_map in all_energy_pipelines[process][sensor].items():
                for _object, _ep in _obj_map.items():
                    pipeline = _ep['full_pipeline']

                    for split_label, df_exp in [
                        ('TRAIN', train_datasets[process].get('expanded')),
                        ('TEST',  test_datasets[process].get('expanded') if RUN_TEST_EVALUATION else None),
                    ]:
                        if df_exp is None or df_exp.empty:
                            continue
                        _sc, _ = split_curves(df_exp, sensor, [_activity], [_object],
                                              test_size=0.0, verbose=0)
                        if not _sc:
                            continue
                        _mdf, _ = evaluate_pipeline_on_test(_sc, pipeline, max_plot_curves=0, verbose=0)
                        if _mdf.empty:
                            continue
                        for _, _row in _mdf.iterrows():
                            profile_summary_records.append({
                                'Process':   process,
                                'Sensor':    sensor,
                                'Activity':  _activity,
                                'Object':    _object,
                                'Split':     split_label,
                                'MAE':       _row['MAE'],
                                'RMSE':      _row['RMSE'],
                                'WAPE':      _row['WAPE (%)'],
                                'R2':        _row['R2'],
                            })

if profile_summary_records:
    profile_summary_df = pd.DataFrame(profile_summary_records)
    report("\n" + "="*80)
    report("DETAILED ENERGY PROFILE METRICS PER PROCESS / SENSOR / ACTIVITY")
    report("="*80)
    _psummary = (
        profile_summary_df
        .groupby(['Process', 'Sensor', 'Activity', 'Object', 'Split'])[['MAE', 'RMSE', 'WAPE', 'R2']]
        .median()
        .round(4)
    )
    report(_psummary.to_string())
    display(_psummary)

# ── SIMULATION CURVE VISUALS (SIMULATED VS REAL) ───────────────────────────
report("\n" + "="*80)
report("VISUAL COMPARISON: SIMULATED (TEST RUN) VS REAL DATA")
report("="*80)

if _combined_sim_store:
    from collections import defaultdict
    from scipy.interpolate import interp1d as _interp1d

    _N_SAMPLE_CURVES = 4  # max example curves per activity in the plot grid

    # ── per-activity metrics table ────────────────────────────────────────
    _act_metric_rows = []
    for _entry in _combined_sim_store:
        for (_sensor, _act), _vals in _entry['act_metrics'].items():
            _act_metric_rows.append({
                'Process': _entry['process'], 'Mode': _entry['mode'],
                'Sensor': _sensor, 'Activity': _act,
                **_vals,
            })
    if _act_metric_rows:
        _act_df = pd.DataFrame(_act_metric_rows)
        report("\nPER-ACTIVITY ENERGY METRICS (TEST SET, combined sim):")
        report("-" * 80)
        _act_pivot = (
            _act_df.groupby(['Process', 'Mode', 'Sensor', 'Activity'])[['MAE', 'RMSE', 'WAPE', 'R2']]
            .mean().round(4)
        )
        report(_act_pivot.to_string())
        display(_act_pivot)

    # ── curve plots per (process, mode, sensor) ───────────────────────────
    for _entry in _combined_sim_store:
        _proc   = _entry['process']
        _mode   = _entry['mode']
        _sdf    = _entry['sim_df']
        _edf    = _entry['exp_df']
        _s_list = _entry.get('sensors', [])

        if _sdf is None or _sdf.empty or 'simulated_energy_curves' not in _sdf.columns:
            continue

        _case_col = 'case_id_log' if 'case_id_log' in _sdf.columns else \
                    next((c for c in _sdf.columns if 'case' in c.lower()), None)
        _act_col  = 'activity_log' if 'activity_log' in _sdf.columns else \
                    next((c for c in _sdf.columns if 'activity' in c.lower()), None)
        if not _case_col or not _act_col:
            continue

        for _sensor in _s_list:
            # collect (activity, y_real, y_sim) pairs
            _by_act = defaultdict(list)
            for _, _row in _sdf.iterrows():
                _sc = _row.get('simulated_energy_curves', {})
                if not isinstance(_sc, dict) or _sensor not in _sc:
                    continue
                _cid = _row[_case_col]
                _act = _row[_act_col]
                _real_rows = _edf[
                    (_edf['case_id_log'] == _cid) & (_edf['activity_log'] == _act)
                ] if 'case_id_log' in _edf.columns else pd.DataFrame()
                if _real_rows.empty or _sensor not in _real_rows.columns:
                    continue
                _y_real = _real_rows[_sensor].dropna().values
                if len(_y_real) < 2:
                    continue
                _y_sim = np.asarray(_sc[_sensor], dtype=float)
                if len(_y_sim) != len(_y_real):
                    _f = _interp1d(np.linspace(0, 1, len(_y_sim)), _y_sim,
                                   kind='linear', fill_value='extrapolate')
                    _y_sim = _f(np.linspace(0, 1, len(_y_real)))
                _by_act[_act].append((_y_real, _y_sim))

            if not _by_act:
                continue

            _acts_sorted = sorted(_by_act.keys())
            _n_rows = len(_acts_sorted)
            _n_cols = _N_SAMPLE_CURVES

            fig, axes = plt.subplots(_n_rows, _n_cols,
                                     figsize=(3.5 * _n_cols, 2.2 * _n_rows),
                                     squeeze=False)
            fig.suptitle(f'{_proc}  |  {_mode}  |  {_sensor}\n'
                         f'Real (blue) vs Simulated (orange)',
                         fontsize=10, fontweight='bold')

            for _ai, _act in enumerate(_acts_sorted):
                _pairs = _by_act[_act][:_n_cols]
                # compute per-activity R² for subplot annotation
                _r2_vals = []
                for _yr, _ys in _pairs:
                    _ss = np.sum((_yr - np.mean(_yr))**2)
                    _r2_vals.append(1 - np.sum((_yr - _ys)**2) / _ss if _ss > 0 else 0.0)

                for _ci, (_yr, _ys) in enumerate(_pairs):
                    ax = axes[_ai][_ci]
                    ax.plot(_yr, color='steelblue', lw=1.5, alpha=0.85, label='Real')
                    ax.plot(_ys, color='darkorange', lw=1.5, alpha=0.85,
                            linestyle='--', label='Sim')
                    ax.set_title(f'R²={_r2_vals[_ci]:.3f}', fontsize=7)
                    ax.tick_params(labelsize=6)
                    if _ci == 0:
                        ax.set_ylabel(_act, fontsize=7, rotation=0,
                                      ha='right', va='center', labelpad=60)
                for _ci in range(len(_pairs), _n_cols):
                    axes[_ai][_ci].axis('off')

            _handles = [
                plt.Line2D([0], [0], color='steelblue', lw=1.5, label='Real'),
                plt.Line2D([0], [0], color='darkorange', lw=1.5,
                           linestyle='--', label='Simulated'),
            ]
            fig.legend(handles=_handles, loc='upper right', fontsize=8)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            _savefig(f'sim_curves_{_proc}_{_mode}_{_sensor}')
            plt.show()
else:
    report("  No combined simulation results available for curve plotting.")

# ── FINAL CONSOLIDATED ENERGY PERFORMANCE REPORT ───────────────────────────
report("\n" + "█"*80)
report("█   FINAL ENERGY PERFORMANCE REPORT (SIMULATION QUALITY)             █")
report("█"*80)

energy_metrics_summary = []
records = []
if 'evaluation_results_df' in dir() and not evaluation_results_df.empty:
    energy_cols = [c for c in evaluation_results_df.columns if c.startswith('test_energy_')]
    for _, row in evaluation_results_df.iterrows():
        proc = row['process']
        mode = row['mode']
        # Focus on the energy-aware modes for the detailed report
        if mode not in _ENERGY_AWARE_MODES:
            continue
            
        for c in energy_cols:
            metric = c.split('_')[-1]
            sensor = c.replace('test_energy_', '').replace(f'_{metric}', '')
            val = row[c]
            if pd.notna(val):
                records.append({
                    'Dataset': proc,
                    'Sensor': sensor,
                    'Metric': metric,
                    'Value': val
                })

if records:
    edf = pd.DataFrame(records)
    # Pivot for clean display: Dataset and Sensor as index, Metric as columns
    report_pivot = edf.pivot_table(index=['Dataset', 'Sensor'], columns='Metric', values='Value', aggfunc='median')
    
    # Ensure all requested metrics are in columns
    final_cols = [c for c in ['MAE', 'RMSE', 'WAPE', 'R2'] if c in report_pivot.columns]
    report_pivot = report_pivot[final_cols]
    
    report("\nDETAILED ENERGY PROFILE METRICS PER DATASET (TEST SET):")
    report("-" * 80)
    # Output to both log (via report/string) and notebook (via display)
    # report() handles the log and the safe stdout
    report(report_pivot.round(4).to_string())
    # display() ensures the beautiful interactive table in the notebook
    display(report_pivot.round(4))
else:
    report("\n⚠️  No energy simulation metrics found in the final results.")

report("\n" + "█"*80 + "\n")

# ── FEATURE IMPORTANCE SUMMARY ────────────────────────────────────────────────
report("\n" + "="*80)
report("ENERGY MODEL — FEATURE IMPORTANCE SUMMARY")
report("="*80)
report("(Only activities where an ML model beat the R²>0.05 threshold are shown)")

_fi_records = []
if 'energy_modifiers_by_process' in dir():
    for _proc, _mods in energy_modifiers_by_process.items():
        for _role, _mod_dict in [('Duration', _mods.get('duration', {})),
                                  ('Transition', _mods.get('transition', {}))]:
            for _act, _mdl in _mod_dict.items():
                fi = getattr(_mdl, '_feature_importance', {})
                if not fi:
                    continue
                # Top 3 features by importance
                for rank, (feat, imp) in enumerate(
                    sorted(fi.items(), key=lambda kv: -kv[1])[:3], start=1
                ):
                    _fi_records.append({
                        'Process':   _proc,
                        'Activity':  _act,
                        'Role':      _role,
                        'Rank':      rank,
                        'Feature':   feat,
                        'Importance': round(imp, 4),
                    })

if _fi_records:
    _fi_df = pd.DataFrame(_fi_records)
    _fi_pivot = _fi_df.pivot_table(
        index=['Process', 'Activity', 'Role'],
        columns='Rank',
        values=['Feature', 'Importance'],
        aggfunc='first',
    )
    # Flatten multi-level columns to e.g. "Feature_1", "Importance_1"
    _fi_pivot.columns = [f'{col}_{rank}' for col, rank in _fi_pivot.columns]
    _fi_pivot = _fi_pivot.reset_index()
    # Reorder into readable triples: Feature_1, Imp_1, Feature_2, Imp_2 ...
    _ordered = ['Process', 'Activity', 'Role']
    for _r in [1, 2, 3]:
        for _c in [f'Feature_{_r}', f'Importance_{_r}']:
            if _c in _fi_pivot.columns:
                _ordered.append(_c)
    _fi_pivot = _fi_pivot[[c for c in _ordered if c in _fi_pivot.columns]]
    report(_fi_pivot.to_string(index=False))
    display(_fi_pivot)
else:
    report("  No ML models passed the R²>0.05 threshold — no feature importances to show.")

report("\n" + "█"*80 + "\n")

# ── POST-REPORT VISUALIZATIONS: GENERALIZATION GALLERY ───────────────────────
# (Note: Placed at the very end to provide a final visual verification of curve fitting)
if RUN_TEST_EVALUATION:
    _gallery_dir = None

    for process, sensors in all_energy_pipelines.items():
        exp_test = test_datasets[process].get('expanded')
        if exp_test is None: continue

        for sensor, act_map in sensors.items():
            for _activity, obj_map in act_map.items():
                for _object, _ep in obj_map.items():
                    test_curves, _ = split_curves(exp_test, sensor, [_activity], [_object],
                                                  test_size=0.0, verbose=0)
                    if not test_curves:
                        continue
                    display(Markdown("---"))
                    display(Markdown(f"## Generalization Gallery: {sensor.upper()} | {_activity} | {_object}"))
                    display(Markdown(f"*Predicted vs real curves — Test Set — {process}*"))
                    evaluate_pipeline_on_test(
                        test_curves,
                        _ep['full_pipeline'],
                        max_plot_curves=6,
                        verbose=1,
                        save_dir=_gallery_dir,
                    )

# %%
# ══════════════════════════════════════════════════════════════════════════════
# RESULTS EXPORT — parquet table + HTML notebook snapshot
# ══════════════════════════════════════════════════════════════════════════════
import subprocess

if not EXPORT_RESULTS:
    print("EXPORT_RESULTS=False — skipping export.")
else:
    # _run_dir is created at startup — just flush the log before saving anything
    _log_handler.flush()

    # ── Curve-only evaluation results ────────────────────────────────────────
    _export_df = None
    for _cname in ('_all_df', '_all_records'):
        _cval = globals().get(_cname)
        if _cval is not None:
            _export_df = pd.DataFrame(_cval) if isinstance(_cval, list) else _cval
            if not _export_df.empty:
                break
            _export_df = None

    if _export_df is not None and not _export_df.empty:
        _parquet_path = os.path.join(_run_dir, 'curve_eval_results.parquet')
        _export_df.to_parquet(_parquet_path, index=False)
        print(f"Saved results  → {_parquet_path}")

        # ── Energy heatmaps: rows=Approach, columns=Metric, aggregated over all
        #    processes and sensors. One heatmap per split × aggregation.
        # sMAE / sRMSE are computed per curve by z-scoring y_true with its own
        # mean and std, applying the same transform to y_pred, then computing
        # MAE and RMSE on the standardised values.

        def _plot_curve_energy_heatmap(df, split, agg, save_dir):
            """Heatmap: rows=Approach, columns=sMAE/sRMSE/WAPE/R2, agg across everything."""
            _metrics = [m for m in ['sMAE', 'sRMSE', 'WAPE'] if m in df.columns]
            _sub = df[df['Split'] == split]
            if _sub.empty or not _metrics:
                return
            _fn = 'median' if agg == 'median' else 'mean'
            _agg = _sub.groupby('Approach')[_metrics].agg(_fn)

            # Normalise column-wise: R2 higher-is-better; MAE/RMSE/WAPE lower-is-better
            _agg_norm = _agg.copy().astype(float)
            for _m in _metrics:
                _vals = _agg[_m]
                _mn, _mx = _vals.min(), _vals.max()
                if _mn == _mx:
                    _agg_norm[_m] = 0.5
                elif _m == 'R2':
                    _agg_norm[_m] = (_vals - _mn) / (_mx - _mn)
                else:
                    _agg_norm[_m] = 1 - (_vals - _mn) / (_mx - _mn)

            # Sort rows by mean normalised score, best first
            _agg_norm = _agg_norm.loc[_agg_norm.mean(axis=1).sort_values(ascending=False).index]
            _agg_annot = _agg.reindex(_agg_norm.index).round(3)

            _title = f"Energy Curves — {split} ({agg})  |  sMAE & sRMSE standardised by per-sensor std, WAPE scale-free"
            _fig, _ax = plt.subplots(figsize=(len(_metrics) * 4, max(4, len(_agg_norm) * 0.7)))
            sns.heatmap(_agg_norm, annot=_agg_annot, fmt='', cmap='RdYlGn', vmin=0, vmax=1,
                        linewidths=0.5, ax=_ax, annot_kws={'size': 10},
                        cbar_kws={'label': 'Normalised score (1 = best)'})
            _ax.set_title(_title, fontsize=12, fontweight='bold')
            _ax.set_xlabel('Metric')
            _ax.set_ylabel('Approach')
            _ax.set_xticklabels(_ax.get_xticklabels(), rotation=0, fontsize=11)
            _ax.set_yticklabels(_ax.get_yticklabels(), rotation=0, fontsize=9)
            plt.tight_layout()
            _fname = os.path.join(save_dir, f'energy_{split.lower()}_{agg}.png')
            plt.savefig(_fname, dpi=150, bbox_inches='tight')
            plt.show()
            print(f"Saved {_fname}")

        if '_energy_results_dir' in globals():
            display(Markdown("---"))
            display(Markdown("# 📊 Energy Prediction Heatmaps"))
            display(Markdown(
                "*Rows: approaches — Columns: sMAE / sRMSE / WAPE — "
                "sMAE and sRMSE standardised by per-sensor std (estimated from R²), "
                "all three metrics are scale-free and comparable across sensors*"
            ))
            for _split in sorted(_export_df['Split'].unique()):
                display(Markdown(f"## {_split}"))
                _plot_curve_energy_heatmap(_export_df, _split, 'median', _energy_results_dir)
                _plot_curve_energy_heatmap(_export_df, _split, 'mean',   _energy_results_dir)

        _sw = globals().get('_summary_wide')
        if _sw is not None and not _sw.empty:
            _sw_path = os.path.join(_run_dir, 'summary_train_test.parquet')
            _sw.reset_index().to_parquet(_sw_path, index=False)
            print(f"Saved summary  → {_sw_path}")

        _as = globals().get('_appr_summary')
        if _as is None or _as.empty:
            # rebuild from raw records if the display block didn't produce it
            _as_raw = (
                _export_df
                .groupby(['Approach', 'Split'])[['MAE', 'RMSE', 'WAPE', 'R2']]
                .median()
                .round(4)
                .unstack('Split')
            )
            _as_raw.columns = [f'{m}_{s}' for m, s in _as_raw.columns]
            _a_tr = [c for c in _as_raw.columns if c.endswith('_TRAIN')]
            _a_te = [c for c in _as_raw.columns if c.endswith('_TEST')]
            _as = _as_raw[_a_tr + _a_te]
        if not _as.empty:
            _as_path = os.path.join(_run_dir, 'summary_by_approach.parquet')
            _as.reset_index().to_parquet(_as_path, index=False)
            print(f"Saved approach summary → {_as_path}")
            display(Markdown("## Approach Summary (saved to parquet)"))
            display(_as)
    else:
        print("No curve evaluation results found — skipping parquet export.")

    # ── Profile summary ───────────────────────────────────────────────────────
    _psdf = globals().get('profile_summary_df')
    if _psdf is not None and not _psdf.empty:
        _profile_path = os.path.join(_run_dir, 'profile_summary.parquet')
        _psdf.to_parquet(_profile_path, index=False)
        print(f"Saved profile  → {_profile_path}")
    else:
        print("No profile summary found — skipping profile parquet.")

    # ── Run metadata ─────────────────────────────────────────────────────────
    pd.DataFrame({
        'run_timestamp': [_run_ts],
        'approaches':    [str(APPROACHES)],
    }).to_parquet(os.path.join(_run_dir, 'run_meta.parquet'), index=False)

    # ── HTML export of this notebook ─────────────────────────────────────────
    _this_file = os.path.abspath(__file__)
    _html_path = os.path.join(_run_dir, 'notebook.html')
    try:
        _nb_result = subprocess.run(
            ['jupyter', 'nbconvert', '--to', 'html', '--output', _html_path, _this_file],
            capture_output=True, text=True, timeout=120,
        )
        if _nb_result.returncode == 0:
            print(f"Saved HTML     → {_html_path}")
        else:
            print(f"nbconvert warning: {_nb_result.stderr.strip()[:300]}")
    except Exception as _e:
        print(f"HTML export skipped: {_e}")

    print(f"\nAll outputs in: {os.path.abspath(_run_dir)}")
    print(f"  plots/  → {os.path.join(_run_dir, 'plots')}")

    # ── info.json ─────────────────────────────────────────────────────────────
    import json as _json
    _info = {
        'run_name': _run_name,
        'run_timestamp': _run_ts,
        'data_experiment': experiment,
        'temporal_resolution': TEMPORAL_RESOLUTION,
        'processes': processes_to_run,
        'approaches': APPROACHES,
    }
    _info_path = os.path.join(_run_dir, 'info.json')
    with open(_info_path, 'w') as _f:
        _json.dump(_info, _f, indent=2)
    print(f"Saved info     → {_info_path}")

    _total_elapsed = _time.perf_counter() - _pipeline_start
    _h, _rem = divmod(int(_total_elapsed), 3600)
    _m, _s   = divmod(_rem, 60)
    print(f"\n{'='*60}")
    print(f"  Pipeline finished in {_h:02d}h {_m:02d}m {_s:02d}s  ({_total_elapsed:.1f}s total)")
    print(f"{'='*60}")

# %%