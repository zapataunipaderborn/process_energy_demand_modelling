
# %%

import os

os.environ["TQDM_DISABLE"] = "1"

os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

import logging
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import jensenshannon
import pm4py
import tempfile
import os
import matplotlib.image as mpimg
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
import random
import os

# Master seed for the whole run. 01_pipeline.py exports PIPELINE_RANDOM_SEED (and
# pins PYTHONHASHSEED to the same value) so a run is reproducible end to end;
# sim_extractor.set_global_seeds covers python/numpy/torch, and every parallel
# training task derives its own seed from this one via stable_seed().
RANDOM_SEED = int(os.environ.get('PIPELINE_RANDOM_SEED', '42'))
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
import plotly.io as pio
from pathlib import Path

pio.renderers.default='notebook'
pd.options.mode.chained_assignment = None
from IPython.display import display, Markdown

# Force pm4py to hide progress bars
from pm4py.util import constants
constants.SHOW_PROGRESS_BAR = False

# %%
import pandas as pd
from utils.sim_extractor import extract_process, MIN_CURVE_SAMPLES
from utils.sim_extractor import set_global_seeds, GLOBAL_RANDOM_SEED

# Seed python/numpy/torch from the master seed now that sim_extractor is
# importable. Parallel training workers re-seed themselves per combo.
set_global_seeds(RANDOM_SEED)
if GLOBAL_RANDOM_SEED != RANDOM_SEED:
    print(f"[modelling] WARNING: sim_extractor read seed {GLOBAL_RANDOM_SEED} but this "
          f"run uses {RANDOM_SEED} — PIPELINE_RANDOM_SEED changed after import.")
if os.environ.get('PYTHONHASHSEED') is None:
    print("[modelling] NOTE: PYTHONHASHSEED is unset. Runs launched through "
          "01_pipeline.py pin it; a direct `python 02_modelling.py` does not, so "
          "str-set iteration order may differ between runs.")
from utils.simulation import ProcessSimulation

from utils.sim_extractor import compare_complete_case_curves, build_exog_lookup
from utils.sim_extractor import build_sensor_activity_object_combos
from utils.sim_extractor import predict_raw_curve_step_dtw
from utils.sim_extractor import (
    build_case_level_curves,
    fit_stochastic_profile_generator, fit_bootstrap_profile_generator,
    compare_schedule_and_stochastic_profiles,
    train_case_duration_pipeline, predict_case_duration, rescale_case_curve_to_duration,
    SAVE_TRAINED_MODELS, save_trained_pipelines,
)
from xgboost import XGBRegressor

# %%

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

# Add temporal EF features derived from datetime_energy before any aggregation
for _pname, _pdata in process_datasets.items():
    _exp = _pdata.get('expanded')
    if _exp is not None and 'datetime_energy' in _exp.columns:
        _dt = pd.to_datetime(_exp['datetime_energy'])
        _exp['ef_hour_of_day'] = _dt.dt.hour.astype(float)
        _exp['ef_day_of_week'] = _dt.dt.dayofweek.astype(float)
        print(f"  [{_pname}] ef_hour_of_day range {_exp['ef_hour_of_day'].min():.0f}–{_exp['ef_hour_of_day'].max():.0f}, "
              f"ef_day_of_week range {_exp['ef_day_of_week'].min():.0f}–{_exp['ef_day_of_week'].max():.0f}")
print("Added ef_hour_of_day and ef_day_of_week to all expanded datasets.")

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
processes_to_run = _env_processes.split(',') if _env_processes else ['process_1', 'process_2', 'process_3', 'process_4_1', 'process_4_2', 'process_5']

print(f"Processes to run: {processes_to_run}")

process_datasets_to_model = process_datasets


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
#
#   → MINING_ALGORITHM:  only affects 'statistical' and the ml* modes.
#                        energy-aware modes ignore it — they auto-pick the
#                        best miner from the petri_net_* results.
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

# ── ML+ helpers (mirrors duration_prediction_benchmark.ipynb) ─────────────────
_MLP_ENG_COLS = [
    'feat_hour', 'feat_dayofweek', 'feat_month',
    'feat_act_pos', 'feat_prev_dur', 'feat_prev_dur2', 'feat_case_elapsed',
]

# ef_* series kept OUT of the ML+ duration features because _MLP_ENG_COLS
# already carries the same quantity: ef_hour_of_day / ef_day_of_week (added to
# every expanded frame further up) are window-averaged copies of feat_hour /
# feat_dayofweek. Feeding both makes the design matrix collinear, which flips
# the CV winner in _mlp_fit_model_with_oof from a tree to HuberRegressor and
# lets that linear fit extrapolate to negative minutes on the test period —
# clipped to 0, then floored at 0.1 min by the simulator. Measured 2026-07-24
# (exp 967 vs 964), process_4_1 global model: 50.7% of test rows predicted <= 0
# and a mean of 2509 min against a real mean of 70. Because petri_net_budget
# stops on elapsed TIME, those 0.1-min activities made it fire 25.7 events per
# case against 13.3 real, which is what pushed budget/ml_local below plain
# budget on span and event ratio. Dropping these two restores the tree fit
# (0% non-positive, mean 92 vs 90.5 real). The genuine weather series stay in.
_MLP_EF_DUPLICATE_COLS = ('ef_hour_of_day', 'ef_day_of_week')


def _mlp_flatten_object_attributes(df):
    if 'object_attributes' not in df.columns:
        return df
    import ast as _ast
    def _parse(x):
        if isinstance(x, dict): return x
        if isinstance(x, str):
            try: return _ast.literal_eval(x)
            except: return {}
        return {}
    attrs   = df['object_attributes'].apply(_parse)
    attr_df = pd.json_normalize(attrs.tolist()).add_prefix('attr_')
    attr_df.index = df.index
    return pd.concat([df.drop(columns=['object_attributes']), attr_df], axis=1)


def _mlp_add_features(df):
    out = df.sort_values(['case_id', 'timestamp_start']).copy()
    out['feat_hour']      = out['timestamp_start'].dt.hour
    out['feat_dayofweek'] = out['timestamp_start'].dt.dayofweek
    out['feat_month']     = out['timestamp_start'].dt.month
    out['feat_act_pos']   = out.groupby('case_id').cumcount()
    out['feat_prev_dur']  = out.groupby('case_id')['duration'].shift(1).fillna(0.0)
    out['feat_prev_dur2'] = out.groupby('case_id')['duration'].shift(2).fillna(0.0)
    out['feat_case_elapsed'] = (
        out['timestamp_start']
        - out.groupby('case_id')['timestamp_start'].transform('min')
    ).dt.total_seconds() / 60.0
    return out


def _mlp_fit_model_with_oof(sub, feat_cols, n_splits=5):
    """Select best sklearn model by CV MAE. Returns (model, scaler, name), oof_resid.

    Trained on the RAW duration in minutes. A squared-error regressor on the raw
    scale predicts the arithmetic MEAN, which is what budget mode needs
    (count = budget / duration); the log1p target this used to offer predicted the
    geometric mean (~median) instead, under-filling the budget on right-skewed
    activities (process_5's main activity: real mean 29 vs median 8) and making
    budget_ml_plus over-generate events 1.5-2.7x. It also amplified bad
    extrapolation exponentially. Removed 2026-07-24.
    """
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge, HuberRegressor
    from sklearn.ensemble import RandomForestRegressor

    if len(sub) < 20 or not feat_cols:
        return None, None

    X = sub[feat_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
    y = sub['duration'].values
    n = len(sub)
    k = min(n_splits, n // 2)
    if k < 2:
        return None, None

    y_t = y.copy()
    inv = lambda p: np.clip(p, 0, None)

    candidates = {
        'ridge': lambda: Ridge(alpha=1.0),
        'huber': lambda: HuberRegressor(epsilon=1.35, max_iter=300),
        'rf':    lambda: RandomForestRegressor(n_estimators=50, max_depth=5,
                                               min_samples_leaf=3, random_state=42),
        'xgb':   lambda: XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05,
                                       subsample=0.8, random_state=42, verbosity=0),
    }
    folds = list(KFold(n_splits=k, shuffle=False).split(X))
    cand_scores, cand_oofs = {}, {}

    for name, make_m in candidates.items():
        oof, scores = np.full(n, np.nan), []
        for tr_idx, val_idx in folds:
            if len(tr_idx) < 2:
                continue
            sc_f, m_f = StandardScaler(), make_m()
            try:
                m_f.fit(sc_f.fit_transform(X[tr_idx]), y_t[tr_idx])
                pred_t = m_f.predict(sc_f.transform(X[val_idx]))
                scores.append(np.mean(np.abs(pred_t - y_t[val_idx])))
                oof[val_idx] = y[val_idx] - inv(pred_t)
            except Exception:
                pass
        if scores:
            cand_scores[name] = np.mean(scores)
            cand_oofs[name]   = oof

    if not cand_scores:
        return None, None

    best_name   = min(cand_scores, key=cand_scores.get)
    _best_oof   = cand_oofs[best_name]
    _mask       = ~np.isnan(_best_oof)
    valid_resid = _best_oof[_mask]

    # Mean-bias calibration (5th tuple element). The candidate models are
    # selected by MAE and include a robust regressor, so the point prediction
    # tracks the MEDIAN; on right-skewed activities that sits well below the
    # arithmetic MEAN. Budget mode needs the mean (count = budget / duration),
    # so an uncorrected prediction makes it over-generate events (measured:
    # process_5 duration predicted 0.37x real -> event ratio 2.8x, while the
    # equally-skewed but well-fit process_4 stays ~1.0x). Correct with the
    # OUT-OF-FOLD ratio mean(actual)/mean(oof_pred) -- OOF, not in-sample, so
    # it captures the generalisation bias the simulation actually sees, and it
    # self-targets: ~1.0 for a well-fit activity, larger only where the model
    # genuinely under-predicts. inv(pred) = actual - residual, per row.
    calib = 1.0
    if _mask.sum() >= 5:
        _actual   = y[_mask]
        _mean_pred = float(np.mean(_actual - valid_resid))
        if _mean_pred > 1e-9:
            calib = float(np.clip(float(np.mean(_actual)) / _mean_pred, 0.5, 5.0))

    from sklearn.preprocessing import StandardScaler as _SS
    sc = _SS()
    m  = candidates[best_name]()
    m.fit(sc.fit_transform(X), y_t)

    oof_out = valid_resid if len(valid_resid) >= 5 else None
    # Tuple: (model, scaler, name, mean_calib). mean_calib re-centers the
    # median-like prediction onto the arithmetic mean (see above).
    return (m, sc, best_name, calib), oof_out


def _mlp_train_models(df_train, expanded_df=None, ef_expanded_df=None):
    """Train ML+ global and per-act models from the training event log.

    expanded_df: optional per-timestamp frame carrying the ef_* external-factor
    series. When given, each activity instance gets one feat_ef_* column per
    series — the mean over [activity start, start + median duration of that
    activity) — via sim_extractor.ExternalFactorWindows. The event log itself
    carries no ef_* columns, so without this the duration models see only
    calendar features (hour/dow/month) and never the external factors.

    ef_expanded_df: frame the ExternalFactorWindows lookup is BUILT from,
    defaulting to expanded_df. Pass the unsplit series here: the windows object
    is handed to the simulator, which queries it at TEST-period timestamps, and
    a train-only lookup has no samples there — window_means then falls back to
    the column mean, so every simulated activity is predicted at "average
    weather" while the model was fitted on real per-activity values (measured
    2026-07-24: process_4_1 train ends 2025-03-17, test runs to 2025-05-31, so
    all 48 test cases were generated at a constant 3.62 °C). The ef_* series are
    exogenous (weather), not process observations, so reading them past the
    split is not target leakage — the curve evaluations already do exactly this
    through sim_extractor.build_exog_lookup. Training-row features are
    unaffected: those timestamps are inside the train period either way.

    Returns:
        global_mlp_tuple: (model, scaler, name) or None
        act_mlp_models:   {activity: (model, scaler, name)}
        mlp_feat_cols:    ordered feature column list
        activity_means:   {activity: mean_duration}
        global_mean:      float
        ef_windows:       ExternalFactorWindows or None (the simulator needs the
                          same object to rebuild these features at predict time)
    """
    df = df_train.copy()
    df['timestamp_start'] = pd.to_datetime(df['timestamp_start'])
    df['timestamp_end']   = pd.to_datetime(df['timestamp_end'])
    df['duration'] = (df['timestamp_end'] - df['timestamp_start']).dt.total_seconds() / 60.0
    df = df[df['duration'] > 0].copy()

    df = _mlp_flatten_object_attributes(df)
    df = _mlp_add_features(df)

    activity_means = df.groupby('activity')['duration'].median().to_dict()
    global_mean    = float(df['duration'].median())

    df['feat_act_mean_dur'] = df['activity'].map(activity_means).fillna(global_mean)

    # ── External factors (ef_*) ───────────────────────────────────────────
    # Joined from the expanded frame, since the event log has none. Window =
    # the activity's median duration, so the feature never encodes the target
    # and is reproducible at generation time (see ExternalFactorWindows).
    ef_windows = None
    ef_feat_cols = []
    if expanded_df is not None and not expanded_df.empty:
        from utils.sim_extractor import ExternalFactorWindows
        _ef_src = ef_expanded_df if (ef_expanded_df is not None
                                     and not ef_expanded_df.empty) else expanded_df
        _ef_cols = [c for c in expanded_df.columns
                    if c.startswith('ef_')
                    and pd.api.types.is_numeric_dtype(expanded_df[c])
                    and c not in _MLP_EF_DUPLICATE_COLS]
        if _ef_cols and 'datetime_energy' in expanded_df.columns:
            ef_windows = ExternalFactorWindows(_ef_src, _ef_cols)
            ef_feat_cols = ef_windows.feature_names
            if ef_feat_cols:
                _starts = df['timestamp_start'].astype('int64').to_numpy() / 1e9
                _windows = (df['activity'].map(activity_means)
                            .fillna(global_mean).to_numpy(dtype=float) * 60.0)
                _rows = [ef_windows.window_means(s, w) for s, w in zip(_starts, _windows)]
                for col in ef_feat_cols:
                    df[col] = [r.get(col, np.nan) for r in _rows]
                print(f"  ML+ duration features: added {len(ef_feat_cols)} external-factor "
                      f"column(s) {ef_feat_cols}")

    attr_cols = [c for c in df.columns if c.startswith('attr_')]
    numeric_attr_cols = [
        c for c in attr_cols
        if pd.to_numeric(df[c], errors='coerce').notna().mean() > 0.5
    ]
    _dropped_attrs = [c for c in attr_cols if c not in numeric_attr_cols]
    if _dropped_attrs:
        print(f"  ML+ duration features: dropped {len(_dropped_attrs)} non-numeric "
              f"attribute(s) {_dropped_attrs} — not encoded, so they carry no signal.")
    mlp_feat_cols = [
        c for c in (numeric_attr_cols + _MLP_ENG_COLS + ['feat_act_mean_dur'] + ef_feat_cols)
        if c in df.columns
    ]

    global_mlp_tuple, _ = _mlp_fit_model_with_oof(df, mlp_feat_cols)

    act_mlp_models = {}
    for act, sub in df.groupby('activity'):
        tpl, _ = _mlp_fit_model_with_oof(sub, mlp_feat_cols)
        if tpl is not None:
            act_mlp_models[act] = tpl

    return (global_mlp_tuple, act_mlp_models, mlp_feat_cols,
            activity_means, global_mean, ef_windows)


MODES_TO_COMPARE = [
    'petri_net_alpha',
    'petri_net_alpha_ml_plus_global',
    'petri_net_alpha_ml_plus_per_act',
    'petri_net_heuristic',
    'petri_net_heuristic_ml_plus_global',
    'petri_net_heuristic_ml_plus_per_act',
    'petri_net_inductive',
    'petri_net_inductive_ml_plus_global',
    'petri_net_inductive_ml_plus_per_act',
    'petri_net_budget',   # base PN token game, but each case is generated to match its predicted total-duration budget (train_case_duration_pipeline) — folds the "duration-corrected" span-fix into the generator (no post-hoc step)
    'petri_net_budget_ml_plus_global',   # petri_net_budget + ML+ global duration model (better internal timing under the same total-duration budget)
    'petri_net_budget_ml_plus_per_act',  # petri_net_budget + ML+ per-activity duration models
    # 'petri_net_ilp',
    # 'petri_net_ilp_ml_plus_global',
    # 'petri_net_ilp_ml_plus_per_act',


    # Best-net selection performed during training by the pipeline itself:
    # argmax over the miner modes of the TRAIN mean of Fitness/Precision/
    # Generalization/Simplicity (_combined_selection_score). Emits a
    # 'petri_net_combined' row carrying selected_mining_algorithm, so the
    # choice is recorded in the results instead of being recomputed at
    # reporting time by 03_results_process.ipynb.
    'petri_net_combined',
    # 'petri_net_combined_ml_plus_global',
    # 'petri_net_combined_ml_plus_per_act',
    # 'petri_net_median_duration',
    #'petri_net_statistical',
    #'petri_net_statistical_memory',
    #'ml_duration_only',
    #'ml_duration_only_with_activity_past',
    #'ml_duration_only_with_activity_past_point_estimate',
    #'ml_global_model',
]


# ─────────────────────────────────────────────────────────────────────────────
# PROCESS MINING ALGORITHM
#   'inductive'  – pm4py Inductive Miner → guarantees a sound Petri net
#   'heuristic'  – pm4py Heuristics Miner → better noise filtering
#   'alpha'      – pm4py Alpha Miner → classic algorithm
#   'ilp'        – pm4py ILP Miner → precise/sound, can be strict
#   'manual'     – original manual extraction (no process mining)
# ─────────────────────────────────────────────────────────────────────────────
MINING_ALGORITHM = os.environ.get('PIPELINE_DEFAULT_MINING_ALGORITHM', 'heuristic')

# Petri-net miner variants to compare when mode names include the algorithm.
# Override via PIPELINE_MINING_ALGORITHMS (comma-separated, e.g. "heuristic,inductive").
_env_mining_algorithms = os.environ.get('PIPELINE_MINING_ALGORITHMS')
PETRI_NET_ALGORITHMS = (
    [a.strip() for a in _env_mining_algorithms.split(',') if a.strip()]
    if _env_mining_algorithms else
    ['alpha', 'heuristic', 'inductive']#, 'ilp']
)

# ─────────────────────────────────────────────────────────────────────────────
# MINER HYPERPARAMETER OPTIMIZATION (for inductive + heuristic)
#   Runs local per-group search during extraction and keeps best model.
# ─────────────────────────────────────────────────────────────────────────────
OPTIMIZE_MINING_HYPERPARAMS = True
MINING_SEARCH_SPACE = {
    'inductive_noise_thresholds': [0.05, 0.10, 0.20, 0.30, 0.40],
    'heuristic_params_grid': [
        {'dependency_threshold': 0.30, 'and_threshold': 0.65, 'loop_two_threshold': 0.50},
        {'dependency_threshold': 0.50, 'and_threshold': 0.65, 'loop_two_threshold': 0.50},
        {'dependency_threshold': 0.70, 'and_threshold': 0.65, 'loop_two_threshold': 0.50},
        {'dependency_threshold': 0.50, 'and_threshold': 0.50, 'loop_two_threshold': 0.50},
        {'dependency_threshold': 0.50, 'and_threshold': 0.80, 'loop_two_threshold': 0.50},
    ],
    'ilp_variant_coverages': [0.80, 0.90, 0.95, 1.0],
}

# ─────────────────────────────────────────────────────────────────────────────
# CURVE MODELLING HYPERPARAMETER OPTIMIZATION
#   When True, each build_and_train_pipeline_* call runs an Optuna search over
#   regressor hyperparameters (learning rate, max_depth, n_estimators, etc.)
#   instead of using defaults.  Significantly slower but often improves fit.
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# CURVE-EVAL POPULATION FLOOR
#   Mirrors sim_extractor.MIN_CURVE_SAMPLES, the floor both curve extractors
#   now share, so training and scoring cover the same curves for every
#   approach. Kept as an explicit reporting-side guard: runs produced BEFORE
#   the extractors were unified (<= experiment_964) still contain 2-4 sample
#   curves for the non-prev-activity approaches only, and re-reading those
#   parquets without this filter would compare medians over different
#   populations. On a fresh run it is a no-op.
#   The exported curve_eval_results.parquet always keeps every scored curve.
CURVE_EVAL_MIN_POINTS = MIN_CURVE_SAMPLES

CURVE_OPTIMIZE_HYPERPARAMS = True   # ← Optuna search for sklearn curve models
CURVE_N_OPTUNA_TRIALS      = 50     # ← trials per (sensor, activity, object) combo
# Pipeline-config overrides (PIPELINE_CURVE_OPTIMIZE_HYPERPARAMS / _N_OPTUNA_TRIALS)
# so a light test run can turn the search off or cut trials without editing here.
_env_opt_hp = os.environ.get('PIPELINE_CURVE_OPTIMIZE_HYPERPARAMS')
if _env_opt_hp is not None:
    CURVE_OPTIMIZE_HYPERPARAMS = _env_opt_hp.lower() == 'true'
_env_opt_trials = os.environ.get('PIPELINE_CURVE_N_OPTUNA_TRIALS')
if _env_opt_trials:
    CURVE_N_OPTUNA_TRIALS = int(_env_opt_trials)

# %%
# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION  —  edit everything here, nothing else needs to change
# ══════════════════════════════════════════════════════════════════════════════

# ── Train / test split ────────────────────────────────────────────────────────
TEMPORAL_SPLIT      = True    # True → split cases into train/test; False → use all data
# How cases are assigned to train vs. test within TEMPORAL_SPLIT. 'temporal'
# (default): earliest train_ratio fraction of cases by start time → train, the
# rest (the "future") → test — no leakage, mirrors deployment. 'random': a
# fixed-seed shuffle of case IDs → train/test — same ratio, no time ordering.
# Override via PIPELINE_SPLIT_TYPE=temporal|random. Consumed by
# _split_process_datasets, applied identically to every evaluation (process,
# energy/profile, schedule-profile) since they all key off the same case-id
# partition — see that function's docstring.
SPLIT_TYPE          = os.environ.get('PIPELINE_SPLIT_TYPE', 'temporal').lower()
if SPLIT_TYPE not in ('temporal', 'random'):
    raise ValueError(f"PIPELINE_SPLIT_TYPE must be 'temporal' or 'random', got {SPLIT_TYPE!r}")
RANDOM_SPLIT_SEED   = int(os.environ.get('PIPELINE_RANDOM_SPLIT_SEED', str(RANDOM_SEED)))
# Fraction of cases used for training. Override via PIPELINE_TRAIN_RATIO=0.8 (etc).
TRAIN_RATIO         = float(os.environ.get('PIPELINE_TRAIN_RATIO', '0.70'))

# ── Pipeline execution flags ──────────────────────────────────────────────────
RUN_TEST_EVALUATION       = True   # evaluate on held-out test set
# Energy profile/curve modelling (sensor curve fitting + curve-quality benchmark).
# Turn off to run process modelling only — skips all build_and_train_pipeline_*
# training and evaluation. Override via PIPELINE_RUN_ENERGY_MODELLING=true/false.
RUN_CURVE_ONLY_EVALUATION = os.environ.get('PIPELINE_RUN_ENERGY_MODELLING', 'true').lower() == 'true'
RUN_PROCESS_MODELLING     = os.environ.get('PIPELINE_RUN_PROCESS_MODELLING', 'true').lower() == 'true'
# Schedule Profile Evaluation — complete-profile reference generators: a
# stochastic (no-features-at-all) generator plus a bootstrap resampler,
# compared against the same real test cases used by the complete-curve eval,
# alongside the already-computed "Best" comparators. Off by default — trains
# one extra model per (process, sensor), on top of everything else. Does NOT
# affect any other evaluation. See utils/sim_extractor.py's "Schedule Profile
# Evaluation" section.
RUN_SCHEDULE_PROFILE_EVAL = os.environ.get('PIPELINE_RUN_SCHEDULE_PROFILE_EVAL', 'false').lower() == 'true'

# Autoregressive test-time rollout of the prev-activity curve approaches
# (the "… (autoreg)" rows in the Curve-Only Evaluation). Off by default —
# it only ADDS those extra comparison rows and does not affect any other
# evaluation. See _run_curve_eval_autoregressive_prev_act.
RUN_AUTOREGRESSIVE_EVAL = os.environ.get('PIPELINE_RUN_AUTOREGRESSIVE_EVAL', 'false').lower() == 'true'

# Whether to persist the actual real/predicted curve arrays behind the
# complete-curve and schedule-profile Wasserstein numbers (not just the
# aggregated distances) — one row per (case_id, sensor, series, timestep), so
# they can be reloaded later to compute other metrics or plot without
# re-simulating anything. Off by default — adds a parquet file per (process,
# mode) for complete-curve eval, and one per process for schedule-profile
# eval. See compare_complete_case_curves / compare_schedule_and_stochastic_profiles.
SAVE_PREDICTED_CURVES = os.environ.get('PIPELINE_SAVE_PREDICTED_CURVES', 'false').lower() == 'true'

# Restrict the COMPLETE-CURVE eval (and with it the schedule-profile eval's
# "Best, mine" comparator) to a subset of the trained approaches. Unset ->
# every trained approach is evaluated, the historical behaviour. Set e.g. to
# 'ml_step_dtw' to assemble complete case profiles for that one approach only —
# the complete-curve loop is (process x mode x approach) inference over every
# simulated case, so each dropped approach cuts that stage's runtime by a full
# share. The schedule-profile comparator prefers 'ml_external' when it is in
# the list (the historical column), otherwise the FIRST listed approach.
_ccap_env = os.environ.get('PIPELINE_COMPLETE_CURVE_APPROACHES')
COMPLETE_CURVE_APPROACHES = ([a.strip() for a in _ccap_env.split(',') if a.strip()]
                             if _ccap_env else None)

# Upper bound on worker processes for the parallel training pools. Default: one
# per core, the historical behaviour.
#
# This matters more than a core count suggests. 02_modelling.py is a SCRIPT with no
# `if __name__ == "__main__"` guard, and ProcessPoolExecutor uses the 'spawn'
# start method on macOS, so every worker RE-IMPORTS this module and re-executes
# it top to bottom — including reloading the process data. Peak memory is
# therefore roughly (workers + 1) x the resident size of one full load, not the
# one copy a fork-based pool would share. On a machine with many cores and
# little RAM the pool is killed with no traceback, which looks exactly like a
# silent hang. Set PIPELINE_MAX_WORKERS to fit the pool to available memory
# rather than to the CPU.
_MAX_WORKERS_ENV = os.environ.get('PIPELINE_MAX_WORKERS')
MAX_WORKERS = int(_MAX_WORKERS_ENV) if _MAX_WORKERS_ENV else None
if MAX_WORKERS is not None:
    print(f"[modelling] Worker pools capped at {MAX_WORKERS} process(es) "
          f"(PIPELINE_MAX_WORKERS); machine has {os.cpu_count()} core(s).")


def _report_numba_fork_safety():
    """
    Print, at startup, which numba threading layer this run will fork with.

    Config only — this deliberately does NOT execute a numba kernel, because
    doing so is precisely what arms the unsafe layer's atfork handler. 'workqueue'
    is fork-safe; 'omp' kills every child at fork; 'tbb' is not fork-safe either.
    A run that prints anything other than a fork-safe layer here WILL lose its
    worker pools — stop it and fix the environment rather than wait for it.
    """
    _layer = os.environ.get('NUMBA_THREADING_LAYER', '(unset)')
    _safe  = _layer in ('workqueue', 'forksafe')
    print(f"[modelling] numba threading layer: {_layer} "
          f"({'fork-safe' if _safe else '⚠️  NOT FORK-SAFE — worker pools will die at fork'})",
          flush=True)
    return _safe


_report_numba_fork_safety()


def _pool_workers(n_tasks):
    """Worker count for a pool over `n_tasks`: one per core, capped by
    PIPELINE_MAX_WORKERS when set, and never more than there is work for."""
    n = min(int(n_tasks), os.cpu_count() or 4)
    if MAX_WORKERS is not None:
        n = min(n, MAX_WORKERS)
    return max(1, n)


# Seconds a worker pool may go with NO task completing before it is declared
# deadlocked. The seq2seq pool has now wedged twice — experiment_986 (7h) and
# experiment_2_20260730 (11.5h) — with the identical signature: every worker
# alive, ~4s of CPU each, all threads parked in futex_wait, no output at all.
#
# The root cause is NOT known. The first attempt blamed CUDA fork-poisoning and
# hid the GPUs from the child (01_pipeline.py); the hang came back with that env
# confirmed in place, so that diagnosis was wrong. It has never reproduced
# outside a full run. Until it can be caught in the act, this does not pretend to
# fix it — it bounds the damage: no progress for this long means dump the workers'
# stacks, kill the pool, and retry rather than wait forever.
POOL_STALL_TIMEOUT = float(os.environ.get('PIPELINE_POOL_STALL_TIMEOUT', '1800'))

# How often a running pool reports progress. Without this a pool that is working
# normally is indistinguishable from a dead one for however long its slowest task
# takes — experiment_1_20260801 spent 80 silent minutes in the inline fallback and
# looked hung. Every heartbeat also samples worker CPU, which is what the stall
# rule keys off (see _pool_worker_cpu_seconds).
POOL_HEARTBEAT_SECONDS = float(os.environ.get('PIPELINE_POOL_HEARTBEAT', '300'))


def _kill_pool(pool):
    """
    Tear down a wedged pool. shutdown() alone is not enough: it joins the worker
    processes, and workers stuck in futex_wait never exit, so the join is what
    turned a deadlock into an 11-hour hang. SIGKILL first, then shut down.

    Nothing in here may raise: it runs from a `finally` on the stall path, and an
    exception escaping it would abandon a pool full of live workers — which is
    precisely the state we are here to get out of.
    """
    try:
        _procs = list((getattr(pool, '_processes', None) or {}).values())
    except Exception:
        _procs = []
    for _p in _procs:
        try:
            _p.kill()
        except Exception:
            pass
    for _p in _procs:
        try:
            _p.join(timeout=5)   # reap, so the interpreter does not linger on exit
        except Exception:
            pass
    try:
        pool.shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass


def _pool_worker_cpu_seconds(pool):
    """
    Total CPU seconds burnt so far by this pool's workers (utime+stime, /proc).

    This is what tells a DEADLOCK apart from work that is merely SLOW, and the
    two look identical from the outside: neither completes a task. A wedged
    worker burns no CPU (experiment_986: ~4s each, then parked in futex_wait); a
    worker training a transformer burns a full core. Counting completed tasks
    cannot distinguish them — see _run_pool_with_stall_watchdog.
    """
    _total = 0.0
    _hz = os.sysconf('SC_CLK_TCK')
    for _pid in list(getattr(pool, '_processes', {}) or {}):
        try:
            with open(f'/proc/{_pid}/stat', 'rb') as _fh:
                # Split on the last ')' — comm may itself contain spaces/parens.
                _f = _fh.read().rsplit(b')', 1)[-1].split()
            _total += (int(_f[11]) + int(_f[12])) / _hz   # utime + stime
        except (OSError, IndexError, ValueError):
            pass
    return _total


def _dump_worker_stacks(pool, label):
    """
    SIGUSR1 every live worker. Workers install a faulthandler trap
    (utils.sim_extractor._pool_stall_trap_initializer) that writes all thread
    stacks to PIPELINE_STALL_TRACE_DIR, so a wedged pool can be diagnosed after
    the fact. This is the only route available: yama ptrace_scope=1 on this host
    denies py-spy and gdb without root, so the process has to incriminate itself.
    """
    import signal as _signal
    _dir = os.environ.get('PIPELINE_STALL_TRACE_DIR') or tempfile.gettempdir()
    _pids = list(getattr(pool, '_processes', {}) or {})
    for _pid in _pids:
        try:
            os.kill(_pid, _signal.SIGUSR1)
        except OSError:
            pass
    _time.sleep(2)  # let the handlers finish writing before the SIGKILL below
    print(f"  🔍 {label}: stack dumps for {len(_pids)} worker(s) written to {_dir} "
          f"(pids: {', '.join(str(p) for p in _pids)})")


def _run_pool_with_stall_watchdog(fn, tasks, n_workers, label,
                                  stall_timeout=None, attempts=2):
    """
    Run `fn(*args)` over `tasks` in a process pool that cannot hang forever.

    tasks     — list of (key, args_tuple); `key` only labels the task in messages
    attempts  — how many times to try a fresh pool before giving up on forking

    A pool that completes NOTHING for `stall_timeout` seconds is treated as
    deadlocked: its workers are dumped and killed, and their tasks are retried in
    a brand-new pool. Whatever still stalls after the last attempt is run inline
    in this process — slow, but it terminates, and an in-process run cannot
    inherit whatever fork-time state wedges the workers.

    A pool whose workers DIE (BrokenProcessPool) is treated the same way. That is
    the other half of the same problem and it cost experiment_1_20260731 the whole
    run: numba's OpenMP layer kills every child forked after numba has run in the
    parent, so all 60 tasks failed in 1.7s. Those are not "stalls", so the old code
    retried nothing, returned nothing, and then hung forever in shutdown(wait=True)
    — the executor's queue-feeder threads were blocked in pipe_write against a call
    queue whose readers were all dead, and shutdown() joins them. Hence: a broken
    pool is retried like a stalled one, and is torn down with _kill_pool (which
    never joins) rather than shutdown(wait=True).

    Returns a list of (key, result) for every task that produced one.
    """
    import concurrent.futures
    from concurrent.futures.process import BrokenProcessPool
    stall_timeout = POOL_STALL_TIMEOUT if stall_timeout is None else stall_timeout
    from utils.sim_extractor import _pool_stall_trap_initializer

    remaining = list(tasks)
    results = []
    for _attempt in range(1, attempts + 1):
        if not remaining:
            break
        if _attempt > 1:
            print(f"  ↻ {label}: retrying {len(remaining)} task(s) in a fresh pool "
                  f"(attempt {_attempt}/{attempts})...")
        _stalled_keys = []
        _broken_keys = []
        _pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=max(1, min(n_workers, len(remaining))),
            initializer=_pool_stall_trap_initializer, initargs=(label,))
        try:
            # Submit one at a time, not as a comprehension: once a worker dies the
            # executor marks itself broken and every LATER submit() raises too. In a
            # comprehension that exception escapes mid-build, skipping straight to
            # the finally below with no keys recorded — i.e. the tasks are silently
            # dropped and the pool is shut down with the joining path.
            _futs = {}
            for _key, _args in remaining:
                try:
                    _futs[_pool.submit(fn, *_args)] = _key
                except BrokenProcessPool as _bpe:
                    if not _broken_keys:
                        print(f"  💥 {label}: pool broke while submitting — "
                              f"first affected task [{_key}]: {_bpe}")
                    _broken_keys.append(_key)
            _pending = set(_futs)
            _n_total = len(_pending)
            _t_pool = _time.perf_counter()
            _cpu_prev = _pool_worker_cpu_seconds(_pool)
            _idle_for = 0.0            # seconds of NO task completion AND no CPU burnt
            _since_beat = 0.0
            while _pending:
                _tick = min(stall_timeout, POOL_HEARTBEAT_SECONDS)
                _done, _pending = concurrent.futures.wait(
                    _pending, timeout=_tick,
                    return_when=concurrent.futures.FIRST_COMPLETED)
                # CPU burnt by the workers during this tick, in "cores busy" terms.
                _cpu_now = _pool_worker_cpu_seconds(_pool)
                _cores = max(0.0, _cpu_now - _cpu_prev) / max(_tick, 1e-9)
                _cpu_prev = _cpu_now
                _since_beat += _tick

                if _done or _cores > 0.05:
                    # Either a task finished, or the workers are computing. The
                    # tail of a pool legitimately completes nothing for a long
                    # time — 30 seq2seq_iom combos on long curves ran >30min each
                    # at a full core, and the old "nothing completed in 1800s"
                    # rule killed two pools of healthy work before falling back to
                    # the slowest mode there is (sequential, in this process).
                    _idle_for = 0.0
                else:
                    _idle_for += _tick

                if _pending and _since_beat >= POOL_HEARTBEAT_SECONDS:
                    _since_beat = 0.0
                    print(f"  ⏳ {label}: {_n_total - len(_pending)}/{_n_total} done, "
                          f"{len(_pending)} running on {_cores:.1f} core(s), "
                          f"{(_time.perf_counter() - _t_pool)/60:.0f} min elapsed"
                          f"{'' if _idle_for == 0 else f' — NO cpu for {_idle_for:.0f}s'}")

                if _idle_for >= stall_timeout:
                    _stalled_keys = [_futs[_f] for _f in _pending]
                    print(f"  ⛔ {label}: {len(_stalled_keys)} task(s) outstanding and the "
                          f"workers burnt NO cpu for {stall_timeout:.0f}s — that is a real "
                          f"deadlock, not slow work. Stalled: "
                          f"{', '.join(str(k) for k in _stalled_keys[:4])}"
                          f"{' ...' if len(_stalled_keys) > 4 else ''}")
                    _dump_worker_stacks(_pool, label)
                    for _f in _pending:
                        _f.cancel()
                    break
                for _f in _done:
                    _key = _futs[_f]
                    try:
                        results.append((_key, _f.result()))
                    except BrokenProcessPool as _bpe:
                        # The pool is dead, not this task: every other future will
                        # raise the same thing. Log the first in full, then count —
                        # 60 identical tracebacks buried the cause last time.
                        if not _broken_keys:
                            print(f"  💥 {label}: worker process died — the pool is "
                                  f"broken, not the task [{_key}]: {_bpe}")
                        _broken_keys.append(_key)
                    except Exception as _e:
                        print(f"  ⚠️  {label} worker failed [{_key}]: {_e}")
        finally:
            # Never shutdown(wait=True) a pool that lost workers: that join is what
            # turned both the deadlock and the fork-abort into multi-hour hangs.
            # `_pool._broken` is the belt-and-braces case — an unexpected exception
            # escaping the try must not reach the joining path either.
            if _stalled_keys or _broken_keys or getattr(_pool, '_broken', None):
                _kill_pool(_pool)
            else:
                _pool.shutdown(wait=True)
        if _broken_keys:
            print(f"  💥 {label}: {len(_broken_keys)} task(s) lost to a broken pool "
                  f"(check the run's stderr/nohup.out — a child killed at fork "
                  f"reports there, not in this log).")
        _retry_set = set(map(str, _stalled_keys)) | set(map(str, _broken_keys))
        remaining = [(_k, _a) for _k, _a in remaining if str(_k) in _retry_set]

    if remaining:
        print(f"  ↩︎ {label}: {len(remaining)} task(s) still stalled/broken after "
              f"{attempts} pool attempt(s) — running them in this process. "
              f"This is slow, but it finishes.")
        print(f"  ⚠️  {label}: this run is no longer bit-identical to a clean one. "
              f"The workers re-seed and re-thread the process they run in, so "
              f"running them inline perturbs this process's RNG stream. "
              f"Re-run once the pool stall is fixed if you need reproducibility.")
        import torch as _torch
        _prev_threads = _torch.get_num_threads()
        for _key, _args in remaining:
            try:
                results.append((_key, fn(*_args)))
            except Exception as _e:
                print(f"  ⚠️  {label} inline fallback failed [{_key}]: {_e}")
        # The workers pin this process to 1 thread and leave it there; undo that
        # so the rest of the stage does not silently run single-threaded.
        _torch.set_num_threads(_prev_threads)
        set_global_seeds(GLOBAL_RANDOM_SEED)
    return results

# ── Approaches to train — comment out any you want to skip ───────────────────
#    'baseline'          "Baseline": ONE median LEVEL per SENSOR (a flat line),
#                        pooled over all activities/objects (the coarser floor)
#    'median_activity_sensor'  "Median per Activity & Sensor": one median LEVEL
#                        per (sensor, activity, object), flat, no model
#    'ml_dtw'            DTW + position index (sklearn regressor)
#    'ml_external'  DTW + external factors (ef_* columns) + prev-activity NAME
#                   (no lagged meter values — see split_curves_with_prev_activity)
#    'seq2seq'           DTW + LSTM encoder-decoder
#    'seq2seq_only'              LSTM encoder-decoder, no DTW
#    'seq2seq_iom'       Woerrlein & Strassburger's "Iterating over Metrics":
#                        trained on vanilla targets like seq2seq_only, but the
#                        model is SELECTED by generating whole curves every 10
#                        epochs and scoring them against a softDTW barycenter
#                        (MSE + sigma length) instead of by validation loss.
#                        The faithful port of the competing paper -- see the
#                        SEQ2SEQ IOM section in sim_extractor for the deviations.
#    'ml_only'                 ML (GBM/RF), linear resample encode+decode (no DTW)
#    'ml_step_dtw'       the leaf's STEP STRUCTURE as the regression target: the
#                        DTW medoid is change-point segmented once, the
#                        breakpoints are carried onto every training curve via
#                        DTW, and the models predict per-instance segment
#                        DURATIONS and LEVELS instead of per-position values.
#                        Step edges are sharp by construction because averaging
#                        happens in parameter space, not value space.
#    'ml_step_dtw_smooth'  ml_step_dtw with the per-segment level gains
#                        interpolated between segment midpoints at reconstruction
#                        instead of applied piecewise-constant — same training,
#                        targets and models, but no jumps at segment boundaries,
#                        so a breakpoint inside a ramp no longer terraces it.
#                        A separate approach (not a flag) so the two land side
#                        by side in every results table.
APPROACHES = [
    'baseline',
    'median_activity_sensor',
    'ml_dtw',
    'ml_external',
    'ml_only',
    'ml_step_dtw',
    'ml_step_dtw_smooth',

    # ── Train/eval-gap variants of 'ml_external' ─────────────────────────────
    # Fit in barycenter space like ml_external, but with per-row weights, so the
    # difference against 'ml_external' is attributable:
    #   ml_external_wcounts  row weight = raw samples folded into that canonical
    #                        position (they are not equally informative)
    #   ml_external_wmetric  ... additionally / curve sigma, matching sMAE
    'ml_external_wcounts',
    'ml_external_wmetric',

    'seq2seq',
    'seq2seq_only',
    'seq2seq_external',
    'seq2seq_iom',
]

# Override the curve models to fit from the pipeline config
# (PIPELINE_CURVE_APPROACHES, comma-separated). Lets a light run fit only e.g.
# 'ml_external' without editing this file. Unknown names are dropped
# with a warning so a typo can't silently train nothing.
_env_curve_approaches = os.environ.get('PIPELINE_CURVE_APPROACHES')
if _env_curve_approaches:
    _requested = [a.strip() for a in _env_curve_approaches.split(',') if a.strip()]
    # Full universe of valid approach names (not just the currently-uncommented
    # defaults above) — mirrors the sklearn/seq2seq dispatch sets below.
    _known = {'baseline', 'median_activity_sensor', 'ml_dtw', 'ml_external',
              'ml_only', 'ml_step_dtw', 'ml_step_dtw_smooth',
              'seq2seq', 'seq2seq_only', 'seq2seq_external', 'seq2seq_iom',
              'ml_external_wcounts', 'ml_external_wmetric'}
    _unknown = [a for a in _requested if a not in _known]
    if _unknown:
        print(f"[modelling] WARNING: PIPELINE_CURVE_APPROACHES has names not in "
              f"the default APPROACHES list: {_unknown} — ignoring those.")
    APPROACHES = [a for a in _requested if a in _known] or APPROACHES
    print(f"[modelling] Curve approaches overridden from pipeline config: {APPROACHES}")

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
_process_results_dir = os.path.join(_run_dir, 'process_results')
_energy_results_dir  = os.path.join(_run_dir, 'energy_results')
_predicted_logs_dir  = os.path.join(_run_dir, 'predicted_logs')
_complete_curve_eval_dir = os.path.join(_run_dir, 'complete_curve_eval_results')
_schedule_profile_eval_dir = os.path.join(_run_dir, 'schedule_profile_eval_results')
os.makedirs(_process_results_dir, exist_ok=True)
os.makedirs(_energy_results_dir, exist_ok=True)
os.makedirs(_predicted_logs_dir, exist_ok=True)
os.makedirs(_complete_curve_eval_dir, exist_ok=True)
os.makedirs(_schedule_profile_eval_dir, exist_ok=True)

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
import contextlib
_pipeline_start = _time.perf_counter()

logging.info(f"Run started — output folder: {_run_dir}")
logging.info(f"Approaches: {APPROACHES}")
# Re-emit the fork-safety line now that the log exists. The first call happens at
# import time, before this redirection, so it lands in the caller's stdout (and
# with block buffering may not surface until exit) — useless for diagnosing a run
# from its own log, which is the first place anyone looks.
_report_numba_fork_safety()


# ══════════════════════════════════════════════════════════════════════════════
# RUNTIME PROFILING — every stage of the pipeline, written to runtime_*.csv
# ══════════════════════════════════════════════════════════════════════════════
# One row per timed unit of work: process mining, ML duration training, each
# simulation run, each process evaluation, curve training (per approach, summed
# over the leaves that trained it), every evaluation stage and the export.
#
# Wall clock, deliberately. The two curve-training pools run one worker process
# per (sensor, activity, object), so the per-approach seconds collected inside
# the workers are CPU-parallel: they sum to far more than the pool's own
# wall-clock row ('curve_training_pool'). Both are recorded — the per-approach
# numbers are what a "how expensive is this method?" table needs, the pool row
# is what the run actually cost. `parallel_workers` says which is which.
_RUNTIME_ROWS = []


def _record_runtime(stage, seconds, process=None, detail=None, split=None,
                    n_items=None, parallel_workers=None):
    """Append one timing row. Never raises — a broken timer must not stop a run."""
    try:
        _RUNTIME_ROWS.append({
            'stage':            str(stage),
            'process':          None if process is None else str(process),
            'detail':           None if detail is None else str(detail),
            'split':            None if split is None else str(split),
            'n_items':          None if n_items is None else int(n_items),
            'parallel_workers': None if parallel_workers is None else int(parallel_workers),
            'seconds':          float(seconds),
        })
    except Exception:
        pass


def _record_curve_training_timings(worker_results, process, pool_label):
    """
    Fold one training pool's per-leaf '_timings' dicts into one row per approach:
    total seconds spent training it across the leaves, and how many leaves that
    was. Summed CPU time across pool workers — see the note above.
    """
    _totals, _counts = {}, {}
    for _res in worker_results or []:
        for _appr, _sec in (_res.get('_timings') or {}).items():
            _totals[_appr] = _totals.get(_appr, 0.0) + float(_sec)
            _counts[_appr] = _counts.get(_appr, 0) + 1
    for _appr in sorted(_totals):
        _record_runtime('curve_training', _totals[_appr], process=process,
                        detail=f'{pool_label}:{_appr}', split='TRAIN',
                        n_items=_counts[_appr])


@contextlib.contextmanager
def _timed(stage, process=None, detail=None, split=None, n_items=None,
           parallel_workers=None):
    """Time a block and record it, including when the block raises."""
    _t_start = _time.perf_counter()
    try:
        yield
    finally:
        _record_runtime(stage, _time.perf_counter() - _t_start, process=process,
                        detail=detail, split=split, n_items=n_items,
                        parallel_workers=parallel_workers)


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION UTILITIES — HEATMAP ENGINE
# ══════════════════════════════════════════════════════════════════════════════

# The 5 short metrics + overall shown in the main heatmap (all 0 = best)
CORE_METRIC_BASES = [
    'overall_error',
    'basic_metrics_event_count_error',
    'duration_metrics_mean_duration_error',
    'duration_metrics_activity_duration_error',
    'duration_metrics_case_span_error',
    'activity_metrics_js_divergence',
    'control_flow_metrics_edge_f1_error',
    'conformance_metrics_fitness_error',
    'conformance_metrics_precision_error',
]


def _detect_sensors_for_energy_distribution(process, expanded_df):
    """
    Self-contained sensor auto-detection (mirrors the logic already used
    inside the energy-aware modes block, duplicated here so the
    energy-distribution metrics don't depend on that block having run —
    it must work for ANY simulation mode, not just petri_net_energy_*).
    """
    if expanded_df is None or expanded_df.empty:
        return []
    sensors_from_config = (
        globals()['process_datasets_to_model_sensors'].get(process, {}).get('sensors_to_model', [])
        if 'process_datasets_to_model_sensors' in globals() else []
    )
    if sensors_from_config:
        return sensors_from_config
    return [
        c for c in expanded_df.columns
        if c.endswith('_to_model')
        and expanded_df[c].dtype in ('float64', 'float32', 'int64', 'int32')
    ]


# Which global dict holds each curve-fitting approach's trained pipelines
# (see the "CURVE-ONLY TRAINING" section — each approach gets reassembled
# into its own all_energy_pipelines_<approach> dict; 'baseline' is the one
# exception, stored unsuffixed as all_energy_pipelines for backward compat).
_ENERGY_APPROACH_DICT_NAMES = {
    'baseline': 'all_energy_pipelines',
}


def _save_complete_curve_eval_metrics(process, mode_name, simulated_df, real_expanded_df,
                                      output_root, approach='baseline'):
    """
    Complete-curve (per real test case) curve reconstruction — see
    compare_complete_case_curves in utils/sim_extractor.py.

    For each real test case (matched to this mode's simulated log by
    case_id), concatenates the real per-activity curves into one complete
    real profile and the predicted (curve-fitting pipeline `approach`,
    applied to the simulated log's own activities/durations) curves into one
    complete simulated profile for that same case, and persists both curves
    to predicted_curves*.parquet for the downstream notebooks.

    Silently no-ops when there's nothing to compare against (no trained
    pipelines for this process/approach, no detectable sensor columns, or no
    shared case_ids between the real test set and this mode's simulated log).
    """
    dict_name = _ENERGY_APPROACH_DICT_NAMES.get(approach, f'all_energy_pipelines_{approach}')
    pipelines_for_process = (
        globals()[dict_name].get(process) if dict_name in globals() else None
    )
    if not pipelines_for_process:
        return
    sensors = _detect_sensors_for_energy_distribution(process, real_expanded_df)
    if not sensors or simulated_df is None or simulated_df.empty:
        return

    try:
        _result = compare_complete_case_curves(
            real_expanded_df, simulated_df, pipelines_for_process, sensors,
            activity_exog_means=globals().get('_activity_exog_means', {}),
            temporal_resolution_minutes=globals().get('TEMPORAL_RESOLUTION_MINUTES', 15.0),
            save_curves=SAVE_PREDICTED_CURVES,
        )
    except Exception as exc:
        # Traceback, not just the message. A one-line `⚠️ ... : 'NoneType' object
        # has no attribute 'get'` repeated for every (process, mode) is what let a
        # broken predict_fn closure go unnoticed for a whole 8-hour run: the stage
        # reported "failed", the downstream table showed an empty row, and there
        # was nothing on disk to say WHERE it broke.
        import traceback, textwrap as _tw
        print(f"  ⚠️ Complete-curve eval ({approach}) failed for {process}/{mode_name}: {exc}\n"
              + _tw.indent(traceback.format_exc(), '      '))
        return
    case_curve_df, curve_df = _result if SAVE_PREDICTED_CURVES else (_result, None)
    if case_curve_df.empty:
        return

    safe_mode = str(mode_name).replace(' ', '_').replace('/', '_')
    out_dir = os.path.join(output_root, process, safe_mode)
    os.makedirs(out_dir, exist_ok=True)
    suffix = '' if approach == 'baseline' else f'_{approach}'

    if curve_df is not None and not curve_df.empty:
        curve_df.to_parquet(os.path.join(out_dir, f'predicted_curves{suffix}.parquet'), index=False)
        print(f"  💾 Predicted curves ({approach}) saved → "
              f"complete_curve_eval_results/{process}/{safe_mode}/predicted_curves{suffix}.parquet")


def _save_schedule_profile_eval(process, train_expanded_df, test_expanded_df, sensors, ef_cols,
                                output_root, best_mode_safe=None, complete_curve_dir=None,
                                predicted_logs_dir=None, budget_mode_safe=None,
                                comparator_approach='ml_external'):
    """
    "Schedule Profile Evaluation" — see utils/sim_extractor.py's Schedule Profile
    Evaluation section for the design. Per sensor: trains a stochastic reference
    generator (and a bootstrap resampler) on TRAIN cases, samples them against
    REAL TEST cases, and (if available) folds in the already-computed predicted
    curves for the 'ml_external' approach on this process's best-fidelity
    simulation mode — all on the exact same real cases: stochastic vs. the full
    process-simulation-based pipeline ("Best, mine") vs. that same pipeline
    with its timeline rescaled to a dedicated case-duration prediction
    ("Best, duration-corrected"). Everything is persisted as curves
    (predicted_curves.parquet), which the downstream notebooks read.

    "Best, duration-corrected" exists because the process-simulation
    timeline has no mechanism forcing its total elapsed time to be
    realistic — local per-activity duration errors and resource/shift
    queueing artifacts compound freely (confirmed: simulated case durations
    off by 0.3x-11x vs. real in several processes), even when the
    individual predicted energy values are good. This variant keeps "Best,
    mine"'s per-activity value predictions untouched and only rescales the
    time axis so the total span matches an independently-trained case-level
    duration regressor (train_case_duration_pipeline), instead of trusting
    the raw simulated schedule's own (possibly very wrong) total span.

    "Best, budget" (``budget_mode_safe``) is a FIXED-mode comparator: the
    petri_net_budget family generates each case to match its predicted total
    duration inside the simulator, so its span is already right without any
    post-hoc rescale. It is pinned to one named mode (not the per-process
    overall_error winner like "Best, mine") precisely so this table shows a
    clean, like-for-like evaluation of that method across every process —
    otherwise it is silently mixed in with other modes wherever it happens to
    win, and can never be compared head-to-head with "Best, duration-corrected".

    Silently no-ops (per sensor) when there isn't enough data to train on.
    """
    import time as _t
    _t0 = _t.perf_counter()

    def _load_mode_curve_sources(_mode_safe):
        """Curves-parquet path for one simulation mode's already-computed
        complete-curve outputs of `comparator_approach` (historically always
        'ml_external'; follows the complete-curve restriction when that
        approach was not evaluated)."""
        _curve_path = None
        _sfx = '' if comparator_approach == 'baseline' else f'_{comparator_approach}'
        if _mode_safe and complete_curve_dir:
            _cp = os.path.join(complete_curve_dir, process, _mode_safe,
                               f'predicted_curves{_sfx}.parquet')
            if os.path.exists(_cp):
                _curve_path = _cp
        return _curve_path

    _best_curve_path = _load_mode_curve_sources(best_mode_safe)
    _budget_curve_path = _load_mode_curve_sources(budget_mode_safe)
    if budget_mode_safe and _budget_curve_path is None:
        print(f"  ⚠️ 'Best, budget': no complete-curve output found for "
              f"{process}/{budget_mode_safe} — that series will be missing.")

    # Raw simulated per-case total span (first activity start -> last
    # activity end) for the winning ("best") mode, straight from its own
    # simulated log — independent of which activities a curve pipeline could
    # actually predict, so a missing-pipeline gap doesn't silently distort
    # this reference number too. Used as the "before" scale for rescaling.
    _raw_sim_durations = {}
    if best_mode_safe and predicted_logs_dir:
        _safe_process = str(process).replace(' ', '_').replace('/', '_')
        _sim_log_path = os.path.join(predicted_logs_dir, f'{_safe_process}_{best_mode_safe}.parquet')
        if os.path.exists(_sim_log_path):
            try:
                _sim_log_df = pd.read_parquet(
                    _sim_log_path, columns=['case_id', 'timestamp_start', 'timestamp_end'])
                _sim_log_df['case_id'] = _sim_log_df['case_id'].astype(str)
                _sim_starts = _sim_log_df.groupby('case_id')['timestamp_start'].min()
                _sim_ends   = _sim_log_df.groupby('case_id')['timestamp_end'].max()
                _raw_sim_durations = ((_sim_ends - _sim_starts).dt.total_seconds() / 60.0).to_dict()
            except Exception as _exc:
                print(f"  ⚠️ Could not read simulated log for duration correction "
                      f"({_sim_log_path}): {_exc} — 'Best, duration-corrected' will be skipped.")
                _raw_sim_durations = {}
        else:
            print(f"  ⚠️ Simulated log not found for duration correction ({_sim_log_path}) — "
                  f"'Best, duration-corrected' will be skipped for {process}.")

    all_rows = []
    all_curve_rows = [] if SAVE_PREDICTED_CURVES else None
    # Per-sensor generator/duration family, persisted at the end of this
    # function (see save_trained_pipelines) so schedule-level what-ifs can be
    # re-run later without retraining. Plain dicts + module-level predictors,
    # so they reload with load_trained_pipelines and work immediately.
    _sched_models = {}
    for sensor in sensors:
        train_cases = build_case_level_curves(train_expanded_df, sensor, ef_cols=ef_cols)
        test_cases  = build_case_level_curves(test_expanded_df, sensor, ef_cols=ef_cols)
        if len(train_cases) < 4 or not test_cases:
            continue

        median_case_duration_minutes = float(np.median([c['duration_minutes'] for c in train_cases]))

        stochastic_gen    = fit_stochastic_profile_generator(train_cases)
        bootstrap_gen     = fit_bootstrap_profile_generator(train_cases)
        duration_pipeline = train_case_duration_pipeline(train_cases, verbose=0)
        if duration_pipeline is None:
            print(f"  ⚠️ Case-duration pipeline not trained for {process}/{sensor} "
                  f"(too few train cases) — 'Best, duration-corrected' will be skipped for this sensor.")
        test_case_attrs   = {str(c['case_id']): c['attributes'] for c in test_cases}

        if SAVE_TRAINED_MODELS:
            _sched_models[sensor] = {
                'duration':                 duration_pipeline,
                'stochastic':               stochastic_gen,
                'bootstrap':                bootstrap_gen,
                'median_case_duration_minutes': median_case_duration_minutes,
            }

        # duration_pipeline is threaded through so the per-position stochastic
        # generator AND the bootstrap generator are placed on each case's own
        # predicted span (like "Best, duration-corrected") instead of a single
        # median span for every case.
        _cmp_result = compare_schedule_and_stochastic_profiles(
            test_cases, stochastic_gen, median_case_duration_minutes,
            save_curves=SAVE_PREDICTED_CURVES,
            bootstrap_generator=bootstrap_gen,
            duration_pipeline=duration_pipeline,
        )
        cmp_df, sensor_curve_df = _cmp_result if SAVE_PREDICTED_CURVES else (_cmp_result, None)
        if cmp_df.empty:
            continue
        cmp_df['sensor'] = sensor
        cmp_df['case_id'] = cmp_df['case_id'].astype(str)
        all_rows.append(cmp_df)

        if SAVE_PREDICTED_CURVES and sensor_curve_df is not None and not sensor_curve_df.empty:
            sensor_curve_df = sensor_curve_df.copy()
            sensor_curve_df['sensor'] = sensor
            sensor_curve_df['case_id'] = sensor_curve_df['case_id'].astype(str)
            all_curve_rows.append(sensor_curve_df)

            # Fold in the already-computed "Best, mine" curves for this sensor
            # (from the complete-curve eval parquet, same real test cases) so
            # one file has every series: real, stochastic, bootstrap, best,
            # best_duration_corrected.
            if _best_curve_path is not None:
                try:
                    _best_curves = pd.read_parquet(_best_curve_path)
                except Exception:
                    _best_curves = pd.DataFrame()
                if not _best_curves.empty:
                    _best_curves = _best_curves[
                        (_best_curves['sensor'] == sensor) & (_best_curves['series'] == 'predicted')
                    ].copy()
                    _best_curves['series'] = 'best'
                    _best_curves['case_id'] = _best_curves['case_id'].astype(str)
                    _best_curves = _best_curves[['case_id', 'sensor', 'series', 't_minutes', 'value']]
                    all_curve_rows.append(_best_curves)

                    # -- "Best, duration-corrected": same values, rescaled timeline --
                    if duration_pipeline is not None and _raw_sim_durations:
                        _corrected_rows = []
                        for cid, grp in _best_curves.groupby('case_id'):
                            _attrs   = test_case_attrs.get(cid)
                            _raw_dur = _raw_sim_durations.get(cid)
                            if _attrs is None or not _raw_dur or _raw_dur <= 1e-6:
                                continue
                            _pred_dur = predict_case_duration(_attrs, duration_pipeline)
                            _g = grp.copy()
                            _g['t_minutes'] = rescale_case_curve_to_duration(
                                _g['t_minutes'].to_numpy(), _pred_dur, _raw_dur)
                            _g['series'] = 'best_duration_corrected'
                            _corrected_rows.append(_g)
                        if _corrected_rows:
                            all_curve_rows.append(pd.concat(_corrected_rows, ignore_index=True))

            # -- "Best, budget": the pinned petri_net_budget mode's own curves.
            # Its span is already correct by construction (the budget is spent
            # inside the simulator), so NO rescale is applied here — that's the
            # whole point of comparing it against best_duration_corrected.
            if _budget_curve_path is not None:
                try:
                    _budget_curves = pd.read_parquet(_budget_curve_path)
                except Exception:
                    _budget_curves = pd.DataFrame()
                if not _budget_curves.empty:
                    _budget_curves = _budget_curves[
                        (_budget_curves['sensor'] == sensor)
                        & (_budget_curves['series'] == 'predicted')
                    ].copy()
                    if not _budget_curves.empty:
                        _budget_curves['series'] = 'best_budget'
                        _budget_curves['case_id'] = _budget_curves['case_id'].astype(str)
                        _budget_curves = _budget_curves[
                            ['case_id', 'sensor', 'series', 't_minutes', 'value']]
                        all_curve_rows.append(_budget_curves)

    if not all_rows:
        return

    out_dir = os.path.join(output_root, process)
    os.makedirs(out_dir, exist_ok=True)

    if all_curve_rows:
        curves_out = pd.concat(all_curve_rows, ignore_index=True)
        curves_out.to_parquet(os.path.join(out_dir, 'predicted_curves.parquet'), index=False)
        print(f"  💾 Predicted curves (real/stochastic/bootstrap/best/best_duration_corrected) saved → "
              f"schedule_profile_eval_results/{process}/predicted_curves.parquet")

    if SAVE_TRAINED_MODELS and _sched_models:
        _sm_path = os.path.join(output_root, process, 'trained_schedule_pipelines.joblib')
        if save_trained_pipelines(_sched_models, _sm_path, label=f'{process}/schedule'):
            print(f"  [{process}] schedule-profile pipelines saved "
                  f"({len(_sched_models)} sensors) → {_sm_path}")

    _elapsed = _t.perf_counter() - _t0
    print(f"  💾 Schedule Profile Evaluation saved → schedule_profile_eval_results/{process}/  "
          f"({len(sensors)} sensors, {_elapsed:.1f}s)")


# Default definitions to avoid NameError when testing is skipped
process_test_cols = []
energy_test_cols = []
enabled_test_metrics = set()

def _plot_short_heatmap(target_df, title, save_path=None, agg='mean', split='test'):
    """Short heatmap: error metrics (0=best) + Overall. Works for train or test split."""
    if target_df is None or target_df.empty or 'mode' not in target_df.columns:
        return

    _col_wape   = f'{split}_duration_metrics_activity_duration_wape'
    _col_dur_a  = f'{split}_duration_metrics_activity_duration_error'   # fallback (MAPE)
    _col_mae    = f'{split}_duration_metrics_activity_duration_mae'
    _col_span_a = f'{split}_duration_metrics_case_span_error'   # fallback (relative-error)
    _col_span_w = f'{split}_duration_metrics_case_span_wape'
    _col_span_mae = f'{split}_duration_metrics_case_span_mae'

    # Use WAPE if available (new runs), fall back to the plain relative error
    # for older parquets -- same precedent as DurWAPE/DurMAPE above.
    _col_span   = _col_span_w if _col_span_w in target_df.columns else _col_span_a
    _col_f1     = f'{split}_control_flow_metrics_edge_f1_error'
    _col_fit    = f'{split}_conformance_metrics_fitness_error'
    _col_prec   = f'{split}_conformance_metrics_precision_error'
    _col_ov     = f'{split}_overall_error'
    _col_evt    = f'{split}_basic_metrics_event_count_error'

    # Use WAPE if available (new runs), fall back to MAPE for older parquets
    _col_dur    = _col_wape if _col_wape in target_df.columns else _col_dur_a

    all_metric_cols = [_col_dur, _col_mae, _col_span, _col_span_mae,
                       _col_f1, _col_fit, _col_prec, _col_ov, _col_evt]
    needed = [_col_f1]
    available = [c for c in needed if c in target_df.columns]
    if not available:
        return

    agg_cols = [c for c in all_metric_cols if c in target_df.columns]
    if agg == 'median':
        mode_avg = target_df.groupby('mode')[agg_cols].median()
    else:
        mode_avg = target_df.groupby('mode')[agg_cols].mean()

    hm = pd.DataFrame(index=mode_avg.index)
    _mae_actual_by_mode: dict = {}       # raw mode name → actual activity-duration MAE (minutes)
    _span_mae_actual_by_mode: dict = {}  # raw mode name → actual case-span MAE (minutes)

    if _col_dur in mode_avg.columns:
        _dur_label = 'DurWAPE' if 'wape' in _col_dur else 'DurMAPE'
        hm[_dur_label]   = mode_avg[_col_dur]
    if _col_mae in mode_avg.columns:
        _mae_raw = mode_avg[_col_mae]
        _mae_min, _mae_max = _mae_raw.min(), _mae_raw.max()
        _mae_rng = _mae_max - _mae_min
        hm['DurMAE'] = (_mae_raw - _mae_min) / _mae_rng if _mae_rng > 0 else 0.0
        _mae_actual_by_mode = _mae_raw.to_dict()
    if _col_span in mode_avg.columns:
        _span_label = 'CaseSpanWape' if 'wape' in _col_span else 'CaseSpanErr'
        hm[_span_label] = mode_avg[_col_span]
    if _col_span_mae in mode_avg.columns:
        _span_mae_raw = mode_avg[_col_span_mae]
        _span_mae_min, _span_mae_max = _span_mae_raw.min(), _span_mae_raw.max()
        _span_mae_rng = _span_mae_max - _span_mae_min
        hm['CaseSpanMAE'] = (_span_mae_raw - _span_mae_min) / _span_mae_rng if _span_mae_rng > 0 else 0.0
        _span_mae_actual_by_mode = _span_mae_raw.to_dict()
    if _col_f1 in mode_avg.columns:
        hm['EdgeF1Err']  = mode_avg[_col_f1]
    if _col_fit in mode_avg.columns:
        hm['FitnessErr'] = mode_avg[_col_fit]
    if _col_prec in mode_avg.columns:
        hm['PrecisionErr'] = mode_avg[_col_prec]
    err_cols = [c for c in ['DurWAPE', 'DurMAPE', 'DurMAE', 'CaseSpanWape', 'CaseSpanErr', 'CaseSpanMAE',
                             'EdgeF1Err', 'FitnessErr', 'PrecisionErr']
                if c in hm.columns]
    if err_cols:
        hm['Overall'] = hm[err_cols].mean(axis=1)
    # Diagnostic-only column, added AFTER Overall so it's never part of the
    # composite -- how far the simulated event count is from the real one
    # per case (|sim_n_events/real_n_events - 1|). Kept out of Overall on
    # request, but shown since a mismatch here (stochastic Petri-net
    # branching/looping firing a different number of activities than the
    # matched real case) is often the dominant driver of CaseSpanErr, more
    # so than any per-activity duration mis-prediction.
    if _col_evt in mode_avg.columns:
        hm['EvtRatioErr'] = mode_avg[_col_evt]

    def _display_mode(m):
        m = str(m)
        if not m.startswith('petri_net_'):
            return m
        rest = m[len('petri_net_'):]
        if rest.endswith('_ml_plus_global'):
            return rest[:-len('_ml_plus_global')] + ' / ml_global'
        if rest.endswith('_ml_plus_per_act'):
            return rest[:-len('_ml_plus_per_act')] + ' / ml_local'
        return rest + ' / baseline'

    _raw_modes = list(hm.index)
    _mae_by_display = {_display_mode(m): _mae_actual_by_mode.get(m, float('nan'))
                       for m in _raw_modes}
    _span_mae_by_display = {_display_mode(m): _span_mae_actual_by_mode.get(m, float('nan'))
                            for m in _raw_modes}
    hm.index = [_display_mode(m) for m in _raw_modes]

    hm = hm.sort_values('Overall', ascending=True)

    # Annotation: DurMAE/CaseSpanMAE columns show actual minutes; all others show 3 dp
    _annot_rows = []
    for _dm in hm.index:
        _row = []
        for _hm_col in hm.columns:
            _val = hm.loc[_dm, _hm_col]
            if _hm_col == 'DurMAE':
                _actual = _mae_by_display.get(_dm, float('nan'))
                _row.append(f'{_actual:.1f}' if not pd.isna(_actual) else '')
            elif _hm_col == 'CaseSpanMAE':
                _actual = _span_mae_by_display.get(_dm, float('nan'))
                _row.append(f'{_actual:.1f}' if not pd.isna(_actual) else '')
            else:
                _row.append(f'{_val:.3f}' if not pd.isna(_val) else '')
        _annot_rows.append(_row)
    _annot_arr = np.array(_annot_rows)

    n_rows = max(2, len(hm))
    fig, ax = plt.subplots(figsize=(max(10, len(hm.columns) * 1.5), n_rows * 0.9 + 1.8))
    # Separator before Overall -- computed from its position, not
    # len(columns)-1, since diagnostic-only columns (e.g. EvtRatioErr) can
    # now follow it.
    sep = list(hm.columns).index('Overall') + 1 if 'Overall' in hm.columns else len(hm.columns) - 1
    sns.heatmap(hm.round(3), annot=_annot_arr, fmt='', cmap='RdYlGn_r',
                vmin=0, vmax=1, linewidths=0.5, linecolor='white',
                cbar_kws={'label': 'Error (0 = best)', 'shrink': 0.7}, ax=ax)
    ax.axvline(x=sep, color='navy', linewidth=2.0)
    ax.set_title(
        f'{title}\nAll metrics: 0 = best  |  DurMAE/CaseSpanMAE annotations = actual minutes (colour normalised)  |  Overall = mean of first {sep} cols',
        fontsize=11, fontweight='bold')
    ax.set_ylabel('Mode')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\n{title.upper()} — SHORT METRICS (0 = best)")
    print("-" * 60)
    with pd.option_context('display.max_columns', None, 'display.width', 200):
        print(hm.round(4).to_string())


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

def _dur_js(a, b, n_bins=20):
    """JS divergence between two duration sample arrays using log-space histogram bins."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a) & (a > 0)]
    b = b[np.isfinite(b) & (b > 0)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    lo = max(np.percentile(np.concatenate([a, b]), 1), 1e-3)
    hi = np.percentile(np.concatenate([a, b]), 99)
    if lo >= hi:
        return 0.0
    bins = np.exp(np.linspace(np.log(lo), np.log(hi), n_bins + 1))
    p, _ = np.histogram(a, bins=bins)
    q, _ = np.histogram(b, bins=bins)
    p = p.astype(float) + 1e-9
    q = q.astype(float) + 1e-9
    p /= p.sum()
    q /= q.sum()
    return float(jensenshannon(p, q))


def _per_case_median_metrics(simulated_df, real_df,
                             case_col='case_id', activity_col='activity',
                             start_col='timestamp_start', end_col='timestamp_end'):
    """
    Compute EvtRatioErr, DurErr(whole), DurErr(activ), JS-div, 1-EdgeF1
    independently for every case that exists in both logs, then return
    the MEDIAN across cases.  This removes the global-pooling bias where
    a few large cases dominate the aggregate.

    The returned dict also carries the untouched per-case frame under the
    ``per_case`` key. Every headline process metric here is a median over
    cases, so the per-case sample is the *only* thing from which a confidence
    interval can be bootstrapped after the fact -- collapsing it to a single
    median and throwing the sample away made post-hoc CIs impossible for the
    process metrics (they were already possible for the curve/profile metrics,
    which persist one row per curve). It is stripped from the flattened
    results row before export and written to its own parquet instead.
    """
    # Match by case_id as strings, not raw values — the real and simulated
    # logs can carry the same case_id in different dtypes (e.g. float64 vs
    # object/str for numeric-looking IDs), which silently zeroes out every
    # match under a raw-value set intersection despite full overlap. Same
    # fix already applied in compare_complete_case_curves (utils/sim_extractor.py)
    # for the identical failure mode. Confirmed to actually happen: process_5
    # real case_id is float64 (10708907.0), its simulated log's is object
    # ('10708907.0') — 0/16 cases matched raw, 16/16 matched as strings.
    sim_ids_str  = simulated_df[case_col].astype(str)
    real_ids_str = real_df[case_col].astype(str)
    sim_valid    = simulated_df[case_col].notna()
    real_valid   = real_df[case_col].notna()
    common = sorted(set(sim_ids_str[sim_valid].unique()) &
                    set(real_ids_str[real_valid].unique()))
    if not common:
        return {}

    rows = []
    for cid in common:
        rc = real_df[real_valid & (real_ids_str == cid)]
        sc = simulated_df[sim_valid & (sim_ids_str == cid)]
        if rc.empty:
            continue

        # --- EvtRatioErr ---
        evt_ratio_err = abs(len(sc) / len(rc) - 1.0)

        # --- DurErr(whole): relative error of mean event duration ---
        real_durs = (rc[end_col] - rc[start_col]).dt.total_seconds() / 60.0
        sim_durs  = (sc[end_col] - sc[start_col]).dt.total_seconds() / 60.0
        real_mu = real_durs.mean()
        dur_err_whole = abs(sim_durs.mean() - real_mu) / real_mu if real_mu > 0 else np.nan

        # --- CaseSpanErr: relative error of the case's TOTAL elapsed time
        # (max(end) - min(start)), distinct from dur_err_whole above (mean of
        # individual per-activity-instance durations within a case). This is
        # the metric that actually catches a process model whose overall
        # simulated case length drifts from reality -- e.g. via queueing/
        # resource-availability artifacts inserting large idle gaps between
        # activities, or compounding per-activity duration bias -- even when
        # individual activity durations look fine on average. ---
        real_span = (rc[end_col].max() - rc[start_col].min()).total_seconds() / 60.0
        sim_span  = (sc[end_col].max() - sc[start_col].min()).total_seconds() / 60.0
        case_span_err = abs(sim_span - real_span) / real_span if real_span > 0 else np.nan
        # Absolute-minutes companion to case_span_err, mirroring dur_mae below
        # (per-activity-type MAE in minutes) but for the whole-case total --
        # directly answers "how many minutes off is the simulated case
        # length," which a relative % can obscure for very short/long cases.
        case_span_mae = abs(sim_span - real_span)
        span_real = real_span

        # --- DurErr(activ): mean relative error across activity types in this case ---
        r_act = (rc.assign(_d=(rc[end_col]-rc[start_col]).dt.total_seconds()/60.0)
                   .groupby(activity_col)['_d'].mean())
        s_act = (sc.assign(_d=(sc[end_col]-sc[start_col]).dt.total_seconds()/60.0)
                   .groupby(activity_col)['_d'].mean())
        common_acts = r_act.index.intersection(s_act.index)
        act_errs = [abs(s_act[a] - r_act[a]) / r_act[a]
                    for a in common_acts if r_act[a] > 0]
        dur_err_activ = float(np.mean(act_errs)) if act_errs else np.nan

        # --- MAE / RMSE / WAPE of per-activity mean durations (minutes) ---
        if common_acts.size > 0:
            _abs_min = np.array([abs(s_act[a] - r_act[a]) for a in common_acts])
            _real_min = np.array([r_act[a] for a in common_acts])
            dur_mae  = float(np.mean(_abs_min))
            dur_rmse = float(np.sqrt(np.mean(_abs_min ** 2)))
            _wden    = float(np.sum(_real_min))
            dur_wape = float(np.sum(_abs_min) / _wden) * 100 if _wden > 0 else np.nan
        else:
            dur_mae = dur_rmse = dur_wape = np.nan

        # --- JS divergence of activity-frequency distribution ---
        all_acts = sorted(set(rc[activity_col].dropna()) | set(sc[activity_col].dropna()))
        r_cnt = rc[activity_col].value_counts()
        s_cnt = sc[activity_col].value_counts()
        p = np.array([r_cnt.get(a, 0) for a in all_acts], dtype=float) + 1e-9
        q = np.array([s_cnt.get(a, 0) for a in all_acts], dtype=float) + 1e-9
        p /= p.sum(); q /= q.sum()
        m = 0.5 * (p + q)
        js_div = float(0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m))))

        # --- EdgeF1 on the directly-follows graph of this case ---
        def _case_edges(df):
            grp = df.sort_values(start_col)
            acts = grp[activity_col].tolist()
            return set(zip(acts, acts[1:]))

        r_edges = _case_edges(rc)
        s_edges = _case_edges(sc)
        if r_edges or s_edges:
            tp = len(r_edges & s_edges)
            prec = tp / len(s_edges) if s_edges else 0.0
            rec  = tp / len(r_edges) if r_edges else 0.0
            ef1  = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
        else:
            ef1 = np.nan

        rows.append({'case_id':       cid,
                     'evt_ratio_err': evt_ratio_err,
                     'dur_err_whole': dur_err_whole,
                     'case_span_err': case_span_err,
                     'case_span_mae': case_span_mae,
                     'span_real':     span_real,
                     'dur_err_activ': dur_err_activ,
                     'dur_mae':       dur_mae,
                     'dur_rmse':      dur_rmse,
                     'dur_wape':      dur_wape,
                     'js_div':        js_div,
                     'edge_f1':       ef1})

    if not rows:
        return {}
    df = pd.DataFrame(rows)
    # CaseSpanWape: true WAPE for case span -- sum of absolute case-span
    # errors over sum of real case spans, pooled across ALL matched cases
    # (mirrors dur_wape's sum-over-sum shape, just pooled across cases
    # instead of across activity types within one case, since a case only
    # has a single span value). Distinct from case_span_err below (median
    # of per-case relative errors) -- that one still drives overall_error/
    # mode-selection unchanged; this is a display-only companion metric.
    _valid_span   = df['span_real'] > 0
    _span_abs_sum = float(df.loc[_valid_span, 'case_span_mae'].sum())
    _span_real_sum = float(df.loc[_valid_span, 'span_real'].sum())
    case_span_wape = (_span_abs_sum / _span_real_sum) * 100 if _span_real_sum > 0 else np.nan
    return {
        'evt_ratio_err': float(df['evt_ratio_err'].median()),
        'dur_err_whole': float(df['dur_err_whole'].median()),
        'case_span_err': float(df['case_span_err'].median()),
        'case_span_mae': float(df['case_span_mae'].median()),
        'case_span_wape': case_span_wape,
        'dur_err_activ': float(df['dur_err_activ'].median()),
        'dur_mae':       float(df['dur_mae'].median()),
        'dur_rmse':      float(df['dur_rmse'].median()),
        'dur_wape':      float(df['dur_wape'].median()),
        'js_div':        float(df['js_div'].median()),
        'edge_f1':       float(df['edge_f1'].median()),
        'n_cases':       len(rows),
        # Raw per-case sample behind every median above — kept for post-hoc
        # bootstrap confidence intervals (see docstring).
        'per_case':      df,
    }


# ── Per-case process metrics collector ───────────────────────────────────────
# Every headline process metric (duration MAE/WAPE/RMSE, case span, event-count
# ratio, JS divergence, edge-F1) is a MEDIAN over cases. A median alone cannot
# be turned into a confidence interval after the fact, so the per-case sample it
# was computed from is collected here and written to
# process_eval_per_case.parquet — one row per (process, mode, split, case_id).
# That makes post-hoc bootstrap CIs possible for the process metrics, the way
# curve_eval_results.parquet already allows for the profile metrics.
_per_case_process_metrics = []


def _collect_per_case_metrics(per_case_df, tag):
    """Tag a per-case frame with (process, mode, split) and stash it."""
    if per_case_df is None or not isinstance(per_case_df, pd.DataFrame) or per_case_df.empty:
        return
    process, mode, split = tag
    out = per_case_df.copy()
    out.insert(0, 'split', split)
    out.insert(0, 'mode', mode)
    out.insert(0, 'process', process)
    _per_case_process_metrics.append(out)


def comprehensive_simulation_evaluation(simulated_df, real_df, real_expanded_df=None,
                                       case_col='case_id', activity_col='activity',
                                       start_col='timestamp_start', end_col='timestamp_end',
                                       process_models=None, station_col='higher_level_activity',
                                       per_case_tag=None):
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
    
    # Event counts. Ratio is sim/real (matches the
    # EvtRatio column convention): 1.0 = perfect match, >1 = simulation
    # OVER-counts events, <1 = simulation under-counts. NB: was real/sim here
    # (a reciprocal inconsistency, fixed 2026-07-21) — that both inverted the
    # table and made event_count_error under-penalise over-generation
    # (sim=2x real gave error 0.5 instead of 1.0), masking budget-mode
    # over-generation.
    sim_events = len(simulated_df)
    real_events = len(real_df)
    event_ratio = sim_events / real_events if real_events > 0 else 0

    # Case counts
    sim_cases = simulated_df[case_col].nunique() if case_col in simulated_df.columns else 0
    real_cases = real_df[case_col].nunique() if case_col in real_df.columns else 0
    case_ratio = sim_cases / real_cases if real_cases > 0 else 0

    report(f"Events - Real: {real_events}, Sim: {sim_events}, Ratio (sim/real): {event_ratio:.3f}")
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
    
    # Per-activity duration metrics: MAPE, MAE (min), RMSE (min), WAPE
    # Durations in seconds → convert to minutes for MAE/RMSE
    _real_act_dur = (real_df.assign(_d=(real_df[end_col] - real_df[start_col]).dt.total_seconds())
                     .groupby(activity_col)['_d'].mean())
    _sim_act_dur  = (simulated_df.assign(_d=(simulated_df[end_col] - simulated_df[start_col]).dt.total_seconds())
                     .groupby(activity_col)['_d'].mean())
    _common_acts  = _real_act_dur.index.intersection(_sim_act_dur.index)
    _act_dur_errs = [abs(_sim_act_dur[a] - _real_act_dur[a]) / _real_act_dur[a]
                     for a in _common_acts if _real_act_dur[a] != 0]
    activity_duration_error = float(np.mean(_act_dur_errs)) if _act_dur_errs else np.nan

    if _common_acts.size > 0:
        _abs_diffs_min = np.array([abs(_sim_act_dur[a] - _real_act_dur[a]) / 60.0 for a in _common_acts])
        _real_min      = np.array([_real_act_dur[a] / 60.0 for a in _common_acts])
        activity_duration_mae  = float(np.mean(_abs_diffs_min))
        activity_duration_rmse = float(np.sqrt(np.mean(_abs_diffs_min ** 2)))
        _wape_den = float(np.sum(_real_min))
        activity_duration_wape = float(np.sum(_abs_diffs_min) / _wape_den) * 100 if _wape_den > 0 else np.nan
    else:
        activity_duration_mae  = np.nan
        activity_duration_rmse = np.nan
        activity_duration_wape = np.nan

    report(f"Per-activity mean duration error (MAPE): {activity_duration_error:.4f} (0=perfect)")
    report(f"Per-activity duration MAE:  {activity_duration_mae:.2f} min")
    report(f"Per-activity duration RMSE: {activity_duration_rmse:.2f} min")
    report(f"Per-activity duration WAPE: {activity_duration_wape:.4f} (0=perfect)")

    # Duration distribution JS divergence (whole): JS between all-event duration histograms
    dur_js_whole = _dur_js(sim_durations.values, real_durations.values)

    # Duration distribution JS divergence (per-activity): mean JS across activity types
    _act_dur_js = []
    for _act in _common_acts:
        _r = (real_df[real_df[activity_col] == _act]
              .pipe(lambda d: (d[end_col] - d[start_col]).dt.total_seconds() / 60.0)
              .replace([np.inf, -np.inf], np.nan).dropna())
        _s = (simulated_df[simulated_df[activity_col] == _act]
              .pipe(lambda d: (d[end_col] - d[start_col]).dt.total_seconds() / 60.0)
              .replace([np.inf, -np.inf], np.nan).dropna())
        _js = _dur_js(_r.values, _s.values)
        if pd.notna(_js):
            _act_dur_js.append(_js)
    dur_js_activ = float(np.mean(_act_dur_js)) if _act_dur_js else np.nan

    report(f"Duration JS divergence (whole): {dur_js_whole:.4f} (0=perfect)")
    report(f"Duration JS divergence (per-activity mean): {dur_js_activ:.4f} (0=perfect)")

    results['duration_metrics'] = {
        'ks_statistic': duration_ks_stat,
        'ks_pvalue': duration_ks_pvalue,
        'mean_duration_error': duration_stats.loc['Mean', 'Error'],
        'median_duration_error': duration_stats.loc['Median', 'Error'],
        'std_duration_error': duration_stats.loc['Std', 'Error'],
        'activity_duration_error': activity_duration_error,
        'activity_duration_mae':   activity_duration_mae,
        'activity_duration_rmse':  activity_duration_rmse,
        'activity_duration_wape':  activity_duration_wape,
        'dur_js_whole': dur_js_whole,
        'dur_js_activ': dur_js_activ,
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
        'edge_f1_error': 1.0 - edge_f1,
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

    # Store error variants (1 - score) so they flow through the heatmap as lower-is-better
    for _cm_key in ('fitness', 'precision'):
        _cm_val = conformance_metrics.get(_cm_key, np.nan)
        if pd.notna(_cm_val):
            conformance_metrics[f'{_cm_key}_error'] = 1.0 - _cm_val
        else:
            conformance_metrics[f'{_cm_key}_error'] = np.nan

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

    # Short-heatmap error components: computed per-case, then median across cases.
    # This avoids large cases dominating the pooled aggregate.
    _pc = _per_case_median_metrics(simulated_df, real_df,
                                   case_col=case_col, activity_col=activity_col,
                                   start_col=start_col, end_col=end_col)
    report(f"\nPer-case median metrics ({_pc.get('n_cases', 0)} cases matched):")

    # Fall back to global values for any metric the per-case routine couldn't compute
    _edge_f1_global   = results['control_flow_metrics'].get('edge_f1_score', np.nan)
    _dur_activ_global = results['duration_metrics'].get('activity_duration_error', np.nan)

    # overall_error (drives best-mode selection, see _best_mode_by_process).
    # Each component carries a weight; the score is their weighted mean.
    # case_span_err (total case duration, vs. dur_err_activ's per-activity-
    # type durations) is included deliberately: it's exactly the metric that
    # was missing when a mode with poor control-flow/duration fidelity got
    # selected as "best" despite simulated case lengths drifting 0.3x-11x
    # from real (see the process_2 idle-gap investigation).
    # evt_ratio_err (per-case |sim_events/real_events - 1|, median) is folded
    # in at a *small* weight: event count is strongly upstream of case span
    # (too few events -> short span almost regardless of durations), so a mode
    # can no longer win by nailing per-activity durations while producing half
    # the events -- but it's down-weighted so it informs, not dominates,
    # selection. js_div is still computed/stored as a diagnostic only.
    _EVT_RATIO_WEIGHT = 0.3   # relative to 1.0 for the core components
    short_components = {
        'dur_err_activ':          (_pc.get('dur_err_activ', _dur_activ_global), 1.0),
        'case_span_err':          (_pc.get('case_span_err', np.nan), 1.0),
        'edge_err (1-EdgeF1)':    ((1.0 - _pc['edge_f1']) if 'edge_f1' in _pc else
                                   ((1.0 - _edge_f1_global) if pd.notna(_edge_f1_global) else np.nan), 1.0),
        'fitness_err (1-Fitness)':    (results['conformance_metrics'].get('fitness_error', np.nan), 1.0),
        'precision_err (1-Prec)':     (results['conformance_metrics'].get('precision_error', np.nan), 1.0),
        'evt_ratio_err':          (_pc.get('evt_ratio_err', np.nan), _EVT_RATIO_WEIGHT),
    }

    report("\nShort-heatmap error components (0 = best):")
    _weighted_sum = 0.0
    _weight_total = 0.0
    for comp_name, (comp_val, comp_wt) in short_components.items():
        if pd.isna(comp_val):
            report(f"  {comp_name:30}: n/a")
        else:
            report(f"  {comp_name:30}: {float(comp_val):.4f}  (w={comp_wt})")
            _weighted_sum += float(comp_val) * comp_wt
            _weight_total += comp_wt

    overall_error = (_weighted_sum / _weight_total) if _weight_total > 0 else np.nan
    report(f"\nOVERALL ERROR SCORE: {overall_error:.4f} (0=perfect)")

    if pd.notna(overall_error) and overall_error <= 0.05:
        quality_assessment = "EXCELLENT"
    elif pd.notna(overall_error) and overall_error <= 0.15:
        quality_assessment = "GOOD"
    elif pd.notna(overall_error) and overall_error <= 0.30:
        quality_assessment = "FAIR"
    else:
        quality_assessment = "POOR"

    report(f"QUALITY ASSESSMENT: {quality_assessment}")

    # Overwrite the individual result-dict entries with per-case median values so
    # the flattening → heatmap pipeline uses per-case medians everywhere.
    if _pc:
        # event_count_ratio/error are left as the simple pooled real/sim
        # count ratio computed above (not overwritten with the per-case
        # median here) -- straightforward to read, and comparable directly
        # across modes.
        results['duration_metrics']['mean_duration_error']         = _pc['dur_err_whole']
        if pd.notna(_pc.get('case_span_err', np.nan)):
            results['duration_metrics']['case_span_error']          = _pc['case_span_err']
        if pd.notna(_pc.get('case_span_mae', np.nan)):
            results['duration_metrics']['case_span_mae']            = _pc['case_span_mae']
        if pd.notna(_pc.get('case_span_wape', np.nan)):
            results['duration_metrics']['case_span_wape']           = _pc['case_span_wape']
        if pd.notna(_pc.get('dur_err_activ', np.nan)):
            results['duration_metrics']['activity_duration_error'] = _pc['dur_err_activ']
        if pd.notna(_pc.get('dur_mae', np.nan)):
            results['duration_metrics']['activity_duration_mae']   = _pc['dur_mae']
        if pd.notna(_pc.get('dur_rmse', np.nan)):
            results['duration_metrics']['activity_duration_rmse']  = _pc['dur_rmse']
        if pd.notna(_pc.get('dur_wape', np.nan)):
            results['duration_metrics']['activity_duration_wape']  = _pc['dur_wape']
        if pd.notna(_pc.get('js_div', np.nan)):
            results['activity_metrics']['js_divergence']           = _pc['js_div']
        if pd.notna(_pc.get('edge_f1', np.nan)):
            results['control_flow_metrics']['edge_f1_score']       = _pc['edge_f1']
            results['control_flow_metrics']['edge_f1_error']       = 1.0 - _pc['edge_f1']

    results['overall_error'] = overall_error
    results['quality_assessment'] = quality_assessment

    # Per-case sample behind every median-based process metric above. It is
    # deliberately NOT put into `results`: callers flatten that dict into one
    # wide row per (process, mode), and a DataFrame landing in a scalar column
    # would break the parquet export. Instead it goes straight to the module
    # level collector, tagged with whatever the caller passed. Callers that
    # don't pass a tag simply don't contribute per-case rows — a missed call
    # site loses data for that mode rather than corrupting the run.
    if per_case_tag is not None:
        _collect_per_case_metrics(_pc.get('per_case'), per_case_tag)

    return results




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

def _order_case_ids_for_split(case_start, split_type, seed=42):
    """
    Return case_start's case-id index ordered so that a plain head/tail slice
    at n_train produces the requested split:
      'temporal' → earliest-start-first (case_start is already sorted this
                   way), so the tail is the chronologically later "future".
      'random'   → a fixed-seed shuffle, so the tail is an arbitrary random
                   subset at the same ratio, with no time ordering.
    Keeping this as the only place that decides ordering means every caller
    (main event-log split, the production-plan fallback below) stays
    identical apart from which ordering feeds the same slicing logic.
    """
    if split_type == 'random':
        rng = np.random.RandomState(seed)
        return case_start.index[rng.permutation(len(case_start))]
    return case_start.index


def _split_process_datasets(datasets, train_ratio=0.80, split_type='temporal',
                            random_seed=42):
    """
    Split *every* process inside ``datasets`` into two copies:
    ``train_datasets`` and ``test_datasets``.

    For each process the event_log is used to determine the case partition
    (by case start time for split_type='temporal', or a fixed-seed shuffle
    for split_type='random' — see _order_case_ids_for_split). The same
    case-id partition is then applied to **production_plan** and
    **expanded** (using ``case_id_log``), so every downstream evaluation
    (process, energy/profile, schedule-profile) is scored against the exact
    same held-out cases regardless of split_type.

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

        ordered_ids = _order_case_ids_for_split(case_start, split_type, random_seed)
        train_cases = set(ordered_ids[:n_train])
        test_cases  = set(ordered_ids[n_train:])

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
            pp_ordered_ids = _order_case_ids_for_split(pp_case_start, split_type, random_seed)
            pp_train_ids = set(pp_ordered_ids[:n_pp_train])
            pp_test_ids  = set(pp_ordered_ids[n_pp_train:])
            pp_train = production_plan[production_plan['case_id'].isin(pp_train_ids)].copy()
            pp_test  = production_plan[production_plan['case_id'].isin(pp_test_ids)].copy()

        # ── expanded df split (uses case_id_log) ─────────────────────────
        exp_train, exp_test = None, None
        if expanded is not None:
            if 'case_id_log' in expanded.columns:
                exp_train = expanded[expanded['case_id_log'].isin(train_cases)].copy()
                exp_test  = expanded[expanded['case_id_log'].isin(test_cases)].copy()
            else:
                # fallback: use datetime_energy and the cutoff date. No case
                # identifier is available on these rows, so a case-level
                # random split isn't possible here -- always falls back to a
                # temporal cutoff regardless of split_type.
                if split_type == 'random':
                    print(f"  ⚠️ {proc_name}: 'expanded' has no case_id_log — "
                          f"can't apply a random split without a case identifier; "
                          f"falling back to a temporal cutoff for this table.")
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


def _retarget_cases_to_event_span(cases, event_log,
                                  case_col='case_id', start_col='timestamp_start',
                                  end_col='timestamp_end'):
    """
    Replace each case's 'duration_minutes' with its EVENT-LOG span.

    build_case_level_curves measures a case by its energy recording, which ends
    with the last activity that has a sensor. Any predictor whose output is
    consumed as a wall-clock case duration — and scored against 'span_real',
    which is the event-log span — has to be trained on the event-log span too,
    or it inherits a fixed negative offset equal to the un-metered head and tail
    of the case. Cases missing from the event log keep their original value.

    Returns a new list; the input dicts are not mutated, so the callers that
    legitimately want the energy-curve span are unaffected.
    """
    if event_log is None or len(event_log) == 0:
        return cases
    _el = event_log.dropna(subset=[case_col])
    _g = _el.groupby(_el[case_col].astype(str))
    _span = ((_g[end_col].max() - _g[start_col].min()).dt.total_seconds() / 60.0)

    out, _n_hit = [], 0
    for c in cases:
        s = _span.get(str(c['case_id']))
        if s is not None and pd.notna(s) and s > 0:
            c = {**c, 'duration_minutes': float(s)}
            _n_hit += 1
        out.append(c)
    if _n_hit < len(cases):
        print(f"  ℹ️ case-duration retarget: {_n_hit}/{len(cases)} cases matched the "
              f"event log; the rest keep their energy-curve span.")
    return out


# ── Build the two dictionaries ───────────────────────────────────────────────
if TEMPORAL_SPLIT:
    print(f"\n📌 {SPLIT_TYPE.upper()} SPLIT ({TRAIN_RATIO:.0%} train / "
          f"{1-TRAIN_RATIO:.0%} test) — splitting all processes:")
    train_datasets, test_datasets = _split_process_datasets(
        process_datasets_to_model, TRAIN_RATIO,
        split_type=SPLIT_TYPE, random_seed=RANDOM_SPLIT_SEED,
    )
else:
    print("\n📌 NO SPLIT – using all data for extraction & training")
    train_datasets = process_datasets_to_model
    test_datasets  = process_datasets_to_model

all_energy_pipelines = {}

### models to compare

_filtered_modes = []
for _mode_name in MODES_TO_COMPARE:
    if _mode_name.startswith('petri_net_'):
        _mode_alg = _mode_name.replace('petri_net_', '', 1).strip().lower()
        if (_mode_alg in ('combined', 'median_duration', 'budget') or
                _mode_alg.endswith('_ml_plus_global') or
                _mode_alg.endswith('_ml_plus_per_act')):
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
# Energy-distribution metrics can't be computed yet here — all_energy_pipelines
# is only populated later, by the module-level "CURVE-ONLY TRAINING" section
# (RUN_CURVE_ONLY_EVALUATION), which runs AFTER this entire per-process loop
# finishes for every process. So we collect what we need now and defer the
# actual computation until after that section runs (see below).
_energy_distribution_pending = []  # (process, mode, sim_df, real_expanded_df, core_metrics_row)

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
    with _timed('process_mining', process=process, detail=MINING_ALGORITHM, split='TRAIN'):
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
        with _timed('process_mining', process=process, detail=mode_alg, split='TRAIN'):
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

    # Legacy SimModeller ('ml'/'ml_duration_only' modes) removed — the
    # evaluated ML+ modes carry their own duration models (mlp_* tuples).
    ml_models = None

    # ── Train the case-duration predictor for petri_net_budget mode ───────
    # One per-case total-duration regressor (schedule-only attributes -> total
    # minutes), the same predictor the "duration-corrected" post-processing
    # used — but here it's handed to the simulator so petri_net_budget can
    # generate each case to match its predicted duration natively. Trained on
    # TRAIN cases only. None -> the mode falls back to a plain-span timeline.
    #
    # NOTE the retarget below. build_case_level_curves sets each case's
    # 'duration_minutes' to the span of its ENERGY RECORDING
    # (max(datetime_energy) - min(timestamp_start_log)), which is the right
    # quantity for the schedule-profile curve reconstruction that also consumes
    # it (see median_case_duration_minutes further down). It is the WRONG
    # quantity here: the simulator spends this number as a wall-clock budget
    # for the whole case, and the process eval scores the result against the
    # event-log span (max(timestamp_end) - min(timestamp_start), 'span_real').
    # Those differ by every station that has no sensor and therefore no row in
    # the expanded table — in process_1 that is Water_supply at the head and
    # Warehousing at the tail, the latter being the last station of every
    # single case. The energy span is short by ~12 min (std 0.7, i.e. a fixed
    # offset, not noise) on all 300 cases, so the budget mode was closing every
    # case ~12 min early before the regressor made any error at all.
    _case_duration_pipeline = None
    if 'petri_net_budget' in MODES_TO_COMPARE:
        _bud_train_exp = train_datasets.get(process, {}).get('expanded') if train_datasets else None
        if _bud_train_exp is not None and not _bud_train_exp.empty:
            _bud_sensors = _detect_sensors_for_energy_distribution(process, _bud_train_exp)
            _bud_ef_cols = [c for c in _bud_train_exp.columns
                            if c.startswith('ef_') and _bud_train_exp[c].dtype in ('float64', 'float32', 'int64', 'int32')]
            if _bud_sensors:
                try:
                    # Case duration/attributes are case-level (sensor-independent),
                    # so any one sensor's case list yields the same duration model.
                    with _timed('case_duration_training', process=process,
                                detail='petri_net_budget', split='TRAIN'):
                        _bud_cases = build_case_level_curves(_bud_train_exp, _bud_sensors[0], ef_cols=_bud_ef_cols)
                        _bud_cases = _retarget_cases_to_event_span(_bud_cases, df_train)
                        _case_duration_pipeline = train_case_duration_pipeline(_bud_cases, verbose=0)
                except Exception as _bud_exc:
                    print(f"  ⚠️ petri_net_budget: case-duration pipeline training failed "
                          f"({_bud_exc}) — mode will run as plain petri_net for {process}.")
        if _case_duration_pipeline is None:
            print(f"  ⚠️ petri_net_budget: no case-duration predictor available for {process} "
                  f"— mode runs as plain petri_net (no budget).")

    # ── Loop over modes ───────────────────────────────────────────────────
    process_mode_results = []

    for sim_mode in MODES_TO_COMPARE:
        if (sim_mode in ('petri_net_combined', 'petri_net_median_duration') or
                sim_mode.endswith('_ml_plus_global') or
                sim_mode.endswith('_ml_plus_per_act')):
            # Derived after all base modes are evaluated.
            continue

        print("\n" + "─"*80)
        print(f"  ▶ SIMULATION MODE: {sim_mode.upper()}")
        print("─"*80)

        mode_algorithm = None
        simulation_mode = sim_mode
        mode_activity_stats_df = activity_stats_df

        if sim_mode == 'petri_net_budget':
            # Base PN token game on MINING_ALGORITHM's net, plus a per-case
            # duration budget spent inside the generator (case_duration_pipeline
            # passed to ProcessSimulation below). simulation_mode stays
            # 'petri_net_budget' so the dispatch routes to the budget variant.
            mode_algorithm = MINING_ALGORITHM
        elif sim_mode.startswith('petri_net_'):
            mode_algorithm = sim_mode.replace('petri_net_', '', 1).strip().lower()
            if mode_algorithm not in extraction_by_algorithm:
                print(
                    f"⚠️ Skipping unsupported Petri-net mode '{sim_mode}'. "
                    f"Expected one of: {[f'petri_net_{a}' for a in PETRI_NET_ALGORITHMS]}"
                )
                continue
            simulation_mode = 'petri_net'
            mode_activity_stats_df = extraction_by_algorithm[mode_algorithm]['activity_stats_df']

        mode_ml = ml_models if simulation_mode not in ('statistical', 'petri_net', 'petri_net_budget') else None
        mode_pm = extraction_by_algorithm.get(mode_algorithm, {}).get('process_models') if simulation_mode in ('petri_net', 'petri_net_budget', 'petri_net_statistical', 'petri_net_statistical_memory') else None

        with _timed('simulation', process=process, detail=sim_mode, split='TRAIN'):
            simulated_log_train = ProcessSimulation(
                mode_activity_stats_df, production_plan,
                mode=simulation_mode, ml_models=mode_ml,
                process_models=mode_pm,
                case_duration_pipeline=(_case_duration_pipeline
                                        if simulation_mode == 'petri_net_budget' else None),
            ).run()

        print(f"\n  Simulated log TRAIN ({sim_mode}): {len(simulated_log_train)} events")

        # ══════════════════════════════════════════════════════════════════
        # EVALUATION — TRAIN SET
        # ══════════════════════════════════════════════════════════════════
        split_label = "TRAIN" if TEMPORAL_SPLIT else "ALL DATA"
        print(f"\n  🔍 EVALUATION ON {split_label}  [{sim_mode}]")
        print("  " + "="*76)

        with _timed('process_evaluation', process=process, detail=sim_mode, split=split_label):
            eval_train = comprehensive_simulation_evaluation(simulated_log_train, df_train,
                                                              process_models=mode_pm,
                                                              per_case_tag=(process, sim_mode, split_label))

        print(f"\n  📊 COMPARISON PLOTS ({split_label})  [{sim_mode}]")
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
            with _timed('simulation', process=process, detail=sim_mode, split='TEST'):
                simulated_log_test = ProcessSimulation(
                    mode_activity_stats_df,
                    production_plan_test,
                    mode=simulation_mode,
                    ml_models=mode_ml,
                    process_models=mode_pm,
                    case_duration_pipeline=(_case_duration_pipeline
                                            if simulation_mode == 'petri_net_budget' else None),
                ).run()

            print(f"\n  Simulated log TEST  ({sim_mode}): {len(simulated_log_test)} events")

            # ── Save predicted log + test set to predicted_logs/ ──────────
            if EXPORT_RESULTS:
                _safe_process = str(process).replace(' ', '_').replace('/', '_')
                _safe_mode    = str(sim_mode).replace(' ', '_').replace('/', '_')

                # Test set — same for all modes, overwriting is fine
                _test_path = os.path.join(_predicted_logs_dir, f'{_safe_process}_test_set.parquet')
                df_test.to_parquet(_test_path, index=False)

                # Predicted log — drop simulated_energy_curves (numpy arrays, not parquet-safe)
                _pred_df = simulated_log_test.drop(
                    columns=[c for c in simulated_log_test.columns
                              if c == 'simulated_energy_curves'],
                    errors='ignore',
                )
                _pred_path = os.path.join(_predicted_logs_dir, f'{_safe_process}_{_safe_mode}.parquet')
                _pred_df.to_parquet(_pred_path, index=False)
                print(f"  💾 Saved predicted log → predicted_logs/{_safe_process}_{_safe_mode}.parquet")

            print(f"\n  🔍 EVALUATION ON TEST SET  [{sim_mode}]")
            print("  " + "="*76)

            with _timed('process_evaluation', process=process, detail=sim_mode, split='TEST'):
                eval_test = comprehensive_simulation_evaluation(simulated_log_test, df_test,
                                                                  process_models=mode_pm,
                                                                  per_case_tag=(process, sim_mode, 'TEST'))

            print(f"\n  📊 COMPARISON PLOTS (TEST)  [{sim_mode}]")
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

            # Feed the simulated test log into the store so the joint
            # duration + profile evaluation can use predicted durations
            # (energy-aware modes add their own entries later; this covers
            # all plain petri_net_* and statistical modes).
            _combined_sim_store.append({
                'process':     process,
                'mode':        sim_mode,
                'sim_df':      simulated_log_test,
                'exp_df':      test_datasets[process].get('expanded'),
                'sensors':     [],   # no energy curves for plain modes
                'act_metrics': {},
            })

            _energy_distribution_pending.append((
                process, sim_mode, simulated_log_test,
                test_datasets[process].get('expanded'),
                dict(flattened),
            ))

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

    if combined_candidates and 'petri_net_combined' in MODES_TO_COMPARE:
        def _combined_selection_score(r):
            """Select best PN on TRAIN: mean of the four classic process-discovery
            dimensions — Fitness, Precision, Generalization, Simplicity (1 = best).

            These are quality scores, so the winner is the *maximum*; -inf is the
            'no usable metric' sentinel. Kept in sync with the Combined-best rule
            in results_process / results_process_discovery /
            results_complete_energy_profile."""
            vals = [float(r.get('train_conformance_metrics_' + k, np.nan))
                    for k in ('fitness', 'precision', 'generalization', 'simplicity')]
            vals = [v for v in vals if np.isfinite(v)]
            return float(np.mean(vals)) if vals else -np.inf

        best_row = max(combined_candidates, key=_combined_selection_score)

        combined_row = dict(best_row)
        combined_row['mode'] = 'petri_net_combined'
        combined_row['selected_mode'] = best_row.get('mode')
        combined_row['selected_mining_algorithm'] = best_row.get('mining_algorithm')

        print("\n" + "─"*80)
        print("  ▶ SIMULATION MODE: PETRI_NET_COMBINED")
        print("─"*80)
        print(
            "  Selected mode based on TRAIN mean of Fitness/Precision/"
            "Generalization/Simplicity (1=best): "
            f"{combined_row['selected_mode']} "
            f"(score={_combined_selection_score(best_row):.4f})"
        )

        process_mode_results.append(combined_row)
        evaluation_results_list.append(combined_row)

        # ── ML+ combined variants (best PN transitions, ML+ duration prediction) ──
        _mlp_modes_requested = [
            m for m in MODES_TO_COMPARE
            if m in ('petri_net_combined_ml_plus_global', 'petri_net_combined_ml_plus_per_act')
        ]
        if _mlp_modes_requested:
            print("\n" + "─"*80)
            print("  TRAINING ML+ MODELS FOR petri_net_combined_ml_plus_* MODES MODES")
            print("─"*80)

            _best_alg   = best_row.get('mining_algorithm')
            _best_pm    = extraction_by_algorithm[_best_alg]['process_models']
            _best_stats = extraction_by_algorithm[_best_alg]['activity_stats_df']

            with _timed('duration_model_training', process=process, detail='ml_plus', split='TRAIN'):
                _glb_tpl, _pa_tpls, _mlp_feat_cols, _act_means, _glb_mean, _mlp_ef_windows = \
                    _mlp_train_models(df_train, train_datasets.get(process, {}).get('expanded'))

            print(f"  feat_cols ({len(_mlp_feat_cols)}): {_mlp_feat_cols}")
            print(f"  Global model: {_glb_tpl[2] if _glb_tpl else 'None'}")
            print(f"  Per-act models trained: {len(_pa_tpls)} activities")

            for _mlp_mode in _mlp_modes_requested:
                _use_global  = _mlp_mode == 'petri_net_combined_ml_plus_global'
                _sim_mode    = 'petri_net_ml_plus_global' if _use_global else 'petri_net_ml_plus_per_act'
                _g_arg       = _glb_tpl if _use_global else None
                _pa_arg      = None     if _use_global else _pa_tpls

                print("\n" + "─"*80)
                print(f"  ▶ SIMULATION MODE: {_mlp_mode.upper()}")
                print("─"*80)

                with _timed('simulation', process=process, detail=_mlp_mode, split='TRAIN'):
                    sim_mlp_train = ProcessSimulation(
                        _best_stats, production_plan,
                        mode=_sim_mode,
                        process_models=_best_pm,
                        mlp_global_tuple=_g_arg,
                        mlp_per_act_tuples=_pa_arg,
                        mlp_feat_cols=_mlp_feat_cols,
                        mlp_ef_windows=_mlp_ef_windows,
                        mlp_activity_means=_act_means,
                        mlp_global_mean=_glb_mean,
                    ).run()
                print(f"\n  Simulated log TRAIN ({_mlp_mode}): {len(sim_mlp_train)} events")

                with _timed('process_evaluation', process=process, detail=_mlp_mode, split=split_label):
                    eval_mlp_train = comprehensive_simulation_evaluation(
                        sim_mlp_train, df_train, process_models=_best_pm,
                        per_case_tag=(process, _mlp_mode, split_label)
                    )

                flattened_mlp = {
                    'process':          process,
                    'mode':             _mlp_mode,
                    'simulation_mode':  _sim_mode,
                    'mining_algorithm': _best_alg,
                    'split':            split_label,
                    'selected_mode':    best_row.get('mode'),
                }
                for _cat, _mets in eval_mlp_train.items():
                    if isinstance(_mets, dict):
                        for _mn, _mv in _mets.items():
                            flattened_mlp[f"train_{_cat}_{_mn}"] = _mv
                    else:
                        flattened_mlp[f"train_{_cat}"] = _mets

                _df_test_mlp = test_datasets[process]['event_log'] if test_datasets else None
                if TEMPORAL_SPLIT and _df_test_mlp is not None and len(_df_test_mlp) > 0:
                    _pp_test_mlp = test_datasets[process]['production_plan']
                    with _timed('simulation', process=process, detail=_mlp_mode, split='TEST'):
                        sim_mlp_test = ProcessSimulation(
                            _best_stats, _pp_test_mlp,
                            mode=_sim_mode,
                            process_models=_best_pm,
                            mlp_global_tuple=_g_arg,
                            mlp_per_act_tuples=_pa_arg,
                            mlp_feat_cols=_mlp_feat_cols,
                            mlp_ef_windows=_mlp_ef_windows,
                            mlp_activity_means=_act_means,
                            mlp_global_mean=_glb_mean,
                        ).run()
                    print(f"\n  Simulated log TEST  ({_mlp_mode}): {len(sim_mlp_test)} events")

                    if EXPORT_RESULTS:
                        _safe_process = str(process).replace(' ', '_').replace('/', '_')
                        _safe_mode    = str(_mlp_mode).replace(' ', '_').replace('/', '_')
                        _pred_df = sim_mlp_test.drop(
                            columns=[c for c in sim_mlp_test.columns
                                     if c == 'simulated_energy_curves'],
                            errors='ignore',
                        )
                        _pred_path = os.path.join(
                            _predicted_logs_dir, f'{_safe_process}_{_safe_mode}.parquet'
                        )
                        _pred_df.to_parquet(_pred_path, index=False)
                        print(f"  Saved predicted log → predicted_logs/{_safe_process}_{_safe_mode}.parquet")

                    with _timed('process_evaluation', process=process, detail=_mlp_mode, split='TEST'):
                        eval_mlp_test = comprehensive_simulation_evaluation(
                            sim_mlp_test, _df_test_mlp, process_models=_best_pm,
                            per_case_tag=(process, _mlp_mode, 'TEST')
                        )
                    for _cat, _mets in eval_mlp_test.items():
                        if isinstance(_mets, dict):
                            for _mn, _mv in _mets.items():
                                flattened_mlp[f"test_{_cat}_{_mn}"] = _mv
                        else:
                            flattened_mlp[f"test_{_cat}"] = _mets

                    _combined_sim_store.append({
                        'process':     process,
                        'mode':        _mlp_mode,
                        'sim_df':      sim_mlp_test,
                        'exp_df':      test_datasets[process].get('expanded'),
                        'sensors':     [],
                        'act_metrics': {},
                    })

                process_mode_results.append(flattened_mlp)
                evaluation_results_list.append(flattened_mlp)

        # ── Median-duration: same best-base PN, but use per-activity median ──
        if 'petri_net_median_duration' in MODES_TO_COMPARE:
            print("\n" + "─"*80)
            print("  ▶ SIMULATION MODE: PETRI_NET_MEDIAN_DURATION")
            print("─"*80)
            _med_alg   = best_row.get('mining_algorithm')
            _med_pm    = extraction_by_algorithm[_med_alg]['process_models']
            _med_stats = extraction_by_algorithm[_med_alg]['activity_stats_df']

            with _timed('simulation', process=process, detail='petri_net_median_duration', split='TRAIN'):
                sim_med_train = ProcessSimulation(
                    _med_stats, production_plan,
                    mode='petri_net_median_duration',
                    process_models=_med_pm,
                ).run()
            print(f"\n  Simulated log TRAIN (petri_net_median_duration): {len(sim_med_train)} events")

            with _timed('process_evaluation', process=process,
                        detail='petri_net_median_duration', split=split_label):
                eval_med_train = comprehensive_simulation_evaluation(
                    sim_med_train, df_train, process_models=_med_pm,
                    per_case_tag=(process, 'petri_net_median_duration', split_label)
                )

            flattened_med = {
                'process':           process,
                'mode':              'petri_net_median_duration',
                'simulation_mode':   'petri_net_median_duration',
                'mining_algorithm':  _med_alg,
                'split':             split_label,
            }
            for _cat, _mets in eval_med_train.items():
                if isinstance(_mets, dict):
                    for _mn, _mv in _mets.items():
                        flattened_med[f"train_{_cat}_{_mn}"] = _mv
                else:
                    flattened_med[f"train_{_cat}"] = _mets

            _df_test_med = test_datasets[process]['event_log'] if test_datasets else None
            if TEMPORAL_SPLIT and _df_test_med is not None and len(_df_test_med) > 0:
                _pp_test_med = test_datasets[process]['production_plan']
                with _timed('simulation', process=process, detail='petri_net_median_duration', split='TEST'):
                    sim_med_test = ProcessSimulation(
                        _med_stats, _pp_test_med,
                        mode='petri_net_median_duration',
                        process_models=_med_pm,
                    ).run()
                print(f"\n  Simulated log TEST  (petri_net_median_duration): {len(sim_med_test)} events")

                with _timed('process_evaluation', process=process,
                            detail='petri_net_median_duration', split='TEST'):
                    eval_med_test = comprehensive_simulation_evaluation(
                        sim_med_test, _df_test_med, process_models=_med_pm,
                        per_case_tag=(process, 'petri_net_median_duration', 'TEST')
                    )
                for _cat, _mets in eval_med_test.items():
                    if isinstance(_mets, dict):
                        for _mn, _mv in _mets.items():
                            flattened_med[f"test_{_cat}_{_mn}"] = _mv
                    else:
                        flattened_med[f"test_{_cat}"] = _mets

            process_mode_results.append(flattened_med)
            evaluation_results_list.append(flattened_med)

    # ── ML+ variants for every mined algorithm ───────────────────────────────
    # Train ML models once (algorithm-agnostic: trained on raw event log).
    _any_algo_mlp = any(
        m for m in MODES_TO_COMPARE
        if m.startswith('petri_net_') and 'combined' not in m
        and (m.endswith('_ml_plus_global') or m.endswith('_ml_plus_per_act'))
    )
    if _any_algo_mlp:
        print("\n" + "─"*80)
        print("  TRAINING ML+ MODELS (shared across all algorithms)")
        print("─"*80)
        # Timed: these are the activity-duration models behind every reported
        # ml_plus_global / ml_plus_per_act mode, so the runtime table needs their
        # cost. Fitted once per process, shared by all algorithms.
        with _timed('duration_model_training', process=process,
                    detail='ml_plus (shared)', split='TRAIN'):
            _mlp_glb_tpl, _mlp_pa_tpls, _mlp_feat_cols, _mlp_act_means, _mlp_glb_mean, _mlp_ef_windows = \
                _mlp_train_models(df_train,
                                  train_datasets.get(process, {}).get('expanded'),
                                  # ef_* lookup over the unsplit series — the
                                  # simulator queries it at test timestamps.
                                  ef_expanded_df=process_datasets_to_model.get(process, {}).get('expanded'))
        print(f"  feat_cols ({len(_mlp_feat_cols)}): {_mlp_feat_cols}")
        print(f"  Global model: {_mlp_glb_tpl[2] if _mlp_glb_tpl else 'None'}")
        print(f"  Per-act models trained: {len(_mlp_pa_tpls)} activities")

        for _algo in sorted(extraction_by_algorithm.keys()):
            _algo_mlp_modes = [
                m for m in MODES_TO_COMPARE
                if m in (f'petri_net_{_algo}_ml_plus_global',
                         f'petri_net_{_algo}_ml_plus_per_act')
            ]
            if not _algo_mlp_modes:
                continue

            _algo_pm    = extraction_by_algorithm[_algo]['process_models']
            _algo_stats = extraction_by_algorithm[_algo]['activity_stats_df']

            for _algo_mode in _algo_mlp_modes:
                _use_global = _algo_mode.endswith('_ml_plus_global')
                _sim_mode   = 'petri_net_ml_plus_global' if _use_global else 'petri_net_ml_plus_per_act'
                _g_arg      = _mlp_glb_tpl if _use_global else None
                _pa_arg     = None          if _use_global else _mlp_pa_tpls

                print("\n" + "─"*80)
                print(f"  ▶ SIMULATION MODE: {_algo_mode.upper()}")
                print("─"*80)

                with _timed('simulation', process=process, detail=_algo_mode, split='TRAIN'):
                    sim_amlp_train = ProcessSimulation(
                        _algo_stats, production_plan,
                        mode=_sim_mode,
                        process_models=_algo_pm,
                        mlp_global_tuple=_g_arg,
                        mlp_per_act_tuples=_pa_arg,
                        mlp_feat_cols=_mlp_feat_cols,
                        mlp_ef_windows=_mlp_ef_windows,
                        mlp_activity_means=_mlp_act_means,
                        mlp_global_mean=_mlp_glb_mean,
                    ).run()
                print(f"\n  Simulated log TRAIN ({_algo_mode}): {len(sim_amlp_train)} events")

                with _timed('process_evaluation', process=process, detail=_algo_mode, split=split_label):
                    eval_amlp_train = comprehensive_simulation_evaluation(
                        sim_amlp_train, df_train, process_models=_algo_pm,
                        per_case_tag=(process, _algo_mode, split_label)
                    )

                flattened_amlp = {
                    'process':          process,
                    'mode':             _algo_mode,
                    'simulation_mode':  _sim_mode,
                    'mining_algorithm': _algo,
                    'split':            split_label,
                    'selected_mode':    f'petri_net_{_algo}',
                }
                for _cat, _mets in eval_amlp_train.items():
                    if isinstance(_mets, dict):
                        for _mn, _mv in _mets.items():
                            flattened_amlp[f"train_{_cat}_{_mn}"] = _mv
                    else:
                        flattened_amlp[f"train_{_cat}"] = _mets

                _df_test_amlp = test_datasets[process]['event_log'] if test_datasets else None
                if TEMPORAL_SPLIT and _df_test_amlp is not None and len(_df_test_amlp) > 0:
                    _pp_test_amlp = test_datasets[process]['production_plan']
                    with _timed('simulation', process=process, detail=_algo_mode, split='TEST'):
                        sim_amlp_test = ProcessSimulation(
                            _algo_stats, _pp_test_amlp,
                            mode=_sim_mode,
                            process_models=_algo_pm,
                            mlp_global_tuple=_g_arg,
                            mlp_per_act_tuples=_pa_arg,
                            mlp_feat_cols=_mlp_feat_cols,
                            mlp_ef_windows=_mlp_ef_windows,
                            mlp_activity_means=_mlp_act_means,
                            mlp_global_mean=_mlp_glb_mean,
                        ).run()
                    print(f"\n  Simulated log TEST  ({_algo_mode}): {len(sim_amlp_test)} events")

                    if EXPORT_RESULTS:
                        _safe_process = str(process).replace(' ', '_').replace('/', '_')
                        _safe_mode    = str(_algo_mode).replace(' ', '_').replace('/', '_')
                        _pred_df = sim_amlp_test.drop(
                            columns=[c for c in sim_amlp_test.columns
                                     if c == 'simulated_energy_curves'],
                            errors='ignore',
                        )
                        _pred_path = os.path.join(
                            _predicted_logs_dir, f'{_safe_process}_{_safe_mode}.parquet'
                        )
                        _pred_df.to_parquet(_pred_path, index=False)
                        print(f"  Saved predicted log → predicted_logs/{_safe_process}_{_safe_mode}.parquet")

                    with _timed('process_evaluation', process=process, detail=_algo_mode, split='TEST'):
                        eval_amlp_test = comprehensive_simulation_evaluation(
                            sim_amlp_test, _df_test_amlp, process_models=_algo_pm,
                            per_case_tag=(process, _algo_mode, 'TEST')
                        )
                    for _cat, _mets in eval_amlp_test.items():
                        if isinstance(_mets, dict):
                            for _mn, _mv in _mets.items():
                                flattened_amlp[f"test_{_cat}_{_mn}"] = _mv
                        else:
                            flattened_amlp[f"test_{_cat}"] = _mets

                    _combined_sim_store.append({
                        'process':     process,
                        'mode':        _algo_mode,
                        'sim_df':      sim_amlp_test,
                        'exp_df':      test_datasets[process].get('expanded'),
                        'sensors':     [],
                        'act_metrics': {},
                    })

                    _energy_distribution_pending.append((
                        process, _algo_mode, sim_amlp_test,
                        test_datasets[process].get('expanded'),
                        dict(flattened_amlp),
                    ))

                process_mode_results.append(flattened_amlp)
                evaluation_results_list.append(flattened_amlp)

    # ── Budget + ML+ duration variants ───────────────────────────────────────
    # petri_net_budget (each case generated to its predicted total-duration
    # budget) combined with ML+ per-activity / global duration prediction — for
    # better INTERNAL timing under the same correct total span. Reuses the ML+
    # tuples trained above (_mlp_glb_tpl/_mlp_pa_tpls; guaranteed present since
    # requesting a budget_ml_plus mode sets _any_algo_mlp) and this process's
    # case-duration predictor (_case_duration_pipeline). Uses MINING_ALGORITHM's
    # net, plain ProcessSimulation (no WIP/RO pass).
    _bud_mlp_modes = [
        m for m in MODES_TO_COMPARE
        if m in ('petri_net_budget_ml_plus_global', 'petri_net_budget_ml_plus_per_act')
    ]
    if _bud_mlp_modes and _any_algo_mlp:
        _bud_mlp_pm    = extraction_by_algorithm[MINING_ALGORITHM]['process_models']
        _bud_mlp_stats = extraction_by_algorithm[MINING_ALGORITHM]['activity_stats_df']

        for _bud_mode in _bud_mlp_modes:
            _use_global = _bud_mode.endswith('_ml_plus_global')
            _g_arg  = _mlp_glb_tpl if _use_global else None
            _pa_arg = None if _use_global else _mlp_pa_tpls

            print("\n" + "─"*80)
            print(f"  ▶ SIMULATION MODE: {_bud_mode.upper()}")
            print("─"*80)

            with _timed('simulation', process=process, detail=_bud_mode, split='TRAIN'):
                sim_budmlp_train = ProcessSimulation(
                    _bud_mlp_stats, production_plan,
                    mode=_bud_mode,
                    process_models=_bud_mlp_pm,
                    case_duration_pipeline=_case_duration_pipeline,
                    mlp_global_tuple=_g_arg, mlp_per_act_tuples=_pa_arg,
                    mlp_feat_cols=_mlp_feat_cols, mlp_activity_means=_mlp_act_means,
                    mlp_ef_windows=_mlp_ef_windows,
                    mlp_global_mean=_mlp_glb_mean,
                ).run()
            print(f"\n  Simulated log TRAIN ({_bud_mode}): {len(sim_budmlp_train)} events")

            with _timed('process_evaluation', process=process, detail=_bud_mode, split=split_label):
                eval_budmlp_train = comprehensive_simulation_evaluation(
                    sim_budmlp_train, df_train, process_models=_bud_mlp_pm,
                    per_case_tag=(process, _bud_mode, split_label)
                )
            flattened_budmlp = {
                'process':          process,
                'mode':             _bud_mode,
                'simulation_mode':  _bud_mode,
                'mining_algorithm': MINING_ALGORITHM,
                'split':            split_label,
            }
            for _cat, _mets in eval_budmlp_train.items():
                if isinstance(_mets, dict):
                    for _mn, _mv in _mets.items():
                        flattened_budmlp[f"train_{_cat}_{_mn}"] = _mv
                else:
                    flattened_budmlp[f"train_{_cat}"] = _mets

            _df_test_budmlp = test_datasets[process]['event_log'] if test_datasets else None
            if TEMPORAL_SPLIT and _df_test_budmlp is not None and len(_df_test_budmlp) > 0:
                _pp_test_budmlp = test_datasets[process]['production_plan']
                with _timed('simulation', process=process, detail=_bud_mode, split='TEST'):
                    sim_budmlp_test = ProcessSimulation(
                        _bud_mlp_stats, _pp_test_budmlp,
                        mode=_bud_mode,
                        process_models=_bud_mlp_pm,
                        case_duration_pipeline=_case_duration_pipeline,
                        mlp_global_tuple=_g_arg, mlp_per_act_tuples=_pa_arg,
                        mlp_feat_cols=_mlp_feat_cols, mlp_activity_means=_mlp_act_means,
                        mlp_ef_windows=_mlp_ef_windows,
                        mlp_global_mean=_mlp_glb_mean,
                    ).run()
                print(f"\n  Simulated log TEST  ({_bud_mode}): {len(sim_budmlp_test)} events")

                if EXPORT_RESULTS:
                    _safe_process = str(process).replace(' ', '_').replace('/', '_')
                    _safe_mode    = str(_bud_mode).replace(' ', '_').replace('/', '_')
                    _pred_df = sim_budmlp_test.drop(
                        columns=[c for c in sim_budmlp_test.columns
                                 if c == 'simulated_energy_curves'],
                        errors='ignore',
                    )
                    _pred_path = os.path.join(
                        _predicted_logs_dir, f'{_safe_process}_{_safe_mode}.parquet'
                    )
                    _pred_df.to_parquet(_pred_path, index=False)
                    print(f"  Saved predicted log → predicted_logs/{_safe_process}_{_safe_mode}.parquet")

                with _timed('process_evaluation', process=process, detail=_bud_mode, split='TEST'):
                    eval_budmlp_test = comprehensive_simulation_evaluation(
                        sim_budmlp_test, _df_test_budmlp, process_models=_bud_mlp_pm,
                        per_case_tag=(process, _bud_mode, 'TEST')
                    )
                for _cat, _mets in eval_budmlp_test.items():
                    if isinstance(_mets, dict):
                        for _mn, _mv in _mets.items():
                            flattened_budmlp[f"test_{_cat}_{_mn}"] = _mv
                    else:
                        flattened_budmlp[f"test_{_cat}"] = _mets

                _combined_sim_store.append({
                    'process':     process,
                    'mode':        _bud_mode,
                    'sim_df':      sim_budmlp_test,
                    'exp_df':      test_datasets[process].get('expanded'),
                    'sensors':     [],
                    'act_metrics': {},
                })

                _energy_distribution_pending.append((
                    process, _bud_mode, sim_budmlp_test,
                    test_datasets[process].get('expanded'),
                    dict(flattened_budmlp),
                ))

            process_mode_results.append(flattened_budmlp)
            evaluation_results_list.append(flattened_budmlp)



if RUN_PROCESS_MODELLING:
    # Convert the results list into a DataFrame
    evaluation_results_df = pd.DataFrame(evaluation_results_list)

    # Reorder columns to place key columns first
    priority_cols = ['process', 'mode', 'split']
    for prefix in ['train', 'test']:
        for col_name in ['overall_error', 'quality_assessment']:
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

        # ── Per-case process metrics (for post-hoc confidence intervals) ────
        # process_eval_results.parquet holds ONE aggregated row per
        # (process, mode) — every process metric in it is a median over cases,
        # so nothing in that file can produce a CI. This companion file keeps
        # the underlying sample: one row per (process, mode, split, case_id),
        # with the same columns the medians are computed from. Bootstrap any
        # of them directly, e.g.
        #     d = pd.read_parquet('process_eval_per_case.parquet')
        #     s = d.query("process=='process_1' and mode=='petri_net_budget' "
        #                 "and split=='TEST'")['dur_wape'].dropna().to_numpy()
        #     boot = [np.median(np.random.choice(s, len(s), replace=True))
        #             for _ in range(10000)]
        #     lo, hi = np.percentile(boot, [2.5, 97.5])
        if _per_case_process_metrics:
            _pc_df = pd.concat(_per_case_process_metrics, ignore_index=True)
            _pc_path = os.path.join(_run_dir, 'process_eval_per_case.parquet')
            _pc_df.to_parquet(_pc_path, index=False)
            print(f"Saved per-case process metrics → {_pc_path} "
                  f"({len(_pc_df)} rows, "
                  f"{_pc_df.groupby(['process', 'mode', 'split']).ngroups} groups)")
        else:
            print("⚠️  No per-case process metrics collected — "
                  "process metric CIs will not be computable for this run.")

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
    display(Markdown("---"))
    display(Markdown("## 📊 Training Quality: ALL Processes Combined"))
    display(Markdown("*Aggregated across all processes on training data — same metrics as test (0 = best)*"))
    _hm_train_all_mean_path   = os.path.join(_process_results_dir, 'process_train_heatmap_all_mean.png')   if EXPORT_RESULTS and '_process_results_dir' in dir() else None
    _hm_train_all_median_path = os.path.join(_process_results_dir, 'process_train_heatmap_all_median.png') if EXPORT_RESULTS and '_process_results_dir' in dir() else None
    _plot_short_heatmap(evaluation_results_df, "Training Quality (Mean): All Processes",   save_path=_hm_train_all_mean_path,   agg='mean',   split='train')
    _plot_short_heatmap(evaluation_results_df, "Training Quality (Median): All Processes", save_path=_hm_train_all_median_path, agg='median', split='train')

    # ── Per-process breakdown: show modes sorted by test_overall_error ───────
    report("\n" + "="*80)
    report("PER-PROCESS RESULTS — MODES SORTED BY test_overall_error (0=best)")
    report("="*80)

    sort_col = 'test_overall_error'
    if sort_col in evaluation_results_df.columns:
        # Columns to display (key columns only for readability)
        display_cols = ['process', 'mode', 'split']
        for prefix in ['test', 'train']:
            for col_name in ['overall_error', 'quality_assessment']:
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
            sorted_grp = grp.sort_values(sort_col, ascending=True)
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
    display(Markdown("---"))
    display(Markdown("# 📊 FINAL CONSOLIDATED PERFORMANCE: TEST SET"))
    display(Markdown("*EvtRatioErr · DurErr(whole) · DurErr(activ) · EdgeF1Err · FitnessErr · PrecisionErr · Overall  (0 = best)*"))

    # ── All-process heatmap ────────────────────────────────────────────────────
    _short_path_mean   = os.path.join(_process_results_dir, 'process_test_short_mean.png')   if EXPORT_RESULTS and '_process_results_dir' in dir() else None
    _short_path_median = os.path.join(_process_results_dir, 'process_test_short_median.png') if EXPORT_RESULTS and '_process_results_dir' in dir() else None
    _plot_short_heatmap(evaluation_results_df, "Process Results — All Processes (Mean)",   save_path=_short_path_mean,   agg='mean')
    _plot_short_heatmap(evaluation_results_df, "Process Results — All Processes (Median)", save_path=_short_path_median, agg='median')

    # ── Per-process heatmaps ───────────────────────────────────────────────────
    display(Markdown("## 📊 Per Process"))
    for _proc_name, _proc_grp in evaluation_results_df.groupby('process'):
        display(Markdown(f"### {_proc_name.upper()}"))
        _short_proc_mean   = os.path.join(_process_results_dir, f'process_test_short_{_proc_name}_mean.png')   if EXPORT_RESULTS and '_process_results_dir' in dir() else None
        _short_proc_median = os.path.join(_process_results_dir, f'process_test_short_{_proc_name}_median.png') if EXPORT_RESULTS and '_process_results_dir' in dir() else None
        _plot_short_heatmap(_proc_grp, f"{_proc_name} (Mean)",   save_path=_short_proc_mean,   agg='mean')
        _plot_short_heatmap(_proc_grp, f"{_proc_name} (Median)", save_path=_short_proc_median, agg='median')



# %%
# ── CURVE-ONLY: train ALL approaches without process modelling ────────────────
# Trains every approach in APPROACHES side by side so results can be compared
# in the evaluation cell below.
if RUN_CURVE_ONLY_EVALUATION:
    # Only the predictors are needed here: training goes through
    # _train_curve_only_worker / _train_seq2seq_worker in sim_extractor, which
    # call the build_and_train_pipeline_* functions themselves.
    from utils.sim_extractor import (
        split_curves,
        split_curves_with_prev_activity,
        build_and_train_pipeline_median,
        predict_raw_curve,
        predict_raw_curve_median,
        predict_raw_curve_exog,
        predict_raw_curve_exog_prev_activity,
        evaluate_pipeline_on_test,
        predict_raw_curve_ml_only,
        predict_raw_curve_seq2seq,
        predict_raw_curve_seq2seq_only,
        predict_raw_curve_seq2seq_external,
        predict_raw_curve_seq2seq_iom,
    )

    all_energy_pipelines                  = {}   # "Baseline": ONE median LEVEL per SENSOR (flat), pooled over all activities/objects (naive floor)
    all_energy_pipelines_median_activity_sensor = {}   # "Median per Activity & Sensor": one median LEVEL per (sensor, activity, object), flat
    all_energy_pipelines_ml_dtw    = {}   # ml_dtw (DBA + DTW + regression -- the former 'baseline')
    all_energy_pipelines_ml_external   = {}   # DTW + Ext. Factors (+ prev-activity name)
    all_energy_pipelines_seq2seq              = {}   # DTW + Seq2Seq
    all_energy_pipelines_seq2seq_only         = {}   # Seq2Seq only (no DTW)
    all_energy_pipelines_seq2seq_external = {}  # DTW + Seq2Seq + Ext. Factors (+ prev-activity name)
    all_energy_pipelines_seq2seq_iom          = {}   # Seq2Seq + IOM selection (Woerrlein & Strassburger)
    all_energy_pipelines_ml_only                  = {}   # ML, linear resample encode+decode
    all_energy_pipelines_ml_step_dtw              = {}   # segment durations+levels via DTW correspondence
    all_energy_pipelines_ml_step_dtw_smooth       = {}   # ml_step_dtw with interpolated (jump-free) level gains
    all_energy_pipelines_ml_external_wcounts      = {}   # DTW + ML + Ext. (count-weighted)
    all_energy_pipelines_ml_external_wmetric      = {}   # DTW + ML + Ext. (metric-weighted)

    # Curve regressors are defined once in sim_extractor._make_curve_models and
    # used there by _train_curve_only_worker; imported here only so the run log
    # records which candidates competed.
    from utils.sim_extractor import _make_curve_models
    _CURVE_MODELS = _make_curve_models()
    print(f"  Curve regressors competed per (sensor, activity, object): "
          f"{', '.join(_CURVE_MODELS)}")

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
                if c.endswith('_to_model')
                and _df_train_exp[c].dtype in ('float64', 'float32', 'int64', 'int32')
            ]
        if not _sensors:
            print(f"  Skipping {_proc}: no sensor columns found.")
            continue

        # Scope limiter for cheap end-to-end validation runs: cut the curve stage
        # to the first N sensors of each process so the whole path (training,
        # reassembly, evaluation, parquet write) executes in minutes instead of
        # hours. Unset -> no limit, so a real run is untouched. Deliberately
        # NOT applied to the energy-modelling section, which is a different stage
        # with its own sensor resolution.
        _curve_max_sensors = os.environ.get('PIPELINE_CURVE_MAX_SENSORS')
        if _curve_max_sensors:
            _n_keep = max(1, int(_curve_max_sensors))
            if len(_sensors) > _n_keep:
                print(f"  ⚠️  PIPELINE_CURVE_MAX_SENSORS={_n_keep}: curve stage limited to "
                      f"{_n_keep}/{len(_sensors)} sensor(s) for {_proc} — THIS IS A SCOPED "
                      f"VALIDATION RUN, its results are not comparable to a full run.")
                _sensors = _sensors[:_n_keep]

        if not _activities and 'activity_log' in _df_train_exp.columns:
            _activities = _df_train_exp['activity_log'].dropna().unique().tolist()
        if not _objects and 'object_log' in _df_train_exp.columns:
            _objects = _df_train_exp['object_log'].dropna().unique().tolist()
        if not _objects:
            # No object dimension — inject sentinel so _combos is non-empty
            print(f"  No object_log values for {_proc} — treating as single-object process ('_all_').")
            _df_train_exp = _df_train_exp.copy()
            if 'object_log' not in _df_train_exp.columns:
                _df_train_exp['object_log'] = '_all_'
            else:
                _df_train_exp['object_log'] = _df_train_exp['object_log'].fillna('_all_')
            if 'object_attributes_log' not in _df_train_exp.columns:
                _df_train_exp['object_attributes_log'] = [{} for _ in range(len(_df_train_exp))]
            _objects = ['_all_']

        # Detect ef_* columns. They are attached to EVERY curve's 'exog_values',
        # but only the exog approaches (ml_external, seq2seq_external)
        # actually consume them as features -- that
        # difference is what the "+ Ext. Factors" comparison isolates.
        _ef_cols = [
            c for c in _df_train_exp.columns
            if c.startswith('ef_')
            and _df_train_exp[c].dtype in ('float64', 'float32', 'int64', 'int32')
        ]

        print(f"\n{'='*60}")
        print(f"CURVE-ONLY TRAINING — {_proc.upper()}")
        print(f"  Sensors          : {_sensors}")
        print(f"  Activities       : {_activities}")
        print(f"  Objects          : {_objects}")
        print(f"  External factors : {_ef_cols}")
        print(f"{'='*60}")

        _pipelines_baseline            = {}   # per-SENSOR median (built below, pooled)
        _pipelines_median_activity_sensor = {}   # per-(sensor,activity,object) median (from worker)
        _pipelines_ml_dtw        = {}

        _pipelines_ml_external   = {}
        _pipelines_seq2seq                  = {}
        _pipelines_seq2seq_only             = {}
        _pipelines_seq2seq_external    = {}
        _pipelines_seq2seq_iom              = {}
        _pipelines_ml_only                = {}
        _pipelines_ml_step_dtw            = {}   # segment durations+levels via DTW correspondence
        _pipelines_ml_step_dtw_smooth     = {}   # ml_step_dtw with interpolated (jump-free) level gains
        _pipelines_ml_external_wcounts      = {}
        _pipelines_ml_external_wmetric      = {}

        # ── Parallel training for all sklearn-based approaches ───────────────
        # One worker per (sensor, activity, object) combo — each trains its own
        # barycenter and model on a homogeneous set of curves.
        # Seq2seq approaches run afterwards in their own process pool.
        from utils.sim_extractor import _train_curve_only_worker, build_prev_activity_energy_map
        import concurrent.futures, os as _os

        # Previous-activity energy levels for 'ml_external': per (sensor, activity)
        # medians over the TRAINING frame. Built here, once per process and across
        # ALL activities, because each worker only sees the one activity it trains
        # on and could never assemble a predecessor's level itself. Train-only by
        # construction, and stored in the pipeline so predict re-derives it from
        # the same numbers -- no test information reaches the features.
        _prev_act_energy_maps = build_prev_activity_energy_map(_df_train_exp, _sensors)
        print(f"  Prev-activity energy map : {len(_prev_act_energy_maps)}/{len(_sensors)} sensors")

        # 'baseline' (per-sensor pooled median) is NOT trained by the per-combo
        # worker — it is built separately below. The worker trains the per-combo
        # median under 'median_activity_sensor'.
        _sklearn_approaches = [a for a in APPROACHES
                               if a in {'median_activity_sensor','ml_dtw','ml_external','ml_only',
                                        'ml_step_dtw','ml_step_dtw_smooth',
                                        'ml_external_wcounts','ml_external_wmetric'}]
        _seq2seq_approaches = [a for a in APPROACHES
                               if a in {'seq2seq','seq2seq_only','seq2seq_external','seq2seq_iom'}]

        # Restrict per-sensor to the objects/activities where that sensor actually
        # carries signal — e.g. process_1's destillation sensors have nothing to do
        # with autoclaving activities, so don't train/evaluate that cross product.
        _combos    = build_sensor_activity_object_combos(_df_train_exp, _sensors, _activities, _objects)
        _n_workers = _pool_workers(len(_combos))

        # ── Seq2seq approaches — one worker per (sensor, activity, object) combo ──
        # Trained FIRST, before the sklearn pool. These are the only approaches
        # that have ever wedged or died in the pool (numba/softDTW in a forked
        # child), so they run while the log is short and a failure is visible in
        # the first minutes of the process — not after the sklearn pool has burnt
        # 7-11 minutes per process and buried the cause under 5k lines of output.
        if _seq2seq_approaches and _combos:
            from utils.sim_extractor import _train_seq2seq_worker
            _s2s_n_workers = _pool_workers(len(_combos))
            print(f"\n  Parallel seq2seq training: {len(_combos)} combos × "
                  f"{len(_seq2seq_approaches)} approaches across {_s2s_n_workers} workers...")
            _s2s_t0 = _time.perf_counter()
            # Watchdogged pool — see _run_pool_with_stall_watchdog. A plain
            # ProcessPoolExecutor here deadlocked twice and blocked the whole run
            # indefinitely; this bounds a stall to POOL_STALL_TIMEOUT and still
            # produces every model, via a fresh pool or inline as a last resort.
            _s2s_tasks = [
                ((_s, _a, _o), (
                    _s, _a, _o, _df_train_exp, _seq2seq_approaches, _ef_cols,
                    SEQ2SEQ_HIDDEN_SIZE, SEQ2SEQ_NUM_LAYERS, SEQ2SEQ_DROPOUT,
                    SEQ2SEQ_EPOCHS, SEQ2SEQ_BATCH_SIZE, SEQ2SEQ_LR,
                    SEQ2SEQ_TEACHER_FORCING, SEQ2SEQ_PATIENCE,
                ))
                for _s, _a, _o in _combos
            ]
            _s2s_results = []
            for _key, _res in _run_pool_with_stall_watchdog(
                    _train_seq2seq_worker, _s2s_tasks, _s2s_n_workers, 'seq2seq'):
                # '_elapsed' was per-task wall clock before; with retries and an
                # inline fallback in play a per-task figure would be misleading,
                # so report the stage total instead.
                _res['_elapsed'] = _time.perf_counter() - _s2s_t0
                _s2s_results.append(_res)
            _s2s_elapsed = _time.perf_counter() - _s2s_t0
            print(f"  seq2seq training done in {_s2s_elapsed:.1f}s")
            _record_runtime('curve_training_pool', _s2s_elapsed, process=_proc,
                            detail='seq2seq', split='TRAIN', n_items=len(_combos),
                            parallel_workers=_s2s_n_workers)
            _record_curve_training_timings(_s2s_results, _proc, 'seq2seq')
            # Same fixed key order as the sklearn pool above.
            _s2s_results.sort(key=lambda r: (str(r['sensor']), str(r['activity']), str(r['object'])))

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
                    print(f"  [{_s}|{_a}|{_o}] seq2seq       val_loss={_ep['val_loss']:.5f}  cell={_ep.get('cell_type', 'lstm')}  ({_s_elapsed:.1f}s)")
                if 'seq2seq_only' in _r2:
                    _ep = _r2['seq2seq_only']
                    _ep['variable_name'] = _s
                    _pipelines_seq2seq_only.setdefault(_s, {}).setdefault(_a, {})[_o] = {
                        'reference_curve': None,
                        'predict_fn':      (lambda ep: lambda rv, act, attrs: predict_raw_curve_seq2seq_only(rv, act, attrs, pipeline=ep))(_ep),
                        'full_pipeline':   _ep,
                    }
                    print(f"  [{_s}|{_a}|{_o}] seq2seq_only  val_loss={_ep['val_loss']:.5f}  cell={_ep.get('cell_type', 'lstm')}  ({_s_elapsed:.1f}s)")
                if 'seq2seq_iom' in _r2:
                    _ep = _r2['seq2seq_iom']
                    _ep['variable_name'] = _s
                    _pipelines_seq2seq_iom.setdefault(_s, {}).setdefault(_a, {})[_o] = {
                        # The softDTW reference this leaf was SELECTED against —
                        # metadata only, the predictor does no DTW decode.
                        'reference_curve': _ep['reference_curve'],
                        'predict_fn':      (lambda ep: lambda rv, act, attrs: predict_raw_curve_seq2seq_iom(rv, act, attrs, pipeline=ep))(_ep),
                        'full_pipeline':   _ep,
                    }
                    # val_loss here is the IOM metric (MSE vs the DTW reference),
                    # not the pointwise validation loss the lines above print —
                    # both are shown so they are never silently conflated.
                    print(f"  [{_s}|{_a}|{_o}] seq2seq_iom   iom_mse={_ep['val_loss']:.5f}  "
                          f"pointwise={_ep.get('val_loss_pointwise', float('nan')):.5f}  "
                          f"sigma={_ep.get('iom_sigma_length', float('nan')):.3f}  "
                          f"epoch={_ep.get('iom_epoch')}  cell={_ep.get('cell_type', 'lstm')}  ({_s_elapsed:.1f}s)")
                if 'seq2seq_external' in _r2:
                    _ep = _r2['seq2seq_external']
                    _ep['variable_name'] = _s
                    _pipelines_seq2seq_external.setdefault(_s, {}).setdefault(_a, {})[_o] = {
                        'reference_curve': _ep['reference_curve'],
                        'predict_fn':      (lambda ep: lambda rv, act, attrs, exog=None: predict_raw_curve_seq2seq_external(rv, act, attrs, pipeline=ep, exog_values=exog or {}))(_ep),
                        'full_pipeline':   _ep,
                    }
                    print(f"  [{_s}|{_a}|{_o}] seq2seq_external  val_loss={_ep['val_loss']:.5f}  cell={_ep.get('cell_type', 'lstm')}  ({_s_elapsed:.1f}s)")

        if _sklearn_approaches and _combos:
            print(f"\n  Parallel sklearn training: {len(_combos)} combos × {len(_sklearn_approaches)} approaches "
                  f"across {_n_workers} workers...")
            _sklearn_t0 = _time.perf_counter()
            # Same watchdogged pool as seq2seq below. This one has never wedged,
            # but it forks from the same parent, so a bare ProcessPoolExecutor
            # here carries the identical failure mode: a dead or stuck worker
            # would hang the run with no bound and no fallback.
            _sk_tasks = [
                ((s, a, o), (
                    s, a, o, _df_train_exp, _sklearn_approaches, _ef_cols,
                    None, 0.2, CURVE_OPTIMIZE_HYPERPARAMS, CURVE_N_OPTUNA_TRIALS,
                    _prev_act_energy_maps.get(s),
                ))
                for s, a, o in _combos
            ]
            _worker_results = [
                _res for _key, _res in _run_pool_with_stall_watchdog(
                    _train_curve_only_worker, _sk_tasks, _n_workers, 'sklearn')
            ]
            _sklearn_elapsed = _time.perf_counter() - _sklearn_t0
            print(f"  sklearn training done in {_sklearn_elapsed:.1f}s")
            _record_runtime('curve_training_pool', _sklearn_elapsed, process=_proc,
                            detail='sklearn', split='TRAIN', n_items=len(_combos),
                            parallel_workers=_n_workers)
            _record_curve_training_timings(_worker_results, _proc, 'sklearn')

            # Reassemble in a fixed key order, not pool-completion order: each
            # worker's models are already seeded from its own key, so only the
            # dict INSERTION order was still schedule-dependent — and that
            # leaks into the row order of every saved results table.
            _worker_results.sort(key=lambda r: (str(r['sensor']), str(r['activity']), str(r['object'])))

            # Reassemble into per-sensor dicts keyed [sensor][activity][object]
            for _r in _worker_results:
                _s, _a, _o = _r['sensor'], _r['activity'], _r['object']
                if _r.get('skipped'):
                    print(f"  ⚠️  Skipped {_s}|{_a}|{_o} (too few curves).")
                    continue
                if 'median_activity_sensor' in _r:
                    # "Median per Activity & Sensor": median curve for this
                    # (sensor, activity, object), no model.
                    _pipelines_median_activity_sensor.setdefault(_s, {}).setdefault(_a, {})[_o] = {
                        'reference_curve': _r['median_activity_sensor']['reference_curve'],
                        'predict_fn':      (lambda ep: lambda rv, act, attrs: predict_raw_curve_median(rv, act, attrs, pipeline=ep))(_r['median_activity_sensor']),
                        'full_pipeline':   _r['median_activity_sensor'],
                    }
                if 'ml_dtw' in _r:
                    # The former 'baseline': DBA barycenter + DTW alignment + regression.
                    _pipelines_ml_dtw.setdefault(_s, {}).setdefault(_a, {})[_o] = {
                        'reference_curve': _r['ml_dtw']['reference_curve'],
                        'predict_fn':      (lambda ep: lambda rv, act, attrs: predict_raw_curve(rv, act, attrs, pipeline=ep))(_r['ml_dtw']),
                        'full_pipeline':   _r['ml_dtw'],
                    }
                if 'ml_external' in _r:
                    _pipelines_ml_external.setdefault(_s, {}).setdefault(_a, {})[_o] = {
                        'reference_curve': _r['ml_external']['reference_curve'],
                        'predict_fn':      (lambda ep: lambda rv, act, attrs, exog=None: predict_raw_curve_exog_prev_activity(rv, act, attrs, pipeline=ep, exog_values=exog or {}))(_r['ml_external']),
                        'full_pipeline':   _r['ml_external'],
                    }
                # Train/eval-gap variants: share ml_external's predictor
                # (they differ only in the FIT's row weights).
                for _v, _dst in (('ml_external_wcounts', _pipelines_ml_external_wcounts),
                                 ('ml_external_wmetric', _pipelines_ml_external_wmetric)):
                    if _v in _r:
                        _dst.setdefault(_s, {}).setdefault(_a, {})[_o] = {
                            'reference_curve': _r[_v]['reference_curve'],
                            'predict_fn':      (lambda ep: lambda rv, act, attrs, exog=None: predict_raw_curve_exog_prev_activity(rv, act, attrs, pipeline=ep, exog_values=exog or {}))(_r[_v]),
                            'full_pipeline':   _r[_v],
                        }
                if 'ml_only' in _r:
                    _pipelines_ml_only.setdefault(_s, {}).setdefault(_a, {})[_o] = {
                        'reference_curve': None,
                        'predict_fn':      (lambda ep: lambda rv, act, attrs: predict_raw_curve_ml_only(rv, act, attrs, pipeline=ep))(_r['ml_only']),
                        'full_pipeline':   _r['ml_only'],
                    }
                if 'ml_step_dtw' in _r:
                    # exog-aware signature like ml_external: the duration and
                    # level models read ef_* window means, so the values have to
                    # reach the predictor.
                    _pipelines_ml_step_dtw.setdefault(_s, {}).setdefault(_a, {})[_o] = {
                        'reference_curve': _r['ml_step_dtw']['reference_curve'],
                        'predict_fn':      (lambda ep: lambda rv, act, attrs, exog=None: predict_raw_curve_step_dtw(rv, act, attrs, pipeline=ep, exog_values=exog or {}))(_r['ml_step_dtw']),
                        'full_pipeline':   _r['ml_step_dtw'],
                    }
                if 'ml_step_dtw_smooth' in _r:
                    # Same predictor as ml_step_dtw — the pipeline's stored
                    # gain_mode='smooth' is what makes the reconstruction differ.
                    _pipelines_ml_step_dtw_smooth.setdefault(_s, {}).setdefault(_a, {})[_o] = {
                        'reference_curve': _r['ml_step_dtw_smooth']['reference_curve'],
                        'predict_fn':      (lambda ep: lambda rv, act, attrs, exog=None: predict_raw_curve_step_dtw(rv, act, attrs, pipeline=ep, exog_values=exog or {}))(_r['ml_step_dtw_smooth']),
                        'full_pipeline':   _r['ml_step_dtw_smooth'],
                    }

        # NOTE: the old inline "mean baseline" block (flat constant = training
        # mean at every timestep, approach='mean_baseline') has been retired.
        # 'baseline' is now a properly registered approach (median training
        # LEVEL -- see build_and_train_pipeline_median), trained via the same
        # worker/dispatch path as every other approach instead of a bespoke
        # inline computation. It is likewise a flat line; what differs from the
        # retired block is the statistic (median, not mean), the pooling, and
        # the fact that it goes through the registered path.

        # ── "Baseline": ONE median level per SENSOR ──────────────────────────
        # The naive floor: pool EVERY value of EVERY training curve of a sensor
        # (across all its activities/objects) into one vector, take the median,
        # and predict that one number as a horizontal line for every (activity,
        # object) leaf of the sensor. This is the coarser sibling of
        # _pipelines_median_activity_sensor (from the worker), which takes a
        # separate level per (sensor, activity, object). Same predictor
        # (predict_raw_curve_median emits the stored level at the target
        # length); the two differ only in how much they condition the median.
        # Cheap — no model, no DTW, no shape.
        # Gated on 'baseline' in APPROACHES (like every other approach); it can't
        # run in the per-(sensor,activity,object) worker because the per-sensor
        # pool spans activities/objects a single combo worker never sees together.
        if 'baseline' in APPROACHES:
            _baseline_t0 = _time.perf_counter()
            _combos_by_sensor = {}
            for _cs, _ca, _co in _combos:
                _combos_by_sensor.setdefault(_cs, []).append((_ca, _co))
            for _cs, _leaves in _combos_by_sensor.items():
                _acts_s = sorted({_la for _la, _lo in _leaves})
                _objs_s = sorted({_lo for _la, _lo in _leaves})
                _curves_s, _ = split_curves(_df_train_exp, variable=_cs,
                                            activities=_acts_s, objects=_objs_s,
                                            test_size=0.0, verbose=0)
                if len(_curves_s) < 5:
                    continue
                _ep_s = build_and_train_pipeline_median(_curves_s, variable=_cs, verbose=0)
                _ep_s['approach'] = 'baseline'   # per-sensor median = the Baseline
                for _la, _lo in _leaves:
                    _pipelines_baseline.setdefault(_cs, {}).setdefault(_la, {})[_lo] = {
                        'reference_curve': _ep_s['reference_curve'],
                        # Capture the pipeline in an OUTER lambda, never as a default
                        # argument (`ep=_ep_s`): a default arg leaves a 4th positional
                        # slot open, and predict_curve_for_instance probes calling
                        # conventions positionally — its (rv, act, attrs, exog) attempt
                        # would bind exog (None here, this approach has no exog_cols)
                        # into `ep` and predict_raw_curve_median would then call .get
                        # on None. That AttributeError is not the TypeError the probe
                        # loop falls through on, so it killed the whole eval. Keep this
                        # signature identical to ENERGY_PREDICT_REBUILDERS['baseline'].
                        'predict_fn':      (lambda ep: lambda rv, act, attrs: predict_raw_curve_median(rv, act, attrs, pipeline=ep))(_ep_s),
                        'full_pipeline':   _ep_s,
                    }
            # Single-threaded, unlike the two pools above — directly comparable
            # to their wall clock ('curve_training_pool'), not to the summed
            # per-approach worker seconds.
            _record_runtime('curve_training', _time.perf_counter() - _baseline_t0,
                            process=_proc, detail='inline:baseline', split='TRAIN',
                            n_items=len(_combos_by_sensor))

        # ── Median-floor report (see sim_extractor.CURVE_MEDIAN_FLOOR) ────────
        # Every learned leaf is kept only if it beat its own median curve on
        # held-out instances; the ones that lost now PREDICT that median. Report
        # how many, because it is the honest read on where the features actually
        # carry signal — a high share is a finding about the data, not a bug, and
        # without this line the fallback is invisible in the results tables.
        _floor_counts = {}
        for _lbl, _pipes in (('DTW + ML',                _pipelines_ml_dtw),
                             ('DTW + ML + Ext. Factors', _pipelines_ml_external),
                             ('ML only (no DTW)',        _pipelines_ml_only),
                             ('Step DTW + ML + Ext.',    _pipelines_ml_step_dtw),
                             ('Step DTW smooth + ML + Ext.', _pipelines_ml_step_dtw_smooth),
                             ('DTW + Seq2Seq',           _pipelines_seq2seq),
                             ('Seq2Seq only (no DTW)',   _pipelines_seq2seq_only),
                             ('Seq2Seq IOM (DTW-selected)', _pipelines_seq2seq_iom),
                             ('DTW + Seq2Seq + Ext. Factors', _pipelines_seq2seq_external)):
            _tot = _fell = 0
            for _sd in _pipes.values():
                for _ad in _sd.values():
                    for _entry in _ad.values():
                        _tot += 1
                        _fell += bool(_entry.get('full_pipeline', {}).get('median_floor_active'))
            if _tot:
                _floor_counts[_lbl] = (_fell, _tot)
        if _floor_counts:
            print(f"  [{_proc}] median floor — leaves that lost to their own median curve "
                  f"and now predict it:")
            for _lbl, (_fell, _tot) in _floor_counts.items():
                print(f"      {_lbl:<32} {_fell:>4}/{_tot:<4} ({100.0 * _fell / _tot:5.1f}%)")

        all_energy_pipelines[_proc]                   = _pipelines_baseline
        all_energy_pipelines_median_activity_sensor[_proc] = _pipelines_median_activity_sensor
        all_energy_pipelines_ml_dtw[_proc]      = _pipelines_ml_dtw
        all_energy_pipelines_ml_external[_proc]   = _pipelines_ml_external

        all_energy_pipelines_seq2seq[_proc]               = _pipelines_seq2seq
        all_energy_pipelines_seq2seq_only[_proc]          = _pipelines_seq2seq_only
        all_energy_pipelines_seq2seq_external[_proc] = _pipelines_seq2seq_external
        all_energy_pipelines_seq2seq_iom[_proc]           = _pipelines_seq2seq_iom
        all_energy_pipelines_ml_only[_proc]                 = _pipelines_ml_only
        all_energy_pipelines_ml_step_dtw[_proc]             = _pipelines_ml_step_dtw
        all_energy_pipelines_ml_step_dtw_smooth[_proc]      = _pipelines_ml_step_dtw_smooth
        all_energy_pipelines_ml_external_wcounts[_proc] = _pipelines_ml_external_wcounts
        all_energy_pipelines_ml_external_wmetric[_proc] = _pipelines_ml_external_wmetric

        # ── Persist this process's trained pipelines ──────────────────────────
        # One joblib per (process, approach) under <run>/trained_models/, saved
        # as soon as the process finishes training so a crash later in the run
        # loses nothing. predict_fn closures are stripped at save and rebuilt at
        # load — see save_trained_pipelines / rebuild_energy_predict_fns in
        # utils/sim_extractor.py for the round trip.
        if SAVE_TRAINED_MODELS:
            _tm_t0 = _time.perf_counter()
            _tm_saved = []
            for _tm_approach, _tm_pipes in [
                ('baseline',                _pipelines_baseline),
                ('median_activity_sensor',  _pipelines_median_activity_sensor),
                ('ml_dtw',                  _pipelines_ml_dtw),
                ('ml_external',             _pipelines_ml_external),
                ('ml_external_wcounts',     _pipelines_ml_external_wcounts),
                ('ml_external_wmetric',     _pipelines_ml_external_wmetric),
                ('ml_only',                 _pipelines_ml_only),
                ('ml_step_dtw',             _pipelines_ml_step_dtw),
                ('ml_step_dtw_smooth',      _pipelines_ml_step_dtw_smooth),
                ('seq2seq',                 _pipelines_seq2seq),
                ('seq2seq_only',            _pipelines_seq2seq_only),
                ('seq2seq_external',        _pipelines_seq2seq_external),
                ('seq2seq_iom',             _pipelines_seq2seq_iom),
            ]:
                if not _tm_pipes:
                    continue
                _tm_path = os.path.join(_run_dir, 'trained_models', _proc,
                                        f'{_tm_approach}.joblib')
                if save_trained_pipelines(_tm_pipes, _tm_path,
                                          label=f'{_proc}/{_tm_approach}'):
                    _tm_saved.append(_tm_approach)
            if _tm_saved:
                print(f"  [{_proc}] trained models saved "
                      f"({_time.perf_counter() - _tm_t0:.1f}s): {', '.join(_tm_saved)}")

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
    import importlib, utils.sim_extractor as _se
    importlib.reload(_se)
    from utils.sim_extractor import (
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
                # Which approaches need ef_* attached to the evaluation curves,
                # and which splitter mirrors how they were TRAINED. Getting this
                # wrong does not raise: the curves simply arrive without
                # 'exog_values', the ef_* features go in as NaN, and the approach
                # is silently scored without the external factors it was fitted
                # with. The exemplar family was in exactly that state before —
                # trained on ef_* through split_curves(exog_columns=ef_cols), but
                # evaluated without them.
                #   _exog_prev_approaches — trained on split_curves_with_prev_activity
                #   _exog_plain_approaches — trained on plain split_curves + ef_*
                _exog_prev_approaches  = ('ml_external', 'seq2seq_external',
                                          'ml_step_dtw', 'ml_step_dtw_smooth')
                _exog_plain_approaches = ()   # (was the exemplar family — retired)
                _exog_approaches = _exog_prev_approaches + _exog_plain_approaches
                _exog_cols_eval = _fp.get('exog_cols', []) if _approach_eval in _exog_approaches else None
                if _approach_eval in _exog_prev_approaches:
                    # Must mirror training: previous-activity NAME + ef_* only.
                    # include_prev_energy stays False so nothing here reads the
                    # test set's real meter values for the preceding activity.
                    _curves, _ = split_curves_with_prev_activity(
                        _df, _sensor, _leaf_acts, _leaf_objs,
                        test_size=0.0, verbose=0,
                        exog_columns=_exog_cols_eval,
                        include_prev_energy=False,
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
                    # Carry EVERY column evaluate_pipeline_on_test produced into
                    # the export, not just the five pointwise metrics. That
                    # includes 'Instance' (so a row can be traced back to one
                    # activity instance and joined against the event log), the
                    # shape statistics (std/acf1/zeros/max of both the real and
                    # the predicted curve — see _curve_shape_stats), and the
                    # full y_true/y_pred arrays when PIPELINE_SAVE_CURVE_VALUES
                    # is on. Listing the columns by hand is what made the old
                    # export a dead end: the values were gone once the run
                    # finished, so any metric not thought of in advance meant
                    # re-running the whole pipeline.
                    _passthrough = [c for c in _metrics_df.columns
                                    if c not in ('activity', 'n_points', 'MAE',
                                                 'RMSE', 'WAPE (%)', 'sMAE',
                                                 'sRMSE', 'instance_id')]
                    for _, _r in _metrics_df.iterrows():
                        _rec = {
                            'Approach': approach_label,
                            'Process':  _proc,
                            'Sensor':   _sensor,
                            'Split':    split_label,
                            'Activity': _r['activity'],
                            'Instance': _r.get('instance_id'),
                            'N':        _r['n_points'],
                            'MAE':      _r['MAE'],
                            'RMSE':     _r['RMSE'],
                            'WAPE':     _r['WAPE (%)'],
                            'sMAE':     _r.get('sMAE'),
                            'sRMSE':    _r.get('sRMSE'),
                        }
                        for _c in _passthrough:
                            _rec[_c] = _r.get(_c)
                        records.append(_rec)
                except Exception as _eval_e:
                    print(f"  [ERROR] {approach_label} | {_proc} | {_sensor} | act={_leaf_acts}: {_eval_e}")
    return records


def _run_curve_eval_autoregressive_prev_act(pipelines_dict, approach_label, split_label,
                                            df_lookup, activities, objects, save_dir=None):
    """
    Autoregressive evaluator for the ml_external approach.

    ONLY meaningful when the pipelines were trained with
    split_curves_with_prev_activity(include_prev_energy=True). With the default
    (name-only prev-activity context, which is what the reported runs use) the
    model has no lagged-energy feature to chain, so this evaluator returns the
    same numbers as _run_curve_eval. Kept for the explicit lagged-energy
    experiment only; RUN_AUTOREGRESSIVE_EVAL is off for paper runs.

    For each test case in chronological order:
      • Activity 1 of the case has no real predecessor → uses the pipeline's
        per-activity training-median defaults (first_of_case_defaults).
      • Activity k (k≥2) uses prev_act_* features computed from the PREDICTED
        curve of activity k-1, never from real test values.
      • If an intermediate activity has no trained pipeline (or its curve is
        too short), the next activity is treated as first-of-case (prev reset).
    """
    import importlib, utils.sim_extractor as _se
    importlib.reload(_se)
    from utils.sim_extractor import _dispatch_predict
    from sklearn.metrics import mean_squared_error, mean_absolute_error

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

        # sentinel: mirror split_curves behaviour for processes with no object_log
        if 'object_log' not in _df.columns:
            _df = _df.copy()
            _df['object_log'] = '_all_'
            _proc_objs = ['_all_']
        elif not _proc_objs or _df['object_log'].isna().all():
            _df = _df.copy()
            _df['object_log'] = _df['object_log'].fillna('_all_')
            _proc_objs = ['_all_']

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


# ── Complete-curve eval (deferred) ──────────────────────────────────────────
# Every (process, mode) simulated log was collected into
# _energy_distribution_pending during the per-process loop above, but
# all_energy_pipelines only exists from here onward (populated by the
# CURVE-ONLY TRAINING section just above). Compute + save now that the
# curve pipelines actually exist.
if 'all_energy_pipelines' in dir() and all_energy_pipelines and _energy_distribution_pending:
    # Complete-curve eval is pure inference (all pipelines are already trained
    # above), so it runs against every approach that actually trained
    # pipelines for this run (i.e. every entry in APPROACHES with a non-empty
    # all_energy_pipelines_<approach> dict).
    #
    # Only the prev_activity-conditioned seq2seq* approaches are excluded, not
    # the whole seq2seq family. Root-caused 2026-07-21: predict_curve_for_instance
    # calls predict_fn statelessly, one activity instance at a time. The
    # experiment_802 "every seq2seq* attempt failing" observation that used to
    # justify excluding ALL of them was actually a missing df_seq.fillna(0)
    # step in the seq2seq* predict functions (present in the sklearn/DTW
    # X_ref.fillna(0) path, absent here) -- any trained feature not present in
    # production object_attributes (e.g. hour_of_day/day_of_week, injected
    # only at TRAIN time by split_curves*) came through as NaN and silently
    # propagated to an all-NaN prediction. That's now fixed (see the
    # df_seq.fillna(0) calls in predict_raw_curve_seq2seq*), and verified: a
    # plain seq2seq/seq2seq_only pipeline predicts finite values
    # on real production-shaped attributes.
    #
    # The *_external approaches used to raise a second concern: their
    # prev_act_* features were lagged energy of the previous activity, which a
    # simulation cannot know without chaining predictions
    # (_run_curve_eval_autoregressive_prev_act). That is moot now --
    # split_curves_with_prev_activity defaults to include_prev_energy=False, so
    # the only prev-activity feature is the categorical prev_act_name, an event-
    # log fact the simulator already has. Nothing to chain, nothing to exclude.
    # (The guard below is a leftover no-op: no APPROACHES key contains
    # 'prev_activity'.)
    _complete_curve_approaches_available = []
    for _appr_candidate in APPROACHES:
        if _appr_candidate.startswith('seq2seq') and 'prev_activity' in _appr_candidate:
            continue
        _dict_name_c = _ENERGY_APPROACH_DICT_NAMES.get(_appr_candidate, f'all_energy_pipelines_{_appr_candidate}')
        if _dict_name_c in dir() and globals().get(_dict_name_c):
            _complete_curve_approaches_available.append(_appr_candidate)
    if not _complete_curve_approaches_available:
        _complete_curve_approaches_available = ['baseline']

    # Opt-in restriction — see COMPLETE_CURVE_APPROACHES. Intersected with what
    # actually trained, so a typo or a not-trained approach cannot silently
    # produce an empty eval; it falls back to everything with a warning.
    if COMPLETE_CURVE_APPROACHES is not None:
        _cc_missing = [a for a in COMPLETE_CURVE_APPROACHES
                       if a not in _complete_curve_approaches_available]
        _cc_kept = [a for a in COMPLETE_CURVE_APPROACHES
                    if a in _complete_curve_approaches_available]
        if _cc_missing:
            print(f"  ⚠️ PIPELINE_COMPLETE_CURVE_APPROACHES: {_cc_missing} not among the "
                  f"trained approaches {_complete_curve_approaches_available} — ignored.")
        if _cc_kept:
            _complete_curve_approaches_available = _cc_kept
        else:
            print(f"  ⚠️ PIPELINE_COMPLETE_CURVE_APPROACHES left nothing evaluable — "
                  f"keeping all trained approaches instead.")

    print("\n" + "="*50)
    print(f"COMPLETE-CURVE EVAL ({len(_energy_distribution_pending)} process/mode combos "
          f"x {len(_complete_curve_approaches_available)} approach(es): {_complete_curve_approaches_available})")
    print("="*50)
    for _proc_p, _mode_p, _sim_p, _exp_p, _core_p in _energy_distribution_pending:
        for _approach_p in _complete_curve_approaches_available:
            with _timed('complete_curve_eval', process=_proc_p,
                        detail=f'{_mode_p}:{_approach_p}', split='TEST'):
                _save_complete_curve_eval_metrics(
                    _proc_p, _mode_p, _sim_p, _exp_p, _complete_curve_eval_dir,
                    approach=_approach_p,
                )

    # ── Schedule Profile Evaluation (opt-in — off by default) ─────────────
    # Runs once per process (not per mode): trains a stochastic generator
    # (+ bootstrap resampler) on train cases, evaluates it against
    # real test cases, and merges in the already-computed per-case
    # 'ml_external' numbers for this process's best-fidelity mode as
    # the "Best, mine" comparator — no retraining/resimulating needed for
    # that column, it's already sitting on disk from the loop just above.
    #
    # Which mode is "best" is decided on TRAIN. The comparator it feeds is then
    # scored against real TEST cases, so picking it by test_overall_error would
    # be choosing the comparator on the very data the comparison reports.
    if RUN_SCHEDULE_PROFILE_EVAL:
        _best_mode_by_process = {}
        _best_error_by_process = {}
        _best_sim_by_process = {}
        _best_core_by_process = {}
        for _proc_p, _mode_p, _sim_p, _exp_p, _core_p in _energy_distribution_pending:
            _err = _core_p.get('train_overall_error')
            if _err is None:
                continue
            if _proc_p not in _best_error_by_process or _err < _best_error_by_process[_proc_p]:
                _best_error_by_process[_proc_p] = _err
                _best_mode_by_process[_proc_p] = str(_mode_p).replace(' ', '_').replace('/', '_')
                _best_sim_by_process[_proc_p] = _sim_p
                _best_core_by_process[_proc_p] = _core_p

        print("\n" + "="*50)
        print(f"SCHEDULE PROFILE EVALUATION ({len({p for p, *_ in _energy_distribution_pending})} processes)")
        print("="*50)
        for _proc_p in sorted({p for p, *_ in _energy_distribution_pending}):
            _train_exp_p = train_datasets.get(_proc_p, {}).get('expanded')
            _test_exp_p  = test_datasets.get(_proc_p, {}).get('expanded')
            if _train_exp_p is None or _train_exp_p.empty or _test_exp_p is None or _test_exp_p.empty:
                continue
            _sensors_p = _detect_sensors_for_energy_distribution(_proc_p, _test_exp_p)
            _ef_cols_p = [c for c in _train_exp_p.columns
                         if c.startswith('ef_') and _train_exp_p[c].dtype in ('float64', 'float32', 'int64', 'int32')]
            # "Best, budget" is pinned to one named budget mode for EVERY
            # process (unlike "Best, mine", which is the per-process
            # overall_error winner) so the budget method gets a clean,
            # like-for-like row instead of being silently blended in wherever
            # it happens to win. Falls back through the budget family in
            # order of preference; None -> series simply absent.
            _budget_mode_pin = next(
                (m for m in ('petri_net_budget_ml_plus_per_act',
                             'petri_net_budget_ml_plus_global',
                             'petri_net_budget')
                 if m in MODES_TO_COMPARE),
                None,
            )
            # "Best, mine" reads the complete-curve outputs from disk, so its
            # source approach must be one the (possibly restricted) loop above
            # actually wrote: ml_external when available — the historical
            # column — otherwise the first evaluated approach.
            _sched_comparator = ('ml_external'
                                 if 'ml_external' in _complete_curve_approaches_available
                                 else _complete_curve_approaches_available[0])
            if _sched_comparator != 'ml_external':
                print(f"  ℹ️ Schedule-profile 'Best, mine' comparator: "
                      f"'{_sched_comparator}' (ml_external not in the complete-curve eval).")
            with _timed('schedule_profile_eval', process=_proc_p,
                        detail=f'comparator:{_sched_comparator}', split='TEST',
                        n_items=len(_sensors_p)):
                _save_schedule_profile_eval(
                    _proc_p, _train_exp_p, _test_exp_p, _sensors_p, _ef_cols_p,
                    _schedule_profile_eval_dir,
                    best_mode_safe=_best_mode_by_process.get(_proc_p),
                    complete_curve_dir=_complete_curve_eval_dir,
                    predicted_logs_dir=_predicted_logs_dir,
                    budget_mode_safe=_budget_mode_pin,
                    comparator_approach=_sched_comparator,
                )


if RUN_CURVE_ONLY_EVALUATION and 'all_energy_pipelines' in dir() and all_energy_pipelines:
    import importlib, utils.sim_extractor as _se
    importlib.reload(_se)
    from utils.sim_extractor import evaluate_pipeline_on_test, split_curves, split_curves_with_prev_activity

    display(Markdown("---"))
    display(Markdown("# Curve-Only Evaluation — Baseline vs Approach 2 (B-spline) vs Approach 3 (DTW-phase)"))

    _proc_cfg_lookup = {
        _p: process_datasets_to_model_sensors.get(_p, {})
        for _p in process_datasets_to_model.keys()
    } if 'process_datasets_to_model_sensors' in dir() else {}
    _activities_map = {p: c.get('activities_to_model', []) for p, c in _proc_cfg_lookup.items()}
    _objects_map    = {p: c.get('objects_to_model',    []) for p, c in _proc_cfg_lookup.items()}

    _all_records = []

    # Crash-safe checkpointing: each approach's rows land on disk the moment
    # its eval finishes, as one shard per (split, approach), so a failure in a
    # later approach or stage cannot lose the finished ones (experiment_986
    # lost 7 hours to exactly that — a hang after training left NOTHING on
    # disk). The final curve_eval_results.parquet is still written as before
    # and supersedes these; the shards are raw per-curve records (y_true still
    # on every row — the dedup into real_test_curves.parquet happens only in
    # the final export) meant for salvage, not for the notebooks.
    _curve_ckpt_dir = os.path.join(_run_dir, 'curve_eval_checkpoints')
    def _ckpt_curve_eval(_recs_c, _split_c, _label_c):
        if not _recs_c:
            return
        try:
            os.makedirs(_curve_ckpt_dir, exist_ok=True)
            _safe_c = ''.join(ch if ch.isalnum() else '_' for ch in _label_c).strip('_')
            _fp_c = os.path.join(_curve_ckpt_dir, f'{_split_c}_{_safe_c}.parquet')
            pd.DataFrame(_recs_c).to_parquet(_fp_c, index=False)
            print(f"  💾 Checkpoint: {len(_recs_c)} rows → curve_eval_checkpoints/"
                  f"{os.path.basename(_fp_c)}")
        except Exception as _ck_exc:
            # A checkpoint must never take down the run it exists to protect.
            print(f"  [WARN] curve-eval checkpoint failed for {_split_c}/{_label_c}: {_ck_exc}")

    # ── Evaluate both approaches on TRAIN and TEST ───────────────────────────
    for _split_label, _df_src in [('TRAIN', train_datasets),
                                   ('TEST',  test_datasets if TEMPORAL_SPLIT else train_datasets)]:

        display(Markdown(f"## Split: {_split_label}"))

        for _approach_label, _pipelines in [
            ('Baseline',                          all_energy_pipelines),
            ('Median per Activity & Sensor',      all_energy_pipelines_median_activity_sensor),
            ('ML DTW',                    all_energy_pipelines_ml_dtw),
            ('ML + Ext. Factors',     all_energy_pipelines_ml_external),
            ('ML only (no DTW)',                all_energy_pipelines_ml_only),
            ('Step DTW + ML + Ext.',            all_energy_pipelines_ml_step_dtw),
            ('Step DTW smooth + ML + Ext.',     all_energy_pipelines_ml_step_dtw_smooth),
            ('DTW + ML + Ext. (count-weighted)', all_energy_pipelines_ml_external_wcounts),
            ('DTW + ML + Ext. (metric-weighted)', all_energy_pipelines_ml_external_wmetric),

            ('DTW + Seq2Seq',                     all_energy_pipelines_seq2seq),
            ('Seq2Seq only (no DTW)',              all_energy_pipelines_seq2seq_only),
            ('Seq2Seq IOM (DTW-selected)',         all_energy_pipelines_seq2seq_iom),
            ('DTW + Seq2Seq + Ext. Factors', all_energy_pipelines_seq2seq_external),
        ]:
            if not _pipelines:
                continue
            display(Markdown(f"### {_approach_label}"))
            with _timed('curve_only_eval', detail=_approach_label, split=_split_label):
                _recs = _run_curve_eval(
                    _pipelines, _approach_label, _split_label,
                    _df_src, _activities_map, _objects_map,
                    save_dir=None,
                )
            _all_records.extend(_recs)
            _ckpt_curve_eval(_recs, _split_label, _approach_label)

        # ── Autoregressive rollout for ml_external ───────────────────
        if RUN_AUTOREGRESSIVE_EVAL and all_energy_pipelines_ml_external:
            display(Markdown("### DTW + Ext. Factors + Prev Act (autoreg)"))
            with _timed('curve_only_eval', detail='ml_external (autoreg)', split=_split_label):
                _ar_recs = _run_curve_eval_autoregressive_prev_act(
                    all_energy_pipelines_ml_external,
                    'DTW + Ext. Factors + Prev Act (autoreg)',
                    _split_label,
                    _df_src, _activities_map, _objects_map,
                    save_dir=None,
                )
            _all_records.extend(_ar_recs)
            _ckpt_curve_eval(_ar_recs, _split_label, 'ml_external_autoreg')

        # ── Autoregressive rollout for seq2seq_external ─────────────────
        if RUN_AUTOREGRESSIVE_EVAL and all_energy_pipelines_seq2seq_external:
            display(Markdown("### DTW + Seq2Seq + Ext. Factors + Prev Act (autoreg)"))
            with _timed('curve_only_eval', detail='seq2seq_external (autoreg)', split=_split_label):
                _ar_s2s_recs = _run_curve_eval_autoregressive_prev_act(
                    all_energy_pipelines_seq2seq_external,
                    'DTW + Seq2Seq + Ext. Factors + Prev Act (autoreg)',
                    _split_label,
                    _df_src, _activities_map, _objects_map,
                    save_dir=None,
                )
            _all_records.extend(_ar_s2s_recs)
            _ckpt_curve_eval(_ar_s2s_recs, _split_label, 'seq2seq_external_autoreg')

    # ── Side-by-side comparison table ───────────────────────────────────────
    if not _all_records:
        print("[ERROR] _all_records is empty — no curves were evaluated. Check WARN messages above.")
    if _all_records:
        _all_df = pd.DataFrame(_all_records)

        # Level the population across approaches before any comparison table
        # (see CURVE_EVAL_MIN_POINTS). _all_df itself stays complete — it is
        # what gets exported to curve_eval_results.parquet.
        _cmp_df = _all_df[_all_df['N'] >= CURVE_EVAL_MIN_POINTS]
        _dropped = len(_all_df) - len(_cmp_df)
        if _dropped:
            _drop_by_appr = (
                _all_df[_all_df['N'] < CURVE_EVAL_MIN_POINTS]
                .groupby('Approach').size().to_dict()
            )
            print(f"  ℹ️ Curve comparison levelled to curves with >= {CURVE_EVAL_MIN_POINTS} "
                  f"samples: {_dropped} of {len(_all_df)} curve rows excluded "
                  f"({_drop_by_appr}). Full set kept in curve_eval_results.parquet.")
        _cmp_counts = _cmp_df.groupby(['Approach', 'Split']).size().unstack('Split', fill_value=0)
        display(Markdown("### Curves scored per approach (after levelling)"))
        display(_cmp_counts)

        # ── Top-level table: Approach × Split (all sensors/processes aggregated) ─
        display(Markdown("---"))
        display(Markdown(
            f"## Approach Comparison — Train & Test (median over ALL sensors, processes, "
            f"curves with >= {CURVE_EVAL_MIN_POINTS} samples)"))
        _appr_summary = (
            _cmp_df
            .groupby(['Approach', 'Split'])[['MAE', 'RMSE', 'WAPE', 'sMAE', 'sRMSE']]
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
        # round(6), not round(4): this frame is persisted as
        # summary_train_test.parquet and read back by the results notebooks.
        # Sensors with small absolute magnitudes (process_1 bottling /
        # individual_packaging draw single-digit kW) have median MAEs around
        # 1e-4, which round(4) flattened to 0.0 in the stored file -- so the
        # notebooks could not tell "no error at all" from "too small to show".
        _summary = (
            _cmp_df
            .groupby(['Process', 'Sensor', 'Approach', 'Split'])[['MAE', 'RMSE', 'WAPE', 'sMAE', 'sRMSE']]
            .median()
            .round(6)
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

        _test_df = _cmp_df[_cmp_df['Split'] == 'TEST']
        if not _test_df.empty:
            _compare = (
                _test_df
                .groupby(['Approach', 'Process', 'Sensor'])[['MAE', 'RMSE', 'WAPE', 'sMAE', 'sRMSE']]
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
                .groupby(['Approach', 'Process', 'Sensor', 'Activity'])[['MAE', 'RMSE', 'WAPE', 'sMAE', 'sRMSE']]
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
            except Exception as _e:
                print(f"[WARN] Plotly WAPE distribution failed: {_e}")

            # ── sMAE heatmap — one subplot per approach ───────────────────────
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
                        values='sMAE', aggfunc='median'
                    )
                    sns.heatmap(
                        _ph, annot=True, fmt='.3f', cmap='RdYlGn_r',
                        linewidths=0.5, ax=ax_h,
                        cbar_kws={'label': 'sMAE'}
                    )
                    ax_h.set_title(f'sMAE — {_appr}', fontsize=11, fontweight='bold')
                    ax_h.set_xticklabels(ax_h.get_xticklabels(), rotation=30, ha='right', fontsize=8)

                plt.suptitle('Curve sMAE per Activity — TEST set', fontsize=13, fontweight='bold', y=1.02)
                plt.tight_layout()
                plt.show()
            except Exception as _e:
                print(f"[WARN] sMAE heatmap failed: {_e}")

            # ── sMAE heatmap — one file per process → energy_results ─────────
            try:
                for _proc_hm in sorted(_test_df['Process'].unique()):
                    _pdf = _test_df[_test_df['Process'] == _proc_hm]
                    _appr_hm = _pdf['Approach'].unique()
                    _fig_p, _axes_p = plt.subplots(
                        1, len(_appr_hm),
                        figsize=(max(8, _pdf['Activity'].nunique() * 1.4) * len(_appr_hm),
                                 max(3, _pdf['Sensor'].nunique() * 1.2))
                    )
                    _axes_p = np.array(_axes_p).reshape(-1)
                    for _ax_p, _appr in zip(_axes_p, _appr_hm):
                        _sub_p = _pdf[_pdf['Approach'] == _appr]
                        _ph_p = _sub_p.pivot_table(
                            index='Sensor', columns='Activity',
                            values='sMAE', aggfunc='median'
                        )
                        sns.heatmap(
                            _ph_p, annot=True, fmt='.3f', cmap='RdYlGn_r',
                            linewidths=0.5, ax=_ax_p,
                            cbar_kws={'label': 'sMAE'}
                        )
                        _ax_p.set_title(f'sMAE — {_appr}', fontsize=11, fontweight='bold')
                        _ax_p.set_xticklabels(_ax_p.get_xticklabels(), rotation=30, ha='right', fontsize=8)
                    _fig_p.suptitle(f'Curve sMAE — {_proc_hm} — TEST set', fontsize=13, fontweight='bold', y=1.02)
                    plt.tight_layout()
                    if EXPORT_RESULTS and '_energy_results_dir' in dir():
                        _ep = os.path.join(_energy_results_dir, f'smae_heatmap_{_proc_hm}.png')
                        _fig_p.savefig(_ep, dpi=150, bbox_inches='tight')
                    plt.show()
                    plt.close(_fig_p)
            except Exception as _e:
                print(f"[WARN] Per-process sMAE heatmaps failed: {_e}")

            # ── 5 BEST / 5 WORST curves per approach (TEST WAPE) ────────────────
            try:
                display(Markdown("---"))
                display(Markdown("## 5 Best & 5 Worst Curve Fits per Approach — TEST set"))

                _APPROACH_PIPELINES = {
                    'Baseline':                           all_energy_pipelines
                        if 'all_energy_pipelines' in dir() else {},
                    'Median per Activity & Sensor':       all_energy_pipelines_median_activity_sensor
                        if 'all_energy_pipelines_median_activity_sensor' in dir() else {},
                    'ML DTW':                     all_energy_pipelines_ml_dtw
                        if 'all_energy_pipelines_ml_dtw' in dir() else {},
                    'ML + Ext. Factors':      all_energy_pipelines_ml_external
                        if 'all_energy_pipelines_ml_external' in dir() else {},
                    'ML only (no DTW)':                 all_energy_pipelines_ml_only
                        if 'all_energy_pipelines_ml_only' in dir() else {},
                    'Step DTW + ML + Ext.':             all_energy_pipelines_ml_step_dtw
                        if 'all_energy_pipelines_ml_step_dtw' in dir() else {},
                    'Step DTW smooth + ML + Ext.':      all_energy_pipelines_ml_step_dtw_smooth
                        if 'all_energy_pipelines_ml_step_dtw_smooth' in dir() else {},
                    'DTW + Seq2Seq':                      all_energy_pipelines_seq2seq
                        if 'all_energy_pipelines_seq2seq' in dir() else {},
                    'Seq2Seq only (no DTW)':              all_energy_pipelines_seq2seq_only
                        if 'all_energy_pipelines_seq2seq_only' in dir() else {},
                    'Seq2Seq IOM (DTW-selected)':         all_energy_pipelines_seq2seq_iom
                        if 'all_energy_pipelines_seq2seq_iom' in dir() else {},
                    'DTW + Seq2Seq + Ext. Factors': all_energy_pipelines_seq2seq_external
                        if 'all_energy_pipelines_seq2seq_external' in dir() else {},
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
                                if _bw_approach in ('ml_external',) else None
                            if _bw_approach == 'ml_external':
                                _tc_bw, _ = split_curves_with_prev_activity(
                                    _df_test_bw, _sensor_bw, [_act_bw], [_obj_bw],
                                    test_size=0.0, verbose=0,
                                    exog_columns=_bw_exog,
                                    include_prev_energy=False,
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
                                # mean_absolute_error rejects NaN outright; nanmean matches its
                                # result when there's nothing to skip, and degrades gracefully
                                # (like _wape_c already does) instead of crashing this plot.
                                _mae_c   = float(np.nanmean(np.abs(np.asarray(_rv_bw) - np.asarray(_yp_bw))))
                                _ax.plot(_rv_bw, label='Actual', color='steelblue', linewidth=2)
                                _ax.plot(_yp_bw, label='Predicted', color='tomato',
                                         linewidth=2, linestyle='--')
                                _ax.set_title(
                                    f"{_act_bw[:28]} | {_sensor_bw[:20]}\n"
                                    f"WAPE={_wape_c:.1f}%  MAE={_mae_c:.4f}  "
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
            plt.show()
else:
    report("  No combined simulation results available for curve plotting.")

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
    # The runtime table is written even here: it costs nothing, and a timing
    # run is exactly the kind of run somebody does with the export off.
    _record_runtime('pipeline_total', _time.perf_counter() - _pipeline_start,
                    detail='wall clock, whole run')
    if _RUNTIME_ROWS:
        _rt_df = pd.DataFrame(_RUNTIME_ROWS)
        _rt_df['minutes'] = _rt_df['seconds'] / 60.0
        _rt_df['run_name'] = _run_name
        _rt_df['run_timestamp'] = _run_ts
        _rt_df.to_csv(os.path.join(_run_dir, 'runtime_profile.csv'), index=False)
        print(f"Saved runtime  → {os.path.join(_run_dir, 'runtime_profile.csv')} "
              f"({len(_rt_df)} timed stages)")
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

        # ── Real curves are written ONCE, not once per approach ──────────────
        # Every approach is scored on the same test curves, so carrying y_true on
        # every row stores the identical array as many times as there are
        # approaches -- roughly half the file, duplicated tenfold on a full run.
        # It is split into its own table keyed on the curve identity instead;
        # y_pred stays on the per-approach rows, where it genuinely differs.
        # Join on the key below to recover the pairing.
        if 'y_true' in _export_df.columns:
            _rc_key = ['Process', 'Sensor', 'Activity', 'Instance', 'Split']
            _rc_key = [c for c in _rc_key if c in _export_df.columns]
            _real_df = (_export_df[_rc_key + ['N', 'y_true']]
                        .drop_duplicates(subset=_rc_key)
                        .rename(columns={'y_true': 'y_real'}))
            _real_path = os.path.join(_run_dir, 'real_test_curves.parquet')
            _real_df.to_parquet(_real_path, index=False)
            _export_df = _export_df.drop(columns=['y_true'])
            print(f"Saved real curves → {_real_path}  "
                  f"({len(_real_df):,} unique curves, deduplicated from "
                  f"{len(_export_df):,} scored rows)")
            print(f"  join key: {' + '.join(_rc_key)}")

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
            # Levelled like every other cross-approach comparison — the heatmap
            # ranks approaches against each other, so it must not mix populations
            # (see CURVE_EVAL_MIN_POINTS).
            _hm_df = _export_df[_export_df['N'] >= CURVE_EVAL_MIN_POINTS]
            for _split in sorted(_hm_df['Split'].unique()):
                display(Markdown(f"## {_split}"))
                _plot_curve_energy_heatmap(_hm_df, _split, 'median', _energy_results_dir)
                _plot_curve_energy_heatmap(_hm_df, _split, 'mean',   _energy_results_dir)

        _sw = globals().get('_summary_wide')
        if _sw is not None and not _sw.empty:
            _sw_path = os.path.join(_run_dir, 'summary_train_test.parquet')
            _sw.reset_index().to_parquet(_sw_path, index=False)
            print(f"Saved summary  → {_sw_path}")

        _as = globals().get('_appr_summary')
        if _as is None or _as.empty:
            # rebuild from raw records if the display block didn't produce it
            _as_raw = (
                _export_df[_export_df['N'] >= CURVE_EVAL_MIN_POINTS]
                .groupby(['Approach', 'Split'])[['MAE', 'RMSE', 'WAPE', 'sMAE', 'sRMSE']]
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

    # ── Runtime profile ──────────────────────────────────────────────────────
    # Every timed stage of this run (see the RUNTIME PROFILING section):
    # runtime_profile.* is the raw rows, runtime_summary.csv the per-stage
    # totals. Written before the HTML export so the table exists even if
    # nbconvert fails.
    _record_runtime('pipeline_total', _time.perf_counter() - _pipeline_start,
                    detail='wall clock, whole run')
    if _RUNTIME_ROWS:
        _rt_df = pd.DataFrame(_RUNTIME_ROWS)
        _rt_df['minutes'] = _rt_df['seconds'] / 60.0
        _rt_df['run_name'] = _run_name
        _rt_df['run_timestamp'] = _run_ts
        _rt_df.to_parquet(os.path.join(_run_dir, 'runtime_profile.parquet'), index=False)
        _rt_df.to_csv(os.path.join(_run_dir, 'runtime_profile.csv'), index=False)

        # Per-stage totals. 'pipeline_total' is the wall clock of everything,
        # so it is listed separately rather than summed with the stages it
        # contains, and the parallel pools' summed worker seconds
        # ('curve_training') legitimately exceed it.
        _rt_summary = (
            _rt_df[_rt_df['stage'] != 'pipeline_total']
            .groupby('stage')
            .agg(seconds=('seconds', 'sum'), minutes=('minutes', 'sum'),
                 n_rows=('seconds', 'size'))
            .sort_values('seconds', ascending=False)
            .reset_index()
        )
        _rt_total = float(_rt_df.loc[_rt_df['stage'] == 'pipeline_total', 'seconds'].sum())
        _rt_summary['pct_of_wall_clock'] = (
            100.0 * _rt_summary['seconds'] / _rt_total if _rt_total > 0 else float('nan'))
        _rt_summary.to_csv(os.path.join(_run_dir, 'runtime_summary.csv'), index=False)
        print(f"Saved runtime  → {os.path.join(_run_dir, 'runtime_profile.parquet')} "
              f"({len(_rt_df)} timed stages)")
        display(Markdown("## Runtime by stage"))
        display(_rt_summary.round(2))
        report("\nRUNTIME BY STAGE (seconds)")
        report(_rt_summary.round(2).to_string())

        # Per-approach curve training, the "what does each method cost?" view.
        _rt_methods = _rt_df[_rt_df['stage'] == 'curve_training']
        if not _rt_methods.empty:
            _rt_by_method = (
                _rt_methods.assign(
                    method=_rt_methods['detail'].astype(str).str.split(':').str[-1],
                    # object dtype: n_items is None for stages that have no
                    # natural item count, so sum() needs a numeric cast first.
                    n_items=pd.to_numeric(_rt_methods['n_items'], errors='coerce'))
                .groupby('method')
                .agg(seconds=('seconds', 'sum'), leaves=('n_items', 'sum'))
                .sort_values('seconds', ascending=False)
            )
            _rt_by_method['seconds_per_leaf'] = (
                _rt_by_method['seconds'] / _rt_by_method['leaves'].replace(0, np.nan))
            _rt_by_method.reset_index().to_csv(
                os.path.join(_run_dir, 'runtime_by_method.csv'), index=False)
            display(Markdown("## Curve training cost per method "
                             "(summed across pool workers — not wall clock)"))
            display(_rt_by_method.round(2))
            report("\nCURVE TRAINING COST PER METHOD (summed worker seconds)")
            report(_rt_by_method.round(2).to_string())

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

    # ── info.json ─────────────────────────────────────────────────────────────
    import json as _json
    import utils.sim_extractor as _sx
    _info = {
        'run_name': _run_name,
        'run_timestamp': _run_ts,
        'data_experiment': experiment,
        'temporal_resolution': TEMPORAL_RESOLUTION,
        'processes': processes_to_run,
        'approaches': APPROACHES,
        'split_type': SPLIT_TYPE,
        'train_ratio': TRAIN_RATIO,
        # Recorded so a results table can never be misread: both of these change
        # what the numbers mean, not just their value.
        'curve_loss': _sx.CURVE_MODEL_LOSS,
        'dtw_shape_blind_decode': _sx.DTW_DECODE_SHAPE_BLIND,
        'curve_select_by_realism': _sx.CURVE_SELECT_BY_REALISM,
        'curve_selection_metrics': list(_sx.CURVE_SELECTION_METRICS),
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