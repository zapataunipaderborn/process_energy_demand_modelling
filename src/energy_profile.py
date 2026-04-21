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

# --- LOGGING AND REDIRECTION SETUP ---
# Silence pm4py internal logs and TBR replay messages
logging.getLogger("pm4py").setLevel(logging.ERROR)
# Silence other potential noise
logging.getLogger("fsspec").setLevel(logging.ERROR)

LOG_FILE = "pipeline_execution.log"

# Configure logging
logging.basicConfig(
    filename=LOG_FILE,
    filemode='w',  # Overwrite each run
    level=logging.INFO, # <-- MUST BE INFO. DEBUG causes Numba/Matplotlib to flood bytecode logs
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Capture all library warnings (pm4py, pandas, etc.)
logging.captureWarnings(True)

# Save original stdout for notebook reports
_original_stdout = sys.stdout

def report(*args, **kwargs):
    """
    Prints to both the original notebook console and the log file.
    Use this for high-level metrics and summaries.
    """
    # Print to the 'real' console/notebook
    print(*args, file=_original_stdout, flush=True, **kwargs)
    # Log the report content to the file as well
    logging.info("[REPORT] " + " ".join(map(str, args)))

class StreamToLogger:
    """Redirects stdout/stderr to logging."""
    def __init__(self, logger_func):
        self.logger_func = logger_func
        self.buffer = ""
    def write(self, data):
        for line in data.splitlines():
            if line.strip():
                self.logger_func(line.strip())
    def flush(self):
        pass

# Redirect all standard prints and errors to the log file
sys.stdout = StreamToLogger(logging.info)
sys.stderr = StreamToLogger(logging.error)

import random
import itertools
import os
from datetime import datetime, timedelta
RANDOM_SEED = 42
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

from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr


pio.renderers.default='notebook'
pd.options.mode.chained_assignment = None
from IPython.display import display, Markdown

# Force pm4py to hide progress bars
from pm4py.util import constants
constants.SHOW_PROGRESS_BAR = False


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
    'overall_score',
    'basic_metrics_event_count_ratio',
    'duration_metrics_mean_duration_error',
    'duration_metrics_median_duration_error', 
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
        
        # Check if it's lower-is-better
        is_lower = clean_c in base_lower_names or any(m in clean_c for m in ['MAE', 'RMSE', 'WAPE'])
        # (Exception: R2 is higher is better)
        if 'R2' in clean_c: is_lower = False

        if is_lower:
            norm_df[c] = (vmax - df[c]) / rng
        else:
            norm_df[c] = (df[c] - vmin) / rng
    return norm_df


def _plot_results_heatmap(cols, title, metric_type='process', local_df=None):
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
        'train_basic_metrics_event_count_ratio': 'EvtRatio',
        'test_basic_metrics_event_count_ratio':  'EvtRatio',
        'train_duration_metrics_mean_duration_error': 'MeanDurErr',
        'test_duration_metrics_mean_duration_error':  'MeanDurErr',
        'train_duration_metrics_median_duration_error': 'MedDurErr',
        'test_duration_metrics_median_duration_error':  'MedDurErr',
        'train_conformance_metrics_fitness': 'Fitness',
        'test_conformance_metrics_fitness':  'Fitness',
        'train_conformance_metrics_precision': 'Precision',
        'test_conformance_metrics_precision':  'Precision',
        'train_conformance_metrics_generalization': 'Generaliz',
        'test_conformance_metrics_generalization':  'Generaliz',
        'train_conformance_metrics_simplicity': 'Simplicity',
        'test_conformance_metrics_simplicity':  'Simplicity',
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
    annot_df = mode_avg[valid_cols].reindex(mode_avg_norm.index).round(3)
    annot_df.columns = disp_cols

    fig, ax = plt.subplots(figsize=(max(10, len(valid_cols) * 1.2), max(3, len(mode_avg_norm) * 0.8)))
    sns.heatmap(plot_df, annot=annot_df, fmt='', cmap='RdYlGn', vmin=0, vmax=1, linewidths=0.5, ax=ax,
                cbar_kws={'label': 'Normalised score (1 = best)'})
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel('')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=9)
    plt.tight_layout()
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
# ## Process 2

# %%


# Define the path to the current file's location
current_path = Path(__file__).resolve().parent if '__file__' in globals() else Path().resolve()

# Define the folder path
files_folder_silver = current_path.parent / 'data' / 'silver' / 'process_2'

# Load the Parquet file
df = pd.read_parquet(files_folder_silver / "df_combined_legend.parquet")

print(df)

# Identify shared columns: those without 'l01' or 'l02' in the name, excluding 'datetime' and mat_id (but mat_id are shared)
shared_columns = [col for col in df.columns if not any(x in col.lower() for x in ['l01', 'l02']) and col not in ['datetime']]

# Columns for L01: those containing 'l01' (case insensitive), plus 'datetime', mat_id columns, and shared columns
l01_columns = ['datetime'] + [col for col in df.columns if 'l01' in col.lower()] + shared_columns

# Columns for L02: those containing 'l02' (case insensitive), plus 'datetime', mat_id columns, and shared columns
l02_columns = ['datetime'] + [col for col in df.columns if 'l02' in col.lower()] + shared_columns

# Create separate DataFrames
df_l01 = df[l01_columns].copy()
df_l02 = df[l02_columns].copy()

# Rename columns in df_l01 to remove 'l01' related suffixes/prefixes
df_l01.columns = [col.replace('_l01', '').replace('l01_', '').replace('l01', '') for col in df_l01.columns]

# Rename columns in df_l02 to remove 'l02' related suffixes/prefixes
df_l02.columns = [col.replace('_l02', '').replace('l02_', '').replace('l02', '') for col in df_l02.columns]

# Ensure both DataFrames have the same columns for appending (after renaming)
common_columns = list(set(df_l01.columns) & set(df_l02.columns))
df_l01 = df_l01[common_columns]
df_l02 = df_l02[common_columns]

# Append the DataFrames
df_expanded = pd.concat([df_l01.assign(temp_object='l01'), df_l02.assign(temp_object='l02')], ignore_index=True)

# Sort by temp_object and datetime to ensure proper ordering
df_expanded = df_expanded.sort_values(['temp_object', 'datetime']).reset_index(drop=True)

# Identify activity changes: group consecutive rows with the same status_name per object
df_expanded['activity_change'] = (df_expanded['status_name'] != df_expanded['status_name'].shift()).cumsum()

# Group by temp_object and activity_change to create activity intervals
activity_groups = df_expanded.groupby(['temp_object', 'activity_change'])

# Create a new DataFrame for activities
activity_rows = []
for (obj, change), group in activity_groups:
    group = group.sort_values('datetime')
    start_time = group['datetime'].min()
    end_time = group['datetime'].max()
    status = group['status_name'].iloc[0]

    row = group.iloc[0].copy()
    row['timestamp_start'] = start_time
    row['timestamp_end'] = end_time
    row['activity'] = status
    activity_rows.append(row)

df_activities = pd.DataFrame(activity_rows)

# Now set the required columns on df_activities
df_activities['higher_level_activity'] = 'shift'
df_activities['object_type'] = 'production_line'
df_activities['object'] = df_activities['temp_object']

# Define a function to determine the shift based on datetime hour
def get_shift(dt):
    hour = dt.hour
    if 6 <= hour < 14:
        return '1'
    elif 14 <= hour < 22:
        return '2'
    else:
        return '3'

df_activities['case_id'] = 'shift_' + df_activities['timestamp_start'].apply(get_shift) + '_' + df_activities['timestamp_start'].dt.date.astype(str)

# FIX 1: Use df_activities.columns (not df.columns) — mat_id cols must exist in df_activities at this point
mat_id_cols = [col for col in df_activities.columns if
               col.startswith('mat_id_m01_') or
               col.startswith('mat_id_m02_') or
               col.startswith('mat_id_m03_')]

df_activities['object_attributes'] = df_activities.apply(
    lambda row: {col: row[col] for col in mat_id_cols},
    axis=1
)

# Drop unwanted columns: schritt_, mat_id_m01_, mat_id_m02_, mat_id_m03_, status_name, temp_object, activity_change
df_activities = df_activities.drop(columns=[col for col in df_activities.columns if
    'schritt_' in col or 'mat_id_m01_' in col or 'mat_id_m02_' in col or 'mat_id_m03_' in col or
    col in ['status_name', 'temp_object', 'activity_change']], errors='ignore')

# Add _log suffix to activity columns
activity_cols = ['case_id', 'activity', 'timestamp_start', 'timestamp_end', 'object', 'object_type', 'higher_level_activity', 'object_attributes']
df_activities = df_activities.rename(columns={col: col + '_log' for col in activity_cols if col in df_activities.columns})

# Add _energy suffix to remaining sensor/datetime columns
sensor_cols = [col for col in df_activities.columns if not col.endswith('_log')]
df_activities = df_activities.rename(columns={col: col + '_energy' for col in sensor_cols if col in df_activities.columns})

# Initialize log columns in df_expanded with None
log_columns = [col for col in df_activities.columns if col.endswith('_log')]
for col in log_columns:
    df_expanded[col] = None

# FIX 2: When assigning dict values, wrap in a list to prevent pandas unpacking the dict into columns
for obj in df_expanded['temp_object'].unique():
    df_obj = df_expanded[df_expanded['temp_object'] == obj]
    df_act_obj = df_activities[df_activities['object_log'] == obj]

    for _, act_row in df_act_obj.iterrows():
        mask = (
            (df_obj['datetime'] >= act_row['timestamp_start_log']) &
            (df_obj['datetime'] <= act_row['timestamp_end_log'])
        )
        matched_index = df_obj[mask].index
        for col in log_columns:
            val = act_row[col]
            if isinstance(val, dict):
                # Wrap in list to stop pandas unpacking the dict across columns
                df_expanded.loc[matched_index, col] = [val] * len(matched_index)
            else:
                df_expanded.loc[matched_index, col] = val

# Drop temp_object, activity_change, and mat_id columns (now stored in object_attributes_log)
df_expanded = df_expanded.drop(columns=['temp_object', 'activity_change'] +
    [col for col in df_expanded.columns if
     'mat_id_m01_' in col or 'mat_id_m02_' in col or 'mat_id_m03_' in col], errors='ignore')

# Rename all non-log columns (sensor + datetime) to _energy
sensor_cols_combined = [col for col in df_expanded.columns if not col.endswith('_log')]
df_expanded = df_expanded.rename(columns={col: col + '_energy' for col in sensor_cols_combined})

df_expanded['timestamp_start_log'] = pd.to_datetime(df_expanded['timestamp_start_log'])
df_expanded['timestamp_end_log'] = pd.to_datetime(df_expanded['timestamp_end_log'])

df_expanded = df_expanded.drop(columns=['schritt__energy'], errors='ignore')
df_expanded.columns = [col.replace('__', '_') for col in df_expanded.columns]

print(f"Expanded df")
print(df_expanded)

# Extract only the event log
df = df_expanded.copy()

# Keep only columns ending with '_log'
df = df[[col for col in df.columns if col.endswith('_log')]]

# Remove '_log' suffix
df.columns = [col.replace('_log', '') for col in df.columns]

df_event_log = df.copy()

# Remove duplicate rows where all values are the same, excluding dict columns
# (drop_duplicates can't handle dicts, so exclude 'object_attributes' column)
subset_cols = [col for col in df_event_log.columns if col != 'object_attributes']
df_event_log = df_event_log.drop_duplicates(subset=subset_cols)

print(f"Event log")
print(df_event_log)

### Prodcution plan
df = df_event_log.copy()
df = df[['case_id', 'activity', 'timestamp_start', 'timestamp_end', 'object_attributes']]

# Only leave the case ids, the oders
df = df.drop_duplicates(subset=['case_id'])

production_plan = df.copy()

print(f"production plan")
print(production_plan)

process_datasets['process_2'] = {
    'expanded': df_expanded,
    'event_log': df_event_log,
    'production_plan': production_plan
}

# %%
variable = 'pro_volstrom_l/h_energy'

# %%

df = df_expanded.copy()

# Assuming your dataframe is named 'df' with columns 'datetime_energy' and 'n'
# Filter by the specific day
filtered_df = df[df['datetime_energy'].dt.date == pd.to_datetime('2024-11-07').date()]

# Create the plot
fig = px.line(filtered_df, x='datetime_energy', y=variable, title='Energy Demand on 2024-11-07')
fig.show()

# %% [markdown]
# ## Process 3

# %%
# Define the path to the current file's location
current_path = Path(__file__).resolve().parent if '__file__' in globals() else Path().resolve()

# Define the folder path
files_folder_silver = current_path.parent / 'data' / 'silver' / 'process_3'

# Load the Parquet file
df = pd.read_parquet(files_folder_silver / "df_combined_legend.parquet")

df

# %%
df.columns

# %%
# Define the path to the current file's location
current_path = Path(__file__).resolve().parent if '__file__' in globals() else Path().resolve()

# Define the folder path
files_folder_silver = current_path.parent / 'data' / 'silver' / 'process_3'

# Load the Parquet file
df = pd.read_parquet(files_folder_silver / "df_combined_legend.parquet")

# Sort by temp_object and datetime to ensure proper ordering
df['temp_object'] = 'tower_1'  # Assign 'tower_1' to the new column first
df_expanded = df.sort_values(['temp_object', 'datetime']).reset_index(drop=True)

df_expanded['activity'] = df_expanded['status_name']

# Identify activity changes: group consecutive rows with the same status_name per object
df_expanded['activity_change'] = (df_expanded['status_name'] != df_expanded['status_name'].shift()).cumsum()

# Group by temp_object and activity_change to create activity intervals
activity_groups = df_expanded.groupby(['temp_object', 'activity_change'])

# Create a new DataFrame for activities
activity_rows = []
for (obj, change), group in activity_groups:
    group = group.sort_values('datetime')
    start_time = group['datetime'].min()
    end_time = group['datetime'].max()
    status = group['status_name'].iloc[0]

    row = group.iloc[0].copy()
    row['timestamp_start'] = start_time
    row['timestamp_end'] = end_time
    row['activity'] = status
    row['object_attributes'] = {"none": "none"}  # Add the "none": "none" key-value pairk
    activity_rows.append(row)

df_activities = pd.DataFrame(activity_rows)

# Now set the required columns on df_activities
df_activities['higher_level_activity'] = 'shift'
df_activities['object_type'] = 'tower'
df_activities['object'] = df_activities['temp_object']

# Define a function to determine the shift based on datetime hour
def get_shift(dt):
    hour = dt.hour
    if 6 <= hour < 14:
        return '1'
    elif 14 <= hour < 22:
        return '2'
    else:
        return '3'

df_activities['case_id'] = 'shift_' + df_activities['timestamp_start'].apply(get_shift) + '_' + df_activities['timestamp_start'].dt.date.astype(str)

# Add _log suffix to activity columns
activity_cols = ['case_id', 'activity', 'timestamp_start', 'timestamp_end', 'object', 'object_type', 'higher_level_activity', 'object_attributes']
df_activities = df_activities.rename(columns={col: col + '_log' for col in activity_cols if col in df_activities.columns})

# Add _energy suffix to remaining sensor/datetime columns
sensor_cols = [col for col in df_activities.columns if not col.endswith('_log')]
df_activities = df_activities.rename(columns={col: col + '_energy' for col in sensor_cols if col in df_activities.columns})

# Initialize log columns in df_expanded with None
log_columns = [col for col in df_activities.columns if col.endswith('_log')]
for col in log_columns:
    df_expanded[col] = None

# FIX 2: When assigning dict values, wrap in a list to prevent pandas unpacking the dict into columns
for obj in df_expanded['temp_object'].unique():
    df_obj = df_expanded[df_expanded['temp_object'] == obj]
    df_act_obj = df_activities[df_activities['object_log'] == obj]

    for _, act_row in df_act_obj.iterrows():
        mask = (
            (df_obj['datetime'] >= act_row['timestamp_start_log']) &
            (df_obj['datetime'] <= act_row['timestamp_end_log'])
        )
        matched_index = df_obj[mask].index
        for col in log_columns:
            val = act_row[col]
            if isinstance(val, dict):
                # Wrap in list to stop pandas unpacking the dict across columns
                df_expanded.loc[matched_index, col] = [val] * len(matched_index)
            else:
                df_expanded.loc[matched_index, col] = val

# Drop temp_object, activity_change, and mat_id columns (now stored in object_attributes_log)
df_expanded = df_expanded.drop(columns=['temp_object', 'activity_change'] +
    [col for col in df_expanded.columns if
     '(1)_status' in col or 'status_name' in col], errors='ignore')

# Rename all non-log columns (sensor + datetime) to _energy
sensor_cols_combined = [col for col in df_expanded.columns if not col.endswith('_log')]
df_expanded = df_expanded.rename(columns={col: col + '_energy' for col in sensor_cols_combined})

df_expanded['timestamp_start_log'] = pd.to_datetime(df_expanded['timestamp_start_log'])
df_expanded['timestamp_end_log'] = pd.to_datetime(df_expanded['timestamp_end_log'])

df_expanded.columns = [col.replace('__', '_') for col in df_expanded.columns]

print(f"Expanded df")
print(df_expanded)

# Extract only the event log
df = df_expanded.copy()

# Keep only columns ending with '_log'
df = df[[col for col in df.columns if col.endswith('_log')]]

# Remove '_log' suffix
df.columns = [col.replace('_log', '') for col in df.columns]

df_event_log = df.copy()

# Remove duplicate rows where all values are the same, excluding dict columns
# (drop_duplicates can't handle dicts, so exclude 'object_attributes' column)
subset_cols = [col for col in df_event_log.columns if col != 'object_attributes']
df_event_log = df_event_log.drop_duplicates(subset=subset_cols)

print(f"Event log")
print(df_event_log)

### Production plan
df = df_event_log.copy()
df = df[['case_id', 'activity', 'timestamp_start', 'timestamp_end', 'object_attributes']]

# Only leave the case ids, the orders
df = df.drop_duplicates(subset=['case_id'])

production_plan = df.copy()

print(f"production plan")
print(production_plan)

process_datasets['process_3'] = {
    'expanded': df_expanded,
    'event_log': df_event_log,
    'production_plan': production_plan
}

# %%
# # Define the path to the current file's location
# current_path = Path(__file__).resolve().parent if '__file__' in globals() else Path().resolve()

# # Define the folder path
# files_folder_silver = current_path.parent / 'data' / 'silver' / 'process_3'

# # Load the Parquet file
# df = pd.read_parquet(files_folder_silver / "df_combined_legend.parquet")

# # Sort by temp_object and datetime to ensure proper ordering
# df['temp_object'] = 'tower_1'  # Assign 'tower_1' to the new column first
# df_expanded = df.sort_values(['temp_object', 'datetime']).reset_index(drop=True)



# df_expanded['activity'] = df_expanded['status_name']

# # Identify activity changes: group consecutive rows with the same status_name per object
# df_expanded['activity_change'] = (df_expanded['status_name'] != df_expanded['status_name'].shift()).cumsum()

# # Group by temp_object and activity_change to create activity intervals
# activity_groups = df_expanded.groupby(['temp_object', 'activity_change'])

# # Create a new DataFrame for activities
# activity_rows = []
# for (obj, change), group in activity_groups:
#     group = group.sort_values('datetime')
#     start_time = group['datetime'].min()
#     end_time = group['datetime'].max()
#     status = group['status_name'].iloc[0]

#     row = group.iloc[0].copy()
#     row['timestamp_start'] = start_time
#     row['timestamp_end'] = end_time
#     row['activity'] = status
#     activity_rows.append(row)

# df_activities = pd.DataFrame(activity_rows)

# # Now set the required columns on df_activities
# df_activities['higher_level_activity'] = 'shift'
# df_activities['object_type'] = 'tower'
# df_activities['object'] = df_activities['temp_object']

# # Define a function to determine the shift based on datetime hour
# def get_shift(dt):
#     hour = dt.hour
#     if 6 <= hour < 14:
#         return '1'
#     elif 14 <= hour < 22:
#         return '2'
#     else:
#         return '3'

# df_activities['case_id'] = 'shift_' + df_activities['timestamp_start'].apply(get_shift) + '_' + df_activities['timestamp_start'].dt.date.astype(str)

# df_activities['object_attributes'] = {}

# # Add _log suffix to activity columns
# activity_cols = ['case_id', 'activity', 'timestamp_start', 'timestamp_end', 'object', 'object_type', 'higher_level_activity', 'object_attributes']
# df_activities = df_activities.rename(columns={col: col + '_log' for col in activity_cols if col in df_activities.columns})

# # Add _energy suffix to remaining sensor/datetime columns
# sensor_cols = [col for col in df_activities.columns if not col.endswith('_log')]
# df_activities = df_activities.rename(columns={col: col + '_energy' for col in sensor_cols if col in df_activities.columns})

# # Initialize log columns in df_expanded with None
# log_columns = [col for col in df_activities.columns if col.endswith('_log')]
# for col in log_columns:
#     df_expanded[col] = None

# # FIX 2: When assigning dict values, wrap in a list to prevent pandas unpacking the dict into columns
# for obj in df_expanded['temp_object'].unique():
#     df_obj = df_expanded[df_expanded['temp_object'] == obj]
#     df_act_obj = df_activities[df_activities['object_log'] == obj]

#     for _, act_row in df_act_obj.iterrows():
#         mask = (
#             (df_obj['datetime'] >= act_row['timestamp_start_log']) &
#             (df_obj['datetime'] <= act_row['timestamp_end_log'])
#         )
#         matched_index = df_obj[mask].index
#         for col in log_columns:
#             val = act_row[col]
#             if isinstance(val, dict):
#                 # Wrap in list to stop pandas unpacking the dict across columns
#                 df_expanded.loc[matched_index, col] = [val] * len(matched_index)
#             else:
#                 df_expanded.loc[matched_index, col] = val

# # Drop temp_object, activity_change, and mat_id columns (now stored in object_attributes_log)
# df_expanded = df_expanded.drop(columns=['temp_object', 'activity_change'] +
#     [col for col in df_expanded.columns if
#      '(1)_status' in col or 'status_name' in col], errors='ignore')

# # Rename all non-log columns (sensor + datetime) to _energy
# sensor_cols_combined = [col for col in df_expanded.columns if not col.endswith('_log')]
# df_expanded = df_expanded.rename(columns={col: col + '_energy' for col in sensor_cols_combined})

# df_expanded['timestamp_start_log'] = pd.to_datetime(df_expanded['timestamp_start_log'])
# df_expanded['timestamp_end_log'] = pd.to_datetime(df_expanded['timestamp_end_log'])

# df_expanded.columns = [col.replace('__', '_') for col in df_expanded.columns]

# print(f"Expanded df")
# print(df_expanded)

# # Extract only the event log
# df = df_expanded.copy()

# # Keep only columns ending with '_log'
# df = df[[col for col in df.columns if col.endswith('_log')]]

# # Remove '_log' suffix
# df.columns = [col.replace('_log', '') for col in df.columns]

# df_event_log = df.copy()

# # Remove duplicate rows where all values are the same, excluding dict columns
# # (drop_duplicates can't handle dicts, so exclude 'object_attributes' column)
# subset_cols = [col for col in df_event_log.columns if col != 'object_attributes']
# df_event_log = df_event_log.drop_duplicates(subset=subset_cols)

# print(f"Event log")
# print(df_event_log)

# ### Prodcution plan
# df = df_event_log.copy()
# df = df[['case_id', 'activity', 'timestamp_start', 'timestamp_end', 'object_attributes']]

# # Only leave the case ids, the oders
# df = df.drop_duplicates(subset=['case_id'])

# production_plan = df.copy()

# print(f"production plan")
# print(production_plan)

# process_datasets['process_3'] = {
#     'expanded': df_expanded,
#     'event_log': df_event_log,
#     'production_plan': production_plan
# }

# %% [markdown]
# ## Process 4

# %%
objects_to_analyze = ['Erhitzer']

# %%
# Define the path to the current file's location
current_path = Path(__file__).resolve().parent if '__file__' in globals() else Path().resolve()

# Define the folder path
files_folder_silver = current_path.parent / 'data' / 'silver' / 'process_4'

## Get the sensor data
# Strom in kWh und Dampf in kg
df_energy = pd.read_parquet(files_folder_silver/"df_sensor_joined.parquet")
print(df_energy.columns)

# Assuming 'datetime' is the timestamp column
df_energy['datetime'] = pd.to_datetime(df_energy['datetime'])
df_energy.set_index('datetime', inplace=True)

# Resample by minute and aggregate (e.g., mean for averages, sum for totals - adjust as needed)
df_energy = df_energy.resample('min').mean()  # or .sum() if appropriate

df_energy.reset_index(inplace=True)


### Get the log data
# Load the Parquet file
df = pd.read_parquet(files_folder_silver / "df_events.parquet")


df_with_higher_level = df[df['higher_level_activity'].notna()].copy()


df_higher_level = df[df['higher_level_activity'].isna() & (df['activity'] == 'production')].copy()

def find_matching_higher_level(row):

    matches = df_higher_level[
        (df_higher_level['timestamp_start'] <= row['timestamp_start']) &
        (df_higher_level['timestamp_end'] >= row['timestamp_end'])
    ]
    if not matches.empty:

        match = matches.iloc[0]
        return match['case_id'], match['object_attributes']
    else:
        return None, None

# Apply the function to each row in df_with_higher_level
df_with_higher_level[['matched_case_id', 'matched_object_attributes']] = df_with_higher_level.apply(
    lambda row: pd.Series(find_matching_higher_level(row)), axis=1
)

# Now, update the case_id and merge object_attributes
df_with_higher_level['case_id'] = df_with_higher_level['matched_case_id']
# For object_attributes, you can merge or update as needed. Assuming you want to add the higher level's attributes
df_with_higher_level['object_attributes'] = df_with_higher_level.apply(
    lambda row: {**row['object_attributes'], **row['matched_object_attributes']} if row['matched_object_attributes'] else row['object_attributes'],
    axis=1
)

# Drop the temporary columns
df_with_higher_level = df_with_higher_level.drop(columns=['matched_case_id', 'matched_object_attributes'])

# Now, combine the updated lower-level activities with the unchanged higher-level activities
df_final = pd.concat([df_with_higher_level, df_higher_level], ignore_index=True)

# Sort by timestamp_start for better order
df_event_log = df_final.sort_values(by='timestamp_start').reset_index(drop=True)


# Filter for rows where object == 'order'
df_orders = df_event_log[df_event_log['object_type'] == 'order'].copy()

# Group by case_id and find the maximum timestamp_end for each case_id
max_ends = df_orders.groupby('case_id')['timestamp_end'].max()

# Prepare a list to hold new rows
new_rows = []

# For each case_id, create a new activity row
for case_id, max_end in max_ends.items():
    # Get the original row for this case_id (assuming there's one per case_id for 'order')
    original_row = df_orders[df_orders['case_id'] == case_id].iloc[0]
    
    # Create the new row with the same properties, but new activity and timestamps
    new_row = {
        'case_id': case_id,
        'activity': 'end',  # New activity name; adjust if needed
        'timestamp_start': max_end + pd.Timedelta(seconds=1),
        'timestamp_end': max_end + pd.Timedelta(seconds=1),  # Ends at the same moment as start (zero duration)
        'object_type': 'order',
        'object': original_row['object'],
        'higher_level_activity': original_row['higher_level_activity'],
        'object_attributes': original_row['object_attributes']
    }
    new_rows.append(new_row)

# Convert new rows to DataFrame
df_new = pd.DataFrame(new_rows)

# Append the new rows to df_event_log
df = pd.concat([df_event_log, df_new], ignore_index=True)

# Sort the DataFrame by timestamp_start for proper ordering
df_event_log = df_event_log.sort_values(by='timestamp_start').reset_index(drop=True)

df_event_log = df_event_log[df_event_log['object'].isin(objects_to_analyze)]

print(f"Event log")
print(df_event_log)


### Get the expanded df with sensor data

def expand_activities_to_timeseries(df_activities, df_energy, energy_cols=None):
    """
    For each activity, filter df_energy to the [timestamp_start, timestamp_end) interval,
    then create rows combining activity info with each energy timestamp's data.
    Append all into one DataFrame, preserving the complete time series per activity.
    """
    if energy_cols is None:
        energy_cols = [col for col in df_energy.columns if col != 'datetime']
    
    expanded_rows = []
    
    for _, activity_row in df_activities.iterrows():
        start = activity_row['timestamp_start_log']
        end = activity_row['timestamp_end_log']
        
        # Filter energy data within the activity's time interval
        mask = (df_energy['datetime_energy'] >= start) & (df_energy['datetime_energy'] < end)
        energy_subset = df_energy.loc[mask, ['datetime_energy'] + energy_cols]
        
        # Create new rows: activity data + each energy row
        for _, energy_row in energy_subset.iterrows():
            new_row = activity_row.to_dict()
            new_row.update(energy_row.to_dict())  # Merge energy data
            expanded_rows.append(new_row)
    
    return pd.DataFrame(expanded_rows)

# Rename columns in df_final_clean to add suffix '_log'
df_activites = df_event_log.add_suffix('_log')

# Rename columns in df_energy to add prefix '_sen', except 'datetime'
df_energys = df_energy.add_suffix('_energy')

# Usage: Use all sensor columns
df_expanded = expand_activities_to_timeseries(df_activites, df_energys)

# View result
print("Expanded df with sensor data")
print(df_expanded)


## Get the production plan
df = df_event_log.copy()

df = df[df['object'].isin(objects_to_analyze)]

df = df[['case_id', 'activity', 'timestamp_start', 'timestamp_end', 'object_attributes']]

df = df.dropna(subset=['case_id'])

df = df.drop_duplicates(subset=['case_id'])

production_plan = df.copy()

print("Expanded production plan")
print(production_plan)


process_datasets['process_4'] = {
    'expanded': df_expanded,
    'event_log': df_event_log,
    'production_plan': production_plan
}


# %% [markdown]
# # Model the process

# %%
# Functions

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
                                       start_col='timestamp_start', end_col='timestamp_end'):
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

    if len(sim_for_dfg) > 0 and len(real_for_dfg) > 0:
        try:
            # Mine a reference model from the real log and replay simulated traces on it.
            ref_net, ref_im, ref_fm = pm4py.discover_petri_net_inductive(real_log)

            conformance_metrics['fitness'] = _mean_trace_fitness(sim_log, ref_net, ref_im, ref_fm)
            conformance_metrics['precision'] = _safe_precision(sim_log, ref_net, ref_im, ref_fm)
            conformance_metrics['generalization'] = _safe_generalization(sim_log, ref_net, ref_im, ref_fm)
            conformance_metrics['simplicity'] = _safe_simplicity(ref_net)
        except Exception as e:
            print(f"Conformance dimensions unavailable: {e}")

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
            if not sim_curves:
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
                        wape = np.sum(np.abs(y_real - y_sim_resampled)) / denom if denom > 0 else 0
                        
                        if sensor not in sensor_metrics:
                            sensor_metrics[sensor] = {'mae': [], 'rmse': [], 'wape': []}
                        sensor_metrics[sensor]['mae'].append(mae)
                        sensor_metrics[sensor]['rmse'].append(rmse)
                        sensor_metrics[sensor]['wape'].append(wape)
        
        # Aggregate per sensor
        agg_energy = {}
        for sensor, m in sensor_metrics.items():
            s_mae = np.mean(m['mae'])
            s_rmse = np.mean(m['rmse'])
            s_wape = np.mean(m['wape'])
            agg_energy[sensor] = {'MAE': s_mae, 'RMSE': s_rmse, 'WAPE': s_wape}
            report(f"  {sensor:40}: MAE={s_mae:.2f}, RMSE={s_rmse:.2f}, WAPE={s_wape:.4f}")
        
        results['energy_metrics'] = agg_energy
        energy_results = agg_energy

    # ========== 8. OVERALL QUALITY SCORE ==========
    report("\n8. OVERALL QUALITY ASSESSMENT")
    report("-" * 40)

    # Active score requested by user:
    # keep conformance metrics + mean duration + event_count_ratio.
    # all active metrics have equal weight.
    active_components = {
        'event_count_ratio': results['basic_metrics']['event_count_ratio'],
        'mean_duration_similarity': (1.0 - results['duration_metrics']['mean_duration_error']),
        'fitness': results['conformance_metrics'].get('fitness'),
        'precision': results['conformance_metrics'].get('precision'),
        'generalization': results['conformance_metrics'].get('generalization'),
        'simplicity': results['conformance_metrics'].get('simplicity'),
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

# %%
print(process_datasets.keys())

# %%
# Extract only process_4 while maintaining the dictionary structure
#rocess_datasets_to_model = {'process_4': process_datasets['process_4']}

process_datasets_to_model = process_datasets

process_datasets_to_model_sensors = process_datasets_to_model.copy()
# process_datasets_to_model_sensors['process_4'] = process_datasets_to_model_sensors.get('process_4', {})
# process_datasets_to_model_sensors['process_4']['objects_to_model'] = ['Erhitzer']
# process_datasets_to_model_sensors['process_4']['activities_to_model'] = ['Step-032 = Umlauf', 'Step-030 = Produktion']
# process_datasets_to_model_sensors['process_4']['sensors_to_model'] = ['temp_nach_WR2_(WT2)_5s_energy']

# process_datasets_to_model_sensors['process_2'] = process_datasets_to_model_sensors.get('process_2', {})
# process_datasets_to_model_sensors['process_2']['objects_to_model'] = ['l01']
# process_datasets_to_model_sensors['process_2']['activities_to_model'] = ['Produktion']
# process_datasets_to_model_sensors['process_2']['sensors_to_model'] = ['pro_volstrom_l/h_energy']

# process_datasets_to_model_sensors['process_3'] = process_datasets_to_model_sensors.get('process_3', {})
# process_datasets_to_model_sensors['process_3']['objects_to_model'] = ['tower_1']
# process_datasets_to_model_sensors['process_3']['activities_to_model'] = ['Produktion']
# process_datasets_to_model_sensors['process_3']['sensors_to_model'] = ['(8)_abluft_mas_kg/h_energy']

processes_to_run = ['process_2']
# Filter the original dictionary
process_datasets_to_model = {
    k: v for k, v in process_datasets.items() 
    if k in processes_to_run
}
# The rest of your configuration will now only see process_3
process_datasets_to_model_sensors = process_datasets_to_model.copy()


# %% [markdown]
# # Visualize All Sensor Profiles (One Graph per Sensor)

# %%
def visualize_all_sensor_profiles(process_datasets, process_name, max_points_per_sensor=20000):
    """
    Plot one time-series graph per sensor for a selected process dataset.
    """
    if process_name not in process_datasets:
        raise ValueError(f"Process '{process_name}' not found. Available: {list(process_datasets.keys())}")

    df_proc = process_datasets[process_name].get('expanded')
    if df_proc is None or df_proc.empty:
        raise ValueError(f"No expanded dataframe found for process '{process_name}'.")

    df_plot = df_proc.copy()
    time_col = 'datetime_energy' if 'datetime_energy' in df_plot.columns else 'datetime'
    if time_col not in df_plot.columns:
        raise ValueError(f"No datetime column found in '{process_name}' expanded data.")

    df_plot[time_col] = pd.to_datetime(df_plot[time_col], errors='coerce')
    df_plot = df_plot.dropna(subset=[time_col]).sort_values(time_col)

    # Prefer columns with _energy suffix and numeric values.
    candidate_cols = [
        c for c in df_plot.columns
        if c.endswith('_energy') and c != time_col
    ]
    sensor_cols = [
        c for c in candidate_cols
        if pd.api.types.is_numeric_dtype(df_plot[c])
    ]

    if not sensor_cols:
        raise ValueError(f"No numeric sensor columns found for process '{process_name}'.")

    report(f"\n[VISUAL] Process: {process_name} | Sensors to plot: {len(sensor_cols)}")

    for sensor in sensor_cols:
        sensor_df = df_plot[[time_col, sensor]].dropna()
        if sensor_df.empty:
            continue

        # Keep plots responsive if data is very dense.
        if len(sensor_df) > max_points_per_sensor:
            step = max(1, len(sensor_df) // max_points_per_sensor)
            sensor_df = sensor_df.iloc[::step].copy()

        fig = px.line(
            sensor_df,
            x=time_col,
            y=sensor,
            title=f"{process_name} | {sensor}",
            labels={time_col: 'Timestamp', sensor: sensor}
        )
        fig.update_layout(height=450, width=1400, hovermode='x unified')
        fig.show()


# Choose the process you want to visualize here.
process_to_visualize_profiles = 'process_2'
#visualize_all_sensor_profiles(process_datasets, process_to_visualize_profiles)

#%%

display(process_datasets['process_2'])
# %%


# %% [markdown]
# # Causal Discovery (PCMCI with Tigramite)
#
# Method summary:
# - Run PCMCI on numeric time-series columns from a selected process.
# - Model variables are columns that end with the configured suffix and do not start with 'ef'.
# - All other variables are treated as parent-only drivers: they can be causes but are never shown as children.

# %%
def _select_pcmci_variables(df, model_suffix='enegy', fallback_suffix='energy'):
    """Select model and exogenous variables for constrained causal discovery.

    Columns starting with 'ef' are always exogenous: they can be parents of
    other variables but never have parents themselves.
    """
    numeric_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
    ]

    suffix = model_suffix.lower()
    fallback = fallback_suffix.lower()

    # ef-prefixed columns are always exogenous — exclude from model vars
    exogenous_vars = [c for c in numeric_cols if c.lower().startswith('ef')]

    model_vars = [
        c for c in numeric_cols
        if c.lower().endswith(suffix) and not c.lower().startswith('ef')
    ]

    # Graceful fallback for common spelling/column naming in this project.
    if not model_vars and suffix != fallback:
        model_vars = [
            c for c in numeric_cols
            if c.lower().endswith(fallback) and not c.lower().startswith('ef')
        ]
        if model_vars:
            report(
                f"[PCMCI] No columns matched suffix '{model_suffix}'. "
                f"Using fallback suffix '{fallback_suffix}'."
            )

    # Everything that is not a model var becomes parent-only (includes ef cols)
    parent_only_vars = [c for c in numeric_cols if c not in model_vars]
    return model_vars, parent_only_vars, exogenous_vars


#%%


def run_causal_discovery(
    dataset,
    method='pcmci',           # 'pc' or 'pcmci'
    model_suffix='enegy',
    tau_max=6,
    pc_alpha=0.05,
    alpha_level=0.05,
    max_rows=30000,
    train_df=None,            # if provided, use only this subset for fitting
    activity_col=None,        # column whose lags are forced as parents of all others
):
    """
    Run constrained causal discovery (PC or PCMCI) and plot a time-series graph.

    Constraints:
    - Non-model variables are parent-only (never shown as children).
    - ef-prefixed columns are always exogenous (parents only).
    - activity_col links are forced into the graph at all lags toward every
      other variable.

    Parameters
    ----------
    method : 'pc' or 'pcmci'
    train_df : pd.DataFrame or None
        If given, causal discovery runs on this subset (e.g. training split).
    activity_col : str or None
        Column name that is forced as a parent of all other variables.
    """
    if PCMCI is None or ParCorr is None or pp is None or tp is None:
        raise ImportError(
            "tigramite is not installed. Install with: pip install tigramite"
        )

    method = method.lower()
    if method not in ('pc', 'pcmci'):
        raise ValueError(f"method must be 'pc' or 'pcmci', got '{method}'.")

    df = (train_df if train_df is not None else dataset).copy()
    if train_df is not None:
        report(f"[{method.upper()}] Using train_df ({len(df)} rows) for causal discovery.")

    model_vars, parent_only_vars, exogenous_vars = _select_pcmci_variables(df, model_suffix=model_suffix)

    if not model_vars:
        raise ValueError(
            "No model variables found. Check your suffix and column naming (example: *_energy)."
        )

    all_vars = model_vars + parent_only_vars
    df_model = df[all_vars].replace([np.inf, -np.inf], np.nan).dropna()

    if df_model.empty:
        raise ValueError("No valid rows left after dropping NaNs/Infs.")

    # Keep runtime manageable on dense sensor logs.
    if len(df_model) > max_rows:
        stride = max(1, len(df_model) // max_rows)
        df_model = df_model.iloc[::stride].copy()
        report(f"[{method.upper()}] Downsampled to {len(df_model)} rows (stride={stride}).")

    tig_df = pp.DataFrame(data=df_model.values, var_names=all_vars)
    pcmci_obj = PCMCI(dataframe=tig_df, cond_ind_test=ParCorr())

    if method == 'pc':
        # PC stable gives the skeleton; run MCI on top to get p_matrix/val_matrix
        pc_results = pcmci_obj.run_pc_stable(tau_min=1, tau_max=tau_max, pc_alpha=pc_alpha)
        results = pcmci_obj.run_mci(
            selected_links=pc_results.get('selected_links'),
            tau_min=1,
            tau_max=tau_max,
        )
    else:
        results = pcmci_obj.run_pcmci(tau_min=1, tau_max=tau_max, pc_alpha=pc_alpha)

    q_matrix = pcmci_obj.get_corrected_pvalues(
        p_matrix=results['p_matrix'],
        tau_max=tau_max,
        fdr_method='fdr_bh'
    )
    graph = q_matrix < alpha_level

    # Enforce parent-only rule: non-model variables (including ef cols) can never be children.
    parent_only_idx = [i for i, v in enumerate(all_vars) if v in parent_only_vars]
    for j in parent_only_idx:
        graph[:, j, :] = False

    # Force activity column as parent of every other variable at all lags.
    if activity_col is not None and activity_col in all_vars:
        act_i = all_vars.index(activity_col)
        for j, v in enumerate(all_vars):
            if j != act_i:
                graph[act_i, j, 1:] = True   # tau=0 slice is contemporaneous; lags start at index 1
        report(f"[{method.upper()}] Forced '{activity_col}' as parent of all variables.")

    # Print strong links for quick inspection.
    report(f"\n[{method.upper()}] Significant links (after child-constraint):")
    pcmci_obj.print_significant_links(
        p_matrix=q_matrix,
        val_matrix=results['val_matrix'],
        alpha_level=alpha_level,
        graph=graph,
    )

    # Time-series causal discovery graph.
    tp.plot_graph(
        val_matrix=results['val_matrix'],
        graph=graph,
        var_names=all_vars,
        figsize=(16, 12),
        node_size=0.25,
    )
    plt.title(
        f"{method.upper()} Causal Discovery | model_suffix='{model_suffix}'",
        fontsize=12,
        fontweight='bold'
    )
    plt.tight_layout()
    plt.show()

    # Extract causal links: for each target variable, which (parent, lag) pairs are significant.
    # Shape of graph: (n_vars, n_vars, tau_max+1); graph[i, j, tau] = True means var i at lag tau -> var j.
    causal_links = {}
    for j, target in enumerate(all_vars):
        parents = []
        for i, parent in enumerate(all_vars):
            for tau in range(1, tau_max + 1):
                if graph[i, j, tau]:
                    parents.append((parent, tau))
        causal_links[target] = parents

    report(f"\n[{method.upper()}] Causal links extracted:")
    for var, parents in causal_links.items():
        if parents:
            report(f"  {var[:45]:45s} <- {parents}")

    return {
        'method': method,
        'model_vars': model_vars,
        'parent_only_vars': parent_only_vars,
        'exogenous_vars': exogenous_vars,
        'all_vars': all_vars,
        'results': results,
        'q_matrix': q_matrix,
        'graph': graph,
        'causal_links': causal_links,
        'used_rows': len(df_model),
    }


#%%


df_proc = process_datasets['process_2'].get('expanded')


display(df_proc.columns)

display(df_proc)

#%%

df = df_proc

columns = ['datetime_energy',
           'ef_precipitation_energy', 'medium_power_kW_energy',
       'ef_temperature_2m_energy', 'ef_global_tilted_irradiance_energy',
       'status_name_energy', 'pro_volstrom_l/h_energy',
       'ef_apparent_temperature_energy', 'dampfmenge_kg/h_nmb+cip_energy',
       'cip_turm_f_energy', 'ef_wind_speed_10m_energy', 'pro_temp_out_energy',
        
       'ef_relative_humidity_2m_energy',
       'pro_menge_kg_energy', 'pro_power_kW_energy',
       'ef_wind_direction_100m_energy', 'cip_turm_g_energy',
       #'timestamp_start_log', 'timestamp_end_log',
       'activity_log',
       #'higher_level_activity_log', 'object_type_log', 'object_log',
       #'case_id_log',
       'object_attributes_log'
       ]

df = df[columns]
display(df)




#%%

df = df_proc.copy()

display(df)

df = df[columns]

# Resample to 15-minute intervals: average all non-log columns, keep last value for log columns
datetime_col = 'datetime_energy'
df[datetime_col] = pd.to_datetime(df[datetime_col])
df = df.set_index(datetime_col)

log_cols = [c for c in df.columns if c.endswith('_log')]
# activity_energy is non-numeric string — keep it with last() like log cols
categorical_cols = log_cols + (['activity_energy'] if 'activity_energy' in df.columns else [])
numeric_non_log_cols = [
    c for c in df.columns
    if c not in categorical_cols and pd.api.types.is_numeric_dtype(df[c])
]
resampling = '2min'

df_resampled = df[numeric_non_log_cols].resample(resampling).mean()
if categorical_cols:
    df_resampled[categorical_cols] = df[categorical_cols].resample(resampling).last()

df = df_resampled.reset_index()

df = df.rename(columns={'activity_log': 'ef_activity_log'})

# Expand object_attributes_log dict → flat attr_* columns (one per attribute key).
# Each attr_* column is categorical and will be treated as a forced exogenous
# parent of all targets at all lags, just like ef_activity_log.
if 'object_attributes_log' in df.columns:
    _attr_expanded = df['object_attributes_log'].apply(
        lambda x: x if isinstance(x, dict) else {}
    )
    _attr_df = pd.DataFrame(_attr_expanded.tolist(), index=df.index)
    _attr_df.columns = [f'attr_{c}' for c in _attr_df.columns]
    for c in _attr_df.columns:
        _attr_df[c] = _attr_df[c].astype(str).replace('nan', np.nan)
    df = pd.concat([df.drop(columns=['object_attributes_log']), _attr_df], axis=1)
    report(f"[Data] Expanded object_attributes_log → {list(_attr_df.columns)}")

df = df.drop(columns=['datetime_energy'], errors='ignore')

display(df)

#%%

# Per-activity step index: resets to 1 each time a new activity starts
df['ef_activity_step'] = df.groupby(
    (df['ef_activity_log'] != df['ef_activity_log'].shift()).cumsum()
).cumcount() + 1

df_clean = df.copy()

display(df)

#%%

# Plot each activity's curve for '(21)_zuluft_turm_mas_kg/h_energy' by activity step
sensor_col = variable

activity_segments = (df['ef_activity_log'] != df['ef_activity_log'].shift()).cumsum()
unique_segments = activity_segments.unique()

fig, ax = plt.subplots(figsize=(14, 6))

for seg_id in unique_segments:
    seg = df[activity_segments == seg_id]
    activity_name = seg['ef_activity_log'].iloc[0]
    ax.plot(seg['ef_activity_step'], seg[sensor_col], alpha=0.5, label=activity_name)

ax.set_xlabel('Step within activity')
ax.set_ylabel(sensor_col)
ax.set_title(f'Activity curves — {sensor_col}')

# Deduplicate legend entries
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=8)

plt.tight_layout()
plt.show()

#%%

# pcmci_output = run_causal_discovery(
#     dataset=df,
#     method='pcmci',           # switch to 'pc' for faster PC algorithm
#     model_suffix='enegy',
#     tau_max=6,
#     pc_alpha=0.05,
#     alpha_level=0.05,
# )
# %%

display(df_clean)

# %%

# ══════════════════════════════════════════════════════════════════════════════
# STRUCTURAL CAUSAL MODEL — TRAIN, EVALUATE, SIMULATE
# ══════════════════════════════════════════════════════════════════════════════
# Step 1: causal discovery on the train split → produces causal_links
# Step 2: train_scm uses those links to build per-target feature sets
# Step 3: simulate_scm rolls the SCM forward step-by-step

#%%

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import warnings

SCM_TAU_MAX = 6
SCM_N_TEST_ACTIVITIES = 30
SCM_SIM_STEPS = 2000


def _build_lag_features(df, target_cols, exog_cols, tau_max, causal_links=None):
    """
    Build lag features. If causal_links is provided (dict: target -> [(parent, lag)]),
    only the causally-justified (parent, lag) pairs are used as features for each target.
    Otherwise falls back to all lags of all columns for every target.
    """
    all_cols = target_cols + exog_cols

    if causal_links is not None:
        # Collect all (col, lag) pairs needed across all targets
        needed = set()
        for target in target_cols:
            for parent, lag in causal_links.get(target, []):
                if parent in all_cols:
                    needed.add((parent, lag))
        # Fallback: if a target has no links at all, give it all lags of all cols
        for target in target_cols:
            if not causal_links.get(target):
                for col in all_cols:
                    for lag in range(1, tau_max + 1):
                        needed.add((col, lag))
    else:
        needed = {(col, lag) for col in all_cols for lag in range(1, tau_max + 1)}

    # ef_activity_log and attr_* columns are unconditional parents of every target
    for col in all_cols:
        if col == 'ef_activity_log' or col.startswith('attr_'):
            for lag in range(1, tau_max + 1):
                needed.add((col, lag))

    lagged = {}
    for col, lag in sorted(needed):
        key = f'{col}_lag{lag}'
        lagged[key] = df[col].shift(lag)

    feat_df = pd.DataFrame(lagged, index=df.index)
    combined = pd.concat([df[all_cols], feat_df], axis=1).dropna()
    return combined, list(feat_df.columns), causal_links


def train_scm(df_train, causal_output, df_test=None, tau_max=SCM_TAU_MAX, n_test_activities=SCM_N_TEST_ACTIVITIES, use_all_features=False):
    """
    Train one GradientBoostingRegressor per target variable (*_energy, not ef*).

    df_train         : training split (rows used for fitting)
    df_test          : test split (rows used for evaluation / simulation); if None a
                       time-based fallback split is derived from df_train's parent df.
    causal_output    : dict returned by run_causal_discovery.
    use_all_features : if True, ignore causal links and use ALL lagged variables as
                       features for every target (the "all" baseline).
    """
    causal_links = {} if use_all_features else causal_output['causal_links']

    # Cast ef_activity_log and all attr_* columns to pandas Categorical.
    # Categories are fit on train only and applied to test.
    # XGBoost with enable_categorical=True treats these as unordered splits.
    _cat_cols = (
        ['ef_activity_log'] if 'ef_activity_log' in df_train.columns
                               and not pd.api.types.is_numeric_dtype(df_train['ef_activity_log'])
        else []
    ) + [
        c for c in df_train.columns
        if c.startswith('attr_') and not pd.api.types.is_numeric_dtype(df_train[c])
    ]
    if _cat_cols:
        df_train = df_train.copy()
        if df_test is not None:
            df_test = df_test.copy()
        for _c in _cat_cols:
            _cats = df_train[_c].astype('category').cat.categories
            df_train[_c] = pd.Categorical(df_train[_c], categories=_cats)
            if df_test is not None:
                df_test[_c] = pd.Categorical(df_test[_c], categories=_cats)
        report(f"[SCM] Categorical columns: {_cat_cols}")

    if df_test is None:
        # Fallback: derive split internally
        df_all = df_train
        activity_col = 'ef_activity_log' if 'ef_activity_log' in df_all.columns else None
        if activity_col:
            seg_ids = (df_all[activity_col] != df_all[activity_col].shift()).cumsum()
            unique_segs = seg_ids.unique()
            test_segs = unique_segs[-min(n_test_activities, max(1, len(unique_segs) // 5)):]
            test_mask = seg_ids.isin(test_segs)
        else:
            cutoff = int(len(df_all) * 0.7)
            test_mask = pd.Series([False] * cutoff + [True] * (len(df_all) - cutoff), index=df_all.index)
        df_train = df_all[~test_mask].copy()
        df_test  = df_all[test_mask].copy()

    target_cols = [
        c for c in df_train.columns
        if c.endswith('_energy') and not c.startswith('ef')
        and pd.api.types.is_numeric_dtype(df_train[c])
    ]
    # Exogenous: numeric ef_* cols + ef_activity_log + all attr_* columns
    exog_cols = [
        c for c in df_train.columns
        if (c.startswith('ef') and (pd.api.types.is_numeric_dtype(df_train[c]) or c == 'ef_activity_log'))
        or c.startswith('attr_')
    ]

    report(f"\n[SCM] Targets: {len(target_cols)} | Exogenous: {len(exog_cols)}")
    report(f"[SCM] Train rows: {len(df_train)} | Test rows: {len(df_test)}")

    # Build lag features on train only for fitting.
    # For test evaluation, prepend the last tau_max train rows so the first test
    # rows get valid lag values that look back into real training history.
    # When use_all_features=True, pass causal_links=None so all lags are built.
    _links_for_features = None if use_all_features else causal_links
    train_combined, all_feature_cols, _ = _build_lag_features(
        df_train, target_cols, exog_cols, tau_max, causal_links=_links_for_features
    )
    _context = df_train[target_cols + exog_cols].tail(tau_max)
    _df_test_with_context = pd.concat([_context, df_test[target_cols + exog_cols]])
    test_combined_full, _, _ = _build_lag_features(
        _df_test_with_context, target_cols, exog_cols, tau_max, causal_links=_links_for_features
    )
    # Drop the context rows — keep only the actual test rows
    test_combined = test_combined_full[test_combined_full.index.isin(df_test.index)]

    # ef_activity_log and attr_* lag columns are forced for every target
    _forced_bases = ['ef_activity_log'] + [c for c in exog_cols if c.startswith('attr_')]
    activity_lag_cols = [
        f'{base}_lag{lag}'
        for base in _forced_bases
        for lag in range(1, tau_max + 1)
        if f'{base}_lag{lag}' in all_feature_cols
    ]

    models, metrics, target_feature_cols, cat_col_categories = {}, {}, {}, {}

    for target in target_cols:
        # Per-target features: causal parents + activity lags always included.
        # When use_all_features=True, every lagged variable is a feature.
        if use_all_features:
            t_feat_cols = list(all_feature_cols)
        else:
            parents = causal_links.get(target, [])
            if parents:
                t_feat_cols = [
                    f'{parent}_lag{lag}'
                    for parent, lag in parents
                    if f'{parent}_lag{lag}' in all_feature_cols
                ]
            else:
                t_feat_cols = list(all_feature_cols)

            if not t_feat_cols:
                t_feat_cols = list(all_feature_cols)

            # Always include activity lags — they are unconditional parents of every target
            for c in activity_lag_cols:
                if c not in t_feat_cols:
                    t_feat_cols.append(c)

        X_train = train_combined[t_feat_cols].copy()
        y_train = train_combined[target]
        X_test  = test_combined[t_feat_cols].copy() if len(test_combined) > 0 else pd.DataFrame(columns=t_feat_cols)
        y_test  = test_combined[target] if len(test_combined) > 0 else pd.Series(dtype=float)

        if len(X_train) < 10 or y_train.std() == 0:
            continue

        # Cast ef_activity_log and attr_* lag columns to category for XGBoost.
        # Must be string first — shift() introduces float NaNs which break XGBoost.
        cat_feat_cols = [c for c in t_feat_cols
                         if c.startswith('ef_activity_log') or
                         any(c.startswith(f'{base}_lag') for base in exog_cols if base.startswith('attr_'))]
        for c in cat_feat_cols:
            _cats = sorted(set(X_train[c].astype(str).tolist()) | {'nan'})
            cat_col_categories[c] = _cats
            X_train[c] = pd.Categorical(X_train[c].astype(str), categories=_cats)
            if len(X_test) > 0:
                X_test[c] = pd.Categorical(X_test[c].astype(str), categories=_cats)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model = XGBRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, random_state=42,
                enable_categorical=True, tree_method='hist',
            )
            model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test  = model.predict(X_test) if len(X_test) > 0 else np.array([])

        def _wape(y_true, y_pred):
            denom = np.abs(y_true).sum()
            return np.abs(y_true - y_pred).sum() / denom if denom > 0 else np.nan

        has_test = len(y_pred_test) > 0
        metrics[target] = {
            'train_r2':   r2_score(y_train, y_pred_train),
            'train_mae':  mean_absolute_error(y_train, y_pred_train),
            'train_rmse': np.sqrt(np.mean((y_train.values - y_pred_train) ** 2)),
            'train_wape': _wape(y_train.values, y_pred_train),
            'test_r2':    r2_score(y_test, y_pred_test)            if has_test else np.nan,
            'test_mae':   mean_absolute_error(y_test, y_pred_test) if has_test else np.nan,
            'test_rmse':  np.sqrt(np.mean((y_test.values - y_pred_test) ** 2)) if has_test else np.nan,
            'test_wape':  _wape(y_test.values, y_pred_test)        if has_test else np.nan,
        }
        models[target] = model
        target_feature_cols[target] = t_feat_cols
        report(
            f"  {target[:50]:50s} | "
            f"train R²={metrics[target]['train_r2']:.3f} | "
            f"test  R²={metrics[target]['test_r2']:.3f}  MAE={metrics[target]['test_mae']:.3f}"
        )

    # ── training metrics table ────────────────────────────────────────────────
    if metrics:
        train_rows = [
            {
                'target':     t,
                'R²':         round(m['train_r2'],   4),
                'MAE':        round(m['train_mae'],  4),
                'RMSE':       round(m['train_rmse'], 4),
                'WAPE':       round(m['train_wape'], 4),
            }
            for t, m in metrics.items()
        ]
        train_metrics_df = (
            pd.DataFrame(train_rows)
            .set_index('target')
            .sort_values('R²', ascending=False)
        )
        print("\n── SCM training metrics ────────────────────────────────")
        print(train_metrics_df.to_string())
        print("────────────────────────────────────────────────────────\n")

    return {
        'models': models,
        'target_feature_cols': target_feature_cols,
        'cat_col_categories': cat_col_categories,   # col → list of valid string categories
        'all_feature_cols': all_feature_cols,
        'target_cols': target_cols,
        'exog_cols': exog_cols,
        'causal_links': causal_links,
        'tau_max': tau_max,
        'df_train': df_train,
        'df_test': df_test,
        'metrics': metrics,
        'feature_mode': 'all' if use_all_features else 'causal',
    }


def simulate_scm(scm, mode='one_step'):
    """
    Predict targets over the test set.

    mode='one_step'    — one-step-ahead: at each test step lags are built from
                         real observed history (train tail + real test rows so far).
                         No error compounding. This is the honest evaluation mode.

    mode='autoregressive' — lags are built from previously predicted targets.
                            Errors compound over time but reflects true forecasting
                            when you have no future ground truth for the targets.

    In both modes exogenous variables (ef_*, ef_activity_log) always come from
    real test values — they are known inputs, not predicted.
    """
    models              = scm['models']
    target_feature_cols = scm['target_feature_cols']
    target_cols         = scm['target_cols']
    exog_cols           = scm['exog_cols']
    tau_max             = scm['tau_max']
    df_train            = scm['df_train']
    df_test             = scm['df_test']

    all_cols = target_cols + exog_cols

    # Real history: train tail + test (used for one_step lags on target cols)
    real_history = pd.concat([
        df_train[all_cols].tail(tau_max),
        df_test[all_cols]
    ]).reset_index(drop=True)
    n_warmup = tau_max  # index offset: real_history[n_warmup + i] == df_test row i

    # Autoregressive buffer seeded from train tail
    ar_buffer = df_train[all_cols].tail(tau_max).copy().reset_index(drop=True)

    cat_col_categories = scm['cat_col_categories']

    def _cast_activity(X, feat_cols):
        for c in feat_cols:
            if c in cat_col_categories:
                X[c] = pd.Categorical(X[c].astype(str), categories=cat_col_categories[c])
        return X

    sim_rows = []
    for step in range(len(df_test)):

        new_row = {}

        for target in target_cols:
            if target not in models:
                new_row[target] = ar_buffer[target].iloc[-1] if len(ar_buffer) > 0 else np.nan
                continue

            t_feat_cols = target_feature_cols[target]
            row_feats = {}

            for feat in t_feat_cols:
                # Parse "colname_lagN"
                for lag in range(1, tau_max + 1):
                    suffix = f'_lag{lag}'
                    if feat.endswith(suffix):
                        col = feat[:-len(suffix)]
                        break
                else:
                    row_feats[feat] = np.nan
                    continue

                is_exog = col in exog_cols or col == 'ef_activity_log'

                if mode == 'one_step' or is_exog:
                    # Always use real values for exog; use real history for targets in one_step
                    hist_idx = n_warmup + step - lag
                    row_feats[feat] = real_history[col].iloc[hist_idx] if hist_idx >= 0 else np.nan
                else:
                    # Autoregressive / trajectory: use predicted buffer for target lags
                    buf_idx = len(ar_buffer) - lag
                    row_feats[feat] = ar_buffer[col].iloc[buf_idx] if buf_idx >= 0 else np.nan

            X_step = pd.DataFrame([row_feats])
            X_step = _cast_activity(X_step, t_feat_cols)
            new_row[target] = float(models[target].predict(X_step)[0])

        # Exogenous cols always come from real test row
        for col in exog_cols:
            new_row[col] = df_test[col].iloc[step]

        sim_rows.append(new_row)

        # Update autoregressive buffer with predicted targets + real exog
        if mode == 'autoregressive':
            ar_buffer = pd.concat([ar_buffer, pd.DataFrame([new_row])], ignore_index=True)

    return pd.DataFrame(sim_rows, index=df_test.index)


def _compute_metrics(real_vals, pred_vals):
    valid = ~(np.isnan(real_vals) | np.isnan(pred_vals))
    rv, pv = real_vals[valid], pred_vals[valid]
    if valid.sum() > 1 and np.std(rv) > 0:
        denom = np.abs(rv).sum()
        return {
            'R²':      round(r2_score(rv, pv),                                          4),
            'MAE':     round(mean_absolute_error(rv, pv),                               4),
            'RMSE':    round(float(np.sqrt(np.mean((rv - pv) ** 2))),                   4),
            'WAPE':    round(float(np.abs(rv - pv).sum() / denom) if denom > 0 else np.nan, 4),
            'n_steps': int(valid.sum()),
        }
    return {'R²': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'WAPE': np.nan, 'n_steps': int(valid.sum())}


def plot_scm_simulation(scm, sim_one_step, sim_autoregressive,
                        scm_all=None, sim_all_one_step=None, sim_all_autoregressive=None):
    """
    One figure per target variable.

    Causal model: Real (blue), One-step (green dashed), Autoregressive (red dotted).
    All-features model (optional): One-step (orange dashed), Autoregressive (purple dotted).

    Vertical dotted lines mark activity transitions with the activity name.
    Followed by a side-by-side metrics table comparing all available modes.
    """
    target_cols  = [t for t in scm['target_cols'] if t in sim_one_step.columns]
    df_test      = scm['df_test']
    activity_col = 'ef_activity_log' if 'ef_activity_log' in df_test.columns else None

    has_all = scm_all is not None and sim_all_one_step is not None and sim_all_autoregressive is not None

    if not target_cols:
        print("[plot_scm_simulation] No target columns found.")
        return

    x = np.arange(len(df_test))
    activity_starts = []
    if activity_col:
        seg_ids = (df_test[activity_col] != df_test[activity_col].shift()).cumsum()
        for _, seg_index in seg_ids.groupby(seg_ids).groups.items():
            pos   = df_test.index.get_loc(seg_index[0])
            label = df_test.loc[seg_index[0], activity_col]
            activity_starts.append((pos, label))

    def _mstr(m):
        return f"R²={m['R²']:.3f}  MAE={m['MAE']:.3f}  RMSE={m['RMSE']:.3f}  WAPE={m['WAPE']:.3f}"

    def _get(sim, target, n):
        return sim[target].values if sim is not None and target in sim.columns else np.full(n, np.nan)

    n = len(df_test)

    # ── one figure per target ────────────────────────────────────────────────
    for target in target_cols:
        real_vals    = df_test[target].values if target in df_test.columns else np.full(n, np.nan)
        os_vals      = _get(sim_one_step,        target, n)
        ar_vals      = _get(sim_autoregressive,  target, n)
        all_os_vals  = _get(sim_all_one_step,    target, n)
        all_ar_vals  = _get(sim_all_autoregressive, target, n)

        m_os     = _compute_metrics(real_vals, os_vals)
        m_ar     = _compute_metrics(real_vals, ar_vals)
        m_all_os = _compute_metrics(real_vals, all_os_vals) if has_all else None
        m_all_ar = _compute_metrics(real_vals, all_ar_vals) if has_all else None

        fig, ax = plt.subplots(figsize=(16, 4))
        ax.plot(x, real_vals, label='Real',                                      color='steelblue', linewidth=1,   alpha=0.9)
        ax.plot(x, os_vals,   label=f'Causal One-step    {_mstr(m_os)}',         color='seagreen',  linewidth=1,   alpha=0.85, linestyle='--')
        ax.plot(x, ar_vals,   label=f'Causal Autoregress {_mstr(m_ar)}',         color='tomato',    linewidth=1,   alpha=0.85, linestyle=':')
        if has_all:
            ax.plot(x, all_os_vals, label=f'All One-step      {_mstr(m_all_os)}', color='darkorange', linewidth=1, alpha=0.85, linestyle='--')
            ax.plot(x, all_ar_vals, label=f'All Autoregress   {_mstr(m_all_ar)}', color='purple',     linewidth=1, alpha=0.85, linestyle=':')

        ymin, ymax = ax.get_ylim()
        for pos, label in activity_starts:
            ax.axvline(pos, color='gray', linewidth=0.5, alpha=0.5, linestyle=':')
            ax.text(pos + 0.5, ymax, label, fontsize=5, rotation=90,
                    va='top', ha='left', color='dimgray', clip_on=True)

        ax.set_title(target, fontsize=9)
        ax.set_xlabel('Test step')
        ax.legend(fontsize=6, loc='upper left')
        ax.tick_params(labelsize=7)
        plt.tight_layout()
        plt.show()

    # ── side-by-side metrics table ───────────────────────────────────────────
    rows = []
    for target in target_cols:
        real_all_vals = df_test[target].values if target in df_test.columns else np.full(n, np.nan)
        m_os     = _compute_metrics(real_all_vals, _get(sim_one_step,           target, n))
        m_ar     = _compute_metrics(real_all_vals, _get(sim_autoregressive,     target, n))
        row = {
            'target':         target,
            'causal_os_R²':   m_os['R²'],  'causal_os_MAE':  m_os['MAE'],  'causal_os_RMSE':  m_os['RMSE'],  'causal_os_WAPE':  m_os['WAPE'],
            'causal_ar_R²':   m_ar['R²'],  'causal_ar_MAE':  m_ar['MAE'],  'causal_ar_RMSE':  m_ar['RMSE'],  'causal_ar_WAPE':  m_ar['WAPE'],
        }
        if has_all:
            m_all_os = _compute_metrics(real_all_vals, _get(sim_all_one_step,       target, n))
            m_all_ar = _compute_metrics(real_all_vals, _get(sim_all_autoregressive, target, n))
            row.update({
                'all_os_R²':  m_all_os['R²'],  'all_os_MAE':  m_all_os['MAE'],  'all_os_RMSE':  m_all_os['RMSE'],  'all_os_WAPE':  m_all_os['WAPE'],
                'all_ar_R²':  m_all_ar['R²'],  'all_ar_MAE':  m_all_ar['MAE'],  'all_ar_RMSE':  m_all_ar['RMSE'],  'all_ar_WAPE':  m_all_ar['WAPE'],
            })
        rows.append(row)

    metrics_df = pd.DataFrame(rows).set_index('target').sort_values('causal_os_R²', ascending=False)
    header = "── Causal (causal_os/ar_*) vs All-features (all_os/ar_*) — test set ─" if has_all else "── One-step (causal_os_*) vs Autoregressive (causal_ar_*) — test set ─"
    print(f"\n{header}")
    print(metrics_df.to_string())
    print("─" * len(header) + "\n")
    display(metrics_df)
    return metrics_df

#%% Step 0 — train / test split by activity segment

display(df_clean)

df = df_clean.copy()

# relevant_columns = [
#         '(2)_zuluft_vor_entfeuchter_kon_g/kg_energy',
#        '(3)_zuluft_nach_entfeuchter_kon_g/kg_energy',
#     #    '(4)_frostschutz_%_energy', '(5)_vor_vent_hauptzuluft_temp_c_energy',
#     #    '(7)_zuluft_turmt_temp_c_energy', '(9)_abluft2_kon_g/kg_energy',
#     #    '(11)_wm_mas_kg/h_energy', '(12)_mpt_fb(mpt?)_kg/h_energy',
#     #    '(13)_lanzen_mas_kg/h_energy', '(14)_filter_mas_kg/h_energy',
#     #    '(15)_filter_mas_kg/h_energy', '(16)_konditionierung_mas_kg/h_energy',
#     #    '(8)_abluft_vol_m3/h_energy', '(23)_f10_speise_temp_c_energy',
#        '(23)_f11_speise_temp_c_energy',
#        '(19)_zuluft_vor_entfeuchter_temp_c_energy', 'dampf_nmb_energy',
#        'nach_nt_(c)_energy', 'f10_speise_kg/h_energy',
#     #    'f11_speise_kg/h_energy', 
#         'f10_speise_kg/m³_energy',
#     #    'f11_speise_kg/m³_energy', 'f10_speise_l/h_energy',
#     #    'f11_speise_l/h_energy', '(18)_leistung_turmF_lufterhitzer_kw_energy',
#     #    '(17)_leistung_turmF_luftentfeuchter_kw_energy',
#     #    '(6)_nach_recu_reg_temp_c_energy', 
#        #'id_original_energy',
#        '(10)_abluft_temp_c_energy', '(8)_abluft_mas_kg/h_energy',
#        'ef_temperature_2m_energy', 'ef_relative_humidity_2m_energy',
#        'ef_apparent_temperature_energy', #'ef_precipitation_energy',
#        #'ef_wind_speed_10m_energy', 'ef_wind_direction_100m_energy',
#        #'ef_global_tilted_irradiance_energy',
#        '(21)_zuluft_turm_mas_kg/h_energy', '(31)_q_waerme_recu_kw_energy',
#        '(22)_t_waermereg_c_energy',
#        '(29)_q_lufterwearmung_von_T6_nach_T7_kw_energy',
#        '(30)_q_lufterwearmung_brechenet_T5_T6_und_T6_T7_und_berechnete_recu_kw_energy',
#        '(33)_q_lufterwearmung_dampgemessen_und_berechnete_recu_kw_energy',
#        '(34)_q_lufterwearmung_berechnet_nach_temp_in_out_kw_energy',
#        'ef_activity_log', 'ef_activity_step']

relevant_columns = df.columns

df_relevant = df[relevant_columns].copy()

display(df_relevant)

#%% Step 0 — train / test split by activity segment

df_in = df_relevant.copy()

_activity_col = 'ef_activity_log' if 'ef_activity_log' in df_in.columns else None
if _activity_col:
    _seg_ids      = (df_in[_activity_col] != df_in[_activity_col].shift()).cumsum()
    _unique_segs  = _seg_ids.unique()
    _n_test       = min(SCM_N_TEST_ACTIVITIES, max(1, len(_unique_segs) // 5))
    _test_segs    = _unique_segs[-_n_test:]
    _test_mask    = _seg_ids.isin(_test_segs)
else:
    _cutoff    = int(len(df_in) * 0.8)
    _test_mask = pd.Series([False] * _cutoff + [True] * (len(df_in) - _cutoff), index=df_in.index)

df_train = df_in[~_test_mask].copy()
df_test  = df_in[_test_mask].copy()
report(f"[Split] Train: {len(df_train)} rows | Test: {len(df_test)} rows | "
       f"Test activities: {df_test[_activity_col].nunique() if _activity_col else 'N/A'}")

# Encode ef_activity_log as integer so it enters causal discovery as numeric.
# Fit the mapping on train only, apply to both splits.
# The original string column is kept as ef_activity_log; the encoded version
# is added as ef_activity_code (still ef-prefixed → parent-only in discovery).
if _activity_col and _activity_col in df_train.columns:
    _activity_categories = df_train[_activity_col].astype('category').cat.categories
    _activity_map = {v: i for i, v in enumerate(_activity_categories)}
    df_train['ef_activity_code'] = df_train[_activity_col].map(_activity_map).fillna(-1).astype(int)
    df_test['ef_activity_code']  = df_test[_activity_col].map(_activity_map).fillna(-1).astype(int)
    report(f"[Split] Activity encoding: {dict(list(_activity_map.items())[:10])}")

display(df_train)


#%% Step 1 — causal discovery (train split only, activity forced as parent)

pcmci_output = run_causal_discovery(
    dataset=df_train,
    method='pc',
    model_suffix='enegy',
    tau_max=SCM_TAU_MAX,
    pc_alpha=0.01,
    alpha_level=0.01,
    activity_col='ef_activity_code',
)

#%% Step 1b — causal features per target variable

_causal_rows = []
for _target, _parents in pcmci_output['causal_links'].items():
    if _parents:
        for _parent, _lag in _parents:
            _causal_rows.append({'target': _target, 'parent': _parent, 'lag': _lag})
    else:
        _causal_rows.append({'target': _target, 'parent': '(none found)', 'lag': None})

_causal_df = pd.DataFrame(_causal_rows).sort_values(['target', 'lag'])
display(_causal_df.reset_index(drop=True))

#%% Step 2 — train SCM using causal links from discovery

scm = train_scm(
    df_train,
    causal_output=pcmci_output,
    df_test=df_test,
    tau_max=SCM_TAU_MAX,
)

#%% Step 2b — train SCM with ALL features (baseline — no causal selection)

scm_all = train_scm(
    df_train,
    causal_output=pcmci_output,
    df_test=df_test,
    tau_max=SCM_TAU_MAX,
    use_all_features=True,
)

#%% Step 3 — run both simulation modes and compare

sim_one_step       = simulate_scm(scm, mode='one_step')
#sim_autoregressive = simulate_scm(scm, mode='autoregressive')

sim_all_one_step       = simulate_scm(scm_all, mode='one_step')
#sim_all_autoregressive = simulate_scm(scm_all, mode='autoregressive')

#%% Step 4 — visuals and metrics comparison

plot_scm_simulation(
    scm, sim_one_step,# sim_autoregressive,
    scm_all=scm_all,
    sim_all_one_step=sim_all_one_step,
    #sim_all_autoregressive=sim_all_autoregressive,
)

# %%