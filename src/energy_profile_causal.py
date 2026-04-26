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
process_datasets_to_model_sensors['process_4'] = process_datasets_to_model_sensors.get('process_4', {})
process_datasets_to_model_sensors['process_4']['objects_to_model'] = ['Erhitzer']
process_datasets_to_model_sensors['process_4']['activities_to_model'] = ['Step-032 = Umlauf', 'Step-030 = Produktion']
process_datasets_to_model_sensors['process_4']['sensors_to_model'] = ['temp_nach_WR2_(WT2)_5s_energy']

process_datasets_to_model_sensors['process_2'] = process_datasets_to_model_sensors.get('process_2', {})
process_datasets_to_model_sensors['process_2']['objects_to_model'] = ['l01']
process_datasets_to_model_sensors['process_2']['activities_to_model'] = ['Produktion']
process_datasets_to_model_sensors['process_2']['sensors_to_model'] = ['pro_volstrom_l/h_energy']

# process_datasets_to_model_sensors['process_3'] = process_datasets_to_model_sensors.get('process_3', {})
# process_datasets_to_model_sensors['process_3']['objects_to_model'] = ['tower_1']
# process_datasets_to_model_sensors['process_3']['activities_to_model'] = ['Produktion']
# process_datasets_to_model_sensors['process_3']['sensors_to_model'] = ['(8)_abluft_mas_kg/h_energy']

processes_to_run = ['process_3']
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
process_to_visualize_profiles = 'process_3'
#visualize_all_sensor_profiles(process_datasets, process_to_visualize_profiles)

#%%

display(process_datasets['process_3'])
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
    """Select model and parent-only variables for constrained causal discovery."""
    numeric_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
    ]

    suffix = model_suffix.lower()
    fallback = fallback_suffix.lower()

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

    parent_only_vars = [c for c in numeric_cols if c not in model_vars]
    return model_vars, parent_only_vars


#%%


def run_pcmci_causal_discovery(
    dataset,
    model_suffix='enegy',
    tau_max=6,
    pc_alpha=0.05,
    alpha_level=0.05,
    max_rows=30000,
):
    """
    Run constrained PCMCI causal discovery and plot a time-series causal graph.

    Constraint:
    - Non-model variables are parent-only (never shown as children).
    """
    if PCMCI is None or ParCorr is None or pp is None or tp is None:
        raise ImportError(
            "tigramite is not installed. Install with: pip install tigramite"
        )



    df = dataset.copy()
    model_vars, parent_only_vars = _select_pcmci_variables(df, model_suffix=model_suffix)

    if not model_vars:
        raise ValueError(
            "No model variables found. Check your suffix and column naming (example: *_energy)."
        )

    all_vars = model_vars + parent_only_vars
    df_model = df[all_vars].replace([np.inf, -np.inf], np.nan).dropna()

    if df_model.empty:
        raise ValueError("No valid rows left after dropping NaNs/Infs for PCMCI.")

    # Keep runtime manageable on dense sensor logs.
    if len(df_model) > max_rows:
        stride = max(1, len(df_model) // max_rows)
        df_model = df_model.iloc[::stride].copy()
        report(f"[PCMCI] Downsampled to {len(df_model)} rows (stride={stride}).")

    tig_df = pp.DataFrame(data=df_model.values, var_names=all_vars)
    pcmci = PCMCI(dataframe=tig_df, cond_ind_test=ParCorr())

    results = pcmci.run_pcmci(tau_min=1, tau_max=tau_max, pc_alpha=pc_alpha)

    q_matrix = pcmci.get_corrected_pvalues(
        p_matrix=results['p_matrix'],
        tau_max=tau_max,
        fdr_method='fdr_bh'
    )
    graph = q_matrix < alpha_level

    # Enforce parent-only rule: non-model variables can never be children.
    parent_only_idx = [i for i, v in enumerate(all_vars) if v in parent_only_vars]
    for j in parent_only_idx:
        graph[:, j, :] = False

    # Print strong links for quick inspection.
    report("\n[PCMCI] Significant links (after child-constraint):")
    pcmci.print_significant_links(
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
        f"PCMCI Causal Discovery | model_suffix='{model_suffix}'",
        fontsize=12,
        fontweight='bold'
    )
    plt.tight_layout()
    plt.show()

    return {
        'model_vars': model_vars,
        'parent_only_vars': parent_only_vars,
        'all_vars': all_vars,
        'results': results,
        'q_matrix': q_matrix,
        'graph': graph,
        'used_rows': len(df_model),
    }


#%%


df_proc = process_datasets['process_3'].get('expanded')




display(df_proc)

#%%

display(df_proc.columns)

columns = [
    #'datetime_energy', 
       '(2)_zuluft_vor_entfeuchter_kon_g/kg_energy',
       '(3)_zuluft_nach_entfeuchter_kon_g/kg_energy',
       '(4)_frostschutz_%_energy', '(5)_vor_vent_hauptzuluft_temp_c_energy',
       '(7)_zuluft_turmt_temp_c_energy', '(9)_abluft2_kon_g/kg_energy',
       '(11)_wm_mas_kg/h_energy', '(12)_mpt_fb(mpt?)_kg/h_energy',
       '(13)_lanzen_mas_kg/h_energy', '(14)_filter_mas_kg/h_energy',
       '(15)_filter_mas_kg/h_energy', '(16)_konditionierung_mas_kg/h_energy',
       '(8)_abluft_vol_m3/h_energy', '(23)_f10_speise_temp_c_energy',
       '(23)_f11_speise_temp_c_energy',
       '(19)_zuluft_vor_entfeuchter_temp_c_energy', 'dampf_nmb_energy',
       'nach_nt_(c)_energy', 'f10_speise_kg/h_energy',
       'f11_speise_kg/h_energy', 'f10_speise_kg/m³_energy',
       'f11_speise_kg/m³_energy', 'f10_speise_l/h_energy',
       'f11_speise_l/h_energy', '(6)_nach_recu_reg_temp_c_old_energy',
       '(18)_leistung_turmF_lufterhitzer_kw_energy',
       '(17)_leistung_turmF_luftentfeuchter_kw_energy',
       '(6)_nach_recu_reg_temp_c_energy', 'id_original_energy',
       '(10)_abluft_temp_c_energy', '(8)_abluft_mas_kg/h_energy',
       'ef_temperature_2m_energy', 
       'ef_relative_humidity_2m_energy',
       'ef_apparent_temperature_energy', 
       'ef_precipitation_energy',
       'ef_wind_speed_10m_energy', 
       'ef_wind_direction_100m_energy',
       'ef_global_tilted_irradiance_energy',
       '(21)_zuluft_turm_mas_kg/h_energy', '(31)_q_waerme_recu_kw_energy',
       '(22)_t_waermereg_c_energy',
       '(29)_q_lufterwearmung_von_T6_nach_T7_kw_energy',
       '(30)_q_lufterwearmung_brechenet_T5_T6_und_T6_T7_und_berechnete_recu_kw_energy',
       '(33)_q_lufterwearmung_dampgemessen_und_berechnete_recu_kw_energy',
       '(34)_q_lufterwearmung_berechnet_nach_temp_in_out_kw_energy',
       'activity_energy', 
       #'activity_log', 'timestamp_start_log',
       #'timestamp_end_log', 'object_attributes_log',
       #'higher_level_activity_log', 'object_type_log', 'object_log',
       #'case_id_log'
       ]




#%%

df = df_proc.copy()

df = df[columns]

df = df.reset_index()

df = df.rename(columns={'index': 'ef_index', 'activity_energy': 'ef_activity_energy'})

df = df.head(10000)

df_clean = df.copy()

display(df)

#%%

pcmci_output = run_pcmci_causal_discovery(
    dataset=df,
    model_suffix='enegy',
    tau_max=6,
    pc_alpha=0.05,
    alpha_level=0.05,
)

# %%
# ══════════════════════════════════════════════════════════════════════════════
# CAUSAL-INFORMED DTW CURVE PREDICTION
#
# Strategy
# --------
# PCMCI gives us a graph[i, j, tau] = True when variable i (at lag tau)
# significantly causes variable j.  For a chosen target sensor we extract its
# causal parents (variable + lag), then build a DTW-baseline pipeline where
# every canonical position row is enriched with the lag-shifted resampled
# values of those parents.  Only causally confirmed links are used —
# no noise from irrelevant ef_ columns.
# ══════════════════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from tslearn.barycenters import dtw_barycenter_averaging
from dtw import dtw as _dtw


def _dtw_align(query, reference):
    """DTW-align query onto reference grid, returns aligned array of len(reference)."""
    alignment = _dtw(query, reference, keep_internals=True)
    aligned = np.zeros(len(reference))
    counts  = np.zeros(len(reference))
    for qi, ri in zip(alignment.index1, alignment.index2):
        aligned[ri] += query[qi]
        counts[ri]  += 1
    counts = np.where(counts == 0, 1, counts)
    return aligned / counts


def _resample(arr, n):
    """Linear resample arr to length n."""
    arr = np.asarray(arr, dtype=float).flatten()
    arr = arr[np.isfinite(arr)] if not np.all(np.isfinite(arr)) else arr
    if len(arr) == 0:
        return np.zeros(n)
    if len(arr) == n:
        return arr
    if len(arr) < 2:
        return np.full(n, arr[0])
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(arr)), arr)


def extract_causal_parents(pcmci_out, target_col):
    """
    Extract significant causal parents of target_col from PCMCI output.

    Parameters
    ----------
    pcmci_out : dict   — returned by run_pcmci_causal_discovery()
    target_col : str   — name of the target variable

    Returns
    -------
    parents : list of (col_name: str, lag: int, strength: float)
        Sorted by |strength| descending.  Lag is the number of time steps
        the parent precedes the target (positive integer).
    """
    all_vars = pcmci_out['all_vars']
    graph    = pcmci_out['graph']      # shape (n_vars, n_vars, tau_max+1)
    val_mat  = pcmci_out['results']['val_matrix']

    if target_col not in all_vars:
        raise ValueError(f"'{target_col}' not in PCMCI variables: {all_vars}")

    j = all_vars.index(target_col)
    parents = []
    for i, col in enumerate(all_vars):
        for tau in range(1, graph.shape[2]):   # tau=0 is contemporaneous, skip
            if graph[i, j, tau]:
                parents.append((col, tau, float(val_mat[i, j, tau])))

    parents.sort(key=lambda x: abs(x[2]), reverse=True)
    return parents


def split_curves_causal(df_expanded, variable, activities, objects,
                         causal_parents, test_size=0.15, random_state=42):
    """
    Like split_curves but stores causal-parent time series per curve.

    Each returned curve dict has an extra key:
        'parent_values': {(col, lag): np.ndarray}
            Raw time series for each parent, same length as original_values.
            The lag is stored as metadata; the actual shifting is applied
            during feature-matrix construction so the arrays stay raw here.

    Parameters
    ----------
    causal_parents : list of (col, lag, strength)
        From extract_causal_parents().
    """
    parent_cols = [col for col, _lag, _s in causal_parents]
    # Keep only parents whose columns actually exist in df_expanded
    present = [
        (col, lag, s)
        for col, lag, s in causal_parents
        if col in df_expanded.columns
    ]
    if not present:
        report("[CausalDTW] No causal parent columns found in df_expanded — "
               "falling back to baseline (no parent features).")

    keep_cols = [
        'case_id_log', 'activity_log', 'object_log', 'timestamp_start_log',
        'datetime_energy', variable, 'object_attributes_log',
    ] + [col for col, _, _ in present if col not in
         ['case_id_log','activity_log','object_log','timestamp_start_log',
          'datetime_energy', variable, 'object_attributes_log']]

    df = df_expanded[[c for c in keep_cols if c in df_expanded.columns]].copy()
    df = df[df['object_log'].isin(objects)]
    df = df[df['activity_log'].isin(activities)]
    df['datetime_energy'] = pd.to_datetime(df['datetime_energy'])
    df = df.sort_values(['case_id_log', 'timestamp_start_log', 'datetime_energy'])

    df['_iid'] = df.groupby(
        ['case_id_log', 'object_log', 'activity_log', 'timestamp_start_log']
    ).ngroup()

    curves = []
    for iid, grp in df.groupby('_iid'):
        grp    = grp.sort_values('datetime_energy').reset_index(drop=True)
        values = grp[variable].dropna().values
        if len(values) < 5:
            continue
        attrs = grp['object_attributes_log'].iloc[0] \
            if 'object_attributes_log' in grp.columns else {}

        parent_values = {}
        for col, lag, _ in present:
            if col in grp.columns:
                raw_col = grp[col]
                # Guard against duplicate columns returning a DataFrame
                if isinstance(raw_col, pd.DataFrame):
                    raw_col = raw_col.iloc[:, 0]
                parent_values[(col, lag)] = (
                    pd.to_numeric(raw_col, errors='coerce')
                    .fillna(0.0).values.astype(float).flatten()
                )

        curves.append({
            'instance_id':     iid,
            'activity':        grp['activity_log'].iloc[0],
            'original_values': values,
            'original_length': len(values),
            'attributes':      attrs,
            'parent_values':   parent_values,
        })

    all_ids = list(range(len(curves)))
    if test_size <= 0:
        train_ids, test_ids = all_ids, []
    elif test_size >= 1.0:
        train_ids, test_ids = [], all_ids
    else:
        train_ids, test_ids = train_test_split(
            all_ids, test_size=test_size, random_state=random_state
        )
    return [curves[i] for i in train_ids], [curves[i] for i in test_ids]


def build_and_train_pipeline_causal_dtw(
    train_curves,
    variable,
    causal_parents,
    fixed_length=100,
    val_size=0.2,
    random_state=42,
    models=None,
    verbose=True,
):
    """
    DTW pipeline enriched with PCMCI-selected causal parents at their discovered lags.

    For every canonical position p (0..fixed_length-1) the feature row contains:
        position_idx, curve_length, activity, <scalar attributes>,
        <parent_col>_lag<tau>  for each (col, tau) in causal_parents

    The parent signal is resampled to fixed_length and then accessed at
    position max(0, p - tau) to encode the lead time.

    Parameters
    ----------
    train_curves   : list[dict]  — from split_curves_causal()
    variable       : str
    causal_parents : list of (col, lag, strength)
    fixed_length   : int
    val_size       : float
    random_state   : int
    models         : dict {name: ModelClass}
    verbose        : bool

    Returns
    -------
    pipeline dict (same structure as build_and_train_pipeline + 'causal_parents', 'approach')
    """
    if models is None:
        models = {
            'Linear Regression':  LinearRegression,
            'Gradient Boosting':  GradientBoostingRegressor,
            'Random Forest':      RandomForestRegressor,
        }

    # Filter to parents that appear in at least one curve
    present_keys = {k for c in train_curves for k in c.get('parent_values', {}).keys()}
    active_parents = [(col, lag, s) for col, lag, s in causal_parents
                      if (col, lag) in present_keys]

    if verbose:
        report(f"\n[CausalDTW] Training with {len(active_parents)} causal parents:")
        for col, lag, s in active_parents:
            report(f"  {col}  lag={lag}  strength={s:.3f}")

    # ── DBA barycenter ────────────────────────────────────────────────────────
    resampled_for_dba = np.array([
        _resample(c['original_values'], fixed_length) for c in train_curves
    ])[:, :, np.newaxis]

    dba_bary = dtw_barycenter_averaging(resampled_for_dba, barycenter_size=fixed_length)
    reference_curve = dba_bary[:, 0]

    # ── DTW-align train curves ────────────────────────────────────────────────
    for c in train_curves:
        c['resampled_values'] = _dtw_align(c['original_values'], reference_curve)
        # Pre-resample parent signals to fixed_length (raw, no lag yet)
        c['_parent_rs'] = {
            (col, lag): _resample(c['parent_values'][(col, lag)], fixed_length)
            for col, lag, _ in active_parents
            if (col, lag) in c.get('parent_values', {})
        }

    # ── Attribute key types ───────────────────────────────────────────────────
    all_keys = sorted({k for c in train_curves for k in c['attributes'].keys()})
    key_types = {}
    for key in all_keys:
        is_num = True
        for c in train_curves:
            if key in c['attributes']:
                try:
                    float(c['attributes'][key])
                except (ValueError, TypeError):
                    is_num = False
                    break
        key_types[key] = 'numeric' if is_num else 'category'

    # Parent feature column names (deterministic order)
    parent_feat_names = [f"{col}__lag{lag}" for col, lag, _ in active_parents]

    # ── Build feature matrix ──────────────────────────────────────────────────
    rows = []
    for c in train_curves:
        prs = c.get('_parent_rs', {})
        for pos in range(fixed_length):
            row = {
                'instance_id':  c['instance_id'],
                'activity':     c['activity'],
                'position_idx': pos,
                'curve_length': c['original_length'],
                'y':            c['resampled_values'][pos],
            }
            for key in all_keys:
                val = c['attributes'].get(key, None)
                if key_types[key] == 'numeric':
                    try:
                        row[key] = float(val) if val is not None else np.nan
                    except (ValueError, TypeError):
                        row[key] = np.nan
                else:
                    row[key] = str(val) if val is not None else 'None'
            for (col, lag, _), fname in zip(active_parents, parent_feat_names):
                sig = prs.get((col, lag))
                # Access lag-shifted position: what the parent looked like `lag` steps ago
                src_pos = max(0, pos - lag)
                row[fname] = float(sig[src_pos]) if sig is not None else np.nan
            rows.append(row)

    df_reg = pd.DataFrame(rows)
    categorical_cols = ['activity'] + [k for k in all_keys if key_types[k] == 'category']
    X_all = df_reg[['position_idx', 'curve_length']].copy()
    X_all = X_all.assign(activity=df_reg['activity'].values)
    for key in all_keys:
        X_all[key] = df_reg[key].values
    for fname in parent_feat_names:
        X_all[fname] = df_reg[fname].values
    X_all = pd.get_dummies(X_all, columns=categorical_cols, drop_first=True)
    y_all = df_reg['y'].copy()

    unique_inst = df_reg['instance_id'].unique()
    tr_inst, va_inst = train_test_split(unique_inst, test_size=val_size,
                                         random_state=random_state)
    tr_mask = df_reg['instance_id'].isin(tr_inst)
    va_mask  = df_reg['instance_id'].isin(va_inst)
    X_train, X_val = X_all[tr_mask].copy(), X_all[va_mask].copy()
    y_train, y_val = y_all[tr_mask].copy(), y_all[va_mask].copy()

    feature_columns = X_all.columns.tolist()
    numeric_feat_cols = (
        ['position_idx', 'curve_length']
        + [k for k in all_keys if key_types[k] == 'numeric']
        + parent_feat_names
    )
    numeric_feat_cols = [c for c in numeric_feat_cols if c in X_train.columns]

    scaler = None
    if numeric_feat_cols:
        scaler = StandardScaler()
        X_train.loc[:, numeric_feat_cols] = scaler.fit_transform(X_train[numeric_feat_cols])
        X_val.loc[:,   numeric_feat_cols] = scaler.transform(X_val[numeric_feat_cols])

    # ── Train models ──────────────────────────────────────────────────────────
    best_model, best_name, best_r2 = None, None, -np.inf
    all_results = {}

    for name, cls in models.items():
        mdl = cls()
        mdl.fit(X_train, y_train)
        tr_r2 = float(r2_score(y_train, mdl.predict(X_train)))
        va_r2 = float(r2_score(y_val,   mdl.predict(X_val)))
        all_results[name] = {'train_r2': tr_r2, 'val_r2': va_r2}
        if verbose:
            report(f"  {name:25}  train R²={tr_r2:.4f}  val R²={va_r2:.4f}")
        if va_r2 > best_r2:
            best_r2, best_model, best_name = va_r2, mdl, name

    report(f"\n[CausalDTW] Best: {best_name}  val R²={best_r2:.4f}")

    return {
        'model':               best_model,
        'model_name':          best_name,
        'reference_curve':     reference_curve,
        'fixed_length':        fixed_length,
        'all_keys':            all_keys,
        'key_types':           key_types,
        'feature_columns':     feature_columns,
        'feature_scaler':      scaler,
        'numeric_feature_cols': numeric_feat_cols,
        'val_r2':              best_r2,
        'all_results':         all_results,
        'causal_parents':      active_parents,
        'parent_feat_names':   parent_feat_names,
        'approach':            'causal_dtw',
    }


def predict_raw_curve_causal_dtw(raw_values, activity, attributes, pipeline,
                                  parent_signals=None):
    """
    Predict a single curve using the causal-DTW pipeline.

    Parameters
    ----------
    raw_values     : np.ndarray
    activity       : str
    attributes     : dict
    pipeline       : dict  — from build_and_train_pipeline_causal_dtw()
    parent_signals : dict | None
        {(col, lag): np.ndarray}  — raw time series of each causal parent
        for this activity window.  Missing keys are filled with zeros.
    """
    reference_curve      = pipeline['reference_curve']
    fixed_length         = len(reference_curve)
    model                = pipeline['model']
    feature_columns      = pipeline['feature_columns']
    scaler               = pipeline.get('feature_scaler')
    numeric_feat_cols    = pipeline.get('numeric_feature_cols', [])
    all_keys             = pipeline['all_keys']
    key_types            = pipeline['key_types']
    active_parents       = pipeline['causal_parents']
    parent_feat_names    = pipeline['parent_feat_names']

    if parent_signals is None:
        parent_signals = {}

    # Resample each parent signal to fixed_length
    parent_rs = {}
    for col, lag, _ in active_parents:
        sig = parent_signals.get((col, lag))
        if sig is not None:
            parent_rs[(col, lag)] = _resample(sig, fixed_length)
        else:
            parent_rs[(col, lag)] = np.zeros(fixed_length)

    # Build feature rows for all canonical positions
    rows = []
    for pos in range(fixed_length):
        row = {
            'position_idx': pos,
            'curve_length': len(raw_values),
            'activity':     activity,
        }
        for key in all_keys:
            val = attributes.get(key, None)
            if key_types[key] == 'numeric':
                try:
                    row[key] = float(val) if val is not None else np.nan
                except (ValueError, TypeError):
                    row[key] = np.nan
            else:
                row[key] = str(val) if val is not None else 'None'
        for (col, lag, _), fname in zip(active_parents, parent_feat_names):
            src_pos = max(0, pos - lag)
            row[fname] = float(parent_rs[(col, lag)][src_pos])
        rows.append(row)

    X_ref = pd.DataFrame(rows)
    categorical_cols = ['activity'] + [k for k in all_keys if key_types[k] == 'category']
    X_ref = pd.get_dummies(X_ref, columns=categorical_cols, drop_first=True)
    for col in feature_columns:
        if col not in X_ref.columns:
            X_ref[col] = 0
    X_ref = X_ref[feature_columns]

    if scaler is not None and numeric_feat_cols:
        cols_to_scale = [c for c in numeric_feat_cols if c in X_ref.columns]
        if cols_to_scale:
            X_ref.loc[:, cols_to_scale] = scaler.transform(X_ref[cols_to_scale])

    y_ref_pred = model.predict(X_ref)

    # DTW decode: canonical predictions → raw timeline
    alignment = _dtw(raw_values, reference_curve, keep_internals=True)
    buckets   = [[] for _ in range(len(raw_values))]
    path_pairs = list(zip(alignment.index1, alignment.index2))
    for qi, ri in path_pairs:
        if 0 <= qi < len(raw_values) and 0 <= ri < fixed_length:
            buckets[qi].append(y_ref_pred[ri])

    y_raw = np.empty(len(raw_values), dtype=float)
    for qi in range(len(raw_values)):
        if buckets[qi]:
            y_raw[qi] = float(np.mean(buckets[qi]))
        else:
            nearest = min(path_pairs, key=lambda p: abs(p[0] - qi))
            y_raw[qi] = float(y_ref_pred[min(max(nearest[1], 0), fixed_length - 1)])

    return y_raw


# ──────────────────────────────────────────────────────────────────────────────
# EVALUATION — Causal-DTW vs Baseline
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_causal_dtw(test_curves, pipeline, parent_col_map=None):
    """
    Evaluate causal-DTW pipeline on test curves.

    Parameters
    ----------
    test_curves    : list[dict]  — from split_curves_causal(), test split
    pipeline       : dict        — from build_and_train_pipeline_causal_dtw()
    parent_col_map : ignored (parent_values already in curve dicts)

    Returns
    -------
    metrics_df : pd.DataFrame  per-curve metrics
    agg        : dict          aggregate metrics
    """
    per_curve = []
    all_true, all_pred = [], []

    for c in test_curves:
        raw   = c['original_values']
        y_pred = predict_raw_curve_causal_dtw(
            raw, c['activity'], c['attributes'], pipeline,
            parent_signals=c.get('parent_values', {}),
        )
        mae  = float(mean_absolute_error(raw, y_pred))
        rmse = float(np.sqrt(mean_squared_error(raw, y_pred)))
        r2   = float(r2_score(raw, y_pred))
        denom = np.sum(np.abs(raw))
        wape = float(np.sum(np.abs(raw - y_pred)) / denom * 100) if denom > 0 else np.nan

        per_curve.append({
            'instance_id': c['instance_id'],
            'activity':    c['activity'],
            'n_points':    len(raw),
            'MAE':  mae, 'RMSE': rmse, 'WAPE (%)': wape, 'R2': r2,
        })
        all_true.extend(raw.tolist())
        all_pred.extend(y_pred.tolist())

    all_true  = np.array(all_true)
    all_pred  = np.array(all_pred)
    denom_agg = np.sum(np.abs(all_true))
    agg = {
        'MAE':      float(mean_absolute_error(all_true, all_pred)),
        'RMSE':     float(np.sqrt(mean_squared_error(all_true, all_pred))),
        'WAPE (%)': float(np.sum(np.abs(all_true - all_pred)) / denom_agg * 100)
                    if denom_agg > 0 else np.nan,
        'R2':       float(r2_score(all_true, all_pred)),
    }
    return pd.DataFrame(per_curve), agg


# ──────────────────────────────────────────────────────────────────────────────
# RUN — pick a target sensor, extract parents, train, compare
# ──────────────────────────────────────────────────────────────────────────────

#%%

_TARGET_SENSOR = '(8)_abluft_mas_kg/h_energy'

# Make sure pcmci_output exists (ran above)
_causal_parents = extract_causal_parents(pcmci_output, _TARGET_SENSOR)
report(f"\nCausal parents of '{_TARGET_SENSOR}':")
for col, lag, s in _causal_parents:
    report(f"  {col}  lag={lag}  strength={s:.4f}")

#%%

# Use process_3 expanded data
_df_exp = process_datasets['process_3']['expanded']

_activities = _df_exp['activity_log'].dropna().unique().tolist() \
    if 'activity_log' in _df_exp.columns else []
_objects    = _df_exp['object_log'].dropna().unique().tolist() \
    if 'object_log' in _df_exp.columns else []

report(f"\nActivities: {_activities}")
report(f"Objects   : {_objects}")

#%%

_train_causal, _test_causal = split_curves_causal(
    _df_exp,
    variable=_TARGET_SENSOR,
    activities=_activities,
    objects=_objects,
    causal_parents=_causal_parents,
    test_size=0.20,
    random_state=42,
)
report(f"\nTrain curves: {len(_train_causal)}  Test curves: {len(_test_causal)}")

#%%

# ── Train causal-DTW pipeline ─────────────────────────────────────────────────
_CAUSAL_MODELS = {
    'Linear Regression': LinearRegression,
    'Gradient Boosting': GradientBoostingRegressor,
    'Random Forest':     RandomForestRegressor,
}

_pipeline_causal = build_and_train_pipeline_causal_dtw(
    _train_causal,
    variable=_TARGET_SENSOR,
    causal_parents=_causal_parents,
    fixed_length=100,
    val_size=0.2,
    models=_CAUSAL_MODELS,
    verbose=True,
)

#%%

# ── Train baseline DTW pipeline on the same curves for a fair comparison ─────
from sim_extractor import build_and_train_pipeline, predict_raw_curve

# Re-split without parent data so split_curves signature is satisfied
_train_base_curves = [
    {k: v for k, v in c.items() if k != 'parent_values'}
    for c in _train_causal
]
_test_base_curves  = [
    {k: v for k, v in c.items() if k != 'parent_values'}
    for c in _test_causal
]

_pipeline_base = build_and_train_pipeline(
    _train_base_curves,
    variable=_TARGET_SENSOR,
    fixed_length=100,
    val_size=0.2,
    models={
        'Linear Regression': LinearRegression,
        'Gradient Boosting': GradientBoostingRegressor,
    },
    verbose=False,
)
report(f"\n[Baseline] Best: {_pipeline_base['model_name']}  "
       f"val R²={_pipeline_base['val_r2']:.4f}")

#%%

# ── Evaluate both on the held-out test curves ─────────────────────────────────
_metrics_causal, _agg_causal = evaluate_causal_dtw(_test_causal, _pipeline_causal)
report(f"\n=== Causal-DTW TEST results ===")
report(f"  MAE={_agg_causal['MAE']:.4f}  RMSE={_agg_causal['RMSE']:.4f}  "
       f"WAPE={_agg_causal['WAPE (%)']:.2f}%  R²={_agg_causal['R2']:.4f}")

# Evaluate baseline on same test curves
from sim_extractor import evaluate_pipeline_on_test as _eval_base
_metrics_base_df, _agg_base = _eval_base(
    _test_base_curves, _pipeline_base, max_plot_curves=0, verbose=False,
)
report(f"\n=== Baseline DTW TEST results ===")
report(f"  MAE={_agg_base['MAE']:.4f}  RMSE={_agg_base['RMSE']:.4f}  "
       f"WAPE={_agg_base['WAPE (%)']:.2f}%  R²={_agg_base['R2']:.4f}")

#%%

# ── Side-by-side comparison table ────────────────────────────────────────────
_compare_rows = []
for _act in sorted(set(_metrics_causal['activity'].unique()) |
                   set(_metrics_base_df['activity'].unique())):
    _c = _metrics_causal[_metrics_causal['activity'] == _act]
    _b = _metrics_base_df[_metrics_base_df['activity'] == _act]
    _compare_rows.append({
        'Activity':          _act,
        'Causal-DTW  R²':   round(_c['R2'].mean(),       4) if len(_c) else np.nan,
        'Baseline    R²':   round(_b['R2'].mean(),       4) if len(_b) else np.nan,
        'Causal-DTW  MAE':  round(_c['MAE'].mean(),      4) if len(_c) else np.nan,
        'Baseline    MAE':  round(_b['MAE'].mean(),      4) if len(_b) else np.nan,
        'Causal-DTW  WAPE': round(_c['WAPE (%)'].mean(), 2) if len(_c) else np.nan,
        'Baseline    WAPE': round(_b['WAPE (%)'].mean(), 2) if len(_b) else np.nan,
    })

_compare_df = pd.DataFrame(_compare_rows)
display(Markdown("## Causal-DTW vs Baseline — TEST set"))
display(_compare_df)

#%%

# ── ΔR² bar chart: Causal-DTW minus Baseline per activity ────────────────────
import matplotlib.pyplot as plt
import seaborn as sns

_compare_df['ΔR² (Causal − Baseline)'] = (
    _compare_df['Causal-DTW  R²'] - _compare_df['Baseline    R²']
)

fig, ax = plt.subplots(figsize=(max(6, len(_compare_df) * 1.4), 4))
colors = ['#2ecc71' if v >= 0 else '#e74c3c'
          for v in _compare_df['ΔR² (Causal − Baseline)']]
ax.bar(_compare_df['Activity'], _compare_df['ΔR² (Causal − Baseline)'],
       color=colors, edgecolor='black', linewidth=0.6)
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_title(f'ΔR² Causal-DTW vs Baseline — {_TARGET_SENSOR}',
             fontsize=11, fontweight='bold')
ax.set_xlabel('Activity')
ax.set_ylabel('ΔR²  (green = Causal better)')
ax.tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.show()

#%%

# ── Overlay plot: predicted vs actual for a few test curves ──────────────────
_n_show = min(6, len(_test_causal))
_sample  = _test_causal[:_n_show]

fig, axes = plt.subplots(2, max(1, _n_show // 2), figsize=(14, 6))
axes = np.array(axes).flatten()

for ax, c in zip(axes[:_n_show], _sample):
    raw = c['original_values']
    y_causal = predict_raw_curve_causal_dtw(
        raw, c['activity'], c['attributes'], _pipeline_causal,
        parent_signals=c.get('parent_values', {}),
    )
    y_base = predict_raw_curve(raw, c['activity'], c['attributes'], _pipeline_base)

    ax.plot(raw,      label='Real',       color='black',   linewidth=1.5)
    ax.plot(y_causal, label='Causal-DTW', color='#2ecc71', linewidth=1.2, linestyle='--')
    ax.plot(y_base,   label='Baseline',   color='#e74c3c', linewidth=1.0, linestyle=':')
    ax.set_title(f"{c['activity']} | id={c['instance_id']}", fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

for ax in axes[_n_show:]:
    ax.set_visible(False)

plt.suptitle(f'Causal-DTW vs Baseline — {_TARGET_SENSOR}',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.show()
# %%
