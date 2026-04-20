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

process_datasets_to_model_sensors['process_3'] = process_datasets_to_model_sensors.get('process_3', {})
process_datasets_to_model_sensors['process_3']['objects_to_model'] = ['tower_1']
process_datasets_to_model_sensors['process_3']['activities_to_model'] = ['Produktion']
process_datasets_to_model_sensors['process_3']['sensors_to_model'] = ['(8)_abluft_mas_kg/h_energy']

processes_to_run = ['process_3']
# Filter the original dictionary
process_datasets_to_model = {
    k: v for k, v in process_datasets.items() 
    if k in processes_to_run
}
process_datasets_to_model_sensors = {
    k: v for k, v in process_datasets_to_model_sensors.items() 
    if k in processes_to_run
}
report(f"[VERIFY] Processes to run: {processes_to_run}")
report(f"[VERIFY] Available in dataset: {list(process_datasets.keys())}")
if processes_to_run[0] in process_datasets:
    p = processes_to_run[0]
    report(f"[VERIFY] Dataset '{p}' has expanded: {'expanded' in process_datasets[p]}")
    if 'expanded' in process_datasets[p]:
        report(f"[VERIFY] Expanded columns: {list(process_datasets[p]['expanded'].columns[:10])}")

# %%
import pandas as pd
from sim_extractor import extract_process, build_energy_features_for_raw_df
from simulation import ProcessSimulation
from sim_modeller import SimModeller

# ─────────────────────────────────────────────────────────────────────────────
# ENERGY MODIFIER CONFIGURATION
#   Runs after the energy modelling section (predict_raw_curve etc).
#   Configure which sklearn estimator to use for each modifier.
#   Both modifiers accept any estimator with .fit() and .predict()/.predict_proba().
# ─────────────────────────────────────────────────────────────────────────────
from sklearn.linear_model import Lasso, LogisticRegression
from xgboost import XGBRegressor

# ── Duration modifier models ───────────────────────────────────────────────
# List of sklearn-compatible regressor types to compete per activity
ENERGY_DURATION_MODELS    = ['xgboost', 'linear', 'lasso', 'mlp']

# ── Transition modifier models ─────────────────────────────────────────────
# List of classifier types to compete per activity
ENERGY_TRANSITION_MODELS  = ['logistic', 'random_forest', 'gradient_boosting']

ENERGY_DURATION_SCALE_CLIP = (0.7, 1.3)   # max ±30% shift per activity
ENERGY_LOGIT_BIAS_CLIP     = (-1.0, 1.0)  # max ~2.7× odds-ratio shift per competing activity
ENERGY_MIN_SAMPLES         = 1           # STRICT: skip ML (use statistical) if n_samples < 30

# Will be populated per process after energy modelling:
energy_modifiers_by_process = {}
energy_pipelines_by_process  = {}  # store dynamic curves for simulation

# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION MODE TOGGLE
#   'statistical'   – sample durations/transitions from best-fit distributions
#   'ml_no_energy'  – use ML models with log features
#   'ml_with_energy'– use ML models with log + energy features
# ─────────────────────────────────────────────────────────────────────────────
SIMULATION_MODE = 'ml_no_energy'

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
# ML MODEL CONFIGURATION (only used when SIMULATION_MODE matches an 'ml_*' mode)
#   model_types: list of models to train — best is selected per activity key
#                Supported: 'xgboost', 'linear', 'lasso', 'mlp'
#   optimize_hyperparams: True  → Optuna hyper-parameter search
#                         False → use default model parameters
#   n_optuna_trials: number of Optuna trials per model (ignored if optimize=False)
# ─────────────────────────────────────────────────────────────────────────────
ML_MODEL_TYPES          = ['xgboost', 'linear', 'lasso', 'mlp']  # ← train all, pick best
ML_MODEL_TYPES          = ['xgboost', 'mean', 'median']
ML_OPTIMIZE_HYPERPARAMS = False    # ← set True to enable Optuna tuning
ML_OPTUNA_TRIALS        = 20

# ─────────────────────────────────────────────────────────────────────────────
# TEMPORAL TRAIN / TEST SPLIT (at the pipeline level, BEFORE extraction)
#   TEMPORAL_SPLIT : True  → split all data by case start time (no leakage)
#                    False → use all data (no split, single evaluation)
#   TRAIN_RATIO    : fraction of cases used for training (e.g. 0.80 = 80%)
# ─────────────────────────────────────────────────────────────────────────────
TEMPORAL_SPLIT     = True
TRAIN_RATIO        = 0.80
RUN_TEST_EVALUATION = True # Fast mode: skip heavy test-set simulation & curve extraction



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



# ─────────────────────────────────────────────────────────────────────────────
# MODES TO COMPARE
#   We run BOTH modes on every process so results appear side by side.
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION MODES
#   'statistical'       - Purely stochastic (distributions + weights)
#   'petri_net_*'       - Petri net token game (stochastic)
#   'ml_no_energy'      - Unified ML prediction with log features
#   'ml_with_energy'    - Unified ML prediction with log + energy features
# ─────────────────────────────────────────────────────────────────────────────
MODES_TO_COMPARE = [
    'statistical',
    'petri_net_alpha',
    'petri_net_heuristic',
    'petri_net_inductive',
    'ml_no_energy',
    'ml_with_energy',
]

# Initialize a list to store results for each process × mode
evaluation_results_list = []
# Initialize a list to store results for each process × mode
evaluation_results_list = []

for process in process_datasets_to_model.keys():
    report(f"\n[DEBUG] === STARTING PIPELINE FOR PROCESS: {process} ===")
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
            
            try:
                title_str = f"PETRI NET VIEW: {obj_name} ({obj_type}) | Process: {process} | Alg: {alg_name}"
                report("\n" + "="*len(title_str))
                report(title_str)
                report("="*len(title_str) + "\n")
                display(Markdown(f"### 🌐 Petri Net: {obj_name} ({obj_type})"))
                display(Markdown(f"*Process: {process} | Mining Algorithm: {alg_name}*"))
                
                import pm4py
                view_obj = pm4py.view_petri_net(model['net'], model['im'], model['fm'], format='png')
                if view_obj:
                        display(view_obj)
            except Exception as e:
                print(f"    [!] Petri net visualization failed for {obj_name}: {e}")
    # ── Detect sensor columns ──────────────────────────────────────────────
    df_expanded_proc = train_datasets[process].get("expanded")
    report(f"\n[DEBUG] Process: {process}")
    if df_expanded_proc is None:
        report(f"[DEBUG]  (!) No \"expanded\" dataframe found for {process}")
    else:
        report(f"[DEBUG]  Found \"expanded\" df with {len(df_expanded_proc)} rows. Columns: {list(df_expanded_proc.columns[:5])}...")
    
    sensors = []
    cfg_dict = globals().get("process_datasets_to_model_sensors", {})
    sensors = cfg_dict.get(process, {}).get("sensors_to_model", [])
    
    if sensors:
        report(f"[DEBUG]  Used sensors from CONFIG: {sensors}")
    elif df_expanded_proc is not None:
        sensors = [c for c in df_expanded_proc.columns if (c.endswith("_energy") or "_energy_" in c or "power" in c.lower()) 
                   and not c.endswith("_log") and df_expanded_proc[c].dtype in ("float64", "float32", "int64")]
        report(f"[DEBUG]  Detected sensors via FALLBACK: {sensors}")
    
    if not sensors:
        report(f"[DEBUG]  [!] FINAL SENSOR LIST IS EMPTY")


    # ── Train ML models ───────────────────────────────────────────────────
    ml_models_no_energy = None
    ml_models_with_energy = None
    
    print("\n" + "="*50)
    print(f"TRAINING ML MODELS (types={ML_MODEL_TYPES})")
    print("="*50)

    # 1. Base ML models (no energy features)
    ml_models_no_energy = SimModeller(
        model_types=ML_MODEL_TYPES,
        optimize_hyperparams=ML_OPTIMIZE_HYPERPARAMS,
        n_optuna_trials=ML_OPTUNA_TRIALS,
    )
    ml_models_no_energy.train(raw_df, activity_stats_df)
    report("✓ Trained [ml_no_energy] models.")

    # 2. Energy-aware ML models
    if sensors and df_expanded_proc is not None:
        try:
            from sim_extractor import build_energy_features_for_raw_df
            raw_df_enriched = build_energy_features_for_raw_df(raw_df, df_expanded_proc, sensors)
            ml_models_with_energy = SimModeller(
                model_types=ML_MODEL_TYPES,
                optimize_hyperparams=ML_OPTIMIZE_HYPERPARAMS,
                n_optuna_trials=ML_OPTUNA_TRIALS,
            )
            ml_models_with_energy.train(raw_df_enriched, activity_stats_df)
            report("✓ Trained [ml_with_energy] models.")
        except Exception as e:
            print(f"    [!] Energy-aware models failed to train: {e}")

    # ── Train Dynamic ML Curve Predictors ──────────────────────────────────
    energy_pipelines = {}
    if sensors and df_expanded_proc is not None:
        print("\n  ── Training Dynamic Energy Curve Predictors for Feedback ──")
        from sim_extractor import split_curves, build_and_train_pipeline
        for sensor in sensors:
            try:
                train_curves, _ = split_curves(df_expanded_proc, sensor)
                if train_curves:
                    pipeline = build_and_train_pipeline(train_curves, sensor, verbose=False)
                    energy_pipelines[sensor] = pipeline
                    print(f"    ✓ trained curve pipeline for {sensor}")
            except Exception as e:
                print(f"    [!] Failed curves for {sensor}: {e}")
    all_energy_pipelines[process] = energy_pipelines

    # ── Simulation Loop ───────────────────────────────────────────────────
    process_mode_results = []
    
    for sim_mode in MODES_TO_COMPARE:
        print("\n" + "─"*80)
        print(f"  ▶ SIMULATION MODE: {sim_mode.upper()}")
        print("─"*80)

        simulation_mode = 'ml' if sim_mode.startswith('ml_') else sim_mode
        mode_ml = None
        mode_eps = None
        mode_pm = None
        mode_stats = activity_stats_df

        if sim_mode == 'ml_no_energy':
            mode_ml = ml_models_no_energy
        elif sim_mode == 'ml_with_energy':
            if ml_models_with_energy is None:
                print(f"    ⚠️ Skipping 'ml_with_energy' for {process}")
                continue
            mode_ml = ml_models_with_energy
            mode_eps = energy_pipelines
        elif sim_mode.startswith('petri_net_'):
            mode_algorithm = sim_mode.replace('petri_net_', '', 1).strip().lower()
            if mode_algorithm not in extraction_by_algorithm:
                continue
            simulation_mode = 'petri_net'
            mode_stats = extraction_by_algorithm[mode_algorithm]['activity_stats_df']
            mode_pm = extraction_by_algorithm[mode_algorithm]['process_models']

        # Simulation
        sim_log_train = ProcessSimulation(
            mode_stats, production_plan,
            mode=simulation_mode, ml_models=mode_ml,
            process_models=mode_pm, energy_pipelines=mode_eps
        ).run()

        df_train_expanded = train_datasets[process].get('expanded')
        eval_train = comprehensive_simulation_evaluation(sim_log_train, df_train, real_expanded_df=df_train_expanded)
        flattened = {
            'process': process, 'mode': sim_mode, 'simulation_mode': simulation_mode,
            'split': "TRAIN" if TEMPORAL_SPLIT else "ALL DATA"
        }
        for cat, metrics in eval_train.items():
            if isinstance(metrics, dict):
                for m, v in metrics.items(): flattened[f"train_{cat}_{m}"] = v
            else:
                flattened[f"train_{cat}"] = metrics

        # Test Simulation
        df_test = test_datasets[process]['event_log'] if test_datasets else None
        if TEMPORAL_SPLIT and df_test is not None and len(df_test) > 0:
            pp_test = test_datasets[process]['production_plan']
            sim_log_test = ProcessSimulation(
                mode_stats, pp_test,
                mode=simulation_mode, ml_models=mode_ml,
                process_models=mode_pm, energy_pipelines=mode_eps
            ).run()
            df_test_expanded = test_datasets[process].get('expanded')
            eval_test = comprehensive_simulation_evaluation(sim_log_test, df_test, real_expanded_df=df_test_expanded)
            for cat, metrics in eval_test.items():
                if isinstance(metrics, dict):
                    for m, v in metrics.items(): flattened[f"test_{cat}_{m}"] = v
                else:
                    flattened[f"test_{cat}"] = metrics

        process_mode_results.append(flattened)
        evaluation_results_list.append(flattened)

    if process_mode_results:
        best = max(process_mode_results, key=lambda r: float(r.get('train_overall_score', -1)))
        report(f"\n  🏆 BEST MODE: {best['mode']} (Score: {best.get('train_overall_score'):.4f})")

# Final Reporting
evaluation_results_df = pd.DataFrame(evaluation_results_list)
report("\n" + "="*80)
report("FINAL PERFORMANCE RANKING")
report("="*80)

if not evaluation_results_df.empty:
    sort_col = 'test_overall_score'
    if sort_col in evaluation_results_df.columns:
        display_cols = ['process', 'mode', 'split', 'train_overall_score', 'test_overall_score']
        for b in CORE_METRIC_BASES:
            if f"test_{b}" in evaluation_results_df.columns:
                display_cols.append(f"test_{b}")
        for process_name, grp in evaluation_results_df.groupby('process'):
            report(f"\n  ▶ PROCESS: {process_name}")
            sorted_grp = grp.sort_values(sort_col, ascending=False)
            report(sorted_grp[display_cols].to_string(index=False))
            display(sorted_grp[display_cols])

# 2. PERFORMANCE HEATMAP COMPARISON
if not evaluation_results_df.empty:
    report("\n" + "█"*80)
    report("█   PERFORMANCE HEATMAPS (Normalized: 1.0 = Best)                    █")
    report("█"*80)
    
    heatmap_bases = CORE_METRIC_BASES
    
    # ─── TRAINING SET HEATMAP ───
    train_cols = [f"train_{b}" for b in heatmap_bases]
    train_energy = [c for c in evaluation_results_df.columns if c.startswith('train_energy_')]
    _plot_results_heatmap(train_cols + train_energy, "Simulation Performance Gallery: TRAINING SET")
    
    # ─── TESTING SET HEATMAP (if available) ───
    if TEMPORAL_SPLIT:
        test_cols = [f"test_{b}" for b in heatmap_bases]
        test_energy = [c for c in evaluation_results_df.columns if c.startswith('test_energy_')]
        # Only plot if we actually have some test results
        if any(c in evaluation_results_df.columns for c in test_cols):
             _plot_results_heatmap(test_cols + test_energy, "Simulation Performance Gallery: TESTING SET")

report("\n" + "█"*80)
report("█   FINAL ENERGY PERFORMANCE REPORT                                  █")
report("█"*80)

if not evaluation_results_df.empty:
    energy_results = evaluation_results_df[evaluation_results_df['mode'] == 'ml_with_energy']
    if not energy_results.empty:
        energy_cols = [c for c in energy_results.columns if c.startswith('test_energy_')]
        if energy_cols:
            records = []
            for _, row in energy_results.iterrows():
                for c in energy_cols:
                    if pd.notna(row[c]):
                        parts = c.split('_')
                        metric = parts[-1]
                        sensor = "_".join(parts[2:-1])
                        records.append({'Process': row['process'], 'Sensor': sensor, 'Metric': metric, 'Value': row[c]})
            if records:
                edf = pd.DataFrame(records)
                pivot = edf.pivot_table(index=['Process', 'Sensor'], columns='Metric', values='Value')
                report(pivot.round(4).to_string())
                display(pivot.round(4))

print("\nPIPELINE EXECUTION COMPLETE.")

# 3. ENERGY CURVE VISUALIZATION (GENERALIZATION GALLERY)
print("\n[DIAGNOSTIC] Checking for visualization data...")

# Use globals() to be safe across different execution environments (Notebook vs Script)
energy_pipes = globals().get('all_energy_pipelines', {})
trained_processes = [p for p, pipes in energy_pipes.items() if pipes]

if not trained_processes:
    print("[DIAGNOSTIC] No energy models were trained. Check if your columns end with '_energy'.")
else:
    print(f"[DIAGNOSTIC] Found trained energy models for: {trained_processes}")
    report("\n" + "█"*80)
    report("█   3. ENERGY CURVE VISUALIZATION (GENERALIZATION GALLERY)           █")
    report("█"*80)
    
    from sim_extractor import split_curves, evaluate_pipeline_on_test
    
    # Determine split to visualize
    use_test = globals().get('TEMPORAL_SPLIT', False) and globals().get('test_datasets')
    target_datasets = globals().get('test_datasets') if use_test else globals().get('train_datasets')
    split_label = "TEST SET" if use_test else "TRAINING SET (ALL DATA)"
    
    if target_datasets:
        for process in trained_processes:
            if process not in target_datasets:
                print(f"[DIAGNOSTIC] Process {process} not found in {split_label}")
                continue
                
            exp_df = target_datasets[process].get('expanded')
            if exp_df is None or exp_df.empty: 
                print(f"[DIAGNOSTIC] No expanded energy data found for {process} in {split_label}")
                continue
                
            sensors = list(energy_pipes[process].keys())
            for sensor in sensors:
                report(f"\nEvaluating generalization for {process} - {sensor} ({split_label})...")
                
                # Fetch config for activities/objects if it exists
                config = globals().get('process_datasets_to_model_sensors', {}).get(process, {})
                
                # Extract ground-truth curves
                gt_curves, _ = split_curves(
                    exp_df, sensor, 
                    config.get('activities_to_model', []), 
                    config.get('objects_to_model', []), 
                    test_size=0.0, 
                    verbose=0
                )
                
                if gt_curves:
                    display(Markdown(f"### 🎨 Visual Validation: {sensor} (Dataset: {process}, Split: {split_label})"))
                    display(Markdown(f"*Comparing predicted curves vs real sensor measurements.*"))
                    
                    pipeline = energy_pipes[process][sensor]
                    evaluate_pipeline_on_test(gt_curves, pipeline, max_plot_curves=6, verbose=1)
                else:
                    report(f"  [!] No valid curves found in {split_label} for {sensor}")

report("\n" + "█"*80 + "\n")
#%%
