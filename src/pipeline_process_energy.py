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
# process_datasets_to_model_sensors['process_4']['objects_to_model'] = ['Erhitzer']
# process_datasets_to_model_sensors['process_4']['activities_to_model'] = ['Step-032 = Umlauf', 'Step-030 = Produktion']
# process_datasets_to_model_sensors['process_4']['sensors_to_model'] = ['temp_nach_WR2_(WT2)_5s_energy']

process_datasets_to_model_sensors['process_2'] = process_datasets_to_model_sensors.get('process_2', {})
process_datasets_to_model_sensors['process_2']['objects_to_model'] = ['l01']
process_datasets_to_model_sensors['process_2']['activities_to_model'] = ['Produktion']
process_datasets_to_model_sensors['process_2']['sensors_to_model'] = ['pro_volstrom_l/h_energy']

process_datasets_to_model_sensors['process_3'] = process_datasets_to_model_sensors.get('process_3', {})
# process_datasets_to_model_sensors['process_3']['objects_to_model'] = ['tower_1']
# process_datasets_to_model_sensors['process_3']['activities_to_model'] = ['Produktion']
# process_datasets_to_model_sensors['process_3']['sensors_to_model'] = ['(8)_abluft_mas_kg/h_energy']


# display(process_datasets_to_model_sensors['process_3']['expanded'])

# %%

processes_to_run = ['process_2', 'process_3', 'process_4']
processes_to_run = ['process_2']
# Filter the original dictionary
process_datasets_to_model = {
    k: v for k, v in process_datasets.items() 
    if k in processes_to_run
}
# The rest of your configuration will now only see process_3
process_datasets_to_model_sensors = process_datasets_to_model.copy()

# %%
import pandas as pd
from sim_extractor import extract_process
from simulation import ProcessSimulation
from sim_modeller import SimModeller

# ─────────────────────────────────────────────────────────────────────────────
# ENERGY MODIFIER CONFIGURATION
#   Runs after the energy modelling section (predict_raw_curve etc).
#   Configure which sklearn estimator to use for each modifier.
#   Both modifiers accept any estimator with .fit() and .predict()/.predict_proba().
# ─────────────────────────────────────────────────────────────────────────────
from sklearn.linear_model import Lasso, LogisticRegression
from sim_extractor import extract_energy_modifiers, extract_energy_direct_models

from xgboost import XGBRegressor

# ── Duration modifier models ───────────────────────────────────────────────
# List of sklearn-compatible regressor types to compete per activity
ENERGY_DURATION_MODELS    = ['xgboost', 'linear', 'lasso', 'mlp', 'statistical']

# ── Transition modifier models ─────────────────────────────────────────────
# List of classifier types to compete per activity
ENERGY_TRANSITION_MODELS  = ['logistic', 'random_forest', 'gradient_boosting']

ENERGY_DURATION_SCALE_CLIP = (0.7, 1.3)   # max ±30% shift per activity
ENERGY_LOGIT_BIAS_CLIP     = (-1.0, 1.0)  # max ~2.7× odds-ratio shift per competing activity
ENERGY_MIN_SAMPLES         = 1           # STRICT: skip ML (use statistical) if n_samples < 30

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

# ─────────────────────────────────────────────────────────────────────────────
# CURVE-ONLY EVALUATION MODE
#   True  → after energy pipelines are trained, run a full curve-quality
#            evaluation (MAE / RMSE / WAPE / R²) per activity and sensor,
#            with plots, WITHOUT running any process simulation.
#            Use this to benchmark curve models in isolation before picking
#            one of the improved approaches (seq2seq, basis expansion, DTW).
#   False → skip this block (default when running full pipeline).
# ─────────────────────────────────────────────────────────────────────────────
RUN_CURVE_ONLY_EVALUATION = True


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

if RUN_CURVE_ONLY_EVALUATION:
    print("RUN_CURVE_ONLY_EVALUATION=True — skipping process modelling loop.")

for process in process_datasets_to_model.keys() if not RUN_CURVE_ONLY_EVALUATION else []:
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
            except Exception as e:
                print(f"    (Could not render Petri net: {e})")

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

        eval_train = comprehensive_simulation_evaluation(simulated_log_train, df_train)

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

            eval_test = comprehensive_simulation_evaluation(simulated_log_test, df_test)

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
                    # Fall back: all *_energy columns that are numeric and not log columns
                    _sensors = [
                        c for c in _df_expanded_train.columns
                        if c.endswith('_energy')
                        and not c.endswith('_log')
                        and _df_expanded_train[c].dtype in ('float64', 'float32', 'int64', 'int32')
                    ]
                    if _sensors:
                        print(f"  ℹ️  Auto-detected {len(_sensors)} sensor column(s): {_sensors}")

                if not _sensors:
                    print("⚠️  No sensors found for this process — skipping energy modifiers.")
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
                        # Only run this if we actually want to evaluate on Test results
                        # as this DTW-based training is the slowest part of the pipeline.
                        if RUN_TEST_EVALUATION:
                            from sim_extractor import split_curves, build_and_train_pipeline, predict_raw_curve
                            from sklearn.linear_model import LinearRegression
                            from sklearn.ensemble import GradientBoostingRegressor

                            _energy_pipelines = {}
                            
                            _config = process_datasets_to_model_sensors.get(process, {}) if 'process_datasets_to_model_sensors' in dir() else {}
                            _activities_list = _config.get('activities_to_model', _df_expanded_train['activity_log'].dropna().unique().tolist())
                            _objects_list = _config.get('objects_to_model', _df_expanded_train['object_log'].dropna().unique().tolist())
                            
                            for _sensor in _sensors:
                                print(f"\n  ℹ️ Training dynamic ML curve for sensor: {_sensor}")
                                _train_curves, _ = split_curves(
                                    _df_expanded_train,
                                    variable=_sensor,
                                    activities=_activities_list,
                                    objects=_objects_list,
                                    test_size=0.0, # All data into train since we do global temporal split
                                    verbose=0,
                                )
                                _ep_pipeline = build_and_train_pipeline(
                                    _train_curves,
                                    variable=_sensor,
                                    fixed_length=100,
                                    val_size=0.2, # Validation internally handles R2 evaluation
                                    models={
                                        'Linear Regression': LinearRegression,
                                        'Gradient Boosting': GradientBoostingRegressor,
                                    },
                                    optimize_hyperparams=False,
                                    verbose=VERBOSE_EVAL
                                )
                                
                                def _make_predict_fn(ep_bound):
                                    return lambda raw_values, activity, object_attributes: predict_raw_curve(
                                        raw_values, activity, object_attributes, pipeline=ep_bound
                                    )
                                    
                                _energy_pipelines[_sensor] = {
                                    'reference_curve': _ep_pipeline['reference_curve'],
                                    'predict_fn': _make_predict_fn(_ep_pipeline),
                                    'full_pipeline': _ep_pipeline 
                                }
                                
                            all_energy_pipelines[process] = _energy_pipelines
                        else:
                            _energy_pipelines = {}

                    except Exception as _exc:
                        print(f"⚠️  extract_energy_modifiers or ML curve modeling failed: {_exc}")
                        _energy_dur_mods, _energy_tr_mods, _energy_state_cols = {}, {}, []
                        _energy_pipelines = {}

                # ── Simulate energy-aware modes ───────────────────────────
                for _energy_mode in _energy_modes_requested:
                    print("\n" + "─"*80)
                    print(f"  ▶ SIMULATION MODE: {_energy_mode.upper()}")
                    print("─"*80)

                    def _run_energy_sim(plan, stats_df, pm):
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
                            duration_scale_clip=ENERGY_DURATION_SCALE_CLIP,
                            logit_bias_clip=ENERGY_LOGIT_BIAS_CLIP,
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
                        _energy_sim_train, df_train, real_expanded_df=_df_expanded_train
                    )

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
                            _pp_test = test_datasets[process]['production_plan']
                            _exp_test = test_datasets[process]['expanded']
                            _energy_sim_test = _run_energy_sim(
                                _pp_test, _best_base_stats, _best_base_pm
                            )
                            if VERBOSE_EVAL:
                                print(f"\n  Simulated log TEST  ({_energy_mode}): "
                                      f"{len(_energy_sim_test)} events")

                            _eval_test = comprehensive_simulation_evaluation(
                                _energy_sim_test, _df_test, real_expanded_df=_exp_test
                            )
                            for _cat, _met in _eval_test.items():
                                if _cat == 'energy_metrics' and isinstance(_met, dict):
                                    for _sensor, _vals in _met.items():
                                        for _mn, _mv in _vals.items():
                                            _energy_flattened[f"test_energy_{_sensor}_{_mn}"] = _mv
                                elif isinstance(_met, dict):
                                    for _mn, _mv in _met.items():
                                        _energy_flattened[f"test_{_cat}_{_mn}"] = _mv
                                else:
                                    _energy_flattened[f"test_{_cat}"] = _met

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
            _plot_results_heatmap(_train_cols, f"Training Quality: {process}", local_df=_proc_df)


if not RUN_CURVE_ONLY_EVALUATION:
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

    evaluation_results_df.to_parquet(
        "evaluation_results.parquet",
        engine="pyarrow",
        index=False
    )
    evaluation_results_df

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

if RUN_TEST_EVALUATION and not RUN_CURVE_ONLY_EVALUATION:
    # Final consolidated summary of TEST set performance across ALL processes
    # (Focuses strictly on the core metrics to maintain clarity)
    _final_test_cols = [f"test_{b}" for b in CORE_METRIC_BASES if f"test_{b}" in evaluation_results_df.columns]
    if _final_test_cols:
        display(Markdown("---"))
        display(Markdown("# 📊 FINAL CONSOLIDATED PERFORMANCE: TEST SET ENSEMBLE"))
        display(Markdown("*Consolidated simulation quality across all processes on unseen data.*"))
        _plot_results_heatmap(_final_test_cols, "Generalization Performance: Test Set Ensemble")



# %%
# ── CURVE-ONLY: train ALL approaches without process modelling ────────────────
# Trains baseline, Approach 2 (B-spline basis), and Approach 3 (DTW-phase)
# side by side so results can be compared in the evaluation cell below.
if RUN_CURVE_ONLY_EVALUATION:
    from sim_extractor import (
        split_curves,
        build_and_train_pipeline,            predict_raw_curve,
        build_and_train_pipeline_instance_stats, predict_raw_curve_instance_stats,
        build_and_train_pipeline_istats_leakfree, predict_raw_curve_istats_leakfree,
        build_and_train_pipeline_dtw_phase,  predict_raw_curve_dtw_phase,
        build_and_train_pipeline_basis,      predict_raw_curve_basis,
        build_and_train_pipeline_exog,       predict_raw_curve_exog,
        build_and_train_pipeline_seq2seq,    predict_raw_curve_seq2seq,
    )
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import GradientBoostingRegressor

    all_energy_pipelines                  = {}   # baseline
    all_energy_pipelines_instance_stats  = {}   # Instance Stats (leaky)
    all_energy_pipelines_istats_leakfree = {}   # Instance Stats (leak-free)
    all_energy_pipelines_dtw_phase       = {}   # Approach 3
    all_energy_pipelines_basis           = {}   # Approach 2
    all_energy_pipelines_exog            = {}   # DTW + External Factors
    all_energy_pipelines_seq2seq         = {}   # DTW + Seq2Seq

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
        _pipelines_seq2seq             = {}

        for _sensor in _sensors:
            # Split with exog columns so curves carry the ef_ time series
            _train_curves, _ = split_curves(
                _df_train_exp,
                variable=_sensor,
                activities=_activities,
                objects=_objects,
                test_size=0.0,
                verbose=0,
                exog_columns=_ef_cols,
            )
            if not _train_curves:
                print(f"  [{_sensor}] No curves found — skipping.")
                continue

            # ── Baseline ────────────────────────────────────────────────────
            print(f"  [{_sensor}] Training baseline (DTW + position index)...")
            _ep_base = build_and_train_pipeline(
                _train_curves,
                variable=_sensor,
                fixed_length=100,
                val_size=0.2,
                models=_CURVE_MODELS,
                optimize_hyperparams=False,
                verbose=False,
            )
            print(f"    Best model: {_ep_base['model_name']}  val R²={_ep_base['val_r2']:.4f}")

            def _make_pred_base(ep):
                return lambda rv, act, attrs: predict_raw_curve(rv, act, attrs, pipeline=ep)

            _pipelines_baseline[_sensor] = {
                'reference_curve': _ep_base['reference_curve'],
                'predict_fn':      _make_pred_base(_ep_base),
                'full_pipeline':   _ep_base,
            }

            # ── Instance Stats — DTW + instance curve statistics ─────────
            print(f"  [{_sensor}] Training Instance Stats (DTW + curve stats)...")
            _ep_istats = build_and_train_pipeline_instance_stats(
                _train_curves,
                variable=_sensor,
                fixed_length=100,
                val_size=0.2,
                models=_CURVE_MODELS,
                optimize_hyperparams=False,
                verbose=False,
            )
            print(f"    Best model: {_ep_istats['model_name']}  val R²={_ep_istats['val_r2']:.4f}")

            def _make_pred_istats(ep):
                return lambda rv, act, attrs: predict_raw_curve_instance_stats(rv, act, attrs, pipeline=ep)

            _pipelines_instance_stats[_sensor] = {
                'reference_curve': _ep_istats['reference_curve'],
                'predict_fn':      _make_pred_istats(_ep_istats),
                'full_pipeline':   _ep_istats,
            }

            # ── Instance Stats (Leak-Free) — two-stage ───────────────────
            print(f"  [{_sensor}] Training Instance Stats Leak-Free (two-stage)...")
            _ep_lf = build_and_train_pipeline_istats_leakfree(
                _train_curves,
                variable=_sensor,
                fixed_length=100,
                val_size=0.2,
                models=_CURVE_MODELS,
                optimize_hyperparams=False,
                verbose=False,
            )
            print(f"    Best model: {_ep_lf['model_name']}  val R²={_ep_lf['val_r2']:.4f}")

            def _make_pred_lf(ep):
                return lambda rv, act, attrs: predict_raw_curve_istats_leakfree(rv, act, attrs, pipeline=ep)

            _pipelines_istats_leakfree[_sensor] = {
                'reference_curve': _ep_lf['reference_curve'],
                'predict_fn':      _make_pred_lf(_ep_lf),
                'full_pipeline':   _ep_lf,
            }

            # ── Approach 3 — DTW-phase ───────────────────────────────────
            print(f"  [{_sensor}] Training Approach 3 (DTW + phase features)...")
            _ep_phase = build_and_train_pipeline_dtw_phase(
                _train_curves,
                variable=_sensor,
                fixed_length=100,
                val_size=0.2,
                models=_CURVE_MODELS,
                optimize_hyperparams=False,
                verbose=False,
            )
            print(f"    Best model: {_ep_phase['model_name']}  val R²={_ep_phase['val_r2']:.4f}")

            def _make_pred_phase(ep):
                return lambda rv, act, attrs: predict_raw_curve_dtw_phase(rv, act, attrs, pipeline=ep)

            _pipelines_dtw_phase[_sensor] = {
                'reference_curve': _ep_phase['reference_curve'],
                'predict_fn':      _make_pred_phase(_ep_phase),
                'full_pipeline':   _ep_phase,
            }

            # ── Approach 2 — B-spline basis expansion ───────────────────
            print(f"  [{_sensor}] Training Approach 2 (B-spline basis expansion)...")
            _ep_basis = build_and_train_pipeline_basis(
                _train_curves,
                variable=_sensor,
                fixed_length=100,
                n_basis=20,
                val_size=0.2,
                models=_CURVE_MODELS,
                optimize_hyperparams=False,
                verbose=False,
            )
            print(f"    Best model: {_ep_basis['model_name']}  val R²={_ep_basis['val_r2']:.4f}")

            def _make_pred_basis(ep):
                return lambda rv, act, attrs: predict_raw_curve_basis(rv, act, attrs, pipeline=ep)

            _pipelines_basis[_sensor] = {
                'reference_curve': _ep_basis['reference_curve'],
                'predict_fn':      _make_pred_basis(_ep_basis),
                'full_pipeline':   _ep_basis,
            }

            # ── DTW + External Factors ───────────────────────────────────
            if _ef_cols:
                print(f"  [{_sensor}] Training DTW + External Factors ({len(_ef_cols)} ef_ signals)...")
                _ep_exog = build_and_train_pipeline_exog(
                    _train_curves,
                    variable=_sensor,
                    fixed_length=100,
                    val_size=0.2,
                    models=_CURVE_MODELS,
                    optimize_hyperparams=False,
                    verbose=False,
                )
                print(f"    Best model: {_ep_exog['model_name']}  val R²={_ep_exog['val_r2']:.4f}")

                def _make_pred_exog(ep):
                    return lambda rv, act, attrs, exog=None: predict_raw_curve_exog(
                        rv, act, attrs, pipeline=ep, exog_values=exog or {}
                    )

                _pipelines_exog[_sensor] = {
                    'reference_curve': _ep_exog['reference_curve'],
                    'predict_fn':      _make_pred_exog(_ep_exog),
                    'full_pipeline':   _ep_exog,
                }
            else:
                print(f"  [{_sensor}] No ef_ columns found — skipping DTW+Exog.")

            # ── DTW + Seq2Seq ────────────────────────────────────────────
            print(f"  [{_sensor}] Training DTW + Seq2Seq (LSTM encoder-decoder)...")
            _ep_seq2seq = build_and_train_pipeline_seq2seq(
                _train_curves,
                variable=_sensor,
                fixed_length=100,
                val_size=0.2,
                hidden_size=128,
                num_layers=2,
                dropout=0.1,
                epochs=80,
                batch_size=32,
                lr=1e-3,
                teacher_forcing_ratio=0.5,
                patience=10,
                verbose=False,
            )
            print(f"    val_loss={_ep_seq2seq['val_loss']:.5f}")

            def _make_pred_seq2seq(ep):
                return lambda rv, act, attrs: predict_raw_curve_seq2seq(rv, act, attrs, pipeline=ep)

            _pipelines_seq2seq[_sensor] = {
                'reference_curve': _ep_seq2seq['reference_curve'],
                'predict_fn':      _make_pred_seq2seq(_ep_seq2seq),
                'full_pipeline':   _ep_seq2seq,
            }

        all_energy_pipelines[_proc]                   = _pipelines_baseline
        all_energy_pipelines_instance_stats[_proc]   = _pipelines_instance_stats
        all_energy_pipelines_istats_leakfree[_proc]  = _pipelines_istats_leakfree
        all_energy_pipelines_dtw_phase[_proc]        = _pipelines_dtw_phase
        all_energy_pipelines_basis[_proc]            = _pipelines_basis
        all_energy_pipelines_exog[_proc]             = _pipelines_exog
        all_energy_pipelines_seq2seq[_proc]          = _pipelines_seq2seq

# %%
# ══════════════════════════════════════════════════════════════════════════════
# CURVE-ONLY EVALUATION — BASELINE vs APPROACH 2 (B-spline) vs APPROACH 3 (DTW-phase)
# ══════════════════════════════════════════════════════════════════════════════

def _run_curve_eval(pipelines_dict, approach_label, split_label,
                    df_lookup, activities, objects):
    """
    Evaluate every (process, sensor) in pipelines_dict against curves from
    df_lookup.  Returns a list of per-curve metric dicts.
    """
    from sim_extractor import evaluate_pipeline_on_test, split_curves
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

        for _sensor, _ep in _sensors.items():
            _fp = _ep.get('full_pipeline', {})
            _exog_cols_eval = _fp.get('exog_cols', []) if _fp.get('approach') == 'exog' else None
            _curves, _ = split_curves(_df, _sensor, _acts, _objs,
                                      test_size=0.0, verbose=0,
                                      exog_columns=_exog_cols_eval)
            if not _curves:
                continue
            show_plots = (split_label == 'TEST')
            _metrics_df, _agg = evaluate_pipeline_on_test(
                _curves, _ep['full_pipeline'],
                max_plot_curves=6 if show_plots else 0,
                verbose=1 if show_plots else 0,
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
                })
    return records


if RUN_CURVE_ONLY_EVALUATION and 'all_energy_pipelines' in dir() and all_energy_pipelines:
    from sim_extractor import evaluate_pipeline_on_test, split_curves

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
            ('Baseline (DTW + pos)',        all_energy_pipelines),
            ('Instance Stats (leaky)',      all_energy_pipelines_instance_stats),
            ('Instance Stats (leak-free)',  all_energy_pipelines_istats_leakfree),
            ('Approach 2 (B-spline)',       all_energy_pipelines_basis),
            ('Approach 3 (DTW-phase)',      all_energy_pipelines_dtw_phase),
            ('DTW + Ext. Factors',          all_energy_pipelines_exog),
            ('DTW + Seq2Seq',               all_energy_pipelines_seq2seq),
        ]:
            if not _pipelines:
                continue
            display(Markdown(f"### {_approach_label}"))
            _recs = _run_curve_eval(
                _pipelines, _approach_label, _split_label,
                _df_src, _activities_map, _objects_map,
            )
            _all_records.extend(_recs)

    # ── Side-by-side comparison table ───────────────────────────────────────
    if _all_records:
        _all_df = pd.DataFrame(_all_records)

        display(Markdown("---"))
        display(Markdown("## Summary — Baseline vs Approach 3 (TEST set)"))

        _test_df = _all_df[_all_df['Split'] == 'TEST']
        if not _test_df.empty:
            _compare = (
                _test_df
                .groupby(['Approach', 'Process', 'Sensor'])[['MAE', 'RMSE', 'WAPE', 'R2']]
                .mean()
                .round(4)
            )
            display(_compare)
            report("\nCURVE MODEL COMPARISON (TEST SET)")
            report(_compare.to_string())

            # ── Per-activity table ───────────────────────────────────────────
            display(Markdown("### Per-activity breakdown (TEST)"))
            _act_compare = (
                _test_df
                .groupby(['Approach', 'Process', 'Sensor', 'Activity'])[['MAE', 'RMSE', 'WAPE', 'R2']]
                .mean()
                .round(4)
            )
            display(_act_compare)

            # ── R² heatmap — one subplot per approach ────────────────────────
            _approaches = _test_df['Approach'].unique()
            fig_h, axes_h = plt.subplots(
                1, len(_approaches),
                figsize=(max(8, _test_df['Activity'].nunique() * 1.4) * len(_approaches),
                         max(3, _test_df[['Process', 'Sensor']].drop_duplicates().shape[0] * 1.2))
            )
            if len(_approaches) == 1:
                axes_h = [axes_h]

            for ax_h, _appr in zip(axes_h, _approaches):
                _sub = _test_df[_test_df['Approach'] == _appr]
                _ph = _sub.pivot_table(
                    index=['Process', 'Sensor'], columns='Activity',
                    values='R2', aggfunc='mean'
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
            plt.show()

            # ── Delta heatmaps: each new approach minus baseline ─────────────
            _base_pivot = _test_df[_test_df['Approach'] == 'Baseline (DTW + pos)'].pivot_table(
                index=['Process', 'Sensor'], columns='Activity', values='R2', aggfunc='mean'
            )
            for _delta_label, _delta_appr in [
                ('Instance Stats',         'Instance Stats'),
                ('Approach 2 (B-spline)', 'Approach 2 (B-spline)'),
                ('Approach 3 (DTW-phase)', 'Approach 3 (DTW-phase)'),
                ('DTW + Ext. Factors',     'DTW + Ext. Factors'),
                ('DTW + Seq2Seq',          'DTW + Seq2Seq'),
            ]:
                _new_pivot = _test_df[_test_df['Approach'] == _delta_appr].pivot_table(
                    index=['Process', 'Sensor'], columns='Activity', values='R2', aggfunc='mean'
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
                plt.show()

elif RUN_CURVE_ONLY_EVALUATION:
    display(Markdown(
        "> **Curve-Only Evaluation skipped** — `all_energy_pipelines` is empty. "
        "Make sure the energy pipeline training block ran successfully."
    ))

# %%
# ── STANDALONE PROFILE EVALUATION (TRAIN & TEST) ──────────────────────────────
from sim_extractor import evaluate_pipeline_on_test, split_curves
profile_summary_records = []

if 'process_datasets_to_model_sensors' in dir():
    for process, config in process_datasets_to_model_sensors.items():
        sensors_to_model = config.get('sensors_to_model', [])
        activities_to_model = config.get('activities_to_model', [])
        objects_to_model = config.get('objects_to_model', [])

        for sensor in sensors_to_model:
            if process not in all_energy_pipelines or sensor not in all_energy_pipelines[process]:
                continue
            
            pipeline = all_energy_pipelines[process][sensor]['full_pipeline']
            
            # Evaluate on TRAIN (60%)
            df_train_exp = train_datasets[process].get('expanded')
            if df_train_exp is not None:
                # We use verbose=0 because we just want the metrics here
                train_curves, _ = split_curves(df_train_exp, sensor, activities_to_model, objects_to_model, test_size=0.0, verbose=0)
                _, tr_agg = evaluate_pipeline_on_test(train_curves, pipeline, verbose=0)
                tr_agg.update({'process': process, 'sensor': sensor, 'split': 'TRAIN'})
                profile_summary_records.append(tr_agg)
                
            # Evaluate on TEST (40%) - Guarded for speed
            if RUN_TEST_EVALUATION:
                df_test_exp = test_datasets[process].get('expanded')
                if df_test_exp is not None and not df_test_exp.empty:
                    test_curves, _ = split_curves(df_test_exp, sensor, activities_to_model, objects_to_model, test_size=0.0, verbose=0)
                    if test_curves:
                        metrics_df, ts_agg = evaluate_pipeline_on_test(test_curves, pipeline, verbose=0)
                        
                        # Get per-activity averages to show 'Subprocess' granularity
                        if not metrics_df.empty:
                            activity_metrics = metrics_df.groupby('activity')[['MAE', 'RMSE', 'WAPE (%)', 'R2']].mean().reset_index()
                            for _, act_row in activity_metrics.iterrows():
                                summary_rec = {
                                    'Dataset': process,
                                    'Subprocess': act_row['activity'],
                                    'Sensor': sensor,
                                    'MAE': act_row['MAE'],
                                    'RMSE': act_row['RMSE'],
                                    'WAPE': act_row['WAPE (%)'],
                                    'R2': act_row['R2']
                                }
                                profile_summary_records.append(summary_rec)


if profile_summary_records:
    profile_summary_df = pd.DataFrame(profile_summary_records)
    report("\n" + "="*80)
    report("DETAILED ENERGY PROFILE METRICS PER DATASET & SUBPROCESS (TEST SET)")
    report("="*80)
    
    # Pivot for clean display: Dataset, Subprocess, Sensor as index
    pivot_cols = ['MAE', 'RMSE', 'WAPE', 'R2']
    available_metrics = [c for c in pivot_cols if c in profile_summary_df.columns]
    summary_pivot = profile_summary_df.pivot_table(
        index=['Dataset', 'Subprocess', 'Sensor'], 
        values=available_metrics
    )
    
    # Reorder columns as requested
    final_cols = [c for c in ['MAE', 'RMSE', 'WAPE', 'R2'] if c in summary_pivot.columns]
    summary_pivot = summary_pivot[final_cols]
    
    report(summary_pivot.round(4).to_string())
    display(summary_pivot.round(4))

# ── SIMULATION CURVE VISUALS (SIMULATED VS REAL) ───────────────────────────
# We look for simulated logs in evaluation_results_list that have 'simulated_energy_curves'
report("\n" + "="*80)
report("VISUAL COMPARISON: SIMULATED (TEST RUN) VS REAL DATA")
report("="*80)




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
    report_pivot = edf.pivot_table(index=['Dataset', 'Sensor'], columns='Metric', values='Value', aggfunc='mean')
    
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
    for process, sensors in all_energy_pipelines.items():
        exp_test = test_datasets[process].get('expanded')
        if exp_test is None: continue
        
        _proc_sensor_config = process_datasets_to_model_sensors.get(process, {})
        for sensor in sensors:
            # We filter for the test curves for this specific process/sensor combo
            _activities = _proc_sensor_config.get('activities_to_model', exp_test['activity_log'].dropna().unique().tolist())
            _objects    = _proc_sensor_config.get('objects_to_model',    exp_test['object_log'].dropna().unique().tolist())
            test_curves, _ = split_curves(exp_test, sensor, _activities, _objects,
                                          test_size=0.0, verbose=0)
            
            if test_curves:
                display(Markdown("---"))
                display(Markdown(f"## 🎨 Generalization Gallery: {sensor.upper()}"))
                display(Markdown(f"*Visual verification of ML curve prediction vs Real ground-truth (Test Set)*"))
                
                # evaluate_pipeline_on_test uses a 3x2 grid by default for max_plot_curves=6
                evaluate_pipeline_on_test(
                    test_curves, 
                    all_energy_pipelines[process][sensor]['full_pipeline'], 
                    max_plot_curves=6, 
                    verbose=1 
                )

# %%