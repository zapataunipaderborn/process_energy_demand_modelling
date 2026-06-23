
# %%
import os
import numpy as np
import pandas as pd
from pathlib import Path
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt




# %%

experiment = '1'
resolution = 'original_resolution'
visuals = True

current_path = Path(__file__).resolve().parent if '__file__' in globals() else Path().resolve()
folder_gold_base = current_path.parent / 'data' / 'gold' / f'experiment_{experiment}'
folder_silver_base = current_path.parent / 'data' / 'silver'

info = {
    'experiment': experiment,
    'resolution': resolution,
    'folder_silver_base': str(folder_silver_base),
    'folder_gold_base': str(folder_gold_base),
    'Note': ""
}

json_path = folder_gold_base / 'info.json'
json_path.parent.mkdir(parents=True, exist_ok=True)
with open(json_path, 'w') as f:
    json.dump(info, f, indent=4)


## %% functions 

# %%

def plot_activity_sensor_curves_by_index(df_expanded, activity_col='activity_log'):
    energy_cols = [
        c for c in df_expanded.columns
        if c.endswith('_energy') and pd.api.types.is_numeric_dtype(df_expanded[c])
    ]

    base_cols = [
        activity_col,
        'case_id_log',
        'object_log',
        'timestamp_start_log',
        'timestamp_end_log'
    ]
    base_cols = [c for c in base_cols if c in df_expanded.columns]

    df_plot = df_expanded[base_cols + energy_cols].copy()

    # One id per activity execution
    df_plot['activity_exec_id'] = (
        df_plot[activity_col].astype(str) + ' | ' +
        df_plot['case_id_log'].astype(str) + ' | ' +
        df_plot['timestamp_start_log'].astype(str) + ' | ' +
        df_plot['timestamp_end_log'].astype(str)
    )

    # Sort so the index reflects the time order inside each execution
    sort_cols = [c for c in ['activity_exec_id', 'timestamp_start_log'] if c in df_plot.columns]
    df_plot = df_plot.sort_values(sort_cols).copy()

    # Local sample index per execution
    df_plot['t_idx'] = df_plot.groupby('activity_exec_id').cumcount()

    # Long format for seaborn
    long_df = df_plot.melt(
        id_vars=[activity_col, 'activity_exec_id', 't_idx'],
        value_vars=energy_cols,
        var_name='sensor',
        value_name='value'
    ).dropna(subset=['value'])

    for activity_name, activity_df in long_df.groupby(activity_col, sort=False):
        g = sns.FacetGrid(
            activity_df,
            col='sensor',
            col_wrap=3,
            height=3.2,
            sharex=True,
            sharey=False,
            aspect=1.3
        )

        g.map_dataframe(
            sns.lineplot,
            x='t_idx',
            y='value',
            units='activity_exec_id',
            estimator=None,
            alpha=0.35,
            linewidth=1,
            color='steelblue',
            legend=False
        )

        g.set_titles(col_template='{col_name}')
        g.set_axis_labels('Index inside activity execution', 'Sensor value')
        g.fig.suptitle(f'Activity: {activity_name}', y=1.02)

        # No legend
        if g._legend is not None:
            g._legend.remove()

        plt.tight_layout()
        plt.show()
# %%
######## Process 2 ########

process = 2

# Define the path to the current file's location


# Define the folder path
files_folder_silver = folder_silver_base / f'process_{process}'
files_folder_gold = folder_gold_base / f'process_{process}'
files_folder_gold.mkdir(parents=True, exist_ok=True)

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
df_activities['higher_level_activity'] = df_activities['temp_object']#'production_line'#'shift'
df_activities['object_type'] = df_activities['temp_object']#'production_line'
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

# Total material per case: integrate flow rate over actual measurement intervals
_vol = df_expanded[['case_id_log', 'datetime_energy', 'pro_volstrom_l/h_energy']].copy()
_vol = _vol.sort_values(['case_id_log', 'datetime_energy'])
_vol['_dt_h'] = _vol.groupby('case_id_log')['datetime_energy'].diff().dt.total_seconds() / 3600
_vol['_dt_h'] = _vol['_dt_h'].fillna(0)
_vol['_liters'] = _vol['pro_volstrom_l/h_energy'] * _vol['_dt_h']
_case_total = _vol.groupby('case_id_log')['_liters'].sum()
df_expanded['object_attributes_log'] = df_expanded.apply(
    lambda row: {**row['object_attributes_log'], 'total_material': _case_total.get(row['case_id_log'], None)}
    if isinstance(row['object_attributes_log'], dict) else row['object_attributes_log'],
    axis=1
)

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

df_production_plan = df.copy()

print(f"production plan")
print(df_production_plan)


relevant_columns = [
       'datetime_energy',
       'status_name_energy', 

    #    'ef_wind_speed_10m_energy',
    #    'ef_precipitation_energy', 
    #    'ef_wind_direction_100m_energy', 
       'ef_temperature_2m_energy',
       'ef_relative_humidity_2m_energy', 
       'ef_global_tilted_irradiance_energy',
    #    'ef_apparent_temperature_energy'

       'pro_menge_kg_energy',
       'pro_volstrom_l/h_energy',
       'dampfmenge_kg/h_nmb+cip_energy',

    #    'cip_turm_f_energy',
    #    'cip_turm_g_energy',

       'pro_power_kW_energy',
       'pro_temp_out_energy', 
       'medium_power_kW_energy',

       'timestamp_start_log', ''
       'timestamp_end_log', 'activity_log',
       'higher_level_activity_log', 'object_type_log', 'object_log',
       'case_id_log', 'object_attributes_log']


relevant_activites = [
 'Grundstellung', 'Vorwärmen des Warmwasserkreislaufs', 'Warten auf Ziel'
 'Materialvalidierung', 
 'Fehler Tankanwahl Bestand', 
 'Leitung zum Vorheizer füllen',
 'Leitung zum HSM füllen', 'Produktion' 'Spülen mit Wasser' 'Nachlauf'
 'Spülen auf Gully', 
 'Warten auf Quelle']

df_expanded = df_expanded[relevant_columns].copy()



files_folder_gold_datasets = files_folder_gold / 'datasets'
files_folder_gold_datasets.mkdir(parents=True, exist_ok=True)

df_expanded.to_parquet(files_folder_gold_datasets / "df_expanded.parquet", index=False)
df_event_log.to_parquet(files_folder_gold_datasets / "df_event_log.parquet", index=False)
df_production_plan.to_parquet(files_folder_gold_datasets / "df_production_plan.parquet", index=False)

# # %%
# display(df_expanded)

# # %%

# # }
# display(df_event_log)

# # %%
# # df_expanded['activity_log'].value_counts()

# # print(df_expanded['activity_log'].unique())


# # plot_activity_sensor_curves_by_index(df_expanded)

# # process_datasets['process_2'] = {
# #     'expanded': df_expanded,
# #     'event_log': df_event_log,
# #     'production_plan': production_plan
# # }
# display(df_expanded)

# # %%

# ######## Process 3 ########

# process = 3

# # Define the path to the current file's location


# # Define the path to the current file's location
# current_path = Path(__file__).resolve().parent if '__file__' in globals() else Path().resolve()

# # Define the folder path
# files_folder_silver = folder_silver_base / f'process_{process}'
# files_folder_gold = folder_gold_base / f'process_{process}'
# files_folder_gold.mkdir(parents=True, exist_ok=True)

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
#     row['object_attributes'] = {"none": "none"}  # Add the "none": "none" key-value pairk
#     activity_rows.append(row)

# df_activities = pd.DataFrame(activity_rows)

# # Now set the required columns on df_activities
# df_activities['higher_level_activity'] = 'tower'#'shift'
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

# # Combine f10/f11 feed rates: take element-wise maximum
# df_expanded['speise_current_kg/h_energy'] = df_expanded[['f10_speise_kg/h_energy', 'f11_speise_kg/h_energy']].max(axis=1)

# # Total material per case: integrate feed rate over actual measurement intervals
# _vol = df_expanded[['case_id_log', 'datetime_energy', 'speise_current_kg/h_energy']].copy()
# _vol = _vol.sort_values(['case_id_log', 'datetime_energy'])
# _vol['_dt_h'] = _vol.groupby('case_id_log')['datetime_energy'].diff().dt.total_seconds() / 3600
# _vol['_dt_h'] = _vol['_dt_h'].fillna(0)
# _vol['_kg'] = _vol['speise_current_kg/h_energy'] * _vol['_dt_h']
# _case_total = _vol.groupby('case_id_log')['_kg'].sum()
# df_expanded['object_attributes_log'] = df_expanded.apply(
#     lambda row: {**row['object_attributes_log'], 'total_material': _case_total.get(row['case_id_log'], None)}
#     if isinstance(row['object_attributes_log'], dict) else row['object_attributes_log'],
#     axis=1
# )

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

# ### Production plan
# df = df_event_log.copy()
# df = df[['case_id', 'activity', 'timestamp_start', 'timestamp_end', 'object_attributes']]

# # Only leave the case ids, the orders
# df = df.drop_duplicates(subset=['case_id'])

# df_production_plan = df.copy()

# print(f"production plan")
# print(df_production_plan)

# relevant_columns = ['datetime_energy',

#         '(2)_zuluft_vor_entfeuchter_kon_g/kg_energy',
#        '(3)_zuluft_nach_entfeuchter_kon_g/kg_energy',
#        #'(4)_frostschutz_%_energy', 
       
#        '(5)_vor_vent_hauptzuluft_temp_c_energy',
#        '(7)_zuluft_turmt_temp_c_energy', '(9)_abluft2_kon_g/kg_energy',

#     #    '(11)_wm_mas_kg/h_energy', 

#        '(12)_mpt_fb(mpt?)_kg/h_energy',
#        '(13)_lanzen_mas_kg/h_energy', '(14)_filter_mas_kg/h_energy',
#        #'(15)_filter_mas_kg/h_energy', 
#        '(16)_konditionierung_mas_kg/h_energy',
#        '(8)_abluft_vol_m3/h_energy', '(23)_f10_speise_temp_c_energy',
#        '(23)_f11_speise_temp_c_energy',
#        '(19)_zuluft_vor_entfeuchter_temp_c_energy', 'dampf_nmb_energy',
#        'nach_nt_(c)_energy', 
       
#        'speise_current_kg/h_energy',
#     #    'f10_speise_kg/h_energy',
#     #    'f11_speise_kg/h_energy',
#     #    'f10_speise_kg/m³_energy',
#     #    'f11_speise_kg/m³_energy', 'f10_speise_l/h_energy',
#     #    'f11_speise_l/h_energy',
       
#        '(6)_nach_recu_reg_temp_c_old_energy',
#        '(18)_leistung_turmF_lufterhitzer_kw_energy',
#        '(17)_leistung_turmF_luftentfeuchter_kw_energy',
#        '(6)_nach_recu_reg_temp_c_energy', 
#        #'id_original_energy',
#        '(10)_abluft_temp_c_energy', '(8)_abluft_mas_kg/h_energy',

#        'ef_temperature_2m_energy', 'ef_relative_humidity_2m_energy',
#        #'ef_apparent_temperature_energy', 'ef_precipitation_energy',
#        #'ef_wind_speed_10m_energy', 'ef_wind_direction_100m_energy',
#        'ef_global_tilted_irradiance_energy',
       
#     #    '(21)_zuluft_turm_mas_kg/h_energy', '(31)_q_waerme_recu_kw_energy',
#     #    '(22)_t_waermereg_c_energy',
#     #    '(29)_q_lufterwearmung_von_T6_nach_T7_kw_energy',
#     #    '(30)_q_lufterwearmung_brechenet_T5_T6_und_T6_T7_und_berechnete_recu_kw_energy',
#     #    '(33)_q_lufterwearmung_dampgemessen_und_berechnete_recu_kw_energy',
#     #    '(34)_q_lufterwearmung_berechnet_nach_temp_in_out_kw_energy',

#        'activity_energy', 'activity_log', 'timestamp_start_log',
#        'timestamp_end_log', 'object_attributes_log',
#        'higher_level_activity_log', 'object_type_log', 'object_log',
#        'case_id_log']

# df_expanded = df_expanded[relevant_columns].copy()

# files_folder_gold_datasets = files_folder_gold / 'datasets'
# files_folder_gold_datasets.mkdir(parents=True, exist_ok=True)

# df_expanded.to_parquet(files_folder_gold_datasets / "df_expanded.parquet", index=False)
# df_event_log.to_parquet(files_folder_gold_datasets / "df_event_log.parquet", index=False)
# df_production_plan.to_parquet(files_folder_gold_datasets / "df_production_plan.parquet", index=False)
# # %%
# display(df_expanded['activity_log'].value_counts())

# display(df_expanded['activity_log'].unique())

# #plot_activity_sensor_curves_by_index(df_expanded)

# display(df_expanded.columns)
# display(df_expanded)

# %%

######## Process 4 ########

process = 4

# Define the folder path
files_folder_silver = folder_silver_base / f'process_{process}'
files_folder_gold = folder_gold_base / f'process_{process}'
files_folder_gold.mkdir(parents=True, exist_ok=True)

objects_to_analyze = ['Erhitzer']

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

# Total material per case: integrate Vorlaufpumpe flow rate (l/h) over actual measurement intervals
_vol = df_expanded[['case_id_log', 'datetime_energy', 'Vorlaufpumpe_30110FT301_5s_energy']].copy()
_vol = _vol.sort_values(['case_id_log', 'datetime_energy'])
_vol['_dt_h'] = _vol.groupby('case_id_log')['datetime_energy'].diff().dt.total_seconds() / 3600
_vol['_dt_h'] = _vol['_dt_h'].fillna(0)
_vol['_liters'] = _vol['Vorlaufpumpe_30110FT301_5s_energy'] * _vol['_dt_h']
_case_total = _vol.groupby('case_id_log')['_liters'].sum()
df_expanded['object_attributes_log'] = df_expanded.apply(
    lambda row: {**row['object_attributes_log'], 'total_material': _case_total.get(row['case_id_log'], None)}
    if isinstance(row['object_attributes_log'], dict) else row['object_attributes_log'],
    axis=1
)

# Propagate total_material back into df_event_log (built before df_expanded)
df_event_log['object_attributes'] = df_event_log.apply(
    lambda row: {**row['object_attributes'], 'total_material': _case_total.get(row['case_id'], None)}
    if isinstance(row['object_attributes'], dict) else {'total_material': _case_total.get(row['case_id'], None)},
    axis=1
)


## Get the production plan
df = df_event_log.copy()

df = df[df['object'].isin(objects_to_analyze)]

df = df[['case_id', 'activity', 'timestamp_start', 'timestamp_end', 'object_attributes']]

df = df.dropna(subset=['case_id'])

df = df.drop_duplicates(subset=['case_id'])

df_production_plan = df.copy()

print("Expanded production plan")
print(df_production_plan)


relevant_columns = ['case_id_log', 'activity_log', 'timestamp_start_log',
       'timestamp_end_log', 'object_log', 'object_type_log',
       'higher_level_activity_log', 'object_attributes_log', 
       'datetime_energy',

       'temp_Auslauf_EG_(WT2)_5s_energy', 'temp_Einlauf_EG_(WT_2)_5s_energy',
       'flow_Kuehlturmwasser_30120FT701_5s_energy',
       'flow_Kaltwasser_(WT7)_5s_energy', 'Vorlaufpumpe_30110FT301_5s_energy',
       'Produkt_Einlauf_30110TT001_5s_energy',
       'vor_Vorwärmer_(WT_2)_5s_energy', 'Kuehlturmwassertemp_(WT6)_5s_energy',
       'Kaltwassertemp_(WT_7)_5s_energy', 'nach_Kuehler_(WT7)_5s_energy',
       'temp_nach_Kuehlturmkuehler_(WT6)_5s_energy',
       'Fuellstand_Steriltank_30140LT001_5s_energy',
       'Fuellstand_Steriltank_30141LT001_5s_energy',
       'flow_Dampf_WT3a/5a)_5s_energy',
       'flow_Heisswasser_30120FT721(WT5a)_5s_energy',
       'temp_nach_WR2,_vor_Druckerhoehungspumpe_(WT4)_5s_energy',
       'temp_nach_Erhitzer_(WT5)_5s_energy', 'temp_nach_WR2_(WT2)_5s_energy',
       'temp_nach_Austauscher_2_(WT4)_5s_energy',
       'Druck_HW_Anwaermer_(WT3a)_5s_energy',
       'temp_HW_Anwaermer_(WT3a)_5s_energy',
       'temp_Produkt_Einlauf_30110TT001_1h_energy',
       'flow_Vorlaufpumpe_30110FT301_1h_energy',
       'temp_vor_Vorwärmer_(WT_2)_1h_energy', 'strom_PERT2_KZE_5s_energy',
       'dampf_PET2_KZE_5s_energy', 
       #'strom_gesamt_PERT2_KZE_15m_energy',
       #'dampf_gesamt_PET2_KZE_15m_energy',
       'A016_temp_vor_Vorwärmer_30121TT001_(WT_2)_5s_energy',
       'A047_temp_Produkt_Einlauf_30110TT001_5s_energy',
       'A054_flow_Vorlaufpumpe_30110FT301_5s_energy',

       'ef_temperature_2m_energy', 'ef_relative_humidity_2m_energy',
       #'ef_apparent_temperature_energy', 'ef_precipitation_energy',
       #'ef_wind_speed_10m_energy', 'ef_wind_direction_100m_energy',
       'ef_global_tilted_irradiance_energy'
       ]

df_expanded = df_expanded[relevant_columns].copy()

files_folder_gold_datasets = files_folder_gold / 'datasets'
files_folder_gold_datasets.mkdir(parents=True, exist_ok=True)

df_expanded.to_parquet(files_folder_gold_datasets / "df_expanded.parquet", index=False)
df_event_log.to_parquet(files_folder_gold_datasets / "df_event_log.parquet", index=False)
df_production_plan.to_parquet(files_folder_gold_datasets / "df_production_plan.parquet", index=False)

# %%
display(df_expanded['activity_log'].value_counts())

display(df_expanded['activity_log'].unique())

#plot_activity_sensor_curves_by_index(df_expanded)

display(df_expanded.columns)
display(df_expanded)



#%%

######## Process 3 ########

process = 3

# Define the folder path
files_folder_silver = folder_silver_base / f'process_{process}'
files_folder_gold = folder_gold_base / f'process_{process}'
files_folder_gold.mkdir(parents=True, exist_ok=True)


df = pd.read_parquet(files_folder_silver / "data_prepared_for_analysis.parquet")

relevant_columns = ['datetime', 
        
        'f11_speise_den_kg/m3', '(10)_abluft_temp_c',
       'f10_speise_den_kg/m3', '(12)_mpt_fb(mpt?)_kg/h', 'f11_speise_mas_kg/h',
       'f11_speise_vol_l/h', '(6)_nach_recu_reg_temp_c',
       'dampf_nassmischbereich', '(3)_zuluft_nach_entfeuchter_kon_g/kg',
       'f10_speise_vol_l/h', '(9)_abluft_kon_g/kg',
       '(16)_konditionierung_mas_kg/h', '(14)_filter_mas_kg/h', 
       #'(1)_status',
       'f10_speise_mas_kg/h', '(5)_vor_vent_hauptzuluft_temp_c',
       '(7)_zuluft_turm_temp_c', '(2)_zuluft_vor_entfeuchter_kon_g/kg',
       '(13)_lanzen_mas_kg/h', '(8)_abluft_vol_m3/h',
       '(17)_leistung_turmF_luftentfeuchter_kw', 
        
        '(8)_abluft_mas_kg/h',
       '(21)_zuluft_turm_mas_kg/h', '(31)_q_waerme_recu_kw',
       '(22)_t_waermereg_c', '(29)_q_lufterwearmung_von_T6_nach_T7_kw',
       '(30)_q_lufterwearmung_brechenet_T5_T6_und_T6_T7_und_berechnete_recu_kw',
       '(33)_q_lufterwearmung_dampgemessen_und_berechnete_recu_kw',
       '(34)_q_lufterwearmung_berechnet_nach_temp_in_out_kw',
       '(35)_speise_max_kg/h',
       
       'temperature_2m',
       'relative_humidity_2m', 
       
       'status_name'
       
       ]

display(df)
print(df.columns)

df = df[relevant_columns].copy()

status_to_keep = ['Produktion', 
        #'Stopp Produkt HPPx', 
        'zurück speisen',
       'Stabilisiert und angepasst', 
       #'Grundstellung',
       #'Start Produkt HPPx', 
       'Feed vorwärts',
       'Fließbett starten für Produktion', 
       #'Wasserlaufphase deaktivieren',
       #None, #'Starten Wasserfahrt', 
       #'HPPx für Produkt anpassen'
       ]

df = df[df['status_name'].isin(status_to_keep)].copy()

df['status_name'].value_counts()


### Build event log: one case_id per day, detect consecutive activity spans

df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

df['case_id'] = df['datetime'].dt.date.astype(str)

# Detect consecutive runs of the same status within a day
df['_activity_change'] = (
    (df['status_name'] != df['status_name'].shift()) |
    (df['case_id'] != df['case_id'].shift())
).cumsum()

span_info = df.groupby('_activity_change').agg(
    timestamp_start=('datetime', 'min'),
    timestamp_end=('datetime', 'max')
).reset_index()

df = df.merge(span_info, on='_activity_change')
df = df.drop(columns='_activity_change')

# Event log: one row per consecutive activity span
df_event_log = (
    df.groupby(['case_id', 'status_name', 'timestamp_start', 'timestamp_end'])
    .size()
    .reset_index(drop=True)
    .pipe(lambda _: df[['case_id', 'status_name', 'timestamp_start', 'timestamp_end']]
          .drop_duplicates()
          .rename(columns={'status_name': 'activity'})
          .sort_values('timestamp_start')
          .reset_index(drop=True))
)

print("Event log")
print(df_event_log)


### Build df_expanded: flat timeseries with log info attached

energy_cols = [c for c in df.columns if c not in [
    'datetime', 'status_name', 'case_id', 'temperature_2m', 'relative_humidity_2m',
    'timestamp_start', 'timestamp_end'
]]
ef_cols = ['temperature_2m', 'relative_humidity_2m']

df_expanded = df.rename(columns={
    'status_name': 'activity_log',
    'case_id': 'case_id_log',
    'datetime': 'datetime_energy',
    'timestamp_start': 'timestamp_start_log',
    'timestamp_end': 'timestamp_end_log',
})
df_expanded = df_expanded.rename(columns={col: f'{col}_energy' for col in energy_cols + ef_cols})

print("Expanded df with sensor data")
print(df_expanded)

# Combine f10/f11 feed rates: element-wise max
df_expanded['speise_current_kg/h_energy'] = df_expanded[['f10_speise_mas_kg/h_energy', 'f11_speise_mas_kg/h_energy']].max(axis=1)

# Total material per case: integrate feed rate (kg/h) over actual measurement intervals
_vol = df_expanded[['case_id_log', 'datetime_energy', 'speise_current_kg/h_energy']].copy()
_vol = _vol.sort_values(['case_id_log', 'datetime_energy'])
_vol['_dt_h'] = _vol.groupby('case_id_log')['datetime_energy'].diff().dt.total_seconds() / 3600
_vol['_dt_h'] = _vol['_dt_h'].fillna(0)
_vol['_kg'] = _vol['speise_current_kg/h_energy'] * _vol['_dt_h']
_case_total = _vol.groupby('case_id_log')['_kg'].sum()

# Inject total_material into df_expanded as object_attributes_log
df_expanded['object_attributes_log'] = df_expanded['case_id_log'].map(
    lambda cid: {'total_material': _case_total.get(cid, None)}
)

# Propagate to event log so extract_process() sees attr_total_material
df_event_log['object_attributes'] = df_event_log['case_id'].map(
    lambda cid: {'total_material': _case_total.get(cid, None)}
)


### Build production plan: one row per case_id (day)

df_production_plan = (
    df_event_log.groupby('case_id')
    .agg(timestamp_start=('timestamp_start', 'min'),
         timestamp_end=('timestamp_end', 'max'))
    .reset_index()
)
_attrs_map = df_event_log.groupby('case_id')['object_attributes'].first()
df_production_plan['object_attributes'] = df_production_plan['case_id'].map(_attrs_map)

print("Production plan")
print(df_production_plan)


### Select relevant columns for df_expanded

energy_cols_energy = [f'{c}_energy' for c in energy_cols]
ef_cols_energy = [f'{c}_energy' for c in ef_cols]

relevant_columns_p5 = (
    ['case_id_log', 'activity_log', 'timestamp_start_log', 'timestamp_end_log', 'datetime_energy',
     'object_attributes_log', 'speise_current_kg/h_energy']
    + energy_cols_energy
    + ef_cols_energy
)

df_expanded = df_expanded[relevant_columns_p5].copy()


### Save datasets

files_folder_gold_datasets = files_folder_gold / 'datasets'
files_folder_gold_datasets.mkdir(parents=True, exist_ok=True)

df_expanded.to_parquet(files_folder_gold_datasets / "df_expanded.parquet", index=False)
df_event_log.to_parquet(files_folder_gold_datasets / "df_event_log.parquet", index=False)
df_production_plan.to_parquet(files_folder_gold_datasets / "df_production_plan.parquet", index=False)


# %%
