import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry

def load_join_weather(dataset, dataset_column_name = 'dataset', 
                      timestamp_column_name = 'timestamp_utc', 
                      latitude_column_name = 'latitude', 
                      longitude_column_name = 'longitude'):
    
    df_input = dataset.copy().sort_values(by = timestamp_column_name)

    df_input[timestamp_column_name] = df_input[timestamp_column_name].dt.tz_localize(None)
    
    df_input['latitude_longitude']  = df_input[latitude_column_name].astype(str) + '-' + df_input[longitude_column_name].astype(str)

    df_list = []

    for dataset_value in df_input[dataset_column_name].unique():
        
        df_dataset_input = df_input[df_input[dataset_column_name] == dataset_value].copy()

        df_dataset_input['date'] = df_dataset_input[timestamp_column_name].dt.date
        
        first_date_dataset = str(df_dataset_input['date'].iloc[0])
        last_date_dataset = str(df_dataset_input['date'].iloc[-1])

        for latitude_longitude in df_dataset_input['latitude_longitude'].unique():
            
            df_subset = df_dataset_input[df_dataset_input['latitude_longitude'] == latitude_longitude].copy()
            
            cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
            retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
            openmeteo = openmeteo_requests.Client(session=retry_session)
    
            url = "https://archive-api.open-meteo.com/v1/archive"

            params = {
                "latitude": df_subset[latitude_column_name].iloc[0],
                "longitude": df_subset[longitude_column_name].iloc[0],
                "start_date": first_date_dataset,
                "end_date": last_date_dataset,
                "hourly": ["temperature_2m", "relative_humidity_2m", "apparent_temperature", "precipitation", "wind_speed_10m", "wind_direction_100m", "global_tilted_irradiance"]
            }
            responses = openmeteo.weather_api(url, params=params)
    
            # Process hourly data
            response = responses[0]
            hourly = response.Hourly()
            
            hourly_data = {
                "timestamp_utc": pd.date_range(start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                                      end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                                      freq=pd.Timedelta(seconds=hourly.Interval()), inclusive="left"
                                      ),
                "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
                "relative_humidity_2m": hourly.Variables(1).ValuesAsNumpy(),
                "apparent_temperature": hourly.Variables(2).ValuesAsNumpy(),
                "precipitation": hourly.Variables(3).ValuesAsNumpy(),
                "wind_speed_10m": hourly.Variables(4).ValuesAsNumpy(),
                "wind_direction_100m": hourly.Variables(5).ValuesAsNumpy(),
                "global_tilted_irradiance": hourly.Variables(6).ValuesAsNumpy()
            }
    
            hourly_dataframe = pd.DataFrame(data=hourly_data)
            hourly_dataframe['timestamp_utc'] = hourly_dataframe['timestamp_utc'].dt.tz_localize(None)
            hourly_dataframe['latitude_longitude'] = latitude_longitude

            # Merge the hourly dataframe with the original dataset based on timestamp and location
            df_subset = pd.merge(
                df_subset,
                hourly_dataframe,
                left_on=[timestamp_column_name, 'latitude_longitude'],
                right_on=['timestamp_utc', 'latitude_longitude'],
                how='left'
            )
            if timestamp_column_name != 'timestamp_utc':
                df_subset = df_subset.drop(columns=['timestamp_utc'])

            df_list.append(df_subset)

    if not df_list:
        return df_input.drop(columns=['latitude_longitude'])

    df_output = pd.concat(df_list).reset_index(drop=True)
    if 'latitude_longitude' in df_output.columns:
        df_output = df_output.drop(columns=['latitude_longitude'])
    return df_output

def get_weather(start_date, end_date, latitude, longitude):
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["temperature_2m", "relative_humidity_2m", "apparent_temperature", "precipitation", "wind_speed_10m", "wind_direction_100m", "global_tilted_irradiance"]
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process hourly data
    response = responses[0]
    hourly = response.Hourly()

    hourly_data = {
        "timestamp_utc": pd.date_range(start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                                freq=pd.Timedelta(seconds=hourly.Interval()), inclusive="left"
                                ),
        "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
        "relative_humidity_2m": hourly.Variables(1).ValuesAsNumpy(),
        "apparent_temperature": hourly.Variables(2).ValuesAsNumpy(),
        "precipitation": hourly.Variables(3).ValuesAsNumpy(),
        "wind_speed_10m": hourly.Variables(4).ValuesAsNumpy(),
        "wind_direction_100m": hourly.Variables(5).ValuesAsNumpy(),
        "global_tilted_irradiance": hourly.Variables(6).ValuesAsNumpy()
    }

    hourly_dataframe = pd.DataFrame(data=hourly_data)
    hourly_dataframe['timestamp_utc'] = hourly_dataframe['timestamp_utc'].dt.tz_localize(None)

    return hourly_dataframe