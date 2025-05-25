import pandas as pd
import requests
import zipfile
import io
import os
import re
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def download_historical_data(data_dir='data/historical_pm25'):
    """
    Downloads and extracts historical PM2.5 data.

    Args:
        data_dir (str): Directory to store the downloaded and extracted data.

    Returns:
        str: Path to the directory containing Excel files.
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    url = 'http://air4thai.pcd.go.th/webV2/history/pm25_2011_2020.zip'
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(data_dir)
        print(f"Data downloaded and extracted to {data_dir}")
        return data_dir
    except requests.exceptions.RequestException as e:
        print(f"Error downloading data: {e}")
        return None
    except zipfile.BadZipFile as e:
        print(f"Error extracting zip file: {e}")
        return None

def load_excel_station_data(excel_path, station_id_pattern=r'^[0-9].*t$'):
    """
    Loads PM2.5 data from a single Excel file.

    Args:
        excel_path (str): Path to the Excel file.
        station_id_pattern (str): Regex pattern to filter station ID columns.

    Returns:
        pd.DataFrame: DataFrame with DatetimeIndex and station PM2.5 values.
    """
    try:
        df = pd.read_excel(excel_path)
        if 'Date' not in df.columns:
            print(f"Warning: 'Date' column not found in {excel_path}. Skipping this file.")
            return pd.DataFrame()

        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')

        station_columns = [col for col in df.columns if re.match(station_id_pattern, str(col))]
        df_station = df[station_columns]
        
        return df_station
    except FileNotFoundError:
        print(f"Error: File not found at {excel_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading Excel file {excel_path}: {e}")
        return pd.DataFrame()

def load_all_historical_data(historical_data_path):
    """
    Loads all historical PM2.5 data from Excel files in a directory.

    Args:
        historical_data_path (str): Path to the directory containing historical Excel files.

    Returns:
        pd.DataFrame: Combined DataFrame with data from all years.
    """
    all_data = []
    if not os.path.exists(historical_data_path):
        print(f"Error: Historical data path {historical_data_path} does not exist.")
        return pd.DataFrame()

    for filename in os.listdir(historical_data_path):
        if filename.endswith('.xlsx') or filename.endswith('.xls'):
            excel_path = os.path.join(historical_data_path, filename)
            print(f"Loading data from {excel_path}...")
            df_year = load_excel_station_data(excel_path)
            if not df_year.empty:
                all_data.append(df_year)
    
    if not all_data:
        print("No data loaded. Returning empty DataFrame.")
        return pd.DataFrame()

    combined_df = pd.concat(all_data)
    combined_df = combined_df.sort_index()
    # Handle potential duplicates by taking the mean (or choose another strategy)
    combined_df = combined_df.groupby(combined_df.index).mean() 
    
    print("Successfully loaded all historical data.")
    return combined_df

def fetch_hourly_data(start_date_str, end_date_str, station_list, data_dir='data/hourly_pm25'):
    """
    Fetches recent hourly PM2.5 data from the Air4Thai API.

    Args:
        start_date_str (str): Start date in 'YYYY-MM-DD' format.
        end_date_str (str): End date in 'YYYY-MM-DD' format.
        station_list (list): List of station IDs (e.g., ['50t', '52t']).
        data_dir (str): Directory to save fetched data.

    Returns:
        pd.DataFrame: DataFrame with DatetimeIndex and station PM2.5 values.
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    all_station_data = []

    for station_id in station_list:
        params = {
            'sdate': start_date_str,
            'edate': end_date_str,
            'stime': '00',
            'etime': '23',
            'param': 'PM25',
            'station': station_id
        }
        url = "http://air4thai.com/forweb/getHistoryData.php"
        
        try:
            print(f"Fetching data for station {station_id} from {start_date_str} to {end_date_str}...")
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if not data or 'stations' not in data or not data['stations']:
                print(f"No data returned for station {station_id}.")
                continue

            station_data = data['stations'][0]['data']
            df_station = pd.DataFrame(station_data)
            
            if df_station.empty or 'DATETIMEDATA' not in df_station.columns:
                print(f"DATETIMEDATA not found or empty data for station {station_id}")
                continue

            df_station['DATETIMEDATA'] = pd.to_datetime(df_station['DATETIMEDATA'])
            df_station = df_station.set_index('DATETIMEDATA')
            df_station = df_station.rename(columns={'PM25': station_id})
            df_station = df_station[[station_id]] # Keep only the station column
            all_station_data.append(df_station)

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for station {station_id}: {e}")
            continue
        except (ValueError, KeyError) as e: # Handles JSON parsing errors or missing keys
            print(f"Error parsing JSON response for station {station_id}: {e}")
            continue
            
    if not all_station_data:
        print("No hourly data fetched. Returning empty DataFrame.")
        return pd.DataFrame()

    # Combine data from all stations
    combined_df = pd.concat(all_station_data, axis=1)
    combined_df = combined_df.sort_index()
    
    # Save to cache
    cache_filename = f"hourly_data_{start_date_str}_to_{end_date_str}.pkl"
    cache_path = os.path.join(data_dir, cache_filename)
    try:
        combined_df.to_pickle(cache_path)
        print(f"Fetched data saved to {cache_path}")
    except Exception as e:
        print(f"Error saving data to cache: {e}")
        
    return combined_df

def preprocess_data(df, interpolation_method='linear', scaler_type='minmax', scaler=None):
    """
    Performs basic data cleaning (interpolation) and normalization/scaling.

    Args:
        df (pd.DataFrame): Input DataFrame with DatetimeIndex and station PM2.5 values.
        interpolation_method (str): Pandas interpolation method (e.g., 'linear', 'time').
        scaler_type (str): Type of scaler ('minmax' or 'standard') if a new scaler is to be fitted.
        scaler (sklearn.preprocessing.Scaler, optional): A pre-fitted scaler to use for transforming data.
                                                       If None, a new scaler is fitted.

    Returns:
        tuple: (pd.DataFrame, sklearn.preprocessing.Scaler)
               The preprocessed DataFrame and the (fitted or provided) scaler.
    """
    if df.empty:
        print("Input DataFrame is empty. Skipping preprocessing.")
        return df, scaler # Return original df and potentially None scaler

    # Convert all columns to numeric, coercing errors
    # Create a copy to avoid SettingWithCopyWarning if df is a slice
    df_processed = df.copy()
    for col in df_processed.columns:
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')

    # Interpolate missing values
    if interpolation_method:
        df_processed = df_processed.interpolate(method=interpolation_method, axis=0) # Interpolate column-wise

    # Handle potential all-NaN columns after interpolation
    df_processed = df_processed.dropna(axis=1, how='all')
    if df_processed.empty:
        print("DataFrame became empty after dropping all-NaN columns post-interpolation. Cannot scale.")
        return df_processed, scaler

    # If a scaler is provided, use it
    if scaler:
        print(f"Using provided scaler to transform data.")
        # Ensure columns match if possible, though scaler should handle it based on fitted features
        # Scaler expects a NumPy array
        scaled_values = scaler.transform(df_processed)
        df_scaled = pd.DataFrame(scaled_values, index=df_processed.index, columns=df_processed.columns)
    else:
        # Initialize a new scaler if none provided
        print(f"Fitting a new '{scaler_type}' scaler.")
        if scaler_type == 'minmax':
            scaler = MinMaxScaler()
        elif scaler_type == 'standard':
            scaler = StandardScaler()
        else:
            print(f"Warning: Unknown scaler_type '{scaler_type}'. No scaling will be applied.")
            return df_processed, None # Return df_processed as it is, and None scaler
        
        # Fit and transform data
        scaled_values = scaler.fit_transform(df_processed)
        df_scaled = pd.DataFrame(scaled_values, index=df_processed.index, columns=df_processed.columns)
        print(f"Data preprocessed using {interpolation_method} interpolation and new {scaler_type} scaling.")
    
    return df_scaled, scaler

if __name__ == '__main__':
    # Example Usage (Optional: for testing the functions directly)
    
    # 1. Download historical data
    historical_dir = download_historical_data() # Downloads to 'data/historical_pm25'
    
    if historical_dir:
        # 2. Load all historical data
        df_historical = load_all_historical_data(historical_dir)
        if not df_historical.empty:
            print("\n--- Historical Data Sample ---")
            print(df_historical.head())
            print(f"Shape of historical data: {df_historical.shape}")

            # 3. Preprocess historical data
            df_historical_processed, historical_scaler = preprocess_data(df_historical.copy()) # Use .copy() to avoid modifying original
            if not df_historical_processed.empty:
                print("\n--- Processed Historical Data Sample ---")
                print(df_historical_processed.head())
                print(f"Scaler used: {historical_scaler}")

    # 4. Fetch recent hourly data
    # Example: Fetch data for stations '50t' (Bangkok) and '67t' (Chiang Mai) for a specific period
    # Note: API might have limitations on date range and availability for certain stations.
    # Adjust dates and stations as needed for testing.
    # stations_to_fetch = ['50t', '52t', '53t', '54t', '58t', '59t', '61t'] # Example Bangkok stations
    stations_to_fetch = ['70t', '71t', '74t'] # Example Chiang Mai stations
    
    # Ensure the date range is recent enough for API data availability
    # For example, fetch data for the last 7 days
    from datetime import datetime, timedelta
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    print(f"\nFetching hourly data from {start_date_str} to {end_date_str} for stations: {stations_to_fetch}")
    df_hourly = fetch_hourly_data(start_date_str, end_date_str, stations_to_fetch) # Saves to 'data/hourly_pm25'
    
    if not df_hourly.empty:
        print("\n--- Hourly Data Sample ---")
        print(df_hourly.head())
        print(f"Shape of hourly data: {df_hourly.shape}")

        # 5. Preprocess hourly data
        df_hourly_processed, hourly_scaler = preprocess_data(df_hourly.copy())
        if not df_hourly_processed.empty:
            print("\n--- Processed Hourly Data Sample ---")
            print(df_hourly_processed.head())
            print(f"Scaler used: {hourly_scaler}")

    print("\n--- Script Execution Finished ---")


def create_sequences(data_array, n_past_steps, n_future_steps):
    """
    Creates input (X) and target (y) sequences for time series forecasting from a NumPy array.
    This version is designed for grid-like data (e.g., output from map_stations_to_grid).

    Args:
        data_array (np.ndarray): Input NumPy array, typically of shape 
                                 (n_time_samples, grid_rows, grid_cols, n_features_per_cell).
                                 For PM2.5, n_features_per_cell is usually 1.
        n_past_steps (int): Number of past time steps to use as input features.
        n_future_steps (int): Number of future time steps to predict.

    Returns:
        tuple: (np.ndarray, np.ndarray)
               X: Input sequences of shape 
                  (num_sequences, n_past_steps, grid_rows, grid_cols, n_features_per_cell)
               y: Target sequences of shape 
                  (num_sequences, n_future_steps, grid_rows, grid_cols, n_features_per_cell)
    """
    if not isinstance(data_array, np.ndarray) or data_array.ndim < 2: # Basic check, expect at least (time, features)
        print("Input data must be a NumPy array with at least 2 dimensions.")
        return np.array([]), np.array([])
    
    if data_array.shape[0] < n_past_steps + n_future_steps:
        print(f"Not enough time samples in data_array ({data_array.shape[0]}) to create "
              f"sequences with {n_past_steps} past steps and {n_future_steps} future steps.")
        return np.array([]), np.array([])

    X_list, y_list = [], []
    n_total_steps = n_past_steps + n_future_steps

    for i in range(data_array.shape[0] - n_total_steps + 1):
        past_end_idx = i + n_past_steps
        future_end_idx = past_end_idx + n_future_steps
        
        X_list.append(data_array[i:past_end_idx, ...]) # Ellipsis takes care of grid_rows, grid_cols, features
        y_list.append(data_array[past_end_idx:future_end_idx, ...])
        
    if not X_list:
        print("Could not create any sequences. Check input data and step parameters.")
        return np.array([]), np.array([])

    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"Created sequences: X shape {X.shape}, y shape {y.shape}")
    return X, y


def map_stations_to_grid(data_df, grid_rows, grid_cols, station_locations=None):
    """
    Placeholder function to map station-based PM2.5 data to a 2D grid.
    The actual implementation will require station coordinates and a gridding strategy.

    Args:
        data_df (pd.DataFrame): DataFrame with DatetimeIndex and station PM2.5 values.
                                Each column is a station.
        grid_rows (int): Number of rows in the target grid.
        grid_cols (int): Number of columns in the target grid.
        station_locations (dict, optional): Dictionary mapping station IDs (columns in data_df)
                                            to their (row, col) coordinates in the grid.
                                            Example: {'station_A': (0,0), 'station_B': (0,1), ...}
                                            If None, a placeholder mapping is used.

    Returns:
        np.ndarray: A NumPy array of shape (n_samples, grid_rows, grid_cols, 1),
                    representing the data mapped to the grid.
                    The last dimension is 1 (for PM2.5 feature).
                    Returns None if data_df is empty.
    """
    if data_df.empty:
        print("Input DataFrame is empty. Cannot map to grid.")
        return None

    num_samples = len(data_df)
    # Initialize an empty grid with NaNs or zeros
    grid_data = np.full((num_samples, grid_rows, grid_cols, 1), np.nan) # (samples, rows, cols, features)

    print(f"Attempting to map {len(data_df.columns)} stations to a {grid_rows}x{grid_cols} grid.")

    if station_locations:
        for station_id, (r, c) in station_locations.items():
            if station_id in data_df.columns:
                if 0 <= r < grid_rows and 0 <= c < grid_cols:
                    grid_data[:, r, c, 0] = data_df[station_id].values
                else:
                    print(f"Warning: Station {station_id} location ({r},{c}) is outside grid dimensions.")
            else:
                print(f"Warning: Station {station_id} in station_locations not found in data_df.")
    else:
        # Placeholder: Distribute stations sequentially into the grid as a naive approach
        # This is NOT a meteorologically sound approach, just for structure.
        print("Warning: station_locations not provided. Using naive sequential mapping.")
        stations = data_df.columns
        for idx, station_id in enumerate(stations):
            r = idx // grid_cols
            c = idx % grid_cols
            if r < grid_rows:
                grid_data[:, r, c, 0] = data_df[station_id].values
            else:
                print(f"Warning: Station {station_id} (index {idx}) cannot fit into the {grid_rows}x{grid_cols} grid. Max index is {grid_rows*grid_cols-1}")
                break # Stop if we run out of grid cells
    
    # A simple check: if many NaNs remain, the mapping was likely incomplete or stations didn't fit.
    num_grid_cells = grid_rows * grid_cols
    non_nan_cells_per_sample = np.sum(~np.isnan(grid_data[0, :, :, 0])) # Check first sample
    if non_nan_cells_per_sample < min(len(data_df.columns), num_grid_cells):
        print(f"Warning: Only {non_nan_cells_per_sample} out of {num_grid_cells} grid cells "
              f"(or {len(data_df.columns)} stations) were filled in the first sample. "
              "Check station_locations and grid size.")

    print("Placeholder map_stations_to_grid executed.")
    # In a real implementation, you might interpolate NaNs if appropriate
    # For now, we return it with NaNs, which ConvLSTM might not handle well without imputation.
    return grid_data
