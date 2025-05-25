import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib # For loading the scaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta # For potential test data selection

# Assuming data_utils is in the PYTHONPATH or same directory level
from data_utils.loader import (
    download_historical_data, # Using this to get data easily for now
    load_all_historical_data,
    preprocess_data, 
    map_stations_to_grid, 
    create_sequences
)

# --- Prediction Parameters ---
HISTORICAL_DATA_DIR = 'data/historical_pm25_test' # Can be same as training or a dedicated test set dir
MODEL_PATH = 'logs/weights/pm25_convlstm_20240501-120000_best.keras' # Placeholder - replace with actual best model path
SCALER_PATH = 'logs/data_scaler.joblib' # Path where the scaler was saved by train.py
LOG_DIR_PREDICT = 'logs/predictions'
PLOT_SAVE_DIR = os.path.join(LOG_DIR_PREDICT, 'plots')

# Model and Data Configuration (should match training)
N_PAST_STEPS = 24
N_FUTURE_STEPS = 12
GRID_ROWS = 10
GRID_COLS = 10
# N_INPUT_FEATURES is implicitly 1 (PM2.5) after map_stations_to_grid
# N_OUTPUT_FEATURES is implicitly GRID_ROWS * GRID_COLS

def inverse_transform_grid_data(grid_data_scaled, scaler, original_df_columns, grid_rows, grid_cols, station_locations=None):
    """
    Inverse transforms scaled grid data back to original PM2.5 values.
    This is complex because scaling was done per-station BEFORE gridding.

    Args:
        grid_data_scaled (np.ndarray): Scaled data in grid format, 
                                     shape (n_samples, n_steps, grid_rows, grid_cols, n_features_per_cell=1).
        scaler (sklearn.preprocessing.Scaler): The scaler object (fitted on original station data).
        original_df_columns (pd.Index): Columns of the DataFrame before gridding (station IDs).
        grid_rows (int): Number of rows in the grid.
        grid_cols (int): Number of columns in the grid.
        station_locations (dict, optional): Dictionary mapping station IDs to their (row, col) coordinates.
                                            If None, assumes naive sequential mapping was used for gridding.

    Returns:
        np.ndarray: Data in original PM2.5 scale, same shape as input grid_data_scaled.
    """
    if grid_data_scaled.ndim != 5:
        raise ValueError("grid_data_scaled must be 5D (samples, steps, rows, cols, features)")
    
    n_samples, n_steps, _, _, n_features_per_cell = grid_data_scaled.shape
    
    # Create a placeholder DataFrame that matches the structure the scaler was fit on
    # This means it needs to have the same columns (stations) as the original data used for fitting the scaler
    num_original_features = len(original_df_columns)
    
    # This will hold the inverse transformed data, station by station, then re-grid
    # It's very tricky because one grid cell might not perfectly map back to one original station
    # or some stations might not be in the grid.
    
    # Simplified approach:
    # If the scaler was fit on N_STATIONS features, and the grid is R*C,
    # we need to "unmap" grid cells to these N_STATIONS features to use inverse_transform.
    # This is only truly possible if each grid cell uniquely corresponds to a single station,
    # or if an aggregate scaler was used (which it wasn't, scaler is per-station).

    # *** Current major simplification / placeholder: ***
    # This assumes that the grid cells (GRID_ROWS * GRID_COLS) can be conceptually
    # treated as the "features" the scaler can inverse_transform, IF the scaler was
    # somehow fit on the gridded data directly. But the scaler was fit on STATION data.
    # A more robust solution requires mapping grid cells back to original stations.
    
    # For now, let's assume the scaler can be applied to each cell's timeseries if we flatten
    # the grid cells into a "features" dimension. This is an approximation.
    # The `MinMaxScaler` expects input of shape (n_samples, n_features) for `inverse_transform`.
    
    print("Warning: Inverse transform for grid data is highly complex due to per-station scaling "
          "before gridding. The current implementation is a major simplification and may not be accurate. "
          "It attempts to inverse scale cell values as if they were independent features seen by the scaler.")

    # Reshape grid_data_scaled to (n_samples * n_steps, grid_rows * grid_cols * n_features_per_cell)
    # Each (grid_row, grid_col, feature) timeseries is treated as a feature by the scaler.
    # This is not ideal as scaler was fit on specific stations.
    
    # Flatten the grid part: (n_samples, n_steps, grid_rows*grid_cols*n_features_per_cell)
    flattened_grid_for_scaling = grid_data_scaled.reshape(n_samples * n_steps, -1)
    
    # The scaler was fit on a DataFrame with `len(original_df_columns)` features.
    # If `flattened_grid_for_scaling.shape[1]` (which is grid_rows*grid_cols*n_features_per_cell)
    # is different from `scaler.n_features_in_`, this will fail or be incorrect.
    
    if flattened_grid_for_scaling.shape[1] != scaler.n_features_in_:
        print(f"Error: Number of features in grid data ({flattened_grid_for_scaling.shape[1]}) "
              f"does not match scaler's expected input features ({scaler.n_features_in_}). "
              "Cannot perform accurate inverse transform this way. Returning scaled data.")
        return grid_data_scaled # Return as is if dimensions don't match

    inversed_flat_data = scaler.inverse_transform(flattened_grid_for_scaling)
    
    # Reshape back to original grid structure
    # (n_samples * n_steps, grid_rows, grid_cols, n_features_per_cell)
    # then (n_samples, n_steps, grid_rows, grid_cols, n_features_per_cell)
    inversed_grid_data = inversed_flat_data.reshape(n_samples, n_steps, grid_rows, grid_cols, n_features_per_cell)
    
    return inversed_grid_data


def main():
    print("--- Starting PM2.5 Forecasting Model Prediction & Evaluation ---")

    # --- 1. Create Directories ---
    os.makedirs(LOG_DIR_PREDICT, exist_ok=True)
    os.makedirs(PLOT_SAVE_DIR, exist_ok=True)
    print(f"Required directories ensured: {LOG_DIR_PREDICT}, {PLOT_SAVE_DIR}")

    # --- 2. Load Scaler ---
    print(f"\n--- Loading Scaler from {SCALER_PATH} ---")
    if not os.path.exists(SCALER_PATH):
        print(f"Error: Scaler file not found at {SCALER_PATH}. Ensure train.py has been run and saved the scaler.")
        return
    try:
        scaler = joblib.load(SCALER_PATH)
        print(f"Scaler loaded successfully: {scaler}")
    except Exception as e:
        print(f"Error loading scaler: {e}. Exiting.")
        return

    # --- 3. Load Test Data ---
    # For simplicity, re-downloading and loading all historical data.
    # In a real scenario, you'd have a dedicated test set or split strategy.
    print("\n--- Loading Test Data ---")
    test_data_path = download_historical_data(data_dir=HISTORICAL_DATA_DIR) # fresh download
    if not test_data_path:
        print("Failed to download or locate test data. Exiting.")
        return

    df_all_data = load_all_historical_data(test_data_path)
    if df_all_data.empty:
        print("No test data loaded. Exiting.")
        return
    
    # Simple way to get a "test set": take the last N days or a fixed percentage
    # For this example, let's take the last 20% of the data as a pseudo-test set
    # Ensure enough data for N_PAST_STEPS + N_FUTURE_STEPS
    split_idx = int(len(df_all_data) * 0.8) 
    df_test_set = df_all_data.iloc[split_idx:]
    
    if len(df_test_set) < N_PAST_STEPS + N_FUTURE_STEPS:
        print(f"Test set too small ({len(df_test_set)} samples) for sequence creation "
              f"with N_PAST_STEPS={N_PAST_STEPS} and N_FUTURE_STEPS={N_FUTURE_STEPS}. "
              "Consider using more data or adjusting parameters. Exiting.")
        # Fallback to using a small portion of earlier data if tail is too small
        # This is just for robust example running, not ideal test practice
        fallback_end_idx = N_PAST_STEPS + N_FUTURE_STEPS + 100 # e.g. 100 sequences
        if len(df_all_data) > fallback_end_idx:
             df_test_set = df_all_data.iloc[fallback_end_idx - (N_PAST_STEPS + N_FUTURE_STEPS + 100) : fallback_end_idx]
             print(f"Warning: Using fallback test data slice. Shape: {df_test_set.shape}")
        else:
            print("Not enough data for fallback either. Exiting.")
            return

    original_test_columns = df_test_set.columns # Save for inverse transform reference
    print(f"Test data loaded. Shape: {df_test_set.shape}")

    # --- 4. Preprocess Test Data ---
    print("\n--- Preprocessing Test Data ---")
    # Use the loaded scaler to transform the test data
    df_test_processed, _ = preprocess_data(df_test_set.copy(), scaler=scaler, interpolation_method='linear')
    if df_test_processed.empty:
        print("Test data preprocessing resulted in an empty DataFrame. Exiting.")
        return
    print(f"Test data preprocessed. Shape: {df_test_processed.shape}")

    # Map station data to grid
    grid_data_test = map_stations_to_grid(df_test_processed, GRID_ROWS, GRID_COLS, station_locations=None)
    if grid_data_test is None or grid_data_test.size == 0:
        print("Failed to map test station data to grid. Exiting.")
        return
    print(f"Test data mapped to grid. Shape: {grid_data_test.shape}")
    
    if np.isnan(grid_data_test).any():
        print(f"NaNs found in test grid_data. Filling with 0. Percentage: {np.isnan(grid_data_test).mean()*100:.2f}%")
        grid_data_test = np.nan_to_num(grid_data_test, nan=0.0)

    # --- 5. Create Test Sequences ---
    print("\n--- Creating Test Sequences ---")
    X_test_seq, y_test_seq_grid = create_sequences(grid_data_test, N_PAST_STEPS, N_FUTURE_STEPS)
    if X_test_seq.size == 0 or y_test_seq_grid.size == 0:
        print("Failed to create test sequences. Exiting.")
        return
    print(f"Test sequences created. X_test_seq shape: {X_test_seq.shape}, y_test_seq_grid shape: {y_test_seq_grid.shape}")

    # Reshape y_test_seq_grid for the model's output target
    # y_target shape: (num_sequences, N_FUTURE_STEPS, GRID_ROWS * GRID_COLS * features_per_cell)
    y_test_target = y_test_seq_grid.reshape(y_test_seq_grid.shape[0], N_FUTURE_STEPS, GRID_ROWS * GRID_COLS * y_test_seq_grid.shape[-1])
    print(f"Test target y reshaped for model. y_test_target shape: {y_test_target.shape}")
    
    # For metric calculation, usually flatten the time and feature dimensions
    y_test_target_flat_for_metrics = y_test_target.reshape(y_test_target.shape[0], -1)


    # --- 6. Load Trained Model ---
    print(f"\n--- Loading Trained Model from {MODEL_PATH} ---")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}. Train the model first or check path. Exiting.")
        return
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        model.summary()
    except Exception as e:
        print(f"Error loading model: {e}. Exiting.")
        return

    # --- 7. Make Predictions ---
    print("\n--- Making Predictions ---")
    y_pred_flat = model.predict(X_test_seq) # Output is (num_sequences, N_FUTURE_STEPS * N_OUTPUT_FEATURES_MODEL)
    print(f"Predictions made. y_pred_flat shape: {y_pred_flat.shape}")

    # --- 8. Inverse Transform Predictions and True Values ---
    print("\n--- Inverse Transforming Data (Simplified) ---")
    # y_pred_flat is (samples, N_FUTURE_STEPS * GRID_R * GRID_C * n_feat_cell)
    # y_test_target_flat_for_metrics is (samples, N_FUTURE_STEPS * GRID_R * GRID_C * n_feat_cell)
    
    # Reshape predictions to grid format: (samples, N_FUTURE_STEPS, GRID_ROWS, GRID_COLS, n_feat_cell)
    n_features_per_cell = X_test_seq.shape[-1] # Should be 1 for PM2.5
    y_pred_grid_scaled = y_pred_flat.reshape(-1, N_FUTURE_STEPS, GRID_ROWS, GRID_COLS, n_features_per_cell)
    
    # y_true_grid_scaled is y_test_seq_grid
    y_true_grid_scaled = y_test_seq_grid # Shape: (samples, N_FUTURE_STEPS, GRID_ROWS, GRID_COLS, n_feat_cell)

    # Perform inverse transformation using the loaded scaler
    # This is where the major simplification/approximation happens (see function docstring)
    try:
        y_pred_grid_inversed = inverse_transform_grid_data(
            y_pred_grid_scaled.copy(), scaler, original_test_columns, GRID_ROWS, GRID_COLS
        )
        y_true_grid_inversed = inverse_transform_grid_data(
            y_true_grid_scaled.copy(), scaler, original_test_columns, GRID_ROWS, GRID_COLS
        )
        print("Inverse transform attempted.")
        # Flatten for metrics
        y_pred_flat_inversed = y_pred_grid_inversed.reshape(y_pred_grid_inversed.shape[0], -1)
        y_true_flat_inversed = y_true_grid_inversed.reshape(y_true_grid_inversed.shape[0], -1)
        
    except Exception as e:
        print(f"Error during inverse transform: {e}. Metrics will be on SCALED data.")
        # Fallback to using scaled data for metrics if inverse transform fails
        y_pred_flat_inversed = y_pred_flat # Already flat and scaled
        y_true_flat_inversed = y_test_target_flat_for_metrics # Already flat and scaled

    # --- 9. Calculate Metrics ---
    print("\n--- Calculating Metrics ---")
    # Ensure shapes are compatible for metrics
    if y_true_flat_inversed.shape != y_pred_flat_inversed.shape:
         print(f"Shape mismatch between true ({y_true_flat_inversed.shape}) and predicted ({y_pred_flat_inversed.shape}) values for metrics. Skipping.")
    else:
        rmse = np.sqrt(mean_squared_error(y_true_flat_inversed, y_pred_flat_inversed))
        mae = mean_absolute_error(y_true_flat_inversed, y_pred_flat_inversed)
        print(f"Overall Test RMSE: {rmse:.4f}")
        print(f"Overall Test MAE:  {mae:.4f}")
        print("Note: If inverse transform failed or was inaccurate, these metrics are on scaled/approximated data.")

    # --- 10. Visualize Predictions ---
    print("\n--- Visualizing Predictions ---")
    # Example: Plot for the first test sample, first future step, and a specific grid cell
    sample_idx_to_plot = 0
    # future_step_to_plot = 0 # This would be just one point in time. Let's plot all future steps for one cell.
    cell_row_to_plot = GRID_ROWS // 2
    cell_col_to_plot = GRID_COLS // 2

    if sample_idx_to_plot < y_true_grid_inversed.shape[0]:
        true_vals_cell = y_true_grid_inversed[sample_idx_to_plot, :, cell_row_to_plot, cell_col_to_plot, 0]
        pred_vals_cell = y_pred_grid_inversed[sample_idx_to_plot, :, cell_row_to_plot, cell_col_to_plot, 0]

        plt.figure(figsize=(10, 6))
        plt.plot(range(N_FUTURE_STEPS), true_vals_cell, marker='o', linestyle='-', label='True Future Values')
        plt.plot(range(N_FUTURE_STEPS), pred_vals_cell, marker='x', linestyle='--', label='Predicted Future Values')
        plt.title(f'PM2.5 Prediction vs True for Sample {sample_idx_to_plot}, Cell ({cell_row_to_plot},{cell_col_to_plot})')
        plt.xlabel('Future Time Step (Hour)')
        plt.ylabel('PM2.5 Value (potentially scaled if inverse transform failed)')
        plt.legend()
        plt.grid(True)
        
        plot_filename = f"prediction_sample{sample_idx_to_plot}_cell{cell_row_to_plot}-{cell_col_to_plot}.png"
        plot_path = os.path.join(PLOT_SAVE_DIR, plot_filename)
        plt.savefig(plot_path)
        print(f"Example plot saved to {plot_path}")
        # plt.show() # Comment out for non-interactive environments
    else:
        print("Not enough samples to plot.")

    print("\n--- Prediction and Evaluation Finished ---")

if __name__ == '__main__':
    # Replace with the actual path to your best model
    # Find the latest .keras file in logs/weights
    weights_dir = 'logs/weights'
    model_files = [os.path.join(weights_dir, f) for f in os.listdir(weights_dir) if f.endswith('.keras')]
    if model_files:
        MODEL_PATH = max(model_files, key=os.path.getctime)
        print(f"Using latest model found: {MODEL_PATH}")
    else:
        print(f"Warning: No model file found in {weights_dir}. Using placeholder: {MODEL_PATH}")
        # MODEL_PATH remains the placeholder if no model is found.

    main()
