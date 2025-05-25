import os
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

# Assuming data_utils and models are in the PYTHONPATH or same directory level
from data_utils.loader import (
    download_historical_data, 
    load_all_historical_data,
    fetch_hourly_data, 
    preprocess_data, 
    map_stations_to_grid, 
    create_sequences
)
from models.pm25_model import build_convlstm_model

# --- Training Parameters ---
HISTORICAL_DATA_DIR = 'data/historical_pm25'
HOURLY_DATA_DIR = 'data/hourly_pm25' # For future use
LOG_DIR = 'logs'
MODEL_NAME_PREFIX = 'pm25_convlstm'

# Derived log/weight paths
WEIGHTS_DIR = os.path.join(LOG_DIR, 'weights')
TENSORBOARD_LOG_DIR = os.path.join(LOG_DIR, 'tensorboard')

# Model and Data Configuration
N_PAST_STEPS = 24  # Number of past hours to use as input
N_FUTURE_STEPS = 12 # Number of future hours to predict
GRID_ROWS = 10     # Target grid rows
GRID_COLS = 10     # Target grid columns
# N_INPUT_FEATURES is implicitly 1 (PM2.5) after map_stations_to_grid
# N_OUTPUT_FEATURES is implicitly GRID_ROWS * GRID_COLS for the model's dense layer

# Training Hyperparameters
EPOCHS = 50 # Number of epochs to train for
BATCH_SIZE = 32 # Batch size for training
LEARNING_RATE = 0.001 # Learning rate for the optimizer
TEST_SPLIT_SIZE = 0.2 # Proportion of data to use for validation

def main():
    print("--- Starting PM2.5 Forecasting Model Training ---")

    # --- 1. Create Directories ---
    os.makedirs(HISTORICAL_DATA_DIR, exist_ok=True)
    os.makedirs(HOURLY_DATA_DIR, exist_ok=True)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)
    print(f"Required directories ensured: {HISTORICAL_DATA_DIR}, {HOURLY_DATA_DIR}, {WEIGHTS_DIR}, {TENSORBOARD_LOG_DIR}")

    # --- 2. Load Data ---
    print("\n--- Loading Data ---")
    historical_data_path = download_historical_data(data_dir=HISTORICAL_DATA_DIR)
    if not historical_data_path:
        print("Failed to download or locate historical data. Exiting.")
        return

    df_historical = load_all_historical_data(historical_data_path)
    if df_historical.empty:
        print("No historical data loaded. Exiting.")
        return
    print(f"Historical data loaded. Shape: {df_historical.shape}")
    
    # (Optional: Fetch and merge recent hourly data - to be implemented later if needed)
    # For now, we proceed with only historical data.

    # --- 3. Preprocess Data ---
    print("\n--- Preprocessing Data ---")
    # Using a copy for preprocessing to keep the original df_historical intact
    # preprocess_data now returns the scaler
    df_processed, scaler = preprocess_data(df_historical.copy(), interpolation_method='linear', scaler_type='minmax', scaler=None)
    if df_processed.empty:
        print("Data preprocessing resulted in an empty DataFrame. Exiting.")
        return
    
    # Save the scaler
    import joblib
    SCALER_SAVE_PATH = os.path.join(LOG_DIR, 'data_scaler.joblib')
    joblib.dump(scaler, SCALER_SAVE_PATH)
    print(f"Scaler saved to {SCALER_SAVE_PATH}")
    print(f"Data preprocessed. Shape: {df_processed.shape}. Scaler: {scaler}")

    # Map station data to grid
    # station_locations would ideally be loaded from a config file or defined based on actual geography.
    # Using None for now, map_stations_to_grid will use its naive sequential mapping.
    grid_data = map_stations_to_grid(df_processed, GRID_ROWS, GRID_COLS, station_locations=None)
    if grid_data is None or grid_data.size == 0:
        print("Failed to map station data to grid. Exiting.")
        return
    # Expected shape: (n_time_samples, GRID_ROWS, GRID_COLS, 1)
    print(f"Data mapped to grid. Shape: {grid_data.shape}")
    
    # Handle potential NaNs from grid mapping (e.g., if not all cells were filled)
    # A common strategy is to fill with a value (e.g., 0 after scaling, or mean/median)
    # For ConvLSTM, it's often better if the input doesn't have NaNs.
    if np.isnan(grid_data).any():
        print(f"NaNs found in grid_data. Filling with 0. Percentage of NaNs: {np.isnan(grid_data).mean()*100:.2f}%")
        grid_data = np.nan_to_num(grid_data, nan=0.0) # Replace NaNs with 0

    # --- 4. Create Sequences ---
    print("\n--- Creating Sequences ---")
    # X shape: (num_sequences, N_PAST_STEPS, GRID_ROWS, GRID_COLS, 1)
    # y_grid shape: (num_sequences, N_FUTURE_STEPS, GRID_ROWS, GRID_COLS, 1)
    X, y_grid = create_sequences(grid_data, N_PAST_STEPS, N_FUTURE_STEPS)
    if X.size == 0 or y_grid.size == 0:
        print("Failed to create sequences. Exiting.")
        return
    print(f"Sequences created. X shape: {X.shape}, y_grid shape: {y_grid.shape}")

    # Reshape y_grid for the model's output layer.
    # The model's final Dense layer outputs (N_FUTURE_STEPS * N_OUTPUT_FEATURES),
    # where N_OUTPUT_FEATURES = GRID_ROWS * GRID_COLS.
    # The Reshape layer in the model then converts this to (N_FUTURE_STEPS, N_OUTPUT_FEATURES).
    # So, the target y should match this shape after the batch dimension.
    # y_target shape: (num_sequences, N_FUTURE_STEPS, GRID_ROWS * GRID_COLS)
    y_target = y_grid.reshape(y_grid.shape[0], N_FUTURE_STEPS, GRID_ROWS * GRID_COLS * y_grid.shape[-1])
    print(f"Target y reshaped for model output. y_target shape: {y_target.shape}")
    
    # --- 5. Split Data ---
    print("\n--- Splitting Data ---")
    X_train, X_val, y_train, y_val = train_test_split(X, y_target, test_size=TEST_SPLIT_SIZE, random_state=42)
    print(f"Data split into training and validation sets:")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

    # --- 6. Build Model ---
    print("\n--- Building Model ---")
    # Input shape for ConvLSTM2D: (n_past_steps, n_rows, n_cols, n_features_per_cell)
    input_model_shape = (N_PAST_STEPS, GRID_ROWS, GRID_COLS, X.shape[-1]) # X.shape[-1] is n_features_per_cell (should be 1)
    n_output_features_model = GRID_ROWS * GRID_COLS * y_grid.shape[-1] # Total features to predict per future step

    model = build_convlstm_model(
        input_shape=input_model_shape,
        n_future_steps=N_FUTURE_STEPS,
        n_output_features=n_output_features_model, # This is GRID_ROWS * GRID_COLS * features_per_cell
        learning_rate=LEARNING_RATE
    )
    # model.summary() is called within build_convlstm_model

    # --- 7. Define Callbacks ---
    print("\n--- Defining Callbacks ---")
    # Model Checkpoint
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_filename = f"{MODEL_NAME_PREFIX}_{timestamp}_best.keras" # Use .keras for modern format
    checkpoint_path = os.path.join(WEIGHTS_DIR, model_filename)
    model_checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        save_best_only=True,
        monitor='val_loss',
        verbose=1
    )
    print(f"Model checkpoints will be saved to: {checkpoint_path}")

    # TensorBoard
    tensorboard_run_log_dir = os.path.join(TENSORBOARD_LOG_DIR, f"{MODEL_NAME_PREFIX}_{timestamp}")
    tensorboard_callback = TensorBoard(
        log_dir=tensorboard_run_log_dir,
        histogram_freq=1 # Log histograms for weights and biases once per epoch
    )
    print(f"TensorBoard logs will be saved to: {tensorboard_run_log_dir}")

    # Early Stopping
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=10, # Number of epochs with no improvement after which training will be stopped
        verbose=1,
        restore_best_weights=True # Restores model weights from the epoch with the best value of the monitored quantity.
    )
    print("Early stopping configured with patience 10 on val_loss.")

    callbacks_list = [model_checkpoint, tensorboard_callback, early_stopping_callback]

    # --- 8. Train Model ---
    print("\n--- Training Model ---")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=callbacks_list,
        verbose=1 # Or 2 for less output per epoch
    )

    print("\n--- Training Finished ---")
    print(f"Best validation loss: {min(history.history.get('val_loss', [float('inf')])):.4f}")
    print(f"Model training completed. Final model weights saved for the best validation epoch at {checkpoint_path} (if improved).")
    print("Check TensorBoard logs for more details on training progress.")

if __name__ == '__main__':
    main()
