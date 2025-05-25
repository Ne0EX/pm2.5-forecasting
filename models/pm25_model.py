import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, ConvLSTM2D, Flatten, Dense, Reshape, BatchNormalization, Dropout
from tensorflow.keras.models import Model

def build_convlstm_model(input_shape, n_future_steps, n_output_features, learning_rate=0.001):
    """
    Builds, compiles, and returns a ConvLSTM2D model for PM2.5 forecasting.

    Args:
        input_shape (tuple): Shape of the input data, e.g., (n_past_steps, n_rows, n_cols, n_input_features).
                             n_input_features is typically 1 for PM2.5.
        n_future_steps (int): Number of future time steps to predict for each output feature.
        n_output_features (int): Number of features/stations to predict at each future step.
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        tensorflow.keras.models.Model: Compiled Keras ConvLSTM2D model.
    """
    
    # Define the input layer
    inputs = Input(shape=input_shape) # (n_past, n_rows, n_cols, n_features_in)

    # Example ConvLSTM2D architecture
    # Layer 1
    x = ConvLSTM2D(
        filters=64, 
        kernel_size=(3, 3), 
        padding='same', 
        return_sequences=True, # True because the next ConvLSTM2D layer expects sequences
        activation='relu'
    )(inputs)
    x = BatchNormalization()(x) # Normalize after activation

    # Layer 2
    x = ConvLSTM2D(
        filters=128, 
        kernel_size=(3, 3), 
        padding='same', 
        return_sequences=False, # False as this is the last ConvLSTM layer before Flatten
        activation='relu'
    )(x)
    x = BatchNormalization()(x)
    
    # Flatten the output from ConvLSTM2D to feed into Dense layers
    # The output of ConvLSTM2D (with return_sequences=False) is (batch_size, n_rows, n_cols, filters)
    # We need to flatten it to (batch_size, n_rows * n_cols * filters)
    x = Flatten()(x)
    
    # Dense layers for further processing
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x) # Add dropout for regularization
    
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)

    # Output layer: predicts n_future_steps * n_output_features values
    # These values will be reshaped to (n_future_steps, n_output_features)
    output_units = n_future_steps * n_output_features
    outputs = Dense(output_units, activation='linear')(x)  # Linear activation for regression tasks

    # Reshape the output to the desired format: (batch_size, n_future_steps, n_output_features)
    outputs = Reshape((n_future_steps, n_output_features))(outputs)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae']) # Mean Squared Error for loss

    print("ConvLSTM Model built and compiled successfully.")
    model.summary() # Print model summary

    return model

if __name__ == '__main__':
    # Example usage:
    # These dimensions are placeholders and will depend on the actual data preprocessing.
    N_PAST_STEPS = 24  # e.g., use 24 hours of past data
    N_ROWS_GRID = 10   # e.g., data is mapped to a 10x10 grid
    N_COLS_GRID = 10   # e.g., data is mapped to a 10x10 grid
    N_INPUT_FEATURES = 1 # Just PM2.5 values as input features
    
    # For the output
    N_FUTURE_STEPS = 6   # e.g., predict 6 hours into the future
    N_OUTPUT_FEATURES = N_ROWS_GRID * N_COLS_GRID # Predict PM2.5 for each cell in the grid

    # Define the input shape for the model
    # (n_past_steps, n_rows, n_cols, n_input_features)
    example_input_shape = (N_PAST_STEPS, N_ROWS_GRID, N_COLS_GRID, N_INPUT_FEATURES)

    print(f"Building model with input shape: {example_input_shape}")
    print(f"Predicting {N_FUTURE_STEPS} future steps for {N_OUTPUT_FEATURES} output features (grid cells).")

    convlstm_model = build_convlstm_model(
        input_shape=example_input_shape,
        n_future_steps=N_FUTURE_STEPS,
        n_output_features=N_OUTPUT_FEATURES
    )
    
    # To verify the output shape, you can create some dummy data
    import numpy as np
    num_samples = 2 # Batch size
    dummy_input_data = np.random.rand(num_samples, *example_input_shape)
    dummy_predictions = convlstm_model.predict(dummy_input_data)
    print(f"Shape of dummy predictions: {dummy_predictions.shape}") # Expected: (num_samples, N_FUTURE_STEPS, N_OUTPUT_FEATURES)

    print("\nExample ConvLSTM model created and tested with dummy data.")
