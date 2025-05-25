# PM2.5 Spatio-Temporal Forecasting using Deep Learning

This project aims to predict PM2.5 (Particulate Matter 2.5 microns) concentrations using a deep learning approach, treating spatio-temporal data as image-like sequences. The current model utilizes a Convolutional LSTM (ConvLSTM) network. Data is sourced from Air4Thai, provided by the Pollution Control Department (PCD) of Thailand.

## About The Project

-   **Goal**: To develop a model for spatio-temporal prediction of PM2.5 concentrations.
-   **Model**: The core model is based on ConvLSTM layers, designed to capture both spatial relationships and temporal dependencies from PM2.5 data structured in a grid format.
-   **Data Source**: Historical PM2.5 data is obtained from the [Air4Thai website](http://air4thai.pcd.go.th).

## Getting Started

### Dependencies

-   **Python**: Python 3.8+ is recommended.
-   **Packages**: All required Python packages are listed in `requirements.txt`. Key dependencies include TensorFlow, Pandas, NumPy, Scikit-learn, Requests, and Matplotlib.

### Installation & Setup

1.  **Clone the Repository**:
    ```sh
    git clone https://github.com/Ne0EX/pm2.5-forecasting.git
    cd pm2.5-forecasting
    ```

2.  **Set up Python Environment and Install Dependencies**:
    It is recommended to use a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
    Then, install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

3.  **Data Download**:
    The historical PM2.5 data (2011-2020) needs to be downloaded from Air4Thai. This process is automated:
    -   Running the training script `python train.py` will automatically trigger the download and extraction of historical data into the `data/historical_pm25/` directory.
    -   Alternatively, you can download the data by running the data loader script directly:
        ```sh
        python data_utils/loader.py
        ```
    This will create the `data/historical_pm25` directory and populate it with Excel files.

## Usage

### Training the Model

1.  **Run the Training Script**:
    ```sh
    python train.py
    ```
2.  **Process**: This script performs the following steps:
    -   Loads the historical PM2.5 data (downloading it if not already present).
    -   Preprocesses the data: This includes cleaning, interpolation, scaling (MinMax scaling by default), and mapping station data to a 2D grid.
    -   Saves the fitted data scaler to `logs/data_scaler.joblib`.
    -   Builds the ConvLSTM model as defined in `models/pm25_model.py`.
    -   Splits the data into training and validation sets.
    -   Trains the model using the prepared sequences.
    -   Saves the best model checkpoints (based on validation loss) to the `logs/weights/` directory (e.g., `pm25_convlstm_YYYYMMDD-HHMMSS_best.keras`).
    -   Generates TensorBoard logs in `logs/tensorboard/` for monitoring training progress.

3.  **Monitor with TensorBoard (Optional)**:
    To visualize training metrics, losses, and model graphs:
    ```sh
    tensorboard --logdir logs/tensorboard
    ```
    Open the URL provided by TensorBoard (usually `http://localhost:6006/`) in your web browser.

### Making Predictions & Evaluation

1.  **Run the Prediction Script**:
    ```sh
    python predict.py
    ```
    This script automatically uses the latest trained model found in `logs/weights/`.

2.  **Process**: This script performs the following:
    -   Loads the data scaler from `logs/data_scaler.joblib`.
    -   Loads a test dataset (currently uses a portion of the historical data; this should be a dedicated, unseen dataset in a production setup).
    -   Preprocesses the test data using the loaded scaler and maps it to the grid format.
    -   Creates input sequences for the model from the test data.
    -   Loads the latest trained ConvLSTM model from `logs/weights/`.
    -   Makes predictions on the test sequences.
    -   Attempts to inverse-transform the predictions and true values back to their original PM2.5 scale (with a note on current simplifications in this process).
    -   Calculates and prints evaluation metrics (RMSE and MAE).
    -   Saves example plots comparing true and predicted values for a sample sequence to `logs/predictions/plots/`.

## Project Structure

-   `data_utils/loader.py`: Contains functions for data fetching (Air4Thai historical data), loading data from Excel files, preprocessing (cleaning, interpolation, scaling), mapping station data to a grid, and creating input/output sequences for the model.
-   `models/pm25_model.py`: Defines the neural network architecture (currently a ConvLSTM model).
-   `train.py`: Main script for training the PM2.5 forecasting model. Handles data loading, preprocessing, model building, training, and saving artifacts.
-   `predict.py`: Main script for making predictions using a trained model, evaluating its performance, and generating sample visualizations.
-   `requirements.txt`: Lists all Python package dependencies.
-   `logs/`: This directory is created to store:
    -   `weights/`: Saved model checkpoints (Keras files).
    -   `tensorboard/`: Logs for TensorBoard visualization.
    -   `predictions/plots/`: Output plots from `predict.py`.
    -   `data_scaler.joblib`: The scaler object fitted on the training data.
-   `data/`: This directory is created to store:
    -   `historical_pm25/`: Downloaded and extracted historical PM2.5 Excel files.
    -   `hourly_pm25/`: (Currently unused by `train.py`/`predict.py` but intended for recent hourly data) Cached data from Air4Thai API calls.
-   `README.md`: This file.

## Roadmap / Future Work

The following are potential areas for improvement and future development:

-   **Station-to-Grid Mapping**: Implement a more sophisticated station-to-grid mapping strategy using actual geographic coordinates of PM2.5 monitoring stations and techniques like kriging or inverse distance weighting for interpolation onto the grid. The current naive sequential mapping is a placeholder.
-   **Inverse Scaling**: Improve the `inverse_transform_grid_data` function in `predict.py` to accurately map grid cell predictions back to station-specific PM2.5 values, considering the per-station scaling applied during preprocessing.
-   **Data Augmentation**: Explore methods for augmenting the spatio-temporal data.
-   **Model Architecture**: Experiment with different neural network architectures, hyperparameter tuning, and attention mechanisms.
-   **Test Data**: Implement a robust test data splitting strategy (e.g., a dedicated hold-out period or specific geographical regions) for more reliable model evaluation on completely unseen future data.
-   **External Features**: Incorporate other relevant data sources, such as meteorological data (wind speed/direction, temperature, humidity), traffic data, or satellite imagery, which can influence PM2.5 concentrations.
-   **Configuration Management**: Introduce configuration files (e.g., YAML or JSON) to manage parameters for data processing, model architecture, and training/prediction, instead of hardcoding them in scripts.
-   **API Integration**: Update or ensure robustness of `fetch_hourly_data` for real-time or near real-time data ingestion if the project moves towards operational forecasting.
