# data_preparation.py

import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import logging
import ast  # For safer evaluation of strings

from logger_setup import setup_logger  # Custom logger setup
logger = logging.getLogger(__name__)  # Initialize logger for this module

def interpolate_coordinates_fixed(x, y, num_points=100):
    """
    Interpolates x and y coordinates to a fixed number of points.

    Parameters:
    - x (list or array): Original x coordinates.
    - y (list or array): Original y coordinates.
    - num_points (int): Number of points to interpolate to.

    Returns:
    - np.ndarray: Concatenated array of interpolated x and y coordinates.
    """
    logger.info(f"Starting interpolation with {len(x)} points.")
    
    # Check if there are enough points to perform interpolation
    if len(x) < 2:
        logger.warning("Insufficient points for interpolation. Returning zero array.")
        return np.zeros(num_points * 2)  # Return an array of zeros if not enough points
    
    try:
        # Convert coordinates to NumPy arrays of type float
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)
        
        # Create interpolation functions for x and y
        f_x = interp1d(np.linspace(0, 1, len(x)), x, kind='linear', fill_value="extrapolate")
        f_y = interp1d(np.linspace(0, 1, len(y)), y, kind='linear', fill_value="extrapolate")
        
        # Generate new evenly spaced points for interpolation
        x_interp = f_x(np.linspace(0, 1, num_points))
        y_interp = f_y(np.linspace(0, 1, num_points))
        
        logger.info("Interpolation successful.")
        
        # Concatenate interpolated x and y coordinates
        return np.concatenate([x_interp, y_interp])
    
    except Exception as e:
        # Log any errors that occur during interpolation
        logger.error(f"Error during interpolation: {e}")
        return np.zeros(num_points * 2)  # Return an array of zeros in case of error

def prepare_data(csv_output):
    """
    Prepares data for model training by loading, processing, and normalizing the dataset.

    Parameters:
    - csv_output (str): Path to the CSV file containing the data.

    Returns:
    - tuple: A tuple containing the feature matrix X and labels vector y.
             Returns (None, None) if any step fails.
    """
    logger.info(f"Starting data preparation for CSV file: {csv_output}")
    
    # Check if the CSV file exists
    if not os.path.exists(csv_output):
        logger.error(f"CSV file {csv_output} does not exist.")
        return None, None
    
    # Check if the CSV file contains data by verifying its size
    try:
        file_size = os.path.getsize(csv_output)
        logger.debug(f"CSV file size: {file_size} bytes.")
    except Exception as e:
        logger.error(f"Error accessing CSV file {csv_output}: {e}")
        return None, None
    
    # Proceed only if the file is not empty
    if file_size > 0:
        try:
            # Load the CSV file into a pandas DataFrame
            df = pd.read_csv(csv_output)
            logger.info(f"CSV file loaded successfully with {len(df)} records.")
        except Exception as e:
            logger.error(f"Failed to read CSV file {csv_output}: {e}")
            return None, None

        try:
            # Start interpolating coordinates with a progress bar
            logger.info("Starting coordinate interpolation.")
            coordinates = []
            for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Interpolating coordinates"):
                try:
                    # Safely evaluate the string representation of lists
                    x_coords = ast.literal_eval(row['x_coords'])
                    y_coords = ast.literal_eval(row['y_coords'])
                    
                    # Interpolate the coordinates to fixed length
                    coord = interpolate_coordinates_fixed(x_coords, y_coords)
                    coordinates.append(coord)
                except Exception as row_e:
                    # Log errors for individual rows and append a zero array
                    logger.error(f"Error processing row {index}: {row_e}")
                    coordinates.append(np.zeros(100 * 2))  # Append zero array for failed rows
            # Add the interpolated coordinates as a new column in the DataFrame
            df['coordinates'] = coordinates
            logger.info("Coordinate interpolation completed.")
        except Exception as e:
            logger.error(f"Failed during coordinate interpolation: {e}")
            return None, None

        try:
            # Remove rows that have empty or invalid coordinates
            initial_count = len(df)
            df = df[df['coordinates'].map(len) > 0]
            removed_count = initial_count - len(df)
            if removed_count > 0:
                logger.warning(f"Removed {removed_count} rows due to empty coordinates.")
            else:
                logger.info("No rows removed; all have valid coordinates.")
        except Exception as e:
            logger.error(f"Error removing invalid rows: {e}")
            return None, None

        try:
            # Create the feature matrix X and labels vector y
            X = np.stack(df['coordinates'].values)  # Stack all coordinate arrays
            # Reshape X from (num_samples, 200) to (num_samples, 100, 2) for time steps and features
            X = X.reshape((X.shape[0], 100, 2))
            # Map string labels to numerical values
            y = df['label'].map({'normal': 0, 'abnormal': 1}).values
            logger.info(f"Feature matrix X shape: {X.shape}")
            logger.info(f"Labels vector y shape: {y.shape}")
        except Exception as e:
            logger.error(f"Error creating features and labels: {e}")
            return None, None

        try:
            # Start data normalization using StandardScaler
            logger.info("Starting data normalization.")
            scaler = StandardScaler()
            # Reshape X to (samples, features) for scaling
            X_reshaped = X.reshape((X.shape[0], -1))
            # Fit the scaler and transform the data
            X_scaled = scaler.fit_transform(X_reshaped)
            # Reshape back to (samples, time_steps, features)
            X = X_scaled.reshape((X.shape[0], 100, 2))
            logger.info("Data normalization completed.")
        except Exception as e:
            logger.error(f"Error during data normalization: {e}")
            return None, None

        # Log successful completion of data preparation
        logger.info("Data preparation completed successfully.")
        return X, y
    else:
        # Log error if the CSV file is empty
        logger.error(f"CSV file {csv_output} is empty. No data to process.")
        return None, None
