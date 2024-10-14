# utils.py

import os
import pandas as pd
from PIL import Image, ImageDraw
import numpy as np
from tqdm import tqdm
import logging
from logger_setup import setup_logger  # Custom logger setup

# Initialize the logger for this module
logger = logging.getLogger(__name__)


def generate_png_images(csv_path, output_dir, image_size=(500, 500)):
    """
    Generates PNG images with black dots on a white background based on coordinates from a CSV file.

    This function reads a CSV file containing filenames, labels, and lists of normalized x and y coordinates.
    For each row in the CSV, it creates a PNG image of specified size, draws black dots at the given coordinates,
    and saves the image to the designated output directory under subdirectories based on layer and label.

    Parameters:
    - csv_path (str): Path to the input CSV file containing the data.
                      The CSV should have columns: 'filename', 'label', 'x_coords', 'y_coords'.
                      'x_coords' and 'y_coords' should be string representations of lists.
    - output_dir (str): Path to the directory where the generated PNG images will be saved.
                        The function will create necessary subdirectories based on layers and labels.
    - image_size (tuple): Size of the generated images in pixels (width, height).
                          Defaults to (500, 500).

    Returns:
    - None
    """
    # Log the start of the PNG image generation process
    logger.info(f"Starting PNG image generation from CSV: {csv_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Image size: {image_size}")

    # Attempt to create the output directory if it doesn't exist
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Verified/created output directory: {output_dir}")
    except Exception as e:
        logger.error(f"Error creating output directory {output_dir}: {e}")
        return  # Exit the function if the output directory cannot be created

    # Attempt to read the CSV file into a pandas DataFrame
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"CSV file loaded with {len(df)} rows.")
    except Exception as e:
        logger.error(f"Error reading CSV file {csv_path}: {e}")
        return  # Exit the function if the CSV cannot be read

    # Iterate over each row in the DataFrame with a progress bar
    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Generating PNG images"):
        try:
            # Extract necessary information from the current row
            filename = row['filename']  # Name of the output image file
            label = row['label']        # Label indicating 'normal' or 'abnormal'
            x_coords = eval(row['x_coords'])  # Convert string representation to list
            y_coords = eval(row['y_coords'])  # Convert string representation to list

            logger.info(f"Processing file {filename} with label {label}.")

            # Create a new white RGB image with the specified size
            img = Image.new('RGB', image_size, 'white')
            draw = ImageDraw.Draw(img)  # Initialize drawing context

            # Iterate over each pair of x and y coordinates to draw black dots
            for i, (x, y) in enumerate(zip(x_coords, y_coords)):
                try:
                    # Convert normalized coordinates to pixel positions
                    x_pixel = int(float(x) * (image_size[0] - 1))
                    y_pixel = int(float(y) * (image_size[1] - 1))
                    radius = 2  # Radius of the black dot

                    # Draw a black ellipse (circle) at the specified pixel position
                    draw.ellipse(
                        (x_pixel - radius, y_pixel - radius, x_pixel + radius, y_pixel + radius),
                        fill='black'
                    )
                    # Optional: Uncomment the following line to log each dot drawn
                    # logger.info(f"Drew dot {i} at ({x_pixel}, {y_pixel}).")
                except Exception as coord_e:
                    # Log a warning if there's an error processing specific coordinates
                    logger.warning(f"Error processing coordinates ({x}, {y}) for file {filename}: {coord_e}")

            # Define the path where the image will be saved
            image_filename = filename  # Use the provided filename from the CSV
            save_path = os.path.join(output_dir, image_filename)

            # Save the generated image in PNG format
            img.save(save_path, 'PNG')
            logger.info(f"PNG image saved as {save_path}.")
        except Exception as row_e:
            # Log an error if there's an issue processing the current row
            logger.error(f"Error processing row {idx} ({row}): {row_e}")

    # Log the completion of the PNG image generation process
    logger.info("PNG image generation completed.")

