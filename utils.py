# utils.py

import os
import pandas as pd
from PIL import Image, ImageDraw
import numpy as np
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

def generate_png_images(csv_path, output_dir, image_size=(500, 500)):
    """Generates PNG images with black dots on white background based on CSV coordinates."""
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Generating PNG images"):
        filename = row['filename']
        label = row['label']
        x_coords = eval(row['x_coords'])  # String to list
        y_coords = eval(row['y_coords'])  # String to list

        # Create white image
        img = Image.new('RGB', image_size, 'white')
        draw = ImageDraw.Draw(img)

        # Draw black dots
        for x, y in zip(x_coords, y_coords):
            x_pixel = int(float(x) * (image_size[0] - 1))
            y_pixel = int(float(y) * (image_size[1] - 1))  # Invert Y-axis
            radius = 2  # Dot radius
            draw.ellipse((x_pixel - radius, y_pixel - radius, x_pixel + radius, y_pixel + radius), fill='black')

        # Save image
        image_filename = filename
        save_path = os.path.join(output_dir, image_filename)
        img.save(save_path, 'PNG')
        logger.info(f"PNG image saved as {save_path}.")
