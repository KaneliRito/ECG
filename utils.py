# utils.py

import os
import pandas as pd
from PIL import Image, ImageDraw
import numpy as np
from tqdm import tqdm
import logging
from logger_setup import setup_logger


logger = logging.getLogger(__name__)

def generate_png_images(csv_path, output_dir, image_size=(500, 500)):
    """Generates PNG images with black dots on white background based on CSV coordinates."""
    logger.info(f"Start het genereren van PNG afbeeldingen vanuit CSV: {csv_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Image size: {image_size}")
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory gecontroleerd/bgemaakt: {output_dir}")
    except Exception as e:
        logger.error(f"Fout bij het maken van output directory {output_dir}: {e}")
        return

    try:
        df = pd.read_csv(csv_path)
        logger.info(f"CSV bestand ingelezen met {len(df)} rijen.")
    except Exception as e:
        logger.error(f"Fout bij het lezen van CSV bestand {csv_path}: {e}")
        return

    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Generating PNG images"):
        try:
            filename = row['filename']
            label = row['label']
            x_coords = eval(row['x_coords'])  # String naar lijst
            y_coords = eval(row['y_coords'])  # String naar lijst

            logger.info(f"Verwerken van bestand {filename} met label {label}.")

            # Maak een witte afbeelding
            img = Image.new('RGB', image_size, 'white')
            draw = ImageDraw.Draw(img)

            # Teken zwarte stippen
            for i, (x, y) in enumerate(zip(x_coords, y_coords)):
                try:
                    x_pixel = int(float(x) * (image_size[0] - 1))
                    y_pixel = int(float(y) * (image_size[1] - 1))  # Y-as omkeren
                    radius = 2  # Straal van de stip
                    draw.ellipse((x_pixel - radius, y_pixel - radius, x_pixel + radius, y_pixel + radius), fill='black')
                    logger.info(f"Stip {i} getekend op ({x_pixel}, {y_pixel}).")
                except Exception as coord_e:
                    logger.warning(f"Fout bij het verwerken van coördinaten ({x}, {y}) voor bestand {filename}: {coord_e}")

            # Sla de afbeelding op
            image_filename = filename
            save_path = os.path.join(output_dir, image_filename)
            img.save(save_path, 'PNG')
            logger.info(f"PNG afbeelding opgeslagen als {save_path}.")
        except Exception as row_e:
            logger.error(f"Fout bij het verwerken van rij {idx} ({row}): {row_e}")

    logger.info("PNG afbeeldingen generatie voltooid.")
 