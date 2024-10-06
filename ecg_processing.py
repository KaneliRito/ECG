# ecg_processing.py

import fitz  # PyMuPDF for working with PDFs
from PIL import Image
import numpy as np
import cv2  # OpenCV for image processing
import os
import pandas as pd
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

def extract_ecg_image_from_pdf(pdf_path, output_path, clip_rect, zoom=2.0):
    """Extracts the ECG image from a PDF and saves it as PNG."""
    try:
        pdf_document = fitz.open(pdf_path)
        page = pdf_document.load_page(0)  # Load first page
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, clip=clip_rect)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img.save(output_path, dpi=(300, 300))
        logger.info(f"ECG image saved as {output_path}.")
        return img
    except Exception as e:
        logger.error(f"Error extracting ECG image from {pdf_path}: {e}")
        return None

def preprocess_image(img):
    """Converts image to grayscale and binarizes it."""
    try:
        gray = img.convert("L")  # Convert to grayscale
        gray_np = np.array(gray)
        _, binary = cv2.threshold(gray_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary_img = Image.fromarray(binary)
        logger.info("Image converted to grayscale and binarized.")
        return binary_img
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return None

def extract_ecg_signal(img):
    """Extracts ECG signal from binarized image by averaging pixel intensities per column."""
    img_array = np.array(img)
    signal = img_array.mean(axis=0)
    return signal

def detect_r_peaks(signal, height=None, distance=None):
    """Finds R-peaks in ECG signal."""
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(signal, height=height, distance=distance)
    return peaks

def split_image_by_peaks(img, peaks, window=50):
    """Splits image around R-peaks."""
    width, height = img.size
    sections = []
    for peak in peaks:
        left = max(peak - window, 0)
        right = min(peak + window, width)
        section = img.crop((left, 0, right, height))
        sections.append(section)
    return sections

def extract_coordinates(img):
    """Extracts X and Y coordinates of white pixels in binarized image."""
    img_array = np.array(img)
    y_coords, x_coords = np.where(img_array > 0)  # White pixels
    return x_coords, y_coords

def determine_label(pdf_text):
    """Determines if ECG is normal or abnormal based on text."""
    if 'Sinus Rhythm' in pdf_text or 'Normal' in pdf_text:
        return 'normal'
    else:
        return 'abnormal'

def process_pdfs(output_csv):
    """Processes PDFs and extracts ECG data."""
    output_image_dir = 'ECG/training_images'
    os.makedirs(output_image_dir, exist_ok=True)
    pdf_folder = 'ECG/pdfs'  # Folder with PDFs
    clip_rect = fitz.Rect(20, 200, 850, 525)  # Area to clip ECG image
    data = []

    logger.info("Starting PDF processing and ECG coordinate extraction.")

    # Process each PDF in the folder with progress bar
    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]
    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        pdf_path = os.path.join(pdf_folder, pdf_file)
        try:
            pdf_document = fitz.open(pdf_path)
            page = pdf_document.load_page(0)
            page_text = page.get_text()
            label = determine_label(page_text)
            logger.info(f"Processing PDF: {pdf_file} with label: {label}")

            # Extract ECG image from PDF
            img = extract_ecg_image_from_pdf(pdf_path, "temp_ecg.png", clip_rect)
            if img is None:
                continue

            # Preprocess the image
            preprocessed_img = preprocess_image(img)
            if preprocessed_img is None:
                continue

            # Split the image into layers
            num_layers = 3  # Number of layers
            width, height = preprocessed_img.size
            layer_height = height // num_layers
            layers = []
            for i in range(num_layers):
                top = i * layer_height
                bottom = (i + 1) * layer_height if i < num_layers - 1 else height
                layer = preprocessed_img.crop((0, top, width, bottom))
                layers.append(layer)
            logger.info(f"Number of layers: {num_layers} for {pdf_file}.")

            # Process each layer with progress bar
            for layer_idx, layer in enumerate(tqdm(layers, desc=f"Processing layers for {pdf_file}", leave=False), start=1):
                logger.info(f"Processing Layer {layer_idx} for {pdf_file}.")

                # Extract ECG signal from layer
                signal = extract_ecg_signal(layer)

                # Find R-peaks
                peaks = detect_r_peaks(signal, height=np.max(signal)*0.5, distance=150)
                logger.info(f"{len(peaks)} R-peaks found for Layer {layer_idx} of {pdf_file}.")

                # Split image into heartbeats with progress bar
                heartbeats = split_image_by_peaks(layer, peaks, window=100)
                logger.info(f"{len(heartbeats)} heartbeat segments split for Layer {layer_idx} of {pdf_file}.")

                for hb_idx, heartbeat in enumerate(tqdm(heartbeats, desc=f"Processing heartbeats Layer {layer_idx}", leave=False), start=1):
                    x_coords, y_coords = extract_coordinates(heartbeat)
                    if len(x_coords) == 0 or len(y_coords) == 0:
                        logger.warning(f"Heartbeat {hb_idx} of Layer {layer_idx} of {pdf_file} contains no lines.")
                        continue
                    # Normalize coordinates
                    x_norm = x_coords / heartbeat.size[0]
                    y_norm = y_coords / heartbeat.size[1]
                    # Interpolate coordinates (handled later)
                    # Add data
                    data.append({
                        'filename': f"{os.path.splitext(pdf_file)[0]}_layer{layer_idx}_heartbeat_{hb_idx}.png",
                        'label': label,
                        'x_coords': list(x_norm),
                        'y_coords': list(y_norm)
                    })
                    logger.info(f"Heartbeat {hb_idx} of Layer {layer_idx} of {pdf_file} added to data.")

                    # Save the heartbeat image
                    heartbeat_filename = f"{os.path.splitext(pdf_file)[0]}_layer{layer_idx}_heartbeat_{hb_idx}.png"
                    save_path = os.path.join(output_image_dir, f'layer{layer_idx}', label, heartbeat_filename)
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    heartbeat.save(save_path, dpi=(300, 300))
                    logger.info(f"Heartbeat image saved as {save_path}.")

            # Remove temporary ECG image
            if os.path.exists("temp_ecg.png"):
                os.remove("temp_ecg.png")
                logger.info("Temporary ECG image removed.")

        except Exception as e:
            logger.error(f"Error processing {pdf_file}: {e}", exc_info=True)

    # Save collected data as CSV
    if data:
        df = pd.DataFrame(data)
        df.to_csv(output_csv, index=False)
        logger.info(f"ECG coordinates and labels saved in {output_csv}.")
    else:
        # Create empty CSV file
        df = pd.DataFrame(columns=['filename', 'label', 'x_coords', 'y_coords'])
        df.to_csv(output_csv, index=False)
        logger.warning(f"No data collected. Empty CSV file created: {output_csv}")
