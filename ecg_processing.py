# ecg_processing.py

import fitz  # PyMuPDF for working with PDFs
from PIL import Image
import numpy as np
import cv2  # OpenCV for image processing
import os
import pandas as pd
from tqdm import tqdm
import logging

# Initialize the logger for this module
logger = logging.getLogger(__name__)


def extract_ecg_image_from_pdf(pdf_path, output_path, clip_rect, zoom=2.0):
    """
    Extracts the ECG image from a PDF and saves it as a PNG file.

    Parameters:
    - pdf_path (str): Path to the input PDF file containing the ECG.
    - output_path (str): Path where the extracted ECG image will be saved.
    - clip_rect (fitz.Rect): Rectangle area to clip from the PDF page where the ECG is located.
    - zoom (float): Zoom factor for rendering the PDF page. Defaults to 2.0 for better resolution.

    Returns:
    - PIL.Image.Image: The extracted ECG image as a PIL Image object.
                        Returns None if extraction fails.
    """
    try:
        # Open the PDF document
        pdf_document = fitz.open(pdf_path)
        
        # Load the first page of the PDF
        page = pdf_document.load_page(0)  # Pages are zero-indexed
        
        # Define the transformation matrix for zooming
        matrix = fitz.Matrix(zoom, zoom)
        
        # Render the specified rectangle area of the page to a pixmap (image)
        pix = page.get_pixmap(matrix=matrix, clip=clip_rect)
        
        # Convert the pixmap to a PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Save the extracted image with specified DPI
        img.save(output_path, dpi=(300, 300))
        logger.info(f"ECG image saved as {output_path}.")
        
        return img
    except Exception as e:
        # Log any errors encountered during extraction
        logger.error(f"Error extracting ECG image from {pdf_path}: {e}")
        return None


def preprocess_image(img):
    """
    Converts the extracted ECG image to grayscale and binarizes it.

    Parameters:
    - img (PIL.Image.Image): The ECG image to preprocess.

    Returns:
    - PIL.Image.Image: The preprocessed (grayscale and binarized) image.
                        Returns None if preprocessing fails.
    """
    try:
        # Convert the image to grayscale
        gray = img.convert("L")  
        gray_np = np.array(gray)
        logger.info("Image converted to grayscale.")
        
        # Apply Otsu's thresholding to binarize the image
        _, binary = cv2.threshold(gray_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Convert the NumPy array back to a PIL Image
        binary_img = Image.fromarray(binary)
        logger.info("Image binarized.")
        
        return binary_img
    except Exception as e:
        # Log any errors encountered during preprocessing
        logger.error(f"Error preprocessing image: {e}")
        return None


def extract_ecg_signal(img):
    """
    Extracts the ECG signal from a binarized image by averaging pixel intensities per column.

    Parameters:
    - img (PIL.Image.Image): The binarized ECG image.

    Returns:
    - np.ndarray: 1D array representing the averaged pixel intensities across each column,
                  which approximates the ECG signal.
    """
    # Convert the image to a NumPy array for numerical operations
    img_array = np.array(img)
    
    # Compute the mean pixel intensity for each column (axis=0)
    signal = img_array.mean(axis=0)
    
    return signal


def detect_r_peaks(signal, height=None, distance=None):
    """
    Detects R-peaks in the ECG signal using the SciPy find_peaks function.

    Parameters:
    - signal (np.ndarray): The ECG signal from which to detect R-peaks.
    - height (float or None): Required height of peaks. If None, no height filtering is applied.
    - distance (int or None): Required minimum horizontal distance (in samples) between neighboring peaks.
                              If None, no distance filtering is applied.

    Returns:
    - tuple: A tuple containing:
             - peaks (np.ndarray): Indices of the detected peaks in the signal.
             - properties (dict): Properties of the detected peaks.
    """
    from scipy.signal import find_peaks
    
    # Use find_peaks to detect peaks based on specified height and distance
    peaks, properties = find_peaks(signal, height=height, distance=distance)
    
    return peaks, properties


def split_image_by_peaks(img, peaks, window=50):
    """
    Splits the ECG image into segments (heartbeats) around detected R-peaks.

    Parameters:
    - img (PIL.Image.Image): The binarized ECG image.
    - peaks (np.ndarray): Indices of detected R-peaks in the ECG signal.
    - window (int): Number of pixels to include on either side of each peak for splitting.

    Returns:
    - list of PIL.Image.Image: List of image segments corresponding to individual heartbeats.
    """
    width, height = img.size
    sections = []
    
    for peak in peaks:
        # Define the left and right boundaries for splitting, ensuring they are within image bounds
        left = max(peak - window, 0)
        right = min(peak + window, width)
        
        # Crop the image to the defined window around the peak
        section = img.crop((left, 0, right, height))
        sections.append(section)
    
    return sections


def extract_coordinates(img):
    """
    Extracts the X and Y coordinates of white pixels in a binarized ECG image.

    Parameters:
    - img (PIL.Image.Image): The binarized ECG image.

    Returns:
    - tuple: A tuple containing two 1D NumPy arrays:
             - x_coords (np.ndarray): X-coordinates of white pixels.
             - y_coords (np.ndarray): Y-coordinates of white pixels.
    """
    # Convert the image to a NumPy array for processing
    img_array = np.array(img)
    
    # Find the indices of pixels with value greater than 0 (white pixels)
    y_coords, x_coords = np.where(img_array > 0)  # Returns (row_indices, column_indices)
    
    return x_coords, y_coords


def determine_label(pdf_text):
    """
    Determines if the ECG is labeled as 'normal' or 'abnormal' based on the PDF text content.

    Parameters:
    - pdf_text (str): The text content extracted from the PDF.

    Returns:
    - str: 'normal' if certain keywords are found, otherwise 'abnormal'.
    """
    # Check for keywords indicating a normal ECG
    if 'Sinus Rhythm' in pdf_text or 'Normal' in pdf_text:
        return 'normal'
    else:
        return 'abnormal'


def process_pdfs(output_csv):
    """
    Processes all PDF files in the designated folder to extract ECG data and coordinates.

    This function performs the following steps for each PDF:
    1. Extracts the ECG image from the PDF.
    2. Preprocesses the image (grayscale conversion and binarization).
    3. Splits the image into multiple layers if necessary.
    4. Detects R-peaks in the ECG signal.
    5. Splits the image into individual heartbeats based on detected peaks.
    6. Extracts and normalizes the coordinates of each heartbeat.
    7. Saves the processed heartbeat images and records their metadata.

    Parameters:
    - output_csv (str): Path to the CSV file where the extracted data will be saved.
    """
    # Define directories for saving output images and locating input PDFs
    output_image_dir = 'ECG/training_images'
    os.makedirs(output_image_dir, exist_ok=True)  # Create directory if it doesn't exist
    pdf_folder = 'ECG/pdfs'  # Directory containing input PDF files
    
    # Define the rectangle area to clip from the PDF page where the ECG image is located
    clip_rect = fitz.Rect(20, 200, 850, 525)  # (x0, y0, x1, y1) in points
    
    data = []  # List to store extracted data for CSV

    logger.info("Starting PDF processing and ECG coordinate extraction.")

    # Retrieve all PDF files in the specified folder
    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]
    
    # Iterate over each PDF file with a progress bar
    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        pdf_path = os.path.join(pdf_folder, pdf_file)
        try:
            # Open the PDF document
            pdf_document = fitz.open(pdf_path)
            
            # Load the first page of the PDF
            page = pdf_document.load_page(0)
            
            # Extract text from the page to determine the label
            page_text = page.get_text()
            label = determine_label(page_text)  # Determine if ECG is 'normal' or 'abnormal'
            logger.info(f"Processing PDF: {pdf_file} with label: {label}")

            # Extract the ECG image from the PDF and save it temporarily
            img = extract_ecg_image_from_pdf(pdf_path, "temp_ecg.png", clip_rect)
            if img is None:
                # Skip processing if image extraction failed
                continue

            # Preprocess the extracted ECG image
            preprocessed_img = preprocess_image(img)
            if preprocessed_img is None:
                # Skip processing if preprocessing failed
                continue

            # Split the image into multiple layers if the ECG has overlapping signals
            num_layers = 3  # Number of layers to split the image into
            width, height = preprocessed_img.size
            layer_height = height // num_layers
            layers = []
            for i in range(num_layers):
                top = i * layer_height
                bottom = (i + 1) * layer_height if i < num_layers - 1 else height  # Ensure last layer includes all remaining pixels
                layer = preprocessed_img.crop((0, top, width, bottom))
                layers.append(layer)
            logger.info(f"Number of layers: {num_layers} for {pdf_file}.")

            # Process each layer individually with a nested progress bar
            for layer_idx, layer in enumerate(tqdm(layers, desc=f"Processing layers for {pdf_file}", leave=False), start=1):
                logger.info(f"Processing Layer {layer_idx} for {pdf_file}.")

                # Extract the ECG signal from the current layer
                signal = extract_ecg_signal(layer)

                # Detect R-peaks in the ECG signal
                # Parameters: height is set to 50% of the max signal value, distance to 150 samples
                peaks, properties = detect_r_peaks(signal, height=np.max(signal)*0.5, distance=150)
                logger.info(f"{len(peaks)} R-peaks found for Layer {layer_idx} of {pdf_file}.")

                # Split the image into individual heartbeats based on detected R-peaks
                heartbeats = split_image_by_peaks(layer, peaks, window=100)
                logger.info(f"{len(heartbeats)} heartbeat segments split for Layer {layer_idx} of {pdf_file}.")

                # Process each heartbeat segment with another nested progress bar
                for hb_idx, heartbeat in enumerate(tqdm(heartbeats, desc=f"Processing heartbeats Layer {layer_idx}", leave=False), start=1):
                    # Extract coordinates of white pixels in the heartbeat image
                    x_coords, y_coords = extract_coordinates(heartbeat)
                    
                    # Skip if no white pixels are found
                    if len(x_coords) == 0 or len(y_coords) == 0:
                        logger.warning(f"Heartbeat {hb_idx} of Layer {layer_idx} of {pdf_file} contains no lines.")
                        continue
                    
                    # Normalize the coordinates to range [0, 1] based on image dimensions
                    x_norm = x_coords / heartbeat.size[0]
                    y_norm = y_coords / heartbeat.size[1]
                    
                    # Add the normalized coordinates and metadata to the data list
                    data.append({
                        'filename': f"{os.path.splitext(pdf_file)[0]}_layer{layer_idx}_heartbeat_{hb_idx}.png",
                        'label': label,
                        'x_coords': list(x_norm),
                        'y_coords': list(y_norm)
                    })
                    logger.info(f"Heartbeat {hb_idx} of Layer {layer_idx} of {pdf_file} added to data.")

                    # Define the filename and path for saving the heartbeat image
                    heartbeat_filename = f"{os.path.splitext(pdf_file)[0]}_layer{layer_idx}_heartbeat_{hb_idx}.png"
                    save_path = os.path.join(output_image_dir, f'layer{layer_idx}', label, heartbeat_filename)
                    
                    # Create the necessary directories if they don't exist
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    
                    # Save the heartbeat image with specified DPI
                    heartbeat.save(save_path, dpi=(300, 300))
                    logger.info(f"Heartbeat image saved as {save_path}.")

            # Remove the temporary ECG image file after processing
            if os.path.exists("temp_ecg.png"):
                os.remove("temp_ecg.png")
                logger.info("Temporary ECG image removed.")

        except Exception as e:
            # Log any errors encountered while processing the current PDF
            logger.error(f"Error processing {pdf_file}: {e}", exc_info=True)

    # After processing all PDFs, save the collected data to a CSV file
    if data:
        # Create a DataFrame from the collected data
        df = pd.DataFrame(data)
        
        # Save the DataFrame to the specified CSV file without the index
        df.to_csv(output_csv, index=False)
        logger.info(f"ECG coordinates and labels saved in {output_csv}.")
    else:
        # If no data was collected, create an empty CSV with the required columns
        df = pd.DataFrame(columns=['filename', 'label', 'x_coords', 'y_coords'])
        df.to_csv(output_csv, index=False)
        logger.warning(f"No data collected. Empty CSV file created: {output_csv}")
