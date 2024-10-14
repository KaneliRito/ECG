# app.py (webapp)

import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import os
import tempfile
from fpdf import FPDF  # For PDF generation
from io import BytesIO, StringIO  # For BytesIO and StringIO
import logging
from logging import StreamHandler
from tqdm import tqdm
from ecg_processing import (
    extract_ecg_image_from_pdf,
    preprocess_image,
    extract_ecg_signal,
    detect_r_peaks,
    split_image_by_peaks,
    extract_coordinates,
    determine_label
)

from data_preparation import (
    interpolate_coordinates_fixed as interpolate_coordinates
)

# Initialize per-session logging
if 'log_buffer' not in st.session_state:
    st.session_state['log_buffer'] = StringIO()

# Set up logger
logger = logging.getLogger('streamlit_logger')
logger.setLevel(logging.INFO)
# Remove all existing handlers
for handler in logger.handlers[:]:
    logger.removeHandler(handler)
# Create a StreamHandler to write to the log buffer
stream_handler = StreamHandler(st.session_state['log_buffer'])
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# Add the stream handler to the root logger to capture logs from other modules
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)
root_logger.addHandler(stream_handler)

logger.info("Start of the ECG processing and model training script via Streamlit.")

# Set seed for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    logger.info("Seed set for reproducibility.")

set_seed()

# Load the trained model with caching to optimize load times
@st.cache_resource
def load_trained_model(model_path):
    logger.info(f"Starting to load the model from path: {model_path}")
    model = load_model(model_path)
    logger.info("final_model.keras successfully loaded.")
    return model

# Provide the path to the model
model_path = 'final_model.keras'

# Load the model
model = load_trained_model(model_path)

# Function to split the image into layers (since not available in other modules)
def split_into_layers(img, num_layers=3):
    """
    Splits the image horizontally into a given number of layers.

    Parameters:
    - img: PIL Image
    - num_layers: Number of layers to split into

    Returns:
    - List of split layers as PIL Images
    """
    logger.info(f"Starting to split the image into {num_layers} layers.")
    width, height = img.size
    layer_height = height // num_layers
    layers = []
    for i in range(num_layers):
        top = i * layer_height
        bottom = (i + 1) * layer_height if i < num_layers - 1 else height
        layer = img.crop((0, top, width, bottom))
        layers.append(layer)
        logger.debug(f"Layer {i+1} cropped from pixels {top} to {bottom}.")
    logger.info(f"Image successfully split into {num_layers} layers.")
    return layers

# Function to generate PDF with predictions (remains in app.py)
def generate_pdf(layers_info, original_image):
    """Generate a PDF with layers, predictions, and percentages."""
    logger.info("Starting to generate the PDF with analysis results.")
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Title page
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "ECG Analysis Results", ln=True, align='C')
    pdf.ln(10)
    
    
    
    # Add ROC Curve Image with Explanation
    roc_image_path = '\\roc_curves\\roc_foldfinal.png'
    if os.path.exists(roc_image_path):
        pdf.set_font("Arial", '', 12)
        pdf.cell(0, 10, "Receiver Operating Characteristic (ROC) Curve:", ln=True)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_roc:
            Image.open(roc_image_path).save(tmp_roc.name, format='PNG')
            pdf.image(tmp_roc.name, w=180)  # Adjust width if necessary
            logger.debug("ROC curve image added to the PDF.")
        pdf.ln(5)
        explanation = (
            "The ROC curve illustrates the diagnostic ability of the AI model by plotting the True Positive Rate against the False Positive Rate at various threshold settings. "
            "A model with a higher area under the curve (AUC) demonstrates better performance in distinguishing between normal and abnormal ECG readings."
        )
        pdf.multi_cell(0, 10, explanation)
        logger.debug("Explanation for ROC curve added to the PDF.")
        os.remove(tmp_roc.name)
    else:
        logger.warning(f"ROC curve image not found at path: {roc_image_path}")
        pdf.set_font("Arial", 'I', 12)
        pdf.cell(0, 10, "ROC curve image not available.", ln=True)
    # Add the original ECG image
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 14, "Original ECG Image:", ln=True)
    # Save the original image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
        original_image.save(tmp_img.name, format='PNG')
        pdf.image(tmp_img.name, w=180)  # Adjust width if necessary
        logger.debug("Original ECG image added to the PDF.")
    pdf.ln(10)
    os.remove(tmp_img.name)
    pdf.ln(10)
    
    # Add each layer and its predictions
    for info in layers_info:
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, f"Layer {info['Laag']}", ln=True)
        pdf.ln(5)
        
        # Add the layer image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_layer:
            info['Layer Image'].save(tmp_layer.name, format='PNG')
            pdf.image(tmp_layer.name, w=180)  # Adjust width if necessary
            logger.debug(f"Layer {info['Laag']} image added to the PDF.")
        os.remove(tmp_layer.name)
        
        pdf.ln(5)
        
        # Add the prediction
        pdf.set_font("Arial", '', 12)
        pdf.cell(0, 10, f"Predicted Class: {info['Predicted Class']}", ln=True)
        pdf.cell(0, 10, f"Prediction Probability: {info['Prediction Probability']:.2f}", ln=True)
    
    # Save the PDF to a bytes buffer
    pdf_output = pdf.output(dest='S').encode('latin1')
    pdf_buffer = BytesIO(pdf_output)
    pdf_buffer.seek(0)
    
    logger.info("PDF successfully generated with the analysis results.")
    return pdf_buffer

# Function to download the log buffer
def download_log():
    """Create a download link for the log file."""
    logger.info("User activated the download button for the log file.")
    st.session_state['log_buffer'].seek(0)
    log_content = st.session_state['log_buffer'].getvalue()
    log_bytes = log_content.encode('utf-8')
    st.download_button(
        label="Download Log File",
        data=log_bytes,
        file_name="ECG_Analysis_Log.txt",
        mime="text/plain"
    )
    logger.info("Log file made available for download.")

# Streamlit App Layout
st.title("ECG PDF Processor and Classification")

# Disclaimer and acceptance
st.subheader("Disclaimer")
st.write("""
This application processes medical data in the form of ECG PDF files. By clicking **"I Agree"**, you consent to uploading your medical data. All data will be deleted after processing. No liability is accepted for any consequences of using this application, as it is intended for demonstration purposes and is not a substitute for professional medical advice. Please note that the AI algorithms used in this application may contain biases, making the results not entirely accurate.""")
    
accept = st.checkbox("I Agree")

if accept:
    logger.info("User has agreed to the disclaimer.")
    st.write("Upload a PDF file with an ECG image to test whether the ECG is **Normal** or **Abnormal**. The ECG image will be split into three layers and each layer will be analyzed. You can download the results as a PDF and download the log file.")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    
    if uploaded_file is not None:
        logger.info(f"User uploaded a file: {uploaded_file.name}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_pdf_path = tmp_file.name
            logger.debug(f"Uploaded file saved as temporary file: {tmp_pdf_path}")
        
        # Define the area where the ECG image is located in the PDF
        # Adjust these values based on your PDFs
        clip_rect = fitz.Rect(20, 200, 850, 525)  # [left, top, right, bottom]
        logger.debug(f"Using clip_rect for ECG extraction: {clip_rect}")
    
        # Extract text from the PDF to determine the label
        try:
            with fitz.open(tmp_pdf_path) as pdf_document:
                page = pdf_document.load_page(0)
                page_text = page.get_text()
                label = determine_label(page_text)
                logger.info(f"Predicted Label of the ECG: {label}")
        except Exception as e:
            logger.error(f"Error reading PDF text: {e}")
            label = 'unknown'
            st.error("Error reading PDF text.")
    
        # Extract ECG image
        try:
            logger.info("Starting extraction of the ECG image from the PDF.")
            extract_ecg_image_from_pdf(tmp_pdf_path, "ECG\\temp\\temp_ecg.png", clip_rect)
            logger.info("ECG image successfully extracted from the PDF.")
        except Exception as e:
            logger.error(f"Error extracting ECG image: {e}")
            st.error("Error extracting ECG image.")
        
        # Load the image from the saved path
        try:
            img = Image.open("ECG\\temp\\temp_ecg.png")
            logger.info("ECG image successfully loaded.")
        except Exception as e:
            logger.error(f"Error loading ECG image: {e}")
            st.error("Error loading ECG image.")
            img = None

        if img is not None:
            st.image(img, caption='Extracted ECG Image', use_column_width=True)
            logger.debug("Extracted ECG image displayed to the user.")
    
            # Preprocess image
            try:
                logger.info("Starting preprocessing of the ECG image.")
                preprocessed_img = preprocess_image(img)
                logger.info("Preprocessing of the ECG image completed.")
            except Exception as e:
                logger.error(f"Error preprocessing ECG image: {e}")
                st.error("Error preprocessing ECG image.")
                preprocessed_img = None

            if preprocessed_img is not None:
                st.image(preprocessed_img, caption='Preprocessed ECG Image (Grayscale & Binarization)', use_column_width=True)
                logger.debug("Preprocessed ECG image displayed to the user.")
    
                # Split the image into layers (fixed at 3 layers)
                num_layers = 3  # Number of layers fixed at 3
                layers = split_into_layers(preprocessed_img, num_layers=num_layers)
                logger.info(f"Number of layers: {num_layers}")
                st.write(f"Number of layers: {num_layers}")
    
                # Display the layers with predictions
                st.subheader("Layers of the ECG Image with Predictions")
                layers_info = []  # List to store information per layer
    
                for layer_idx, layer in enumerate(layers, start=1):
                    st.write(f"### Layer {layer_idx}")
                    logger.info(f"Starting processing of Layer {layer_idx}.")
    
                    # Extract ECG signal
                    try:
                        logger.debug(f"Extracting ECG signal for Layer {layer_idx}.")
                        signal = extract_ecg_signal(layer)
                        logger.debug(f"ECG signal for Layer {layer_idx} successfully extracted.")
                    except Exception as e:
                        logger.error(f"Error extracting ECG signal for Layer {layer_idx}: {e}")
                        st.error(f"Error extracting ECG signal for Layer {layer_idx}.")
                        continue
    
                    # Detect R-peaks
                    height = np.max(signal) * 0.5
                    distance = 150  # Adjust based on your ECG data
                    try:
                        peaks = detect_r_peaks(signal, height=height, distance=distance)
                        logger.info(f"Number of detected R-peaks for Layer {layer_idx}: {len(peaks)}")
                    except Exception as e:
                        logger.error(f"Error detecting R-peaks for Layer {layer_idx}: {e}")
                        st.error(f"Error detecting R-peaks for Layer {layer_idx}.")
                        peaks = []
    
                    st.write(f"Number of detected R-peaks: {len(peaks)}")
    
                    # Split the image into heartbeats
                    try:
                        heartbeats = split_image_by_peaks(layer, peaks, window=100)
                        logger.info(f"Number of heartbeat segments for Layer {layer_idx}: {len(heartbeats)}")
                    except Exception as e:
                        logger.error(f"Error splitting heartbeat segments for Layer {layer_idx}: {e}")
                        st.error(f"Error splitting heartbeat segments for Layer {layer_idx}.")
                        heartbeats = []
    
                    st.write(f"Number of heartbeat segments: {len(heartbeats)}")
    
                    # Process all heartbeat segments and collect coordinates
                    all_coordinates = []
                    for hb_idx, heartbeat in enumerate(heartbeats, start=1):
                        try:
                            x_coords, y_coords = extract_coordinates(heartbeat)
                            logger.debug(f"Coordinates extracted for heartbeat {hb_idx} of Layer {layer_idx}.")
                        except Exception as e:
                            logger.error(f"Error extracting coordinates for heartbeat {hb_idx} of Layer {layer_idx}: {e}")
                            st.error(f"Error extracting coordinates for heartbeat {hb_idx} of Layer {layer_idx}.")
                            continue
                        
                        if len(x_coords) == 0 or len(y_coords) == 0:
                            logger.warning(f"Heartbeat {hb_idx} of Layer {layer_idx} contains no line pixels.")
                            st.warning(f"Heartbeat {hb_idx} of Layer {layer_idx} contains no line pixels.")
                            continue
                        # Normalize coordinates
                        x_norm = x_coords / heartbeat.size[0]
                        y_norm = y_coords / heartbeat.size[1]
                        # Interpolate coordinates
                        try:
                            coordinates = interpolate_coordinates(x_norm, y_norm)
                            logger.debug(f"Coordinates interpolated for heartbeat {hb_idx} of Layer {layer_idx}.")
                            all_coordinates.append(coordinates)
                        except Exception as e:
                            logger.error(f"Error interpolating coordinates for heartbeat {hb_idx} of Layer {layer_idx}: {e}")
                            st.error(f"Error interpolating coordinates for heartbeat {hb_idx} of Layer {layer_idx}.")
                            continue
    
                    if not all_coordinates:
                        logger.error(f"No valid heartbeat segments found for Layer {layer_idx}.")
                        st.error(f"No valid heartbeat segments found for Layer {layer_idx}.")
                        continue
    
                    # Average coordinates over all heartbeat segments in this layer
                    try:
                        average_coordinates = np.mean(all_coordinates, axis=0).reshape(1, -1)
                        logger.debug(f"Average coordinates calculated for Layer {layer_idx}.")
                    except Exception as e:
                        logger.error(f"Error calculating average coordinates for Layer {layer_idx}: {e}")
                        st.error(f"Error calculating average coordinates for Layer {layer_idx}.")
                        continue
    
                    # Reshape data for LSTM [samples, timesteps, features]
                    try:
                        X_pred = average_coordinates.reshape((1, 100, 2))
                        logger.debug(f"Data reshaped for LSTM prediction for Layer {layer_idx}.")
                    except Exception as e:
                        logger.error(f"Error reshaping data for Layer {layer_idx}: {e}")
                        st.error(f"Error reshaping data for Layer {layer_idx}.")
                        continue
    
                    # Predict with the model
                    try:
                        prediction_prob = model.predict(X_pred)[0][0]
                        predicted_class = 'Abnormal' if prediction_prob > 0.5 else 'Normal'
                        prediction_prob = prediction_prob if prediction_prob > 0.5 else 1 - prediction_prob
                        logger.info(f"Predicted Class for Layer {layer_idx}: {predicted_class} with probability {prediction_prob:.2f}")
                    except Exception as e:
                        logger.error(f"Error predicting class for Layer {layer_idx}: {e}")
                        st.error(f"Error predicting class for Layer {layer_idx}.")
                        predicted_class = 'Unknown'
                        prediction_prob = 0.0
    
                    st.write(f"**Predicted Class for Layer {layer_idx}:** {predicted_class}")
                    st.write(f"**Prediction Probability:** {prediction_prob:.2f}")
    
                    # Add information to layers_info for PDF
                    layers_info.append({
                        'Laag': layer_idx,
                        'Predicted Class': predicted_class,
                        'Prediction Probability': prediction_prob,
                        'Layer Image': layer  # Save the PIL Image for PDF
                    })
                    logger.debug(f"Information added to layers_info for Layer {layer_idx}.")
    
                # Generate PDF if layers are processed
                if layers_info:
                    try:
                        logger.info("Starting to generate the results PDF.")
                        pdf_buffer = generate_pdf(layers_info, img)
                        logger.info("Results PDF successfully generated.")
                        st.download_button(
                            label="Download Results as PDF",
                            data=pdf_buffer,
                            file_name="ECG_Analysis_Results.pdf",
                            mime="application/pdf"
                        )
                        logger.info("Download button for PDF made available to the user.")
                    except Exception as e:
                        logger.error(f"Error generating PDF: {e}")
                        st.error("Error generating PDF.")
                else:
                    st.write("No layers processed. PDF cannot be generated.")
                    logger.warning("No layers processed. PDF will not be generated.")
                
                # Show download button for log file
                st.subheader("Download Log File")
                download_log()
            else:
                st.error("Error processing the ECG image.")
                logger.error("Error processing the ECG image.")
    
        # Optionally: Delete the temporary files after processing
        try:
            if uploaded_file is not None and os.path.exists(tmp_pdf_path):
                os.remove(tmp_pdf_path)
                logger.info("Temporary PDF file successfully removed.")
            if os.path.exists("ECG\\temp\\temp_ecg.png"):
                os.remove("ECG\\temp\\temp_ecg.png")
                logger.info("Temporary ECG image successfully removed.")
        except PermissionError:
            logger.warning("Cannot remove the temporary file. Check if it's still in use.")
            st.warning("Cannot remove the temporary file. Check if it's still in use. If not, contact the administrator or email to: s1141810@student.hsleiden.nl")
    else:
        st.warning("You must agree to the disclaimer before uploading a file.")
        logger.info("User did not agree to the disclaimer.")