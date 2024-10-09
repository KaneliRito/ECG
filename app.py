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

# Set seed for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

set_seed()

# Load the trained model with caching to optimize load times
@st.cache_resource
def load_trained_model(model_path):
    model = load_model(model_path)
    return model

# Provide the path to the model
model_path = 'ecg\\final_model.keras'

# Load the model
model = load_trained_model(model_path)

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

# Import functions from ecg_processing.py and data_preparation.py
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
    width, height = img.size
    layer_height = height // num_layers
    layers = []
    for i in range(num_layers):
        top = i * layer_height
        bottom = (i + 1) * layer_height if i < num_layers - 1 else height
        layer = img.crop((0, top, width, bottom))
        layers.append(layer)
    logger.info(f"Image successfully split into {num_layers} layers.")
    return layers

# Function to generate PDF with predictions (remains in app.py)
def generate_pdf(layers_info, original_image):
    """Generate a PDF with layers, predictions, and percentages."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Title page
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "ECG Analyse Resultaten", ln=True, align='C')
    pdf.ln(10)
    
    # Add the original ECG image
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, "Originele ECG Afbeelding:", ln=True)
    # Save the original image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
        original_image.save(tmp_img.name, format='PNG')
        pdf.image(tmp_img.name, w=180)  # Adjust width if necessary
    pdf.ln(10)
    os.remove(tmp_img.name)
    
    # Add each layer and its predictions
    for info in layers_info:
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, f"Laag {info['Laag']}", ln=True)
        pdf.ln(5)
        
        # Add the layer image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_layer:
            info['Layer Image'].save(tmp_layer.name, format='PNG')
            pdf.image(tmp_layer.name, w=180)  # Adjust width if necessary
        os.remove(tmp_layer.name)
        
        pdf.ln(5)
        
        # Add the prediction
        pdf.set_font("Arial", '', 12)
        pdf.cell(0, 10, f"Voorspelde Klasse: {info['Predicted Class']}", ln=True)
        pdf.cell(0, 10, f"Voorspelde Kans: {info['Prediction Probability']:.2f}", ln=True)
    
    # Save the PDF to a bytes buffer
    pdf_output = pdf.output(dest='S').encode('latin1')
    pdf_buffer = BytesIO(pdf_output)
    pdf_buffer.seek(0)
    
    logger.info("PDF succesvol gegenereerd met de analyse resultaten.")
    return pdf_buffer

# Function to download the log buffer
def download_log():
    """Create a download link for the log file."""
    st.session_state['log_buffer'].seek(0)
    log_content = st.session_state['log_buffer'].getvalue()
    log_bytes = log_content.encode('utf-8')
    st.download_button(
        label="Download Logbestand",
        data=log_bytes,
        file_name="ECG_Analyse_Log.txt",
        mime="text/plain"
    )

# Streamlit App Layout
st.title("ECG PDF Verwerker en Classificatie")

# Disclaimer and acceptance
st.subheader("Disclaimer")
st.write("""
Deze applicatie verwerkt medische gegevens in de vorm van ECG PDF-bestanden. Door op **"Ik ga akkoord"** te klikken, stem je in met het uploaden van je medische gegevens. Alle gegevens worden na verwerking verwijderd. Er wordt geen aansprakelijkheid geaccepteerd voor eventuele gevolgen van het gebruik van deze applicatie, aangezien deze bedoeld is voor demonstratiedoeleinden en geen vervanging is voor professioneel medisch advies. Houd er rekening mee dat de AI-algoritmen die in deze applicatie worden gebruikt mogelijk bias bevatten, waardoor de resultaten niet volledig nauwkeurig kunnen zijn.""")

accept = st.checkbox("Ik ga akkoord")

if accept:
    st.write("Upload een PDF-bestand met een ECG-afbeelding om te testen of de ECG **Normal** of **Abnormal** is. De ECG-afbeelding wordt gesplitst in drie lagen en elke laag wordt geanalyseerd. Je kunt de resultaten downloaden als een PDF en het logbestand downloaden.")
    
    uploaded_file = st.file_uploader("Kies een PDF-bestand", type=["pdf"])
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_pdf_path = tmp_file.name
        
        # Define the area where the ECG image is located in the PDF
        # Adjust these values based on your PDFs
        clip_rect = fitz.Rect(20, 200, 850, 525)  # [left, top, right, bottom]
    
        # Extract text from the PDF to determine the label
        try:
            with fitz.open(tmp_pdf_path) as pdf_document:
                page = pdf_document.load_page(0)
                page_text = page.get_text()
                label = determine_label(page_text)
                logger.info(f"Voorspeld Label van de ECG: {label}")
        except Exception as e:
            logger.error(f"Fout bij het lezen van de PDF-tekst: {e}")
            label = 'unknown'
            st.error("Fout bij het lezen van de PDF-tekst.")
    
        # Extract ECG image
        extract_ecg_image_from_pdf(tmp_pdf_path, "temp_ecg.png", clip_rect)
        # Load the image from the saved path
        try:
            img = Image.open("temp_ecg.png")
        except Exception as e:
            logger.error(f"Fout bij het laden van de ECG-afbeelding: {e}")
            st.error("Fout bij het laden van de ECG-afbeelding.")
            img = None

        if img is not None:
            st.image(img, caption='Geëxtraheerde ECG Afbeelding', use_column_width=True)
    
            # Preprocess image
            preprocessed_img = preprocess_image(img)
            if preprocessed_img is not None:
                st.image(preprocessed_img, caption='Geprocesste ECG Afbeelding (Grijswaarden & Binarisatie)', use_column_width=True)
    
                # Split the image into layers (fixed at 3 layers)
                num_layers = 3  # Number of layers fixed at 3
                layers = split_into_layers(preprocessed_img, num_layers=num_layers)
                logger.info(f"Aantal lagen: {num_layers}")
                st.write(f"Aantal lagen: {num_layers}")
    
                # Display the layers with predictions
                st.subheader("Lagen van de ECG Afbeelding met Voorspellingen")
                layers_info = []  # List to store information per layer
    
                for layer_idx, layer in enumerate(layers, start=1):
                    st.write(f"### Laag {layer_idx}")
    
                    # Extract ECG signal
                    signal = extract_ecg_signal(layer)
    
                    # Detect R-peaks
                    height = np.max(signal) * 0.5
                    distance = 150  # Adjust based on your ECG data
                    peaks = detect_r_peaks(signal, height=height, distance=distance)
                    st.write(f"Aantal gedetecteerde R-peaks: {len(peaks)}")
                    logger.info(f"Aantal gedetecteerde R-peaks voor Laag {layer_idx}: {len(peaks)}")
    
                    # Split the image into heartbeats
                    heartbeats = split_image_by_peaks(layer, peaks, window=100)
                    st.write(f"Aantal hartslagsegmenten: {len(heartbeats)}")
                    logger.info(f"Aantal hartslagsegmenten voor Laag {layer_idx}: {len(heartbeats)}")
    
                    # Process all heartbeat segments and collect coordinates
                    all_coordinates = []
                    for hb_idx, heartbeat in enumerate(heartbeats, start=1):
                        x_coords, y_coords = extract_coordinates(heartbeat)
                        if len(x_coords) == 0 or len(y_coords) == 0:
                            logger.warning(f"Hartslag {hb_idx} van Laag {layer_idx} bevat geen lijnpixels.")
                            st.warning(f"Hartslag {hb_idx} van Laag {layer_idx} bevat geen lijnpixels.")
                            continue
                        # Normalize coordinates
                        x_norm = x_coords / heartbeat.size[0]
                        y_norm = y_coords / heartbeat.size[1]
                        # Interpolate coordinates
                        coordinates = interpolate_coordinates(x_norm, y_norm)
                        all_coordinates.append(coordinates)
    
                    if not all_coordinates:
                        logger.error(f"Geen geldige hartslagsegmenten gevonden voor Laag {layer_idx}.")
                        st.error(f"Geen geldige hartslagsegmenten gevonden voor Laag {layer_idx}.")
                        continue
    
                    # Average coordinates over all heartbeat segments in this layer
                    average_coordinates = np.mean(all_coordinates, axis=0).reshape(1, -1)
    
                    # Reshape data for LSTM [samples, timesteps, features]
                    X_pred = average_coordinates.reshape((1, 100, 2))
    
                    # Predict with the model
                    prediction_prob = model.predict(X_pred)[0][0]
                    # Determine predicted class
                    predicted_class = 'Abnormal' if prediction_prob > 0.5 else 'Normal'
                    prediction_prob = prediction_prob if prediction_prob > 0.5 else 1 - prediction_prob
    
                    st.write(f"**Voorspelde Klasse voor Laag {layer_idx}:** {predicted_class}")
                    st.write(f"**Voorspelde Kans:** {prediction_prob:.2f}")
                    logger.info(f"Voorspelde Klasse voor Laag {layer_idx}: {predicted_class} met kans {prediction_prob:.2f}")
    
                    # Add information to layers_info for PDF
                    layers_info.append({
                        'Laag': layer_idx,
                        'Predicted Class': predicted_class,
                        'Prediction Probability': prediction_prob,
                        'Layer Image': layer  # Save the PIL Image for PDF
                    })
    
                # Generate PDF if layers are processed
                if layers_info:
                    pdf_buffer = generate_pdf(layers_info, img)
                    st.download_button(
                        label="Download Resultaten als PDF",
                        data=pdf_buffer,
                        file_name="ECG_Analyse_Resultaten.pdf",
                        mime="application/pdf"
                    )
                else:
                    st.write("Geen lagen verwerkt. PDF kan niet worden gegenereerd.")
                
                # Show download button for log file
                st.subheader("Download Logbestand")
                download_log()
            else:
                st.error("Fout bij het verwerken van de ECG-afbeelding.")
    
        # Optionally: Delete the temporary files after processing
        try:
            if uploaded_file is not None and os.path.exists(tmp_pdf_path):
                os.remove(tmp_pdf_path)
                logger.info("Tijdelijke PDF-bestand succesvol verwijderd.")
            if os.path.exists("temp_ecg.png"):
                os.remove("temp_ecg.png")
                logger.info("Temporary ECG image removed.")
        except PermissionError:
            logger.warning("Kan het tijdelijke bestand niet verwijderen. Controleer of het nog in gebruik is.")
            st.warning("Kan het tijdelijke bestand niet verwijderen. Controleer of het nog in gebruik is.")
else:
    st.warning("Je moet akkoord gaan met de disclaimer voordat je een bestand kunt uploaden.")
