import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
from scipy.signal import find_peaks
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy.interpolate import interp1d
import cv2
import os
import tempfile
from fpdf import FPDF  # Voor PDF generatie
from io import BytesIO, StringIO  # Voor BytesIO en StringIO
import logging
from logging import StreamHandler
from tqdm import tqdm, trange  # Voor progress bars

# Zorg voor reproduceerbaarheid
def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

set_seed()

# Laad het getrainde model met caching om laadtijden te optimaliseren
@st.cache_resource
def load_trained_model(model_path):
    model = load_model(model_path)
    return model

# Geef het pad van het model
model_path = 'final_model.keras'

# Laad het model
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

logger.info("Start van het ECG-verwerkings- en modeltrainingsscript via Streamlit.")

# Functie om een ECG-afbeelding uit een PDF te extraheren
def extract_ecg_image_from_pdf(pdf_path, clip_rect, zoom=2.0):
    """Extraheer de ECG-afbeelding uit een PDF en converteer deze naar een PIL Image."""
    try:
        with fitz.open(pdf_path) as pdf_document:
            page = pdf_document.load_page(0)
            matrix = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=matrix, clip=clip_rect)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            logger.info(f"ECG-afbeelding succesvol geëxtraheerd uit {pdf_path}.")
            return img
    except Exception as e:
        logger.error(f"Fout bij het extraheren van ECG-afbeelding uit PDF: {e}")
        return None

# Functie om afbeelding te preprocessen: grayscaling en binarisatie
def preprocess_image(img):
    """Converteer afbeelding naar grijswaarden en maak deze binair."""
    try:
        # Converteer naar grijswaarden
        gray = img.convert("L")
        
        # Binariseer de afbeelding met Otsu's thresholding
        gray_np = np.array(gray)
        _, binary = cv2.threshold(gray_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Converteer terug naar PIL Image
        binary_img = Image.fromarray(binary)
        logger.info("Afbeelding succesvol omgezet naar grijswaarden en binair gemaakt.")
        return binary_img
    except Exception as e:
        logger.error(f"Fout bij preprocessing van afbeelding: {e}")
        return None

# Functie om ECG-signaal te extraheren uit een afbeelding
def extract_ecg_signal(img):
    """Extraheer een ECG-signaal uit een binariseerde afbeelding door de gemiddelde intensiteit per kolom te nemen."""
    img_array = np.array(img)
    signal = img_array.mean(axis=0)
    logger.info("ECG-signaal succesvol geëxtraheerd uit afbeelding.")
    return signal

# Functie om R-peaks te detecteren in een ECG-signaal
def detect_r_peaks(signal, height=None, distance=None):
    """Detecteer R-peaks in het ECG-signaal."""
    peaks, properties = find_peaks(signal, height=height, distance=distance)
    logger.info(f"{len(peaks)} R-peaks succesvol gedetecteerd in het ECG-signaal.")
    return peaks

# Functie om de afbeelding te splitsen rondom R-peaks
def split_image_by_peaks(img, peaks, window=50):
    """
    Splits de afbeelding rondom de R-peaks.

    Parameters:
    - img: PIL Image
    - peaks: Lijst van indices waar R-peaks zijn gedetecteerd
    - window: Aantal pixels voor en na de R-peak om de sectie te definiëren

    Returns:
    - Lijst van gesplitste afbeeldingen
    """
    width, height = img.size
    sections = []
    for peak in peaks:
        left = max(peak - window, 0)
        right = min(peak + window, width)
        section = img.crop((left, 0, right, height))
        sections.append(section)
    logger.info(f"{len(sections)} hartslagsegmenten succesvol gesplitst rond R-peaks.")
    return sections

# Functie om X en Y coördinaten van lijnpixels te extraheren
def extract_coordinates(img):
    """Haal X en Y coördinaten op van de lijnpixels in de binariseerde afbeelding."""
    img_array = np.array(img)
    y_coords, x_coords = np.where(img_array > 0)  # Veronderstel dat lijnpixels wit zijn
    logger.info(f"{len(x_coords)} lijnpixels succesvol geëxtraheerd uit de afbeelding.")
    return x_coords, y_coords

# Functie om label te bepalen op basis van tekst in PDF
def determine_label(pdf_text):
    """Bepaal of de ECG normaal of abnormaal is op basis van de tekst in de PDF."""
    if 'Sinus Rhythm' in pdf_text or 'Normal' in pdf_text:
        logger.info("Label bepaald als 'normal' op basis van PDF-tekst.")
        return 'normal'
    else:
        logger.info("Label bepaald als 'abnormal' op basis van PDF-tekst.")
        return 'abnormal'

# Functie om coördinaten te interpoleren naar vaste lengte
def interpolate_coordinates(x, y, num_points=100):
    """Interpoleer de coördinaten naar een vaste lengte."""
    if len(x) < 2 or len(y) < 2:
        logger.warning("Niet genoeg punten om te interpoleren.")
        return np.zeros(num_points*2)
    try:
        f_x = interp1d(np.linspace(0, 1, len(x)), x, kind='linear', fill_value="extrapolate")
        f_y = interp1d(np.linspace(0, 1, len(y)), y, kind='linear', fill_value="extrapolate")
        x_interp = f_x(np.linspace(0, 1, num_points))
        y_interp = f_y(np.linspace(0, 1, num_points))
        interpolated = np.concatenate([x_interp, y_interp])
        logger.info("Coördinaten succesvol geïnterpoleerd naar vaste lengte.")
        return interpolated
    except Exception as e:
        logger.error(f"Fout bij interpolatie: {e}")
        return np.zeros(num_points*2)

# Functie om de afbeelding te splitsen in lagen
def split_into_layers(img, num_layers=3):
    """
    Splits de afbeelding horizontaal in een opgegeven aantal lagen.

    Parameters:
    - img: PIL Image
    - num_layers: Aantal lagen om in te splitsen

    Returns:
    - Lijst van gesplitste lagen als PIL Images
    """
    width, height = img.size
    layer_height = height // num_layers
    layers = []
    for i in range(num_layers):
        top = i * layer_height
        bottom = (i + 1) * layer_height if i < num_layers - 1 else height
        layer = img.crop((0, top, width, bottom))
        layers.append(layer)
    logger.info(f"Afbeelding succesvol gesplitst in {num_layers} lagen.")
    return layers

# Functie om PDF te genereren met de voorspellingen
def generate_pdf(layers_info, original_image):
    """Genereer een PDF met de lagen, voorspellingen en percentages."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Titelpagina
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "ECG Analyse Resultaten", ln=True, align='C')
    pdf.ln(10)
    
    # Voeg de originele ECG-afbeelding toe
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, "Originele ECG Afbeelding:", ln=True)
    # Sla de originele afbeelding tijdelijk op
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
        original_image.save(tmp_img.name, format='PNG')
        pdf.image(tmp_img.name, w=180)  # Pas de breedte aan indien nodig
    pdf.ln(10)
    os.remove(tmp_img.name)
    
    # Voeg elke laag en de bijbehorende voorspellingen toe
    for info in layers_info:
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, f"Laag {info['Laag']}", ln=True)
        pdf.ln(5)
        
        # Voeg de laagafbeelding toe
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_layer:
            info['Layer Image'].save(tmp_layer.name, format='PNG')
            pdf.image(tmp_layer.name, w=180)  # Pas de breedte aan indien nodig
        os.remove(tmp_layer.name)
        
        pdf.ln(5)
        
        # Voeg de voorspelling toe
        pdf.set_font("Arial", '', 12)
        pdf.cell(0, 10, f"Voorspelde Klasse: {info['Predicted Class']}", ln=True)
        pdf.cell(0, 10, f"Voorspelde Kans: {info['Prediction Probability']:.2f}", ln=True)
    
    # Sla de PDF op in een bytes buffer
    pdf_output = pdf.output(dest='S').encode('latin1')  # Genereer PDF als string en codeer naar bytes
    pdf_buffer = BytesIO(pdf_output)
    pdf_buffer.seek(0)
    
    logger.info("PDF succesvol gegenereerd met de analyse resultaten.")
    return pdf_buffer

# Functie om de log buffer te downloaden
def download_log():
    """Maak een download link voor het logbestand."""
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

# Disclaimer en acceptatie
st.subheader("Disclaimer")
st.write("""
Deze applicatie verwerkt medische gegevens in de vorm van ECG PDF-bestanden. Door op **"Ik ga akkoord"** te klikken, stem je in met het uploaden van je medische gegevens. Alle gegevens worden na verwerking verwijderd. Er wordt geen aansprakelijkheid geaccepteerd voor eventuele gevolgen van het gebruik van deze applicatie, aangezien deze bedoeld is voor demonstratiedoeleinden en geen vervanging is voor professioneel medisch advies.
""")

accept = st.checkbox("Ik ga akkoord")

if accept:
    st.write("Upload een PDF-bestand met een ECG-afbeelding om te testen of de ECG **Normal** of **Abnormal** is. De ECG-afbeelding wordt gesplitst in drie lagen en elke laag wordt geanalyseerd. Je kunt de resultaten downloaden als een PDF en het logbestand downloaden.")
    
    uploaded_file = st.file_uploader("Kies een PDF-bestand", type=["pdf"])
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_pdf_path = tmp_file.name
        
        # Definieer het gebied waar de ECG-afbeelding zich bevindt in de PDF
        # Pas deze waarden aan op basis van jouw PDF's
        clip_rect = fitz.Rect(20, 200, 850, 525)  # [left, top, right, bottom]
    
        # Extract de tekst uit de PDF om het label te bepalen
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
        img = extract_ecg_image_from_pdf(tmp_pdf_path, clip_rect)
        if img is not None:
            st.image(img, caption='Geëxtraheerde ECG Afbeelding', use_column_width=True)
    
            # Preprocess afbeelding
            preprocessed_img = preprocess_image(img)
            if preprocessed_img is not None:
                st.image(preprocessed_img, caption='Geprocesste ECG Afbeelding (Grijswaarden & Binarisatie)', use_column_width=True)
    
                # Splits de afbeelding in lagen (vast op 3 lagen)
                num_layers = 3  # Aantal lagen vastgezet op 3
                layers = split_into_layers(preprocessed_img, num_layers=num_layers)
                logger.info(f"Aantal lagen: {num_layers}")
                st.write(f"Aantal lagen: {num_layers}")
    
                # Toon de lagen onder elkaar met voorspellingen
                st.subheader("Lagen van de ECG Afbeelding met Voorspellingen")
                layers_info = []  # Lijst om informatie per laag op te slaan
    
                for layer_idx, layer in enumerate(layers, start=1):
                    st.write(f"### Laag {layer_idx}")
    
                    # Extract ECG-signaal
                    signal = extract_ecg_signal(layer)
    
                    # Detecteer R-peaks
                    height = np.max(signal) * 0.5
                    distance = 150  # Pas aan op basis van je ECG-data
                    peaks = detect_r_peaks(signal, height=height, distance=distance)
                    st.write(f"Aantal gedetecteerde R-peaks: {len(peaks)}")
                    logger.info(f"Aantal gedetecteerde R-peaks voor Laag {layer_idx}: {len(peaks)}")
    
                    # Split de afbeelding in hartslagen
                    heartbeats = split_image_by_peaks(layer, peaks, window=100)
                    st.write(f"Aantal hartslagsegmenten: {len(heartbeats)}")
                    logger.info(f"Aantal hartslagsegmenten voor Laag {layer_idx}: {len(heartbeats)}")
    
                    # Verwerk alle hartslagsegmenten en verzamel coördinaten
                    all_coordinates = []
                    for hb_idx, heartbeat in enumerate(heartbeats, start=1):
                        x_coords, y_coords = extract_coordinates(heartbeat)
                        if len(x_coords) == 0 or len(y_coords) == 0:
                            logger.warning(f"Hartslag {hb_idx} van Laag {layer_idx} bevat geen lijnpixels.")
                            st.warning(f"Hartslag {hb_idx} van Laag {layer_idx} bevat geen lijnpixels.")
                            continue
                        # Normaliseer coördinaten
                        x_norm = x_coords / heartbeat.size[0]
                        y_norm = y_coords / heartbeat.size[1]
                        # Interpoleer coördinaten
                        coordinates = interpolate_coordinates(x_norm, y_norm, num_points=100)
                        all_coordinates.append(coordinates)
    
                    if not all_coordinates:
                        logger.error(f"Geen geldige hartslagsegmenten gevonden voor Laag {layer_idx}.")
                        st.error(f"Geen geldige hartslagsegmenten gevonden voor Laag {layer_idx}.")
                        continue
    
                    # Gemiddelde coördinaten over alle hartslagsegmenten in deze laag
                    average_coordinates = np.mean(all_coordinates, axis=0).reshape(1, -1)
    
                    # Reshape data voor LSTM [samples, timesteps, features]
                    X_pred = average_coordinates.reshape((1, 100, 2))
    
                    # Voorspel met het model
                    prediction_prob = model.predict(X_pred)[0][0]
                    # Bepaal voorspelde klasse
                    predicted_class = 'Abnormal' if prediction_prob > 0.5 else 'Normal'
                    prediction_prob = prediction_prob if prediction_prob > 0.5 else 1 - prediction_prob
    
                    st.write(f"**Voorspelde Klasse voor Laag {layer_idx}:** {predicted_class}")
                    st.write(f"**Voorspelde Kans:** {prediction_prob:.2f}")
                    logger.info(f"Voorspelde Klasse voor Laag {layer_idx}: {predicted_class} met kans {prediction_prob:.2f}")
    
                    # Voeg informatie toe aan layers_info voor PDF
                    layers_info.append({
                        'Laag': layer_idx,
                        'Predicted Class': predicted_class,
                        'Prediction Probability': prediction_prob,
                        'Layer Image': layer  # Sla de PIL Image op voor PDF
                    })
    
                # Genereer PDF als er lagen zijn verwerkt
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
            
            # Toon download button voor logbestand
            st.subheader("Download Logbestand")
            download_log()
        else:
            st.error("Fout bij het verwerken van de ECG-afbeelding.")
    
        # Optioneel: Verwijder het tijdelijke bestand na verwerking
        try:
            if uploaded_file is not None and os.path.exists(tmp_pdf_path):
                os.remove(tmp_pdf_path)
                logger.info("Tijdelijke PDF-bestand succesvol verwijderd.")
        except PermissionError:
            logger.warning("Kan het tijdelijke bestand niet verwijderen. Controleer of het nog in gebruik is.")
            st.warning("Kan het tijdelijke bestand niet verwijderen. Controleer of het nog in gebruik is.")
else:
    st.warning("Je moet akkoord gaan met de disclaimer voordat je een bestand kunt uploaden.")
