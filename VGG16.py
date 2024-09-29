import fitz  # PyMuPDF voor het werken met PDF's
from PIL import Image, ImageDraw
import os
import shutil
import logging
from logging.handlers import RotatingFileHandler
import numpy as np
from scipy.signal import find_peaks
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import cv2  # OpenCV voor beeldverwerking
from scipy.interpolate import interp1d
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from collections import Counter

# Instellen van logging naar bestand en console
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Handler voor logbestand met rotatie
file_handler = RotatingFileHandler('model.log', maxBytes=5*1024*1024, backupCount=3)  # 5 MB per bestand, maximaal 3 bestanden
file_handler.setLevel(logging.INFO)

# Handler voor console output
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Formatter voor logberichten
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Voeg handlers toe aan de logger
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

logger.info("Start van het ECG-verwerkings- en modeltrainingsscript.")

# Functie om ECG-afbeelding uit PDF te halen
def extract_ecg_image_from_pdf(pdf_path, output_path, clip_rect, zoom=2.0):
    """Haalt de ECG-afbeelding uit een PDF en slaat deze op als PNG."""
    try:
        pdf_document = fitz.open(pdf_path)
        page = pdf_document.load_page(0)  # Laad eerste pagina
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, clip=clip_rect)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img.save(output_path, dpi=(300, 300))
        logger.info(f"ECG-afbeelding opgeslagen als {output_path}.")
        return img
    except Exception as e:
        logger.error(f"Fout bij het extraheren van ECG-afbeelding uit {pdf_path}: {e}")
        return None

# Functie om afbeelding te preprocessen
def preprocess_image(img):
    """Converteert afbeelding naar grijs en maakt deze binair."""
    try:
        gray = img.convert("L")  # Naar grijswaarden
        gray_np = np.array(gray)
        _, binary = cv2.threshold(gray_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary_img = Image.fromarray(binary)
        logger.info("Afbeelding omgezet naar grijswaarden en binair gemaakt.")
        return binary_img
    except Exception as e:
        logger.error(f"Fout bij preprocessing van afbeelding: {e}")
        return None

# Functie om ECG-signaal te halen uit afbeelding
def extract_ecg_signal(img):
    """Haalt ECG-signaal uit binariseerde afbeelding door gemiddelde intensiteit per kolom."""
    img_array = np.array(img)
    signal = img_array.mean(axis=0)
    return signal

# Functie om R-peaks te vinden
def detect_r_peaks(signal, height=None, distance=None):
    """Vindt R-peaks in ECG-signaal."""
    peaks, properties = find_peaks(signal, height=height, distance=distance)
    return peaks

# Functie om afbeelding te splitsen rond R-peaks
def split_image_by_peaks(img, peaks, window=50):
    """Splitst afbeelding rondom R-peaks."""
    width, height = img.size
    sections = []
    for peak in peaks:
        left = max(peak - window, 0)
        right = min(peak + window, width)
        section = img.crop((left, 0, right, height))
        sections.append(section)
    return sections

# Functie om coördinaten van lijnen te halen
def extract_coordinates(img):
    """Haal X en Y coördinaten van witte pixels in binariseerde afbeelding."""
    img_array = np.array(img)
    y_coords, x_coords = np.where(img_array > 0)  # Witte pixels
    return x_coords, y_coords

# Functie om label te bepalen op basis van tekst in PDF
def determine_label(pdf_text):
    """Bepaalt of ECG normaal of abnormaal is op basis van tekst."""
    if 'Sinus Rhythm' in pdf_text or 'Normal' in pdf_text:
        return 'normal'
    else:
        return 'abnormal'

# Functie om coördinaten te interpoleren naar vaste lengte
def interpolate_coordinates(x, y, num_points=100):
    """Interpoleert coördinaten naar vaste lengte."""
    if len(x) < 2:
        return np.zeros(num_points*2)
    try:
        # Zorg dat x en y floats zijn
        x = [float(coord) for coord in x]
        y = [float(coord) for coord in y]
        
        f_x = interp1d(np.linspace(0, 1, len(x)), x, kind='linear', fill_value="extrapolate")
        f_y = interp1d(np.linspace(0, 1, len(y)), y, kind='linear', fill_value="extrapolate")
        x_interp = f_x(np.linspace(0, 1, num_points))
        y_interp = f_y(np.linspace(0, 1, num_points))
        return np.concatenate([x_interp, y_interp])
    except Exception as e:
        logger.error(f"Fout bij interpolatie: {e}")
        return np.zeros(num_points*2)

# Functie om PNG-afbeeldingen te maken vanuit CSV
def generate_png_images(csv_path, output_dir, image_size=(500, 500)):
    """Maakt PNG-afbeeldingen met zwarte stippen op witte achtergrond op basis van CSV-coördinaten."""
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    for idx, row in df.iterrows():
        filename = row['filename']
        label = row['label']
        x_coords = eval(row['x_coords'])  # String naar lijst
        y_coords = eval(row['y_coords'])  # String naar lijst

        # Maak witte afbeelding
        img = Image.new('RGB', image_size, 'white')
        draw = ImageDraw.Draw(img)

        # Teken zwarte stippen
        for x, y in zip(x_coords, y_coords):
            x_pixel = int(x * (image_size[0] - 1))
            y_pixel = int((1 - y) * (image_size[1] - 1))  # Y-as omkeren
            radius = 2  # Straal van stip
            draw.ellipse((x_pixel - radius, y_pixel - radius, x_pixel + radius, y_pixel + radius), fill='black')

        # Sla afbeelding op
        image_filename = f"{filename}"
        save_path = os.path.join(output_dir, image_filename)
        img.save(save_path, 'PNG')
        logger.info(f"PNG-afbeelding opgeslagen als {save_path}.")

# Instellen van output directories
output_image_dir = 'ECG/training_images'
os.makedirs(output_image_dir, exist_ok=True)

pdf_folder = 'ECG/pdfs'  # Map met PDF's
clip_rect = fitz.Rect(20, 200, 850, 525)  # Gebied om ECG-afbeelding te knippen

csv_output = 'ecg_data.csv'
data = []

logger.info("Start van PDF-verwerking en ECG-coördinaten extractie.")

# Verwerken van elke PDF in de map
for pdf_file in os.listdir(pdf_folder):
    if pdf_file.lower().endswith('.pdf'):
        pdf_path = os.path.join(pdf_folder, pdf_file)
        try:
            pdf_document = fitz.open(pdf_path)
            page = pdf_document.load_page(0)
            page_text = page.get_text()
            label = determine_label(page_text)
            logger.info(f"Verwerken van PDF: {pdf_file} met label: {label}")

            # Haal ECG-afbeelding uit PDF
            img = extract_ecg_image_from_pdf(pdf_path, "temp_ecg.png", clip_rect)
            if img is None:
                continue

            # Preprocess de afbeelding
            preprocessed_img = preprocess_image(img)
            if preprocessed_img is None:
                continue

            # Splits de afbeelding in lagen
            num_layers = 3  # Aantal lagen
            width, height = preprocessed_img.size
            layer_height = height // num_layers
            layers = []
            for i in range(num_layers):
                top = i * layer_height
                bottom = (i + 1) * layer_height if i < num_layers - 1 else height
                layer = preprocessed_img.crop((0, top, width, bottom))
                layers.append(layer)
            logger.info(f"Aantal lagen: {num_layers} voor {pdf_file}.")

            # Verwerk elke laag
            for layer_idx, layer in enumerate(layers, start=1):
                logger.info(f"Verwerken van Laag {layer_idx} voor {pdf_file}.")

                # Haal ECG-signaal uit laag
                signal = extract_ecg_signal(layer)

                # Zoek R-peaks
                peaks = detect_r_peaks(signal, height=np.max(signal)*0.5, distance=150)
                logger.info(f"{len(peaks)} R-peaks gevonden voor Laag {layer_idx} van {pdf_file}.")

                # Splits afbeelding in hartslagen
                heartbeats = split_image_by_peaks(layer, peaks, window=100)
                logger.info(f"{len(heartbeats)} hartslagsegmenten gesplitst voor Laag {layer_idx} van {pdf_file}.")

                for hb_idx, heartbeat in enumerate(heartbeats, start=1):
                    x_coords, y_coords = extract_coordinates(heartbeat)
                    if len(x_coords) == 0 or len(y_coords) == 0:
                        logger.warning(f"Hartslag {hb_idx} van Laag {layer_idx} van {pdf_file} bevat geen lijnen.")
                        continue
                    # Normaliseer coördinaten
                    x_norm = x_coords / heartbeat.size[0]
                    y_norm = y_coords / heartbeat.size[1]
                    # Interpoleer coördinaten
                    coordinates = interpolate_coordinates(x_norm, y_norm, num_points=100)
                    # Voeg data toe
                    data.append({
                        'filename': f"{os.path.splitext(pdf_file)[0]}_layer{layer_idx}_heartbeat_{hb_idx}.png",
                        'label': label,
                        'x_coords': list(x_norm),
                        'y_coords': list(y_norm)
                    })
                    logger.info(f"Hartslag {hb_idx} van Laag {layer_idx} van {pdf_file} toegevoegd aan data.")

                    # Sla de hartslagafbeelding op
                    heartbeat_filename = f"{os.path.splitext(pdf_file)[0]}_layer{layer_idx}_heartbeat_{hb_idx}.png"
                    save_path = os.path.join(output_image_dir, f'layer{layer_idx}', label, heartbeat_filename)
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    heartbeat.save(save_path, dpi=(300, 300))
                    logger.info(f"Hartslagafbeelding opgeslagen als {save_path}.")

            # Verwijder tijdelijke ECG-afbeelding
            if os.path.exists("temp_ecg.png"):
                os.remove("temp_ecg.png")
                logger.info("Tijdelijke ECG-afbeelding verwijderd.")

        except Exception as e:
            logger.error(f"Fout bij het verwerken van {pdf_file}: {e}", exc_info=True)

# Sla verzamelde data op als CSV
if data:
    df = pd.DataFrame(data)
    df.to_csv(csv_output, index=False)
    logger.info(f"ECG-coördinaten en labels opgeslagen in {csv_output}.")
else:
    # Maak leeg CSV-bestand aan
    df = pd.DataFrame(columns=['filename', 'label', 'x_coords', 'y_coords'])
    df.to_csv(csv_output, index=False)
    logger.warning(f"Geen data verzameld. Leeg CSV-bestand aangemaakt: {csv_output}")

# STAP 2: Data Voorbereiden voor Modeltraining
logger.info("Start van data voorbereiding voor modeltraining.")

# Controleer of CSV-bestand data bevat
if os.path.getsize(csv_output) > 0:
    # Laad CSV
    df = pd.read_csv(csv_output)

    # Functie om coördinaten te interpoleren naar vaste lengte
    def interpolate_coordinates_fixed(x, y, num_points=100):
        """Interpoleert coördinaten naar vaste lengte."""
        if len(x) < 2:
            return np.zeros(num_points*2)
        try:
            f_x = interp1d(np.linspace(0, 1, len(x)), x, kind='linear', fill_value="extrapolate")
            f_y = interp1d(np.linspace(0, 1, len(y)), y, kind='linear', fill_value="extrapolate")
            x_interp = f_x(np.linspace(0, 1, num_points))
            y_interp = f_y(np.linspace(0, 1, num_points))
            return np.concatenate([x_interp, y_interp])
        except Exception as e:
            logger.error(f"Fout bij interpolatie: {e}")
            return np.zeros(num_points*2)

    # Interpoleer alle coördinaten
    df['coordinates'] = df.apply(lambda row: interpolate_coordinates_fixed(
        row['x_coords'].strip('[]').split(','), 
        row['y_coords'].strip('[]').split(',')), axis=1)

    # Verwijder rijen zonder data
    df = df[df['coordinates'].map(len) > 0]

    # Maak features en labels
    X = np.stack(df['coordinates'].values)
    y = df['label'].map({'normal': 0, 'abnormal': 1}).values

    # Split data in train, val en test
    X_train, X_temp, y_train, y_temp, train_indices, temp_indices = train_test_split(
        X, y, df.index, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test, val_indices, test_indices = train_test_split(
        X_temp, y_temp, temp_indices, test_size=0.5, random_state=42, stratify=y_temp)

    logger.info(f"Data gesplitst in {X_train.shape[0]} trainings, {X_val.shape[0]} validatie en {X_test.shape[0]} test samples.")

    # Sla splits op als aparte CSV-bestanden
    train_df = df.loc[train_indices]
    val_df = df.loc[val_indices]
    test_df = df.loc[test_indices]

    train_csv = 'ecg_train_data.csv'
    val_csv = 'ecg_val_data.csv'
    test_csv = 'ecg_test_data.csv'

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    logger.info("Train, validatie en test CSV-bestanden opgeslagen.")

    # Organiseer afbeeldingen in aparte mappen
    base_dir = 'ECG/dataset_split'
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')
    test_dir = os.path.join(base_dir, 'test')

    # Maak directories aan
    for directory in [train_dir, val_dir, test_dir]:
        os.makedirs(directory, exist_ok=True)
        for layer in range(1, 4):  # Aantal lagen
            for label in ['normal', 'abnormal']:
                os.makedirs(os.path.join(directory, f'layer{layer}', label), exist_ok=True)

    # Functie om afbeeldingen te kopiëren naar juiste map
    def copy_images(df_subset, destination_dir):
        for idx, row in df_subset.iterrows():
            for layer in range(1, 4):
                if f'layer{layer}' in row['filename']:
                    src = os.path.join(output_image_dir, f'layer{layer}', row['label'], row['filename'])
                    dest = os.path.join(destination_dir, f'layer{layer}', row['label'], row['filename'])
                    if os.path.exists(src):
                        shutil.copy(src, dest)
                        logger.info(f"Gekopieerd {src} naar {dest}.")
                    else:
                        logger.warning(f"Bronbestand {src} bestaat niet.")

    # Kopieer afbeeldingen naar mappen
    copy_images(train_df, train_dir)
    copy_images(val_df, val_dir)
    copy_images(test_df, test_dir)

    logger.info("Afbeeldingen georganiseerd in train, validatie en test mappen.")

    # Check distributie van labels
    logger.info(f"Train labels distribution: {Counter(y_train)}")
    logger.info(f"Validation labels distribution: {Counter(y_val)}")
    logger.info(f"Test labels distribution: {Counter(y_test)}")

    # STAP 3: Model Bouw en Training
    logger.info("Start van modelbouw en training.")

    # Eenvoudig neuraal netwerk
    model = Sequential([
        Input(shape=(200,)),  # Vervangt input_shape in eerste Dense-laag
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    # Compileer model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    logger.info("Model gedefinieerd en gecompileerd.")

    # Defineer callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max')
    logger.info("Callbacks ingesteld voor training.")

    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stop, checkpoint]
    )

    logger.info("Modeltraining voltooid.")

    # Evalueer model
    test_loss, test_acc = model.evaluate(X_test, y_test)
    logger.info(f"Test accuraatheid: {test_acc}")
    print(f"Test accuraatheid: {test_acc}")

    # Classification Report
    y_pred_prob = model.predict(X_test).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Abnormal']))

    auroc = roc_auc_score(y_test, y_pred_prob)
    auprc = average_precision_score(y_test, y_pred_prob)
    print(f"AUROC: {auroc}")
    print(f"AUPRC: {auprc}")

    # Log resultaten
    logger.info(f"Classification Report:\n{classification_report(y_test, y_pred, target_names=['Normal', 'Abnormal'])}")
    logger.info(f"AUROC: {auroc}")
    logger.info(f"AUPRC: {auprc}")

    # Sla model op
    try:
        model.save('final_ecg_model.keras')
        logger.info("Definitieve model opgeslagen als 'final_ecg_model.keras'.")
    except Exception as e:
        logger.error(f"Fout bij opslaan van model: {e}", exc_info=True)

    # Genereer PNG-afbeeldingen vanuit CSV
    logger.info("Start van het genereren van PNG-afbeeldingen vanuit CSV data.")
    image_csv_dir = 'image_csv'
    generate_png_images(csv_output, image_csv_dir, image_size=(500, 500))
    logger.info(f"PNG-afbeeldingen opgeslagen in de map '{image_csv_dir}'.")

    logger.info("Modeltrainingscript voltooid.")
    print("Modeltrainingscript voltooid.")
else:
    logger.error(f"CSV-bestand {csv_output} is leeg. Geen data om te verwerken.")
    print(f"CSV-bestand {csv_output} is leeg. Geen data om te verwerken.")
