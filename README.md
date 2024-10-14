# ECG-Analyse met LSTM en Streamlit

Welkom bij het **ECG-Analyse Project**, een Python-gebaseerde applicatie die gebruikmaakt van Long Short-Term Memory-netwerken (LSTM) om ECG-gegevens te analyseren en te classificeren als normaal of abnormaal. Deze tool is ontworpen om wachttijden voor de diagnose van hartafwijkingen te verkorten door gebruik te maken van ECG-gegevens afkomstig van smartwatches, specifiek de Apple Watch.

Dit project richt zich op het ontwikkelen van een AI-model dat ECG-gegevens kan analyseren en classificeren als normaal of abnormaal. Het model maakt gebruik van LSTM-netwerken vanwege hun efficiëntie in het verwerken van sequentiële data. De toepassing wordt ondersteund door een gebruiksvriendelijke webapplicatie gebouwd met Streamlit, waarmee gebruikers hun ECG-PDF's kunnen uploaden en direct resultaten kunnen ontvangen.

## Technologieën

- **Programmeertaal:** Python
- **Machine Learning:** TensorFlow, Keras
- **Data Processing:** Pandas, NumPy, OpenCV, PyMuPDF
- **Webapplicatie:** Streamlit
- **Visualisatie:** Matplotlib
- **PDF Generatie:** FPDF


## Installatie

Volg de onderstaande stappen om de benodigde dependencies te installeren en het project op te zetten:

1. **Clone de repository:**
    ```bash
    git clone https://github.com/KaneliRito/ECG.git
    cd ecg-analyse
    ```

2. **Maak een virtuele omgeving aan (optioneel maar aanbevolen):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Voor Windows: venv\Scripts\activate
    ```

3. **Installeer de benodigde packages:**
    ```bash
    pip install -r requirements.txt
    ```

## Gebruik

### Webapplicatie

1. **Start de Streamlit webapplicatie:**
    ```bash
    streamlit run webapp/app.py
    ```

2. **Interactieve Interface:**
    - Ga naar de URL die in de terminal wordt weergegeven (meestal `http://localhost:8501`).
    - Accepteer de disclaimer.
    - Upload een ECG-PDF-bestand van een Apple Watch.
    - Bekijk de geanalyseerde lagen en de voorspellingen.
    - Download de resultaten als PDF en het logbestand.

### Model Training

1. **Voer het hoofdscript uit:**
    ```bash
    python scripts/main.py
    ```

    Dit script:
    - Verwerkt de PDF's en extraheert ECG-gegevens.
    - Bereidt de data voor en normaliseert deze.
    - Trained en valideert het LSTM-model met stratified K-Fold cross-validatie.
    - Slaat het getrainde model op als `final_model.keras`.

## Data

### Beschikbaarheid

Hoewel er verschillende openbare datasets beschikbaar zijn voor aritmie-detectie, is ervoor gekozen om een eigen dataset te creëren om de betrouwbaarheid en relevantie voor smartwatch-ECG's te waarborgen. Deze dataset is gebaseerd op ECG-PDF's verzameld via een Apple Watch.

### ETL Proces

Het ETL-proces omvat:
- **Extractie:** Verzamelen van ECG-PDF's van de Apple Watch.
- **Transformatie:** Omzetten van PDF's naar PNG-afbeeldingen, extraheren van ECG-signalen, detecteren van R-peaks en normaliseren van coördinaten.
- **Laden:** Opslaan van de verwerkte gegevens in CSV-bestanden en gestructureerde mappen.

## Model Training

### Architectuur

Het LSTM-model bestaat uit:
- **Twee LSTM-lagen:** Met respectievelijk 128 en 64 geheugencellen.
- **Batch Normalization en Dropout:** Voor stabilisatie en het voorkomen van overfitting.
- **Dense-lagen:** Met ReLU-activatie voor verwerking en Sigmoid-activatie voor binaire classificatie.

### Validatie

- **Stratified K-Fold Cross-Validation:** Met 5 folds om de modelprestaties te evalueren.
- **Metric Evaluatie:** F1-score, Precision, Recall, AUROC en AUPRC worden gebruikt om de nauwkeurigheid te waarborgen.

### Resultaten

Het uiteindelijke model heeft een gemiddelde AUROC van ongeveer 0.78 en een F1-score van 0.71, wat aangeeft dat het model effectief is in het onderscheiden van normale en abnormale ECG-gegevens.

Fold 1:	                        Fold 4:
F1 Score: 0.7407407407407407	F1 Score: 0.7411764705882353
Precision: 0.8955223880597015	Fold 4 - Precision: 0.84
Recall: 0.631578947368421	    Recall: 0.6631578947368421
AUROC: 0.8510964912280703	    AUROC: 0.8224780701754387
AUPRC: 0.8864009924159035	    AUPRC: 0.8657702110900989
Fold 2:	Fold 5:
F1 Score: 0.75	                F1 Score: 0.6666666666666666
Precision: 0.863013698630137	Precision: 0.828125
Recall: 0.6631578947368421	    Recall: 0.5578947368421052
AUROC: 0.8396929824561404	    AUROC: 0.7721883656509695
AUPRC: 0.8627379863634974	    AUPRC: 0.8158282779556207
Fold 3:	                        Final model:
F1 Score: 0.7407407407407407	Accuracy: 0.74737948179245
Precision: 0.8955223880597015	F1 Score: 0.7085852478839177
Recall: 0.631578947368421	    Precision: 0.8347578347578347
AUROC: 0.8510964912280703	    Recall: 0.615546218487395
AUPRC: 0.8864009924159035	    AUROC: 0.7886457930452516
	                            AUPRC: 0.8385686271442826


## Webapplicatie

De webapplicatie, gebouwd met Streamlit, biedt een gebruiksvriendelijke interface voor het uploaden en analyseren van ECG-PDF's. Belangrijke functionaliteiten zijn onder andere:

- **Uploaden van PDF's:** Gebruikers kunnen eenvoudig hun ECG-PDF's uploaden.
- **Realtime Analyse:** Het model verwerkt de data en toont de voorspellingen.
- **Downloaden van Resultaten:** Gebruikers kunnen de analyseresultaten en logbestanden downloaden.
- **Privacy:** Alle tijdelijke bestanden worden na verwerking verwijderd om de privacy te waarborgen.

## Logging

Tijdens het gehele proces wordt uitgebreid gelogd om transparantie en foutopsporing te waarborgen. Alle logs worden opgeslagen in `model.log` en kunnen gedownload worden via de webapplicatie.

**Logformaat:**
2024-10-06 12:29:01,498 - INFO - Verwerken van Laag 1 voor abnormal.pdf.
