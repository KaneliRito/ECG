ECG-Analyse met LSTM en Streamlit

Welkom bij het ECG-Analyse Project, een Python-gebaseerde applicatie die gebruikmaakt van Long Short-Term Memory-netwerken (LSTM) om ECG-gegevens te analyseren en te classificeren als normaal of abnormaal. Deze tool is ontworpen om wachttijden voor de diagnose van hartafwijkingen te verkorten door gebruik te maken van ECG-gegevens afkomstig van smartwatches, specifiek de Apple Watch.


Installatie

Volg de onderstaande stappen om de benodigde dependencies te installeren en het project op te zetten:

    Clone de repository:

    bash

git clone https://github.com/jouw-gebruikersnaam/ecg-analyse.git
cd ecg-analyse

Maak een virtuele omgeving aan (optioneel maar aanbevolen):

bash

python -m venv venv
source venv/bin/activate  # Voor Windows: venv\Scripts\activate

Installeer de benodigde packages:

bash

    pip install -r requirements.txt

Gebruik
Webapplicatie

    Start de Streamlit webapplicatie:

    bash

    streamlit run webapp/app.py

    Interactieve Interface:
        Ga naar de URL die in de terminal wordt weergegeven (meestal http://localhost:8501).
        Accepteer de disclaimer.
        Upload een ECG-PDF-bestand van een Apple Watch.
        Bekijk de geanalyseerde lagen en de voorspellingen.
        Download de resultaten als PDF en het logbestand.

Model Training

    Voer het hoofdscript uit:

    bash

    python scripts/main.py

    Dit script:
        Verwerkt de PDF's en extraheert ECG-gegevens.
        Bereidt de data voor en normaliseert deze.
        Trained en valideert het LSTM-model met stratified K-Fold cross-validatie.
        Slaat het getrainde model op als final_model.keras.

Data
Beschikbaarheid

Hoewel er verschillende openbare datasets beschikbaar zijn voor aritmie-detectie, is ervoor gekozen om een eigen dataset te creëren om de betrouwbaarheid en relevantie voor smartwatch-ECG's te waarborgen. Deze dataset is gebaseerd op ECG-PDF's verzameld via een Apple Watch.
ETL Proces

Het ETL-proces omvat:

    Extractie: Verzamelen van ECG-PDF's van de Apple Watch.
    Transformatie: Omzetten van PDF's naar PNG-afbeeldingen, extraheren van ECG-signalen, detecteren van R-peaks en normaliseren van coördinaten.
    Laden: Opslaan van de verwerkte gegevens in CSV-bestanden en gestructureerde mappen.

Model Training
Architectuur

Het LSTM-model bestaat uit:

    Twee LSTM-lagen: Met respectievelijk 128 en 64 geheugencellen.
    Batch Normalization en Dropout: Voor stabilisatie en het voorkomen van overfitting.
    Dense-lagen: Met ReLU-activatie voor verwerking en Sigmoid-activatie voor binaire classificatie.

Validatie

    Stratified K-Fold Cross-Validation: Met 5 folds om de modelprestaties te evalueren.
    Metric Evaluatie: F1-score, Precision, Recall, AUROC en AUPRC worden gebruikt om de nauwkeurigheid te waarborgen.

Resultaten

Het uiteindelijke model heeft een gemiddelde AUROC van ongeveer 0.78 en een F1-score van 0.71, wat aangeeft dat het model effectief is in het onderscheiden van normale en abnormale ECG-gegevens.
Webapplicatie

De webapplicatie, gebouwd met Streamlit, biedt een gebruiksvriendelijke interface voor het uploaden en analyseren van ECG-PDF's. Belangrijke functionaliteiten zijn onder andere:

    Uploaden van PDF's: Gebruikers kunnen eenvoudig hun ECG-PDF's uploaden.
    Realtime Analyse: Het model verwerkt de data en toont de voorspellingen.
    Downloaden van Resultaten: Gebruikers kunnen de analyseresultaten en logbestanden downloaden.
    Privacy: Alle tijdelijke bestanden worden na verwerking verwijderd om de privacy te waarborgen.

Logging

Tijdens het gehele proces wordt uitgebreid gelogd om transparantie en foutopsporing te waarborgen. Alle logs worden opgeslagen in model.log en kunnen gedownload worden via de webapplicatie.

Logformaat:

yaml

2024-10-06 12:29:01,498 - INFO - Verwerken van Laag 1 voor abnormal.pdf.

Bias en Ethische Aspecten
Bias

    Dataset Beperkingen: De dataset is gebaseerd op gegevens van één persoon, wat leidt tot selectiebias en beperkte generaliseerbaarheid.
    Apparaat-specifiek: Data is verzameld via een Apple Watch, waardoor variaties tussen verschillende smartwatches niet gedekt worden.
    Kleine Dataset: Met slechts 40 PDF's bestaat het risico op overfitting, wat de prestaties op nieuwe data kan beïnvloeden.

Ethische Overwegingen

    Privacy: Alle geüploade gegevens worden na verwerking verwijderd om misbruik te voorkomen.
    Transparantie: Het model is transparant over hoe beslissingen worden genomen en biedt gedetailleerde logs.
    Disclaimer: De applicatie is bedoeld voor demonstratiedoeleinden en vervangt geen professioneel medisch advies.

Onderzoeksresultaten
Hoofdvraag

Hoe kan AI worden ingezet om de wachttijden voor de diagnose van hartafwijkingen te verkorten op basis van ECG-gegevens uit smartwatches?
Deelvragen

    Welke AI-algoritmen zijn het meest geschikt voor de analyse van ECG-gegevens?
        Convolutionele Neurale Netwerken (CNN) en Long Short-Term Memory-netwerken (LSTM) zijn het meest geschikt.

    Hoe wordt de nauwkeurigheid van AI-modellen voor ECG-analyse gewaarborgd?
        Door data voorbereiding, validatie met AUROC en uitgebreide cross-validatie.

    Hoe kan AI afwijkingen in ECG-gegevens detecteren?
        Door het omzetten van ECG-afbeeldingen naar genormaliseerde X- en Y-coördinaten en het herkennen van patronen met LSTM.

    Wat zijn de ethische aspecten bij de implementatie van de ECG-analyse?
        Privacybescherming, transparantie, en het benadrukken van de beperkingen van het model.

    Welke juridische aspecten komen er kijken bij de implementatie van de ECG-analyse?
        Naleving van AVG en EU AI Act, expliciete toestemming voor dataverwerking, en CE-markering voor medische hulpmiddelen.

    Wat voor soort dataset kan het best worden gebruikt voor aritmie-detectie?
        Een diverse en representatieve dataset met variatie in personen, leeftijden, en apparatuur.

    Welke tooling past het best bij het realiseren van de training van AI-modellen?
        Python, TensorFlow, Keras, Pandas, Matplotlib, OpenCV, en PyMuPDF.

    Wat voor meetmethode past er bij dit AI-model?
        Classificatie gebaseerd op binaire labels (normaal of abnormaal) met evaluatie via F1-score, Precision, Recall, AUROC en AUPRC.

Contributie

Bijdragen aan dit project zijn welkom! Volg de onderstaande stappen om bij te dragen:

    Fork de repository.
    Maak een feature branch:

    bash

git checkout -b feature/NaamVanJeFeature

Commit je wijzigingen:

bash

git commit -m "Beschrijving van je feature"

Push naar de branch:

bash

    git push origin feature/NaamVanJeFeature

    Open een Pull Request.

Licentie

Dit project is gelicentieerd onder de MIT License. Zie het LICENSE bestand voor details.
Acknowledgements

    TensorFlow en Keras: Voor krachtige machine learning tools.
    Streamlit: Voor het mogelijk maken van interactieve webapplicaties.
    OpenAI's ChatGPT: Voor het ondersteunen bij de ontwikkeling van de front-end.
    Beth Israel Deaconess Medical Center en MIT: Voor het leveren van de MIT-BIH Arrhythmia Database.
    Alle bijdragers: Voor hun waardevolle input en ondersteuning.
