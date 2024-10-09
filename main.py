# main.py

from logger_setup import setup_logger
from ecg_processing import process_pdfs
from data_preparation import prepare_data
from model_training import perform_cross_validation, train_final_model
from utils import generate_png_images
import logging

def main():
    # Set up logging
    setup_logger()
    logger = logging.getLogger(__name__)
    logger.info("Start of the ECG processing and model training script.")

    # Process PDFs and extract ECG data
    csv_output = 'ecg_data.csv'
    
    process_pdfs(output_csv=csv_output)

    # Prepare data for model training
    X, y = prepare_data(csv_output)

    if X is not None and y is not None:
        image_csv_dir = 'image_csv'
        generate_png_images(csv_output, image_csv_dir, image_size=(500, 500))
        logger.info(f"PNG images saved in the folder '{image_csv_dir}'.")

        # Perform Stratified K-Fold Cross-Validation
        perform_cross_validation(X, y, n_splits=5)

        # Train final model on the entire dataset
        train_final_model(X, y)

        # Generate PNG images from CSV data

    else:
        logger.error(f"CSV file {csv_output} is empty or data preparation failed. No data to process.")

    logger.info("Model training script completed.")

if __name__ == "__main__":
    main()
