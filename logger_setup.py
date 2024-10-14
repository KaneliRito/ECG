# logger_setup.py

import logging
from logging import FileHandler

def setup_logger():
    # Set up logging to file only
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Handler for log file 
    file_handler = FileHandler('ecg\\model.log')  
    file_handler.setLevel(logging.INFO)

    # Formatter for log messages
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add only the file handler to the logger
    if not logger.handlers:
        logger.addHandler(file_handler)
