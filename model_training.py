# model_training.py

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, roc_curve, auc
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
import os
from logger_setup import setup_logger  # Custom logger setup

# Initialize the logger for this module
logger = logging.getLogger(__name__)

# Directory paths
ROC_DIR = 'ECG\\roc_curves'  # Directory to save ROC curve images


def create_dirs(*dirs):
    """
    Creates directories if they do not exist.

    Parameters:
    - dirs (str): Variable number of directory paths to create.

    Returns:
    - None
    """
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
        logger.debug(f"Verified/created directory: {directory}")


def create_model(input_shape=(100, 2)):
    """
    Creates and compiles an LSTM-based neural network model.

    The model architecture includes:
    - Two LSTM layers with Batch Normalization and Dropout.
    - A Dense layer with ReLU activation.
    - An output Dense layer with Sigmoid activation for binary classification.

    Parameters:
    - input_shape (tuple): Shape of the input data (time_steps, features). Defaults to (100, 2).

    Returns:
    - tensorflow.keras.models.Sequential: Compiled LSTM model.
    """
    model = Sequential([
        Input(shape=input_shape),
        LSTM(128, return_sequences=True),
        BatchNormalization(),
        Dropout(0.5),
        LSTM(64),
        BatchNormalization(),
        Dropout(0.5),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    logger.debug("Model created and compiled.")
    return model


def log_metrics(fold_num, metrics):
    """
    Logs evaluation metrics for a specific fold or the final model.

    Parameters:
    - fold_num (int or str): Identifier for the fold (e.g., fold number or 'final').
    - metrics (list): List of metric values in the order:
                      [F1 Score, Precision, Recall, AUROC, AUPRC].

    Returns:
    - None
    """
    f1, precision_val, recall_val, auroc_val, auprc_val = metrics
    logger.info(f"Fold {fold_num} - F1 Score: {f1:.4f}")
    logger.info(f"Fold {fold_num} - Precision: {precision_val:.4f}")
    logger.info(f"Fold {fold_num} - Recall: {recall_val:.4f}")
    logger.info(f"Fold {fold_num} - AUROC: {auroc_val:.4f}")
    logger.info(f"Fold {fold_num} - AUPRC: {auprc_val:.4f}")


def compute_class_weights(y):
    """
    Computes class weights to handle class imbalance.

    Parameters:
    - y (np.ndarray): Array of target labels.

    Returns:
    - dict: Dictionary mapping class indices to their respective weights.
    """
    class_weights_values = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weights_dict = {0: class_weights_values[0], 1: class_weights_values[1]}
    logger.debug(f"Computed class weights: {class_weights_dict}")
    return class_weights_dict


def compute_roc_curve(y_val, y_pred_prob, fold_num):
    """
    Computes and saves the ROC curve for a given fold.

    Parameters:
    - y_val (np.ndarray): True labels for the validation set.
    - y_pred_prob (np.ndarray): Predicted probabilities for the positive class.
    - fold_num (int or str): Identifier for the fold (e.g., fold number or 'final').

    Returns:
    - tuple: False Positive Rates (fpr), True Positive Rates (tpr), and Area Under the Curve (roc_auc).
    """
    # Calculate ROC curve metrics
    fpr, tpr, _ = roc_curve(y_val, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line for random guessing
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - Fold {fold_num}')
    plt.legend(loc="lower right")
    
    # Save ROC curve plot
    roc_path = os.path.join(ROC_DIR, f'roc_fold{fold_num}.png')
    plt.savefig(roc_path)
    plt.close()
    logger.info(f"Fold {fold_num} - ROC curve saved as {roc_path}.")
    
    return fpr, tpr, roc_auc


def perform_cross_validation(X, y, n_splits=5):
    """
    Performs Stratified K-Fold Cross-Validation to evaluate the model.

    Parameters:
    - X (np.ndarray): Feature matrix.
    - y (np.ndarray): Target labels.
    - n_splits (int): Number of folds for cross-validation. Defaults to 5.

    Returns:
    - None
    """
    logger.info("Starting Stratified K-Fold Cross-Validation.")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Compute class weights for the entire dataset
    class_weights_dict_full = compute_class_weights(y)
    logger.info(f"Class weights: {class_weights_dict_full}")

    # Create directory for ROC curves if it doesn't exist
    create_dirs(ROC_DIR)

    # Initialize lists to store metrics for each fold
    fold_accuracies, fold_f1_scores, fold_precision, fold_recall, fold_auroc, fold_auprc = [], [], [], [], [], []
    mean_fpr = np.linspace(0, 1, 100)  # Common False Positive Rates for mean ROC
    tprs, aucs = [], []

    # Iterate over each fold
    for fold_num, (train_index, val_index) in enumerate(tqdm(skf.split(X, y), total=n_splits, desc="Stratified K-Fold")):
        logger.info(f"--- Starting Fold {fold_num + 1} ---")
        # Split data into training and validation sets
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Compute class weights based on training data to handle class imbalance
        class_weights_dict = compute_class_weights(y_train)

        # Create and compile the LSTM model
        model = create_model()
        
        # Define callbacks for early stopping and model checkpointing
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint(f'ecg/best_model_fold{fold_num + 1}.keras', monitor='val_accuracy', save_best_only=True, mode='max')
        ]

        # Train the model on the training set with validation
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            class_weight=class_weights_dict,
            verbose=0  # Suppress training logs
        )

        # Evaluate the model on the validation set
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        fold_accuracies.append(val_acc)
        logger.info(f"Fold {fold_num + 1} - Validation Accuracy: {val_acc:.4f}")

        # Predict probabilities on the validation set
        y_pred_prob = model.predict(X_val).flatten()
        # Convert probabilities to binary predictions with a threshold of 0.5
        y_pred = (y_pred_prob > 0.5).astype(int)

        # Calculate evaluation metrics
        metrics = [
            f1_score(y_val, y_pred),
            precision_score(y_val, y_pred),
            recall_score(y_val, y_pred),
            roc_auc_score(y_val, y_pred_prob),
            average_precision_score(y_val, y_pred_prob)
        ]

        # Log the computed metrics for the current fold
        log_metrics(fold_num + 1, metrics)

        # Append metrics to their respective lists
        fold_f1_scores.append(metrics[0])
        fold_precision.append(metrics[1])
        fold_recall.append(metrics[2])
        fold_auroc.append(metrics[3])
        fold_auprc.append(metrics[4])

        # Compute and save ROC curve for the current fold
        fpr, tpr, roc_auc = compute_roc_curve(y_val, y_pred_prob, fold_num + 1)
        aucs.append(roc_auc)
        # Interpolate TPR for mean ROC curve
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        # Ensure that the interpolated TPR starts at 0
        tprs[-1][0] = 0.0

    # After all folds, plot the mean ROC curve
    plt.figure()
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0  # Ensure that the last TPR value is 1
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='blue',
             label=f'Mean ROC (AUC = {mean_auc:.2f})', lw=2, alpha=.8)

    # Plot the diagonal line for reference
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    # Set plot limits and labels
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Mean Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")

    # Save the mean ROC curve plot
    mean_roc_path = os.path.join(ROC_DIR, 'mean_roc_curve.png')
    plt.savefig(mean_roc_path)
    plt.close()
    logger.info(f"Mean ROC curve saved as {mean_roc_path}.")

    # Log average metrics across all folds
    logger.info("Cross-validation results:")
    logger.info(f"Average Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
    logger.info(f"Average F1 Score: {np.mean(fold_f1_scores):.4f} ± {np.std(fold_f1_scores):.4f}")
    logger.info(f"Average Precision: {np.mean(fold_precision):.4f} ± {np.std(fold_precision):.4f}")
    logger.info(f"Average Recall: {np.mean(fold_recall):.4f} ± {np.std(fold_recall):.4f}")
    logger.info(f"Average AUROC: {np.mean(fold_auroc):.4f} ± {np.std(fold_auroc):.4f}")
    logger.info(f"Average AUPRC: {np.mean(fold_auprc):.4f} ± {np.std(fold_auprc):.4f}")

    logger.info("Stratified K-Fold Cross-Validation completed.")


def train_final_model(X, y):
    """
    Trains the final LSTM model on the entire dataset and saves the trained model.

    Parameters:
    - X (np.ndarray): Feature matrix.
    - y (np.ndarray): Target labels.

    Returns:
    - None
    """
    logger.info("Training final model on the entire dataset.")

    # Create and compile the final LSTM model
    final_model = create_model()
    class_weights_dict_full = compute_class_weights(y)

    # Define callbacks for early stopping and model checkpointing
    callbacks = [
        EarlyStopping(monitor='loss', patience=10, restore_best_weights=True),
        ModelCheckpoint('ecg/final_model.keras', monitor='accuracy', save_best_only=True, mode='max')
    ]

    # Train the final model on the entire dataset
    history = final_model.fit(
        X, y,
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        class_weight=class_weights_dict_full,
        verbose=0  # Suppress training logs
    )

    # Evaluate the final model on the entire dataset
    loss, acc = final_model.evaluate(X, y, verbose=0)
    logger.info(f"Final Model - Accuracy: {acc:.4f}")

    # Predict probabilities and convert them to binary predictions
    y_pred_prob = final_model.predict(X).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Calculate evaluation metrics for the final model
    metrics = [
        f1_score(y, y_pred),
        precision_score(y, y_pred),
        recall_score(y, y_pred),
        roc_auc_score(y, y_pred_prob),
        average_precision_score(y, y_pred_prob)
    ]

    # Log the computed metrics for the final model
    log_metrics('final', metrics)

    # Compute and save ROC curve for the final model
    compute_roc_curve(y, y_pred_prob, 'final')
    
    # Save the final trained model to disk
    final_model.save('ecg/final_model.keras')
    logger.info("Final model saved as 'ecg/final_model.keras'.")
