# model_training.py

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight
from sklearn.metrics import (f1_score, precision_score, recall_score, roc_auc_score, average_precision_score)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tqdm import tqdm, trange
import logging

logger = logging.getLogger(__name__)

def create_model():
    model = Sequential([
        Input(shape=(100, 2)),              # Input layer with sequence length 100 and 2 features
        LSTM(128, return_sequences=True),    # First LSTM layer
        BatchNormalization(),                # Batch Normalization
        Dropout(0.5),                        # Dropout for regularization
        LSTM(64),                            # Second LSTM layer
        BatchNormalization(),
        Dropout(0.5),
        Dense(64, activation='relu'),        # Dense layer
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')       # Output layer for binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def perform_cross_validation(X, y, n_splits=5):
    """Performs Stratified K-Fold Cross-Validation."""
    logger.info("Starting Stratified K-Fold Cross-Validation.")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Compute class weights for the entire dataset
    class_weights_values_full = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weights_dict_full = {0: class_weights_values_full[0], 1: class_weights_values_full[1]}
    logger.info(f"Class weights: {class_weights_dict_full}")

    # Initialize lists to store performance per fold
    fold_accuracies = []
    fold_f1_scores = []
    fold_precision = []
    fold_recall = []
    fold_auroc = []
    fold_auprc = []

    for fold, (train_index, val_index) in enumerate(tqdm(skf.split(X, y), total=n_splits, desc="Stratified K-Fold Cross-Validation")):
        fold_num = fold + 1
        logger.info(f"--- Starting Fold {fold_num} ---")

        # Split the data
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Compute class weights for this fold
        class_weights_values_fold = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights_dict = {0: class_weights_values_fold[0], 1: class_weights_values_fold[1]}
        logger.info(f"Fold {fold_num} - Class weights: {class_weights_dict}")

        # Define the model
        model = create_model()
        logger.info(f"Fold {fold_num} - Model defined and compiled.")

        # Define callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        checkpoint = ModelCheckpoint(f'ecg/best_model_fold{fold_num}.keras', monitor='val_accuracy', save_best_only=True, mode='max')
        logger.info(f"Fold {fold_num} - Callbacks set for training.")

        # Train the model
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[early_stop, checkpoint],
            class_weight=class_weights_dict,
            verbose=0  # Suppress Keras' own progress bar
        )

        # Manual progress bar for epochs
        for epoch in trange(len(history.history['loss']), desc=f"Fold {fold_num} Training", leave=False):
            logger.info(f"Fold {fold_num} - Epoch {epoch + 1}/{len(history.history['loss'])}")
            logger.info(f"Fold {fold_num} - Train Loss: {history.history['loss'][epoch]:.4f}, Train Accuracy: {history.history['accuracy'][epoch]:.4f}")
            logger.info(f"Fold {fold_num} - Val Loss: {history.history['val_loss'][epoch]:.4f}, Val Accuracy: {history.history['val_accuracy'][epoch]:.4f}")

        logger.info(f"Fold {fold_num} - Model training completed.")

        # Evaluate the model on the validation set
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        logger.info(f"Fold {fold_num} - Validation Accuracy: {val_acc}")
        fold_accuracies.append(val_acc)

        # Make predictions
        y_pred_prob = model.predict(X_val).flatten()
        y_pred = (y_pred_prob > 0.5).astype(int)

        # Compute additional evaluation metrics
        f1 = f1_score(y_val, y_pred)
        precision_val = precision_score(y_val, y_pred)
        recall_val = recall_score(y_val, y_pred)
        auroc_val = roc_auc_score(y_val, y_pred_prob)
        auprc_val = average_precision_score(y_val, y_pred_prob)

        # Append metrics to lists
        fold_f1_scores.append(f1)
        fold_precision.append(precision_val)
        fold_recall.append(recall_val)
        fold_auroc.append(auroc_val)
        fold_auprc.append(auprc_val)

        # Log the metrics
        logger.info(f"Fold {fold_num} - F1 Score: {f1}")
        logger.info(f"Fold {fold_num} - Precision: {precision_val}")
        logger.info(f"Fold {fold_num} - Recall: {recall_val}")
        logger.info(f"Fold {fold_num} - AUROC: {auroc_val}")
        logger.info(f"Fold {fold_num} - AUPRC: {auprc_val}")

    # Summary of average performance over all folds
    summary = (
        f"--- Summary of Stratified K-Fold Cross-Validation ---\n"
        f"Average Validation Accuracy over {n_splits} folds: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}\n"
        f"Average F1 Score over {n_splits} folds: {np.mean(fold_f1_scores):.4f} ± {np.std(fold_f1_scores):.4f}\n"
        f"Average Precision over {n_splits} folds: {np.mean(fold_precision):.4f} ± {np.std(fold_precision):.4f}\n"
        f"Average Recall over {n_splits} folds: {np.mean(fold_recall):.4f} ± {np.std(fold_recall):.4f}\n"
        f"Average AUROC over {n_splits} folds: {np.mean(fold_auroc):.4f} ± {np.std(fold_auroc):.4f}\n"
        f"Average AUPRC over {n_splits} folds: {np.mean(fold_auprc):.4f} ± {np.std(fold_auprc):.4f}\n"
    )

    logger.info(summary)

    # Save the summary in a text file
    with open('cross_validation_summary.txt', 'w') as f:
        f.write(summary)
    logger.info("Cross-validation summary saved as 'cross_validation_summary.txt'.")

def train_final_model(X, y):
    """Trains the final model on the entire dataset."""
    logger.info("Training final model on the entire dataset.")

    # Define the final model architecture
    final_model = create_model()
    logger.info("Final model defined and compiled.")

    # Compute class weights on the full dataset
    class_weights_values_full = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weights_dict_full = {0: class_weights_values_full[0], 1: class_weights_values_full[1]}
    logger.info(f"Class weights for full dataset: {class_weights_dict_full}")

    # Define callbacks for final model
    early_stop_final = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    checkpoint_final = ModelCheckpoint('ecg/final_model.keras', monitor='accuracy', save_best_only=True, mode='max')
    logger.info("Callbacks set for final model training.")

    # Train the final model
    final_model.fit(
        X, y,
        epochs=100,
        batch_size=32,
        callbacks=[early_stop_final, checkpoint_final],
        class_weight=class_weights_dict_full,
        verbose=1
    )

    # Save the final model
    final_model.save('ecg/final_model.keras')
    logger.info("Final model trained on the entire dataset and saved as 'ecg/final_model.keras'.")
