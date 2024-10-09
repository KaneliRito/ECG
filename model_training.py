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
from logger_setup import setup_logger

logger = logging.getLogger(__name__)

# Directory paths
ROC_DIR = 'roc_curves'

def create_dirs(*dirs):
    """Creates directories if they do not exist."""
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)

def create_model(input_shape=(100, 2)):
    """Creates and compiles an LSTM-based model."""
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
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def log_metrics(fold_num, metrics):
    """Logs evaluation metrics."""
    f1, precision_val, recall_val, auroc_val, auprc_val = metrics
    logger.info(f"Fold {fold_num} - F1 Score: {f1}")
    logger.info(f"Fold {fold_num} - Precision: {precision_val}")
    logger.info(f"Fold {fold_num} - Recall: {recall_val}")
    logger.info(f"Fold {fold_num} - AUROC: {auroc_val}")
    logger.info(f"Fold {fold_num} - AUPRC: {auprc_val}")

def compute_class_weights(y):
    """Computes class weights."""
    class_weights_values = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
    return {0: class_weights_values[0], 1: class_weights_values[1]}

def compute_roc_curve(y_val, y_pred_prob, fold_num):
    """Computes and saves ROC curve."""
    fpr, tpr, _ = roc_curve(y_val, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - Fold {fold_num}')
    plt.legend(loc="lower right")
    roc_path = os.path.join(ROC_DIR, f'roc_fold{fold_num}.png')
    plt.savefig(roc_path)
    plt.close()
    logger.info(f"Fold {fold_num} - ROC-curve opgeslagen als {roc_path}.")
    return fpr, tpr, roc_auc

def perform_cross_validation(X, y, n_splits=5):
    """Performs Stratified K-Fold Cross-Validation."""
    logger.info("Starting Stratified K-Fold Cross-Validation.")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    class_weights_dict_full = compute_class_weights(y)
    logger.info(f"Class weights: {class_weights_dict_full}")

    create_dirs(ROC_DIR)

    fold_accuracies, fold_f1_scores, fold_precision, fold_recall, fold_auroc, fold_auprc = [], [], [], [], [], []
    mean_fpr = np.linspace(0, 1, 100)
    tprs, aucs = [], []

    for fold_num, (train_index, val_index) in enumerate(tqdm(skf.split(X, y), total=n_splits, desc="Stratified K-Fold")):
        logger.info(f"--- Starting Fold {fold_num + 1} ---")
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        class_weights_dict = compute_class_weights(y_train)

        model = create_model()
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint(f'ecg/best_model_fold{fold_num + 1}.keras', monitor='val_accuracy', save_best_only=True, mode='max')
        ]

        history = model.fit(
            X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val),
            callbacks=callbacks, class_weight=class_weights_dict, verbose=0
        )

        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        fold_accuracies.append(val_acc)

        y_pred_prob = model.predict(X_val).flatten()
        y_pred = (y_pred_prob > 0.5).astype(int)

        metrics = [
            f1_score(y_val, y_pred),
            precision_score(y_val, y_pred),
            recall_score(y_val, y_pred),
            roc_auc_score(y_val, y_pred_prob),
            average_precision_score(y_val, y_pred_prob)
        ]

        log_metrics(fold_num + 1, metrics)

        fold_f1_scores.append(metrics[0])
        fold_precision.append(metrics[1])
        fold_recall.append(metrics[2])
        fold_auroc.append(metrics[3])
        fold_auprc.append(metrics[4])

        fpr, tpr, roc_auc = compute_roc_curve(y_val, y_pred_prob, fold_num + 1)
        aucs.append(roc_auc)
        tprs.append(np.interp(mean_fpr, fpr, tpr))

    logger.info("Cross-validation completed.")
    # (Continue with Mean ROC Curve and Summary like before)

def train_final_model(X, y):
    """Trains the final model on the entire dataset."""
    logger.info("Training final model on the entire dataset.")

    final_model = create_model()
    class_weights_dict_full = compute_class_weights(y)

    callbacks = [
        EarlyStopping(monitor='loss', patience=10, restore_best_weights=True),
        ModelCheckpoint('ecg/final_model.keras', monitor='accuracy', save_best_only=True, mode='max')
    ]

    history = final_model.fit(X, y, epochs=100, batch_size=32, callbacks=callbacks, class_weight=class_weights_dict_full, verbose=0)

    loss, acc = final_model.evaluate(X, y, verbose=0)
    logger.info(f"Final Model - Accuracy: {acc}")

    y_pred_prob = final_model.predict(X).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)

    metrics = [
        f1_score(y, y_pred),
        precision_score(y, y_pred),
        recall_score(y, y_pred),
        roc_auc_score(y, y_pred_prob),
        average_precision_score(y, y_pred_prob)
    ]

    log_metrics('final', metrics)

    compute_roc_curve(y, y_pred_prob, 'final')
    final_model.save('ecg/final_model.keras')
    logger.info("Final model saved as 'ecg/final_model.keras'.")
