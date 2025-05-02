#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_classifier.py

Обучение RandomForest на ваших TIFF-патчах с березами (pos/) и без (neg/).
"""

import csv
import os
import rasterio
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
from sklearn.model_selection import train_test_split
import joblib

# --- Параметры ---
LABELS_CSV = "labels.csv"   # ваш файл
TEST_SIZE  = 0.2            # доля теста
VAL_SIZE   = 0.25           # доля валидации от оставшихся после test_split
RANDOM_SEED= 42
N_TREES    = 150

# --- Функция для извлечения признаков из TIFF ---
def extract_features(tif_path):
    """
    Считает средние значения каналов R, G, B.
    Возвращает список из трёх признаков.
    """
    with rasterio.open(tif_path) as src:
        img = src.read().astype(float)  # (bands, H, W)
    # Предполагаем порядок каналов [R, G, B, ...]
    R, G, B = img[0], img[1], img[2]
    feats = [
        R.mean(), G.mean(), B.mean()
    ]
    return feats

# --- Main ---
def main():
    # 1. Читаем CSV
    df = pd.read_csv(LABELS_CSV)
    filepaths = df['filepath'].tolist()
    labels    = df['label'].astype(int).tolist()

    # 2. Сплит: сначала test, потом val
    X_temp, X_test, y_temp, y_test = train_test_split(
        filepaths, labels,
        test_size=TEST_SIZE,
        stratify=labels,
        random_state=RANDOM_SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=VAL_SIZE,
        stratify=y_temp,
        random_state=RANDOM_SEED
    )

    # 3. Извлечение признаков
    print("Extracting features...")
    X_train_feats = [extract_features(p) for p in X_train]
    X_val_feats   = [extract_features(p) for p in X_val]
    X_test_feats  = [extract_features(p) for p in X_test]

    # 4. Обучение RandomForest
    print("Training RandomForest...")
    model = RandomForestClassifier(
        n_estimators=N_TREES,
        random_state=RANDOM_SEED
    )
    model.fit(X_train_feats, y_train)

    # 5. Оценка на валидации
    y_pred_val = model.predict(X_val_feats)
    print("\nValidation metrics:")
    print(classification_report(y_val, y_pred_val))
    print("Confusion Matrix (val):")
    print(confusion_matrix(y_val, y_pred_val))
    print("Cohen's kappa:", cohen_kappa_score(y_val, y_pred_val))

    # 6. Оценка на тесте
    y_pred = model.predict(X_test_feats)
    print("\nTest metrics:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix (test):")
    print(confusion_matrix(y_test, y_pred))
    print("Cohen's kappa:", cohen_kappa_score(y_test, y_pred))

    # 7. Сохраняем модель (опционально)
    joblib.dump(model, "rf_birch_model.pkl")
    print("Model saved to rf_birch_model.pkl")

if __name__ == "__main__":
    main()
