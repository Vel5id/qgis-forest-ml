#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_next.py

Добавляет часть 'hard' патчей в тренировочный набор и переобучает модель.
Затем проверяет её на оставшихся 'hard' примерах.
"""

import pandas as pd
import joblib
import rasterio
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score

# --- Параметры ---
LABELS_CSV     = "labels_aug.csv"         # исходный набор
HARD_LABELS_CSV= "hard_labels.csv"    # hard-набор (оба класса)
TEST_SIZE_HARD = 0.5                  # доля hard, отложенная в финальный тест
VAL_SIZE       = 0.2                  # доля валидации от trainval
RANDOM_SEED    = 42
N_TREES        = 150

# --- Функция извлечения признаков ---
def extract_features(path):
    with rasterio.open(path) as src:
        img = src.read().astype(float)
    R, G, B = img[0], img[1], img[2]
    return [R.mean(), G.mean(), B.mean()]

def main():
    # 1. Загрузить метки
    df_orig = pd.read_csv(LABELS_CSV)
    df_hard = pd.read_csv(HARD_LABELS_CSV)

    # 2. Разделить hard на дообучение и финальный тест
    hard_train, hard_test = train_test_split(
        df_hard, test_size=TEST_SIZE_HARD,
        stratify=df_hard.label, random_state=RANDOM_SEED
    )

    # 3. Объединить исходный и hard_train для trainval
    df_trainval = pd.concat([df_orig, hard_train], ignore_index=True)

    # 4. Train/Val split
    train_df, val_df = train_test_split(
        df_trainval, test_size=VAL_SIZE,
        stratify=df_trainval.label, random_state=RANDOM_SEED
    )

    # 5. Подготовить списки путей и меток
    X_train_paths = train_df.filepath.tolist(); y_train = train_df.label.tolist()
    X_val_paths   = val_df.filepath.tolist();   y_val   = val_df.label.tolist()
    X_test_paths  = hard_test.filepath.tolist(); y_test  = hard_test.label.tolist()

    # 6. Извлечь признаки
    print("Extracting features for train ...")
    X_train_feats = [extract_features(p) for p in X_train_paths]
    print("Extracting features for val ...")
    X_val_feats   = [extract_features(p) for p in X_val_paths]
    print("Extracting features for hard-test ...")
    X_test_feats  = [extract_features(p) for p in X_test_paths]

    # 7. Обучение RandomForest
    print("Training RandomForest...")
    model = RandomForestClassifier(n_estimators=N_TREES, random_state=RANDOM_SEED, class_weight="balanced")
    model.fit(X_train_feats, y_train)

    # 8. Оценка на валидации
    print("\nValidation results:")
    val_pred = model.predict(X_val_feats)
    print(classification_report(y_val, val_pred))
    print("Confusion Matrix (val):")
    print(confusion_matrix(y_val, val_pred))
    print("Cohen's kappa (val):", cohen_kappa_score(y_val, val_pred))

    # 9. Оценка на hard_test
    print("\nHard-test results:")
    test_pred = model.predict(X_test_feats)
    print(classification_report(y_test, test_pred))
    print("Confusion Matrix (hard-test):")
    print(confusion_matrix(y_test, test_pred))
    print("Cohen's kappa (hard-test):", cohen_kappa_score(y_test, test_pred))

    # 10. Сохранить модель
    joblib.dump(model, "rf_birch_model_retrained3.pkl")
    print("\nModel retrained and saved as rf_birch_model_retrained3.pkl")

if __name__ == "__main__":
    main()
