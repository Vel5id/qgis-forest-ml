#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import joblib
from train_classifier import extract_features  # импорт вашей функции

from sklearn.metrics import classification_report, confusion_matrix

# 1. Загрузить hard-метки
hard_df = pd.read_csv("hard_labels.csv")
hard_paths = hard_df.filepath.tolist()
hard_labels= hard_df.label.tolist()

# 2. Загрузить сохранённую модель
model = joblib.load("rf_birch_model.pkl")

# 3. Извлечь признаки так же, как при обучении
hard_feats = [extract_features(p) for p in hard_paths]

# 4. Прогноз и отчёт
hard_pred = model.predict(hard_feats)
print("=== Hard patches test ===")
print(classification_report(hard_labels, hard_pred))
print("Confusion Matrix (hard):")
print(confusion_matrix(hard_labels, hard_pred))
