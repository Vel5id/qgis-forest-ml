#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import rasterio
import joblib
from joblib import Parallel, delayed, Memory
from pathlib import Path
from sklearn.base import clone
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    cohen_kappa_score,
    roc_auc_score,
    precision_recall_curve
)
from tqdm import tqdm

# новые импорты для текстурных признаков
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

# --- Параметры ------------------------------------------------------------
LABELS_CSV   = "labels_multiclass_all.csv"
BASE_DIR     = Path("../Images/new_ver")
RANDOM_SEED  = 42
N_TREES      = 300

# число уровней квантования для GLCM и LBP
LEVELS = 8
# -------------------------------------------------------------------------
memory = Memory(location="cache_dir", verbose=0)

def extract_features(path: str) -> list[float]:
    """Извлекает цветовые, Haralick (GLCM) и LBP-признаки из 3-канального TIFF."""
    # читаем патч
    with rasterio.open(path) as src:
        img = src.read().astype(float)  # shape = (3, H, W)
    R, G, B = img[0], img[1], img[2]

    # 1) Цветовые признаки
    mR, mG, mB = R.mean(), G.mean(), B.mean()
    exg  = (2 * G - R - B).mean()
    exr  = (1.4 * R - G).mean()
    exgr = exg - exr
    feats = [mR, mG, mB, exg, exr, exgr]

    # 2) Haralick (GLCM) по G-каналу
    # квантование до 0..LEVELS-1
    patchG = (G * (LEVELS - 1) / (G.max() + 1e-6)).astype(np.uint8)
    glcm = graycomatrix(
        patchG,
        distances=[1, 5],
        angles=[0, np.pi/2],
        levels=LEVELS,
        symmetric=True,
        normed=True
    )
    # усредняем по всем дистанциям и углам
    for prop in ('contrast', 'correlation'):
        feats.append(graycoprops(glcm, prop).mean())

    # 3) LBP-гистограмма (uniform, P=8, R=1)
    lbp = local_binary_pattern(patchG, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)
    feats.extend(hist.tolist())

    return feats



# При необходимости, обновите список имён признаков для вывода:
feature_names = [
    "mR","mG","mB","ExG","ExR","ExG-ExR",
    "haralick_contrast","haralick_correlation",
] + [f"lbp_{i}" for i in range(10)]

extract_features_cached = memory.cache(extract_features)

def build_dataset(df_split: pd.DataFrame, n_jobs: int = -1, use_cache: bool = True):
    """
    Параллельно строит X и y.
    - n_jobs: число процессов (используйте -1, чтобы взять все ядра).
    - use_cache: держать ли кэш (True) или вызывать оригинальную функцию.
    """
    func = extract_features_cached if use_cache else extract_features

    paths = [str(BASE_DIR / p) for p in df_split.filepath]
    # распараллеливаем: каждый процесс берёт свой набор путей
    X = Parallel(
        n_jobs=n_jobs,
        backend="loky",      # можно заменить на "threading" для IO-bound
        verbose=5
    )(delayed(func)(path) for path in paths)

    y = df_split["class"].astype(int).tolist()
    return np.array(X), y


def get_original_ids(relpaths):
    """
    Из списка относительных путей возвращает set of (class_name, patch_id),
    где patch_id — имя исходного патча до аугментации.
    """
    ids = set()
    for rp in relpaths:
        parts = Path(rp).parts
        if len(parts) < 3:
            continue
        class_aug = parts[1]
        class_name = class_aug.removesuffix("_augmented")
        filename = parts[2]
        stem = Path(filename).stem
        patch_id = stem.rsplit("_", 1)[0]
        ids.add((class_name, patch_id))
    return ids


def main() -> None:
    # 1) Читаем CSV
    df = pd.read_csv(LABELS_CSV)

    # 2) Жёсткое разделение по префиксу пути
    df_train = df[df.filepath.str.startswith("augmented_train/")].reset_index(drop=True)
    df_val   = df[df.filepath.str.startswith("augmented_validation/")].reset_index(drop=True)
    df_test  = df[df.filepath.str.startswith("augmented_test/")].reset_index(drop=True)

    # 2.5) Smoke-tests на data-leak
    ids_train = get_original_ids(df_train.filepath)
    ids_val   = get_original_ids(df_val.filepath)
    ids_test  = get_original_ids(df_test.filepath)
    assert ids_train.isdisjoint(ids_val),  f"LEAK between TRAIN/VAL: {ids_train & ids_val}"
    assert ids_train.isdisjoint(ids_test), f"LEAK between TRAIN/TEST: {ids_train & ids_test}"
    assert ids_val.isdisjoint(ids_test),   f"LEAK between VAL/TEST:   {ids_val & ids_test}"
    print("✅ No data leakage detected between splits.\n")

    print(f"Train samples: {len(df_train)}, Validation: {len(df_val)}, Test: {len(df_test)}")

    # 3) Строим фичи
    print("Постройка фич train")
    X_train_f, y_train = build_dataset(df_train, n_jobs=-1, use_cache=True)
    print("Постройка фич val")
    X_val_f,   y_val   = build_dataset(df_val,   n_jobs=-1, use_cache=True)
    print("Постройка фич test")
    X_test_f,  y_test  = build_dataset(df_test,  n_jobs=-1, use_cache=True)

    # 4) Manual class weights (no automatic balancing)
    classes = np.unique(y_train)
    # задаём всем классам базовый вес = 1.0
    class_w = {c: 1.0 for c in classes}

    # усиливаем берёзу (label=1) в 5 раз
    boost_factor = 0.85
    if 1 in class_w:
        class_w[1] *= boost_factor

    print("Using manual class weights:", class_w, "\n")

    # 5) Обучаем RandomForest с прогресс-баром
    print("Training RandomForest with progress bar:")
    clf = RandomForestClassifier(
        n_estimators=1,
        warm_start=True,
        class_weight=class_w,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        oob_score=True
    )
    for i in tqdm(range(N_TREES), desc="Building trees", unit="tree"):
        clf.n_estimators = i + 1
        clf.fit(X_train_f, y_train)

    # 5.1) Собираем OOB-ошибки **на клоне**, чтобы не ломать `clf`
    oob_errors = []
    for n in [1, 5, 10, 25, 50, 100, 150, 200, 300]:
        temp = clone(clf)  # берём тот же набор параметров
        temp.set_params(n_estimators=n, warm_start=False)  # fresh forest
        temp.fit(X_train_f, y_train)
        oob_errors.append((n, 1 - temp.oob_score_))
    print("OOB errors by n_estimators:", oob_errors)


    # 6) Опциональная кросс-валидация на train+validation
    X_tv_f = np.vstack([X_train_f, X_val_f])
    y_tv   = y_train + y_val
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = cross_val_score(
        clf, X_tv_f, y_tv,
        cv=skf, scoring="f1_macro", n_jobs=-1
    )
    print(f"\n5-fold CV Macro-F1 on train+val: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # 7.1) Найдём порог для класса 1 на валидации
    # бинаризуем y_val: 1 – берёза, 0 – всё остальное
    y_val_bin = np.array(y_val) == 1

    # вероятности «берёзы» на валидации
    proba_val = clf.predict_proba(X_val_f)[:, 0]  # столбец 0 = класс 1

    # считаем precision, recall и пороги
    prec, rec, thresh = precision_recall_curve(y_val_bin, proba_val)

    # высчитываем F1 для каждого порога
    f1_scores = 2 * (prec * rec) / (prec + rec + 1e-12)

    # выбираем индекс максимального F1 (пропускаем последний элемент prec/reс без порога)
    best_idx = np.argmax(f1_scores[:-1])
    best_thresh = thresh[best_idx]
    print(f"Best threshold for class 1: {best_thresh:.3f} → "
        f"Precision={prec[best_idx]:.3f}, Recall={rec[best_idx]:.3f}, F1={f1_scores[best_idx]:.3f}")

    # 8) Оценка на тесте
    print("\n=== Test ===")
    yt_pred = clf.predict(X_test_f)
    print(classification_report(y_test, yt_pred, digits=3))
    print("Confusion Matrix (test):\n", confusion_matrix(y_test, yt_pred))
    print("Balanced accuracy (test):", balanced_accuracy_score(y_test, yt_pred))
    print("Cohen's kappa (test):", cohen_kappa_score(y_test, yt_pred))

    # 8.1) Прогноз с порогом для класса 1
    proba_test = clf.predict_proba(X_test_f)

    def predict_with_threshold(proba, t):
        y_pred = []
        for p in proba:
            # если вероятность класса 1 >= t, то метка = 1
            if p[0] >= t:
                y_pred.append(1)
            else:
                # иначе выбираем класс с наибольшей вероятностью среди остальных
                other = np.argmax(p[1:]) + 2  # сдвиг, т.к. p[1:] → классы 2..6
                y_pred.append(other)
        return y_pred

    y_test_adj = predict_with_threshold(proba_test, best_thresh)

    # далее используем y_test_adj вместо clf.predict(X_test_f)
    print(classification_report(y_test, y_test_adj, digits=3))

    # 9) ROC-AUC OvR
    proba = clf.predict_proba(X_test_f)
    y_onehot = np.zeros((len(y_test), clf.n_classes_))
    for i, lab in enumerate(y_test):
        y_onehot[i, lab-1] = 1
    auc = roc_auc_score(y_onehot, proba, multi_class="ovr")
    print(f"ROC-AUC (ovr) on test: {auc:.3f}")

    # 10) Важность признаков
    print("\nFeature importances:")
    feature_names = [
    "mR","mG","mB","ExG","ExR","ExG-ExR",
    "haralick_contrast","haralick_correlation"
] + [f"lbp_{i}" for i in range(10)]
    for name, imp in sorted(zip(feature_names, clf.feature_importances_), key=lambda x: -x[1]):
        print(f"  {name}: {imp:.3f}")

    # 11) Сохраняем модель
    out_model = "multiclass_main_without_data_leak_0.85weight_birch_300_trees_modified_8levels.pkl"
    joblib.dump(clf, out_model)
    print(f"\nModel saved as {out_model}")

if __name__ == "__main__":
    main()
