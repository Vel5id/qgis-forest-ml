#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import rasterio
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    cohen_kappa_score,
    roc_auc_score
)

# --- Parameters ---
LABELS_CSV  = "labels_multiclass_all_aug.csv"
TEST_SIZE   = 0.2        # test split
VAL_SIZE    = 0.2        # validation split from the remaining data
RANDOM_SEED = 42
N_TREES     = 400
# ------------------

def extract_features(path):
    # Extract features from a single .tif file
    with rasterio.open(path) as src:
        img = src.read().astype(float)
    R, G, B = img[0], img[1], img[2]
    mR, mG, mB = R.mean(), G.mean(), B.mean()
    exg  = (2 * G - R - B).mean()
    exr  = (1.4 * R - G).mean()
    exgr = exg - exr
    return [mR, mG, mB, exg, exr, exgr]

def main():
    # 1) Read CSV
    df = pd.read_csv(LABELS_CSV)
    X = df.filepath.tolist()
    y = df["class"].tolist()

    # 2) Train/test split
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=TEST_SIZE,
        stratify=y, random_state=RANDOM_SEED
    )

    # 3) Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=VAL_SIZE,
        stratify=y_tv, random_state=RANDOM_SEED
    )

    # 4) Feature extraction
    def build_feats(lst): return np.array([extract_features(p) for p in lst])
    X_train_f = build_feats(X_train)
    X_val_f   = build_feats(X_val)
    X_test_f  = build_feats(X_test)

    # 5) Initialize and train model
    clf = RandomForestClassifier(
        n_estimators=N_TREES,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        oob_score=True
    )
    clf.fit(X_train_f, y_train)

    # 5.1) Cross-validation on train+val
    X_tv_f = np.vstack([X_train_f, X_val_f])
    y_tv_all = y_train + y_val
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = cross_val_score(
        clf, X_tv_f, y_tv_all,
        cv=skf, scoring="f1_macro", n_jobs=-1
    )
    print(f"\n5-fold CV Macro-F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # 6) Validation
    print("\n=== Validation ===")
    yv_pred = clf.predict(X_val_f)
    print(classification_report(y_val, yv_pred, digits=3))
    print("Confusion Matrix (val):\n", confusion_matrix(y_val, yv_pred))
    print("Balanced accuracy (val):", balanced_accuracy_score(y_val, yv_pred))
    print("Cohen's kappa (val):", cohen_kappa_score(y_val, yv_pred))

    # 7) Test
    print("\n=== Test ===")
    yt_pred = clf.predict(X_test_f)
    print(classification_report(y_test, yt_pred, digits=3))
    print("Confusion Matrix (test):\n", confusion_matrix(y_test, yt_pred))
    print("Balanced accuracy (test):", balanced_accuracy_score(y_test, yt_pred))
    print("Cohen's kappa (test):", cohen_kappa_score(y_test, yt_pred))

    # 8) ROC-AUC One-vs-Rest (multiclass)
    proba = clf.predict_proba(X_test_f)
    y_onehot = np.zeros((len(y_test), clf.n_classes_))
    for i, label in enumerate(y_test):
        y_onehot[i, label - 1] = 1  # assuming labels start from 1
    auc = roc_auc_score(y_onehot, proba, multi_class="ovr")
    print(f"ROC-AUC (ovr) on test: {auc:.3f}")

    # 9) Feature importances
    print("\nFeature importances:")
    for name, imp in sorted(
        zip(
            ["mR", "mG", "mB", "ExG", "ExR", "ExG-ExR"],
            clf.feature_importances_[:6]
        ),
        key=lambda x: -x[1]
    ):
        print(f"  {name}: {imp:.3f}")

    # 10) Save model
    joblib.dump(clf, "rf_multiclass_all_rgb_v4_second_birch_only_inner_augmented.pkl")
    print("\nModel saved as rf_multiclass_all_rgb_v4_second_birch_only_inner_augmented.pkl")

if __name__ == "__main__":
    main()
