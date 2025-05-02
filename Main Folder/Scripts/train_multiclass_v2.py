#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import rasterio
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# For texture features
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

# --- Parameters ---
LABELS_CSV  = "labels_multiclass_all_aug.csv"
TEST_SIZE   = 0.2        # test split size
VAL_SIZE    = 0.2        # validation split from remaining data
RANDOM_SEED = 42
N_TREES     = 300

LEVELS = 64
# ------------------

def extract_features(path):
    """Extract color + Haralick + LBP features from a 3-channel TIFF."""
    with rasterio.open(path) as src:
        img = src.read().astype(float)  # shape (bands, H, W)
    R, G, B = img[0], img[1], img[2]

    # 1) Color features
    mR, mG, mB = R.mean(), G.mean(), B.mean()
    exg  = (2 * G - R - B).mean()
    exr  = (1.4 * R - G).mean()
    exgr = exg - exr
    feats = [mR, mG, mB, exg, exr, exgr]

    # 2) Haralick (GLCM) on G channel
    patchG = (G * (LEVELS - 1) / (G.max() + 1e-6)).astype(np.uint8)
    glcm = graycomatrix(
        patchG,
        distances=[1, 5],
        angles=[0, np.pi / 2],
        levels=LEVELS,
        symmetric=True,
        normed=True
    )
    # Average over all distances and angles
    for prop in ['contrast', 'correlation']:
        feats.append(graycoprops(glcm, prop).mean())

    # 3) LBP histogram (uniform LBP, P=8, R=1)
    lbp = local_binary_pattern(patchG, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)
    feats.extend(hist.tolist())

    return feats


def main():
    # 1) Load CSV with filepath, class, weight
    df = pd.read_csv(LABELS_CSV)
    X = df['filepath'].tolist()
    y = df['class'].tolist()
    w = df['weight'].tolist()

    # 2) Train/test split (preserve weights)
    X_tv, X_test, y_tv, y_test, w_tv, w_test = train_test_split(
        X, y, w,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_SEED
    )

    # 3) Train/val split
    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
        X_tv, y_tv, w_tv,
        test_size=VAL_SIZE,
        stratify=y_tv,
        random_state=RANDOM_SEED
    )

    # 4) Feature extraction
    def build_feats(paths):
        return np.array([extract_features(p) for p in paths])

    print("Extracting train features...")
    X_train_f = build_feats(X_train)
    print("Extracting val features...")
    X_val_f   = build_feats(X_val)
    print("Extracting test features...")
    X_test_f  = build_feats(X_test)

    # 5) Train Random Forest using sample weights
    clf = RandomForestClassifier(
        n_estimators=N_TREES,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    clf.fit(X_train_f, y_train, sample_weight=w_train)

    # 6) Evaluate on validation set
    print("\n=== Validation ===")
    yv_pred = clf.predict(X_val_f)
    print(classification_report(y_val, yv_pred, digits=3))
    print("Confusion Matrix (val):\n", confusion_matrix(y_val, yv_pred))

    # 7) Evaluate on test set
    print("\n=== Test ===")
    yt_pred = clf.predict(X_test_f)
    print(classification_report(y_test, yt_pred, digits=3))
    print("Confusion Matrix (test):\n", confusion_matrix(y_test, yt_pred))

    # 8) Save model
    joblib.dump(clf, "rf_multiclass_all_rgb_v2_texture.pkl")
    print("\nModel saved as rf_multiclass_all_rgb_v2_texture.pkl")

if __name__ == "__main__":
    main()
