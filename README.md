qgis-forest-ml

Automated pipeline for detecting and counting birch seedlings in high-resolution forest mosaics.

📖 Overview

This repository provides scripts to:

Train Random Forest models on multispectral TIFF datasets (RGB, textures, etc.) for multi-class and binary classification.

Predict classes on large mosaic TIFFs by processing them in patches (64×64 px), with optional CPU thermal management.

Generate GeoTIFF outputs: full class maps and binary birch masks.

Count connected clusters of birch seedlings and export summary reports in Excel.

🚀 Features

Patch-based inference: splits large images into configurable patches and strides.

Parallel processing: utilizes ~80% of CPU cores via ProcessPoolExecutor.

Thermal throttling: monitors CPU temperature and automatically pauses if overheating.

Multi-class & Binary modes:

Multiclass: detect birch vs. other landcover types.

Binary: detect birch (1) vs. non‑birch (2).

Cluster analysis: downsample binary mask to patch-grid, label connected components, count single vs multi-patch clusters.

Error estimation: leverage out‑of‑bag (OOB) score for model error probability.

Excel reporting: save cluster counts and error probabilities to .xlsx.

📦 Requirements

Python 3.8+

rasterio

numpy

pandas

scipy

scikit-learn

joblib

psutil

tqdm

skimage (for texture-based training)

Install via:

pip install rasterio numpy pandas scipy scikit-learn joblib psutil tqdm scikit-image

📂 Repository Structure

qgis-forest-ml/
├── README.md
├── full_script.py          # Multiclass inference + cluster counting pipeline
├── binary_birch_script.py  # Binary (birch vs non-birch) inference pipeline
├── train_rgb.py            # Train RF on RGB features (multiclass)
├── train_texture.py        # Train RF on RGB+texture features
├── labels_multiclass_all_aug.csv  # Example labels CSV for training
└── models/
    ├── rf_multiclass_...pkl
    └── rf_binary_birch_...pkl

⚙️ Usage

1. Inference (Multiclass)

Place your mosaic TIFF at mosaic_path inside full_script.py.

Ensure model_path points to your multiclass RF model .pkl.

Adjust parameters (patch_size, stride, block_size, thermal thresholds).

Run:

python full_script.py

Outputs:

class_map.tif — full-class raster (uint8 labels).

birch_mask.tif — binary birch mask (1/0).

birch_report.xlsx — cluster counts & error probability.

2. Inference (Binary Birch vs Non-Birch)

Configure paths in binary_birch_script.py (mosaic_path, model_path).

Adjust parameters as needed.

Run:

python binary_birch_script.py

Outputs:

class_map_binary.tif — binary-class raster (1: birch, 2: non-birch).

birch_mask_binary.tif — birch mask (1/0).

3. Training Scripts

train_rgb.py: train a RandomForest on color features.

train_texture.py: train a RandomForest with color + Haralick + LBP features.

Usage example:

python train_rgb.py
python train_texture.py

Ensure your labels_multiclass_all_aug.csv is populated with filepath,class,weight.

📋 Configuration

Patch size: patch_size (default: 64)

Stride: stride (default: 64)

Block size: block_size (default: 100)

CPU throttle: temp_threshold, cool_factor

Edit these parameters at the top of each script.

💡 Recommendations

Use Git LFS for large .tif and .pkl files.

Store raw imagery and models in GitHub Releases if >100 MB.

Reference outputs in QGIS by adding the GeoTIFFs as layers.

🤝 Contributing

Fork the repository

Create a feature branch

Commit changes

Push and open a Pull Request

Please follow PEP8 and include tests when applicable.
