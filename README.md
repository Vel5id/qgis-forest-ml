Overview
This repository contains the full implementation of the methods described in the manuscript:

"Automated Detection of Birch Seedlings on Orthophotos Using Random Forest with Texture-Based Feature Engineering"
Vladimir Fominov et al., 2025 [add more authors]

The project provides reproducible code for training, validation, and evaluation of Random Forest classifiers applied to UAV-derived RGB orthomosaics for seedling density estimation and vegetation mapping in post-fire boreal forests. Both the basic (spectral/morphological) and advanced (texture-based) models are included.

Features
Data preprocessing and augmentation (spatial transforms, noise, brightness, and color jitter)
Feature extraction:
Base Model: Mean RGB, Excess Green (ExG), Excess Red (ExR), ExG–ExR
Modified Model: GLCM (gray-level co-occurrence matrix) statistics and Local Binary Patterns (LBP)
Random Forest model training with flexible hyperparameters (class weights, number of trees, etc.)
Out-of-bag (OOB) error analysis and hyperparameter optimization
Evaluation scripts for comparison with manual ground-truthing and visualization (heatmaps, confusion matrices, detection counts)
Jupyter notebooks and Python scripts for reproducible experiments

Getting Started:
Requirements:
Python 3.10+
numpy, pandas, scikit-learn, scikit-image, rasterio, tqdm, matplotlib, etc.

Install dependencies with:
pip install -r requirements.txt
Data
Orthomosaic images and manual annotations (polygon masks) are required.

Example data and pre-trained models are available at Kaggle Collection.
https://www.kaggle.com/work/collections/15977795

Repository Contents
train_multiclass_base.py
Training of the base Random Forest model using spectral features (RGB and vegetation indices).
Output: model, quality metrics, OOB-error visualization.

train_multiclass_modified.py
Training of the modified Random Forest model with an extended set of features (GLCM, LBP).
Output: model, quality comparison, analysis of the influence of the number of trees.

labels_fo_augmented.py
Script for generating a CSV table with class labels for all images required for model training and validation.

augment_images.py
Image augmentation: spatial transformations, noise addition, brightness/color modification. Used to expand the training dataset.

full_script_base.py
A complete script that runs the pipeline: orthophoto processing, prediction with the base model, creation of a heatmap, and exporting results to Excel.

full_script_modified.py
A script similar to the previous one, but uses the modified model and extended feature set.



Citation
If you use this code or data in your research, please cite:
Fominov V., et al. (2025) [add more authors]. Automated Detection of Birch Seedlings on Orthophotos Using Random Forest with Texture-Based Feature Engineering. [Journal, under review].
License

This project is licensed under the MIT License.
See the LICENSE file for details.

Contact
For questions, suggestions, or contributions, please open an issue or contact the author at:
Vladimir Fominov — GitHub Profile, [add more]
