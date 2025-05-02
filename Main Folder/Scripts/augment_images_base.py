# Updated augmentation script: specify input and output directories inside the script

import os
import numpy as np
import rasterio

# === User-defined parameters ===
# Set your source and target folders here:
INPUT_DIR = "../Images/new_ver/base/mistake"       # folder with original patches
OUTPUT_DIR = "../Images/new_ver/augmented/mistake_augmented"  # folder where augmented patches will be saved

# === End of user parameters ===

def augment_and_save(src_path, dst_dir):
    """Read TIFF, create augmentations, save to dst_dir."""
    basename = os.path.splitext(os.path.basename(src_path))[0]
    with rasterio.open(src_path) as src:
        profile = src.profile
        img = src.read()  # (bands, height, width)

    transforms = [
        ("orig",   lambda x: x),
        ("flip_h", lambda x: np.flip(x, axis=2)),
        ("flip_v", lambda x: np.flip(x, axis=1)),
        ("rot90",  lambda x: np.flip(np.transpose(x, (0,2,1)), axis=2)),
        ("rot180", lambda x: np.flip(np.flip(x, axis=1), axis=2)),
        ("rot270", lambda x: np.flip(np.transpose(x, (0,2,1)), axis=1)),
    ]

    for suffix, fn in transforms:
        out_img = fn(img)
        out_name = f"{basename}_{suffix}.tif"
        out_path = os.path.join(dst_dir, out_name)
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(out_img)


def main():
    # Check directories
    if not os.path.isdir(INPUT_DIR):
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Walk through input_dir for TIFF files
    for root, _, files in os.walk(INPUT_DIR):
        # preserve subdirectory structure if needed
        rel = os.path.relpath(root, INPUT_DIR)
        dst_base = os.path.join(OUTPUT_DIR, rel) if rel != "." else OUTPUT_DIR
        os.makedirs(dst_base, exist_ok=True)

        for fname in files:
            if fname.lower().endswith(".tif"):
                src_path = os.path.join(root, fname)
                augment_and_save(src_path, dst_base)
                print(f"Augmented {src_path} -> {dst_base}")


if __name__ == "__main__":
    main()

# Instructions:
# - Edit the INPUT_DIR and OUTPUT_DIR variables at the top to point to your folders.
# - Run the script with: python augment_images.py
