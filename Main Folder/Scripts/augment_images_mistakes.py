import os
import numpy as np
import rasterio

# === User-defined parameters ===
INPUT_DIR = "../Images/new_ver/augmented/birch_only_inner_augmented"       
OUTPUT_DIR = "../Images/new_ver/augmented/second_birch_only_inner_augmented"
N_AUGMENT = 3    # number of augmentations per original

# Augmentation functions
def brightness_jitter(img):
    factor = np.random.uniform(0.8, 1.2)
    out = np.clip(img.astype(float) * factor, 0, np.iinfo(img.dtype).max)
    return out.astype(img.dtype)

def channel_jitter(img):
    out = img.astype(float)
    out[1] = np.clip(out[1] * np.random.uniform(0.9, 1.1), 0, np.iinfo(img.dtype).max)
    return out.astype(img.dtype)

def gaussian_noise(img):
    out = img.astype(float)
    noise = np.random.normal(loc=0.0, scale=2.0, size=out.shape)
    out = np.clip(out + noise, 0, np.iinfo(img.dtype).max)
    return out.astype(img.dtype)

def flip_h(img):
    return np.flip(img, axis=2)

AUG_FUNCS = [
    ("flip_h",  flip_h),
    ("bright",  brightness_jitter),
    ("color",   channel_jitter),
    ("noise",   gaussian_noise),
]

def augment_and_save(src_path, dst_dir, n_augment=N_AUGMENT):
    basename = os.path.splitext(os.path.basename(src_path))[0]
    with rasterio.open(src_path) as src:
        profile = src.profile
        img = src.read()
    # save original
    out_path = os.path.join(dst_dir, f"{basename}_orig.tif")
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(img)
    # select random augmentations (without replacement)
    picks = np.random.choice(len(AUG_FUNCS), size=n_augment, replace=False)
    for idx in picks:
        suffix, fn = AUG_FUNCS[idx]
        aug = fn(img)
        out_name = f"{basename}_{suffix}.tif"
        out_file = os.path.join(dst_dir, out_name)
        with rasterio.open(out_file, 'w', **profile) as dst:
            dst.write(aug)

def main():
    if not os.path.isdir(INPUT_DIR):
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for root, _, files in os.walk(INPUT_DIR):
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
