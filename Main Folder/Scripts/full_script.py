import os
import time
import psutil
import joblib
import rasterio
import numpy as np
import pandas as pd
from scipy.ndimage import label
from concurrent.futures import ProcessPoolExecutor, as_completed
from rasterio.windows import Window
from tqdm import tqdm
from multiprocessing import freeze_support

# === Parameters ===
mosaic_path      = r"C:\Users\vladi\Downloads\AllForScience\results\Kalininskoe\result.tif"
model_path       = "../Models/rf_multiclass_all_rgb_v4_second_birch_only_inner_augmented.pkl"
output_class_map = "../results/Kalininskoe/class_map.tif"
output_birch     = "../results/Kalininskoe/birch_mask.tif"
output_excel     = "../results/Kalininskoe/birch_report.xlsx"
patch_size       = 64
stride           = 64
block_size       = 100     # number of patches per block for batch processing
temp_threshold   = 70.0    # °C, throttle threshold for CPU overheating
cool_factor      = 0.5     # seconds per °C above threshold
n_workers        = max(1, int((os.cpu_count() or 1) * 0.8))

def get_cpu_temp():
    try:
        for entries in psutil.sensors_temperatures().values():
            for e in entries:
                if e.current:
                    return e.current
    except:
        pass
    return None

def init_worker():
    """Worker initialization: load model and open raster."""
    global model, src
    model = joblib.load(model_path)
    src   = rasterio.open(mosaic_path)

def extract_features(patch):
    """Extract features from an RGB patch."""
    R, G, B = patch[0].astype(float), patch[1].astype(float), patch[2].astype(float)
    mR, mG, mB = R.mean(), G.mean(), B.mean()
    exg  = (2 * G - R - B).mean()
    exr  = (1.4 * R - G).mean()
    exgr = exg - exr
    return [mR, mG, mB, exg, exr, exgr]

def process_block(coords):
    """Read and predict labels for a block of patches."""
    feats = []
    for top, left in coords:
        win   = Window(left, top, patch_size, patch_size)
        patch = src.read(window=win)
        feats.append(extract_features(patch))
    labels = model.predict(feats)
    return list(zip(coords, labels))

def count_birch_clusters(birch_mask, patch_size):
    """Count birch clusters on a grid of patches."""
    H, W = birch_mask.shape
    M, N = H // patch_size, W // patch_size
    grid = np.zeros((M, N), dtype=np.uint8)

    for r in range(M):
        for c in range(N):
            block = birch_mask[
                r * patch_size:(r + 1) * patch_size,
                c * patch_size:(c + 1) * patch_size
            ]
            grid[r, c] = 1 if block.max() == 1 else 0

    struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
    labels, n_clusters = label(grid, structure=struct)
    sizes = np.bincount(labels.ravel())[1:]
    return {
        "total_clusters": int(n_clusters),
        "single_patch":   int((sizes == 1).sum()),
        "multi_patch":    int((sizes > 1).sum()),
        "cluster_sizes":  sizes.tolist()
    }

def main():
    freeze_support()
    # Lower the process priority
    try:
        psutil.Process(os.getpid()).nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except:
        pass

    if not os.path.isfile(mosaic_path):
        raise FileNotFoundError(f"Mosaic not found: {mosaic_path}")

    # Read raster metadata
    with rasterio.open(mosaic_path) as tmp:
        meta = tmp.meta.copy()
        H, W = tmp.height, tmp.width

    # Initialize output arrays
    class_map = np.zeros((H, W), dtype=np.uint8)
    birch_map = np.zeros((H, W), dtype=np.uint8)

    # Generate all patch coordinates
    all_coords = [
        (i, j)
        for i in range(0, H - patch_size + 1, stride)
        for j in range(0, W - patch_size + 1, stride)
    ]
    blocks = [all_coords[i:i + block_size] for i in range(0, len(all_coords), block_size)]

    pbar = tqdm(total=len(blocks), desc="Blocks")
    with ProcessPoolExecutor(max_workers=n_workers, initializer=init_worker) as exe:
        futures = {exe.submit(process_block, blk): blk for blk in blocks}
        for fut in as_completed(futures):
            for (top, left), lbl in fut.result():
                class_map[top:top + patch_size, left:left + patch_size] = lbl
                birch_map[top:top + patch_size, left:left + patch_size] = (1 if lbl == 1 else 0)
            pbar.update(1)

            # Thermal throttling
            t = get_cpu_temp()
            if t and t > temp_threshold:
                sl = min((t - temp_threshold) * cool_factor, 5.0)
                pbar.write(f"[WARN] CPU {t:.1f}°C > {temp_threshold}°C → sleep {sl:.1f}s")
                time.sleep(sl)
    pbar.close()

    # Save GeoTIFFs
    meta.update(count=1, dtype='uint8')
    with rasterio.open(output_class_map, 'w', **meta) as dst:
        dst.write(class_map, 1)
    with rasterio.open(output_birch, 'w', **meta) as dst:
        dst.write(birch_map, 1)

    # Count clusters
    stats = count_birch_clusters(birch_map, patch_size=patch_size)

    # Estimate error probability (if OOB score is available)
    mdl = joblib.load(model_path)
    error_prob = None
    if hasattr(mdl, "oob_score_"):
        error_prob = 1.0 - mdl.oob_score_

    # Export summary report to Excel
    df = pd.DataFrame([{
        "total_clusters":    stats["total_clusters"],
        "single_patch":      stats["single_patch"],
        "multi_patch":       stats["multi_patch"],
        "error_probability": error_prob
    }])
    df.to_excel(output_excel, index=False)

    print("Done! Maps saved and summary written to", output_excel)

if __name__ == "__main__":
    main()
