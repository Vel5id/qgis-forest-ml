import rasterio
import numpy as np
import joblib
import os
import psutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from rasterio.windows import Window
from tqdm import tqdm
from multiprocessing import freeze_support

# === Parameters ===
mosaic_path      = r"C:\Users\vladi\Downloads\AllForScience\results\Калининское 1\result.tif"
model_path       = "rf_multiclass_all_rgb_v4_second_birch_only_inner_augmented.pkl"
output_class_map = "class_map.tif"
output_birch     = "birch_mask.tif"
patch_size       = 64
stride           = 64
temp_threshold   = 70.0   # °C
cool_factor      = 0.5    # sec per °C above threshold
block_size       = 100    # patches per block

# Use ~80% of logical cores
n_total  = os.cpu_count() or 1
n_workers = max(1, int(n_total * 0.8))

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
    """Each worker loads model and opens raster."""
    global model, src
    model = joblib.load(model_path)
    src   = rasterio.open(mosaic_path)

def extract_features(patch):
    R,G,B = patch[0].astype(float), patch[1].astype(float), patch[2].astype(float)
    mR, mG, mB = R.mean(), G.mean(), B.mean()
    exg  = (2*G - R - B).mean()
    exr  = (1.4*R - G).mean()
    exgr = exg - exr
    return [mR, mG, mB, exg, exr, exgr]

def process_block(coords_block):
    """Read & predict a small block of patches in one go."""
    feats = []
    for top,left in coords_block:
        win   = Window(left, top, patch_size, patch_size)
        patch = src.read(window=win)
        feats.append(extract_features(patch))
    labels = model.predict(feats)
    return list(zip(coords_block, labels))

def main():
    freeze_support()
    # lower priority
    p = psutil.Process(os.getpid())
    try: p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except: pass

    if not os.path.isfile(mosaic_path):
        raise FileNotFoundError(mosaic_path)

    # read metadata only
    with rasterio.open(mosaic_path) as tmp:
        meta = tmp.meta.copy()
        h, w = tmp.height, tmp.width

    class_map = np.zeros((h, w), dtype=np.uint8)
    birch_map = np.zeros((h, w), dtype=np.uint8)

    # all patch coords
    all_coords = [
        (i, j)
        for i in range(0, h-patch_size+1, stride)
        for j in range(0, w-patch_size+1, stride)
    ]
    # split into blocks
    blocks = [all_coords[i:i+block_size] for i in range(0, len(all_coords), block_size)]

    pbar = tqdm(total=len(blocks), desc="Blocks")
    with ProcessPoolExecutor(max_workers=n_workers,
                             initializer=init_worker) as exe:
        futures = {exe.submit(process_block, blk): blk for blk in blocks}
        for fut in as_completed(futures):
            for (top,left), lbl in fut.result():
                class_map[top:top+patch_size, left:left+patch_size] = lbl
                birch_map[top:top+patch_size, left:left+patch_size] = (1 if lbl==1 else 0)
            pbar.update(1)

            # thermal management
            t = get_cpu_temp()
            if t and t>temp_threshold:
                sl = min((t-temp_threshold)*cool_factor, 5.0)
                pbar.write(f"[WARN] CPU {t:.1f}°C >{temp_threshold}°C → sleep {sl:.1f}s")
                time.sleep(sl)
    pbar.close()

    # save
    meta.update(count=1, dtype='uint8')
    with rasterio.open(output_class_map,'w',**meta) as dst:
        dst.write(class_map,1)
    with rasterio.open(output_birch,'w',**meta) as dst:
        dst.write(birch_map,1)

    print("Done.")

if __name__=='__main__':
    main()
