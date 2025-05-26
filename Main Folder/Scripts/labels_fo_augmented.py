#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
from pathlib import Path

# ========== Parameters ==========
BASE_DIR = Path("../Images/new_ver")
SPLITS = ["augmented_train", "augmented_validation", "augmented_test"]

class_map = {
    "birch_inner":    1,
    "green_grass":    2,
    "yellow_grass":    3,
    "green_moss":     4,
    "stump":    5,
    "dirt":  6,
}

# Assign a weight to each class (e.g., mistakes get higher weight)
weight_map = {subdir: (5 if subdir == "mistake" else 1)
              for subdir in class_map}

OUTPUT_CSV = "labels_multiclass_all.csv"
# ========== End of parameters ==========

def generate_labels(base_dir: Path, split: str, subdir: str, label_value: int, weight_value: int):
    dirpath = base_dir / split / f"{subdir}_augmented"
    if not dirpath.is_dir():
        raise FileNotFoundError(f"Directory not found: {dirpath}")
    for tif in dirpath.glob("*.tif"):
        # relative path from BASE_DIR
        rel = tif.relative_to(base_dir).as_posix()
        yield rel, label_value, weight_value


def main():
    # Check that split directories exist
    missing = [split for split in SPLITS if not (BASE_DIR / split).is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing split dirs in {BASE_DIR}: {missing}")

    counts = {cls: 0 for cls in class_map.values()}
    wcounts = {cls: 0 for cls in class_map.values()}

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "class", "weight"] )

        for split in SPLITS:
            for subdir, cls in class_map.items():
                weight = weight_map[subdir]
                for relpath, label, w in generate_labels(BASE_DIR, split, subdir, cls, weight):
                    writer.writerow([relpath, label, w])
                    counts[cls] += 1
                    wcounts[cls] += 1

    print(f"'{OUTPUT_CSV}' has been created.")
    for subdir, cls in class_map.items():
        print(f"  {subdir}: class={cls}, files={counts[cls]}, weight={weight_map[subdir]}×{wcounts[cls]}")

if __name__ == "__main__":
    main()
