#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import os

# ========== Parameters ==========

BASE_DIR = "../Images/base"

class_map = {
    "birch":    1,
    "grass":    2,
    "moss":     3,
    "stump":    4,
    "water":    5,
    "ash_way":  6,
    "mistake":  2,
}

# Assign a weight to each subfolder
weight_map = {subdir: (5 if subdir == "mistake" else 1)
              for subdir in class_map}

OUTPUT_CSV = "labels_multiclass_all.csv"

# ========== End of parameters ==========

def generate_labels(dirpath, label_value, weight_value):
    for root, _, files in os.walk(dirpath):
        for fname in files:
            if fname.lower().endswith(".tif"):
                full = os.path.join(root, fname)
                rel  = os.path.relpath(full).replace(os.sep, "/")
                yield rel, label_value, weight_value

def main():
    if not os.path.isdir(BASE_DIR):
        raise FileNotFoundError(f"Directory not found: {BASE_DIR}")

    missing = [d for d in class_map
               if not os.path.isdir(os.path.join(BASE_DIR, d))]
    if missing:
        raise FileNotFoundError(f"These subdirectories are missing in {BASE_DIR}: {missing}")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # header row
        writer.writerow(["filepath", "class", "weight"])

        counts = {cls: 0 for cls in class_map.values()}
        wcounts = {cls: 0 for cls in class_map.values()}

        for subdir, cls in class_map.items():
            dirpath = os.path.join(BASE_DIR, subdir)
            w = weight_map[subdir]
            for relpath, label, _ in generate_labels(dirpath, cls, w):
                writer.writerow([relpath, label, w])
                counts[cls] += 1
                wcounts[cls] += 1

    print(f"'{OUTPUT_CSV}' has been created.")
    for subdir, cls in class_map.items():
        print(f"  {subdir}: class={cls}, files={counts[cls]}, weight={weight_map[subdir]}×{wcounts[cls]}")

if __name__ == "__main__":
    main()
