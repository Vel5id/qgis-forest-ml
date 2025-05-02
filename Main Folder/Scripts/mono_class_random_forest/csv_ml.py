import csv
import os

def generate_labels(input_dir, label_value):
    """
    Walk `input_dir` recursively and yield (relative_path, label_value)
    for each .tif file found.
    """
    for root, _, files in os.walk(input_dir):
        for fname in files:
            if fname.lower().endswith(".tif"):
                full_path = os.path.join(root, fname)
                rel_path = os.path.relpath(full_path).replace(os.sep, "/")
                yield rel_path, label_value

def main():
    pos_dir = "pos_test_aug"
    # список папок с негативными примерами
    neg_dirs = ["neg_test_aug", "hard_neg(3iter_augmented"]
    output_csv = "hard_labels.csv"

    with open(output_csv, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filepath", "label"])

        # positive
        for path, label in generate_labels(pos_dir, 1):
            writer.writerow([path, label])

        # negatives (несколько директорий)
        total_neg = 0
        for neg_dir in neg_dirs:
            for path, label in generate_labels(neg_dir, 0):
                writer.writerow([path, label])
                total_neg += 1

    # вывод статистики
    pos_count = sum(1 for _ in generate_labels(pos_dir, 1))
    print(f"'{output_csv}' created:")
    print(f"  Positives (1): {pos_count}")
    print(f"  Negatives (0): {total_neg}")

if __name__ == "__main__":
    main()
