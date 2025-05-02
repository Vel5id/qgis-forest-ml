#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_labels.py

Генерирует CSV-файл labels.csv со списком TIFF-патчей и метками:
    1 — содержит берёзу (pos/)
    0 — не содержит берёзу (neg/)
"""

import csv
import os

def main():
    # Директории с изображениями
    pos_dir = "pos"
    neg_dir = "neg"

    # Префиксы файлов
    pos_prefix = "test_"
    neg_prefix = "nobirch_"

    # Количество файлов (например, test_0.tif … test_53.tif)
    pos_count = 54   # от 0 до 53 включительно
    neg_count = 63   # от 0 до 62 включительно

    output_csv = "labels.csv"

    # Убедимся, что каталоги существуют
    if not os.path.isdir(pos_dir):
        raise FileNotFoundError(f"Директория не найдена: {pos_dir}")
    if not os.path.isdir(neg_dir):
        raise FileNotFoundError(f"Директория не найдена: {neg_dir}")

    # Создаём CSV
    with open(output_csv, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        # Заголовок
        writer.writerow(["filepath", "label"])

        # Положительные образцы — метка = 1
        for i in range(pos_count):
            filename = f"{pos_prefix}{i}.tif"
            path = os.path.join(pos_dir, filename)
            writer.writerow([path, 1])

        # Отрицательные образцы — метка = 0
        for i in range(neg_count):
            filename = f"{neg_prefix}{i}.tif"
            path = os.path.join(neg_dir, filename)
            writer.writerow([path, 0])

    print(f"Файл '{output_csv}' успешно создан: {pos_count} положительных и {neg_count} отрицательных записей.")

if __name__ == "__main__":
    main()
