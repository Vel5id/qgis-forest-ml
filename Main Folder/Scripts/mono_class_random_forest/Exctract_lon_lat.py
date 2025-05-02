# -*- coding: utf-8 -*-
"""
Извлекает lat/lon/accuracy из NoteCam-снимков:
  1) сперва из EXIF-тегов;
  2) если GPS-тегов нет ­— через OCR надписи в левом-нижнем углу.
Пишет результат в seedlings.csv (UTF-8).
"""
import pathlib, csv, re, exifread, cv2, pytesseract

# ─── НАСТРОЙКИ ────────────────────────────────────────────────────────────────
IMG_DIR  = pathlib.Path(r"C:\Users\vladi\Downloads\AllForScience\Images\Baghanaly 2")
OUT_CSV  = IMG_DIR.parent / "seedlings.csv"
SPECIES  = "Саженец Ели"          # подпись в столбце species
CROP_Y0  = 0.74                   # нижние 26 % кадра
CROP_X1  = 0.50                   # левая половина (широта/долгота всегда слева)
# ──────────────────────────────────────────────────────────────────────────────

# Regex для всей надписи; двух строк достаточно (Accuracy может отсутствовать)
OCR_RE = re.compile(
    r"Latitude:\s*([-0-9., ]+).*?"
    r"Longitude:\s*([-0-9., ]+).*?"
    r"(?:Accuracy:\s*([-0-9., ]+))?", re.S | re.I)

def dms_to_deg(dms_tag):
    d, m, s = [v.num / v.den for v in dms_tag.values]
    return d + m / 60 + s / 3600

def from_exif(tags):
    try:
        lat = dms_to_deg(tags["GPS GPSLatitude"])
        if tags["GPS GPSLatitudeRef"].values != "N": lat *= -1
        lon = dms_to_deg(tags["GPS GPSLongitude"])
        if tags["GPS GPSLongitudeRef"].values != "E": lon *= -1
        acc_tag = tags.get("GPS GPSHPositioningError")
        acc = float(acc_tag.values[0]) if acc_tag else ""
        return lat, lon, acc
    except KeyError:
        return None

# ▸ вытаскиваем ПЕРВОЕ число вида  64  |  64.1  |  -64.123
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
def parse_num(txt: str) -> float:
    m = NUM_RE.search(txt)
    if not m:
        raise ValueError(f"Не найдено число в «{txt}»")
    return float(m.group())

def from_ocr(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    h, w = img.shape[:2]
    crop = img[int(h * CROP_Y0):, :int(w * CROP_X1)]
    text = pytesseract.image_to_string(crop, lang="eng")
    m = OCR_RE.search(text)
    if not m:
        return None
    lat, lon, acc = m.groups()
    return parse_num(lat), parse_num(lon), parse_num(acc or "0")

with OUT_CSV.open("w", newline="", encoding="utf-8") as out:
    writer = csv.writer(out)
    writer.writerow(["species", "lat", "lon", "accuracy_m", "source"])

    for img_path in IMG_DIR.glob("*.jpg"):
        with img_path.open("rb") as f:
            tags = exifread.process_file(f, stop_tag="UNDEF")

        data = from_exif(tags)
        src  = "EXIF"
        if not data:
            data = from_ocr(img_path)
            src  = "OCR"
        if data:
            writer.writerow([SPECIES, *data, src])
        else:
            print(f"[!]   не удалось извлечь координаты: {img_path.name}")

print(f"✓ Завершено — {OUT_CSV}")
