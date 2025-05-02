import pandas as pd
import math

SRC  = r"C:\Users\vladi\Downloads\AllForScience\Images + Dots\Kalininskoe 1 - Dots.csv"           # исходный файл
DEST = r"C:\Users\vladi\Downloads\AllForScience\Images + Dots\Kalininskoe 1 - Dots_shift4m.csv"   # куда сохранить

DX = -45.5   # м  → восток(вправо) = +, запад = –
DY =  +22.5   # м  → север(вверх) = +, юг  = –

df = pd.read_csv(SRC)

# ── поправка по долготе (зависит от широты) ──────────────────
lat_rad   = df["lat"].astype(float).apply(math.radians)
delta_lon = DX / (111_320 * lat_rad.apply(math.cos))     # градусы

# ── поправка по широте (практически константа) ──────────────
delta_lat = DY / 110_940                                  # градусы

df["lon"] = df["lon"].astype(float) + delta_lon
df["lat"] = df["lat"].astype(float) + delta_lat

df.to_csv(DEST, index=False, float_format="%.8f")
print("✓ Сдвинуто:  dX={} м, dY={} м  → {}".format(DX, DY, DEST))