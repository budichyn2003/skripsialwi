from pathlib import Path
from collections import Counter

# ==========================
# Lokasi dataset
# ==========================
DATASET_PATH = Path("data/raw/valid/images")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

counter = Counter()
total_gambar = 0

for file in DATASET_PATH.iterdir():

    if not file.is_file():
        continue

    if file.suffix.lower() not in IMAGE_EXTENSIONS:
        continue

    total_gambar += 1

    nama = file.stem

    # hapus bagian nomor di belakang
    if "_s_" in nama:
        nama = nama.split("_s_")[0]
    elif "_u_" in nama:
        nama = nama.split("_u_")[0]

    counter[nama] += 1

# ==========================
# Simpan hasil
# ==========================

with open("list_nama_obat.txt", "w", encoding="utf-8") as f:

    f.write("=" * 60 + "\n")
    f.write("LIST NAMA OBAT\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"Total Jenis Obat : {len(counter)}\n")
    f.write(f"Total Gambar     : {total_gambar}\n\n")

    f.write("-" * 60 + "\n")
    f.write(f"{'Nama Obat':40}Jumlah\n")
    f.write("-" * 60 + "\n")

    for nama in sorted(counter):
        f.write(f"{nama:40}{counter[nama]}\n")

    f.write("-" * 60 + "\n")

print(f"Total Jenis Obat : {len(counter)}")
print(f"Total Gambar     : {total_gambar}")
print("Selesai.")