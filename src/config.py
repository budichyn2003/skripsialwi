import os

# ==========================================
# KONFIGURASI PATH DIREKTORI
# ==========================================
# BASE_DIR akan otomatis mendeteksi root folder (SKRIPSI-ALWI)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Folder sumber (raw) dan tujuan (split)
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
SPLIT_DIR = os.path.join(BASE_DIR, 'data', 'split')

# ==========================================
# KONFIGURASI DATASET & KELAS (FASE 1)
# ==========================================
# 7 Kelas Target yang telah disetujui dosen
TARGET_CLASSES = [
    "cataflam_50_mg",
    "cetirizin_10_mg",
    "pantoprazol_sandoz_40_mg",
    "c_vitamin_teva_500_mg",
    "aspirin_ultra_500_mg",
    "algoflex_forte_dolo_400_mg",
    "merckformin_xr_1000_mg"
]

# Parameter Citra
IMG_SIZE = (224, 224)
AUGMENTATION_FACTOR = 7  # 1 gambar asli akan di-generate menjadi 7 variasi baru