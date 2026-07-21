import os
import cv2
import albumentations as A
from tqdm import tqdm
from config import RAW_DIR, SPLIT_DIR, TARGET_CLASSES, IMG_SIZE, AUGMENTATION_FACTOR

def create_directory_structure():
    """Membuat folder tujuan jika belum ada (train, valid, test untuk 7 kelas)"""
    splits = ['train', 'valid', 'test']
    for split in splits:
        for cls in TARGET_CLASSES:
            dir_path = os.path.join(SPLIT_DIR, split, cls)
            os.makedirs(dir_path, exist_ok=True)
    print("✅ Struktur direktori tujuan ('data/split/') berhasil disiapkan.")

def pad_and_resize(image, target_size=IMG_SIZE):
    """
    Mempertahankan Aspect Ratio: 
    Menambahkan padding hitam agar gambar menjadi persegi sebelum di-resize, 
    sehingga bentuk geometri obat (lonjong/bulat) tidak rusak (stretching).
    """
    old_size = image.shape[:2] # (height, width)
    ratio = float(target_size[0]) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    
    # Resize dengan rasio yang benar
    image = cv2.resize(image, (new_size[1], new_size[0]))
    
    # Hitung padding yang dibutuhkan untuk mencapai target_size
    delta_w = target_size[1] - new_size[1]
    delta_h = target_size[0] - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    
    # Tambahkan batas padding berwarna hitam (0,0,0)
    color = [0, 0, 0]
    new_img = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return new_img

def get_augmentation_pipeline():
    """
    Skenario Augmentasi: Rotasi, Shift, Zoom In/Out, Brightness, Contrast, dan Noise.
    Dilarang menggunakan Shear, Stretch, atau Hue Shift.
    """
    return A.Compose([
        A.Rotate(limit=360, p=1.0), # Rotasi acak 0-360 derajat
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8), # Variasi pencahayaan
        A.Affine(scale=(0.8, 1.2), translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, p=0.8), # Zoom & Shift
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5), # Simulasi noise sensor kamera
    ])

def process_and_augment_data():
    """Fungsi utama untuk memproses seluruh dataset (Preprocessing & Augmentasi)"""
    splits = ['train', 'valid', 'test']
    aug_pipeline = get_augmentation_pipeline()
    
    for split in splits:
        print(f"\n🚀 Memproses direktori: {split.upper()}")
        
        for cls in TARGET_CLASSES:
            src_class_dir = os.path.join(RAW_DIR, split, cls)
            dst_class_dir = os.path.join(SPLIT_DIR, split, cls)
            
            if not os.path.exists(src_class_dir):
                print(f"⚠️ Peringatan: Folder {src_class_dir} tidak ditemukan! Lewati.")
                continue
                
            image_files = [f for f in os.listdir(src_class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # Progress bar per kelas
            for img_name in tqdm(image_files, desc=f"Kelas: {cls}"):
                img_path = os.path.join(src_class_dir, img_name)
                
                # Baca gambar menggunakan OpenCV
                image = cv2.imread(img_path)
                if image is None:
                    continue
                
                # PREPROCESSING 1: Pertahankan rasio dan resize (224x224)
                processed_img = pad_and_resize(image)
                
                # Simpan Citra Asli (Base Image)
                base_save_path = os.path.join(dst_class_dir, f"base_{img_name}")
                cv2.imwrite(base_save_path, processed_img)
                
                # DATA AUGMENTATION (Hanya untuk direktori TRAIN)
                if split == 'train':
                    for i in range(AUGMENTATION_FACTOR):
                        # Terapkan transformasi dari pipeline
                        augmented = aug_pipeline(image=processed_img)
                        aug_img = augmented['image']
                        
                        # Simpan hasil augmentasi
                        aug_save_path = os.path.join(dst_class_dir, f"aug_{i}_{img_name}")
                        cv2.imwrite(aug_save_path, aug_img)

if __name__ == "__main__":
    print("=== MEMULAI FASE 1: DATA ENGINEERING PIPELINE ===")
    create_directory_structure()
    process_and_augment_data()
    print("\n✅ FASE 1 SELESAI! Data latih berhasil diagregasi menjadi total 1.568 gambar.")