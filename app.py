import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
import os
import cv2
import scipy.ndimage as ndimage

from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.applications.efficientnet import preprocess_input

# ==========================================
# 1. KONFIGURASI HALAMAN & CUSTOM CSS
# ==========================================
st.set_page_config(
    page_title="AI Klasifikasi Obat Tablet",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    .header-text { text-align: center; color: #4facfe; padding-bottom: 2rem; font-weight: bold; }
    div[data-testid="stSidebar"] { background-color: #161b22; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DEFINISI KELAS & FUNGSI PREPROCESSING
# ==========================================
# 🔥 PERBAIKAN: Urutan ini disamakan 100% dengan urutan indeks generator di Colab
CLASS_NAMES = [
    'Cataflam 50 mg',              # Index 0
    'Cetirizin 10 mg',             # Index 1
    'Pantoprazol Sandoz 40 mg',    # Index 2
    'Vitamin C Teva 500 mg',       # Index 3
    'Aspirin Ultra 500 mg',        # Index 4
    'Algoflex Forte Dolo 400 mg',  # Index 5
    'Merckformin XR 1000 mg'       # Index 6
]

CONFIDENCE_THRESHOLD = 15.0  # Ambang batas penolakan

def apply_clahe_fix(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    enhanced_bgr = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    return enhanced_bgr

def pad_and_resize(image, target_size=(224, 224)):
    old_size = image.shape[:2]
    ratio = float(target_size[0]) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    image_resized = cv2.resize(image, (new_size[1], new_size[0]))
    delta_w = target_size[1] - new_size[1]
    delta_h = target_size[0] - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    new_image = cv2.copyMakeBorder(image_resized, top, bottom, left, right,
                                   cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return new_image

def crop_tablet(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return image
        
    cnt = max(contours, key=cv2.contourArea)
    image_area = image.shape[0] * image.shape[1]
    if cv2.contourArea(cnt) > (0.8 * image_area) or cv2.contourArea(cnt) < 1000:
        return image
        
    x, y, w, h = cv2.boundingRect(cnt)
    margin = 30
    
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(image.shape[1] - x, w + 2*margin)
    h = min(image.shape[0] - y, h + 2*margin)
    
    return image[y:y+h, x:x+w]

# ----- FUNGSI PREDIKSI -----
def predict_with_tta(image_pil, model):
    img = np.array(image_pil.convert('RGB'))[:, :, ::-1]  # RGB -> BGR
    if img is None:
        print("❌ ERROR: Gambar tidak terbaca oleh OpenCV!")
        return None, None, True, None

    # 🔥 LOGGING DIKEMBALIKAN KE TERMINAL
    print("\n" + "="*50)
    print("🔍 MEMULAI LOG PREPROCESSING GAMBAR")
    print("="*50)
    print(f"[1] Gambar Asli       : Shape={img.shape}, Min={img.min()}, Max={img.max()}")

    img_cropped = crop_tablet(img)
    print(f"[2] Setelah Crop      : Shape={img_cropped.shape}, Min={img_cropped.min()}, Max={img_cropped.max()}")
    
    img_clahe = apply_clahe_fix(img_cropped)
    print(f"[3] Setelah CLAHE     : Shape={img_clahe.shape}, Min={img_clahe.min()}, Max={img_clahe.max()}")
    
    img_resized = pad_and_resize(img_clahe, (224, 224))
    print(f"[4] Setelah Resize/Pad: Shape={img_resized.shape}, Min={img_resized.min()}, Max={img_resized.max()}")
    
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    angles = [0, 90, 180, 270]
    probs_list = []
    
    print("\n⚙️ MULAI PREDIKSI TTA (4 Rotasi)")
    for angle in angles:
        if angle != 0:
            rotated = ndimage.rotate(img_rgb, angle, reshape=False, cval=0)
        else:
            rotated = img_rgb
            
        img_array = np.expand_dims(rotated, axis=0)
        
        # Normalisasi Wajib EfficientNet
        img_array = np.array(img_array, dtype=np.float32)
        img_array = preprocess_input(img_array) 
        
        print(f"  -> Input Model (Rotasi {angle:03d}°) : Shape={img_array.shape}, Min={img_array.min():.4f}, Max={img_array.max():.4f}, Mean={img_array.mean():.4f}")
        
        preds = model.predict(img_array, verbose=0)
        probs_list.append(preds[0])
        print(f"  -> Hasil Prediksi Mentah : {preds[0]}")
        
    avg_probs = np.mean(probs_list, axis=0)
    max_conf = np.max(avg_probs) * 100
    idx = np.argmax(avg_probs)

    print("\n📊 KESIMPULAN HASIL:")
    print(f"  -> Rata-rata Probabilitas: {avg_probs}")
    print(f"  -> Kelas Terpilih        : {CLASS_NAMES[idx]} ({max_conf:.2f}%)")
    print("="*50 + "\n")

    if max_conf < CONFIDENCE_THRESHOLD:
        return "❌ BUKAN OBAT / TIDAK DIKENALI", max_conf, True, avg_probs
    else:
        return CLASS_NAMES[idx], max_conf, False, avg_probs

# ==========================================
# 3. LOAD MODEL (FLAT ARCHITECTURE FIX)
# ==========================================
def build_model_architecture(model_name):
    input_shape = (224, 224, 3)
    
    if "scenario_a" in model_name or "scenario_b" in model_name:
        return tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=input_shape),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(7, activation='softmax')
        ])
        
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False, 
        weights=None, 
        input_shape=input_shape
    )
    
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    if any(keyword in model_name for keyword in ["silent", "optimized", "final", "brute"]):
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        
    outputs = tf.keras.layers.Dense(7, activation='softmax')(x)
    return tf.keras.models.Model(inputs=base_model.input, outputs=outputs)

@st.cache_resource(show_spinner=False)
def load_classification_model(model_name):
    model_path = f'models/{model_name}'
    if not os.path.exists(model_path):
        return None

    try:
        model = build_model_architecture(model_name)
        model.load_weights(model_path)
        return model
    except Exception as e:
        st.sidebar.error(f"Error memuat model: {e}")
        return None

# ==========================================
# 4. SIDEBAR
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3014/3014385.png", width=80)
    st.markdown("### Konfigurasi Sistem")

    selected_scenario = st.selectbox(
        "Model Prediksi",
        ["Eksperimen Silent Training (CLAHE Fix)",
         "Skenario Final Push (Terbaik)",
         "Skenario C (EfficientNetB0)",
         "Skenario A (Custom CNN - Baseline)",
         "Skenario B (Custom CNN + Augmentasi)",
         "Skenario C Optimized",
         "Skenario C Brute Force"]
    )

    model_mapping = {
        "Skenario A (Custom CNN - Baseline)": "scenario_a.h5",
        "Skenario B (Custom CNN + Augmentasi)": "scenario_b.h5",
        "Skenario C (EfficientNetB0)": "scenario_c.h5",
        "Skenario C Optimized": "scenario_c_optimized.h5",
        "Skenario Final Push (Terbaik)": "scenario_c_final_retrain_v2.h5",
        "Skenario C Brute Force": "scenario_c_brute_force.h5",
        "Eksperimen Silent Training (CLAHE Fix)": "silent_training.h5"
    }

    target_model_file = model_mapping[selected_scenario]

    st.divider()
    st.markdown("**Status Sistem:**")
    model = load_classification_model(target_model_file)

    if model is None:
        st.warning(f"⚠️ File `{target_model_file}` belum ditemukan. Mode Simulasi Aktif.")
    else:
        st.success(f"✅ Model `{target_model_file}` Aktif & Terhubung!")

# ==========================================
# 5. KONTEN UTAMA
# ==========================================
st.markdown("<h1 class='header-text'>💊 Sistem Klasifikasi Citra Obat Tablet</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Unggah atau ambil foto obat tablet Anda. Sistem akan memproses dan mengklasifikasikannya secara otomatis.</p>", unsafe_allow_html=True)

input_method = st.radio("Pilih Metode Masukan Citra:", ["Unggah Berkas (File Uploader)", "Gunakan Kamera (Kamera Langsung)"], horizontal=True)

uploaded_file = None
if input_method == "Unggah Berkas (File Uploader)":
    uploaded_file = st.file_uploader("Seret dan Lepas (Drag & Drop) Citra Obat di sini", type=["jpg", "jpeg", "png"])
else:
    uploaded_file = st.camera_input("Ambil Foto Obat Tablet dengan Kamera")

if uploaded_file is not None:
    st.divider()
    col1, col2 = st.columns([1, 1.5], gap="large")

    with col1:
        st.markdown("### Citra Input")
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Citra Masukan Sistem", use_column_width=True)

    with col2:
        st.markdown("### Hasil Analisis Model")
        with st.spinner("Memproses analisis citra secara otomatis..."):

            if model is not None:
                pred_label, confidence, is_rejected, probs = predict_with_tta(image, model)

                if is_rejected:
                    st.error(f"🚫 {pred_label}")
                    st.info(f"Confidence tertinggi: {confidence:.2f}% (< {CONFIDENCE_THRESHOLD}%)")
                else:
                    st.success(f"**Identifikasi:** {pred_label}")
                    st.info(f"**Tingkat Keyakinan (Confidence):** {confidence:.2f}%")
                    st.progress(int(confidence))
            else:
                np.random.seed(42)
                dummy_raw = np.random.rand(len(CLASS_NAMES))
                probs = dummy_raw / np.sum(dummy_raw)
                pred_label = CLASS_NAMES[np.argmax(probs)]
                confidence = np.max(probs) * 100
                st.info("ℹ️ Menampilkan hasil simulasi (Model belum dilatih).")
                st.success(f"**Identifikasi:** {pred_label}")
                st.info(f"**Tingkat Keyakinan (Confidence):** {confidence:.2f}%")
                st.progress(int(confidence))

            st.markdown("#### Detail Probabilitas Top-3")
            if probs is not None:
                pred_df = pd.DataFrame({
                    'Obat': CLASS_NAMES,
                    'Probabilitas (%)': probs * 100
                }).sort_values(by='Probabilitas (%)', ascending=False).head(3)
                st.bar_chart(pred_df.set_index('Obat'))
            else:
                st.warning("Tidak ada data probabilitas untuk ditampilkan.")