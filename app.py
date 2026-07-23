import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
import os
import cv2
import scipy.ndimage as ndimage

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
    .main { background-color: #f8f9fa; }
    .header-text { text-align: center; color: #1e3a8a; padding-bottom: 2rem; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DEFINISI KELAS & FUNGSI PREPROCESSING
# ==========================================
CLASS_NAMES = [
    'Algoflex Forte Dolo 400 mg',
    'Aspirin Ultra 500 mg',
    'Vitamin C Teva 500 mg',
    'Cataflam 50 mg',
    'Cetirizin 10 mg',
    'Merckformin XR 1000 mg',
    'Pantoprazol Sandoz 40 mg'
]

CONFIDENCE_THRESHOLD = 10.0  # Turunkan threshold agar lebih longgar

# ----- FUNGSI PREPROCESSING (SAMA PERSIS DENGAN COLAB) -----
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
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    margin = 10
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(image.shape[1] - x, w + 2*margin)
    h = min(image.shape[0] - y, h + 2*margin)
    cropped = image[y:y+h, x:x+w]
    return cropped

# ----- FUNGSI PREDIKSI DENGAN TTA + THRESHOLD -----
def predict_with_tta(image_pil, model):
    img = np.array(image_pil.convert('RGB'))[:, :, ::-1]  # RGB -> BGR
    if img is None:
        return None, None, True, None

    img = crop_tablet(img)
    img = apply_clahe_fix(img)
    img = pad_and_resize(img, (224, 224))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    angles = [0, 90, 180, 270]
    probs_list = []
    for angle in angles:
        if angle != 0:
            rotated = ndimage.rotate(img_rgb, angle, reshape=False, cval=0)
        else:
            rotated = img_rgb
        img_array = np.expand_dims(rotated, axis=0)
        img_array = preprocess_input(img_array)
        preds = model.predict(img_array, verbose=0)
        probs_list.append(preds[0])
    avg_probs = np.mean(probs_list, axis=0)
    max_conf = np.max(avg_probs) * 100
    idx = np.argmax(avg_probs)

    if max_conf < CONFIDENCE_THRESHOLD:
        return "❌ BUKAN OBAT / TIDAK DIKENALI", max_conf, True, avg_probs
    else:
        return CLASS_NAMES[idx], max_conf, False, avg_probs

# ==========================================
# 3. LOAD MODEL
# ==========================================
def build_model_architecture(model_type):
    if model_type == "Custom CNN":
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(7, activation='softmax')
        ])
        return model
    else:
        # Untuk silent_training dan model EfficientNet lainnya
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False, weights=None, input_shape=(224, 224, 3)
        )
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(7, activation='softmax')
        ])
        return model

@st.cache_resource(show_spinner=False)
def load_classification_model(model_name):
    model_path = f'models/{model_name}'
    if not os.path.exists(model_path):
        st.sidebar.error(f"❌ File {model_path} tidak ditemukan!")
        return None

    # Tentukan jenis arsitektur berdasarkan nama file
    if "scenario_a" in model_name or "scenario_b" in model_name:
        model_type = "Custom CNN"
    else:
        model_type = "EfficientNet"

    try:
        model = build_model_architecture(model_type)
        model.load_weights(model_path, by_name=True, skip_mismatch=True)
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
        st.warning(f"⚠️ File `{target_model_file}` belum ditemukan di folder `models/`. Mode Simulasi Aktif.")
    else:
        st.success(f"✅ Model `{target_model_file}` Aktif & Terhubung!")

# ==========================================
# 5. KONTEN UTAMA
# ==========================================
st.markdown("<h1 class='header-text'>💊 Sistem Klasifikasi Citra Obat Tablet</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #64748b;'>Unggah atau ambil foto obat tablet Anda. Sistem akan memproses dan mengklasifikasikannya secara otomatis.</p>", unsafe_allow_html=True)

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

                # Debug: tampilkan semua probabilitas untuk pemeriksaan
                st.caption("Probabilitas per kelas (debug):")
                debug_df = pd.DataFrame({
                    'Kelas': CLASS_NAMES,
                    'Prob': probs * 100
                }).sort_values('Prob', ascending=False)
                st.dataframe(debug_df, use_container_width=True)

                if is_rejected:
                    st.error(f"🚫 {pred_label}")
                    st.info(f"Confidence tertinggi: {confidence:.2f}% (< {CONFIDENCE_THRESHOLD}%)")
                else:
                    st.success(f"**Identifikasi:** {pred_label}")
                    st.info(f"**Tingkat Keyakinan (Confidence):** {confidence:.2f}%")
                    st.progress(int(confidence))
            else:
                # Mode simulasi
                np.random.seed(42)
                dummy_raw = np.random.rand(len(CLASS_NAMES))
                probs = dummy_raw / np.sum(dummy_raw)
                pred_label = CLASS_NAMES[np.argmax(probs)]
                confidence = np.max(probs) * 100
                st.info("ℹ️ Menampilkan hasil simulasi (Model belum dilatih).")
                st.success(f"**Identifikasi:** {pred_label}")
                st.info(f"**Tingkat Keyakinan (Confidence):** {confidence:.2f}%")
                st.progress(int(confidence))

            # Tampilkan top-3 probabilitas
            st.markdown("#### Detail Probabilitas Top-3")
            if probs is not None:
                pred_df = pd.DataFrame({
                    'Obat': CLASS_NAMES,
                    'Probabilitas (%)': probs * 100
                }).sort_values(by='Probabilitas (%)', ascending=False).head(3)
                st.bar_chart(pred_df.set_index('Obat'))
            else:
                st.warning("Tidak ada data probabilitas untuk ditampilkan.")