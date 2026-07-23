import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
import os

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
# 2. DEFINISI KELAS & FUNGSI UTAMA
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

def build_model_architecture(scenario_name):
    """Otomatis membangun kerangka arsitektur model berdasarkan nama skenario."""
    if "Custom CNN" in scenario_name:
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(7, activation='softmax')
        ])
        return model
    elif "Optimized" in scenario_name or "Final Push" in scenario_name or "Brute Force" in scenario_name:
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
    else:
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False, weights=None, input_shape=(224, 224, 3)
        )
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(7, activation='softmax')
        ])
        return model

@st.cache_resource(show_spinner=False)
def load_classification_model(scenario_name, model_name):
    """Memuat bobot model dengan metode bypass arsitektur."""
    model_path = f'models/{model_name}'
    if not os.path.exists(model_path):
        return None

    try:
        model = build_model_architecture(scenario_name)
        model.load_weights(model_path, by_name=True, skip_mismatch=True)
        return model
    except Exception as e:
        st.sidebar.error(f"Error memuat model: {e}")
        return None

def preprocess_image(image, scenario_name):
    """Pipeline otomatis: Normalisasi warna, resize absolut 224x224, dan prapemrosesan input."""
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    if "Custom CNN" in scenario_name:
        img_array = img_array / 255.0
    else:
        img_array = preprocess_input(img_array)
        
    return img_array

# ==========================================
# 3. STRUKTUR SIDEBAR
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3014/3014385.png", width=80)
    st.markdown("### Konfigurasi Sistem")
    
    selected_scenario = st.selectbox(
        "Model Prediksi",
        ["Skenario Final Push (Terbaik)",
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
        "Skenario C Brute Force": "scenario_c_brute_force.h5"
    }
    
    target_model_file = model_mapping[selected_scenario]
    
    st.divider()
    st.markdown("**Status Sistem:**")
    model = load_classification_model(selected_scenario, target_model_file)
    
    if model is None:
        st.warning(f"⚠️ File `{target_model_file}` belum ditemukan di folder `models/`. Mode Simulasi Aktif.")
    else:
        st.success(f"✅ Model `{target_model_file}` Aktif & Terhubung!")

# ==========================================
# 4. KONTEN UTAMA (MAIN AREA)
# ==========================================
st.markdown("<h1 class='header-text'>💊 Sistem Klasifikasi Citra Obat Tablet</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #64748b;'>Unggah atau ambil foto obat tablet Anda. Sistem akan memproses dan mengklasifikasikannya secara otomatis.</p>", unsafe_allow_html=True)

# Pilihan metode masukan secara otomatis (Tanpa Crop Manual)
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
        st.image(image, caption="Citra Masukan Sistem", use_container_width=True)
        
    with col2:
        st.markdown("### Hasil Analisis Model")
        with st.spinner("Memproses analisis citra secara otomatis..."):
            
            if model is not None:
                img_tensor = preprocess_image(image, selected_scenario)
                predictions = model.predict(img_tensor)[0]
            else:
                np.random.seed(42) 
                dummy_raw = np.random.rand(len(CLASS_NAMES))
                predictions = dummy_raw / np.sum(dummy_raw) 
            
            top_index = np.argmax(predictions)
            predicted_class = CLASS_NAMES[top_index]
            confidence = predictions[top_index] * 100
            
            if model is None:
                st.info("ℹ️ Menampilkan hasil simulasi (Model belum dilatih).")
            
            st.success(f"**Identifikasi:** {predicted_class}")
            st.info(f"**Tingkat Keyakinan (Confidence):** {confidence:.2f}%")
            st.progress(int(confidence))
            
            st.markdown("#### Detail Probabilitas Top-3")
            pred_df = pd.DataFrame({
                'Obat': CLASS_NAMES,
                'Probabilitas (%)': predictions * 100
            }).sort_values(by='Probabilitas (%)', ascending=False).head(3)
            
            st.bar_chart(pred_df.set_index('Obat'))