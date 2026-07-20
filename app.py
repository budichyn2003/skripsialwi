import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
import os

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
CLASS_NAMES = sorted([
    'Cataflam', 'Cetirizine', 'Paracetamol', 
    'Vitamin C IPI', 'Aspirin Ultra', 'Ibuprofen', 'Metformin'
])

@st.cache_resource(show_spinner=False)
def load_classification_model(model_name):
    """Fungsi memuat model fisik jika file-nya sudah tersedia"""
    model_path = f'models/{model_name}'
    if os.path.exists(model_path):
        try:
            return tf.keras.models.load_model(model_path)
        except OSError:
            return None
    return None

def preprocess_image(image):
    """Standarisasi input gambar ke 224x224 piksel dan normalisasi RGB"""
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, 0)
    img_array = img_array / 255.0
    return img_array

# ==========================================
# 3. STRUKTUR SIDEBAR
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3014/3014385.png", width=80)
    st.markdown("### Konfigurasi Sistem")
    
    selected_scenario = st.selectbox(
        "Model Prediksi",
        ["Skenario C (EfficientNetB0 - Terbaik)", 
         "Skenario B (Custom CNN + Augmentasi)", 
         "Skenario A (Custom CNN - Baseline)"]
    )
    
    model_mapping = {
        "Skenario C (EfficientNetB0 - Terbaik)": "scenario_c.h5",
        "Skenario B (Custom CNN + Augmentasi)": "scenario_b.h5",
        "Skenario A (Custom CNN - Baseline)": "scenario_a.h5"
    }
    
    target_model_file = model_mapping[selected_scenario]
    
    st.divider()
    st.markdown("**Status Sistem:**")
    model = load_classification_model(target_model_file)
    
    if model is None:
        st.warning("⚠️ Mode Simulasi (Mock Mode Aktif)\nFile `.h5` belum ditemukan. Prediksi akan menggunakan data dummy sementara.")
    else:
        st.success("✅ Model Aktif & Terhubung!")

# ==========================================
# 4. KONTEN UTAMA (MAIN AREA)
# ==========================================
st.markdown("<h1 class='header-text'>💊 Sistem Klasifikasi Citra Obat Tablet</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #64748b;'>Unggah citra obat tablet Anda untuk menguji antarmuka sistem.</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Seret dan Lepas (Drag & Drop) Citra Obat di sini", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.divider()
    col1, col2 = st.columns([1, 1.5], gap="large")
    
    with col1:
        st.markdown("### Citra Input")
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption=f"File: {uploaded_file.name}", use_container_width=True)
        
    with col2:
        st.markdown("### Hasil Analisis Model")
        with st.spinner("Memproses analisis citra..."):
            
            # CEK APAKAH MODEL NYATA ADA ATAU PAKAI SIMULASI
            if model is not None:
                # Inferensi menggunakan model asli (.h5)
                img_tensor = preprocess_image(image)
                predictions = model.predict(img_tensor)[0]
            else:
                # Simulasi/Dummy data sementara agar UI tetap bisa dites
                np.random.seed(42) # Agar nilainya stabil saat gambar di-upload
                dummy_raw = np.random.rand(len(CLASS_NAMES))
                predictions = dummy_raw / np.sum(dummy_raw) # Normalisasi ke probabilitas softmax
            
            top_index = np.argmax(predictions)
            predicted_class = CLASS_NAMES[top_index]
            confidence = predictions[top_index] * 100
            
            # Tampilan Hasil
            if model is None:
                st.info("ℹ️ Menampilkan hasil simulasi (Model belum dilatih).")
            
            st.success(f"**Identifikasi:** {predicted_class}")
            st.info(f"**Tingkat Keyakinan (Confidence):** {confidence:.2f}%")
            st.progress(int(confidence))
            
            # Grafik Probabilitas Top-3
            st.markdown("#### Detail Probabilitas")
            pred_df = pd.DataFrame({
                'Obat': CLASS_NAMES,
                'Probabilitas (%)': predictions * 100
            }).sort_values(by='Probabilitas (%)', ascending=False).head(3)
            
            st.bar_chart(pred_df.set_index('Obat'))