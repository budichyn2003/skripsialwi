import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
# WAJIB IMPORT INI AGAR SINKRON DENGAN TRAINING
from tensorflow.keras.applications.efficientnet import preprocess_input

# 1. Load Model dengan Caching agar web tidak lemot saat refresh
@st.cache_resource
def load_model():
    # Arahkan ke folder models/ sesuai struktur VS Code Anda
    return tf.keras.models.load_model('models/scenario_c_bruteforce.h5')

model = load_model()

# 2. Daftar Kelas (Urutannya HARUS sama persis dengan saat training)
TARGET_CLASSES = [
    "Cataflam 50 mg", 
    "Cetirizin 10 mg", 
    "Pantoprazol Sandoz 40 mg",
    "Vitamin C Teva 500 mg", 
    "Aspirin Ultra 500 mg", 
    "Algoflex Forte Dolo 400 mg", 
    "Merckformin XR 1000 mg"
]

# 3. Fungsi Preprocessing dan Prediksi
def predict_image(image_file):
    # Buka gambar dan pastikan formatnya RGB (bukan RGBA/PNG transparan)
    image = Image.open(image_file).convert('RGB')
    
    # Resize wajib 224x224 (Syarat mutlak EfficientNet)
    image = image.resize((224, 224))
    
    # Ubah ke array dan tambahkan dimensi batch (dari 224,224,3 menjadi 1,224,224,3)
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    
    # NORMALISASI KHUSUS EFFICIENTNET (Ini rahasia akurasinya!)
    img_array = preprocess_input(img_array)
    
    # Lakukan tebakan
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100
    
    predicted_label = TARGET_CLASSES[predicted_class_index]
    
    return predicted_label, confidence

# ==========================================
# CONTOH PENERAPAN DI UI STREAMLIT
# ==========================================
st.title("💊 Klasifikasi Obat Tablet (EfficientNetB0)")
st.write("Akurasi Model: 85.71% (Skenario C - Brute Force)")

uploaded_file = st.file_uploader("Upload Foto Obat (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Tampilkan gambar yang diupload
    st.image(uploaded_file, caption="Gambar yang diunggah", use_column_width=True)
    
    if st.button("Prediksi Obat"):
        with st.spinner("Model sedang berpikir..."):
            label, akurasi = predict_image(uploaded_file)
            
            st.success(f"**Hasil Prediksi:** {label}")
            st.info(f"**Tingkat Kepercayaan:** {akurasi:.2f}%")