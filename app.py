import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# 1. Konfigurasi Halaman Web (Harus paling atas)
st.set_page_config(
    page_title="AI Klasifikasi Obat",
    page_icon="💊",
    layout="centered"
)

# # 2. Fungsi untuk memuat model (di-cache agar tidak berat)
# @st.cache_resource
# def load_classification_model():
#     # Pastikan path mengarah ke model terbaikmu
#     model = tf.keras.models.load_model('models/scenario_a.h5')
#     return model

# model = load_classification_model()

# 2. Fungsi untuk memuat model (di-cache agar tidak berat)
@st.cache_resource
def load_classification_model():
    try:
        # Mencoba memuat model
        model = tf.keras.models.load_model('models/scenario_a.h5')
        return model
    except OSError:
        # Jika file belum ada atau rusak, kembalikan None
        return None

model = load_classification_model()

# Tambahkan peringatan jika model belum ada
if model is None:
    st.warning("⚠️ File model ('models/scenario_a.h5') belum ditemukan atau masih kosong. Silakan lakukan training data terlebih dahulu.")

# Daftar kelas obat (Sesuaikan dengan list_nama_obat.txt kamu)
class_names = ['Obat A', 'Obat B', 'Vitamin C', 'Vitamin D'] 

# 3. Desain Header UI
st.title("💊 Klasifikasi Obat & Vitamin")
st.markdown("""
    Selamat datang di sistem klasifikasi citra obat! 
    Silakan unggah foto obat (tablet/kapsul) untuk dianalisis oleh AI.
""")
st.divider()

# 4. Fitur Upload Gambar
uploaded_file = st.file_uploader("Pilih atau Tarik Gambar Obat ke Sini", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar yang diupload
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Citra Input")
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang akan dianalisis", use_container_width=True)
    
    with col2:
        st.subheader("Hasil Klasifikasi")
        with st.spinner("AI sedang memproses..."):
            # Preprocessing gambar (Sesuaikan target_size dengan input modelmu, misal 224x224)
            img = image.resize((224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = np.expand_dims(img_array, 0) # Create a batch
            img_array = img_array / 255.0 # Normalisasi jika saat training dinormalisasi
            
            # Prediksi
            predictions = model.predict(img_array)
            predicted_class = class_names[np.argmax(predictions[0])]
            confidence = np.max(predictions[0]) * 100
            
            # Tampilkan Hasil yang Cantik
            st.success(f"**Prediksi Obat:** {predicted_class}")
            st.info(f"**Tingkat Keyakinan (Confidence):** {confidence:.2f}%")
            
            st.progress(int(confidence))