# 💻 Web Interface - GUI Klasifikasi Obat

File ini mendokumentasikan bagian Antarmuka Pengguna (GUI) dari proyek Skripsi Klasifikasi Citra Obat. Antarmuka web ini dibangun untuk mendemonstrasikan secara langsung kemampuan model AI dalam mengenali gambar obat dan vitamin secara interaktif.

## 📑 Daftar Isi
- [Teknologi yang Digunakan](#teknologi-yang-digunakan)
- [Persiapan & Instalasi](#persiapan--instalasi)
- [Cara Menjalankan Aplikasi](#cara-menjalankan-aplikasi)
- [Panduan Penggunaan GUI](#panduan-penggunaan-gui)
- [Troubleshooting](#troubleshooting)

---

## 🛠️ Teknologi yang Digunakan
- **Frontend & Web Server:** [Streamlit](https://streamlit.io/) (Framework Python untuk Web GUI cepat)
- **Image Processing:** PIL (Pillow)
- **Machine Learning Engine:** TensorFlow / Keras (Membaca file `.h5`)
- **Manipulasi Array:** NumPy

---

## ⚙️ Persiapan & Instalasi
Aplikasi web ini membutuhkan pustaka tambahan di luar *core machine learning*. Pastikan Anda berada di dalam *virtual environment* (jika menggunakan) dan jalankan:

```bash
# Instalasi Streamlit dan pustaka pendukung
pip install streamlit Pillow numpy tensorflow
```

**Syarat Penting:**
Pastikan Anda sudah memiliki minimal satu model AI yang telah dilatih dan disimpan di dalam folder `models/` (misalnya: `models/scenario_a.h5`). Jika belum, silakan merujuk ke instruksi pelatihan model pada `readmeai.md`.

---

## 🚀 Cara Menjalankan Aplikasi
1. Buka terminal atau command prompt di editor kode Anda (VSCode/Windsurf/Cursor).
2. Pastikan posisi terminal berada di root direktori proyek (sejajar dengan file `app.py`).
3. Jalankan perintah server Streamlit:

```bash
streamlit run app.py
```

4. Sebuah tab baru akan otomatis terbuka di browser default Anda dengan alamat lokal:

```
http://localhost:8501
```

---

## 📱 Panduan Penggunaan GUI
Antarmuka web dirancang agar sangat intuitif dan mudah digunakan:

1. **Area Unggah**: Pada halaman utama, terdapat area drag-and-drop. Tarik gambar obat atau vitamin dari komputer Anda, atau klik untuk memilih file melalui Windows Explorer.
2. **Format Didukung**: Pastikan format gambar yang diunggah adalah `.jpg`, `.jpeg`, atau `.png`.
3. **Pratinjau**: Setelah gambar berhasil diunggah, sistem akan menampilkannya di sisi kiri layar.
4. **Proses Klasifikasi**: Sistem secara otomatis akan memproses gambar, melakukan normalisasi, dan mengirimkannya ke model AI.
5. **Hasil (Output)**: Di sisi kanan layar, sistem akan menampilkan:
   - **Nama Kelas**: Prediksi jenis obat/vitamin (misal: "Vitamin C").
   - **Confidence Score**: Persentase tingkat keyakinan AI terhadap tebakannya (dalam bentuk bar progres).

---

## 🔧 Troubleshooting
- **Error "Model file not found"**: Sistem web gagal memuat model. Periksa kembali isi file `app.py` pada bagian pemuatan model, pastikan path direktori menunjuk tepat ke `models/scenario_a.h5` (huruf besar/kecil sangat berpengaruh).
- **Error "Port 8501 is already in use"**: Port server bertabrakan. Jalankan Streamlit dengan port berbeda: `streamlit run app.py --server.port 8502`.