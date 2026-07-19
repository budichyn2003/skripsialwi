# 🧠 Core AI - Klasifikasi Citra Obat & Vitamin

File ini mendokumentasikan bagian *Core Artificial Intelligence* (AI) dari proyek Skripsi Klasifikasi Citra Obat dan Vitamin. Fokus utama pada repositori ini adalah prapemrosesan gambar, perancangan arsitektur Convolutional Neural Network (CNN), serta skenario pelatihan (*training*) dan evaluasi model.

## 📑 Daftar Isi
1. [Prasyarat Sistem](#prasyarat-sistem)
2. [Instalasi & Persiapan Lingkungan (Virtual Environment)](#instalasi--persiapan-lingkungan-virtual-environment)
3. [Struktur Direktori AI](#struktur-direktori-ai)
4. [Persiapan Dataset](#persiapan-dataset)
5. [Skenario Pelatihan Model](#skenario-pelatihan-model)
6. [Evaluasi & Output](#evaluasi--output)

---

## 💻 Prasyarat Sistem
Pastikan sistem komputer Anda telah memenuhi persyaratan berikut:
- **Sistem Operasi**: Windows 10/11 (direkomendasikan) atau Linux/macOS.
- **Python**: Versi **3.10** (Sangat direkomendasikan untuk stabilitas TensorFlow/Keras).
- **Terminal/Command Line**: PowerShell atau Git Bash (untuk Windows).
- **Hardware**: GPU NVIDIA dengan CUDA support (Opsional namun sangat disarankan untuk mempercepat proses *training*), RAM minimal 8GB.

---

## ⚙️ Instalasi & Persiapan Lingkungan (Virtual Environment)
Langkah ini sangat penting untuk memastikan pustaka (library) yang digunakan dalam proyek ini tidak bentrok dengan proyek Python Anda yang lain. Kita akan menggunakan *Virtual Environment* (`venv`).

Buka terminal (PowerShell) di dalam folder root proyek Anda (`SKRIPSI-ALWI`), lalu jalankan perintah berikut secara berurutan:

1. **Buat Virtual Environment:**
   Jalankan perintah ini untuk membuat folder `venv` yang akan berisi lingkungan Python terisolasi:
   ```powershell
   python -m venv venv
   ```

2. **Aktivasi Virtual Environment:**
   Setelah folder `venv` terbentuk, Anda wajib mengaktifkannya setiap kali akan bekerja. 
   ```powershell
   . env\Scripts\Activate.ps1
   ```
   *(Catatan: Jika muncul error merah terkait "Execution_Policies" di PowerShell, jalankan perintah `Set-ExecutionPolicy Unrestricted -Scope CurrentUser` terlebih dahulu, lalu ulangi perintah aktivasi di atas).*
   
   Jika berhasil, terminal Anda akan memiliki awalan `(venv)` di sebelah kiri, contoh: `(venv) PS D:\Jokian\skripsi-alwi>`.

3. **Instalasi Dependensi (Requirements):**
   Setelah `(venv)` aktif, instal semua pustaka yang dibutuhkan:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 📁 Struktur Direktori AI
Bagian *machine learning* pada repositori ini diatur dengan struktur yang rapi:
- `data/` : Direktori utama untuk manajemen gambar.
  - `raw/` : Dataset asli sebelum augmentasi/pemisahan.
  - `split/` : Dataset yang telah dibagi (Training, Validation, Testing).
- `models/` : Lokasi penyimpanan model yang telah selesai dilatih dengan format HDF5 (`.h5`).
- `outputs/` : Menyimpan metrik pelatihan seperti grafik akurasi, *loss*, dan *confusion matrix*.
- `src/` : Kumpulan *source code* Python.
  - `config.py` : Pengaturan *hyperparameter* (batch size, epochs, learning rate).
  - `dataset.py` : Skrip untuk *data loader* dan augmentasi.
  - `architectures.py` : Definisi rancangan layer CNN.
  - `train_A.py`, `train_B.py`, `train_C.py` : Skrip eksekusi untuk berbagai skenario eksperimen.
- `venv/` : Lingkungan virtual Python (Tidak ikut di-push ke GitHub karena ada di `.gitignore`).

---

## 📊 Persiapan Dataset
1. Letakkan seluruh dataset citra obat Anda ke dalam folder `data/raw/` yang dikelompokkan berdasarkan nama kelas (misal: `data/raw/Obat_A/`, `data/raw/Vitamin_C/`).
2. Pastikan daftar nama kelas sesuai dengan yang tercantum di file `list_nama_obat.txt`.
3. Jalankan skrip pembagian data (misal: `python src/explore_dataset.py`) untuk menghasilkan rasio Train/Valid/Test secara proporsional di dalam folder `data/split/`.

---

## 🚀 Skenario Pelatihan Model
Proyek ini menguji beberapa skenario *hyperparameter* dan arsitektur untuk mendapatkan performa terbaik:

* **Skenario A (`train_A.py`)**: Pengujian arsitektur *baseline* dengan hyperparameter standar.
* **Skenario B (`train_B.py`)**: Pengujian dengan penambahan layer augmentasi ekstra dan *dropout* rasio tinggi.
* **Skenario C (`train_C.py`)**: Eksperimen tingkat lanjut dengan penyesuaian *learning rate scheduler*.

**Cara Menjalankan Training:**
Pastikan `(venv)` masih aktif, lalu eksekusi skrip sesuai skenario yang diinginkan:
```powershell
python src/train_A.py
```
*(Model akhir akan secara otomatis tersimpan di folder `models/scenario_a.h5` setelah epochs selesai)*.

---

## 📈 Evaluasi & Output
- **Grafik Performa:** Tersimpan di `outputs/scenario_x/` (Grafik Akurasi dan Loss per Epoch).
- **Model Final:** Disimpan dalam folder `models/`. Model dengan metrik terbaik akan digunakan oleh antarmuka web.
- **Eksperimen Interaktif:** Dapat dilihat melalui file Jupyter Notebook `cnn_obat_tablet.ipynb`.