import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Fungsi untuk memuat model
@st.cache  # Ganti @st.cache_resource dengan @st.cache
def load_models():
    lstm_model = None
    svm_classifier = None
    scaler = None

    try:
        lstm_model = load_model('/workspaces/blank-app/lstm_model.h5')
    except OSError as e:
        return None, None, None, f"Error loading LSTM model: {e}"

    try:
        with open('/workspaces/blank-app/svm_classifier.pkl', 'rb') as svm_file:  # Tambahkan 'rb' untuk membuka file biner
            svm_classifier = pickle.load(svm_file)
    except Exception as e:
        return lstm_model, None, None, f"Error loading SVM classifier: {e}"

    try:
        with open('/workspaces/blank-app/scaler.pkl', 'rb') as scaler_file:  # Tambahkan 'rb' untuk membuka file biner
            scaler = pickle.load(scaler_file)
    except Exception as e:
        return lstm_model, svm_classifier, None, f"Error loading scaler: {e}"

    return lstm_model, svm_classifier, scaler, None

# Memuat model LSTM, SVM, dan Scaler
lstm_model, svm_classifier, scaler, error_message = load_models()

# Jika ada pesan kesalahan, tampilkan dan hentikan eksekusi lebih lanjut
if error_message:
    st.error(error_message)
    st.stop()

# Judul aplikasi
st.title("Aplikasi Prediksi Intrusi Jaringan")

# Input dari pengguna
st.write("Masukkan nilai fitur untuk prediksi:")

# Membuat form input untuk fitur sesuai dataset Anda
ts = st.number_input("Timestamp", min_value=0)
src_port = st.number_input("Source Port", min_value=0)
dst_port = st.number_input("Destination Port", min_value=0)
duration = st.number_input("Duration", min_value=0.0)
src_bytes = st.number_input("Source Bytes", min_value=0)
dst_bytes = st.number_input("Destination Bytes", min_value=0)
service = st.selectbox("Service", ["-", "http", "dns", "smtp", "ftp-data"])  # Pilihan contoh
label = st.selectbox("Label", [0, 1])  # Kategori label dalam dataset

# Menggabungkan input menjadi array sesuai fitur dalam dataset Anda
input_data = np.array([[ts, src_port, dst_port, duration, src_bytes, dst_bytes]])

# Tombol prediksi
if st.button("Prediksi"):
    # Skala input menggunakan scaler yang sudah dilatih
    input_scaled = scaler.transform(input_data)

    # Bentuk input untuk LSTM
    input_lstm = input_scaled.reshape((input_scaled.shape[0], 1, input_scaled.shape[1]))

    # Ekstrak fitur menggunakan model LSTM
    lstm_features = lstm_model.predict(input_lstm)

    # Prediksi menggunakan model SVM
    prediction = svm_classifier.predict(lstm_features)

    # Tampilkan hasil prediksi
    if prediction[0] == 1:
        st.write("Hasil Prediksi: Potensi Intrusi Teridentifikasi")
    else:
        st.write("Hasil Prediksi: Tidak Ada Potensi Intrusi")

# Membersihkan cache jika diperlukan
if st.button("Bersihkan Cache"):
    st.cache_resource.clear()