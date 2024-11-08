import os
import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model, save_model
from sklearn.preprocessing import MinMaxScaler
import tempfile

# Fungsi untuk menyimpan scaler
@st.cache
def save_scaler(data):
    scaler = MinMaxScaler()
    scaler.fit(data)

    # Menggunakan direktori sementara
    temp_dir = tempfile.gettempdir()
    scaler_path = os.path.join(temp_dir, 'scaler.pkl')

    # Simpan scaler
    with open(scaler_path, 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)

    return scaler_path  # Kembalikan jalur scaler yang disimpan

# Fungsi untuk menyimpan model LSTM
def save_lstm_model(model):
    temp_dir = tempfile.gettempdir()
    model_path = os.path.join(temp_dir, 'lstm_model.h5')
    save_model(model, model_path)
    return model_path

# Fungsi untuk memuat model LSTM, SVM, dan Scaler
@st.cache
def load_models():
    lstm_model = None
    svm_classifier = None
    scaler = None
    error_message = None

    try:
        lstm_model = load_model(os.path.join(tempfile.gettempdir(), 'lstm_model.h5'))
    except FileNotFoundError as e:
        error_message = f"Error loading LSTM model: {e}"
    
    try:
        with open(os.path.join(tempfile.gettempdir(), 'svm_classifier.pkl'), 'rb') as svm_file:
            svm_classifier = pickle.load(svm_file)
    except FileNotFoundError as e:
        error_message = f"Error loading SVM model: {e}"
    
    try:
        # Memuat scaler dari jalur sementara
        scaler_path = os.path.join(tempfile.gettempdir(), 'scaler.pkl')
        with open(scaler_path, 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
    except FileNotFoundError as e:
        error_message = f"Error loading scaler: {e}"

    return lstm_model, svm_classifier, scaler, error_message

# Menyimpan scaler jika diperlukan (gunakan data Anda sendiri untuk fit scaler)
data = np.random.rand(100, 6)  # Contoh data, ganti dengan data aktual Anda
scaler_path = save_scaler(data)

# Tampilkan pesan sukses setelah menyimpan scaler
st.success("Scaler saved successfully!")

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
    prediction = svm_classifier