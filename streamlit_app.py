import os
import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import tempfile

# Fungsi untuk menyimpan scaler
@st.cache_resource
def save_scaler(data):
    scaler = MinMaxScaler()
    scaler.fit(data)

    # Menggunakan direktori sementara
    temp_dir = tempfile.gettempdir()
    scaler_path = os.path.join(temp_dir, 'scaler.pkl')

    # Simpan scaler
    try:
        with open(scaler_path, 'wb') as scaler_file:
            pickle.dump(scaler, scaler_file)
    except Exception as e:
        st.error(f"Error saving scaler: {e}")
        return None

    return scaler_path  # Kembalikan jalur scaler yang disimpan

# Fungsi untuk memuat model LSTM, SVM, dan Scaler
@st.cache_resource
def load_models():
    lstm_model = None
    svm_classifier = None
    scaler = None
    error_message = None

    # Cek keberadaan file model LSTM
    lstm_model_path = '/workspaces/blank-app/lstm_model'  # Pastikan ini adalah direktori model
    if not os.path.exists(lstm_model_path):
        error_message = f"LSTM model directory not found at {lstm_model_path}"
        return lstm_model, svm_classifier, scaler, error_message

    try:
        lstm_model = tf.keras.models.load_model(lstm_model_path)
    except Exception as e:
        error_message = f"Error loading LSTM model: {e}"
    
    # Cek keberadaan file SVM
    svm_model_path = '/workspaces/blank-app/svm_classifier.pkl'
    if not os.path.exists(svm_model_path):
        error_message = f"SVM model file not found at {svm_model_path}"
        return lstm_model, svm_classifier, scaler, error_message

    try:
        with open(svm_model_path, 'rb') as svm_file:
            svm_classifier = pickle.load(svm_file)
    except Exception as e:
        error_message = f"Error loading SVM model: {e}"
    
    # Cek keberadaan file scaler
    scaler_path = os.path.join(tempfile.gettempdir(), 'scaler.pkl')
    if not os.path.exists(scaler_path):
        error_message = f"Scaler file not found at {scaler_path}"
        return lstm_model, svm_classifier, scaler, error_message

    try:
        with open(scaler_path, 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
    except Exception as e:
        error_message = f"Error loading scaler: {e}"

    return lstm_model, svm_classifier, scaler, error_message

# Menyimpan scaler jika diperlukan (gunakan data Anda sendiri untuk fit scaler)
data = np.random.rand(100, 6)  # Contoh data, ganti dengan data aktual Anda
scaler_path = save_scaler(data)

# Tampilkan pesan sukses setelah menyimpan scaler
if scaler_path:
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

#Tombol prediksi
if st.button("Prediksi"):
    if scaler:
        try:
            input_data_scaled = scaler.transform(input_data)
        except Exception as e:
            st.error(f"Error scaling input data: {e}")
            st.stop()
    else:
        st.error("Scaler tidak tersedia!")
        st.stop()

    # Prediksi menggunakan model LSTM
    if lstm_model:
        try:
            # LSTM biasanya membutuhkan input dengan dimensi (samples, timesteps, features)
            input_data_scaled = input_data_scaled.reshape((input_data_scaled.shape[0], 1, input_data_scaled.shape[1]))
            lstm_prediction = lstm_model.predict(input_data_scaled)
            st.write("Prediksi LSTM:", lstm_prediction)
        except Exception as e:
            st.error(f"Error predicting with LSTM model: {e}")
    
    else:
        st.error("Model LSTM tidak tersedia!")
    
    # Prediksi menggunakan model SVM
    if svm_classifier:
        try:
            svm_prediction = svm_classifier.predict(input_data_scaled)
            st.write("Prediksi SVM:", svm_prediction)
        except Exception as e:
            st.error(f"Error predicting with SVM model: {e}")
    else:
        st.error("Model SVM tidak tersedia!")
