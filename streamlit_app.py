import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

@st.cache
def load_models():
    lstm_model = None
    svm_classifier = None
    scaler = None
    error_message = None

    try:
        lstm_model = load_model('/workspaces/blank-app/lstm_model.h5')
    except OSError as e:
        error_message = f"Error loading LSTM model: {e}"

    try:
        with open('/workspaces/blank-app/svm_classifier.pkl', 'rb') as svm_file:
            svm_classifier = pickle.load(svm_file)
    except Exception as e:
        error_message = f"Error loading SVM classifier: {e}"

    try:
        with open('/workspaces/blank-app/scaler.pkl', 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
    except Exception as e:
        error_message = f"Error loading scaler: {e}"

    return lstm_model, svm_classifier, scaler, error_message

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
# ... (Input pengguna dan logika prediksi tetap sama)