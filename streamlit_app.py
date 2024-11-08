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
    except FileNotFoundError as