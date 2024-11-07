import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler

# Fungsi untuk menyimpan scaler
def save_scaler():
    data = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])  # Contoh data
    scaler = MinMaxScaler()
    scaler.fit(data)

    with open('/workspaces/blank-app/scaler.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)
    st.success("Scaler saved successfully!")

# Fungsi untuk memuat scaler
def load_scaler():
    scaler = None
    error_message = None

    try:
        with open('/workspaces/blank-app/scaler.pkl', 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
    except FileNotFoundError as e:
        error_message = f"Error loading scaler: {e}"

    return scaler, error_message

# Menyimpan scaler jika belum ada
if st.button("Save Scaler"):
    save_scaler()

# Memuat scaler
scaler, error_message = load_scaler()

# Jika ada pesan kesalahan, tampilkan dan hentikan eksekusi lebih lanjut
if error_message:
    st.error(error_message)
    st.stop()

# Judul aplikasi
st.title("Aplikasi Prediksi Intrusi Jaringan")

# Input dari pengguna
st.write("Masukkan nilai fitur untuk prediksi:")
# ... (Input pengguna dan logika prediksi tetap sama)
