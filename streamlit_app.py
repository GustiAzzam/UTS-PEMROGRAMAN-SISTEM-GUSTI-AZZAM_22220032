import streamlit as st
import pickle

@st.cache
def load_scaler():
    scaler = None
    error_message = None

    try:
        with open('/workspaces/blank-app/scaler.pkl', 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
    except FileNotFoundError as e:
        error_message = f"Error loading scaler: {e}"

    return scaler, error_message

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