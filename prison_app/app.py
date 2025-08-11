import streamlit as st
import page_home
import page_prediction
import page_recommendation
import page_model_analysis

PAGES = {
    "Anasayfa": page_home,
    "Tahmin Modeli": page_prediction,
    "Tavsiye ve Profil Analizi": page_recommendation,
    "Model Analizleri ve Harita": page_model_analysis
}

st.sidebar.title("Menü")
selection = st.sidebar.radio("Sayfa Seçin", list(PAGES.keys()))
page = PAGES[selection]
page.app()
