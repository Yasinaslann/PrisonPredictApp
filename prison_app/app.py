import streamlit as st
from pages import page_home, page_prediction, page_recommendation, page_model_analysis

PAGES = {
    "🏠 Anasayfa": page_home,
    "📊 Tahmin Modeli": page_prediction,
    "💡 Tavsiye ve Profil Analizi": page_recommendation,
    "📈 Model Analizleri ve Harita": page_model_analysis
}

st.set_page_config(page_title="Prison Predict App", layout="wide")

st.sidebar.title("Menü")
selection = st.sidebar.radio("Gitmek istediğiniz sayfayı seçin:", list(PAGES.keys()))

page = PAGES[selection]
page.app()
