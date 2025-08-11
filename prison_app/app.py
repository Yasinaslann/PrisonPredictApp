import streamlit as st
import pandas as pd
from pathlib import Path

# Genel sayfa ayarları
st.set_page_config(
    page_title="Yeniden Suç İşleme Tahmin Uygulaması",
    page_icon="⚖️",
    layout="wide",
)

BASE = Path(__file__).parent

# Veri yükleme fonksiyonu
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(BASE / "Prisonguncelveriseti.csv")
        return df
    except FileNotFoundError:
        st.error("❌ Veri seti bulunamadı.")
        return None

# Ana sayfa içeriği
st.markdown("""
# 🏛️ Yeniden Suç İşleme Tahmin Uygulaması

Bu uygulama, tahliye sonrası yeniden suç işleme riskini tahmin etmek ve analizler yapmak için geliştirilmiştir.  
Yan menüden ilgili sayfalara geçebilirsiniz:

- 📊 **Tahmin Modeli**  
- 💡 **Tavsiye ve Profil Analizi**  
- 📈 **Model Analizleri ve Harita**  
""")

# Veri setini önizleme
df = load_data()
if df is not None:
    st.write("📂 Veri seti önizleme:", df.head())
