import streamlit as st
import pandas as pd

def app():
    st.title("📈 Model Analizleri ve Harita")
    st.write("Model performansı ve verilerin harita üzerinde görselleştirilmesi burada yer alacak.")

    st.subheader("📊 Veri İstatistikleri")
    df = pd.read_csv("Prisonguncelveriseti.csv")
    st.write(df.describe())

    st.subheader("🗺️ Harita Görselleştirme")
    if "Latitude" in df.columns and "Longitude" in df.columns:
        st.map(df)
    else:
        st.info("Veri setinde konum bilgisi bulunamadı.")
