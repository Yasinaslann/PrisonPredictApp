import streamlit as st

def app():
    st.title("🏠 Anasayfa")
    st.write("""
    **Prison Predict App**'e hoş geldiniz!  
    Bu uygulama, cezaevi verilerini analiz ederek tahminler yapar, öneriler sunar ve modelin detaylı analizlerini gösterir.
    """)
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/f2/Prison_cells_in_Helsinki.jpg/640px-Prison_cells_in_Helsinki.jpg", caption="Temsili Cezaevi Görseli")
