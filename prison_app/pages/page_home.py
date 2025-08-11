import streamlit as st

def app():
    st.title("🏛️ Prison Predict App")
    st.write("""
    Bu uygulama, suç ve ceza verileri üzerinden tahminler, analizler ve tavsiyeler sunar.
    Sol menüden sayfalar arasında geçiş yapabilirsiniz.
    """)
    st.image("https://cdn.pixabay.com/photo/2016/06/09/17/59/prison-1448678_1280.jpg", use_column_width=True)
