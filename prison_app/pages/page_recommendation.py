import streamlit as st

def app():
    st.title("💡 Tavsiye ve Profil Analizi")
    st.write("""
    Bu bölümde kullanıcıya veri girişine göre öneriler ve profil analizi yapılır.
    """)
    
    # Basit örnek öneri
    st.subheader("Öneri Örneği")
    st.info("Ceza süresi yüksek olan mahkumlar için rehabilitasyon programı önerilir.")
