import streamlit as st

def app():
    st.title("💡 Tavsiye ve Profil Analizi")
    st.write("""
    Burada kullanıcı profilinize göre öneriler alabilirsiniz.
    Şu an basit bir örnek metin var, isterseniz veriye bağlı tavsiye sistemi ekleyebiliriz.
    """)
    risk_score = st.slider("Risk Skoru", 0, 100, 50)
    if risk_score > 70:
        st.warning("Yüksek risk! Daha fazla rehabilitasyon programına ihtiyaç duyabilirsiniz.")
    elif risk_score > 40:
        st.info("Orta risk! Davranış iyileştirme programları faydalı olabilir.")
    else:
        st.success("Düşük risk! Mevcut programlar yeterli görünüyor.")
