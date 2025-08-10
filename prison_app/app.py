import streamlit as st

from pages import page_home, page_prediction, page_recommendations, page_eda

PAGES = {
    "Anasayfa": page_home,
    "Tahmin": page_prediction,
    "Tavsiye & Profil": page_recommendations,
    "EDA & Harita": page_eda,
}

selection = st.sidebar.radio("Menü", list(PAGES.keys()))
PAGES[selection].render()

# ================== 1. ANA SAYFA ==================
if menu == "🏠 Ana Sayfa":
    st.title("🚔 Prison Recidivism Prediction App")
    st.markdown("""
    ### Proje Tanımı
    Bu uygulama, tahliye edilen mahkumların yeniden suç işleme olasılığını tahmin etmek ve
    profil analizi yapmak amacıyla geliştirilmiştir.
    
    **Kapsam:**
    - Veri seti: NIJ's Recidivism Challenge verileri
    - Hedef: Tahliye sonrası yeniden tutuklanma riskini tahmin etmek
    - Kullanılan Teknolojiler: Python, Streamlit, CatBoost, Pandas, Scikit-learn
    
    **Sayfa Yapısı:**
    1. **Ana Sayfa** – Proje tanımı, veri seti hikayesi, görseller ve açıklamalar
    2. **Suç Tekrarı Tahmini** – Kullanıcı verileri ile tahmin modeli
    3. **Tavsiye Sistemi ve Profil Analizi** – Tahmine dayalı öneriler
    4. **Veri Analizi & Harita** – Görselleştirilmiş analizler ve harita
    """)

    st.image("https://cdn.pixabay.com/photo/2016/04/27/22/08/prison-1354303_960_720.jpg", use_column_width=True)

# ================== 2. SUÇ TEKRARI TAHMİNİ ==================
elif menu == "🔮 Suç Tekrarı Tahmini":
    st.title("🔮 Suç Tekrarı Tahmini")
    st.write("Burada kullanıcıdan alınan bilgilerle CatBoost modeli üzerinden tahmin yapılacak.")

# ================== 3. TAVSİYE SİSTEMİ & PROFİL ANALİZİ ==================
elif menu == "📊 Tavsiye Sistemi ve Profil Analizi":
    st.title("📊 Tavsiye Sistemi ve Profil Analizi")
    st.write("Burada kullanıcı profiline göre tavsiyeler ve analizler sunulacak.")

# ================== 4. VERİ ANALİZİ & HARİTA ==================
elif menu == "📈 Veri Analizi & Harita":
    st.title("📈 Veri Analizi & Harita")
    st.write("Burada veri setine ait analizler, grafikler ve harita görselleri yer alacak.")

