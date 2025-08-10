import streamlit as st
import pandas as pd
from pathlib import Path

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="🏛 Prison Predict App",
    page_icon="🏛",
    layout="wide"
)

# ----------------- HEADER -----------------
st.title("🏛 Prison Predict App")
st.markdown("""
### 🔍 Yapay Zeka ile Tekrar Suç İşleme Tahmini

Bu uygulama, **mahkumların tekrar suç işleme olasılığını** tahmin eden,
rehabilitasyon tavsiyeleri sunan ve suç istatistiklerini görselleştiren
**çok yönlü bir yapay zeka sistemi**dir.
""")

# ----------------- INTRO IMAGE / BANNER -----------------
st.image(
    "https://images.unsplash.com/photo-1530651781830-06c2a897cf4e",
    caption="Ceza sistemi ve yapay zeka ile suç önleme",
    use_column_width=True
)

st.markdown("---")

# ----------------- PROBLEM TANIMI -----------------
st.header("📌 Problem Tanımı")
st.markdown("""
Suç oranlarının yüksek olduğu bölgelerde, yeniden suç işleme olasılığı (recidivism) ciddi bir problemdir.
Mevcut sistemlerde **mahkumların yeniden topluma kazandırılması** için
kullanılan yöntemler çoğu zaman **genel** ve **kişiselleştirilmemiş** olmaktadır.

Bu proje ile:
- Mahkumun geçmiş verilerine göre **tekrar suç işleme olasılığı** tahmin edilecek.
- Kişisel risk faktörleri analiz edilerek **özel rehabilitasyon önerileri** sunulacak.
- Suç yoğunluğu haritaları oluşturularak **bölgesel analiz** yapılacak.
""")

# ----------------- DATASET HAKKINDA -----------------
st.header("📂 Veri Seti Hakkında")
st.markdown("""
**Kaynak:** `Prisongüncelveriseti.csv`  
Veri setinde mahkumlara ait geçmiş bilgiler, suç tipleri, demografik veriler ve sosyal durum bilgileri yer almaktadır.

**Örnek Sütunlar:**
- `age` → Mahkumun yaşı
- `crime_type` → İşlenen suç tipi
- `sentence_years` → Ceza süresi (yıl)
- `education_level` → Eğitim durumu
- `prior_convictions` → Önceki mahkumiyet sayısı
- `region` → Coğrafi bölge
- `release_year` → Tahliye yılı
""")

# Veri setini yükleme
data_path = Path("Prisongüncelveriseti.csv")
if data_path.exists():
    df = pd.read_csv(data_path)
    st.subheader("📊 Veri Önizleme")
    st.dataframe(df.head(10), use_container_width=True)

    st.info(f"Veri setinde **{df.shape[0]} satır** ve **{df.shape[1]} sütun** bulunuyor.")
else:
    st.warning("⚠ Veri seti bulunamadı. Lütfen `Prisongüncelveriseti.csv` dosyasını proje klasörüne ekleyin.")

# ----------------- PROJENİN AMACI -----------------
st.header("🎯 Projenin Hedefleri")
goals = [
    "1️⃣ **Tahmin Modeli** — CatBoost algoritması ile tekrar suç işleme olasılığını hesaplamak.",
    "2️⃣ **Tavsiye Sistemi** — Risk faktörlerine göre kişiselleştirilmiş öneriler sunmak.",
    "3️⃣ **Profil Analizi** — Mahkumun güçlü ve zayıf yönlerini analiz etmek.",
    "4️⃣ **Harita Analizi** — Bölgesel suç yoğunluklarını görselleştirmek."
]
for g in goals:
    st.markdown(g)

# ----------------- NASIL KULLANILIR -----------------
st.header("🛠 Nasıl Kullanılır?")
st.markdown("""
1. **Tahmin Modeli** sayfasına giderek kendi verinizi girin ve tahmin alın.
2. **Tavsiye ve Profil Analizi** sayfasında kişisel önerilerinizi inceleyin.
3. **Model Analizleri ve Harita** sayfasında genel istatistikleri keşfedin.
""")

st.success("🚀 Başlamak için sol menüden bir sayfa seçebilirsiniz.")
