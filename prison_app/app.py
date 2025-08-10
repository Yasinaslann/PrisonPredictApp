# app.py (Ana Sayfa)
import streamlit as st
import pandas as pd
from pathlib import Path
import requests
from io import BytesIO

# ---------- Sayfa yapılandırma ----------
st.set_page_config(
    page_title="🏛 Prison Predict App — Anasayfa",
    page_icon="🏛",
    layout="wide"
)

# ---------- Dosya yolları (esnek) ----------
# Eğer __file__ mevcutsa script dizinini, yoksa working dir al
BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
LOCAL_CSV = BASE_DIR / "Prisongüncelveriseti.csv"
GITHUB_RAW_CSV = "https://raw.githubusercontent.com/Yasinaslann/PrisonPredictApp/main/prison_app/Prisongüncelveriseti.csv"

# ---------- Yardımcı fonksiyonlar ----------
@st.cache_data
def load_csv_from_path(path: Path):
    return pd.read_csv(path)

def download_csv_from_github(url: str, save_path: Path):
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        save_path.write_bytes(resp.content)
        return True, f"İndirildi: {save_path}"
    except Exception as e:
        return False, str(e)

# ---------- Layout: Başlık & Banner ----------
st.title("🏛 Prison Predict App")
st.markdown("### 🔍 Tekrar Suç İşleme (Recidivism) Tahmin & Tavsiye Sistemi")
st.markdown(
    "Bu uygulama mahkumların yeniden suç işleme riskini tahmin eder, "
    "kişiselleştirilmiş öneriler üretir ve model/veri analizleri ile destek sağlar."
)

# görsel banner (remote görsel; deploy ortamında sorun olursa kaldırabilirsin)
st.image(
    "https://images.unsplash.com/photo-1530651781830-06c2a897cf4e?auto=format&fit=crop&w=1400&q=60",
    caption="Adalet, rehabilitasyon ve veri bilimi — amaç riskleri azaltmak",
    use_container_width=True
)

st.markdown("---")

# ---------- Sol panel: Dosya kontrol + upload ----------
with st.sidebar.expander("📁 Veri & Model Kontrolleri", expanded=True):
    st.markdown("**Veri yükleme seçenekleri:**")
    uploaded_csv = st.file_uploader("Yerel CSV yükle (opsiyonel)", type=["csv"])
    st.markdown("**Alternatif:** Eğer proje dizininde CSV yoksa GitHub'dan indirebilirsiniz.")
    if st.button("GitHub'dan Prisongüncelveriseti.csv indir / güncelle"):
        ok, msg = download_csv_from_github(GITHUB_RAW_CSV, LOCAL_CSV)
        if ok:
            st.success(msg)
        else:
            st.error(f"İndirme hatası: {msg}")
    st.markdown("---")
    st.write("İpucu: Eğer Streamlit Cloud'a deploy ettiysen repo otomatik klonlanır; dosyalar repoda ise local görünür.")
    st.markdown("**Hızlı yardım:** Eğer CSV görünmüyorsa `prison_app` dizininde dosyanın adının tam olarak `Prisongüncelveriseti.csv` olduğundan emin ol.")

# ---------- Veri yükleme mantığı ----------
df = pd.DataFrame()
load_error = None

# 1) Eğer kullanıcı upload ettiyse onu kullan
if uploaded_csv is not None:
    try:
        df = pd.read_csv(uploaded_csv)
        st.sidebar.success("CSV başarıyla upload edildi (runtime).")
    except Exception as e:
        load_error = f"Yüklenen CSV okunamadı: {e}"

# 2) Aksi halde local path'te var mı kontrol et
elif LOCAL_CSV.exists():
    try:
        df = load_csv_from_path(LOCAL_CSV)
    except Exception as e:
        load_error = f"Local CSV okunamadı: {e}"

# 3) Eğer yoksa kullanıcıya uyarı göster (indir butonu sidebar'da)
else:
    st.warning("⚠ Veri seti bulunamadı: `Prisongüncelveriseti.csv` proje dizininde değil.")
    st.info("Çözüm önerileri: (1) Repo'yu Streamlit Cloud'a doğru yükelediğinden emin ol; (2) sidebar'dan GitHub'dan indir butonuna bas; (3) yerel CSV'yi upload et.")
    st.markdown(f"- GitHub raw: [{GITHUB_RAW_CSV}]({GITHUB_RAW_CSV})")

# ---------- Veri önizleme ve meta bilgisi ----------
st.markdown("## 📊 Veri Seti Özeti")
if load_error:
    st.error(load_error)

if df.empty:
    st.info("Henüz veri yüklenmedi veya CSV boş.")
    # gösterilecek bekleme/dummy içerikler
    st.markdown("**Beklenen bazı örnek sütunlar (örnek şablon):**")
    st.write([
        "age", "crime_type", "sentence_years", "education_level",
        "prior_convictions", "region", "release_year"
    ])
    st.caption("Yukarıdaki sütun isimleri örnektir. Kendi CSV'nizde farklı sütunlar olabilir — `feature_names.pkl` ile eşleştirmeyi sonraki sayfalarda yapacağız.")
else:
    # güzel önizleme
    st.success(f"CSV başarıyla yüklendi — {df.shape[0]} satır, {df.shape[1]} sütun.")
    st.dataframe(df.head(10), use_container_width=True)

    # Basit özet istatistikler
    st.markdown("### Temel İstatistikler")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Satır sayısı", df.shape[0])
        st.metric("Sütun sayısı", df.shape[1])
    with col2:
        numeric_count = df.select_dtypes(include=['number']).shape[1]
        st.metric("Sayısal sütunlar", numeric_count)
    with col3:
        cat_count = df.select_dtypes(exclude=['number']).shape[1]
        st.metric("Kategorik sütunlar", cat_count)

    # örnek dağılım grafiği için seçilebilir sütun
    st.markdown("### Hızlı Dağılım Görselleştirme (Örnek)")
    # numeric kolon varsa ilkini seç
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if numeric_cols:
        sel = st.selectbox("Histogram için numeric sütun seç", options=numeric_cols, index=0, key="home_hist_col")
        fig = px.histogram(df, x=sel, nbins=30, title=f"{sel} dağılımı")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Histogram oluşturmak için sayısal sütun gereklidir.")

st.markdown("---")
st.header("🚀 Sonraki Adımlar")
st.markdown("""
1. Sol menüden **Tahmin Modeli** sayfasına git — bireysel tahmin yapabilir ve SHAP açıklamasını görebilirsin.  
2. **Tavsiye & Profil Analizi** sayfasında kişisel önerilere bak.  
3. **Analiz & Harita** sayfasında model performansı ve coğrafi dağılım çalışılacak.
""")

st.caption("Not: Eğer hala CSV bulunamıyor hatası alıyorsan, repo yolunu ve dosya adını tekrar kontrol et. Gerekirse bana deploy log'unu gönder, birlikte çözelim.")
