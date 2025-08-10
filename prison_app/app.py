# prison_app/app.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
from datetime import datetime

# -------------------------
# Sayfa genel yapılandırması
# -------------------------
st.set_page_config(
    page_title="Yeniden Suç İşleme Tahmin Uygulaması",
    page_icon="⚖️",
    layout="wide"
)

BASE = Path(__file__).parent
CANDIDATE_PATHS = [
    BASE / "Prisongüncelveriseti.csv",
    Path("/mnt/data/Prisongüncelveriseti.csv")
]

APP_VERSION = "v1.0 (Ana Sayfa)"

# -------------------------
# Veri yükleme (güvenli, cached)
# -------------------------
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame | None:
    """
    Veri setini bulup yükler. Eğer bulunamazsa None döner.
    Cached olduğu için tekrar tekrar diskten okuyup yavaşlamaz.
    """
    for p in CANDIDATE_PATHS:
        try:
            if p.exists():
                df = pd.read_csv(p)
                return df
        except Exception:
            # okuma hatası olsa diğer dosya yollarına bakar
            continue
    return None

df = load_data()

# -------------------------
# Ana Sayfa içeriği
# -------------------------
def home_page():
    st.title("⚖️ Yeniden Suç İşleme Tahmin Uygulaması")
    st.markdown(
        """
        Bu uygulama, mahpusların tahliye sonrasında yeniden suç işleme (recidivism) riskini
        tahmin etmeye yönelik bir proje için hazırlanmıştır.  
        Aşağıda uygulamanın kısa tanımı, veri önizlemesi ve hızlı analiz araçları yer almaktadır.
        """
    )

    # Row: kısa özet kartları
    col1, col2, col3, col4 = st.columns([2,2,2,2])
    total_rows = df.shape[0] if df is not None else 0
    total_cols = df.shape[1] if df is not None else 0
    data_source = None
    for p in CANDIDATE_PATHS:
        if p.exists():
            data_source = str(p)
            break

    col1.metric("Veri satırı", total_rows)
    col2.metric("Sütun sayısı", total_cols)
    col3.metric("Veri kaynağı", data_source or "Bulunamadı (demo çalışma)")
    col4.metric("Güncelleme", datetime.now().strftime("%Y-%m-%d %H:%M"))

    st.markdown("---")

    # Veri yoksa bilgilendirme ve demo notu
    if df is None:
        st.warning(
            """
            `Prisongüncelveriseti.csv` dosyası bulunamadı.  
            - Lütfen veri dosyasını `prison_app/` veya `/mnt/data/` dizinine koyun.  
            - Uygulamaya devam etmek için örnek demo verisi gösterilmektedir.
            """
        )

        demo = pd.DataFrame({
            "suç_tipi": ["hırsızlık", "dolandırıcılık", "yaralama", "hırsızlık"],
            "ceza_ay": [6, 12, 24, 3],
            "egitim_durumu": ["lise", "ilkokul", "lise", "lise"],
            "gecmis_suc_sayisi": [0, 2, 1, 0],
            "il": ["Istanbul", "Ankara", "Izmir", "Istanbul"],
            "Recidivism_Within_3years": [0, 1, 0, 0]
        })
        with st.expander("📊 Demo veri önizlemesi (ilk 10 satır)"):
            st.dataframe(demo.head(10))
    else:
        # Veri gösterimi
        with st.expander("📊 Veri seti önizlemesi (ilk 10 satır) — " + (data_source or "")):
            st.dataframe(df.head(10))

        # Hedef değişken (recidivism) tespiti
        target_candidates = [c for c in df.columns if "recidiv" in c.lower() or "recid" in c.lower()]
        if target_candidates:
            target = target_candidates[0]
            try:
                recid_rate = df[target].dropna().astype(float).mean()
                st.markdown(f"**Hedef sütun:** `{target}` — Ortalama yeniden suç işleme oranı: **{recid_rate:.2%}**")
            except Exception:
                st.info(f"Hedef sütun `{target}` bulundu fakat oran hesaplanamadı (veri tipi uygun değil).")
        else:
            st.info("Veri setinde otomatik tespit edilebilen bir 'recidivism' hedef sütunu bulunamadı.")

        # Hızlı görselleştirmeler (suç tipine göre, varsa)
        crime_cols = [c for c in df.columns if any(x in c.lower() for x in ("crime", "suç", "offense", "charge"))]
        region_cols = [c for c in df.columns if any(x in c.lower() for x in ("il", "sehir", "city", "region"))]

        viz_col1, viz_col2 = st.columns(2)
        if crime_cols:
            with viz_col1:
                top_col = crime_cols[0]
                top_counts = df[top_col].value_counts().nlargest(10).reset_index()
                top_counts.columns = [top_col, "sayi"]
                fig = px.bar(top_counts, x=top_col, y="sayi", title=f"En sık {top_col} (ilk 10)")
                st.plotly_chart(fig, use_container_width=True)
        else:
            with viz_col1:
                st.info("Veri setinde 'suç tipi' gibi bir sütun bulunamadı (suç tipi görselleştirmesi pasif).")

        if region_cols:
            with viz_col2:
                reg = region_cols[0]
                region_count = df[reg].value_counts().nlargest(10).reset_index()
                region_count.columns = [reg, "sayi"]
                fig2 = px.bar(region_count, x=reg, y="sayi", title=f"{reg} bazlı örnek yoğunluk (ilk 10)")
                st.plotly_chart(fig2, use_container_width=True)
        else:
            with viz_col2:
                st.info("Veri setinde 'il/sehir/region' gibi bölge sütunu bulunamadı (harita/konum pasif).")

    st.markdown("---")
    st.header("🔧 Nasıl İlerleyeceksiniz (Adımlar)")
    st.markdown("""
    1. **Tahmin Modeli** sayfasında bireysel kayıt girerek modelle test edilecek.  
    2. Eğer elinizde eğitilmiş `catboost_model.pkl` gibi dosyalar varsa proje dizinine koyun; sonraki sayfada yüklenecek.  
    3. Model dosyanız yoksa, ben sana model eğitme notebook'u hazırlayıp verebilirim.  
    """)
    st.info("Sıradaki adım: `Tahmin Modeli` sayfasını oluşturayım mı? Hazırsa 'Evet' yaz ve ben devam edeyim — yoksa home sayfasında değişiklik yapalım.")

    st.markdown("---")
    st.caption(f"Repo: https://github.com/Yasinaslann/PrisonPredictApp  •  {APP_VERSION}")

# -------------------------
# Basit placeholder sayfalar (şimdilik)
# -------------------------
def placeholder_page(name: str):
    st.title(name)
    st.info("Bu sayfa henüz hazırlanmadı. 'Ana Sayfa' tasarımını onayladıktan sonra aynı kalite/formatta bu sayfayı da oluşturacağım.")

# -------------------------
# Sol sidebar navigasyon
# -------------------------
st.sidebar.title("Navigasyon")
page = st.sidebar.radio(
    "Sayfa seçin",
    ("Ana Sayfa", "Tahmin Modeli", "Tavsiye ve Profil Analizi", "Model Analizleri ve Harita")
)

if page == "Ana Sayfa":
    home_page()
elif page == "Tahmin Modeli":
    placeholder_page("📊 Tahmin Modeli (Hazırlanıyor)")
elif page == "Tavsiye ve Profil Analizi":
    placeholder_page("💡 Tavsiye ve Profil Analizi (Hazırlanıyor)")
elif page == "Model Analizleri ve Harita":
    placeholder_page("📈 Model Analizleri ve Harita (Hazırlanıyor)")
