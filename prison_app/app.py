# prison_app/app.py
import streamlit as st
import pandas as pd
from pathlib import Path
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

@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame | None:
    for p in CANDIDATE_PATHS:
        try:
            if p.exists():
                df = pd.read_csv(p)
                return df
        except Exception as e:
            st.warning(f"Veri yüklenirken hata oluştu: {e}")
    return None

df = load_data()

def create_demo_data() -> pd.DataFrame:
    demo = pd.DataFrame({
        "suç_tipi": ["hırsızlık", "dolandırıcılık", "yaralama", "hırsızlık", "uyuşturucu", "dolandırıcılık", "dolandırıcılık"],
        "ceza_ay": [6, 12, 24, 3, 18, 9, 6],
        "egitim_durumu": ["lise", "ilkokul", "lise", "lise", "üniversite", "lise", "ilkokul"],
        "gecmis_suc_sayisi": [0, 2, 1, 0, 3, 1, 2],
        "il": ["Istanbul", "Ankara", "Izmir", "Istanbul", "Bursa", "Ankara", "Izmir"],
        "Recidivism_Within_3years": [0, 1, 0, 0, 1, 0, 1]
    })
    return demo

def home_page():
    st.title("🏛️ Yeniden Suç İşleme Tahmin Uygulaması")
    
    st.markdown(
        """
        ## Projenin Amacı ve Hikayesi

        Modern toplumlarda suç ve ceza kavramları, bireylerin ve toplumların güvenliği için büyük önem taşır.  
        Ancak hapishaneden tahliye edilen mahpusların, topluma tekrar suç işleyerek dönme riski (recidivism) önemli bir sosyal sorundur.  

        Bu proje, mahpusların tahliye sonrası yeniden suç işleme olasılıklarını **veri bilimi ve makine öğrenmesi teknikleri** ile analiz etmeyi ve tahmin etmeyi hedefler.  
        Amaç, bu riskleri önceden belirleyerek, rehabilitasyon süreçlerini geliştirmek ve toplumsal yeniden entegrasyon süreçlerine katkı sağlamaktır.

        ## Veri Seti Hakkında

        Kullanılan veri seti, Türkiye’deki mahpusların çeşitli demografik, suç geçmişi ve ceza bilgilerini içermektedir.  
        Veri setinde yer alan bazı temel değişkenler şunlardır:  

        - **Suç Tipi (suç_tipi):** Mahpusların işlediği suçların kategorileri  
        - **Ceza Süresi (ceza_ay):** Hapis cezasının ay cinsinden uzunluğu  
        - **Eğitim Durumu (egitim_durumu):** Mahpusların eğitim seviyeleri  
        - **Geçmiş Suç Sayısı (gecmis_suc_sayisi):** Daha önce işlenen suçların sayısı  
        - **İl (il):** Mahpusun cezaevinin bulunduğu şehir veya bölge  
        - **Yeniden Suç İşleme (Recidivism_Within_3years):** Tahliye sonrası 3 yıl içinde yeniden suç işleyip işlemediği (1=Evet, 0=Hayır)  

        Veri seti, bu tür değişkenler üzerinden modelleme ve analizlere imkan verir.  
        Elinizde `Prisongüncelveriseti.csv` dosyası yoksa, demo veri seti kullanılacaktır.

        """
    )

    st.markdown("---")

    if df is None:
        st.warning(
            """
            **Veri seti bulunamadı veya yüklenemedi.**  
            `Prisongüncelveriseti.csv` dosyasını proje dizinine ekleyerek gerçek verilerle çalışabilirsiniz.  
            Aksi halde demo veri gösterilecektir.
            """
        )
        data_to_show = create_demo_data()
    else:
        data_to_show = df

    st.subheader("📂 Veri Seti Önizlemesi (İlk 10 Satır)")
    st.dataframe(data_to_show.head(10))

    st.markdown("---")
    st.caption(f"📂 Repo: https://github.com/Yasinaslann/PrisonPredictApp • {APP_VERSION}")

# -------------------------
# Placeholder sayfalar
# -------------------------
def placeholder_page(name: str):
    st.title(name)
    st.info("Bu sayfa henüz hazırlanmadı. 'Ana Sayfa' tasarımını onayladıktan sonra aynı kalite/formatta bu sayfayı da oluşturacağım.")

# -------------------------
# Sidebar navigasyon
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
