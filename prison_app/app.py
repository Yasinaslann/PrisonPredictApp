# prison_app/app.py
import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime
import plotly.express as px

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
        "Prison_Offense": ["hırsızlık", "dolandırıcılık", "yaralama", "hırsızlık", "uyuşturucu", "dolandırıcılık", "dolandırıcılık"],
        "Prison_Years": [0.5, 1, 2, 0.25, 1.5, 0.75, 0.5],
        "Education_Level": ["lise", "ilkokul", "lise", "lise", "üniversite", "lise", "ilkokul"],
        "Num_Distinct_Arrest_Crime_Types": [0, 2, 1, 0, 3, 1, 2],
        "Recidivism_Within_3years": [0, 1, 0, 0, 1, 0, 1]
    })
    return demo

def show_basic_stats(df: pd.DataFrame):
    st.subheader("📊 Veri Seti Temel İstatistikler")

    col1, col2, col3, col4 = st.columns(4)
    try:
        total_records = df.shape[0]
        unique_crimes = df["suç_tipi"].nunique() if "suç_tipi" in df.columns else None
        avg_sentence = df["ceza_ay"].mean() if "ceza_ay" in df.columns else None

        recid_col_candidates = [c for c in df.columns if "recid" in c.lower()]
        recid_rate = None
        if recid_col_candidates:
            recid_col = recid_col_candidates[0]
            recid_rate = df[recid_col].dropna().astype(float).mean()

        col1.metric("🗂️ Toplam Kayıt", total_records)
        if unique_crimes is not None:
            col2.metric("📌 Farklı Suç Tipi", unique_crimes)
        else:
            col2.markdown("📌 Farklı Suç Tipi\n**Veri yok**")

        if avg_sentence is not None and not pd.isna(avg_sentence):
            col3.metric("⏳ Ortalama Ceza Süresi (ay)", round(avg_sentence, 2))
        else:
            col3.markdown("⏳ Ortalama Ceza Süresi (ay)\n**Veri yok**")

        if recid_rate is not None and not pd.isna(recid_rate):
            col4.metric("⚠️ Yeniden Suç İşleme Oranı", f"{recid_rate:.2%}")
        else:
            col4.markdown("⚠️ Yeniden Suç İşleme Oranı\n**Veri yok**")

    except Exception as e:
        st.error(f"İstatistikler hesaplanırken hata oluştu: {e}")

def plot_category_distribution(df: pd.DataFrame, col_name: str, title: str):
    counts = df[col_name].value_counts().reset_index()
    counts.columns = [col_name, "Sayısı"]
    fig = px.bar(counts, x=col_name, y="Sayısı", title=title)
    st.plotly_chart(fig, use_container_width=True)

def plot_histogram(df: pd.DataFrame, col_name: str, title: str):
    fig = px.histogram(df, x=col_name, nbins=20, title=title)
    st.plotly_chart(fig, use_container_width=True)

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

        - **Prison_Offense:** Mahpusların işlediği suçların kategorileri  
        - **Prison_Years:** Hapis cezasının yıl cinsinden uzunluğu  
        - **Education_Level:** Mahpusların eğitim seviyeleri  
        - **Num_Distinct_Arrest_Crime_Types:** Daha önce işlenen farklı suç türlerinin sayısı  
        - **Recidivism_Within_3years:** Tahliye sonrası 3 yıl içinde yeniden suç işleyip işlemediği (1=Evet, 0=Hayır)  

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
        data_to_show["ceza_ay"] = data_to_show["Prison_Years"] * 12
        data_to_show["gecmis_suc_sayisi"] = data_to_show["Num_Distinct_Arrest_Crime_Types"]
        data_to_show["suç_tipi"] = data_to_show["Prison_Offense"]
        data_to_show["egitim_durumu"] = data_to_show["Education_Level"]
    else:
        data_to_show = df.copy()
        data_to_show["ceza_ay"] = data_to_show["Prison_Years"] * 12
        data_to_show["gecmis_suc_sayisi"] = data_to_show["Num_Distinct_Arrest_Crime_Types"]
        data_to_show["suç_tipi"] = data_to_show["Prison_Offense"]
        data_to_show["egitim_durumu"] = data_to_show["Education_Level"]

    with st.expander("📂 Veri Seti Önizlemesi (İlk 10 Satır)"):
        st.dataframe(data_to_show.head(10))

    show_basic_stats(data_to_show)

    st.markdown("---")

    st.subheader("📈 Veri Seti Görselleştirmeleri")

    col1, col2 = st.columns(2)

    with col1:
        if "suç_tipi" in data_to_show.columns:
            plot_category_distribution(data_to_show, "suç_tipi", "Suç Tipi Dağılımı")
        else:
            st.info("Suç tipi verisi mevcut değil.")

        if "gecmis_suc_sayisi" in data_to_show.columns:
            plot_histogram(data_to_show, "gecmis_suc_sayisi", "Geçmiş Suç Sayısı Dağılımı")
        else:
            st.info("Geçmiş suç sayısı verisi mevcut değil.")

    with col2:
        st.info("📍 Bu veri setinde coğrafi (şehir veya bölge) bilgisi bulunmamaktadır, bu yüzden coğrafi dağılım grafiği gösterilemiyor.")

        if "ceza_ay" in data_to_show.columns:
            plot_histogram(data_to_show, "ceza_ay", "Ceza Süresi Dağılımı (Ay)")
        else:
            st.info("Ceza süresi verisi mevcut değil.")

    st.markdown("---")
    st.caption(f"📂 Repo: https://github.com/Yasinaslann/PrisonPredictApp • {APP_VERSION}")

def placeholder_page(name: str):
    st.title(name)
    st.info("Bu sayfa henüz hazırlanmadı. 'Ana Sayfa' tasarımını onayladıktan sonra aynı kalite/formatta bu sayfayı da oluşturacağım.")

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
