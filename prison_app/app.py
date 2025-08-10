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
        except Exception as e:
            st.warning(f"Veri yüklenirken hata oluştu: {e}")
            continue
    return None

df = load_data()

# -------------------------
# Demo veri seti oluşturucu
# -------------------------
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

# -------------------------
# Grafik çizme fonksiyonu
# -------------------------
def plot_top_categories(df: pd.DataFrame, col_name: str, top_n: int = 10):
    counts = df[col_name].value_counts().nlargest(top_n).reset_index()
    counts.columns = [col_name, "sayı"]
    fig = px.bar(counts, x=col_name, y="sayı", title=f"En Sık {col_name} Türleri (Top {top_n})")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Ana Sayfa
# -------------------------
def home_page():
    st.title("🏛️ Yeniden Suç İşleme Tahmin Uygulaması")
    st.markdown(
        """
        ### Proje Amacı  
        Bu uygulama, **mahpusların tahliye sonrasında yeniden suç işleme riskini** (recidivism)  
        **veri bilimi ve makine öğrenmesi teknikleri** ile tahmin etmeyi amaçlar.  
        Amaç, topluma yeniden uyum sürecini iyileştirecek stratejiler geliştirmek ve  
        risk analizi yaparak tekrar suç oranlarını azaltmaya katkı sağlamaktır.
        """
    )
    st.markdown(
        """
        **📌 Bu sayfada bulacaklarınız:**  
        - Projenin kısa tanımı  
        - Veri seti hakkında genel bilgiler  
        - Hızlı istatistikler ve görselleştirmeler  
        - İleriye dönük adımlar  
        """
    )
    st.markdown("---")

    total_rows = df.shape[0] if df is not None else 0
    total_cols = df.shape[1] if df is not None else 0
    data_source = None
    for p in CANDIDATE_PATHS:
        if p.exists():
            data_source = str(p)
            break

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📄 Veri Satırı", total_rows)
    col2.metric("📊 Sütun Sayısı", total_cols)
    col3.metric("💾 Veri Kaynağı", data_source or "Bulunamadı")
    col4.metric("⏰ Güncelleme", datetime.now().strftime("%Y-%m-%d %H:%M"))

    st.markdown("---")

    if df is None:
        st.warning(
            """
            **Veri seti yüklenemedi.**  
            `Prisongüncelveriseti.csv` dosyasını aşağıdaki dizinlerden birine ekleyin:  
            - `prison_app/`  
            - `/mnt/data/`  
            Şimdilik örnek bir **demo veri seti** gösterilmektedir.
            """
        )
        demo = create_demo_data()
        with st.expander("📂 Demo Veri Önizlemesi (İlk 10 Satır)"):
            st.dataframe(demo.head(10))
        data_for_viz = demo
    else:
        with st.expander("📂 Veri Seti Önizlemesi (İlk 10 Satır) — " + (data_source or "")):
            st.dataframe(df.head(10))
        data_for_viz = df

    # Hedef değişken analizi
    target_candidates = [c for c in data_for_viz.columns if "recidiv" in c.lower() or "recid" in c.lower()]
    if target_candidates:
        target = target_candidates[0]
        try:
            recid_rate = data_for_viz[target].dropna().astype(float).mean()
            st.markdown(f"**🎯 Hedef Sütun:** `{target}` — Ortalama yeniden suç işleme oranı: **{recid_rate:.2%}**")
        except Exception:
            st.info(f"Hedef sütun `{target}` bulundu fakat oran hesaplanamadı.")
    else:
        st.info("Hedef sütun (recidivism) otomatik olarak tespit edilemedi.")

    # Görselleştirmeler
    crime_cols = [c for c in data_for_viz.columns if any(x in c.lower() for x in ("crime", "suç", "offense", "charge"))]
    region_cols = [c for c in data_for_viz.columns if any(x in c.lower() for x in ("il", "şehir", "city", "region"))]

    viz1, viz2 = st.columns(2)

    if crime_cols:
        with viz1:
            plot_top_categories(data_for_viz, crime_cols[0])
    else:
        with viz1:
            st.info("Suç tipi bilgisi bulunamadı.")

    if region_cols:
        with viz2:
            plot_top_categories(data_for_viz, region_cols[0])
    else:
        with viz2:
            st.info("Bölge bilgisi bulunamadı.")

    st.markdown("---")
    st.header("🚀 Nasıl İlerlenir?")
    st.markdown(
        """
        1. **Tahmin Modeli** sayfasına giderek bireysel kayıt ile test yapın.  
        2. Eğitilmiş model dosyanız varsa (`catboost_model.pkl`) proje dizinine ekleyin.  
        3. Model dosyanız yoksa, eğitim için özel bir **notebook** hazırlanabilir.  
        """
    )
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
