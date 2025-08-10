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
    # Başlık ve ikon
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

    # Özet metrikler
    col1, col2, col3, col4 = st.columns(4)
    total_rows = df.shape[0] if df is not None else 0
    total_cols = df.shape[1] if df is not None else 0
    data_source = None
    for p in CANDIDATE_PATHS:
        if p.exists():
            data_source = str(p)
            break

    col1.metric("📄 Veri Satırı", total_rows)
    col2.metric("📊 Sütun Sayısı", total_cols)
    col3.metric("💾 Veri Kaynağı", data_source or "Bulunamadı")
    col4.metric("⏰ Güncelleme", datetime.now().strftime("%Y-%m-%d %H:%M"))

    st.markdown("---")

    # Veri seti yoksa uyarı + demo
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
        demo = pd.DataFrame({
            "suç_tipi": ["hırsızlık", "dolandırıcılık", "yaralama", "hırsızlık"],
            "ceza_ay": [6, 12, 24, 3],
            "egitim_durumu": ["lise", "ilkokul", "lise", "lise"],
            "gecmis_suc_sayisi": [0, 2, 1, 0],
            "il": ["Istanbul", "Ankara", "Izmir", "Istanbul"],
            "Recidivism_Within_3years": [0, 1, 0, 0]
        })
        with st.expander("📂 Demo Veri Önizlemesi (İlk 10 Satır)"):
            st.dataframe(demo.head(10))
    else:
        # Veri önizleme
        with st.expander("📂 Veri Seti Önizlemesi (İlk 10 Satır) — " + (data_source or "")):
            st.dataframe(df.head(10))

        # Hedef değişken analizi
        target_candidates = [c for c in df.columns if "recidiv" in c.lower() or "recid" in c.lower()]
        if target_candidates:
            target = target_candidates[0]
            try:
                recid_rate = df[target].dropna().astype(float).mean()
                st.markdown(f"**🎯 Hedef Sütun:** `{target}` — Ortalama yeniden suç işleme oranı: **{recid_rate:.2%}**")
            except Exception:
                st.info(f"Hedef sütun `{target}` bulundu fakat oran hesaplanamadı.")
        else:
            st.info("Hedef sütun (recidivism) otomatik olarak tespit edilemedi.")

        # Görselleştirmeler
        crime_cols = [c for c in df.columns if any(x in c.lower() for x in ("crime", "suç", "offense", "charge"))]
        region_cols = [c for c in df.columns if any(x in c.lower() for x in ("il", "şehir", "city", "region"))]

        viz1, viz2 = st.columns(2)

        if crime_cols:
            with viz1:
                top_col = crime_cols[0]
                top_counts = df[top_col].value_counts().nlargest(10).reset_index()
                top_counts.columns = [top_col, "sayı"]
                fig = px.bar(top_counts, x=top_col, y="sayı", title=f"En Sık {top_col} Türleri (Top 10)")
                st.plotly_chart(fig, use_container_width=True)
        else:
            with viz1:
                st.info("Suç tipi bilgisi bulunamadı.")

        if region_cols:
            with viz2:
                reg = region_cols[0]
                region_count = df[reg].value_counts().nlargest(10).reset_index()
                region_count.columns = [reg, "sayı"]
                fig2 = px.bar(region_count, x=reg, y="sayı", title=f"{reg} Bazlı Yoğunluk (Top 10)")
                st.plotly_chart(fig2, use_container_width=True)
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

