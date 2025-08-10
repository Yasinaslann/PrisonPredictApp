import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from datetime import datetime

# Sayfa yapılandırması
st.set_page_config(
    page_title="Yeniden Suç İşleme Tahmin Uygulaması",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
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
        except Exception:
            continue
    return None

df = load_data()

def home_page(df: pd.DataFrame | None):
    # Üst - Proje açıklaması ve veri seti hakkında
    st.markdown("""
    <div style="background-color:#0b1d51; padding:25px; border-radius:12px; margin-bottom: 25px;">
    <h1 style="color:#f0f2f6; margin-bottom: 0;">🏛️ Yeniden Suç İşleme Tahmin Uygulaması</h1>
    <p style="color:#ccd7ff; font-size:16px; line-height:1.5; margin-top: 0.5rem;">
    Bu uygulama, mahpusların tahliye sonrasında yeniden suç işleme riskini (recidivism) veri bilimi ve makine öğrenmesi teknikleri ile tahmin etmeyi amaçlar.<br>
    Amaç, topluma yeniden uyum sürecini iyileştirecek stratejiler geliştirmek ve risk analizi yaparak tekrar suç oranlarını azaltmaya katkı sağlamaktır.
    </p>
    <h3 style="color:#a7b7ff; margin-top:2rem; margin-bottom:0.5rem;">Veri Seti Hakkında</h3>
    <p style="color:#ccd7ff; font-size:14px; line-height:1.5;">
    Veri seti, mahpusların demografik bilgileri, ceza süreleri, geçmiş suç kayıtları ve yeniden suç işleme bilgilerini içermektedir.<br>
    Bu bilgilerle risk faktörleri analiz edilip, model geliştirme için zengin bir kaynak sağlanmıştır.
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Veri seti istatistikleri kartları (şık ve modern)
    if df is not None:
        total_records = len(df)
        total_columns = len(df.columns)
        unique_offenses = df['Prison_Offense'].nunique() if 'Prison_Offense' in df.columns else "Veri Yok"
        avg_age = f"{df['Age_at_Release'].dropna().astype(float).mean():.1f}" if 'Age_at_Release' in df.columns else "N/A"
        recid_rate = "N/A"
        if 'Recidivism_Within_3years' in df.columns:
            try:
                recid_rate = f"{df['Recidivism_Within_3years'].dropna().astype(float).mean() * 100:.2f}%"
            except Exception:
                recid_rate = "N/A"

        st.markdown("### 📊 Veri Seti Temel İstatistikler")
        cols = st.columns(5)
        cols[0].metric(label="🗂️ Toplam Kayıt", value=f"{total_records:,}")
        cols[1].metric(label="📋 Sütun Sayısı", value=f"{total_columns}")
        cols[2].metric(label="📌 Farklı Suç Tipi", value=f"{unique_offenses}")
        cols[3].metric(label="📅 Ortalama Yaş", value=avg_age)
        cols[4].metric(label="🎯 Ortalama Yeniden Suç Oranı", value=recid_rate)
    else:
        st.warning("Veri seti yüklenemedi. 'Prisongüncelveriseti.csv' dosyasını proje dizinine ekleyin.")

    st.markdown("---")

    # Veri seti önizlemesi açılır-kapanır
    if df is not None:
        with st.expander("📂 Veri Seti Önizlemesi (İlk 10 Satır)"):
            st.dataframe(df.head(10))

    st.markdown("---")

    # Grafik seçimleri sütunda, yan yana
    st.markdown("## 📈 Veri Seti Görselleştirmeleri")
    if df is None:
        st.info("Veri yüklenemediği için grafik gösterilemiyor.")
        return

    grafiker = st.columns(3)
    with grafiker[0]:
        chart_type_offense = st.selectbox("Suç Tipi Grafiği Türü", ["Bar Grafiği", "Pasta Grafiği"], key="offense")
    with grafiker[1]:
        chart_type_gender = st.selectbox("Cinsiyet Dağılımı Grafiği Türü", ["Bar Grafiği", "Pasta Grafiği"], key="gender")
    with grafiker[2]:
        chart_type_age = st.selectbox("Yaş Dağılımı Grafiği Türü", ["Histogram", "Box Plot"], key="age")

    # Suç tipi grafiği
    if 'Prison_Offense' in df.columns:
        offense_counts = df['Prison_Offense'].value_counts()
        if chart_type_offense == "Bar Grafiği":
            fig_offense = px.bar(
                x=offense_counts.index,
                y=offense_counts.values,
                labels={"x": "Suç Tipi", "y": "Kayıt Sayısı"},
                title="Suç Tipi Dağılımı"
            )
        else:  # Pasta Grafiği
            fig_offense = px.pie(
                values=offense_counts.values,
                names=offense_counts.index,
                title="Suç Tipi Dağılımı (Pasta Grafiği)"
            )
        st.plotly_chart(fig_offense, use_container_width=True)
    else:
        st.info("Suç tipi verisi bulunamadı.")

    # Cinsiyet grafiği
    if 'Gender' in df.columns:
        gender_counts = df['Gender'].value_counts()
        if chart_type_gender == "Bar Grafiği":
            fig_gender = px.bar(
                x=gender_counts.index,
                y=gender_counts.values,
                labels={"x": "Cinsiyet", "y": "Kayıt Sayısı"},
                title="Cinsiyet Dağılımı"
            )
        else:
            fig_gender = px.pie(
                values=gender_counts.values,
                names=gender_counts.index,
                title="Cinsiyet Dağılımı (Pasta Grafiği)"
            )
        st.plotly_chart(fig_gender, use_container_width=True)
    else:
        st.info("Cinsiyet verisi bulunamadı.")

    # Yaş dağılımı grafiği
    if 'Age_at_Release' in df.columns:
        try:
            df['Age_at_Release'] = pd.to_numeric(df['Age_at_Release'], errors='coerce')
            if chart_type_age == "Histogram":
                fig_age = px.histogram(
                    df,
                    x='Age_at_Release',
                    nbins=20,
                    labels={"Age_at_Release": "Çıkış Yaşı"},
                    title="Yaş Dağılımı (Histogram)"
                )
            else:
                fig_age = px.box(
                    df,
                    y='Age_at_Release',
                    labels={"Age_at_Release": "Çıkış Yaşı"},
                    title="Yaş Dağılımı (Box Plot)"
                )
            st.plotly_chart(fig_age, use_container_width=True)
        except Exception:
            st.info("Yaş verisi işlenirken hata oluştu.")
    else:
        st.info("Yaş verisi bulunamadı.")


def placeholder_page(name: str):
    st.title(name)
    st.info("Bu sayfa henüz hazırlanmadı.")


# Sidebar
st.sidebar.title("Navigasyon")
page = st.sidebar.radio(
    "Sayfa seçin",
    ("Ana Sayfa", "Tahmin Modeli", "Tavsiye ve Profil Analizi", "Model Analizleri ve Harita")
)

if page == "Ana Sayfa":
    home_page(df)
elif page == "Tahmin Modeli":
    placeholder_page("📊 Tahmin Modeli (Hazırlanıyor)")
elif page == "Tavsiye ve Profil Analizi":
    placeholder_page("💡 Tavsiye ve Profil Analizi (Hazırlanıyor)")
elif page == "Model Analizleri ve Harita":
    placeholder_page("📈 Model Analizleri ve Harita (Hazırlanıyor)")
