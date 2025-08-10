import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# --- Sayfa Yapılandırması ve Temel Ayarlar ---
# Streamlit'in kendi tema ayarlarını kullanarak daha temiz bir tasarım oluşturun
st.set_page_config(
    page_title="Yeniden Suç İşleme Tahmin Uygulaması",
    page_icon="⚖️",
    layout="wide",
)

# Koyu mod için özel CSS (isteğe bağlı, Streamlit teması yeterli olabilir)
st.markdown("""
    <style>
        .st-emotion-cache-18ni3l0.e1f1d6gn5 {
            color: #ffffff !important;
            background-color: #0d1b2a !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- Veri Yükleme ---
# @st.cache_data yerine @st.cache_resource kullanmak, büyük veri setleri için daha uygun olabilir
@st.cache_data(show_spinner=False)
def load_data():
    """Veri setini güvenli bir şekilde yükler."""
    data_path = Path(__file__).parent / "Prisongüncelveriseti.csv"
    if data_path.exists():
        return pd.read_csv(data_path)
    else:
        st.error("Veri dosyası (Prisongüncelveriseti.csv) bulunamadı. Lütfen dosyanın uygulamanın olduğu dizinde olduğundan emin olun.")
        return None

# --- Yardımcı Fonksiyonlar ---
def safe_mean(series):
    """Sayısal olmayan değerleri yok sayarak ortalama hesaplar."""
    return pd.to_numeric(series, errors='coerce').dropna().mean()

def get_column_if_exists(df, col_name_list):
    """Veri setinde, verilen liste içindeki ilk uygun sütunu bulur."""
    for col in col_name_list:
        if col in df.columns:
            return col
    return None

def main_page(df):
    """Ana sayfa içeriğini oluşturan fonksiyon."""
    if df is None:
        return

    # --- Başlık ve Açıklama Bölümü ---
    st.markdown("""
        <div style="background-color: #0d1b2a; color: white; padding: 1.8rem 2rem; border-radius: 15px; box-shadow: 0 6px 15px rgba(0,0,0,0.3);">
            <h1 style="margin-bottom: 0.3rem;">⚖️ Yeniden Suç İşleme Tahmin Uygulaması</h1>
            <p style="line-height:1.5; font-size:1.1rem; color:#90caf9;">
                Bu uygulama, mahpusların tahliye sonrası **yeniden suç işleme riskini (recidivism)** analiz eder.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("---")

    # --- Kilit İstatistikler (Metrikler) ---
    st.subheader("📊 Genel İstatistikler")
    recid_col = get_column_if_exists(df, ["Recidivism", "recidivism_status", "recidivism"])

    # Metrikleri hesapla
    total_rows = df.shape[0]
    total_cols = df.shape[1]
    unique_offenses = df["Prison_Offense"].nunique() if "Prison_Offense" in df.columns else "N/A"
    avg_sentence = safe_mean(df["Sentence_Length_Months"])
    recid_rate = safe_mean(df[recid_col]) if recid_col else None
    avg_age = safe_mean(df["Age_at_Release"])

    # st.metric ile daha şık kartlar oluştur
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    col1.metric("Toplam Kayıt", f"{total_rows:,} 🗂️")
    col2.metric("Sütun Sayısı", f"{total_cols} 📋")
    col3.metric("Farklı Suç Tipi", f"{unique_offenses} 📌")
    col4.metric("Ortalama Ceza Süresi", f"{avg_sentence:.1f} ay" if avg_sentence else "N/A")
    if recid_rate:
        col5.metric("Yeniden Suç Oranı", f"{(recid_rate * 100):.1f}%", delta=f"{((recid_rate - 0.5) * 100):.1f}%", delta_color="inverse")
    else:
        col5.metric("Yeniden Suç Oranı", "N/A")
    col6.metric("Ortalama Tahliye Yaşı", f"{avg_age:.1f}" if avg_age else "N/A")
    
    st.markdown("---")
    
    # --- Veri Seti Önizleme ---
    with st.expander("📂 Veri Seti Önizlemesi (İlk 10 Satır)"):
        st.dataframe(df.head(10))

    st.markdown("---")

    # --- Grafik Bölümleri ---
    if recid_col:
        # Yeniden Suç İşleme Pasta Grafiği
        st.subheader("🎯 Yeniden Suç İşleme Durumu")
        recid_counts = df[recid_col].value_counts().sort_index()
        labels = ["Tekrar Suç İşlemedi", "Tekrar Suç İşledi"]
        values = [recid_counts.get(0, 0), recid_counts.get(1, 0)]
        
        # Plotly grafiklerini daha modern hale getirin
        pie_fig = px.pie(names=labels, values=values, title="3 Yıl İçinde Yeniden Suç İşleme Oranı",
                         color_discrete_sequence=px.colors.qualitative.Pastel)
        pie_fig.update_traces(textposition='inside', textinfo='percent+label')
        pie_fig.update_layout(title_x=0.5, font=dict(family="Arial", size=14),
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(pie_fig, use_container_width=True)

    st.markdown("---")

    # Demografik Dağılım Grafikleri
    st.subheader("👥 Demografik Dağılımlar")
    demo_cols = ["Gender", "Education_Level", "Prison_Offense"]
    # Her bir demografik özellik için döngü oluştur
    for col_name in demo_cols:
        if col_name in df.columns:
            st.markdown(f"**{col_name.replace('_', ' ')} Dağılımı**")
            col1, col2 = st.columns(2)
            with col1:
                counts = df[col_name].value_counts()
                bar_fig = px.bar(x=counts.index, y=counts.values,
                                 title=f"{col_name.replace('_', ' ')}'a Göre Kişi Sayısı",
                                 color=counts.index, color_discrete_sequence=px.colors.qualitative.Vivid)
                bar_fig.update_layout(showlegend=False, title_x=0.5)
                st.plotly_chart(bar_fig, use_container_width=True)

            if recid_col:
                with col2:
                    recid_means = df.groupby(col_name)[recid_col].mean()
                    recid_fig = px.bar(x=recid_means.index, y=recid_means.values * 100,
                                       labels={"y": "Yeniden Suç Oranı (%)"},
                                       title=f"{col_name.replace('_', ' ')}'a Göre Yeniden Suç Oranı",
                                       color=recid_means.index, color_discrete_sequence=px.colors.qualitative.Vivid)
                    recid_fig.update_layout(showlegend=False, yaxis=dict(range=[0, 100]), title_x=0.5)
                    st.plotly_chart(recid_fig, use_container_width=True)
            st.markdown("---")

    # --- Footer ---
    st.caption("✨ Uygulama: Yeniden Suç İşleme Tahmin Modeli")

def placeholder_page(name):
    """Geliştirilecek sayfalar için yer tutucu."""
    st.title(name)
    st.info("Bu sayfa şu anda geliştirme aşamasındadır. Lütfen daha sonra tekrar kontrol edin. ⏳")

def main():
    """Ana uygulama akışını yönetir."""
    # Yan çubuk navigasyonu
    st.sidebar.title("Navigasyon")
    page = st.sidebar.radio(
        "Sayfa seçin",
        ("Ana Sayfa", "Tahmin Modeli", "Tavsiye ve Profil Analizi")
    )

    df = load_data()

    if page == "Ana Sayfa":
        main_page(df)
    elif page == "Tahmin Modeli":
        placeholder_page("📊 Tahmin Modeli")
    elif page == "Tavsiye ve Profil Analizi":
        placeholder_page("💡 Tavsiye ve Profil Analizi")

if __name__ == "__main__":
    main()
