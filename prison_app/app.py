import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# --- Sayfa Yapılandırması ve Temel Ayarlar ---
st.set_page_config(
    page_title="Yeniden Suç İşleme Tahmin Uygulaması",
    page_icon="⚖️",
    layout="wide",
)

# --- Veri Yükleme ---
@st.cache_data
def load_data():
    """Veri setini güvenli bir şekilde yükler."""
    data_path = Path(__file__).parent / "Prisongüncelveriseti.csv"
    if data_path.exists():
        df = pd.read_csv(data_path)
        # NaN değerlerini doldurma stratejisi
        df.fillna(method="ffill", inplace=True) # Örneğin, boş değerleri bir önceki değerle doldur
        return df
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
        <div style="
            background-color: #1e3a5f;
            color: white;
            padding: 1.8rem 2rem;
            border-radius: 15px;
            box-shadow: 0 6px 15px rgba(0,0,0,0.4);
        ">
            <h1 style="margin-bottom: 0.3rem; font-size: 2.5rem;">⚖️ Yeniden Suç İşleme Tahmin Uygulaması</h1>
            <p style="line-height:1.5; font-size:1.1rem; color:#d1e0e8;">
                Bu uygulama, mahpusların tahliye sonrası **yeniden suç işleme riskini (recidivism)** analiz eder ve görselleştirir.
            </p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("---")

    # --- Kilit İstatistikler (Metrikler) ---
    st.subheader("📊 Genel İstatistikler")
    recid_col = get_column_if_exists(df, ["Recidivism", "recidivism_status", "recidivism"])

    total_rows = df.shape[0]
    total_cols = df.shape[1]
    unique_offenses = df["Prison_Offense"].nunique() if "Prison_Offense" in df.columns else "N/A"
    avg_sentence = safe_mean(df["Sentence_Length_Months"])
    recid_rate = safe_mean(df[recid_col]) if recid_col else None
    
    # Ortalama Tahliye Yaşı
    avg_age = safe_mean(df["Age_at_Release"]) if "Age_at_Release" in df.columns else None

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Toplam Kayıt", f"{total_rows:,}")
    col2.metric("Sütun Sayısı", f"{total_cols}")
    col3.metric("Farklı Suç Tipi", f"{unique_offenses}")
    col4.metric("Ortalama Ceza Süresi", f"{avg_sentence:.1f} ay" if avg_sentence else "N/A")
    
    if recid_rate is not None:
        col5.metric("Yeniden Suç Oranı", f"{(recid_rate * 100):.1f}%")
    else:
        col5.metric("Yeniden Suç Oranı", "N/A")

    st.markdown("---")

    # --- Veri Seti Önizleme ---
    with st.expander("📂 **Veri Seti Önizlemesi (İlk 15 Satır)**"):
        st.markdown("<p style='font-size: 1.1rem;'>Veri setinin ilk 15 satırını daha modern bir görünümle inceleyin.</p>", unsafe_allow_html=True)
        st.dataframe(df.head(15), use_container_width=True, hide_index=True)

    st.markdown("---")

    # --- Grafik Bölümleri ---
    st.subheader("📊 Keşifsel Veri Analizi")

    if recid_col:
        # Yeniden Suç İşleme Pasta Grafiği
        st.markdown("#### Yeniden Suç İşleme Durumu")
        recid_counts = df[recid_col].value_counts().sort_index()
        labels = ["Tekrar Suç İşlemedi", "Tekrar Suç İşledi"]
        values = [recid_counts.get(0, 0), recid_counts.get(1, 0)]
        
        pie_fig = px.pie(names=labels, values=values,
                         color_discrete_sequence=["#1f77b4", "#d62728"],
                         title="Yeniden Suç İşleme Oranı")
        pie_fig.update_traces(textposition='inside', textinfo='percent+label', pull=[0, 0.1])
        pie_fig.update_layout(title_x=0.5, font=dict(family="Arial", size=14),
                              legend_title="Durum")
        st.plotly_chart(pie_fig, use_container_width=True)

    st.markdown("---")

    # Demografik Dağılım ve Recidivism Grafikleri
    st.markdown("#### Demografik Dağılım Analizi")
    demo_cols = ["Gender", "Education_Level"]
    cols = st.columns(len(demo_cols))
    for idx, col_name in enumerate(demo_cols):
        with cols[idx]:
            if col_name in df.columns and recid_col:
                # Dağılım Grafiği
                counts = df[col_name].value_counts().reset_index()
                counts.columns = [col_name, "Kişi Sayısı"]
                bar_fig_dist = px.bar(counts, x=col_name, y="Kişi Sayısı",
                                      title=f"{col_name.replace('_', ' ')} Dağılımı",
                                      color=col_name, color_discrete_sequence=px.colors.qualitative.Plotly)
                bar_fig_dist.update_layout(showlegend=False, title_x=0.5)
                st.plotly_chart(bar_fig_dist, use_container_width=True)

                # Orana Göre Grafik
                recid_means = df.groupby(col_name)[recid_col].mean().reset_index()
                recid_means.columns = [col_name, "Ortalama Yeniden Suç Oranı"]
                recid_fig = px.bar(recid_means, x=col_name, y="Ortalama Yeniden Suç Oranı",
                                   title=f"{col_name.replace('_', ' ')}'a Göre Yeniden Suç Oranı",
                                   color=col_name, color_discrete_sequence=px.colors.qualitative.Plotly)
                recid_fig.update_layout(showlegend=False, yaxis=dict(range=[0, 1]), title_x=0.5)
                st.plotly_chart(recid_fig, use_container_width=True)
            else:
                st.info(f"{col_name} veya Yeniden Suç İşleme verisi bulunamadı.")

    st.markdown("---")

    # --- Footer ---
    st.caption("✨ Uygulama: Yeniden Suç İşleme Tahmin Modeli")

def placeholder_page(name):
    """Geliştirilecek sayfalar için yer tutucu."""
    st.title(name)
    st.info("Bu sayfa şu anda geliştirme aşamasındadır. Lütfen daha sonra tekrar kontrol edin. ⏳")

def main():
    """Ana uygulama akışını yönetir."""
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
