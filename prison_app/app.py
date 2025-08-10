import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px

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
def load_data():
    for p in CANDIDATE_PATHS:
        try:
            if p.exists():
                df = pd.read_csv(p)
                return df
        except:
            continue
    return None

def convert_sentence_length(val):
    if pd.isna(val):
        return None
    val = str(val).strip().lower()
    if val == "less than 1 year":
        return 0.5
    elif val == "1-2 years":
        return 1.5
    elif val == "greater than 2 to 3 years":
        return 2.5
    elif val == "more than 3 years":
        return 4
    else:
        try:
            return float(val)
        except:
            return None

def create_demo_data():
    demo = pd.DataFrame({
        "Prison_Offense": ["hırsızlık", "dolandırıcılık", "yaralama", "hırsızlık", "uyuşturucu", "dolandırıcılık"],
        "Prison_Years": ["Less than 1 year", "1-2 years", "More than 3 years", "1-2 years", "Less than 1 year", "More than 3 years"],
        "Num_Distinct_Arrest_Crime_Types": [0, 2, 1, 0, 3, 1],
        "Recidivism_Within_3years": [0, 1, 0, 0, 1, 0]
    })
    return demo

def plot_graph(df, column, grafik_tipi, title):
    if grafik_tipi == "Bar Grafiği":
        counts = df[column].value_counts().reset_index()
        counts.columns = [column, "Sayı"]
        fig = px.bar(counts, x=column, y="Sayı", title=title)
    elif grafik_tipi == "Histogram":
        fig = px.histogram(df, x=column, nbins=20, title=title)
    else:  # Box Plot
        fig = px.box(df, y=column, title=title)
    fig.update_layout(template="plotly_white", title_x=0.5)
    return fig

def home_page(df):
    st.title("🏛️ Yeniden Suç İşleme Tahmin Uygulaması")

    st.markdown("""
    ### Proje Amacı

    Tahliye sonrası mahpusların yeniden suç işleme riskini analiz ederek,  
    toplum güvenliğini artırmak ve bireylerin rehabilitasyon süreçlerini desteklemek amacıyla geliştirilmiş bir platformdur.  
    Veri bilimi ve makine öğrenmesi yöntemleri kullanılarak, riskli durumların önceden tespiti ve etkili müdahaleler sağlanması hedeflenmektedir.

    ### Veri Seti

    Veri seti, mahpusların demografik bilgileri, ceza süreleri, suç tipleri ve geçmiş suç kayıtlarını içermektedir.  
    Bu kapsamlı veri, suçun tekrarlanma olasılığını etkileyen faktörlerin derinlemesine incelenmesine olanak tanır.
    """)

    if df is None:
        st.warning("Veri seti bulunamadı. Örnek demo veri gösterilmektedir.")
        df = create_demo_data()

    df["Prison_Years_Numeric"] = df["Prison_Years"].apply(convert_sentence_length)

    with st.expander("📂 Veri Seti Önizlemesi (İlk 10 Satır)"):
        st.dataframe(df.head(10))

    st.subheader("📊 Temel İstatistikler")
    total_records = len(df)
    unique_crimes = df["Prison_Offense"].nunique() if "Prison_Offense" in df.columns else 0
    avg_sentence = df["Prison_Years_Numeric"].mean()
    recid_col = next((c for c in df.columns if "recid" in c.lower()), None)
    recid_rate = df[recid_col].mean() if recid_col else None

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🗂️ Toplam Kayıt", total_records)
    col2.metric("📌 Farklı Suç Tipi", unique_crimes)
    col3.metric("⏳ Ortalama Ceza Süresi (yıl)", f"{avg_sentence:.2f}" if avg_sentence else "Veri yok")
    col4.metric("⚠️ Yeniden Suç İşleme Oranı", f"{recid_rate:.2%}" if recid_rate else "Veri yok")

    st.markdown("""
    - **Toplam Kayıt:** Veri setindeki toplam birey sayısı.  
    - **Farklı Suç Tipi:** Suç kategorilerinin benzersiz sayısı.  
    - **Ortalama Ceza Süresi:** Yıllık bazda ortalama ceza süresi.  
    - **Yeniden Suç İşleme Oranı:** Tahliye sonrası 3 yıl içinde tekrar suç işleme oranı.
    """)

    st.markdown("---")
    st.subheader("📈 Veri Seti Görselleştirmeleri")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Suç Tipi Dağılımı")
        grafik_tipi_suc = st.selectbox(
            "Grafik Tipi:",
            ["Bar Grafiği", "Histogram", "Kutu Grafiği (Box Plot)"],
            key="grafik_suc",
            label_visibility="collapsed"
        )
        if "Prison_Offense" in df.columns:
            if grafik_tipi_suc == "Kutu Grafiği (Box Plot)":
                st.info("Suç tipi kategorik olduğundan kutu grafiği uygun değildir.")
            else:
                fig = plot_graph(df, "Prison_Offense", grafik_tipi_suc, "Suç Tipi Dağılımı")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Suç tipi verisi mevcut değil.")

        st.markdown("### Geçmiş Suç Sayısı Dağılımı")
        grafik_tipi_gecmis = st.selectbox(
            "Grafik Tipi:",
            ["Bar Grafiği", "Histogram", "Kutu Grafiği (Box Plot)"],
            key="grafik_gecmis",
            label_visibility="collapsed"
        )
        if "Num_Distinct_Arrest_Crime_Types" in df.columns:
            fig2 = plot_graph(df, "Num_Distinct_Arrest_Crime_Types", grafik_tipi_gecmis, "Geçmiş Suç Sayısı Dağılımı")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Geçmiş suç sayısı verisi mevcut değil.")

    with col2:
        st.markdown("### Ceza Süresi Dağılımı (Yıl)")
        grafik_tipi_ceza = st.selectbox(
            "Grafik Tipi:",
            ["Bar Grafiği", "Histogram", "Kutu Grafiği (Box Plot)"],
            key="grafik_ceza",
            label_visibility="collapsed"
        )
        if "Prison_Years_Numeric" in df.columns and df["Prison_Years_Numeric"].notnull().any():
            fig3 = plot_graph(df, "Prison_Years_Numeric", grafik_tipi_ceza, "Ceza Süresi Dağılımı (Yıl)")
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Ceza süresi verisi mevcut değil veya sayısal değil.")

    st.caption(f"📂 Repo: https://github.com/Yasinaslann/PrisonPredictApp • {APP_VERSION}")

def placeholder_page(name):
    st.title(name)
    st.info("Bu sayfa henüz hazırlanmadı. Ana sayfa hazırlandıktan sonra bu sayfa geliştirilecektir.")

def main():
    df = load_data()

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

if __name__ == "__main__":
    main()
