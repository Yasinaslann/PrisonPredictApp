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

def home_page(df):
    st.title("🏛️ Yeniden Suç İşleme Tahmin Uygulaması")

    st.markdown("""
    ## Proje Amacı

    Bu uygulama, mahpusların tahliye sonrası topluma yeniden uyum süreçlerinde karşılaşabilecekleri riskleri  
    azaltmak amacıyla geliştirilmiştir. Yeniden suç işleme oranlarını analiz etmek ve tahmin etmek için gelişmiş  
    veri bilimi ve makine öğrenmesi teknikleri kullanılmaktadır. Böylece, riskli bireylerin tespiti sağlanarak,  
    uygun rehabilitasyon ve destek programlarının planlanmasına katkı sağlanır. Bu yaklaşım, toplum güvenliğinin  
    artırılması ve suçun tekrarlanma oranının azaltılması hedeflenmektedir.

    ## Veri Seti Hakkında

    Kullanılan veri seti, mahpusların demografik bilgileri, ceza süreleri, geçmişte işledikleri suç tipleri,  
    yeniden suç işleme durumu ve benzeri çeşitli özelliklerden oluşmaktadır. Veri seti, modelleme ve analizler için  
    zengin ve kapsamlı bir temel oluşturur. Bu sayede farklı özelliklerin yeniden suç işleme üzerindeki etkileri  
    incelenebilir.

    """)

    if df is None:
        st.warning("Veri seti bulunamadı, demo veri gösteriliyor.")
        df = create_demo_data()

    # Ceza süresini sayısal yap
    df["Prison_Years_Numeric"] = df["Prison_Years"].apply(convert_sentence_length)

    with st.expander("📂 Veri Seti Önizlemesi (İlk 10 Satır)"):
        st.dataframe(df.head(10))

    # Temel istatistikler
    st.subheader("📊 Temel İstatistikler")
    total_records = len(df)
    unique_crimes = df["Prison_Offense"].nunique() if "Prison_Offense" in df.columns else 0
    avg_sentence = df["Prison_Years_Numeric"].mean()
    recid_col = None
    for c in df.columns:
        if "recid" in c.lower():
            recid_col = c
            break
    recid_rate = df[recid_col].mean() if recid_col else None

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🗂️ Toplam Kayıt", total_records)
    col2.metric("📌 Farklı Suç Tipi", unique_crimes)
    col3.metric("⏳ Ortalama Ceza Süresi (yıl)", f"{avg_sentence:.2f}" if avg_sentence else "Veri yok")
    col4.metric("⚠️ Yeniden Suç İşleme Oranı", f"{recid_rate:.2%}" if recid_rate else "Veri yok")

    st.markdown("""
    - **Toplam Kayıt:** Veri setindeki toplam mahpus sayısını gösterir.  
    - **Farklı Suç Tipi:** Veri setindeki benzersiz suç kategorilerinin sayısı.  
    - **Ortalama Ceza Süresi:** Ceza sürelerinin sayısal ortalaması, yıllık bazda.  
    - **Yeniden Suç İşleme Oranı:** Veri setindeki mahpusların tahliye sonrası 3 yıl içinde yeniden suç işleme oranı.
    """)

    st.markdown("---")
    st.subheader("📈 Veri Seti Görselleştirmeleri")

    col1, col2 = st.columns(2)

    grafik_tipi = st.selectbox(
        "Grafik tipi seçin:",
        options=["Bar Grafiği", "Histogram", "Kutu Grafiği (Box Plot)"],
        index=0
    )

    with col1:
        if "Prison_Offense" in df.columns:
            counts = df["Prison_Offense"].value_counts().reset_index()
            counts.columns = ["Suç Tipi", "Sayı"]

            if grafik_tipi == "Bar Grafiği":
                fig = px.bar(counts, x="Suç Tipi", y="Sayı", title="Suç Tipi Dağılımı")
            elif grafik_tipi == "Histogram":
                fig = px.histogram(df, x="Prison_Offense", title="Suç Tipi Histogramı")
            else:  # Box Plot
                st.info("Suç tipi için kutu grafiği anlamlı değil, Bar veya Histogram seçin.")
                fig = None

            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Suç tipi verisi mevcut değil.")

        if "Num_Distinct_Arrest_Crime_Types" in df.columns:
            if grafik_tipi == "Bar Grafiği":
                counts2 = df["Num_Distinct_Arrest_Crime_Types"].value_counts().reset_index()
                counts2.columns = ["Geçmiş Suç Sayısı", "Sayı"]
                fig2 = px.bar(counts2.sort_values("Geçmiş Suç Sayısı"), x="Geçmiş Suç Sayısı", y="Sayı", title="Geçmiş Suç Sayısı Dağılımı")
            elif grafik_tipi == "Histogram":
                fig2 = px.histogram(df, x="Num_Distinct_Arrest_Crime_Types", nbins=20, title="Geçmiş Suç Sayısı Histogramı")
            else:
                fig2 = px.box(df, y="Num_Distinct_Arrest_Crime_Types", title="Geçmiş Suç Sayısı Kutu Grafiği")

            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Geçmiş suç sayısı verisi mevcut değil.")

    with col2:
        if "Prison_Years_Numeric" in df.columns and df["Prison_Years_Numeric"].notnull().any():
            if grafik_tipi == "Bar Grafiği":
                counts3 = df["Prison_Years_Numeric"].value_counts().reset_index()
                counts3.columns = ["Ceza Süresi (yıl)", "Sayı"]
                fig3 = px.bar(counts3.sort_values("Ceza Süresi (yıl)"), x="Ceza Süresi (yıl)", y="Sayı", title="Ceza Süresi Dağılımı")
            elif grafik_tipi == "Histogram":
                fig3 = px.histogram(df, x="Prison_Years_Numeric", nbins=20, title="Ceza Süresi Histogramı (Yıl)")
            else:
                fig3 = px.box(df, y="Prison_Years_Numeric", title="Ceza Süresi Kutu Grafiği (Yıl)")

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
