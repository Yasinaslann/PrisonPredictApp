import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="Yeniden Suç İşleme Tahmin Uygulaması",
    page_icon="⚖️",
    layout="wide",
)

BASE = Path(__file__).parent
CANDIDATE_PATHS = [
    BASE / "Prisongüncelveriseti.csv",
    Path("/mnt/data/Prisongüncelveriseti.csv")
]

APP_VERSION = "v1.2 (Ana Sayfa)"

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

def info_icon(text):
    return f"ℹ️ {text}"

def safe_mean(series):
    # Sayısal olmayanları NaN yapıp sonra ortalama al
    return pd.to_numeric(series, errors='coerce').dropna().mean()

def home_page(df):
    # --- Üst Kısım: koyu mavi kutu ---
    st.markdown(
        """
        <div style="
            background-color: #0d1b2a; 
            color: white; 
            padding: 1.8rem 2rem; 
            border-radius: 15px; 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            box-shadow: 0 6px 15px rgba(0,0,0,0.3);
            ">
            <h1 style="margin-bottom: 0.3rem;">🏛️ Yeniden Suç İşleme Tahmin Uygulaması</h1>
            <h3 style="margin-top:0; color:#90caf9;">Proje Amacı</h3>
            <p style="line-height:1.5; font-size:1.1rem;">
                Bu uygulama, mahpusların tahliye sonrasında yeniden suç işleme riskini (recidivism) veri bilimi ve makine öğrenmesi teknikleri ile tahmin etmeyi amaçlar.<br>
                Amaç, topluma yeniden uyum sürecini iyileştirecek stratejiler geliştirmek ve risk analizi yaparak tekrar suç oranlarını azaltmaya katkı sağlamaktır.
            </p>
            <h3 style="margin-top: 1.5rem; color:#90caf9;">Veri Seti Hakkında</h3>
            <p style="line-height:1.5; font-size:1.1rem;">
                Veri seti, mahpusların demografik bilgileri, ceza süreleri, geçmiş suç kayıtları ve yeniden suç işleme bilgilerini içermektedir.<br>
                Bu bilgilerle risk faktörleri analiz edilip, model geliştirme için zengin bir kaynak sağlanmıştır.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("---")

    # --- İstatistik kartları ---
    total_rows = df.shape[0] if df is not None else 0
    total_cols = df.shape[1] if df is not None else 0
    unique_offenses = df["Prison_Offense"].nunique() if df is not None and "Prison_Offense" in df.columns else 0
    avg_sentence = safe_mean(df["Sentence_Length_Months"]) if df is not None and "Sentence_Length_Months" in df.columns else None
    recid_rate = safe_mean(df["Recidivism"]) if df is not None and "Recidivism" in df.columns else None
    avg_age = safe_mean(df["Age_at_Release"]) if df is not None and "Age_at_Release" in df.columns else None
    unique_education = df["Education_Level"].nunique() if df is not None and "Education_Level" in df.columns else 0
    unique_genders = df["Gender"].nunique() if df is not None and "Gender" in df.columns else 0

    cols = st.columns(7)

    card_style = """
        background-color: #e3f2fd;
        border-radius: 14px;
        padding: 1.5rem 1rem;
        text-align: center;
        box-shadow: 0 5px 15px rgb(3 155 229 / 0.25);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        """

    def render_card(col, number, label, emoji, color="#0d47a1"):
        col.markdown(f"""
            <div style="{card_style}">
                <div style="font-size: 2.6rem; font-weight: 800; color: {color};">{number}</div>
                <div style="font-size: 1.15rem; color: {color}; font-weight: 700;">{emoji} {label}</div>
            </div>
        """, unsafe_allow_html=True)

    render_card(cols[0], f"{total_rows:,}", "Toplam Kayıt", "🗂️")
    render_card(cols[1], total_cols, "Sütun Sayısı", "📋")
    render_card(cols[2], unique_offenses, "Farklı Suç Tipi", "📌")
    render_card(cols[3], f"{avg_sentence:.1f} ay" if avg_sentence else "N/A", "Ortalama Ceza Süresi", "⏳", "#1b5e20")
    render_card(cols[4], f"{(recid_rate*100):.1f}%" if recid_rate else "N/A", "Yeniden Suç İşleme Oranı", "⚠️", "#b71c1c")
    render_card(cols[5], f"{avg_age:.1f}" if avg_age else "N/A", "Ortalama Tahliye Yaşı", "👤", "#004d40")
    render_card(cols[6], unique_education, "Eğitim Seviyesi Sayısı", "🎓", "#6a1b9a")

    st.markdown("---")

    # --- Veri seti önizleme ---
    with st.expander("📂 Veri Seti Önizlemesi (İlk 10 Satır)"):
        st.dataframe(df.head(10))

    st.markdown("---")

    recid_col = next((c for c in df.columns if "recid" in c.lower()), None)

    # --- Grafikler ---
    st.subheader("🎯 Yeniden Suç İşleme Oranı (Pasta Grafiği)")
    col1, col2 = st.columns([3,1])
    with col1:
        if recid_col and recid_col in df.columns:
            counts = df[recid_col].value_counts().sort_index()
            labels = ["Tekrar Suç İşlemedi", "Tekrar Suç İşledi"]
            values = [counts.get(0, 0), counts.get(1, 0)]
            fig = px.pie(
                names=labels, values=values,
                title="3 Yıl İçinde Yeniden Suç İşleme Oranı",
                color_discrete_sequence=px.colors.sequential.RdBu
            )
            fig.update_traces(textposition='inside', textinfo='percent+label', pull=[0, 0.1])
            fig.update_layout(title_x=0.5, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Yeniden suç işleme verisi bulunmamaktadır.")
    with col2:
        st.markdown(info_icon("Bu pasta grafik, tahliye sonrası mahpusların yeniden suç işleme durumunu yüzdesel olarak gösterir. 'Tekrar Suç İşledi' dilimi öne çıkarılmıştır."))

    st.markdown("---")

    st.subheader("👥 Demografik Dağılımlar ve Yeniden Suç İşleme Oranları")
    demo_cols = ["Gender", "Education_Level"]
    cols = st.columns(len(demo_cols))
    for idx, col_name in enumerate(demo_cols):
        with cols[idx]:
            if col_name in df.columns:
                counts = df[col_name].value_counts()
                fig_bar = px.bar(
                    x=counts.index, y=counts.values,
                    labels={"x": col_name, "y": "Kişi Sayısı"},
                    title=f"{col_name.replace('_',' ')} Dağılımı",
                    color=counts.index,
                    color_discrete_sequence=px.colors.qualitative.Safe
                )
                fig_bar.update_layout(showlegend=False, template="plotly_white", title_x=0.5)
                st.plotly_chart(fig_bar, use_container_width=True)

                if recid_col:
                    recid_means = df.groupby(col_name)[recid_col].mean()
                    fig_recid = px.bar(
                        x=recid_means.index, y=recid_means.values,
                        labels={"x": col_name, "y": "Ortalama Yeniden Suç İşleme Oranı"},
                        title=f"{col_name.replace('_',' ')} Bazında Yeniden Suç İşleme Oranı",
                        color=recid_means.index,
                        color_discrete_sequence=px.colors.qualitative.Safe
                    )
                    fig_recid.update_layout(showlegend=False, template="plotly_white", title_x=0.5, yaxis=dict(range=[0,1]))
                    st.plotly_chart(fig_recid, use_container_width=True)
            else:
                st.info(f"{col_name} verisi bulunamadı.")
            st.markdown(info_icon(f"{col_name} dağılımı ve ilgili yeniden suç işleme oranları."))

    st.markdown("---")

    st.subheader("📊 Özelliklerin Recidivism ile Korelasyonu")
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if recid_col in numeric_cols:
        numeric_cols.remove(recid_col)

    corr = None
    try:
        corr = df[numeric_cols + [recid_col]].corr()[recid_col].drop(recid_col)
    except:
        corr = None

    if corr is not None and not corr.empty:
        corr_df = pd.DataFrame(corr).reset_index()
        corr_df.columns = ["Özellik", "Recidivism Korelasyonu"]
        corr_df = corr_df.sort_values(by="Recidivism Korelasyonu", key=abs, ascending=False)

        c1, c2 = st.columns([3,1])
        with c1:
            fig_corr = px.bar(
                corr_df, x="Özellik", y="Recidivism Korelasyonu",
                color="Recidivism Korelasyonu",
                color_continuous_scale=px.colors.diverging.RdBu,
                title="Özelliklerin Yeniden Suç İşleme ile Korelasyonu"
            )
            fig_corr.update_layout(template="plotly_white", title_x=0.5)
            st.plotly_chart(fig_corr, use_container_width=True)
        with c2:
            st.markdown(info_icon("Sayısal özelliklerin yeniden suç işleme ile korelasyonunu gösterir."))
    else:
        st.info("Sayısal veriler ve recidivism korelasyon bilgisi mevcut değil veya hesaplanamadı.")

    st.caption(f"📂 Repo: https://github.com/Yasinaslann/PrisonPredictApp • {APP_VERSION}")

def placeholder_page(name):
    st.title(name)
    st.info("Bu sayfa henüz hazırlanmadı. 'Ana Sayfa' hazırlandıktan sonra geliştirilecektir.")

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
