import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# Sayfa yapılandırması
st.set_page_config(
    page_title="Yeniden Suç İşleme Tahmin Uygulaması",
    page_icon="⚖️",
    layout="wide"
)

# Veri yolu (dosyan senin dizininde olmalı)
BASE = Path(__file__).parent
DATA_PATHS = [
    BASE / "Prisongüncelveriseti.csv",
    Path("/mnt/data/Prisongüncelveriseti.csv")
]

APP_VERSION = "v1.2 Modern"

@st.cache_data(show_spinner=False)
def load_data():
    for p in DATA_PATHS:
        try:
            if p.exists():
                df = pd.read_csv(p)
                return df
        except Exception as e:
            pass
    return None

def safe_mean(series):
    return pd.to_numeric(series, errors='coerce').dropna().mean()

def info_box(text):
    return f"""
        <div style="
            background:#e3f2fd; 
            padding: 10px; 
            border-radius: 8px; 
            font-size: 14px; 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: #0d47a1;">
            ℹ️ {text}
        </div>
    """

def render_card(col, number, label, emoji, color="#0d47a1"):
    card_style = f"""
        background-color: #e3f2fd;
        border-radius: 14px;
        padding: 1.5rem 1rem;
        text-align: center;
        box-shadow: 0 5px 15px rgb(3 155 229 / 0.25);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: {color};
    """
    col.markdown(f"""
        <div style="{card_style}">
            <div style="font-size: 2.6rem; font-weight: 800;">{number}</div>
            <div style="font-size: 1.15rem; font-weight: 700;">{emoji} {label}</div>
        </div>
    """, unsafe_allow_html=True)

def home_page(df):
    # Başlık ve açıklama
    st.markdown(
        """
        <div style="
            background-color: #0d1b2a; 
            color: white; 
            padding: 2rem 2rem; 
            border-radius: 15px; 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            box-shadow: 0 6px 15px rgba(0,0,0,0.3);
        ">
            <h1 style="margin-bottom: 0.2rem;">🏛️ Yeniden Suç İşleme Tahmin Uygulaması</h1>
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

    # Kartlar
    total_rows = df.shape[0] if df is not None else 0
    total_cols = df.shape[1] if df is not None else 0
    unique_offenses = df["Prison_Offense"].nunique() if df is not None and "Prison_Offense" in df.columns else 0
    avg_sentence = safe_mean(df["Sentence_Length_Months"]) if df is not None and "Sentence_Length_Months" in df.columns else None
    recid_rate = safe_mean(df["Recidivism"]) if df is not None and "Recidivism" in df.columns else None
    avg_age = safe_mean(df["Age_at_Release"]) if df is not None and "Age_at_Release" in df.columns else None
    unique_education = df["Education_Level"].nunique() if df is not None and "Education_Level" in df.columns else 0
    unique_genders = df["Gender"].nunique() if df is not None and "Gender" in df.columns else 0

    cols = st.columns(7, gap="small")

    render_card(cols[0], f"{total_rows:,}", "Toplam Kayıt", "🗂️")
    render_card(cols[1], total_cols, "Sütun Sayısı", "📋")
    render_card(cols[2], unique_offenses, "Farklı Suç Tipi", "📌")
    render_card(cols[3], f"{avg_sentence:.1f} ay" if avg_sentence else "N/A", "Ortalama Ceza Süresi", "⏳", "#1b5e20")
    render_card(cols[4], f"{(recid_rate*100):.1f}%" if recid_rate else "N/A", "Yeniden Suç İşleme Oranı", "⚠️", "#b71c1c")
    render_card(cols[5], f"{avg_age:.1f}" if avg_age else "N/A", "Ortalama Tahliye Yaşı", "👤", "#004d40")
    render_card(cols[6], unique_education, "Eğitim Seviyesi Sayısı", "🎓", "#6a1b9a")

    st.markdown("---")

    # Veri seti önizleme
    st.subheader("📂 Veri Seti Önizlemesi (İlk 15 Satır)")
    if df is not None and not df.empty:
        st.dataframe(df.head(15), use_container_width=True, height=320)
    else:
        st.info("Veri seti boş veya yüklenemedi.")

    st.markdown("---")

    # Yeniden Suç İşleme Oranı (Pasta Grafik)
    st.subheader("🎯 Yeniden Suç İşleme Oranı")

    if "Recidivism" in df.columns:
        recid_counts = df["Recidivism"].value_counts(dropna=False).sort_index()
        total = recid_counts.sum()
        again_count = recid_counts.get(1, 0)
        no_again_count = recid_counts.get(0, 0)
        again_pct = (again_count / total)*100 if total > 0 else 0

        col1, col2 = st.columns([3, 1], gap="small")
        with col1:
            fig = px.pie(
                names=["Tekrar Suç İşlemedi", "Tekrar Suç İşledi"],
                values=[no_again_count, again_count],
                title="3 Yıl İçinde Yeniden Suç İşleme Oranı",
                color_discrete_sequence=px.colors.sequential.RdBu,
            )
            fig.update_traces(textposition='inside', textinfo='percent+label', pull=[0, 0.1])
            fig.update_layout(title_x=0.5, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown(f"""
                <div style="background:#f0f4f8; padding:1.5rem; border-radius:10px; text-align:center; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
                    <h3 style="color:#b71c1c;">⚠️ Yeniden Suç İşleme</h3>
                    <p><strong>{again_count:,}</strong> kişi tekrar suç işlemiş.</p>
                    <p><strong>{no_again_count:,}</strong> kişi tekrar suç işlememiş.</p>
                    <p><strong>%{again_pct:.1f}</strong> oranında risk var.</p>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Yeniden suç işleme verisi bulunmamaktadır.")

    st.markdown("---")

    # Demografik Dağılımlar ve Recidivism Oranları (Interaktif)
    st.subheader("👥 Demografik Dağılımlar & Yeniden Suç İşleme Oranları")

    demo_options = []
    if "Gender" in df.columns:
        demo_options.append("Gender")
    if "Education_Level" in df.columns:
        demo_options.append("Education_Level")

    if demo_options:
        choice = st.selectbox("Gösterilecek Demografik Özellik", demo_options)

        counts = df[choice].value_counts()
        recid_col = "Recidivism" if "Recidivism" in df.columns else None

        col1, col2 = st.columns(2, gap="small")
        with col1:
            fig_count = px.bar(
                x=counts.index, y=counts.values,
                labels={"x": choice.replace('_',' '), "y": "Kişi Sayısı"},
                title=f"{choice.replace('_',' ')} Dağılımı",
                color=counts.index,
                color_discrete_sequence=px.colors.qualitative.Safe,
            )
            fig_count.update_layout(showlegend=False, template="plotly_white", title_x=0.5)
            st.plotly_chart(fig_count, use_container_width=True)

        with col2:
            if recid_col:
                recid_means = df.groupby(choice)[recid_col].mean()
                fig_recid = px.bar(
                    x=recid_means.index, y=recid_means.values,
                    labels={"x": choice.replace('_',' '), "y": "Ortalama Yeniden Suç İşleme Oranı"},
                    title=f"{choice.replace('_',' ')} Bazında Yeniden Suç İşleme Oranı",
                    color=recid_means.index,
                    color_discrete_sequence=px.colors.qualitative.Safe,
                )
                fig_recid.update_layout(showlegend=False, template="plotly_white", title_x=0.5, yaxis=dict(range=[0,1]))
                st.plotly_chart(fig_recid, use_container_width=True)
            else:
                st.info("Yeniden suç işleme verisi bulunmamaktadır.")

        st.markdown(info_box(f"{choice.replace('_',' ')} dağılımı ve ilgili yeniden suç işleme oranları."))
    else:
        st.info("Demografik veri bulunmamaktadır.")

    st.markdown("---")

    # Özelliklerin Korelasyonu
    st.subheader("📊 Özelliklerin Yeniden Suç İşleme ile Korelasyonu")

    recid_col = "Recidivism" if "Recidivism" in df.columns else None
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if recid_col in numeric_cols:
        numeric_cols.remove(recid_col)

    corr = None
    try:
        corr = df[numeric_cols + [recid_col]].corr()[recid_col].drop(recid_col)
    except Exception:
        corr = None

    if corr is not None and not corr.empty:
        corr_df = pd.DataFrame(corr).reset_index()
        corr_df.columns = ["Özellik", "Recidivism Korelasyonu"]
        corr_df = corr_df.sort_values(by="Recidivism Korelasyonu", key=abs, ascending=False)

        st.dataframe(corr_df.style.background_gradient(cmap='RdBu_r').format({"Recidivism Korelasyonu": "{:.3f}"}), height=320, use_container_width=True)

        fig_corr = px.bar(
            corr_df, x="Özellik", y="Recidivism Korelasyonu",
            color="Recidivism Korelasyonu",
            color_continuous_scale=px.colors.diverging.RdBu,
            title="Özelliklerin Yeniden Suç İşleme ile Korelasyonu",
        )
        fig_corr.update_layout(template="plotly_white", title_x=0.5)
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Sayısal veriler ve recidivism korelasyon bilgisi mevcut değil veya hesaplanamadı.")

    st.markdown("---")

    # Ortalama Tahliye Yaşı
    st.subheader("👤 Ortalama Tahliye Yaşı")

    avg_age = safe_mean(df["Age_at_Release"]) if "Age_at_Release" in df.columns else None
    if avg_age is not None:
        st.metric(label="Ortalama Tahliye Yaşı", value=f"{avg_age:.1f}")
    else:
        st.info("Ortalama tahliye yaşı verisi bulunmamaktadır.")

    # Alt bilgi
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
