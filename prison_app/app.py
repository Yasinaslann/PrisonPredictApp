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

APP_VERSION = "v1.4 (Modern Grafikler)"

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

def safe_mean(series):
    return pd.to_numeric(series, errors='coerce').dropna().mean()

def safe_unique(series):
    return series.nunique() if series is not None else 0

def render_card(col, value, label, emoji, color="#0d47a1"):
    card_style = f"""
        background-color: {color}33;
        border-radius: 14px;
        padding: 1.7rem 1rem;
        text-align: center;
        box-shadow: 0 6px 15px rgb(3 155 229 / 0.3);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        min-height: 110px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        """
    number_style = f"""
        font-size: 2.4rem; 
        font-weight: 800; 
        color: {color};
        """
    label_style = f"""
        font-size: 1.15rem; 
        color: {color};
        font-weight: 700;
        margin-top: 0.2rem;
        """
    col.markdown(f"""
        <div style="{card_style}">
            <div style="{number_style}">{value}</div>
            <div style="{label_style}">{emoji} {label}</div>
        </div>
    """, unsafe_allow_html=True)

def info_icon(text):
    return f"ℹ️ {text}"

def convert_age_to_numeric(age_str):
    if pd.isna(age_str):
        return None
    age_str = str(age_str).strip()
    if "or older" in age_str:
        return 50
    elif "-" in age_str:
        parts = age_str.split("-")
        try:
            low = int(parts[0])
            high = int(parts[1])
            return (low + high) / 2
        except:
            return None
    else:
        try:
            return float(age_str)
        except:
            return None

def home_page(df):
    st.markdown(
        """
        <div style="
            background-color: #0d1b2a; 
            color: white; 
            padding: 2rem 2.5rem; 
            border-radius: 15px; 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            box-shadow: 0 6px 15px rgba(0,0,0,0.35);
            ">
            <h1 style="margin-bottom: 0.3rem;">🏛️ Yeniden Suç İşleme Tahmin Uygulaması</h1>
            <h3 style="margin-top:0; color:#90caf9;">Proje Amacı</h3>
            <p style="line-height:1.6; font-size:1.1rem;">
                Bu uygulama, mahpusların tahliye sonrasında yeniden suç işleme riskini (recidivism) veri bilimi ve makine öğrenmesi teknikleri ile tahmin etmeyi amaçlar.<br>
                Amaç, topluma yeniden uyum sürecini iyileştirecek stratejiler geliştirmek ve risk analizi yaparak tekrar suç oranlarını azaltmaya katkı sağlamaktır.
            </p>
            <h3 style="margin-top: 1.7rem; color:#90caf9;">Veri Seti Hakkında</h3>
            <p style="line-height:1.6; font-size:1.1rem;">
                Veri seti, mahpusların demografik bilgileri, ceza süreleri, geçmiş suç kayıtları ve yeniden suç işleme bilgilerini içermektedir.<br>
                Bu bilgilerle risk faktörleri analiz edilip, model geliştirme için zengin bir kaynak sağlanmıştır.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    if df is None:
        st.error("Veri seti yüklenemedi. Lütfen dosyanın doğru yerde ve formatta olduğundan emin olun.")
        return

    recid_col = None
    for c in df.columns:
        if "recid" in c.lower():
            recid_col = c
            break

    if recid_col:
        df[recid_col] = pd.to_numeric(df[recid_col], errors='coerce')

    if "Age_at_Release" in df.columns:
        df["Age_at_Release_Num"] = df["Age_at_Release"].apply(convert_age_to_numeric)
    else:
        df["Age_at_Release_Num"] = None

    if "Sentence_Length_Months" in df.columns:
        df["Sentence_Length_Months"] = pd.to_numeric(df["Sentence_Length_Months"], errors='coerce')

    info_cards = []
    info_cards.append(("Toplam Kayıt", f"{df.shape[0]:,}", "🗂️", "#0d47a1"))
    info_cards.append(("Sütun Sayısı", df.shape[1], "📋", "#1976d2"))

    if "Prison_Offense" in df.columns:
        n_offense = safe_unique(df["Prison_Offense"])
        if n_offense > 0:
            info_cards.append(("Farklı Suç Tipi", n_offense, "📌", "#0288d1"))

    avg_sentence = safe_mean(df["Sentence_Length_Months"]) if "Sentence_Length_Months" in df.columns else None
    if avg_sentence is not None:
        info_cards.append(("Ortalama Ceza Süresi (Ay)", f"{avg_sentence:.1f}", "⏳", "#388e3c"))

    recid_rate = safe_mean(df[recid_col]) if recid_col else None
    if recid_rate is not None:
        info_cards.append(("Yeniden Suç İşleme Oranı", f"%{recid_rate*100:.1f}", "⚠️", "#d32f2f"))

    avg_age = safe_mean(df["Age_at_Release_Num"]) if "Age_at_Release_Num" in df.columns else None
    if avg_age is not None and not pd.isna(avg_age):
        info_cards.append(("Ortalama Tahliye Yaşı", f"{avg_age:.1f}", "👤", "#00695c"))

    if "Education_Level" in df.columns:
        n_edu = safe_unique(df["Education_Level"])
        if n_edu > 0:
            info_cards.append(("Eğitim Seviyesi Sayısı", n_edu, "🎓", "#6a1b9a"))

    if "Gender" in df.columns:
        n_gender = safe_unique(df["Gender"])
        if n_gender > 0:
            info_cards.append(("Cinsiyet Sayısı", n_gender, "🚻", "#5d4037"))

    n = len(info_cards)
    rows = (n + 3) // 4
    for r in range(rows):
        cols = st.columns(4, gap="small")
        for i in range(4):
            idx = r*4 + i
            if idx >= n:
                break
            label, val, emoji, color = info_cards[idx]
            render_card(cols[i], val, label, emoji, color)

    st.markdown("---")

    with st.expander("📂 Veri Seti Önizlemesi (İlk 10 Satır)"):
        st.dataframe(df.head(10))

    st.markdown("---")

    st.subheader("🎯 Yeniden Suç İşleme Oranı (Pasta Grafiği)")
    col1, col2 = st.columns([3,1])
    with col1:
        if recid_col and not df[recid_col].dropna().empty:
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
                    labels={"x": col_name.replace('_',' '), "y": "Kişi Sayısı"},
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
                        labels={"x": col_name.replace('_',' '), "y": "Ortalama Yeniden Suç İşleme Oranı"},
                        title=f"{col_name.replace('_',' ')} Bazında Yeniden Suç İşleme Oranı",
                        color=recid_means.index,
                        color_discrete_sequence=px.colors.qualitative.Safe
                    )
                    fig_recid.update_layout(showlegend=False, template="plotly_white", title_x=0.5, yaxis=dict(range=[0,1]))
                    st.plotly_chart(fig_recid, use_container_width=True)
            else:
                st.info(f"{col_name} verisi bulunamadı.")
            st.markdown(info_icon(f"{col_name.replace('_',' ')} dağılımı ve ilgili yeniden suç işleme oranları."))

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
        ("Ana Sayfa", "Tahmin Modeli", "Tavsiye ve Profil Analizi", "Model Analizleri ve Harita"),
    )

    if page == "Ana Sayfa":
        home_page(df)
    elif page == "Tahmin Modeli":
        # Tahmin sayfasını ayrı olarak pages/page_prediction.py'de kullanacağız
        st.info("Tahmin sayfası için lütfen sol menüden 'Tahmin Modeli' sayfasına geçiniz veya 'pages/page_prediction.py' dosyasını açınız.")
    elif page == "Tavsiye ve Profil Analizi":
        placeholder_page("💡 Tavsiye ve Profil Analizi (Hazırlanıyor)")
    elif page == "Model Analizleri ve Harita":
        placeholder_page("📈 Model Analizleri ve Harita (Hazırlanıyor)")

if __name__ == "__main__":
    main()
