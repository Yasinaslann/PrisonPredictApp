import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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

def info_box(text):
    return f"ℹ️ {text}"

def home_page(df):
    # Üst başlık ve açıklama kutusu
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

    # Sayısal veri dönüştür
    for col in ["Sentence_Length_Months", "Recidivism", "Age_at_Release"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Kart bilgileri
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

    recid_rate = safe_mean(df["Recidivism"]) if "Recidivism" in df.columns else None
    if recid_rate is not None:
        info_cards.append(("Yeniden Suç İşleme Oranı", f"%{recid_rate*100:.1f}", "⚠️", "#d32f2f"))

    avg_age = safe_mean(df["Age_at_Release"]) if "Age_at_Release" in df.columns else None
    if avg_age is not None:
        info_cards.append(("Ortalama Tahliye Yaşı", f"{avg_age:.1f}", "👤", "#00695c"))

    if "Education_Level" in df.columns:
        n_edu = safe_unique(df["Education_Level"])
        if n_edu > 0:
            info_cards.append(("Eğitim Seviyesi Sayısı", n_edu, "🎓", "#6a1b9a"))

    if "Gender" in df.columns:
        n_gender = safe_unique(df["Gender"])
        if n_gender > 0:
            info_cards.append(("Cinsiyet Sayısı", n_gender, "🚻", "#5d4037"))

    # Kartları 4 sütun halinde göster
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

    # --- Veri Önizleme (Açılır/Kapanır, 15 satır) ---
    with st.expander("📂 Veri Seti Önizlemesi (İlk 15 Satır)", expanded=False):
        st.dataframe(df.head(15), use_container_width=True)

    st.markdown("---")

    # --- DONUT GRAFİK: Yeniden Suç İşleme Oranı ---
    st.subheader("🎯 Yeniden Suç İşleme Oranı (Donut Grafik)")

    recid_col = "Recidivism" if "Recidivism" in df.columns else None
    if recid_col and not df[recid_col].dropna().empty:
        counts = df[recid_col].value_counts().reindex([0, 1], fill_value=0)
        labels = ["Tekrar Suç İşlemedi", "Tekrar Suç İşledi"]
        values = counts.values

        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.5,
            marker=dict(colors=['#10b981', '#ef4444']),
            pull=[0, 0.1],
            textinfo='percent+label',
            textfont=dict(size=16, family='Segoe UI')
        )])
        fig.update_layout(title_text="3 Yıl İçinde Yeniden Suç İşleme Oranı", title_x=0.5, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Yeniden suç işleme verisi bulunmamaktadır.")

    st.markdown(info_box("Bu donut grafik, tahliye sonrası mahpusların yeniden suç işleme durumunu yüzdesel olarak gösterir. 'Tekrar Suç İşledi' dilimi öne çıkarılmıştır."))

    st.markdown("---")

    # --- Demografik Dağılımlar & Yeniden Suç İşleme Oranları ---
    st.subheader("👥 Demografik Dağılımlar & Yeniden Suç İşleme Oranları")

    demo_cols = []
    if "Gender" in df.columns:
        demo_cols.append("Gender")
    if "Education_Level" in df.columns:
        demo_cols.append("Education_Level")

    if demo_cols:
        cols = st.columns(len(demo_cols))
        for idx, col_name in enumerate(demo_cols):
            with cols[idx]:
                counts = df[col_name].value_counts()
                fig_bar = px.bar(
                    x=counts.index, y=counts.values,
                    labels={"x": col_name.replace('_',' '), "y": "Kişi Sayısı"},
                    title=f"{col_name.replace('_',' ')} Dağılımı",
                    color=counts.index,
                    color_discrete_sequence=px.colors.qualitative.Safe,
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
                        color_discrete_sequence=px.colors.qualitative.Safe,
                    )
                    fig_recid.update_layout(showlegend=False, template="plotly_white", title_x=0.5, yaxis=dict(range=[0,1]))
                    st.plotly_chart(fig_recid, use_container_width=True)

            st.markdown(info_box(f"{col_name.replace('_',' ')} dağılımı ve ilgili yeniden suç işleme oranları."))
    else:
        st.info("Demografik veri bulunmamaktadır.")

    st.markdown("---")

    # --- Özelliklerin Yeniden Suç İşleme ile Korelasyonu ---
    st.subheader("📊 Özelliklerin Yeniden Suç İşleme ile Korelasyonu")

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
            st.markdown(info_box("Sayısal özelliklerin yeniden suç işleme ile korelasyonunu gösterir."))
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
        placeholder_page("📊 Tahmin Modeli (Hazırlanıyor)")
    elif page == "Tavsiye ve Profil Analizi":
        placeholder_page("💡 Tavsiye ve Profil Analizi (Hazırlanıyor)")
    elif page == "Model Analizleri ve Harita":
        placeholder_page("📈 Model Analizleri ve Harita (Hazırlanıyor)")

if __name__ == "__main__":
    main()

