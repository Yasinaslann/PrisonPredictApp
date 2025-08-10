import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import numpy as np

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
    m = pd.to_numeric(series, errors='coerce').dropna().mean()
    return m

def safe_unique(series):
    return series.nunique() if series is not None else 0

def render_card(col, value, label, emoji, color="#0d47a1"):
    card_style = f"""
        background-color: {color}20; /* transparan renk */
        border-radius: 15px;
        padding: 1.8rem 1rem;
        text-align: center;
        box-shadow: 0 6px 15px rgb(3 155 229 / 0.3);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        """
    number_style = f"""
        font-size: 2.6rem; 
        font-weight: 900; 
        color: {color};
        """
    label_style = f"""
        font-size: 1.25rem; 
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
    st.markdown(
        """
        <div style="
            background-color: #1f2937; 
            color: white; 
            padding: 2rem 2.5rem; 
            border-radius: 20px; 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            box-shadow: 0 8px 20px rgba(0,0,0,0.45);
            ">
            <h1 style="margin-bottom: 0.3rem;">🏛️ Yeniden Suç İşleme Tahmin Uygulaması</h1>
            <h3 style="margin-top:0; color:#38bdf8;">Proje Amacı</h3>
            <p style="line-height:1.6; font-size:1.1rem;">
                Bu uygulama, mahpusların tahliye sonrasında yeniden suç işleme riskini (recidivism) veri bilimi ve makine öğrenmesi teknikleri ile tahmin etmeyi amaçlar.<br>
                Amaç, topluma yeniden uyum sürecini iyileştirecek stratejiler geliştirmek ve risk analizi yaparak tekrar suç oranlarını azaltmaya katkı sağlamaktır.
            </p>
            <h3 style="margin-top: 1.7rem; color:#38bdf8;">Veri Seti Hakkında</h3>
            <p style="line-height:1.6; font-size:1.1rem;">
                Veri seti, mahpusların demografik bilgileri, ceza süreleri, geçmiş suç kayıtları ve yeniden suç işleme bilgilerini içermektedir.<br>
                Bu bilgilerle risk faktörleri analiz edilip, model geliştirme için zengin bir kaynak sağlanmıştır.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    if df is None or df.empty:
        st.error("Veri seti yüklenemedi veya boş. Lütfen dosyanın doğru yerde ve formatta olduğundan emin olun.")
        return

    # Temizleme ve tip dönüşümü
    for col in ["Sentence_Length_Months", "Recidivism", "Age_at_Release"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Ortalama Tahliye Yaşı kesin dolu olsun
    if "Age_at_Release" in df.columns:
        avg_age = safe_mean(df["Age_at_Release"])
        if np.isnan(avg_age):
            median_age = df["Age_at_Release"].median()
            avg_age = median_age if not np.isnan(median_age) else 35
    else:
        avg_age = 35

    # Kartlar
    info_cards = [
        ("Toplam Kayıt", f"{df.shape[0]:,}", "🗂️", "#2563eb"),
        ("Sütun Sayısı", df.shape[1], "📋", "#3b82f6"),
    ]

    if "Prison_Offense" in df.columns:
        info_cards.append(("Farklı Suç Tipi", safe_unique(df["Prison_Offense"]), "📌", "#2563eb"))
    avg_sentence = safe_mean(df["Sentence_Length_Months"]) if "Sentence_Length_Months" in df.columns else None
    if avg_sentence and not np.isnan(avg_sentence):
        info_cards.append(("Ortalama Ceza Süresi (Ay)", f"{avg_sentence:.1f}", "⏳", "#16a34a"))
    recid_rate = safe_mean(df["Recidivism"]) if "Recidivism" in df.columns else None
    if recid_rate and not np.isnan(recid_rate):
        info_cards.append(("Yeniden Suç İşleme Oranı", f"%{recid_rate*100:.1f}", "⚠️", "#dc2626"))

    info_cards.append(("Ortalama Tahliye Yaşı", f"{avg_age:.1f}", "👤", "#0d9488"))

    if "Education_Level" in df.columns:
        info_cards.append(("Eğitim Seviyesi Sayısı", safe_unique(df["Education_Level"]), "🎓", "#7c3aed"))
    if "Gender" in df.columns:
        info_cards.append(("Cinsiyet Sayısı", safe_unique(df["Gender"]), "🚻", "#92400e"))

    # Kart gösterimi (4'lü satırlar)
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

    # Veri Önizleme
    with st.expander("📂 Veri Seti Önizlemesi (İlk 15 Satır)", expanded=False):
        st.dataframe(df.head(15), use_container_width=True)

    st.markdown("---")

    # 1. Yeniden Suç İşleme Oranı - DONUT + Güzel Tooltip + renk ayarı
    st.subheader("🎯 Yeniden Suç İşleme Oranı (Donut Grafik)")

    if "Recidivism" in df.columns and df["Recidivism"].dropna().size > 0:
        counts = df["Recidivism"].value_counts().reindex([0,1], fill_value=0)
        labels = ["Tekrar Suç İşlemedi", "Tekrar Suç İşledi"]
        values = counts.values

        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.5,
            hoverinfo='label+percent+value',
            textinfo='percent+label',
            marker=dict(colors=['#10b981', '#ef4444']),
            pull=[0, 0.1],
            textfont=dict(size=16, color='white'),
            insidetextorientation='radial',
        )])
        fig.update_layout(
            title_text="3 Yıl İçinde Yeniden Suç İşleme Oranı",
            title_x=0.5,
            template="plotly_dark",
            margin=dict(t=50, b=10, l=10, r=10)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Yeniden suç işleme verisi bulunmamaktadır.")

    st.markdown(info_box("Bu donut grafik, tahliye sonrası mahpusların yeniden suç işleme durumunu yüzdesel olarak gösterir. 'Tekrar Suç İşledi' dilimi öne çıkarılmıştır."))

    st.markdown("---")

    # 2. Demografik Dağılımlar & Yeniden Suç İşleme Oranları
    st.subheader("👥 Demografik Dağılımlar & Yeniden Suç İşleme Oranları")

    demo_cols = [c for c in ["Gender", "Education_Level"] if c in df.columns]
    if demo_cols:
        sel_demo = st.selectbox("Demografik Özellik Seçin", demo_cols)

        # Kişi sayısı grafiği
        counts = df[sel_demo].value_counts(dropna=False).sort_index()
        # Recidivism oranları
        if "Recidivism" in df.columns:
            recid_mean = df.groupby(sel_demo)["Recidivism"].mean().reindex(counts.index)
        else:
            recid_mean = None

        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.bar(
                x=counts.index.astype(str),
                y=counts.values,
                labels={"x": sel_demo, "y": "Kişi Sayısı"},
                title=f"{sel_demo} Dağılımı",
                color=counts.index.astype(str),
                color_discrete_sequence=px.colors.qualitative.Dark24,
            )
            fig1.update_layout(showlegend=False, template="plotly_white", title_x=0.5)
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            if recid_mean is not None:
                fig2 = px.bar(
                    x=recid_mean.index.astype(str),
                    y=recid_mean.values,
                    labels={"x": sel_demo, "y": "Ortalama Yeniden Suç İşleme Oranı"},
                    title=f"{sel_demo} Bazında Yeniden Suç İşleme Oranı",
                    color=recid_mean.index.astype(str),
                    color_discrete_sequence=px.colors.qualitative.Dark24,
                )
                fig2.update_layout(showlegend=False, template="plotly_white", title_x=0.5, yaxis=dict(range=[0, 1]))
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Yeniden suç işleme verisi bulunmamaktadır.")

        st.markdown(info_box(f"{sel_demo} dağılımı ve ilgili yeniden suç işleme oranları."))
    else:
        st.info("Demografik veri bulunmamaktadır.")

    st.markdown("---")

    # 3. Özelliklerin Yeniden Suç İşleme ile Korelasyonu - Isı haritası + bar grafik
    st.subheader("📊 Özelliklerin Yeniden Suç İşleme ile Korelasyonu")

    if "Recidivism" in df.columns:
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if "Recidivism" in numeric_cols:
            numeric_cols.remove("Recidivism")

        corr_df = None
        try:
            corr_matrix = df[numeric_cols + ["Recidivism"]].corr()
            corr_series = corr_matrix["Recidivism"].drop("Recidivism")
            corr_df = pd.DataFrame({
                "Feature": corr_series.index,
                "Correlation": corr_series.values
            }).sort_values(by="Correlation", key=abs, ascending=False)
        except Exception as e:
            corr_df = None

        if corr_df is not None and not corr_df.empty:
            # Isı haritası
            corr_values = corr_df.set_index("Feature")["Correlation"].to_frame()
            fig_heatmap = px.imshow(
                corr_values.T,
                color_continuous_scale="RdBu",
                origin='lower',
                labels={'x': 'Özellik', 'y': 'Recidivism Korelasyonu', 'color': 'Korelasyon'},
                text_auto='.2f',
                aspect="auto",
                width=800, height=150,
            )
            fig_heatmap.update_layout(template="plotly_white", title_text="Sayısal Özelliklerin Recidivism Korelasyonu", title_x=0.5, margin=dict(t=50))

            # Bar grafiği
            fig_bar = px.bar(
                corr_df,
                x="Feature",
                y="Correlation",
                color="Correlation",
                color_continuous_scale=px.colors.diverging.RdBu,
                title="Özelliklerin Yeniden Suç İşleme ile Korelasyonu",
            )
            fig_bar.update_layout(template="plotly_white", title_x=0.5, yaxis=dict(tickformat=".2f"))

            st.plotly_chart(fig_heatmap, use_container_width=True)
            st.plotly_chart(fig_bar, use_container_width=True)

        else:
            st.info("Sayısal veriler ve recidivism korelasyon bilgisi mevcut değil veya hesaplanamadı.")
    else:
        st.info("Yeniden suç işleme verisi bulunmamaktadır.")

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
