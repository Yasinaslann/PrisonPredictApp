import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# Sayfa yapılandırması
st.set_page_config(
    page_title="Yeniden Suç İşleme Tahmin Uygulaması",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
    theme="dark"
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

def info_icon(text):
    return f"ℹ️ {text}"

def home_page(df):
    # Başlık ve açıklamalar (koyu mavi kutu)
    st.markdown(
        """
        <div style="background-color:#0d47a1; padding:20px; border-radius:12px; color:#e3f2fd; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
            <h1 style="margin-bottom:0.3rem;">🏛️ Yeniden Suç İşleme Tahmin Uygulaması</h1>
            <h3 style="margin-top:0; margin-bottom:0.8rem; font-weight:600;">Proje Amacı</h3>
            <p>Bu uygulama, mahpusların tahliye sonrasında yeniden suç işleme riskini (recidivism) veri bilimi ve makine öğrenmesi teknikleri ile tahmin etmeyi amaçlar.</p>
            <p>Amaç, topluma yeniden uyum sürecini iyileştirecek stratejiler geliştirmek ve risk analizi yaparak tekrar suç oranlarını azaltmaya katkı sağlamaktır.</p>
            <h3 style="margin-top:1.2rem; margin-bottom:0.3rem; font-weight:600;">Veri Seti Hakkında</h3>
            <p>Veri seti, mahpusların demografik bilgileri, ceza süreleri, geçmiş suç kayıtları ve yeniden suç işleme bilgilerini içermektedir.</p>
            <p>Bu bilgilerle risk faktörleri analiz edilip, model geliştirme için zengin bir kaynak sağlanmıştır.</p>
        </div>
        """, unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Kartlar (hepsi çok yakın) ---
    total_rows = df.shape[0] if df is not None else 0
    total_cols = df.shape[1] if df is not None else 0
    unique_offenses = df["Prison_Offense"].nunique() if df is not None and "Prison_Offense" in df.columns else 0

    # Ortalama ceza süresi ve tahliye yaşı ortalaması gibi önemli metrikler için try-except
    def safe_mean(col_name):
        try:
            if col_name in df.columns:
                numeric_vals = pd.to_numeric(df[col_name], errors='coerce').dropna()
                if len(numeric_vals) > 0:
                    return f"{numeric_vals.mean():.1f}"
        except:
            pass
        return "N/A"

    avg_sentence = safe_mean("Sentence_Length")
    avg_release_age = safe_mean("Age_at_Release")

    recid_col = next((c for c in df.columns if "recid" in c.lower()), None)
    recid_rate = "N/A"
    if recid_col and recid_col in df.columns:
        recid_vals = pd.to_numeric(df[recid_col], errors='coerce').dropna()
        if len(recid_vals) > 0:
            recid_rate = f"{recid_vals.mean() * 100:.1f}%"

    cards_data = [
        {"title": "🗂️ Toplam Kayıt", "value": f"{total_rows:,}"},
        {"title": "📋 Sütun Sayısı", "value": f"{total_cols}"},
        {"title": "📌 Farklı Suç Tipi", "value": f"{unique_offenses}"},
        {"title": "⏳ Ortalama Ceza Süresi (Ay)", "value": avg_sentence},
        {"title": "👤 Ortalama Tahliye Yaşı", "value": avg_release_age},
        {"title": "⚠️ Yeniden Suç İşleme Oranı", "value": recid_rate},
    ]

    # Kartlar çok yakın, flexbox gibi yan yana
    cols = st.columns(len(cards_data), gap="small")
    card_style = """
        background-color: #1565c0;
        border-radius: 10px;
        padding: 18px 12px;
        text-align: center;
        color: #e3f2fd;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        box-shadow: 0 6px 10px rgba(0,0,0,0.3);
    """

    for col, card in zip(cols, cards_data):
        with col:
            st.markdown(
                f"""
                <div style="{card_style}">
                    <div style="font-size: 2.2rem; font-weight: 700;">{card['value']}</div>
                    <div style="font-size: 1.1rem; font-weight: 600; margin-top: 4px;">{card['title']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("---")

    # --- Veri Seti Önizlemesi (modern, açılır kapanır, büyük font, şık) ---
    st.subheader("📂 Veri Seti Önizlemesi")
    with st.expander("Veri Setini Göster / Gizle", expanded=False):
        # Şık tablo için hafif stil verelim (sadece tablo başlıkları)
        st.write(
            df.style.set_table_styles([
                {'selector': 'thead th', 'props': [('background-color', '#1565c0'), ('color', 'white'), ('font-weight', '600'), ('font-size', '14px')]},
                {'selector': 'tbody tr:nth-child(even)', 'props': [('background-color', '#1e88e5')]},
                {'selector': 'tbody tr:nth-child(odd)', 'props': [('background-color', '#1976d2')]},
                {'selector': 'tbody tr:hover', 'props': [('background-color', '#42a5f5')]}
            ]).set_properties(**{'color': 'white', 'font-family': "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif", 'font-size': '13px'}),
            unsafe_allow_html=True
        )

    st.markdown("---")

    # --- Grafikler (modern interaktif Plotly) ---

    # Yeniden Suç İşleme Oranı (Pie Chart)
    st.subheader("🎯 Yeniden Suç İşleme Oranı Dağılımı")
    if recid_col and recid_col in df.columns:
        counts = df[recid_col].value_counts().sort_index()
        labels = ["Tekrar Suç İşlemedi", "Tekrar Suç İşledi"]
        values = [counts.get(0, 0), counts.get(1, 0)]
        fig = px.pie(
            names=labels,
            values=values,
            title="3 Yıl İçinde Yeniden Suç İşleme Oranı",
            color_discrete_sequence=px.colors.sequential.Blues,
            hole=0.5
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(title_x=0.5, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(info_icon("Bu grafik, tahliye sonrası mahpusların yeniden suç işleme durumunu yüzdesel olarak gösterir."))
    else:
        st.info("Yeniden suç işleme verisi bulunmamaktadır.")

    st.markdown("---")

    # Demografik Dağılımlar ve Recidivism Oranları
    st.subheader("👥 Demografik Dağılımlar ve Yeniden Suç İşleme Oranları")

    demo_cols = ["Gender", "Education_Level"]
    cols = st.columns(len(demo_cols))

    for idx, col_name in enumerate(demo_cols):
        with cols[idx]:
            if col_name in df.columns:
                counts = df[col_name].value_counts()
                fig_bar = px.bar(
                    x=counts.index, y=counts.values,
                    labels={"x": col_name.replace('_', ' '), "y": "Kişi Sayısı"},
                    title=f"{col_name.replace('_',' ')} Dağılımı",
                    color=counts.index,
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig_bar.update_layout(showlegend=False, template="plotly_dark", title_x=0.5)
                st.plotly_chart(fig_bar, use_container_width=True)

                if recid_col:
                    recid_means = df.groupby(col_name)[recid_col].mean()
                    fig_recid = px.bar(
                        x=recid_means.index, y=recid_means.values,
                        labels={"x": col_name.replace('_', ' '), "y": "Ortalama Recidivism Oranı"},
                        title=f"{col_name.replace('_',' ')} Bazında Yeniden Suç İşleme Oranı",
                        color=recid_means.index,
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )
                    fig_recid.update_layout(showlegend=False, template="plotly_dark", title_x=0.5, yaxis=dict(range=[0, 1]))
                    st.plotly_chart(fig_recid, use_container_width=True)

                st.markdown(info_icon(f"{col_name} dağılımı ve ilgili yeniden suç işleme oranları."))
            else:
                st.info(f"{col_name} verisi bulunamadı.")

    st.markdown("---")

    # Korelasyon Grafiği
    st.subheader("📊 Özelliklerin Yeniden Suç İşleme ile Korelasyonu")

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if recid_col in numeric_cols:
        numeric_cols.remove(recid_col)

    try:
        corr = df[numeric_cols + [recid_col]].corr()[recid_col].drop(recid_col)
    except:
        corr = None

    if corr is not None and not corr.empty:
        corr_df = pd.DataFrame(corr).reset_index()
        corr_df.columns = ["Özellik", "Recidivism Korelasyonu"]
        corr_df = corr_df.sort_values(by="Recidivism Korelasyonu", key=abs, ascending=False)

        c1, c2 = st.columns([3, 1])
        with c1:
            fig_corr = px.bar(
                corr_df, x="Özellik", y="Recidivism Korelasyonu",
                color="Recidivism Korelasyonu",
                color_continuous_scale=px.colors.diverging.RdBu,
                title="Özelliklerin Yeniden Suç İşleme ile Korelasyonu"
            )
            fig_corr.update_layout(template="plotly_dark", title_x=0.5)
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
