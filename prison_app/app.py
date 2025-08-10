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
    # --- Üst metin şık modern ---
    st.markdown(
        """
        <div style="padding: 1rem 0;">
            <h1 style="font-weight: 800; color: #0b3d91; margin-bottom: 0.3rem;">🏛️ Yeniden Suç İşleme Tahmin Uygulaması</h1>
            <div style="font-size: 1.1rem; line-height: 1.6; color: #333;">
                <h3 style="margin-bottom: 0.3rem; color: #1a237e;">Proje Amacı</h3>
                <p>Bu uygulama, mahpusların tahliye sonrasında yeniden suç işleme riskini (recidivism)
                veri bilimi ve makine öğrenmesi teknikleri ile tahmin etmeyi amaçlar.</p>
                <p>Amaç, topluma yeniden uyum sürecini iyileştirecek stratejiler geliştirmek ve
                risk analizi yaparak tekrar suç oranlarını azaltmaya katkı sağlamaktır.</p>

                <h3 style="margin-top: 1.2rem; margin-bottom: 0.3rem; color: #1a237e;">Veri Seti Hakkında</h3>
                <p>Veri seti, mahpusların demografik bilgileri, ceza süreleri, geçmiş suç kayıtları ve yeniden suç işleme bilgilerini içermektedir.</p>
                <p>Bu bilgilerle risk faktörleri analiz edilip, model geliştirme için zengin bir kaynak sağlanmıştır.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # --- Modern istatistik kartları ---
    total_rows = df.shape[0] if df is not None else 0
    total_cols = df.shape[1] if df is not None else 0
    unique_offenses = df["Prison_Offense"].nunique() if df is not None and "Prison_Offense" in df.columns else 0

    cols = st.columns(3)

    card_style = """
        background-color: #e8f0fe;
        border-radius: 12px;
        padding: 1.8rem;
        text-align: center;
        box-shadow: 0 6px 12px rgb(13 71 161 / 0.2);
        """

    with cols[0]:
        st.markdown(f"""
        <div style="{card_style}">
            <div style="font-size: 2.8rem; font-weight: 800; color: #0d47a1;">{total_rows:,}</div>
            <div style="font-size: 1.2rem; color: #1565c0; font-weight: 700;">🗂️ Toplam Kayıt</div>
        </div>
        """, unsafe_allow_html=True)

    with cols[1]:
        st.markdown(f"""
        <div style="{card_style}">
            <div style="font-size: 2.8rem; font-weight: 800; color: #0d47a1;">{total_cols}</div>
            <div style="font-size: 1.2rem; color: #1565c0; font-weight: 700;">📋 Sütun Sayısı</div>
        </div>
        """, unsafe_allow_html=True)

    with cols[2]:
        st.markdown(f"""
        <div style="{card_style}">
            <div style="font-size: 2.8rem; font-weight: 800; color: #0d47a1;">{unique_offenses}</div>
            <div style="font-size: 1.2rem; color: #1565c0; font-weight: 700;">📌 Farklı Suç Tipi</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # --- Veri seti önizlemesi ---
    with st.expander("📂 Veri Seti Önizlemesi (İlk 10 Satır)"):
        st.dataframe(df.head(10))

    st.markdown("---")

    recid_col = next((c for c in df.columns if "recid" in c.lower()), None)

    # --- Grafik tipi seçimi ---
    st.subheader("📊 Grafik Türü Seçin")
    graph_options = [
        "Yeniden Suç İşleme Oranı (Pasta Grafiği)",
        "Demografik Dağılımlar (Çubuk Grafik)",
        "Özelliklerin Recidivism ile Korelasyonu"
    ]
    selected_graphs = st.multiselect("Gösterilecek grafikler", options=graph_options, default=graph_options)

    # --- Grafikler ---
    if "Yeniden Suç İşleme Oranı (Pasta Grafiği)" in selected_graphs:
        st.subheader("🎯 Yeniden Suç İşleme Oranı Dağılımı")
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
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(title_x=0.5, template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Yeniden suç işleme verisi bulunmamaktadır.")
        with col2:
            st.markdown(info_icon("Bu grafik, tahliye sonrası mahpusların yeniden suç işleme durumunu yüzdesel olarak gösterir."))

        st.markdown("---")

    if "Demografik Dağılımlar (Çubuk Grafik)" in selected_graphs:
        st.subheader("👥 Demografik Dağılımlar ve Recidivism Oranları")
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
                            labels={"x": col_name, "y": "Ortalama Recidivism Oranı"},
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

    if "Özelliklerin Recidivism ile Korelasyonu" in selected_graphs:
        st.subheader("📊 Özellikler Arası Korelasyon (Recidivism ile)")
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
