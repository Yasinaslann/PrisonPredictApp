import streamlit as st
import pandas as pd
import plotly.express as px

def home_page(df):
    # ... üst kısımlar önceki gibi (başlık, kartlar)

    st.markdown("---")

    # --- Yeni modern veri önizleme ve filtreleme ---
    st.subheader("📂 Veri Seti Önizleme ve Filtreleme")

    if df is not None and not df.empty:
        st.dataframe(df, use_container_width=True, height=300)
    else:
        st.info("Veri seti boş veya yüklenemedi.")

    st.markdown("---")

    # --- Yeniden Suç İşleme Oranı modern ---
    st.subheader("🎯 Yeniden Suç İşleme Oranı")

    if "Recidivism" in df.columns:
        recid_counts = df["Recidivism"].value_counts(dropna=False).sort_index()
        total = recid_counts.sum()
        again_count = recid_counts.get(1, 0)
        no_again_count = recid_counts.get(0, 0)
        again_pct = (again_count / total)*100 if total > 0 else 0

        col1, col2 = st.columns([3,1])
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
                <div style="background:#f0f4f8; padding:1rem; border-radius:10px; text-align:center; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
                    <h3 style="color:#b71c1c;">⚠️ Yeniden Suç İşleme</h3>
                    <p><strong>{again_count:,}</strong> kişi tekrar suç işlemiş.</p>
                    <p><strong>{no_again_count:,}</strong> kişi tekrar suç işlememiş.</p>
                    <p><strong>%{again_pct:.1f}</strong> oranında risk var.</p>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Yeniden suç işleme verisi bulunmamaktadır.")

    st.markdown("---")

    # --- Demografik Dağılımlar + Recidivism oranları (interaktif seçimli grafik) ---
    st.subheader("👥 Demografik Dağılımlar ve Yeniden Suç İşleme Oranları")

    demo_options = []
    if "Gender" in df.columns:
        demo_options.append("Gender")
    if "Education_Level" in df.columns:
        demo_options.append("Education_Level")

    if demo_options:
        choice = st.selectbox("Gösterilecek Demografik Özellik", demo_options)

        counts = df[choice].value_counts()
        recid_col = "Recidivism" if "Recidivism" in df.columns else None

        col1, col2 = st.columns(2)
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

    # --- Özelliklerin Korelasyonu ---
    st.subheader("📊 Özelliklerin Yeniden Suç İşleme ile Korelasyonu")

    recid_col = "Recidivism" if "Recidivism" in df.columns else None
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

    # --- Ortalama Tahliye Yaşı ---
    st.subheader("👤 Ortalama Tahliye Yaşı")

    avg_age = safe_mean(df["Age_at_Release"]) if "Age_at_Release" in df.columns else None
    if avg_age is not None:
        st.metric(label="Ortalama Tahliye Yaşı", value=f"{avg_age:.1f}")
    else:
        st.info("Ortalama tahliye yaşı verisi bulunmamaktadır.")
