# Streamlit multi-page skeleton for PrisonPredictApp
# Place at repo root: app.py
# Place pages in folder: pages/
# Each page should implement a `render()` function which the root app will call.
# This file shows small ready-to-run skeletons for 4 pages:
# 1) pages/1_home.py           -> Anasayfa / Giriş / KPI / Hızlı EDA
# 2) pages/2_prediction.py     -> Tahmin (single & batch) + model seçimi
# 3) pages/3_recommendations.py-> Tavsiye sistemi & profil analizi (skeleton)
# 4) pages/4_eda.py           -> Detaylı EDA, grafikler, harita
#
# USAGE:
# - Create a folder named `pages` next to this `app.py` or deploy both files in your project.
# - Ensure `pages/__init__.py` exists (can be empty) so Python treats it as a package.
# - Run: `streamlit run app.py`

# -------------------- FILE: app.py --------------------
import streamlit as st
from pages import page_home, page_prediction, page_recommendations, page_eda

PAGES = {
    "Anasayfa": page_home,
    "Tahmin": page_prediction,
    "Tavsiye & Profil": page_recommendations,
    "EDA & Harita": page_eda,
}

st.set_page_config(page_title="PrisonPredictApp", layout="wide")

st.sidebar.title("PrisonPredictApp — Menü")
selection = st.sidebar.radio("Sayfalar", list(PAGES.keys()))

page = PAGES[selection]
page.render()


# -------------------- FILE: pages/__init__.py --------------------
# (This file can be empty; it's here so `pages` is a package.)


# -------------------- FILE: pages/page_home.py --------------------
import streamlit as st
import pandas as pd
import plotly.express as px
import io
from pathlib import Path

DEFAULT_PATHS = [
    "/mnt/data/Prisongüncelveriseti.csv",
    "/mnt/data/NIJ_s_Recidivism_Challenge_Full_Dataset_20250729.csv",
]


def find_default_dataset():
    for p in DEFAULT_PATHS:
        if Path(p).exists():
            return p
    return None


def render():
    st.title("Anasayfa — Recidivism Tahmini")
    st.markdown("Bu panel, proje hikayesi, veri özetleri ve hızlı görseller içerir.")
    st.warning("**Etik uyarı:** Bu içerik araştırma amaçlıdır. Gerçek uygulama öncesi hukuki/etik değerlendirme gereklidir.")

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded = st.file_uploader("CSV dosyası yükle", type=["csv"], key="home_upload")
        use_default = st.checkbox("Varsayılan örnek dosyayı kullan (varsa)", value=True)
        df = None
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
            except Exception as e:
                st.error(f"Yükleme okunurken hata: {e}")
        elif use_default:
            p = find_default_dataset()
            if p:
                df = pd.read_csv(p)
                st.caption(f"Varsayılan dosya yüklendi: {p}")

        if df is not None:
            st.subheader("Veri Önizleme (ilk 10 satır)")
            st.dataframe(df.head(10))

    with col2:
        st.subheader("Proje Özeti")
        st.markdown("- Hedef: Tahliye sonrası yeniden tutuklanma tahmini
- Veri: Kaynak, tarih aralığı, satır sayısı vb. (buraya kısa açıklama ekle)")

    if df is not None:
        st.subheader("Hızlı KPI'lar")
        total = len(df)
        missing_pct = df.isna().mean().mean() * 100
        # find candidate target automatically (try common names)
        for cand in ["recidivism", "rearrest", "target", "reoffend"]:
            if cand in df.columns:
                target_col = cand
                break
        else:
            target_col = None

        c1, c2, c3 = st.columns(3)
        c1.metric("Satır sayısı", f"{total:,}")
        c2.metric("Ortalama eksik değer oranı", f"{missing_pct:.2f}%")
        if target_col:
            pos_rate = df[target_col].mean()
            c3.metric(f"Pozitif oran ({target_col})", f"{pos_rate:.3f}")
        else:
            c3.metric("Pozitif oran (target)", "N/A")

        st.markdown("---")
        st.subheader("Bazı görseller")
        # target distribution
        if target_col:
            vc = df[target_col].value_counts().reset_index()
            vc.columns = ["value", "count"]
            fig = px.bar(vc, x='value', y='count', title=f'{target_col} dağılımı')
            st.plotly_chart(fig, use_container_width=True)

        # example numeric histogram
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        if num_cols:
            sel = st.selectbox("Histogram için değişken seç", options=num_cols, index=0)
            fig2 = px.histogram(df, x=sel, nbins=40, title=f"{sel} dağılımı")
            st.plotly_chart(fig2, use_container_width=True)

        # download filtered data
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        st.download_button("Tüm veriyi indir (CSV)", data=buf, file_name='dataset_export.csv')

    else:
        st.info("Henüz veri yüklenmedi. Soldan veya yukarıdan yükleyin.")


# -------------------- FILE: pages/page_prediction.py --------------------
import streamlit as st
import pandas as pd
import joblib
from io import BytesIO


def render():
    st.title("Tahmin — Prediction")
    st.markdown("Tekil örnek tahmini veya toplu (CSV) tahmin yapabilirsiniz.")

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded = st.file_uploader("Toplu tahmin için CSV yükle (model için aynı feature set olmalı)", type=["csv"], key='pred_batch')
        if uploaded is not None:
            try:
                df_batch = pd.read_csv(uploaded)
                st.write("Yüklenen örnek satırları:")
                st.dataframe(df_batch.head(5))
            except Exception as e:
                st.error(f"CSV okunamadı: {e}")
                df_batch = None
        else:
            df_batch = None

        st.markdown("---")
        st.subheader("Tekil örnek tahmini")
        st.markdown("Form üzerinden bir örnek gir ve modelin verdiği olasılığı görüntüle.")
        # NOTE: burada form alanlarını otomatik üretmek için model özellik listesine ihtiyaç var.
        st.info("Bu iskelette örnek form yok; daha sonra feature listesine göre otomatik alan çıkarılacak.")

    with col2:
        st.subheader("Model yükle")
        model_file = st.file_uploader("Model (.joblib) yükle", type=["joblib", "pkl"], key='model_upload')
        model = None
        if model_file is not None:
            try:
                model = joblib.load(model_file)
                st.success("Model yüklendi.")
            except Exception as e:
                st.error(f"Model yüklenirken hata: {e}")

        st.markdown("---")
        if df_batch is not None and model is not None:
            if st.button("Toplu tahmini çalıştır"):
                try:
                    probs = model.predict_proba(df_batch)[:, 1]
                    df_out = df_batch.copy()
                    df_out['pred_proba'] = probs
                    st.dataframe(df_out.head())
                    csv = df_out.to_csv(index=False).encode('utf-8')
                    st.download_button("Tahmin sonuçlarını indir (CSV)", data=csv, file_name='predictions.csv', mime='text/csv')
                except Exception as e:
                    st.error(f"Tahmin sırasında hata: {e}")
        elif model is None:
            st.info("Tahmin yapabilmek için model yükleyin.")


# -------------------- FILE: pages/page_recommendations.py --------------------
import streamlit as st
import pandas as pd


def render():
    st.title("Tavsiye Sistemi & Profil Analizi")
    st.markdown("Bu sayfa: benzer profillere göre öneriler, cluster analizi ve what-if senaryoları için iskelet sunar.")

    st.info("Şu anda bu sayfa skeleton durumunda. Veri ve hedefler doğrultusunda iki yaklaşım önerilir: (1) rule-based / literature-based öneriler veya (2) similarity-based (KNN) öneriler.")

    st.subheader("Profile göre öneri (örnek akış)")
    st.markdown("1. Bir kişi seç / form ile oluştur.  2. Benzer kayıtları KNN ile bul.  3. Benzer kişilerin aldıkları müdahaleleri çıkar ve sıralı öneri oluştur.")

    st.subheader("Cluster analizi")
    st.markdown("K-means / HDBSCAN ile kullanıcı segmentleri oluşturulup her segmente özel politika önerileri üretilebilir.")

    st.warning("Bu sayfayı doldurmak için önce veri yüklenmesi ve hedef/özelliklerin netleşmesi gerekir. Hazır olduğunda otomatik kod üretebilirim.")


# -------------------- FILE: pages/page_eda.py --------------------
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

DEFAULT_PATHS = [
    "/mnt/data/Prisongüncelveriseti.csv",
    "/mnt/data/NIJ_s_Recidivism_Challenge_Full_Dataset_20250729.csv",
]


def find_default_dataset():
    for p in DEFAULT_PATHS:
        if Path(p).exists():
            return p
    return None


def render():
    st.title("EDA & Harita — Detaylı Veri Analizi")
    uploaded = st.file_uploader("EDA için CSV yükle", type=["csv"], key='eda_upload')
    df = None
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        p = find_default_dataset()
        if p:
            df = pd.read_csv(p)
            st.caption(f"Varsayılan dosya yüklendi: {p}")

    if df is None:
        st.info("Önce veri yükleyin veya varsayılan dosyayı kullanın.")
        return

    st.subheader("Özet istatistikler")
    st.write(df.describe(include='all').T)

    st.subheader("Korelasyon matrisi (sayısal)")
    num = df.select_dtypes(include=['number'])
    if not num.empty:
        corr = num.corr()
        st.dataframe(corr)
        st.markdown("(Daha güzel görselleştirme için heatmap eklenebilir.)")
    else:
        st.info("Sayısal sütun yok.")

    # Harita placeholder: lat/lon sütunları varsa göster
    lat_candidates = [c for c in df.columns if c.lower() in ('lat', 'latitude')]
    lon_candidates = [c for c in df.columns if c.lower() in ('lon', 'lng', 'longitude')]
    if lat_candidates and lon_candidates:
        latc = lat_candidates[0]
        lonc = lon_candidates[0]
        map_df = df[[latc, lonc]].dropna()
        if not map_df.empty:
            map_df = map_df.rename(columns={latc: 'lat', lonc: 'lon'})
            fig = px.scatter_mapbox(map_df, lat='lat', lon='lon', zoom=5, height=500)
            fig.update_layout(mapbox_style='open-street-map')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Harita için yeterli koordinat verisi yok.")

    st.markdown("---")
    st.info("Bu EDA iskeletini genişletebilirim: interaktif filtreler, zaman serisi analizleri, group comparisons ve export seçenekleri ekleyebilirim.")
