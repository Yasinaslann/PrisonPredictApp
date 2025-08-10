import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay
)
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Recidivism Risk Prediction", layout="wide", initial_sidebar_state="expanded")

# ---------------------- PATHS ----------------------
BASE_PATH = Path(__file__).parent
MODEL_PATH = BASE_PATH / "catboost_model.pkl"
DATA_PATH = BASE_PATH / "Prisongüncelveriseti.csv"

# ---------------------- LOAD DATA & MODEL ----------------------
@st.cache_resource
def load_resources():
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    return model, df

model, df_data = load_resources()

# ---------------------- FEATURE INFO (TOOLTIPS) ----------------------
FEATURE_HELP = {
    "Gender": "Mahkumun cinsiyeti. Erkek veya Kadın.",
    "Race": "Mahkumun ırkı.",
    "Age_at_Release": "Mahkumun serbest bırakıldığı yaş.",
    "Gang_Affiliated": "Çete bağlantısı (Evet / Hayır).",
    "Education_Level": "Mahkumun eğitim seviyesi.",
    "Prison_Years": "Mahkumun hapiste geçirdiği yıl sayısı.",
    "Recidivism_Within_3years": "3 yıl içinde yeniden suç işleme durumu (1: Evet, 0: Hayır).",
    # Buraya datasetin diğer sütunlarını da ekleyebilirsin
}

FEATURES = [col for col in df_data.columns if col != "Recidivism_Within_3years"]

# ---------------------- SESSION STATE ----------------------
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# ==================== ANA SAYFA ====================
def home_page():
    st.title("🏛️ Recidivism Risk Tahmin ve Analiz Uygulaması")
    st.markdown("""
    ### Proje Hakkında

    Bu uygulama, ABD mahkumlarının 3 yıl içinde tekrar suç işleme riskini tahmin etmek amacıyla hazırlanmıştır.
    Veri seti mahkumların demografik bilgileri, suç geçmişi, eğitim durumu ve diğer sosyal parametreleri içermektedir.

    **Amaç:**  
    - Toplumsal güvenliği artırmak,  
    - Cezaevi sonrası risk değerlendirmesi yapmak,  
    - Kaynakları daha etkin kullanmak için destek sağlamak.

    **Uygulama Bölümleri:**  
    - **Tahmin:** Mahkum bilgileri ile risk tahmini ve açıklaması.  
    - **Veri Analizi:** Gelişmiş interaktif grafikler ve veri keşfi.  
    - **Model Performansı:** Modelin kapsamlı metriklerle değerlendirilmesi.

    ---
    """)
    st.info("Sol menüden diğer sayfalara geçiş yapabilirsiniz.")

# ==================== TAHMİN SAYFASI ====================
def prediction_page():
    st.title("🔮 Recidivism Risk Tahmini")
    st.write("Her alanın yanında `?` işaretine tıklayarak açıklamalarını görebilirsiniz. Değerleri girdikten sonra 'Tahmin Yap' butonuna basınız.")

    # Dinamik input oluşturma
    input_data = {}
    cols = st.columns(2)

    for i, feat in enumerate(FEATURES):
        with cols[i % 2]:
            label = f"{feat}"
            help_text = FEATURE_HELP.get(feat, "Bu özellik hakkında bilgi bulunmamaktadır.")
            # Veri tipi ve benzersiz değerlerine göre seçim kutusu ya da input
            if df_data[feat].dtype == "bool" or df_data[feat].dropna().value_counts().index.isin([True, False]).all():
                options = ["False", "True"]
                default_idx = 0
                val = st.selectbox(label, options=options, index=default_idx, help=help_text)
                input_data[feat] = val == "True"
            elif df_data[feat].dtype == "object" or len(df_data[feat].unique()) < 20:
                options = list(df_data[feat].dropna().unique())
                default_idx = 0 if options else -1
                val = st.selectbox(label, options=options, index=default_idx, help=help_text)
                input_data[feat] = val
            else:
                min_val = float(df_data[feat].min())
                max_val = float(df_data[feat].max())
                median_val = float(df_data[feat].median())
                val = st.number_input(label, min_value=min_val, max_value=max_val, value=median_val, help=help_text)
                input_data[feat] = val

    if st.button("Tahmin Yap"):
        try:
            df_input = pd.DataFrame([input_data])
            pred = model.predict(df_input)[0]
            proba = model.predict_proba(df_input)[0][1] if hasattr(model, "predict_proba") else None

            risk_label = "🔴 Yüksek Risk" if pred == 1 else "🟢 Düşük Risk"
            st.success(f"**Tahmin Sonucu:** {risk_label}")

            if proba is not None:
                st.info(f"Olasılık: **{proba*100:.2f}%**")

            # SHAP Açıklaması
            st.subheader("Tahmin Açıklaması - Özelliklerin Etkisi")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(df_input)

            fig, ax = plt.subplots(figsize=(10, 5))
            shap.plots.waterfall(shap.Explanation(
                values=shap_values[0], base_values=explainer.expected_value, data=df_input.iloc[0]
            ), max_display=10, show=False)
            st.pyplot(fig)

            # Tahmin geçmişine ekle
            rec = input_data.copy()
            rec.update({"Tahmin": int(pred), "Olasılık": proba})
            st.session_state.prediction_history.append(rec)

        except Exception as e:
            st.error(f"Tahmin sırasında hata oluştu: {e}")

    # Tahmin geçmişi gösterimi
    if st.session_state.prediction_history:
        st.markdown("---")
        st.subheader("📋 Tahmin Geçmişi ve Özet")

        df_hist = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(df_hist)

        # Basit Özetler
        st.markdown("**Genel Özet:**")
        total_preds = len(df_hist)
        high_risk_count = (df_hist["Tahmin"] == 1).sum()
        low_risk_count = (df_hist["Tahmin"] == 0).sum()
        st.write(f"- Toplam tahmin sayısı: {total_preds}")
        st.write(f"- Yüksek risk tahmin sayısı: {high_risk_count} (%{high_risk_count/total_preds*100:.2f})")
        st.write(f"- Düşük risk tahmin sayısı: {low_risk_count} (%{low_risk_count/total_preds*100:.2f})")

        csv_data = df_hist.to_csv(index=False).encode("utf-8")
        st.download_button("Tahmin Geçmişini CSV Olarak İndir", csv_data, "tahmin_gecmisi.csv", "text/csv")

# ==================== GELİŞMİŞ VERİ ANALİZİ ====================
def analysis_page(df):
    st.title("📊 Gelişmiş Veri Görselleştirme ve Analiz")

    st.write("""
    Bu sayfada veri setindeki çeşitli özelliklerin dağılımlarını, korelasyonlarını ve kategorik değişkenlerin etkilerini interaktif grafiklerle keşfedebilirsiniz.
    """)

    # Güvenlik Kontrolleri
    for col in ["Age_at_Release", "Gender", "Race", "Education_Level", "Gang_Affiliated"]:
        if col not in df.columns:
            st.error(f"Veride gerekli sütun bulunamadı: {col}")
            return

    # Filtreleme paneli
    st.sidebar.header("Filtreler")

    age_min = int(df["Age_at_Release"].min())
    age_max = int(df["Age_at_Release"].max())
    age_range = st.sidebar.slider("Yaş Aralığı", age_min, age_max, (age_min, age_max))

    gender_options = list(df["Gender"].dropna().unique())
    gender_filter = st.sidebar.multiselect("Cinsiyet", gender_options, default=gender_options)

    race_options = list(df["Race"].dropna().unique())
    race_filter = st.sidebar.multiselect("Irk", race_options, default=race_options)

    education_options = list(df["Education_Level"].dropna().unique())
    education_filter = st.sidebar.multiselect("Eğitim Seviyesi", education_options, default=education_options)

    gang_options = list(df["Gang_Affiliated"].dropna().unique())
    gang_filter = st.sidebar.multiselect("Çete Bağlılığı", gang_options, default=gang_options)

    df_filtered = df[
        (df["Age_at_Release"] >= age_range[0]) & (df["Age_at_Release"] <= age_range[1]) &
        (df["Gender"].isin(gender_filter)) &
        (df["Race"].isin(race_filter)) &
        (df["Education_Level"].isin(education_filter)) &
        (df["Gang_Affiliated"].isin(gang_filter))
    ]

    st.write(f"**Filtrelenmiş Kayıt Sayısı:** {df_filtered.shape[0]}")

    if df_filtered.empty:
        st.warning("Seçilen filtrelere uygun kayıt bulunamadı.")
        return

    # Yaş dağılımı histogramı
    fig_age = px.histogram(df_filtered, x="Age_at_Release", nbins=30, color="Gender",
                           title="Yaş Dağılımı ve Cinsiyet")
    st.plotly_chart(fig_age, use_container_width=True)

    # Kategorik dağılımlar: Race
    race_counts = df_filtered["Race"].value_counts().reset_index()
    race_counts.columns = ["Irk", "Sayısı"]
    fig_race = px.bar(race_counts, x="Irk", y="Sayısı", color="Irk",
                      title="Irk Dağılımı", text="Sayısı")
    st.plotly_chart(fig_race, use_container_width=True)

    # Eğitim seviyesi dağılımı
    edu_counts = df_filtered["Education_Level"].value_counts().reset_index()
    edu_counts.columns = ["Eğitim Seviyesi", "Sayısı"]
    fig_edu = px.pie(edu_counts, names="Eğitim Seviyesi", values="Sayısı", title="Eğitim Seviyesi Dağılımı")
    st.plotly_chart(fig_edu, use_container_width=True)

    # Ceza süresi ve tekrar suç ilişkisinin kutu grafiği
    if "Prison_Years" in df_filtered.columns and "Recidivism_Within_3years" in df_filtered.columns:
        fig_box = px.box(df_filtered, x="Recidivism_Within_3years", y="Prison_Years",
                         color="Recidivism_Within_3years",
                         labels={"Recidivism_Within_3years": "3 Yıl İçinde Yeniden Suç", "Prison_Years": "Ceza Süresi (Yıl)"},
                         title="Ceza Süresi ve Suç Tekrarı İlişkisi")
        st.plotly_chart(fig_box, use_container_width=True)

    # Korelasyon matrisi
    numeric_cols = df_filtered.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols) > 1:
        st.subheader("Sayısal Değişkenler Korelasyon Matrisi")
        corr = df_filtered[numeric_cols].corr()
        fig_corr, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(corr, annot=True, cmap="RdBu_r", ax=ax, center=0)
        st.pyplot(fig_corr)

    st.markdown("""
    ---
    **Not:**  
    - Filtreleri kullanarak veriyi daraltabilir, farklı segmentlerin karşılaştırmasını yapabilirsiniz.  
    - Korelasyon matrisi ile değişkenlerin birbirleriyle ilişkisini görebilirsiniz.  
    - Grafikleri inceleyerek önemli trend ve dağılımları analiz edebilirsiniz.
    """)

# ==================== MODEL PERFORMANS ====================
def performance_page(df):
    st.title("📈 Model Performansı ve Değerlendirme")

    if "Recidivism_Within_3years" not in df.columns:
        st.error("Performans değerlendirmesi için hedef değişken (Recidivism_Within_3years) bulunamadı!")
        return

    y_true = df["Recidivism_Within_3years"].astype(int)
    X = df[FEATURES]

    # Model tahminleri
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None

    # Metrikler
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_proba) if y_proba is not None else None

    # Performans metrikleri gösterimi
    st.markdown(f"""
    ### Temel Başarı Metrikleri
    - **Doğruluk (Accuracy):** {acc:.3f}
    - **Hassasiyet (Precision):** {prec:.3f}
    - **Duyarlılık (Recall):** {rec:.3f}
    - **F1 Skoru:** {f1:.3f}
    - **ROC AUC:** {roc_auc:.3f if roc_auc is not None else 'Modelde olasılık yok'}
    """)

    # Karışıklık matrisi ve görselleştirme
    st.subheader("Karışıklık Matrisi (Confusion Matrix)")
    cm = confusion_matrix(y_true, y_pred)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm,
                xticklabels=["Düşük Risk", "Yüksek Risk"],
                yticklabels=["Düşük Risk", "Yüksek Risk"])
    ax_cm.set_xlabel("Tahmin")
    ax_cm.set_ylabel("Gerçek")
    st.pyplot(fig_cm)

    # ROC eğrisi
    if y_proba is not None:
        st.subheader("ROC Eğrisi")
        fig_roc, ax_roc = plt.subplots()
        RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax_roc)
        st.pyplot(fig_roc)

    # Precision-Recall Eğrisi
    if y_proba is not None:
        st.subheader("Precision-Recall Eğrisi")
        fig_pr, ax_pr = plt.subplots()
        PrecisionRecallDisplay.from_predictions(y_true, y_proba, ax=ax_pr)
        st.pyplot(fig_pr)

    # Sınıf dağılımı analizi
    st.subheader("Sınıf Dağılımı")
    counts = y_true.value_counts()
    fig_count = px.bar(x=["Düşük Risk", "Yüksek Risk"], y=[counts.get(0,0), counts.get(1,0)], title="Gerçek Sınıf Dağılımı", labels={"x":"Sınıf", "y":"Kayıt Sayısı"})
    st.plotly_chart(fig_count, use_container_width=True)

    st.markdown("""
    ---
    **Notlar:**  
    - Model metrikleri eğitim verisi üzerinde hesaplanmıştır.  
    - Metriklere göre modelin güçlü ve zayıf yönleri analiz edilmelidir.  
    - ROC ve Precision-Recall eğrileri modelin ayrıştırma gücünü gösterir.
    """)

# ==================== UYGULAMA ÇALIŞTIRMA ====================
def main():
    pages = {
        "Ana Sayfa": home_page,
        "Tahmin": prediction_page,
        "Veri Analizi": lambda: analysis_page(df_data),
        "Model Performansı": lambda: performance_page(df_data),
    }

    st.sidebar.title("🗂️ Menü")
    choice = st.sidebar.radio("Sayfa Seçiniz", list(pages.keys()))
    pages[choice]()

if __name__ == "__main__":
    main()
