import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import shap
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, RocCurveDisplay
)
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Recidivism Risk Tahmin & Analiz", layout="wide", initial_sidebar_state="expanded")

BASE_PATH = Path(__file__).parent
MODEL_PATH = BASE_PATH / "catboost_model.pkl"
BOOL_COLS_PATH = BASE_PATH / "bool_columns.pkl"
CAT_FEATURES_PATH = BASE_PATH / "cat_features.pkl"
FEATURE_NAMES_PATH = BASE_PATH / "feature_names.pkl"
CAT_UNIQUE_VALUES_PATH = BASE_PATH / "cat_unique_values.pkl"
DATA_PATH = BASE_PATH / "Prisongüncelveriseti.csv"

FEATURE_HELP = {
    "Gender": "Mahkumun cinsiyeti. Erkek veya Kadın olabilir.",
    "Race": "Mahkumun ırkı.",
    "Age_at_Release": "Mahkumun serbest bırakıldığı yaş.",
    "Gang_Affiliated": "Çete bağlantısı (True/Hayır).",
    "Education_Level": "Mahkumun eğitim seviyesi.",
    "Prison_Years": "Mahkumun hapiste geçirdiği yıl sayısı.",
    # İstersen buraya diğer özellik açıklamalarını da ekle.
}

@st.cache_resource
def load_resources():
    model = joblib.load(MODEL_PATH)
    bool_cols = joblib.load(BOOL_COLS_PATH) if BOOL_COLS_PATH.exists() else []
    cat_features = joblib.load(CAT_FEATURES_PATH) if CAT_FEATURES_PATH.exists() else []
    feature_names = joblib.load(FEATURE_NAMES_PATH) if FEATURE_NAMES_PATH.exists() else getattr(model, "feature_names_", None)
    cat_unique_values = joblib.load(CAT_UNIQUE_VALUES_PATH) if CAT_UNIQUE_VALUES_PATH.exists() else {}
    df = pd.read_csv(DATA_PATH)
    return model, bool_cols, cat_features, feature_names, cat_unique_values, df

resources = load_resources()
model, bool_cols, cat_features, feature_names, cat_unique_values, df_data = resources

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# --- Ana Sayfa ---
def home_page():
    st.title("🏛️ Recidivism Risk Tahmin & Analiz Platformu")
    st.markdown("""
    ### Proje Tanıtımı ve Dataset Hikayesi

    Bu proje, ABD mahkumlarının 3 yıl içinde tekrar suç işleme riskini tahmin etmek için geliştirilmiş bir makine öğrenimi uygulamasıdır.

    **Dataset İçeriği:**  
    - Demografik veriler: Cinsiyet, ırk, yaş vb.  
    - Suç geçmişi ve ceza bilgileri  
    - Eğitim, çete bağlantısı ve diğer sosyal faktörler

    **Uygulama Sayfaları:**  
    1. **Tahmin:** Mahkum özelliklerine göre suç işleme riskini anlık tahmin edebilirsiniz.  
    2. **Veri Analizi:** Dataset üzerinde interaktif filtreler ve gelişmiş görselleştirmeler.  
    3. **Model Performansı:** Modelin doğruluk, hassasiyet, ROC eğrisi gibi metrikleri.

    ### Projenin Önemi  
    Mahkumların tekrar suç işlemesi toplumsal güvenlik açısından önemli bir konudur.  
    Bu model, kaynakların etkin kullanımı ve müdahalelerin hedeflenmesi için destek sağlar.

    ### İleriye Dönük Gelişimler  
    - Farklı modellerle karşılaştırma  
    - Veri seti güncellemeleri ve yeni özellikler  
    - Model açıklanabilirliğinin artırılması (Explainable AI)

    """)
    st.info("Uygulama sol menüden diğer sayfalara geçiş yapabilirsiniz.")

# --- Tahmin Sayfası ---
def prediction_page():
    st.title("🔮 Recidivism Risk Tahmini")
    st.markdown("Her alanın yanında bulunan `?` işaretine tıklayarak o özelliğin anlamını görebilirsiniz. \n\n Değerleri girdikten sonra **Tahmin Yap** butonuna basınız.")

    input_data = {}
    cols = st.columns(2)

    for i, feat in enumerate(feature_names):
        container = cols[i % 2]
        with container:
            label = f"{feat}"
            help_text = FEATURE_HELP.get(feat, "Bu alan hakkında bilgi bulunmamaktadır.")
            if feat in bool_cols:
                val = st.selectbox(label, options=["False", "True"], index=0, help=help_text)
            elif feat in cat_features:
                options = cat_unique_values.get(feat, [])
                val = st.selectbox(label, options=options, index=0 if options else -1, help=help_text)
            else:
                col_min = float(df_data[feat].min()) if feat in df_data.columns else 0.0
                col_max = float(df_data[feat].max()) if feat in df_data.columns else 100.0
                default_val = float(df_data[feat].median()) if feat in df_data.columns else col_min
                val = st.number_input(label, value=default_val, min_value=col_min, max_value=col_max, help=help_text)
            input_data[feat] = val

    if st.button("Tahmin Yap"):
        try:
            df_input = pd.DataFrame([input_data], columns=feature_names)
            for b in bool_cols:
                if b in df_input.columns:
                    df_input[b] = df_input[b].astype(str)
            pred = model.predict(df_input)[0]
            proba = model.predict_proba(df_input)[0][1] if hasattr(model, "predict_proba") else None

            risk_text = "🔴 Yüksek Risk" if int(pred) == 1 else "🟢 Düşük Risk"
            st.success(f"**Tahmin:** {risk_text}")
            if proba is not None:
                st.info(f"Olasılık: **{proba*100:.2f}%**")

            # SHAP Açıklaması
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(df_input)

            st.subheader("Tahmin Açıklaması (SHAP Waterfall Plot)")
            fig, ax = plt.subplots(figsize=(10, 5))
            shap.plots.waterfall(shap.Explanation(values=shap_values[0], base_values=explainer.expected_value, data=df_input.iloc[0]), max_display=10, show=False)
            st.pyplot(fig)

            # SHAP etkilerinin bar chartı
            st.subheader("Özelliklerin Tahmine Etkisi (SHAP Değerleri %)")
            shap_sum = np.abs(shap_values[0]).sum()
            shap_df = pd.DataFrame({
                "Özellik": feature_names,
                "SHAP Değeri (%)": (np.abs(shap_values[0]) / shap_sum * 100).round(2)
            }).sort_values(by="SHAP Değeri (%)", ascending=False)
            st.bar_chart(shap_df.set_index("Özellik")["SHAP Değeri (%)"])

            # Tahmin geçmişine ekle
            rec = input_data.copy()
            rec.update({"Tahmin": int(pred), "Olasılık": proba})
            st.session_state.prediction_history.append(rec)

        except Exception as e:
            st.error(f"Tahmin sırasında hata oluştu: {e}")

    # Tahmin geçmişi ve analiz
    if st.session_state.prediction_history:
        st.markdown("---")
        st.subheader("📋 Tahmin Geçmişi ve Analizleri")

        df_hist = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(df_hist)

        # Özet Analiz
        st.markdown("**Tahmin Geçmişi Özetleri:**")
        risk_counts = df_hist["Tahmin"].value_counts(normalize=True).rename({0:"Düşük Risk",1:"Yüksek Risk"})
        st.write(f"- Toplam tahmin sayısı: {len(df_hist)}")
        st.write(f"- Yüksek risk oranı: {risk_counts.get(1, 0)*100:.2f}%")
        st.write(f"- Düşük risk oranı: {risk_counts.get(0, 0)*100:.2f}%")

        # CSV olarak indir
        csv_data = df_hist.to_csv(index=False).encode("utf-8")
        st.download_button("Tahmin Geçmişini CSV Olarak İndir", csv_data, "tahmin_gecmisi.csv", "text/csv")

# --- Veri Analizi ---
def analysis_page(df):
    st.title("📊 Gelişmiş Veri Görselleştirme ve Analiz")

    # Güvenlik: sütun var mı kontrolü
    if "Age_at_Release" not in df.columns:
        st.error("Veride 'Age_at_Release' sütunu bulunamadı! Lütfen datasetinizi kontrol edin.")
        return

    age_min = int(df["Age_at_Release"].dropna().min())
    age_max = int(df["Age_at_Release"].dropna().max())

    age_range = st.sidebar.slider("Yaş Aralığı", age_min, age_max, (age_min, age_max))
    gender_filter = st.sidebar.multiselect("Cinsiyet", options=df["Gender"].dropna().unique(), default=df["Gender"].dropna().unique())
    race_filter = st.sidebar.multiselect("Irk", options=df["Race"].dropna().unique(), default=df["Race"].dropna().unique())
    education_filter = st.sidebar.multiselect("Eğitim Seviyesi", options=df["Education_Level"].dropna().unique(), default=df["Education_Level"].dropna().unique())
    gang_filter = st.sidebar.multiselect("Çete Bağlılığı", options=df["Gang_Affiliated"].dropna().unique(), default=df["Gang_Affiliated"].dropna().unique())

    df_filtered = df[
        (df["Age_at_Release"] >= age_range[0]) & (df["Age_at_Release"] <= age_range[1]) &
        (df["Gender"].isin(gender_filter)) &
        (df["Race"].isin(race_filter)) &
        (df["Education_Level"].isin(education_filter)) &
        (df["Gang_Affiliated"].isin(gang_filter))
    ]

    st.write(f"**Filtrelenmiş Kayıt Sayısı:** {df_filtered.shape[0]}")

    # Grafikler Plotly ile interaktif ve hover destekli
    fig1 = px.histogram(df_filtered, x="Age_at_Release", nbins=30, color="Gender", barmode="overlay", title="Yaş Dağılımı")
    st.plotly_chart(fig1, use_container_width=True)

    if not df_filtered.empty:
        fig2 = px.pie(df_filtered, names="Gender", title="Cinsiyet Oranları")
        st.plotly_chart(fig2, use_container_width=True)

        race_count = df_filtered["Race"].value_counts().reset_index()
        race_count.columns = ["Irk", "Sayısı"]
        fig3 = px.bar(race_count, x="Irk", y="Sayısı", title="Irk Dağılımı")
        st.plotly_chart(fig3, use_container_width=True)

        edu_count = df_filtered["Education_Level"].value_counts().reset_index()
        edu_count.columns = ["Eğitim Seviyesi", "Sayısı"]
        fig4 = px.bar(edu_count, x="Eğitim Seviyesi", y="Sayısı", title="Eğitim Seviyesi Dağılımı")
        st.plotly_chart(fig4, use_container_width=True)

        if "Prison_Years" in df_filtered.columns and "Recidivism_Within_3years" in df_filtered.columns:
            fig5 = px.box(df_filtered, x="Recidivism_Within_3years", y="Prison_Years",
                          color="Recidivism_Within_3years",
                          labels={"Recidivism_Within_3years": "3 Yıl İçinde Yeniden Suç", "Prison_Years": "Ceza Süresi (Yıl)"},
                          title="Ceza Süresi ve Suç Tekrarı İlişkisi")
            st.plotly_chart(fig5, use_container_width=True)

        num_cols = df_filtered.select_dtypes(include=["float64", "int64"]).columns.tolist()
        if len(num_cols) > 1:
            st.subheader("Sayısal Değişkenler Korelasyon Matrisi")
            corr = df_filtered[num_cols].corr()
            fig6, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig6)

        if "Recidivism_Arrest_Year1" in df_filtered.columns:
            ts_data = df_filtered.groupby("Recidivism_Arrest_Year1").size().reset_index(name="Suç Sayısı")
            fig7 = px.line(ts_data, x="Recidivism_Arrest_Year1", y="Suç Sayısı", title="Yıllara Göre Suç Tekrar Sayısı")
            st.plotly_chart(fig7, use_container_width=True)
    else:
        st.warning("Seçilen filtrelerde kayıt bulunamadı.")

    st.markdown("""
    ---
    **Notlar:**  
    - Filtrelerle belirli gruplar üzerinde detaylı analiz yapabilirsiniz.  
    - Korelasyon matrisi değişkenler arasındaki ilişkileri gösterir.  
    - Zaman serisi analizleri suç tekrar trendlerini sunar.
    """)

# --- Model Performans Sayfası ---
def performance_page(df):
    st.title("📈 Model Performansı ve Değerlendirme")

    if "Recidivism_Within_3years" not in df.columns:
        st.error("Performans analizi için hedef değişken (Recidivism_Within_3years) bulunamadı!")
        return

    y_true = df["Recidivism_Within_3years"].astype(int)
    X = df[feature_names]

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_proba) if y_proba is not None else None
    cm = confusion_matrix(y_true, y_pred)

    st.markdown(f"""
    ### Başarı Metrikleri
    - **Doğruluk (Accuracy):** {acc:.3f}
    - **Hassasiyet (Precision):** {prec:.3f}
    - **Duyarlılık (Recall):** {rec:.3f}
    - **F1 Skoru:** {f1:.3f}
    - **ROC AUC:** {roc_auc:.3f if roc_auc is not None else 'Modelde olasılık yok'}
    """)

    st.subheader("Karışıklık Matrisi (Confusion Matrix)")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Düşük Risk", "Yüksek Risk"],
                yticklabels=["Düşük Risk", "Yüksek Risk"])
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")
    st.pyplot(fig)

    if y_proba is not None:
        st.subheader("ROC Eğrisi")
        fig2, ax2 = plt.subplots()
        RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax2)
        st.pyplot(fig2)

    st.markdown("""
    ---
    **Not:**  
    Model performansı mevcut veri üzerinde hesaplanmıştır. İlerleyen aşamalarda model optimizasyonu yapılabilir.
    """)

def main():
    pages = {
        "Ana Sayfa": home_page,
        "Tahmin": prediction_page,
        "Veri Analizi": lambda: analysis_page(df_data),
        "Model Performansı": lambda: performance_page(df_data)
    }
    st.sidebar.title("📋 Menü")
    choice = st.sidebar.radio("Sayfa Seçimi", list(pages.keys()))
    pages[choice]()

if __name__ == "__main__":
    main()
