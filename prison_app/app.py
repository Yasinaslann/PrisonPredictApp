import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import shap
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import io

# --- AYARLAR ---
st.set_page_config(
    page_title="Recidivism Tahmin ve Analiz Uygulaması",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- DOSYA YOLLARI ---
BASE_PATH = Path(__file__).parent
MODEL_PATH = BASE_PATH / "catboost_model.pkl"
BOOL_COLS_PATH = BASE_PATH / "bool_columns.pkl"
CAT_FEATURES_PATH = BASE_PATH / "cat_features.pkl"
FEATURE_NAMES_PATH = BASE_PATH / "feature_names.pkl"
CAT_UNIQUE_VALUES_PATH = BASE_PATH / "cat_unique_values.pkl"
DATA_PATH = BASE_PATH / "Prisongüncelveriseti.csv"

# --- YARDIMCI AÇIKLAMALAR (tooltip gibi) ---
FEATURE_HELP = {
    "Gender": "Mahkumun cinsiyeti (Erkek/Kadın)",
    "Race": "Mahkumun ırkı",
    "Age_at_Release": "Mahkumun serbest bırakıldığı yaş",
    "Gang_Affiliated": "Çete bağlantısı (Evet/Hayır)",
    "Education_Level": "Eğitim seviyesi",
    "Prison_Years": "Ceza süresi (yıl)",
    # ... diğer önemli alanlar için açıklamalar ekleyin
}

# --- VERİ VE MODEL YÜKLEME ---
@st.cache_resource
def load_resources():
    if not MODEL_PATH.exists():
        st.error("Model dosyası bulunamadı: catboost_model.pkl")
        return None

    model = joblib.load(MODEL_PATH)
    bool_cols = joblib.load(BOOL_COLS_PATH) if BOOL_COLS_PATH.exists() else []
    cat_features = joblib.load(CAT_FEATURES_PATH) if CAT_FEATURES_PATH.exists() else []
    feature_names = joblib.load(FEATURE_NAMES_PATH) if FEATURE_NAMES_PATH.exists() else getattr(model, "feature_names_", None)
    cat_unique_values = joblib.load(CAT_UNIQUE_VALUES_PATH) if CAT_UNIQUE_VALUES_PATH.exists() else {}

    if not DATA_PATH.exists():
        st.error("Veri dosyası bulunamadı: Prisongüncelveriseti.csv")
        return None

    df = pd.read_csv(DATA_PATH)

    if feature_names is None:
        st.error("Feature isimleri bulunamadı.")
        return None

    return model, bool_cols, cat_features, feature_names, cat_unique_values, df


resources = load_resources()
if resources is None:
    st.stop()

model, bool_cols, cat_features, feature_names, cat_unique_values, df_data = resources

# --- LOCAL STATE: Tahmin Geçmişi ---
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# --- SAYFA BAŞLIKLARI ---
PAGES = ["Ana Sayfa", "Tahmin", "Veri Analizi", "Model Performansı"]
page = st.sidebar.radio("Sayfa Seçimi", PAGES)

# --- ANA SAYFA ---
def home_page():
    st.title("🏛️ Recidivism Tahmin ve Analiz Uygulamasına Hoşgeldiniz")
    st.markdown("""
    Bu uygulama, mahkumların 3 yıl içinde suç işleme riskini tahmin etmek için geliştirilmiş bir model içerir.
    ---
    ### Uygulama Sayfaları:
    - **Tahmin:** Girdi alanlarını doldurarak risk tahmini yapabilirsiniz.
    - **Veri Analizi:** Eğitim veri setinin detaylı görselleştirmeleri ve analizleri.
    - **Model Performansı:** Modelin başarım metrikleri ve değerlendirmeleri.
    """)

# --- TAHMİN SAYFASI ---
def prediction_page():
    st.title("🔮 Recidivism Risk Tahmini")

    st.markdown("Lütfen aşağıdaki alanları doldurun. `?` işaretine tıklayarak her alan hakkında bilgi alabilirsiniz.")

    # Kullanıcı girdileri için input formu
    input_data = {}
    cols = st.columns(2)

    for i, feat in enumerate(feature_names):
        container = cols[i % 2]
        with container:
            label = f"{feat}  ?"
            help_text = FEATURE_HELP.get(feat, None)

            if feat in bool_cols:
                val = st.selectbox(label, options=["True", "False"], help=help_text)
            elif feat in cat_features:
                options = cat_unique_values.get(feat, [])
                if options:
                    val = st.selectbox(label, options=[""] + options, help=help_text)
                else:
                    val = st.text_input(label, help=help_text)
            else:
                # sayısal inputlar
                val = st.number_input(label, value=0.0, format="%.4f", help=help_text)

            input_data[feat] = val

    if st.button("Tahmin Yap"):
        try:
            # DataFrame oluşturma ve uygun dönüşümler
            df_input = pd.DataFrame([input_data], columns=feature_names)
            # Boolean string olarak kalmalı
            for b in bool_cols:
                if b in df_input.columns:
                    df_input[b] = df_input[b].astype(str)

            pred = model.predict(df_input)[0]
            proba = model.predict_proba(df_input)[0][1] if hasattr(model, "predict_proba") else None

            risk_text = "🔴 Yüksek Risk (1)" if int(pred) == 1 else "🟢 Düşük Risk (0)"
            st.success(f"Tahmin: {risk_text}")
            if proba is not None:
                st.info(f"Olasılık: **{proba*100:.2f}%**")

            # SHAP explainability
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(df_input)
            st.subheader("Tahmin Açıklaması (SHAP)")

            fig_shap, ax = plt.subplots(figsize=(10, 5))
            shap.plots.waterfall(shap.Explanation(values=shap_values[0], 
                                                 base_values=explainer.expected_value,
                                                 data=df_input.iloc[0]),
                                 max_display=10, show=False)
            st.pyplot(fig_shap)

            # Tahmin geçmişine ekle
            record = input_data.copy()
            record.update({
                "Tahmin": int(pred),
                "Olasılık": proba,
            })
            st.session_state.prediction_history.append(record)

        except Exception as e:
            st.error(f"Tahmin sırasında hata oluştu: {e}")

    # Tahmin Geçmişi gösterimi
    if st.session_state.prediction_history:
        st.markdown("---")
        st.subheader("📋 Tahmin Geçmişi")
        df_hist = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(df_hist)

        csv_exp = df_hist.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Tahmin Geçmişini CSV Olarak İndir",
            data=csv_exp,
            file_name="tahmin_gecmisi.csv",
            mime="text/csv"
        )

# --- VERİ ANALİZİ SAYFASI ---
def analysis_page(df):
    st.title("📊 Gelişmiş Veri Görselleştirme ve Analiz")

    st.sidebar.header("Filtreler")

    # Güvenli yaş aralığı alma
    if "Age_at_Release" in df.columns:
        age_min = int(df["Age_at_Release"].dropna().min())
        age_max = int(df["Age_at_Release"].dropna().max())
        age_range = st.sidebar.slider("Yaş Aralığı", age_min, age_max, (age_min, age_max))
    else:
        st.error("'Age_at_Release' sütunu bulunamadı!")
        return

    # Çoklu seçimler
    def safe_unique(col):
        return df[col].dropna().unique().tolist() if col in df.columns else []

    gender_filter = st.sidebar.multiselect("Cinsiyet", options=safe_unique("Gender"), default=safe_unique("Gender"))
    race_filter = st.sidebar.multiselect("Irk", options=safe_unique("Race"), default=safe_unique("Race"))
    education_filter = st.sidebar.multiselect("Eğitim Seviyesi", options=safe_unique("Education_Level"), default=safe_unique("Education_Level"))
    gang_filter = st.sidebar.multiselect("Çete Bağlılığı", options=safe_unique("Gang_Affiliated"), default=safe_unique("Gang_Affiliated"))

    # Filtre uygulama
    df_filtered = df[
        (df["Age_at_Release"] >= age_range[0]) & (df["Age_at_Release"] <= age_range[1]) &
        (df["Gender"].isin(gender_filter)) &
        (df["Race"].isin(race_filter)) &
        (df["Education_Level"].isin(education_filter)) &
        (df["Gang_Affiliated"].isin(gang_filter))
    ]

    st.write(f"**Filtrelenmiş Kayıt Sayısı:** {df_filtered.shape[0]}")

    # Yaş Dağılımı
    fig1 = px.histogram(df_filtered, x="Age_at_Release", nbins=30, title="Yaş Dağılımı", color="Gender", barmode='overlay')
    st.plotly_chart(fig1, use_container_width=True)

    # Cinsiyet Oranı - Pasta Grafiği
    if "Gender" in df_filtered.columns:
        fig2 = px.pie(df_filtered, names="Gender", title="Cinsiyet Oranları")
        st.plotly_chart(fig2, use_container_width=True)

    # Irk Dağılımı - Çubuk Grafik
    if "Race" in df_filtered.columns:
        race_count = df_filtered["Race"].value_counts().reset_index()
        race_count.columns = ["Race", "Count"]
        fig3 = px.bar(race_count, x="Race", y="Count", title="Irk Dağılımı")
        st.plotly_chart(fig3, use_container_width=True)

    # Eğitim Seviyesi Dağılımı
    if "Education_Level" in df_filtered.columns:
        edu_count = df_filtered["Education_Level"].value_counts().reset_index()
        edu_count.columns = ["Education_Level", "Count"]
        fig4 = px.bar(edu_count, x="Education_Level", y="Count", title="Eğitim Seviyesi Dağılımı")
        st.plotly_chart(fig4, use_container_width=True)

    # Ceza Süresi ve Recidivism
    if "Prison_Years" in df_filtered.columns and "Recidivism_Within_3years" in df_filtered.columns:
        fig5 = px.box(df_filtered, x="Recidivism_Within_3years", y="Prison_Years", 
                      color="Recidivism_Within_3years",
                      labels={"Recidivism_Within_3years": "3 Yıl İçinde Yeniden Suç", "Prison_Years": "Ceza Süresi (Yıl)"},
                      title="Ceza Süresi ve Recidivism İlişkisi")
        st.plotly_chart(fig5, use_container_width=True)

    # Korelasyon Matrisi
    numeric_cols = df_filtered.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if len(numeric_cols) > 1:
        st.subheader("Sayısal Değişkenler Korelasyon Matrisi")
        corr = df_filtered[numeric_cols].corr()
        fig6, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig6)

    # Zaman serisi analizi (varsa)
    if "Recidivism_Arrest_Year1" in df_filtered.columns:
        ts_data = df_filtered.groupby("Recidivism_Arrest_Year1").size().reset_index(name="Count")
        fig7 = px.line(ts_data, x="Recidivism_Arrest_Year1", y="Count", title="Yıl Bazında Suç Tekrar Sayısı")
        st.plotly_chart(fig7, use_container_width=True)

    st.markdown("""
    ---
    **Analiz Notları:**  
    - Gelişmiş filtreleme ile farklı grupların risk ve demografik özellikleri incelenebilir.  
    - Korelasyon matrisi özellikler arası ilişkileri gösterir.  
    - Zaman serisi grafikler suç tekrar trendlerini ortaya koyar.
    """)


# --- MODEL PERFORMANS SAYFASI ---
def performance_page(df):
    st.title("📈 Model Performansı ve Değerlendirme")

    # Basit metrikler
    if "Recidivism_Within_3years" not in df.columns:
        st.error("Performans sayfası için hedef değişken (Recidivism_Within_3years) veride bulunamadı!")
        return

    y_true = df["Recidivism_Within_3years"].astype(int)
    X = df[feature_names]

    # Model tahminleri
    y_pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)[:, 1]
    else:
        y_proba = None

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

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
        from sklearn.metrics import RocCurveDisplay
        st.subheader("ROC Eğrisi")
        fig2, ax2 = plt.subplots()
        RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax2)
        st.pyplot(fig2)

    st.markdown("""
    ---
    **Not:**  
    Model performansı, gerçek veriler üzerinde ölçülmüştür. Daha iyi sonuçlar için model parametreleri optimize edilebilir.
    """)

# --- SAYFA YÖNLENDİRME ---
def main():
    if page == "Ana Sayfa":
        home_page()
    elif page == "Tahmin":
        prediction_page()
    elif page == "Veri Analizi":
        analysis_page(df_data)
    elif page == "Model Performansı":
        performance_page(df_data)
    else:
        st.error("Geçersiz sayfa seçimi!")

if __name__ == "__main__":
    main()
