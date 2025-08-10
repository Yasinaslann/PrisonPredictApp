# app.py
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
import plotly.express as px
import numpy as np

# Set page configuration for a wider layout
st.set_page_config(layout="wide", page_title="Recidivism Prediction App")

# File paths
BASE_DIR = Path(__file__).parent
MODEL_FILE = BASE_DIR / "catboost_model.pkl"
BOOL_FILE = BASE_DIR / "bool_columns.pkl"
CAT_FILE = BASE_DIR / "cat_features.pkl"
FEATURES_FILE = BASE_DIR / "feature_names.pkl"
CAT_UNIQUE_FILE = BASE_DIR / "cat_unique_values.pkl"
DATA_FILE = BASE_DIR / "Prisongüncelveriseti.csv" # Dataset file

# Assuming a simple LR model exists for comparison
LR_MODEL_FILE = BASE_DIR / "logistic_regression_model.pkl"

# Feature descriptions
FEATURE_DESCRIPTIONS = {
    "Gender": "Mahkumun cinsiyeti",
    "Race": "Mahkumun ırkı",
    "Age_at_Release": "Tahliye yaşı",
    "Gang_Affiliated": "Çete bağlantısı (True/False)",
    "Supervised_Release_Years": "Gözetimli Serbestlik Süresi (Yıl)",
    "Education_Level": "Eğitim Seviyesi",
    "Prior_Convictions": "Önceki Mahkumiyet Sayısı"
}

# --- Caching Functions for performance ---
@st.cache_resource
def load_models_and_data():
    """Loads all models and data once for efficiency."""
    try:
        model = joblib.load(MODEL_FILE)
        bool_cols = joblib.load(BOOL_FILE)
        cat_features = joblib.load(CAT_FILE)
        feature_names = joblib.load(FEATURES_FILE)
        cat_unique_values = joblib.load(CAT_UNIQUE_FILE)
        df = pd.read_csv(DATA_FILE)
        # Assuming a pre-trained simple LR model for comparison
        try:
            lr_model = joblib.load(LR_MODEL_FILE)
        except FileNotFoundError:
            # Create a simple LR model on the fly if it doesn't exist for the example
            st.warning("Logistic Regression modeli bulunamadı, yeni bir tane eğitiliyor.")
            X = df[feature_names].copy()
            y = df["Recidivism_Within_3years"]
            for col in bool_cols:
                if col in X.columns:
                    X[col] = X[col].astype('category').cat.codes
            lr_model = LogisticRegression(random_state=42, solver='liblinear')
            lr_model.fit(X, y)
            joblib.dump(lr_model, LR_MODEL_FILE)

        return model, lr_model, bool_cols, cat_features, feature_names, cat_unique_values, df
    except FileNotFoundError as e:
        st.error(f"Hata: Gerekli dosyalardan biri bulunamadı: {e}. Lütfen dosyaların (model.pkl, .csv, vb.) uygulamanın dizininde olduğundan emin olun.")
        st.stop()

# Load everything at the start
try:
    model, lr_model, bool_cols, cat_features, feature_names, cat_unique_values, df = load_models_and_data()
except Exception as e:
    st.error(f"Uygulama başlatılırken bir hata oluştu: {e}")
    st.stop()

# --- Shared UI Components ---
def format_df_for_prediction(df_input):
    """Formats the input DataFrame for prediction."""
    for b in bool_cols:
        if b in df_input.columns:
            df_input[b] = df_input[b].astype(str)
    return df_input

# --- Page functions ---
def prediction_page():
    st.title("🔮 Bireysel Risk Tahmini")
    st.write("Alanları doldurarak bir mahkumun suç tekrarı riskini tahmin edin.")

    input_data = {}
    
    with st.form("prediction_form"):
        cols = st.columns(3)
        for i, col in enumerate(feature_names):
            container = cols[i % 3]
            with container:
                help_text = FEATURE_DESCRIPTIONS.get(col, "Açıklama bulunmamaktadır.")
                st.markdown(f"**{col}**")
                if col in bool_cols:
                    v = st.selectbox(col, ["True", "False"], help=help_text, key=f"pred_input_{col}")
                elif col in cat_features:
                    options = cat_unique_values.get(col, [])
                    if options:
                        v = st.selectbox(col, options, help=help_text, key=f"pred_input_{col}")
                    else:
                        v = st.text_input(col, help=help_text, key=f"pred_input_{col}")
                else:
                    v = st.number_input(col, value=float(df[col].mean()), format="%.2f", help=help_text, key=f"pred_input_{col}")
                input_data[col] = v
        
        submitted = st.form_submit_button("🔮 Tahmin Yap")

    if submitted:
        try:
            df_input = pd.DataFrame([input_data], columns=feature_names)
            df_input_for_predict = format_df_for_prediction(df_input)
            
            pred = model.predict(df_input_for_predict)[0]
            proba = model.predict_proba(df_input_for_predict)[0][1]

            st.subheader("Tahmin Sonucu")
            if pred == 1:
                st.markdown(f"<h2 style='color:red;'>Yüksek risk altında: Tekrar suç işleme olasılığı yüksek.</h2>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h2 style='color:green;'>Düşük risk altında: Tekrar suç işleme olasılığı düşük.</h2>", unsafe_allow_html=True)
            st.write(f"Tahmin Olasılığı: **%{proba*100:.2f}**")

            # SHAP Explanation
            st.subheader("Tahmin Açıklaması (SHAP)")
            st.write("Bu grafik, tahmin sonucunu en çok etkileyen faktörleri göstermektedir.")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(df_input_for_predict)
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            shap.force_plot(explainer.expected_value, shap_values[0], df_input_for_predict.iloc[0], matplotlib=True, show=False, ax=ax)
            ax.set_title("SHAP Force Plot")
            plt.tight_layout()
            st.pyplot(fig, bbox_inches='tight')
            plt.close(fig)

            # Personal recommendation based on prediction
            st.subheader("Öneri")
            if pred == 1:
                st.info("📌 Öneri: Eğitim programlarına katılmanız ve denetimli serbestlik programlarına dahil olmanız önerilir.")
            else:
                st.success("🎉 Öneri: Düşük risk grubundasınız. Takip ve destek programlarına devam edin.")

        except Exception as e:
            st.error(f"Tahmin sırasında bir hata oluştu: {e}")

def analysis_page():
    st.title("📊 Veri Analizi ve Görselleştirme")
    st.write("Veri setini filtreleyerek ve görselleştirerek suç tekrarı faktörlerini inceleyin.")

    # Sidebar filters
    st.sidebar.header("Veri Filtreleri")
    age_column = "Age_at_Release"
    gender_column = "Gender"
    
    if age_column in df.columns:
        age_min, age_max = st.sidebar.slider(
            "Yaş Aralığı",
            int(df[age_column].min()),
            int(df[age_column].max()),
            (int(df[age_column].min()), int(df[age_column].max()))
        )
    else:
        st.warning(f"'{age_column}' sütunu veri setinde bulunamadı.")
        age_min, age_max = 0, 100

    if gender_column in df.columns:
        gender_options = df[gender_column].unique().tolist()
        gender_filter = st.sidebar.multiselect("Cinsiyet", options=gender_options, default=gender_options)
    else:
        st.warning(f"'{gender_column}' sütunu veri setinde bulunamadı.")
        gender_filter = []

    filtered_df = df[
        (df[age_column].between(age_min, age_max)) &
        (df[gender_column].isin(gender_filter))
    ].copy()

    st.info(f"Filtrelenmiş Toplam Kayıt Sayısı: {filtered_df.shape[0]}")

    # Class distribution
    st.subheader("Suç Tekrarı Sınıf Dağılımı")
    fig = px.histogram(filtered_df, x="Recidivism_Within_3years", color="Recidivism_Within_3years",
                       category_orders={"Recidivism_Within_3years": [0,1]},
                       labels={"Recidivism_Within_3years": "3 Yıl İçinde Yeniden Suç (0: Hayır, 1: Evet)"},
                       title="Suç Tekrarı Sınıf Dağılımı")
    st.plotly_chart(fig, use_container_width=True)

    # User selected feature chart
    st.subheader("Özelliklere Göre Dağılım")
    selected_feature = st.selectbox("Grafik için bir özellik seçin", options=feature_names)

    if selected_feature in cat_features or selected_feature in bool_cols:
        fig2 = px.histogram(filtered_df, x=selected_feature, color="Recidivism_Within_3years",
                            title=f"{selected_feature} Değişkeninin Suç Tekrarına Göre Dağılımı")
    else:
        fig2 = px.box(filtered_df, x="Recidivism_Within_3years", y=selected_feature,
                      title=f"{selected_feature} Değişkeninin Suç Tekrarına Göre Dağılımı")
    st.plotly_chart(fig2, use_container_width=True)

def performance_page():
    st.title("📈 Model Performansı")
    st.write("Modelin tüm veri seti üzerindeki performans metriklerini inceleyin.")

    y_true = df["Recidivism_Within_3years"]
    X = df[feature_names].copy()
    X_for_predict = format_df_for_prediction(X)
    
    # --- CatBoost Model Performance ---
    st.subheader("CatBoost Model Performansı")
    y_pred_catboost = model.predict(X_for_predict)
    
    st.markdown("### Sınıflandırma Raporu (CatBoost)")
    report_dict_catboost = classification_report(y_true, y_pred_catboost, output_dict=True)
    report_df_catboost = pd.DataFrame(report_dict_catboost).transpose()
    st.dataframe(report_df_catboost)

    st.markdown("### Confusion Matrix (CatBoost)")
    cm_catboost = confusion_matrix(y_true, y_pred_catboost)
    fig, ax = plt.subplots()
    sns.heatmap(cm_catboost, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Tahmin Edilen")
    ax.set_ylabel("Gerçek")
    st.pyplot(fig)

    # --- Logistic Regression Model Performance (for comparison) ---
    st.markdown("---")
    st.subheader("Logistic Regression Model Performansı (Karşılaştırma)")
    X_lr = df[feature_names].copy()
    for col in bool_cols:
        if col in X_lr.columns:
            X_lr[col] = X_lr[col].astype('category').cat.codes
    
    y_pred_lr = lr_model.predict(X_lr)
    
    st.markdown("### Sınıflandırma Raporu (LR)")
    report_dict_lr = classification_report(y_true, y_pred_lr, output_dict=True)
    report_df_lr = pd.DataFrame(report_dict_lr).transpose()
    st.dataframe(report_df_lr)

    st.markdown("### Confusion Matrix (LR)")
    cm_lr = confusion_matrix(y_true, y_pred_lr)
    fig_lr, ax_lr = plt.subplots()
    sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=ax_lr)
    ax_lr.set_xlabel("Tahmin Edilen")
    ax_lr.set_ylabel("Gerçek")
    st.pyplot(fig_lr)

def what_if_page():
    st.title("🧐 'Ne-Olursa-Ne-Olur?' Senaryo Analizi")
    st.write("Özellikleri değiştirerek tahmin sonucunun nasıl değiştiğini inceleyin.")

    st.subheader("Varsayılan Özellikler")
    # Use mean or mode values as a baseline
    baseline_data = {}
    for col in feature_names:
        if col in cat_features or col in bool_cols:
            baseline_data[col] = df[col].mode()[0]
        else:
            baseline_data[col] = df[col].mean()

    # Create a DataFrame for the baseline
    df_baseline = pd.DataFrame([baseline_data])

    # Show baseline prediction
    baseline_pred_df = format_df_for_prediction(df_baseline.copy())
    baseline_proba = model.predict_proba(baseline_pred_df)[0][1]
    st.info(f"**Varsayılan Durum:** Ortalama bir kişinin suç tekrarı olasılığı **%{baseline_proba*100:.2f}**'dir.")

    st.subheader("Özellikleri Değiştirin")
    modified_data = baseline_data.copy()

    with st.form("what_if_form"):
        cols = st.columns(3)
        for i, col in enumerate(feature_names):
            container = cols[i % 3]
            with container:
                help_text = FEATURE_DESCRIPTIONS.get(col, "Açıklama bulunmamaktadır.")
                if col in bool_cols:
                    v = st.selectbox(f"{col}", ["True", "False"], help=help_text, index=1 if modified_data[col] == "False" else 0)
                elif col in cat_features:
                    options = cat_unique_values.get(col, [])
                    v = st.selectbox(f"{col}", options, help=help_text, index=options.index(modified_data[col]))
                else:
                    v = st.slider(f"{col}", float(df[col].min()), float(df[col].max()), float(modified_data[col]), step=1.0)
                modified_data[col] = v
        
        submitted = st.form_submit_button("Analizi Yenile")
    
    if submitted:
        df_modified = pd.DataFrame([modified_data])
        df_modified_for_predict = format_df_for_prediction(df_modified.copy())
        
        modified_proba = model.predict_proba(df_modified_for_predict)[0][1]
        
        st.subheader("Yeni Tahmin Sonucu")
        st.write(f"Değiştirilmiş özelliklerle suç tekrarı olasılığı: **%{modified_proba*100:.2f}**")

        proba_change = modified_proba - baseline_proba
        
        st.subheader("Olasılık Değişimi")
        if proba_change > 0:
            st.success(f"Olasılık **%{proba_change*100:.2f}** arttı.")
        elif proba_change < 0:
            st.error(f"Olasılık **%{-proba_change*100:.2f}** azaldı.")
        else:
            st.info("Olasılıkta bir değişiklik olmadı.")

def main():
    st.sidebar.title("App Menüsü")
    
    tabs = st.tabs(["🔮 Tahmin", "📊 Veri Analizi", "📈 Model Performansı", "🧐 Ne-Olursa-Ne-Olur?"])

    with tabs[0]:
        prediction_page()
    with tabs[1]:
        analysis_page()
    with tabs[2]:
        performance_page()
    with tabs[3]:
        what_if_page()

if __name__ == "__main__":
    main()
