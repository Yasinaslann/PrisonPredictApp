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
from sklearn.preprocessing import LabelEncoder

# Sayfa yapılandırmasını ayarla
st.set_page_config(layout="wide", page_title="Suç Tekrarı Tahmin Uygulaması ⚖️", page_icon="📈")

# Dosya yolları
BASE_DIR = Path(__file__).parent
MODEL_FILE = BASE_DIR / "catboost_model.pkl"
BOOL_FILE = BASE_DIR / "bool_columns.pkl"
CAT_FILE = BASE_DIR / "cat_features.pkl"
FEATURES_FILE = BASE_DIR / "feature_names.pkl"
CAT_UNIQUE_FILE = BASE_DIR / "cat_unique_values.pkl"
DATA_FILE = BASE_DIR / "Prisongüncelveriseti.csv" # Veri seti dosyası

# Karşılaştırma için Logistic Regression modeli ve kodlayıcıları
LR_MODEL_FILE = BASE_DIR / "logistic_regression_model.pkl"

# Özellik açıklamaları
FEATURE_DESCRIPTIONS = {
    "Gender": "Mahkumun cinsiyeti",
    "Race": "Mahkumun ırkı",
    "Age_at_Release": "Tahliye yaşı",
    "Gang_Affiliated": "Çete bağlantısı (True/False)",
    "Supervised_Release_Years": "Gözetimli Serbestlik Süresi (Yıl)",
    "Education_Level": "Eğitim Seviyesi",
    "Prior_Convictions": "Önceki Mahkumiyet Sayısı"
}

# --- Performans için önbelleğe alma fonksiyonları ---
@st.cache_resource
def load_models_and_data():
    """
    Tüm modelleri ve veriyi bir kez yükler. Bu fonksiyon,
    kullanıcı etkileşimlerinde yeniden yüklemeyi önlemek için önbelleğe alınmıştır.
    """
    try:
        model = joblib.load(MODEL_FILE)
        bool_cols = joblib.load(BOOL_FILE)
        cat_features = joblib.load(CAT_FILE)
        feature_names = joblib.load(FEATURES_FILE)
        cat_unique_values = joblib.load(CAT_UNIQUE_FILE)
        df = pd.read_csv(DATA_FILE)
        
        # --- Düzeltme: Logistic Regression için modeli ve kodlayıcıları yükle veya eğit ---
        try:
            lr_model, lr_encoders = joblib.load(LR_MODEL_FILE)
        except (FileNotFoundError, ValueError):
            st.warning("Logistic Regression modeli bulunamadı veya eski formatta, yeni bir tane eğitiliyor.")
            X_for_lr = df[feature_names].copy()
            y = df["Recidivism_Within_3years"]
            lr_encoders = {}
            
            # LabelEncoder kullanarak kategorik özellikleri sayısal hale getir ve kodlayıcıları kaydet
            for col in cat_features + bool_cols:
                if col in X_for_lr.columns:
                    le = LabelEncoder()
                    X_for_lr[col] = le.fit_transform(X_for_lr[col])
                    lr_encoders[col] = le
            
            lr_model = LogisticRegression(random_state=42, solver='liblinear')
            lr_model.fit(X_for_lr, y)
            joblib.dump((lr_model, lr_encoders), LR_MODEL_FILE) # Model ve kodlayıcıları birlikte kaydet
        # --- Düzeltme sonu ---

        return model, lr_model, lr_encoders, bool_cols, cat_features, feature_names, cat_unique_values, df
    except FileNotFoundError as e:
        st.error(f"Hata: Gerekli dosyalardan biri bulunamadı: {e}. Lütfen dosyaların (model.pkl, .csv, vb.) uygulamanın dizininde olduğundan emin olun.")
        st.stop()

# Uygulama başlangıcında tüm veriyi yükle
try:
    model, lr_model, lr_encoders, bool_cols, cat_features, feature_names, cat_unique_values, df = load_models_and_data()
except Exception as e:
    st.error(f"Uygulama başlatılırken bir hata oluştu: {e}")
    st.stop()

# --- Paylaşılan Fonksiyonlar ---
def format_df_for_prediction(df_input):
    """
    CatBoost tahmini için DataFrame'i hazırlar.
    """
    df_output = df_input.copy()
    for b in bool_cols:
        if b in df_output.columns:
            df_output[b] = df_output[b].astype(str)
    return df_output

def format_df_for_lr(df_input, lr_encoders):
    """
    Logistic Regression tahmini için DataFrame'i hazırlar.
    """
    df_output = df_input.copy()
    for col in lr_encoders:
        if col in df_output.columns:
            # Eğitilmiş kodlayıcıyı kullanarak veriyi dönüştür
            df_output[col] = lr_encoders[col].transform(df_output[col])
    return df_output

# --- Sayfa fonksiyonları ---
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

    st.subheader("Suç Tekrarı Sınıf Dağılımı")
    fig = px.histogram(filtered_df, x="Recidivism_Within_3years", color="Recidivism_Within_3years",
                       category_orders={"Recidivism_Within_3years": [0,1]},
                       labels={"Recidivism_Within_3years": "3 Yıl İçinde Yeniden Suç (0: Hayır, 1: Evet)"},
                       title="Suç Tekrarı Sınıf Dağılımı")
    st.plotly_chart(fig, use_container_width=True)

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
    
    st.subheader("CatBoost Model Performansı")
    X_for_catboost = format_df_for_prediction(df[feature_names].copy())
    y_pred_catboost = model.predict(X_for_catboost)
    
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

    st.markdown("---")
    st.subheader("Logistic Regression Model Performansı (Karşılaştırma)")
    X_for_lr = format_df_for_lr(df[feature_names].copy(), lr_encoders)
    y_pred_lr = lr_model.predict(X_for_lr)
    
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

    st.subheader("Senaryo Karşılaştırması")
    
    baseline_data = {}
    for col in feature_names:
        if col in cat_features or col in bool_cols:
            baseline_data[col] = df[col].mode()[0]
        else:
            baseline_data[col] = df[col].mean()

    df_baseline = pd.DataFrame([baseline_data])
    baseline_pred_df = format_df_for_prediction(df_baseline.copy())
    baseline_proba = model.predict_proba(baseline_pred_df)[0][1]

    st.markdown("---")
    st.markdown("### Varsayılan Durum")
    st.write("Ortalama özelliklere sahip bir birey.")
    
    modified_data = baseline_data.copy()
    st.markdown("---")
    st.markdown("### Değiştirilmiş Senaryo")
    with st.form("what_if_form"):
        cols = st.columns(3)
        for i, col in enumerate(feature_names):
            container = cols[i % 3]
            with container:
                help_text = FEATURE_DESCRIPTIONS.get(col, "Açıklama bulunmamaktadır.")
                if col in bool_cols:
                    index = 1 if str(modified_data[col]) == "False" else 0
                    v = st.selectbox(f"{col}", ["True", "False"], help=help_text, index=index, key=f"what_if_{col}")
                elif col in cat_features:
                    options = cat_unique_values.get(col, [])
                    index = options.index(modified_data[col]) if modified_data[col] in options else 0
                    v = st.selectbox(f"{col}", options, help=help_text, index=index, key=f"what_if_{col}")
                else:
                    v = st.slider(f"{col}", float(df[col].min()), float(df[col].max()), float(modified_data[col]), step=1.0, key=f"what_if_{col}")
                modified_data[col] = v
        
        submitted = st.form_submit_button("Analizi Yenile")

    if submitted:
        df_modified = pd.DataFrame([modified_data])
        df_modified_for_predict = format_df_for_prediction(df_modified.copy())
        
        modified_proba = model.predict_proba(df_modified_for_predict)[0][1]
        
        st.markdown("---")
        st.subheader("Olasılık Değişimi")

        col_base, col_modified, col_change = st.columns(3)
        
        col_base.metric("Varsayılan Olasılık", f"%{baseline_proba*100:.2f}")
        col_modified.metric("Değiştirilmiş Olasılık", f"%{modified_proba*100:.2f}")

        proba_change = (modified_proba - baseline_proba) * 100
        
        if proba_change > 0:
            col_change.metric("Değişim", f"↑ %{proba_change:.2f}", delta_color="inverse")
        elif proba_change < 0:
            col_change.metric("Değişim", f"↓ %{-proba_change:.2f}", delta_color="normal")
        else:
            col_change.metric("Değişim", "0%")


def main():
    st.sidebar.title("Menü")
    st.sidebar.markdown("""
    Bu uygulama, makine öğrenimi modelini kullanarak bireylerin suç tekrarı olasılığını tahmin etmek için tasarlanmıştır.
    """)
    
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
