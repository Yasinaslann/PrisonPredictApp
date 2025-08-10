# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report
import shap
from fpdf import FPDF

# ----- Sayfa ayarları -----
st.set_page_config(page_title="Hapisten Tahliye Sonrası Suç Tekrarı Tahmin Uygulaması", layout="wide", page_icon="🔒")

# ----- Global değişkenler -----
DATA_PATH = os.path.join("prison_app", "Prisongüncelveriseti.csv")
MODEL_PATH = os.path.join("prison_app", "catboost_model.pkl")
BOOL_COLS_PATH = os.path.join("prison_app", "bool_columns.pkl")
CAT_FEATURES_PATH = os.path.join("prison_app", "cat_features.pkl")
CAT_UNIQUE_PATH = os.path.join("prison_app", "cat_unique_values.pkl")
FEATURE_NAMES_PATH = os.path.join("prison_app", "feature_names.pkl")

# ----- Yardımcı fonksiyonlar -----

@st.cache_data(show_spinner=True)
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error(f"Veri dosyası bulunamadı! Lütfen '{DATA_PATH}' dosyasını uygulama dizinine koyun.")
        return None
    df = pd.read_csv(DATA_PATH)
    return df

@st.cache_resource(show_spinner=True)
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model dosyası bulunamadı! '{MODEL_PATH}'")
        return None
    model = joblib.load(MODEL_PATH)
    return model

@st.cache_data
def load_pickle(path):
    if not os.path.exists(path):
        st.error(f"Dosya bulunamadı! '{path}'")
        return None
    return joblib.load(path)

def preprocess_data(df, bool_cols):
    # Boolean kolonları bool tipine çevir
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(bool)
    return df

def filter_data(df, cat_features, cat_unique_values):
    st.sidebar.header("Filtreler")
    filtered_df = df.copy()
    # Kategorik filtreleme örneği
    for feature in cat_features:
        options = cat_unique_values.get(feature, [])
        if options:
            selected = st.sidebar.multiselect(f"{feature} Seçimi", options, default=options)
            filtered_df = filtered_df[filtered_df[feature].isin(selected)]
    # Sayısal filtre örneği (ör: yaş)
    if "Age_at_Release" in df.columns:
        min_age = int(df["Age_at_Release"].min())
        max_age = int(df["Age_at_Release"].max())
        age_range = st.sidebar.slider("Yaş Aralığı", min_age, max_age, (min_age, max_age))
        filtered_df = filtered_df[(filtered_df["Age_at_Release"] >= age_range[0]) & (filtered_df["Age_at_Release"] <= age_range[1])]
    return filtered_df

def plot_categorical_distribution(df, cat_features):
    st.subheader("Kategorik Değişkenlerin Dağılımı")
    for feature in cat_features:
        if feature in df.columns:
            fig = px.histogram(df, x=feature, color=feature, title=f"{feature} Dağılımı", labels={feature: feature})
            st.plotly_chart(fig, use_container_width=True)

def plot_numerical_distribution(df):
    st.subheader("Sayısal Değişkenlerin Dağılımı")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    for col in num_cols:
        fig = px.histogram(df, x=col, nbins=30, title=f"{col} Dağılımı")
        st.plotly_chart(fig, use_container_width=True)

def model_prediction(model, df, feature_names, cat_features):
    st.header("Suç Tekrarı Tahmini")

    # Kullanıcıdan girdi alma (interaktif form)
    input_data = {}
    st.markdown("### Tahmin için bilgilerinizi giriniz:")
    with st.form("predict_form"):
        for feature in feature_names:
            if feature in cat_features:
                # Kategorik input
                unique_vals = df[feature].dropna().unique().tolist()
                val = st.selectbox(f"{feature}", unique_vals)
                input_data[feature] = val
            elif df[feature].dtype == bool:
                val = st.checkbox(feature)
                input_data[feature] = val
            else:
                # Sayısal input
                min_val = int(df[feature].min()) if not df[feature].isnull().all() else 0
                max_val = int(df[feature].max()) if not df[feature].isnull().all() else 100
                val = st.number_input(f"{feature}", min_value=min_val, max_value=max_val, value=min_val)
                input_data[feature] = val

        submitted = st.form_submit_button("Tahmin Et")

    if submitted:
        input_df = pd.DataFrame([input_data])
        # Modelin istediği özellik sırasına göre düzenle
        input_df = input_df[feature_names]

        # Kategorik özelliklerin tip dönüşümü
        for cat in cat_features:
            if cat in input_df.columns:
                input_df[cat] = input_df[cat].astype(str)

        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]

        st.success(f"Suç Tekrarı Tahmin Sonucu: {'Evet' if prediction == 1 else 'Hayır'}")
        st.info(f"Tekrar Suç İşleme Olasılığı: %{proba*100:.2f}")

        # SHAP Değerleri ile yorumlama
        st.subheader("Model Açıklaması (SHAP Değerleri)")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)
        shap.initjs()
        st_shap_plot = st.pyplot(shap.summary_plot(shap_values, input_df, show=False))

def model_performance(df, model):
    st.header("Model Performans ve Değerlendirme")

    if "Recidivism" not in df.columns:
        st.warning("Performans değerlendirmesi için hedef değişken 'Recidivism' bulunamadı.")
        return

    y_true = df["Recidivism"]
    X = df.drop(columns=["Recidivism"])
    feature_names = X.columns.tolist()

    # Tahmin ve skor
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    accuracy = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_proba)

    st.markdown(f"**Accuracy:** {accuracy:.3f}")
    st.markdown(f"**ROC AUC:** {roc_auc:.3f}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Tahmin Edilen")
    ax.set_ylabel("Gerçek")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

    # Detaylı sınıflandırma raporu
    st.text("Detaylı Sınıflandırma Raporu:")
    report = classification_report(y_true, y_pred)
    st.text(report)

def generate_pdf_report(df, model, feature_names, cat_features):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Suç Tekrarı Tahmini Raporu", ln=True, align="C")

    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(0, 10, "Veri Seti Özeti:", ln=True)
    pdf.cell(0, 8, f"Toplam Kayıt Sayısı: {len(df)}", ln=True)
    pdf.cell(0, 8, f"Kategorik Özellikler: {', '.join(cat_features)}", ln=True)

    # Model performans örneği
    y_true = df["Recidivism"]
    X = df.drop(columns=["Recidivism"])
    y_pred = model.predict(X)
    accuracy = accuracy_score(y_true, y_pred)
    pdf.cell(0, 8, f"Model Doğruluk (Accuracy): {accuracy:.3f}", ln=True)

    pdf.output("suç_tekrar_raporu.pdf")
    st.success("PDF raporu oluşturuldu: suç_tekrar_raporu.pdf")

# ----- Ana fonksiyon -----
def main():
    st.title("🔒 Hapisten Tahliye Sonrası Suç Tekrarı Tahmin Uygulaması")
    st.markdown("""
    Bu uygulama, hapisten tahliye sonrası suç tekrarını tahmin etmek amacıyla geliştirilmiştir.  
    Veri seti üzerinden detaylı keşif, model tahminleri, performans değerlendirmesi ve raporlama yapılabilir.
    """)

    df = load_data()
    if df is None:
        st.stop()

    bool_cols = load_pickle(BOOL_COLS_PATH)
    cat_features = load_pickle(CAT_FEATURES_PATH)
    cat_unique_values = load_pickle(CAT_UNIQUE_PATH)
    model = load_model()
    feature_names = load_pickle(FEATURE_NAMES_PATH)

    if None in [bool_cols, cat_features, cat_unique_values, model, feature_names]:
        st.error("Gerekli dosyalardan biri veya birkaçı yüklenemedi.")
        st.stop()

    df = preprocess_data(df, bool_cols)
    filtered_df = filter_data(df, cat_features, cat_unique_values)

    st.sidebar.title("Sayfalar")
    page = st.sidebar.radio("Sayfa Seçiniz:", ["Veri Keşfi", "Tahmin", "Model Performansı", "Rapor Oluştur"])

    if page == "Veri Keşfi":
        st.header("📊 Veri Keşfi ve Gelişmiş Analiz")
        st.markdown(f"Filtrelenmiş veri sayısı: {len(filtered_df)}")
        plot_categorical_distribution(filtered_df, cat_features)
        plot_numerical_distribution(filtered_df)

    elif page == "Tahmin":
        model_prediction(model, df, feature_names, cat_features)

    elif page == "Model Performansı":
        model_performance(df, model)

    elif page == "Rapor Oluştur":
        st.header("📄 PDF Rapor Oluşturma")
        if st.button("Raporu Oluştur ve İndir"):
            generate_pdf_report(df, model, feature_names, cat_features)

if __name__ == "__main__":
    main()
