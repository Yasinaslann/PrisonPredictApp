# app.py
import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from catboost import CatBoostClassifier, Pool
from fpdf import FPDF
import base64
from io import BytesIO

# --- AYARLAR ---

st.set_page_config(
    page_title="🔒 Hapisten Tahliye Sonrası Suç Tekrarı Tahmin Uygulaması",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🔒"
)

# Dark theme için CSS (basit)
def local_css():
    st.markdown(
        """
        <style>
        .css-1d391kg {background-color:#121212;}
        .css-ffhzg2 {color:#e0e0e0;}
        .stButton>button {background-color:#4e73df;color:#fff;}
        .css-1v0mbdj {color:#e0e0e0;}
        .css-18e3th9 {background-color:#1e1e1e;}
        </style>
        """, unsafe_allow_html=True)

local_css()

# --- VERİ YÜKLEME ---

@st.cache_data(show_spinner=True)
def load_data():
    possible_files = ["Prisongüncelveriseti.csv", "NIJ_s_Recidivism_Encod_Update.csv"]
    for f in possible_files:
        if os.path.exists(f):
            df = pd.read_csv(f)
            return df
    st.error("Veri dosyası bulunamadı! Lütfen 'Prisongüncelveriseti.csv' veya 'NIJ_s_Recidivism_Encod_Update.csv' dosyasını uygulama dizinine koyun.")
    return None

@st.cache_resource(show_spinner=True)
def load_models_and_metadata():
    # Model, özellik isimleri ve kategorik sütunlar
    model = None
    cat_features = []
    feature_names = []
    bool_cols = []
    cat_unique_values = {}

    # Dosyalar varsa yükle
    try:
        model = joblib.load("catboost_model.pkl")
    except Exception:
        st.warning("Model dosyası yüklenemedi: catboost_model.pkl")

    try:
        cat_features = joblib.load("cat_features.pkl")
    except Exception:
        st.warning("cat_features.pkl yüklenemedi.")

    try:
        feature_names = joblib.load("feature_names.pkl")
    except Exception:
        st.warning("feature_names.pkl yüklenemedi.")

    try:
        bool_cols = joblib.load("bool_columns.pkl")
    except Exception:
        st.warning("bool_columns.pkl yüklenemedi.")

    try:
        cat_unique_values = joblib.load("cat_unique_values.pkl")
    except Exception:
        st.warning("cat_unique_values.pkl yüklenemedi.")

    return model, cat_features, feature_names, bool_cols, cat_unique_values


# --- VERİ ÖN İŞLEME ---

def clean_data(df):
    df = df.copy()
    # Eksik değerlere göre basit doldurma örneği
    for col in df.columns:
        if df[col].dtype == 'O':  # kategorik
            df[col].fillna("Unknown", inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)
    return df


# --- SAYFA: Ana Bilgilendirme ---

def home_page():
    st.title("🔒 Hapisten Tahliye Sonrası Suç Tekrarı Tahmin Uygulaması")
    st.markdown("""
        Bu uygulama, mahkumların tahliye sonrası suç tekrarı yapıp yapmayacağını tahmin etmeye yönelik gelişmiş makine öğrenimi modelleri kullanır.
        
        **Amaç:** Toplumu korumak, kaynakları etkili kullanmak ve riskli bireyleri erken belirleyerek müdahale etmektir.

        **Proje Özellikleri:**
        - Gelişmiş CatBoost modeli ile yüksek doğruluk
        - Kapsamlı veri keşfi ve filtreleme
        - Etkileşimli grafikler ve tablolar
        - Model performans analizleri
        - SHAP açıklayıcı analiz
        - PDF rapor indirme
        - Modern ve kullanıcı dostu arayüz
    """)
    st.image("https://upload.wikimedia.org/wikipedia/commons/6/64/Justice_scales_icon.svg", width=150)

# --- SAYFA: Veri Keşfi ---

def data_exploration_page(df):
    st.header("📊 Veri Keşfi ve Gelişmiş Analiz")

    # Veri filtreleme - interaktif

    with st.expander("Veri Seti Özet Bilgileri"):
        st.write("Toplam Kayıt Sayısı:", df.shape[0])
        st.write("Özellik Sayısı:", df.shape[1])
        st.write("Örnek Veri:")
        st.dataframe(df.head(10))

    st.markdown("---")

    # Filtre seçenekleri
    # Numerik kolonlar
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    with st.sidebar.expander("Filtreleme Seçenekleri"):
        st.markdown("### Sayısal Filtreler")
        filters = {}
        for col in numeric_cols:
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            selected_range = st.slider(f"{col} aralığı", min_val, max_val, (min_val, max_val))
            filters[col] = selected_range

        st.markdown("### Kategorik Filtreler")
        for col in cat_cols:
            unique_vals = list(df[col].dropna().unique())
            selected_vals = st.multiselect(f"{col} seçimi", unique_vals, default=unique_vals)
            filters[col] = selected_vals

    # Filtre uygulama
    df_filtered = df.copy()
    for col, val in filters.items():
        if col in numeric_cols:
            df_filtered = df_filtered[(df_filtered[col] >= val[0]) & (df_filtered[col] <= val[1])]
        elif col in cat_cols:
            df_filtered = df_filtered[df_filtered[col].isin(val)]

    st.markdown(f"### Filtrelenmiş Veri: {df_filtered.shape[0]} kayıt")

    # Kategorik dağılımlar grafik
    if st.checkbox("Kategorik Değişkenlerin Dağılımını Göster"):
        selected_cat = st.selectbox("Kategori seçin", cat_cols)
        fig = px.histogram(df_filtered, x=selected_cat, color=selected_cat, title=f"{selected_cat} Dağılımı")
        st.plotly_chart(fig, use_container_width=True)

    # Sayısal değişken dağılımları
    if st.checkbox("Sayısal Değişkenlerin Histogramını Göster"):
        selected_num = st.selectbox("Sayısal Değişken seçin", numeric_cols)
        fig2 = px.histogram(df_filtered, x=selected_num, nbins=30, title=f"{selected_num} Histogramı")
        st.plotly_chart(fig2, use_container_width=True)

    # Korelasyon matrisi
    if st.checkbox("Korelasyon Matrisini Göster"):
        corr = df_filtered[numeric_cols].corr()
        fig3, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig3)

# --- SAYFA: Tahmin ---

def prediction_page(model, cat_features, feature_names, df):
    st.header("🧠 Kişisel Suç Tekrarı Tahmin Modülü")

    if model is None:
        st.warning("Model yüklenemedi. Tahmin yapılamıyor.")
        return

    st.markdown("Lütfen aşağıdaki bilgileri eksiksiz doldurun.")

    input_data = {}

    # Kategorik ve sayısal feature'lara göre form oluştur
    for feature in feature_names:
        if feature in cat_features:
            options = df[feature].dropna().unique().tolist()
            input_data[feature] = st.selectbox(f"{feature} seçiniz", options)
        else:
            min_val = float(df[feature].min())
            max_val = float(df[feature].max())
            step = (max_val - min_val) / 100 if max_val != min_val else 1
            if df[feature].dtype == 'int64':
                input_data[feature] = st.slider(f"{feature} (int)", int(min_val), int(max_val), int(min_val))
            else:
                input_data[feature] = st.slider(f"{feature} (float)", float(min_val), float(max_val), float(min_val), step=step)

    # Tahmin yap butonu
    if st.button("Tahmin Et"):
        # Dataframe oluştur
        input_df = pd.DataFrame([input_data])
        # Model için Pool objesi yarat
        pool = Pool(input_df, cat_features=cat_features)

        prediction = model.predict(pool)[0]
        proba = model.predict_proba(pool)[0][1]  # 1. sınıf olasılığı

        st.success(f"Tahmin Sonucu: {'Tekrar Suç İşler' if prediction == 1 else 'Tekrar Suç İşlemez'}")
        st.info(f"Model Güveni: %{proba*100:.2f}")

# --- SAYFA: Performans ---

def performance_page(df, model, cat_features, feature_names):
    st.header("📈 Model Performans ve Değerlendirme")

    if model is None:
        st.warning("Model yüklenemedi. Performans gösterilemiyor.")
        return

    # Hedef sütun
    if "Recidivism" not in df.columns:
        st.error("Hedef sütun (Recidivism) veri setinde bulunamadı!")
        return

    y_true = df["Recidivism"]
    X = df[feature_names]

    pool = Pool(X, y_true, cat_features=cat_features)

    preds_proba = model.predict_proba(pool)[:, 1]

    from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report

    roc_auc = roc_auc_score(y_true, preds_proba)
    accuracy = accuracy_score(y_true, model.predict(pool))

    st.markdown(f"**ROC AUC Skoru:** {roc_auc:.4f}")
    st.markdown(f"**Doğruluk (Accuracy):** {accuracy:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_true, model.predict(pool))
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Tahmin Edilen")
    ax.set_ylabel("Gerçek")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

    # Feature importance
    st.subheader("Model Özellik Önem Düzeyi")
    fi = model.get_feature_importance(type="FeatureImportance")
    fi_df = pd.DataFrame({"Feature": feature_names, "Importance": fi})
    fi_df = fi_df.sort_values("Importance", ascending=False)

    fig2 = px.bar(fi_df, x="Importance", y="Feature", orientation='h', title="Feature Importance")
    st.plotly_chart(fig2, use_container_width=True)

# --- SAYFA: SHAP Açıklama ---

def shap_analysis_page(df, model, cat_features, feature_names):
    st.header("🔍 Model Açıklaması ve SHAP Analizleri")

    if model is None:
        st.warning("Model yüklenemedi. SHAP analizleri yapılamıyor.")
        return

    X = df[feature_names]

    pool = Pool(X, cat_features=cat_features)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    st.markdown("### SHAP Summary Plot")
    fig, ax = plt.subplots(figsize=(12,6))
    shap.summary_plot(shap_values, X, show=False)
    st.pyplot(fig)

    st.markdown("### SHAP Feature Importance (Bar)")
    fig2, ax2 = plt.subplots(figsize=(10,6))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    st.pyplot(fig2)

# --- SAYFA: PDF Rapor İndir ---

def generate_pdf_report(prediction, proba, input_data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "Tahliye Sonrası Suç Tekrarı Tahmin Raporu", ln=True, align='C')
    pdf.ln(10)

    for key, val in input_data.items():
        pdf.cell(0, 10, f"{key}: {val}", ln=True)
    pdf.ln(10)
    pdf.cell(0, 10, f"Tahmin Sonucu: {'Tekrar Suç İşler' if prediction == 1 else 'Tekrar Suç İşlemez'}", ln=True)
    pdf.cell(0, 10, f"Model Güveni: %{proba*100:.2f}", ln=True)

    return pdf.output(dest='S').encode('latin-1')

def pdf_download_button(prediction, proba, input_data):
    pdf_bytes = generate_pdf_report(prediction, proba, input_data)
    b64 = base64.b64encode(pdf_bytes).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="tahmin_raporu.pdf">📄 Tahmin Raporunu İndir</a>'
    st.markdown(href, unsafe_allow_html=True)


# --- ANA FONKSİYON ---

def main():
    st.sidebar.title("🔒 Menü")
    menu = ["Anasayfa", "Veri Keşfi", "Tahmin", "Model Performansı", "SHAP Açıklama"]
    choice = st.sidebar.radio("Sayfa Seçiniz", menu)

    df = load_data()
    if df is None:
        st.stop()

    model, cat_features, feature_names, bool_cols, cat_unique_values = load_models_and_metadata()
    df = clean_data(df)

    if choice == "Anasayfa":
        home_page()
    elif choice == "Veri Keşfi":
        data_exploration_page(df)
    elif choice == "Tahmin":
        prediction_page(model, cat_features, feature_names, df)
    elif choice == "Model Performansı":
        performance_page(df, model, cat_features, feature_names)
    elif choice == "SHAP Açıklama":
        shap_analysis_page(df, model, cat_features, feature_names)
    else:
        st.write("Lütfen menüden bir sayfa seçiniz.")


if __name__ == "__main__":
    main()
