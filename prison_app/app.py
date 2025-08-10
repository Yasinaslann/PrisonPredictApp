import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from catboost import CatBoostClassifier, Pool
import joblib
import shap
from fpdf import FPDF
import io

# --- Ayarlar ---
st.set_page_config(page_title="Hapisten Tahliye Sonrası Suç Tekrarı Tahmin Uygulaması",
                   layout="wide",
                   initial_sidebar_state="expanded",
                   page_icon="🔒")

# --- Tema & Renkler ---
PRIMARY_COLOR = "#4e79a7"
SECONDARY_COLOR = "#f28e2b"
BACKGROUND_COLOR = "#f0f2f6"

# CSS ile temayı biraz güzelleştirelim
st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {BACKGROUND_COLOR};
        color: #333333;
    }}
    .css-18e3th9 {{
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }}
    h1, h2, h3, h4 {{
        color: {PRIMARY_COLOR};
    }}
    .stButton>button {{
        background-color: {PRIMARY_COLOR};
        color: white;
        border-radius: 8px;
        padding: 8px 20px;
    }}
    .stButton>button:hover {{
        background-color: {SECONDARY_COLOR};
        color: white;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# --- Veri Yükleme ---
@st.cache_data
def load_data():
    filename = "Prisongüncelveriseti.csv"
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        return df
    else:
        st.error(f"Veri dosyası bulunamadı! Lütfen '{filename}' dosyasını uygulama dizinine koyun.")
        return None

# --- Pickle dosyaları yükle ---
@st.cache_resource
def load_pickle(filename):
    if os.path.exists(filename):
        return joblib.load(filename)
    else:
        st.warning(f"Pickle dosyası bulunamadı: {filename}")
        return None

# --- Model yükle ---
@st.cache_resource
def load_model(filename):
    if os.path.exists(filename):
        model = CatBoostClassifier()
        model.load_model(filename)
        return model
    else:
        st.warning(f"Model dosyası bulunamadı: {filename}")
        return None

# --- PDF raporu için fonksiyon ---
def generate_pdf_report(input_data, prediction, probability):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(0, 10, "Hapisten Tahliye Sonrası Suç Tekrarı Tahmin Raporu", ln=1, align="C")
    pdf.ln(10)
    
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "Girilen Bilgiler:", ln=1)
    
    for key, value in input_data.items():
        pdf.cell(0, 8, f"{key}: {value}", ln=1)
    
    pdf.ln(10)
    pdf.set_font("Arial", style="B", size=12)
    pdf.cell(0, 10, f"Tahmin Sonucu: {'Tekrar Suç İşler' if prediction==1 else 'Tekrar Suç İşlemez'}", ln=1)
    pdf.cell(0, 10, f"Tahmin Olasılığı: %{probability*100:.2f}", ln=1)
    
    pdf.output("Recidivism_Prediction_Report.pdf")
    return "Recidivism_Prediction_Report.pdf"

# --- Veri Keşfi Bölümü ---
def data_exploration(df):
    st.header("📊 Veri Keşfi ve Gelişmiş Analiz")

    st.markdown("### Veri Önizlemesi")
    st.dataframe(df.head(15))

    st.markdown("### Veri Seti Özet Bilgileri")
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    st.markdown("### Sayısal Değişkenlerin Dağılımı")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    selected_num_cols = st.multiselect("Sayısal sütunları seçin", numeric_cols, default=numeric_cols[:3])

    for col in selected_num_cols:
        fig = px.histogram(df, x=col, nbins=30, title=f"{col} Dağılımı", color_discrete_sequence=[PRIMARY_COLOR])
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Kategorik Değişkenlerin Dağılımı")
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    selected_cat_cols = st.multiselect("Kategorik sütunları seçin", cat_cols, default=cat_cols[:3])

    for col in selected_cat_cols:
        fig = px.bar(df[col].value_counts().reset_index(), x="index", y=col,
                     title=f"{col} Frekans Dağılımı", color_discrete_sequence=[SECONDARY_COLOR])
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

# --- Model Tahmin Sayfası ---
def prediction_page(model, cat_features, feature_names, df):
    st.header("🧠 Hapisten Tahliye Sonrası Suç Tekrarı Tahmin Modülü")

    input_data = {}
    st.markdown("### Lütfen aşağıdaki alanları doldurun:")

    for feature in feature_names:
        if feature in cat_features:
            unique_vals = df[feature].dropna().unique().tolist()
            val = st.selectbox(f"{feature} seçin:", options=unique_vals)
            input_data[feature] = val
        else:
            min_val = int(df[feature].min())
            max_val = int(df[feature].max())
            val = st.slider(f"{feature} girin:", min_value=min_val, max_value=max_val, value=min_val)
            input_data[feature] = val

    if st.button("Tahmin Et"):
        try:
            input_df = pd.DataFrame([input_data])
            # CatBoost Pool ile
            pool = Pool(data=input_df, cat_features=cat_features)
            prediction = model.predict(pool)[0]
            prediction_proba = model.predict_proba(pool)[0][1]

            st.success(f"🔮 Tahmin: {'Tekrar Suç İşler' if prediction == 1 else 'Tekrar Suç İşlemez'}")
            st.info(f"Model Olasılığı: %{prediction_proba * 100:.2f}")

            # SHAP açıklaması
            st.markdown("### Model Açıklaması (SHAP Değerleri)")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df)
            shap.initjs()
            shap.force_plot(explainer.expected_value, shap_values[0,:], input_df.iloc[0,:], matplotlib=True, show=False)
            st.pyplot(bbox_inches='tight')
            plt.clf()

            # PDF Raporu
            if st.button("Tahmin Sonucunu PDF olarak indir"):
                pdf_path = generate_pdf_report(input_data, prediction, prediction_proba)
                with open(pdf_path, "rb") as f:
                    st.download_button(label="PDF İndir", data=f, file_name=pdf_path, mime="application/pdf")

        except Exception as e:
            st.error(f"Hata oluştu: {e}")

    st.markdown("---")

# --- Model Performans Sayfası ---
def performance_page(df, model, cat_features):
    st.header("📈 Model Performans ve Değerlendirme")

    # Hedef değişken kontrolü
    if "Recidivism" not in df.columns:
        st.warning("Veri setinde 'Recidivism' hedef sütunu bulunamadı.")
        return

    y_true = df["Recidivism"]

    X = df.drop(columns=["Recidivism"])

    # Gerekirse kategorik filtre
    pool = Pool(data=X, label=y_true, cat_features=cat_features)

    preds_proba = model.predict_proba(pool)[:, 1]
    preds = model.predict(pool)

    from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

    acc = accuracy_score(y_true, preds)
    roc_auc = roc_auc_score(y_true, preds_proba)

    st.write(f"**Doğruluk (Accuracy):** {acc:.3f}")
    st.write(f"**ROC AUC Skoru:** {roc_auc:.3f}")

    cm = confusion_matrix(y_true, preds)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")
    ax.set_title("Karışıklık Matrisi (Confusion Matrix)")
    st.pyplot(fig)

    st.markdown("---")

# --- Ana Fonksiyon ---
def main():
    st.title("🔒 Hapisten Tahliye Sonrası Suç Tekrarı Tahmin Uygulaması")

    df = load_data()
    if df is None:
        st.stop()

    cat_features = load_pickle("cat_features.pkl")
    feature_names = load_pickle("feature_names.pkl")
    model = load_model("catboost_model.pkl")

    if None in (cat_features, feature_names, model):
        st.error("Gerekli dosyalardan biri veya birkaçı yüklenemedi.")
        st.stop()

    pages = {
        "Veri Keşfi": lambda: data_exploration(df),
        "Tahmin": lambda: prediction_page(model, cat_features, feature_names, df),
        "Model Performansı": lambda: performance_page(df, model, cat_features)
    }

    st.sidebar.title("Navigasyon")
    choice = st.sidebar.radio("Sayfa Seçin:", list(pages.keys()))
    pages[choice]()

if __name__ == "__main__":
    main()
