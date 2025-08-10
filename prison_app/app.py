# app.py
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px

# Dosya yolları (app.py ile aynı klasörde olduğunu varsayıyoruz)
BASE_DIR = Path(__file__).parent
MODEL_FILE = BASE_DIR / "catboost_model.pkl"
BOOL_FILE = BASE_DIR / "bool_columns.pkl"
CAT_FILE = BASE_DIR / "cat_features.pkl"
FEATURES_FILE = BASE_DIR / "feature_names.pkl"
CAT_UNIQUE_FILE = BASE_DIR / "cat_unique_values.pkl"
DATA_FILE = BASE_DIR / "Prisongüncelveriseti.csv"

FEATURE_DESCRIPTIONS = {
    "Gender": "Mahkumun cinsiyeti",
    "Race": "Mahkumun ırkı",
    "Age_at_Release": "Tahliye yaşı",
    "Gang_Affiliated": "Çete bağlantısı (True/False)",
    # İstersen diğerlerini de ekle
}

@st.cache_resource
def load_model_and_data():
    model = joblib.load(MODEL_FILE)
    bool_cols = joblib.load(BOOL_FILE)
    cat_features = joblib.load(CAT_FILE)
    feature_names = joblib.load(FEATURES_FILE)
    cat_unique_values = joblib.load(CAT_UNIQUE_FILE)
    df = pd.read_csv(DATA_FILE)
    return model, bool_cols, cat_features, feature_names, cat_unique_values, df

model, bool_cols, cat_features, feature_names, cat_unique_values, df = load_model_and_data()

def prediction_page():
    st.title("📊 Recidivism (3 yıl) Tahmin Uygulaması")
    st.write("Alanları doldurup tahmin yapın. Boolean sütunlar `True/False` string olarak modele verildi.")

    input_data = {}
    cols = st.columns(2)
    for i, col in enumerate(feature_names):
        container = cols[i % 2]
        with container:
            help_text = FEATURE_DESCRIPTIONS.get(col, "")
            if col in bool_cols:
                v = st.selectbox(col, ["True", "False"], help=help_text)
            elif col in cat_features:
                options = cat_unique_values.get(col, [])
                if options:
                    v = st.selectbox(col, options, help=help_text)
                else:
                    v = st.text_input(col, help=help_text)
            else:
                v = st.number_input(col, value=0.0, format="%.6f", help=help_text)
            input_data[col] = v

    if st.button("🔮 Tahmin Yap"):
        try:
            df_input = pd.DataFrame([input_data], columns=feature_names)
            for b in bool_cols:
                if b in df_input.columns:
                    df_input[b] = df_input[b].astype(str)

            pred = model.predict(df_input)[0]
            proba = model.predict_proba(df_input)[0][1] if hasattr(model, "predict_proba") else None

            if pred == 1:
                st.markdown("<h2 style='color:red;'>Yüksek risk (1)</h2>", unsafe_allow_html=True)
            else:
                st.markdown("<h2 style='color:green;'>Düşük risk (0)</h2>", unsafe_allow_html=True)
            if proba is not None:
                st.write(f"Olasılık: **{proba*100:.2f}%**")

            # SHAP açıklaması
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(df_input)

            st.subheader("Tahmin Açıklaması (SHAP)")
            shap.initjs()
            plt.figure(figsize=(10, 4))
            shap.force_plot(explainer.expected_value, shap_values[0], df_input.iloc[0], matplotlib=True, show=False)
            plt.tight_layout()
            st.pyplot(plt.gcf())
            plt.clf()

            # Kişisel öneri
            if pred == 1:
                st.info("📌 Öneri: Eğitim programlarına katılmanız ve denetimli serbestlik programına dahil olmanız önerilir.")
            else:
                st.success("🎉 Öneri: Düşük risk grubundasınız. Takip ve destek programlarına devam edin.")

        except Exception as e:
            st.error(f"Tahmin sırasında hata: {e}")

def analysis_page():
    st.title("📊 Veri Analizi ve Görselleştirme")

    age_column = "Age_at_Release"
    gender_column = "Gender"

    age_min, age_max = st.sidebar.slider(
        "Yaş Aralığı", 
        int(df[age_column].min()), 
        int(df[age_column].max()), 
        (int(df[age_column].min()), int(df[age_column].max()))
    )

    gender_options = df[gender_column].dropna().unique().tolist()
    gender_filter = st.sidebar.multiselect("Cinsiyet", options=gender_options, default=gender_options)

    filtered_df = df[
        (df[age_column] >= age_min) & 
        (df[age_column] <= age_max) & 
        (df[gender_column].isin(gender_filter))
    ]

    st.write(f"Toplam kayıt sayısı: {filtered_df.shape[0]}")

    fig = px.histogram(filtered_df, x="Recidivism_Within_3years", color="Recidivism_Within_3years", 
                       category_orders={"Recidivism_Within_3years": [0,1]},
                       labels={"Recidivism_Within_3years": "3 Yıl İçinde Yeniden Suç"},
                       title="Recidivism Sınıf Dağılımı")
    st.plotly_chart(fig)

    selected_feature = st.selectbox("Grafik için özellik seçin", options=feature_names)

    if selected_feature in cat_features or selected_feature in bool_cols:
        fig2 = px.histogram(filtered_df, x=selected_feature, color="Recidivism_Within_3years",
                            category_orders={selected_feature: filtered_df[selected_feature].dropna().unique().tolist()},
                            title=f"{selected_feature} Dağılımı")
    else:
        fig2 = px.box(filtered_df, x="Recidivism_Within_3years", y=selected_feature,
                      title=f"{selected_feature} Değişkeninin Recidivism'a Göre Dağılımı")
    st.plotly_chart(fig2)

def performance_page():
    st.title("📈 Model Performansı")

    y_true = df["Recidivism_Within_3years"]
    X = df[feature_names].copy()
    for b in bool_cols:
        if b in X.columns:
            X[b] = X[b].astype(str)
    y_pred = model.predict(X)

    st.subheader("Sınıflandırma Raporu")
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    st.dataframe(report_df)

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Tahmin Edilen")
    ax.set_ylabel("Gerçek")
    st.pyplot(fig)

def help_page():
    st.title("❓ Yardım ve Açıklamalar")
    st.markdown("""
    **Bu uygulama hakkında:**  
    - Mahkumların tekrar suç işleyip işlemediğini tahmin etmek için CatBoost modeli kullanılır.  
    - Veri analizi sayfasında veriler filtrelenip grafiklerle incelenebilir.  
    - Model performans sayfasında sınıflandırma raporu ve karışıklık matrisi gösterilir.  
    - Tahmin sayfasında kişisel risk tahmini ve açıklamaları bulunur.

    **Girdi alanlarının açıklamaları:**  
    """)
    for k, v in FEATURE_DESCRIPTIONS.items():
        st.markdown(f"- **{k}**: {v}")

def main():
    st.sidebar.title("Sayfa Seçimi")
    page = st.sidebar.selectbox("Sayfa", ["Tahmin", "Veri Analizi", "Model Performansı", "Yardım"])

    if page == "Tahmin":
        prediction_page()
    elif page == "Veri Analizi":
        analysis_page()
    elif page == "Model Performansı":
        performance_page()
    elif page == "Yardım":
        help_page()

if __name__ == "__main__":
    main()
