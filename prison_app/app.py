import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import shap
import plotly.express as px
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).parent

# Dosya yolları
MODEL_FILE = BASE_DIR / "catboost_model.pkl"
BOOL_FILE = BASE_DIR / "bool_columns.pkl"
CAT_FILE = BASE_DIR / "cat_features.pkl"
FEATURES_FILE = BASE_DIR / "feature_names.pkl"
CATEGORIES_FILE = BASE_DIR / "cat_unique_values.pkl"
DATA_FILE = BASE_DIR / "Prisongüncelveriseti.csv"  # Eğitim verisi

# Sayfa başlıkları
PAGES = ["Tahmin", "Veri Analizi", "Model Performansı", "Yardım"]

# Feature açıklamaları (kendi datasetine göre ekle)
FEATURE_DESCRIPTIONS = {
    "Age": "Kişinin yaşı.",
    "Gender": "Cinsiyet (Erkek/Kadın).",
    "Previous_Convictions": "Önceki suç sayısı.",
    # Buraya datasetindeki diğer özellikler eklenmeli
}

# Yükleme fonksiyonları
@st.cache_resource
def load_model_and_data():
    model = joblib.load(MODEL_FILE)
    bool_cols = joblib.load(BOOL_FILE)
    cat_features = joblib.load(CAT_FILE)
    feature_names = joblib.load(FEATURES_FILE)
    cat_unique_values = joblib.load(CATEGORIES_FILE)
    df = pd.read_csv(DATA_FILE)
    return model, bool_cols, cat_features, feature_names, cat_unique_values, df

def main():
    st.set_page_config(page_title="Recidivism Tahmin Uygulaması", layout="wide")
    st.sidebar.title("Navigasyon")
    page = st.sidebar.radio("Sayfalar", PAGES)

    model, bool_cols, cat_features, feature_names, cat_unique_values, df = load_model_and_data()

    if page == "Tahmin":
        show_prediction_page(model, bool_cols, cat_features, feature_names, cat_unique_values)
    elif page == "Veri Analizi":
        show_analysis_page(df)
    elif page == "Model Performansı":
        show_performance_page(model, df)
    else:
        show_help_page()

def show_prediction_page(model, bool_cols, cat_features, feature_names, cat_unique_values):
    st.title("🔮 Recidivism Tahmin Sayfası")

    input_data = {}
    cols = st.columns(2)
    for i, col in enumerate(feature_names):
        container = cols[i % 2]
        with container:
            label = f"{col}"
            if col in FEATURE_DESCRIPTIONS:
                label += " ⓘ"
            st.markdown(f"**{label}**")
            if col in bool_cols:
                v = st.selectbox(f"{col} seçiniz", ["True", "False"], key=col)
            elif col in cat_features:
                options = cat_unique_values.get(col, [])
                if options:
                    v = st.selectbox(f"{col} seçiniz", options, key=col)
                else:
                    v = st.text_input(f"{col} giriniz", value="", key=col)
            else:
                v = st.number_input(f"{col} değeri", value=0.0, step=0.1, key=col)
            input_data[col] = v
            if col in FEATURE_DESCRIPTIONS:
                st.caption(FEATURE_DESCRIPTIONS[col])

    if st.button("Tahmin Yap"):
        try:
            df_input = pd.DataFrame([input_data], columns=feature_names)
            for b in bool_cols:
                if b in df_input.columns:
                    df_input[b] = df_input[b].astype(str)

            pred = model.predict(df_input)[0]
            proba = model.predict_proba(df_input)[0][1] if hasattr(model, "predict_proba") else None

            if int(pred) == 1:
                st.markdown(f"<h3 style='color:red;'>Tahmin: Yüksek risk (1)</h3>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h3 style='color:green;'>Tahmin: Düşük risk (0)</h3>", unsafe_allow_html=True)

            if proba is not None:
                st.write(f"Olasılık: **{proba*100:.2f}%**")

            # SHAP açıklaması
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(df_input)

            st.subheader("Tahmin Açıklaması (SHAP Değerleri)")
            shap.initjs()
            st.pyplot(shap.plots._force.pyplot(
                shap.force_plot(explainer.expected_value, shap_values[0], df_input.iloc[0]), bbox_inches='tight'
            ))

            # Kişisel öneri
            st.subheader("Kişisel Öneri")
            if int(pred) == 1:
                st.info("Yüksek risk nedeniyle, eğitime katılım ve denetimli serbestlik programlarına dahil olmanız önerilir.")
            else:
                st.success("Düşük risk ile değerlendirilmişsiniz. Takip ve standart prosedürler yeterli olacaktır.")

        except Exception as e:
            st.error(f"Hata oluştu: {e}")

def show_analysis_page(df):
    st.title("📊 Veri Analizi ve Görselleştirme")

    st.sidebar.header("Filtreler")
    age_min, age_max = st.sidebar.slider("Yaş Aralığı", int(df["Age"].min()), int(df["Age"].max()), (int(df["Age"].min()), int(df["Age"].max())))
    gender_options = df["Gender"].unique().tolist()
    gender_filter = st.sidebar.multiselect("Cinsiyet", options=gender_options, default=gender_options)

    filtered_df = df[(df["Age"] >= age_min) & (df["Age"] <= age_max) & (df["Gender"].isin(gender_filter))]

    st.write(f"**Filtrelenmiş veri sayısı:** {filtered_df.shape[0]}")

    st.subheader("Recidivism Sınıf Dağılımı")
    fig1 = px.histogram(filtered_df, x="Recidivism_Within_3years", color="Recidivism_Within_3years",
                        category_orders={"Recidivism_Within_3years": [0, 1]},
                        labels={"Recidivism_Within_3years": "Recidivism (3 yıl içinde)"},
                        color_discrete_map={0: "green", 1: "red"})
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Özellik Dağılımı Seç")
    feature = st.selectbox("Görselleştirilecek Özellik", options=df.columns.drop(["Recidivism_Within_3years", "ID", "Training_Sample"]))
    if df[feature].dtype == "object":
        fig2 = px.histogram(filtered_df, x=feature, color="Recidivism_Within_3years", barmode="group")
    else:
        fig2 = px.box(filtered_df, x="Recidivism_Within_3years", y=feature, points="all")
    st.plotly_chart(fig2, use_container_width=True)

def show_performance_page(model, df):
    st.title("📈 Model Performans Sayfası")

    X = df.drop(columns=["Recidivism_Within_3years", "ID", "Training_Sample"])
    y = df["Recidivism_Within_3years"]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_pred = model.predict(X_test)

    st.subheader("Sınıflandırma Raporu")
    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    st.dataframe(df_report.style.format({"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}", "support": "{:.0f}"}))

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Düşük Risk", "Yüksek Risk"], yticklabels=["Düşük Risk", "Yüksek Risk"])
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")
    st.pyplot(fig)

def show_help_page():
    st.title("❓ Yardım ve Açıklamalar")

    st.markdown("""
    ### Girdi Alanları Açıklamaları
    - **Age**: Kişinin yaşı.
    - **Gender**: Cinsiyet, örn. Erkek veya Kadın.
    - **Previous_Convictions**: Önceki suç sayısı.
    - *Diğer alanlar için dataset belgelerine bakınız.*

    ### Model Kullanımı
    - Bu uygulama kişilerin 3 yıl içindeki suç tekrarı (recidivism) riskini tahmin eder.
    - Tahmin sonuçları sadece destek amaçlıdır, karar verilirken diğer faktörler de göz önünde bulundurulmalıdır.
    - Model, geçmiş verilere dayalı olup %100 doğruluk garanti etmez.

    ### İletişim
    - Herhangi bir sorun veya öneri için [email@example.com](mailto:email@example.com) adresinden ulaşabilirsiniz.
    """)

if __name__ == "__main__":
    main()
