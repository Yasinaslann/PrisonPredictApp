# app.py
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import io

st.set_page_config(page_title="Recidivism Tahmin Uygulaması", layout="wide")

# Dosya yolları
MODEL_PATH = Path("prison_app/catboost_model.pkl")
BOOL_PATH = Path("prison_app/bool_columns.pkl")
CAT_FEAT_PATH = Path("prison_app/cat_features.pkl")
FEATURES_PATH = Path("prison_app/feature_names.pkl")
CAT_UNIQ_PATH = Path("prison_app/cat_unique_values.pkl")
DATA_PATH = Path("prison_app/Prisongüncelveriseti.csv")


@st.cache_resource(show_spinner=False)
def load_resources():
    if not MODEL_PATH.exists():
        st.error(f"Model dosyası bulunamadı: {MODEL_PATH}")
        st.stop()
    if not DATA_PATH.exists():
        st.error(f"Veri dosyası bulunamadı: {DATA_PATH}")
        st.stop()

    model = joblib.load(MODEL_PATH)
    bool_cols = joblib.load(BOOL_PATH)
    cat_features = joblib.load(CAT_FEAT_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    cat_unique_values = joblib.load(CAT_UNIQ_PATH)
    df_data = pd.read_csv(DATA_PATH)
    return model, bool_cols, cat_features, feature_names, cat_unique_values, df_data


def prediction_page(model, bool_cols, cat_features, feature_names, cat_unique_values):
    st.title("📈 Recidivism Tahmin Sayfası")

    st.markdown(
        """
        Bu sayfada girdilerinizi doldurarak suçun tekrar işlenme olasılığını tahmin edebilirsiniz.
        Her alanın yanında bilgi ikonuna tıklayarak açıklamalarını görebilirsiniz.
        """
    )

    input_data = {}
    cols = st.columns(2)

    # Örnek açıklamalar (bunları veri sözlüğüne göre detaylandırabilirsiniz)
    feature_explanations = {
        "Gender": "Cinsiyet (Male/Female)",
        "Age_at_Release": "Serbest bırakılma anındaki yaş",
        "Education_Level": "Eğitim seviyesi",
        "Gang_Affiliated": "Çete ile bağlantılı mı? (True/False)",
        # Diğer özellikler için buraya ekleyin
    }

    for i, feat in enumerate(feature_names):
        container = cols[i % 2]
        with container:
            label = feat
            help_text = feature_explanations.get(feat, "Bilgi yok")
            if feat in bool_cols:
                val = st.selectbox(f"{label} ❓", options=["True", "False"], help=help_text)
            elif feat in cat_features:
                options = cat_unique_values.get(feat, [""])
                val = st.selectbox(f"{label} ❓", options=options, help=help_text)
            else:
                val = st.number_input(f"{label} ❓", value=0.0, format="%.3f", help=help_text)
            input_data[feat] = val

    if st.button("🔮 Tahmin Yap"):
        try:
            df_input = pd.DataFrame([input_data], columns=feature_names)
            # Boolean stringe çevir
            for b in bool_cols:
                if b in df_input.columns:
                    df_input[b] = df_input[b].astype(str)

            pred = model.predict(df_input)[0]
            proba = model.predict_proba(df_input)[0][1] if hasattr(model, "predict_proba") else None

            st.success(f"Tahmin: {'Yüksek Risk (1)' if int(pred) == 1 else 'Düşük Risk (0)'}")
            if proba is not None:
                st.write(f"Olasılık: **{proba*100:.2f}%**")

            # SHAP Açıklaması
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(df_input)
            st.subheader("Tahmin Açıklaması (SHAP Değerleri)")

            # Matplotlib figür ile SHAP görselleştirmesi
            fig, ax = plt.subplots(figsize=(8, 4))
            shap.waterfall_plot(shap.Explanation(values=shap_values[0], 
                                                base_values=explainer.expected_value, 
                                                data=df_input.iloc[0]))
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Tahmin sırasında hata oluştu: {str(e)}")


def analysis_page(df):
    st.title("📊 Veri Analizi ve Görselleştirme")

    st.sidebar.header("Filtreler")

    # Örneğin 'Age_at_Release' sütununa göre filtreleme
    age_min = int(df["Age_at_Release"].min())
    age_max = int(df["Age_at_Release"].max())
    age_range = st.sidebar.slider("Yaş Aralığı", age_min, age_max, (age_min, age_max))

    gender_options = list(df["Gender"].unique())
    gender_filter = st.sidebar.multiselect("Cinsiyet", options=gender_options, default=gender_options)

    race_options = list(df["Race"].unique())
    race_filter = st.sidebar.multiselect("Irk", options=race_options, default=race_options)

    df_filtered = df[
        (df["Age_at_Release"] >= age_range[0]) &
        (df["Age_at_Release"] <= age_range[1]) &
        (df["Gender"].isin(gender_filter)) &
        (df["Race"].isin(race_filter))
    ]

    st.write(f"Filtrelenmiş veri sayısı: {df_filtered.shape[0]}")

    # Plotly grafikler
    fig1 = px.histogram(df_filtered, x="Age_at_Release", nbins=30, title="Yaş Dağılımı")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.pie(df_filtered, names="Gender", title="Cinsiyet Oranları")
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.bar(df_filtered["Race"].value_counts().reset_index(), x="index", y="Race", title="Irk Dağılımı")
    st.plotly_chart(fig3, use_container_width=True)


def performance_page(model, df):
    st.title("📈 Model Performans Sayfası")

    from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
    import seaborn as sns

    # Gerçek ve tahmin değerleri
    X = df.drop(columns=["Recidivism_Within_3years", "ID", "Training_Sample"])
    y = df["Recidivism_Within_3years"]

    # Ön işleme (bool sütunlar stringe çevriliyor)
    bool_cols = X.select_dtypes(include=['bool']).columns
    X[bool_cols] = X[bool_cols].astype(str)

    y_pred = model.predict(X)

    st.subheader("Sınıflandırma Raporu")
    report_dict = classification_report(y, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    st.dataframe(report_df.style.background_gradient(cmap="RdYlGn"))

    st.subheader("ROC Eğrisi")
    y_prob = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    st.pyplot(fig)

    st.subheader("Karışıklık Matrisi (Confusion Matrix)")
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    ax.set_xlabel('Tahmin')
    ax.set_ylabel('Gerçek')
    st.pyplot(fig)


def main():
    model, bool_cols, cat_features, feature_names, cat_unique_values, df_data = load_resources()

    st.sidebar.title("Sayfalar")
    page = st.sidebar.radio("Sayfa Seçiniz", ["Tahmin", "Veri Analizi", "Model Performansı"])

    if page == "Tahmin":
        prediction_page(model, bool_cols, cat_features, feature_names, cat_unique_values)
    elif page == "Veri Analizi":
        analysis_page(df_data)
    elif page == "Model Performansı":
        performance_page(model, df_data)


if __name__ == "__main__":
    main()
