import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve

BASE_DIR = Path(__file__).parent

MODEL_FILE = BASE_DIR / "catboost_model.pkl"
BOOL_FILE = BASE_DIR / "bool_columns.pkl"
CAT_FILE = BASE_DIR / "cat_features.pkl"
FEATURES_FILE = BASE_DIR / "feature_names.pkl"
CAT_UNIQUE_FILE = BASE_DIR / "cat_unique_values.pkl"
DATA_FILE = BASE_DIR / "Prisongüncelveriseti.csv"

FEATURE_DESCRIPTIONS = {
    # Örnek açıklamalar
    "Gender": "Mahkumun cinsiyeti",
    "Race": "Mahkumun ırkı",
    "Age_at_Release": "Tahliye yaşı",
    # Diğer özellikler...
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
                    if len(options) > 20:
                        v = st.text_input(col, value=options[0], help=help_text)
                    else:
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

            color = "red" if pred == 1 else "green"
            risk_text = "Yüksek risk (1)" if pred == 1 else "Düşük risk (0)"
            st.markdown(
                f"""
                <div style="border-radius: 10px; padding: 20px; background-color: #f9f9f9; box-shadow: 0 0 10px {color}; margin-top: 20px;">
                    <h2 style="color: {color}; text-align:center;">{risk_text}</h2>
                    <p style="text-align:center;">Olasılık: <strong>{proba*100:.2f}%</strong></p>
                </div>
                """, unsafe_allow_html=True
            )

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(df_input)

            st.subheader("Tahmin Açıklaması (SHAP Waterfall Plot)")
            fig, ax = plt.subplots(figsize=(12, 5))
            shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[0], df_input.iloc[0], max_display=10, show=False)
            st.pyplot(fig)
            plt.close(fig)

            if pred == 1:
                st.info("📌 Öneri: Eğitim programlarına katılmanız ve denetimli serbestlik programına dahil olmanız önerilir.")
            else:
                st.success("🎉 Öneri: Düşük risk grubundasınız. Destek programlarına devam edin.")

        except Exception as e:
            st.error(f"Tahmin sırasında hata: {e}")

def analysis_page():
    st.title("📊 Veri Analizi ve Görselleştirme")

    age_column = "Age_at_Release"
    gender_column = "Gender"

    df_clean = df.copy()
    df_clean[age_column] = pd.to_numeric(df_clean[age_column], errors='coerce')
    df_clean = df_clean.dropna(subset=[age_column, gender_column])

    age_min_val = int(df_clean[age_column].min())
    age_max_val = int(df_clean[age_column].max())

    age_min, age_max = st.sidebar.slider(
        "Yaş Aralığı",
        age_min_val,
        age_max_val,
        (age_min_val, age_max_val)
    )

    gender_options = df_clean[gender_column].unique().tolist()
    gender_filter = st.sidebar.multiselect("Cinsiyet", options=gender_options, default=gender_options)

    filtered_df = df_clean[
        (df_clean[age_column] >= age_min) &
        (df_clean[age_column] <= age_max) &
        (df_clean[gender_column].isin(gender_filter))
    ]

    st.write(f"**Filtrelenmiş Kayıt Sayısı:** {filtered_df.shape[0]}")

    fig = px.histogram(filtered_df, x="Recidivism_Within_3years", color="Recidivism_Within_3years",
                       category_orders={"Recidivism_Within_3years": [0, 1]},
                       labels={"Recidivism_Within_3years": "3 Yıl İçinde Yeniden Suç"},
                       title="Recidivism Sınıf Dağılımı",
                       color_discrete_map={0: "green", 1: "red"})
    st.plotly_chart(fig, use_container_width=True)

    selected_feature = st.selectbox("Özellik Seçin", options=feature_names)

    if selected_feature in cat_features or selected_feature in bool_cols:
        fig2 = px.histogram(filtered_df, x=selected_feature, color="Recidivism_Within_3years",
                            title=f"{selected_feature} Dağılımı")
    else:
        fig2 = px.box(filtered_df, x="Recidivism_Within_3years", y=selected_feature,
                      title=f"{selected_feature} Değişkeninin Recidivism'a Göre Dağılımı")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Özet İstatistikler")
    st.dataframe(filtered_df.describe().T.style.background_gradient(cmap="viridis"))

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
    st.dataframe(report_df.style.background_gradient(cmap="Blues"))

    st.subheader("Karışıklık Matrisi (Confusion Matrix)")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Tahmin Edilen")
    ax.set_ylabel("Gerçek")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("ROC Eğrisi ve Precision-Recall Eğrisi")
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        fig_roc = px.area(
            x=fpr, y=tpr,
            title=f'ROC Eğrisi (AUC={roc_auc:.2f})',
            labels=dict(x='False Positive Rate', y='True Positive Rate'),
            width=700, height=400
        )
        fig_roc.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )
        st.plotly_chart(fig_roc)

        fig_pr = px.area(
            x=recall, y=precision,
            title='Precision-Recall Eğrisi',
            labels=dict(x='Recall', y='Precision'),
            width=700, height=400
        )
        st.plotly_chart(fig_pr)
    else:
        st.warning("Model predict_proba özelliğine sahip değil, ROC ve PR eğrisi çizilemiyor.")

def feature_importance_page():
    st.title("📌 Özelliklerin Önemi ve Etkisi")

    explainer = shap.TreeExplainer(model)
    X_for_shap = df[feature_names].copy()
    for b in bool_cols:
        if b in X_for_shap.columns:
            X_for_shap[b] = X_for_shap[b].astype(str)

    shap_values = explainer.shap_values(X_for_shap)

    st.subheader("Özelliklerin Genel Önemi (SHAP summary plot)")
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_for_shap, plot_type="bar", show=False)
    st.pyplot(fig)
    plt.close(fig)

    feature_to_inspect = st.selectbox("Detaylı etkisini görmek istediğiniz özellik", options=feature_names)
    st.subheader(f"{feature_to_inspect} Özelliğinin SHAP Değerleri")

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    shap.dependence_plot(feature_to_inspect, shap_values, X_for_shap, show=False)
    st.pyplot(fig2)
    plt.close(fig2)

def main():
    st.sidebar.title("Sayfa Seçimi")
    page = st.sidebar.selectbox("Sayfa", ["Tahmin", "Veri Analizi", "Model Performansı", "Özelliklerin Önemi"])

    if page == "Tahmin":
        prediction_page()
    elif page == "Veri Analizi":
        analysis_page()
    elif page == "Model Performansı":
        performance_page()
    elif page == "Özelliklerin Önemi":
        feature_importance_page()

if __name__ == "__main__":
    main()
