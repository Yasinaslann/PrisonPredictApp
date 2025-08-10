import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# --------- Dosya yolları ---------
DATA_PATH = "prison_app/Prisongüncelveriseti.csv"
MODEL_PATH = "prison_app/catboost_model.pkl"
BOOL_COLS_PATH = "prison_app/bool_columns.pkl"
CAT_FEATURES_PATH = "prison_app/cat_features.pkl"
CAT_UNIQUE_PATH = "prison_app/cat_unique_values.pkl"
FEATURE_NAMES_PATH = "prison_app/feature_names.pkl"

# --------- Load Fonksiyonları ---------
@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error(f"Veri dosyası bulunamadı! Lütfen '{DATA_PATH}' konumunu kontrol edin.")
        return None
    df = pd.read_csv(DATA_PATH)
    return df

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model dosyası bulunamadı! Lütfen '{MODEL_PATH}' konumunu kontrol edin.")
        return None
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_pickle(path):
    if not os.path.exists(path):
        st.error(f"Dosya bulunamadı! Lütfen '{path}' konumunu kontrol edin.")
        return None
    return joblib.load(path)

# --------- Veri Ön İşleme ---------
def preprocess(df, bool_cols):
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(bool)
    return df

# --------- Kullanıcı Arayüzü ve Sayfalar ---------
def sidebar_menu():
    st.sidebar.title("Navigasyon")
    return st.sidebar.radio("Sayfa Seçimi", ["Veri Analizi", "Tahmin", "Model Performansı", "Tahmin Geçmişi", "Yardım"])

# --------- Veri Analizi Sayfası ---------
def page_data_analysis(df, cat_features, cat_unique_values):
    st.header("📊 Veri Analizi ve Görselleştirme")

    # Filtreler
    st.sidebar.subheader("Filtreler")
    df_filtered = df.copy()
    for feat in cat_features:
        options = cat_unique_values.get(feat, [])
        if options:
            selected = st.sidebar.multiselect(f"{feat} seçin", options, default=options)
            df_filtered = df_filtered[df_filtered[feat].isin(selected)]

    # Sayısal filtreler örnek: Yaş
    if "Age_at_Release" in df_filtered.columns:
        min_age = int(df_filtered["Age_at_Release"].min())
        max_age = int(df_filtered["Age_at_Release"].max())
        age_range = st.sidebar.slider("Yaş Aralığı", min_age, max_age, (min_age, max_age))
        df_filtered = df_filtered[(df_filtered["Age_at_Release"] >= age_range[0]) & (df_filtered["Age_at_Release"] <= age_range[1])]

    st.write(f"Filtrelenmiş Kayıt Sayısı: {len(df_filtered)}")

    # Kategorik değişken dağılımı - Plotly
    for cat in cat_features:
        if cat in df_filtered.columns:
            fig = px.histogram(df_filtered, x=cat, color=cat, title=f"{cat} Dağılımı", hover_data=df_filtered.columns)
            st.plotly_chart(fig, use_container_width=True)

    # Sayısal değişken dağılımı
    num_cols = df_filtered.select_dtypes(include=np.number).columns.tolist()
    for num in num_cols:
        fig = px.histogram(df_filtered, x=num, nbins=30, title=f"{num} Dağılımı")
        st.plotly_chart(fig, use_container_width=True)

# --------- Tahmin Sayfası ---------
def page_prediction(df, model, feature_names, cat_features):
    st.header("🧮 Suç Tekrarı Tahmin Sayfası")

    st.markdown("Lütfen tahmin için aşağıdaki bilgileri doldurun:")

    input_dict = {}
    with st.form("predict_form"):
        for feat in feature_names:
            if feat in cat_features:
                vals = df[feat].dropna().unique().tolist()
                val = st.selectbox(f"{feat}", options=vals)
                input_dict[feat] = val
            elif df[feat].dtype == bool:
                val = st.checkbox(feat)
                input_dict[feat] = val
            else:
                min_val = int(df[feat].min()) if not df[feat].isnull().all() else 0
                max_val = int(df[feat].max()) if not df[feat].isnull().all() else 100
                val = st.number_input(feat, min_value=min_val, max_value=max_val, value=min_val)
                input_dict[feat] = val

        submit = st.form_submit_button("Tahmin Et")

    if submit:
        input_df = pd.DataFrame([input_dict])
        input_df = input_df[feature_names]

        for cat in cat_features:
            if cat in input_df.columns:
                input_df[cat] = input_df[cat].astype(str)

        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]

        st.success(f"Suç Tekrarı Tahmin Sonucu: {'Evet' if prediction == 1 else 'Hayır'}")
        st.info(f"Tekrar Suç İşleme Olasılığı: %{proba*100:.2f}")

        # Kişisel öneriler
        advice = generate_advice(proba)
        st.markdown(f"### Kişisel Öneriler:\n- {advice}")

        # SHAP açıklaması
        st.header("Tahmin Açıklaması (SHAP)")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)

        shap.initjs()
        st.pyplot(shap.summary_plot(shap_values, input_df, show=False))

        # Tahmin geçmişine kaydet (Streamlit session_state ile)
        if "predictions" not in st.session_state:
            st.session_state.predictions = []
        st.session_state.predictions.append({
            "input": input_dict,
            "prediction": prediction,
            "probability": proba
        })

# --------- Kişisel Öneri Sistemi ---------
def generate_advice(risk_score):
    if risk_score >= 0.75:
        return "Yüksek risk grubundasınız. Kesinlikle eğitime katılmalı ve denetimli serbestlik programına dahil olmalısınız."
    elif risk_score >= 0.5:
        return "Orta risk grubundasınız. Sosyal destek ve mesleki eğitim programlarına katılmanız önerilir."
    else:
        return "Düşük risk grubundasınız. Riskinizi azaltmak için olumlu sosyal aktiviteleri sürdürmelisiniz."

# --------- Model Performansı Sayfası ---------
def page_model_performance(df, model):
    st.header("📈 Model Performans Değerlendirmesi")

    if "Recidivism" not in df.columns:
        st.warning("Veri setinde hedef değişken 'Recidivism' bulunamadı.")
        return

    y_true = df["Recidivism"]
    X = df.drop(columns=["Recidivism"])

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    accuracy = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_proba)

    st.write(f"Accuracy: {accuracy:.3f}")
    st.write(f"ROC AUC: {roc_auc:.3f}")

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

    st.text("Sınıflandırma Raporu:")
    st.text(classification_report(y_true, y_pred))

# --------- Tahmin Geçmişi Sayfası ---------
def page_prediction_history():
    st.header("📋 Tahmin Geçmişi")

    if "predictions" not in st.session_state or len(st.session_state.predictions) == 0:
        st.info("Henüz tahmin yapılmadı.")
        return

    df_hist = pd.DataFrame([{
        **pred["input"],
        "Prediction": "Evet" if pred["prediction"] == 1 else "Hayır",
        "Risk_Score": f"{pred['probability']*100:.2f}%"
    } for pred in st.session_state.predictions])

    st.dataframe(df_hist)

    csv = df_hist.to_csv(index=False).encode("utf-8")
    st.download_button("Tahmin Geçmişini CSV Olarak İndir", data=csv, file_name="tahmin_gecmisi.csv", mime="text/csv")

# --------- Yardım Sayfası ---------
def page_help():
    st.header("ℹ️ Yardım ve Açıklamalar")
    st.markdown("""
    **Hapisten Tahliye Sonrası Suç Tekrarı Tahmin Uygulaması**:

    - Veri Analizi: Veri setindeki değişkenlerin interaktif grafiklerini inceleyebilirsiniz.
    - Tahmin: Kendi bilgilerinizi girerek suç tekrar riski tahmin edebilirsiniz.
    - Model Performansı: Modelin doğruluk ve ROC AUC gibi performans metriklerini görebilirsiniz.
    - Tahmin Geçmişi: Yaptığınız tahminlerin geçmişini görüntüleyebilir ve CSV olarak indirebilirsiniz.

    Herhangi bir sorun yaşarsanız, lütfen uygulama sahibine ulaşın.
    """)

# --------- Ana Fonksiyon ---------
def main():
    st.title("🔒 Hapisten Tahliye Sonrası Suç Tekrarı Tahmin Uygulaması")

    df = load_data()
    if df is None:
        st.stop()

    model = load_model()
    if model is None:
        st.stop()

    bool_cols = load_pickle(BOOL_COLS_PATH)
    cat_features = load_pickle(CAT_FEATURES_PATH)
    cat_unique_values = load_pickle(CAT_UNIQUE_PATH)
    feature_names = load_pickle(FEATURE_NAMES_PATH)

    if None in [bool_cols, cat_features, cat_unique_values, feature_names]:
        st.error("Bazı gerekli dosyalar yüklenemedi.")
        st.stop()

    df = preprocess(df, bool_cols)

    page = sidebar_menu()

    if page == "Veri Analizi":
        page_data_analysis(df, cat_features, cat_unique_values)
    elif page == "Tahmin":
        page_prediction(df, model, feature_names, cat_features)
    elif page == "Model Performansı":
        page_model_performance(df, model)
    elif page == "Tahmin Geçmişi":
        page_prediction_history()
    elif page == "Yardım":
        page_help()

if __name__ == "__main__":
    main()
