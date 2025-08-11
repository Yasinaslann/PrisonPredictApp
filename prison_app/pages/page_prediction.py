import streamlit as st
import pandas as pd
import joblib
import numpy as np
import sys

# İşte bu satır kesinlikle olmalı, bu hata için:
sys.modules['Index'] = pd.Index

st.set_page_config(page_title="Tahmin Modeli", layout="wide")

@st.cache_data(show_spinner=True)
def load_model_files():
    try:
        model = joblib.load("catboost_model.pkl")
        bool_cols = joblib.load("bool_columns.pkl")
        cat_features = joblib.load("cat_features.pkl")
        feature_names = joblib.load("feature_names.pkl")
        cat_unique_values = joblib.load("cat_unique_values.pkl")
        return model, bool_cols, cat_features, feature_names, cat_unique_values
    except Exception as e:
        st.error(f"Model dosyaları yüklenirken hata oluştu: {e}")
        return None, None, None, None, None

def convert_age_range(age_str):
    if pd.isna(age_str):
        return np.nan
    age_str = str(age_str).strip()
    if "-" in age_str:
        parts = age_str.split("-")
        try:
            low = int(parts[0])
            high = int(parts[1])
            return (low + high) / 2
        except:
            return np.nan
    elif "or older" in age_str:
        try:
            low = int(age_str.split()[0])
            return low + 2
        except:
            return np.nan
    else:
        try:
            return float(age_str)
        except:
            return np.nan

def main():
    st.title("📊 Tahmin Modeli")

    model, bool_cols, cat_features, feature_names, cat_unique_values = load_model_files()
    if model is None:
        return

    st.info("Model yüklendi. Lütfen tahmin için gerekli bilgileri doldurun.")

    input_data = {}

    for feat in feature_names:
        if feat == "Age_at_Release":
            options = ["18-22", "23-27", "28-32", "33-37", "38-42", "43-47", "48 or older"]
            input_data[feat] = st.selectbox(f"{feat} seçin", options)
        elif feat in bool_cols:
            val = st.selectbox(f"{feat} (bool)", ["False", "True"])
            input_data[feat] = True if val == "True" else False
        elif feat in cat_features:
            options = cat_unique_values.get(feat, [])
            if options:
                input_data[feat] = st.selectbox(f"{feat} seçin", options)
            else:
                input_data[feat] = st.text_input(f"{feat} girin")
        else:
            input_data[feat] = st.number_input(f"{feat} girin", value=0)

    if st.button("Tahmin Et"):
        X_pred = pd.DataFrame([input_data], columns=feature_names)

        if "Age_at_Release" in X_pred.columns:
            X_pred["Age_at_Release"] = X_pred["Age_at_Release"].apply(convert_age_range)

        for col in bool_cols:
            if col in X_pred.columns:
                X_pred[col] = X_pred[col].astype(str)

        try:
            pred_prob = model.predict_proba(X_pred)[:,1][0]
            pred_class = model.predict(X_pred)[0]

            st.success(f"✅ Tahmin Sonucu: {'Tekrar Suç İşledi' if pred_class == 1 else 'Tekrar Suç İşlemedi'}")
            st.info(f"📊 Yeniden Suç İşleme Olasılığı: %{pred_prob*100:.2f}")
        except Exception as e:
            st.error(f"Tahmin sırasında hata oluştu: {e}")

if __name__ == "__main__":
    main()
