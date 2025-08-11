# pages/prediction.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Tahmin Sayfası", page_icon="🔍", layout="wide")
st.title("🔍 Tahmin Sayfası")
st.write("Lütfen aşağıdaki formu doldurarak tahmin alın.")

# Model ve yardımcı dosyaları yükle
model = joblib.load("catboost_model.pkl")
feature_names = joblib.load("feature_names.pkl")
cat_features = joblib.load("cat_features.pkl")
bool_columns = joblib.load("bool_columns.pkl")
cat_unique_values = joblib.load("cat_unique_values.pkl")

# Kullanıcı girişlerini saklamak için sözlük
user_input = {}

with st.form("prediction_form"):
    for feature in feature_names:
        if feature in bool_columns:
            user_input[feature] = st.checkbox(feature, value=False)
        elif feature in cat_features:
            options = cat_unique_values.get(feature, [])
            if options:
                user_input[feature] = st.selectbox(feature, options)
            else:
                user_input[feature] = st.text_input(feature, "")
        else:
            user_input[feature] = st.number_input(feature, value=0.0, step=1.0)

    submitted = st.form_submit_button("📊 Tahmin Et")

if submitted:
    # DataFrame oluştur
    input_df = pd.DataFrame([user_input])

    # Bool değerleri int'e çevir
    for col in bool_columns:
        if col in input_df.columns:
            input_df[col] = input_df[col].astype(int)

    # Tahmin yap
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    st.success(f"📌 Tahmin Sonucu: **{prediction}**")
    st.info(f"📈 Olasılıklar: {dict(zip(model.classes_, np.round(proba, 3)))}")
