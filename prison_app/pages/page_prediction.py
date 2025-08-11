import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

st.title("📊 Tahmin Modeli")

BASE = Path(__file__).parent.parent

# Model dosyalarını yükle
try:
    model = pickle.load(open(BASE / "catboost_model.pkl", "rb"))
    feature_names = pickle.load(open(BASE / "feature_names.pkl", "rb"))
    bool_columns = pickle.load(open(BASE / "bool_columns.pkl", "rb"))
    cat_features = pickle.load(open(BASE / "cat_features.pkl", "rb"))
    cat_unique_values = pickle.load(open(BASE / "cat_unique_values.pkl", "rb"))
except FileNotFoundError as e:
    st.error(f"❌ Model dosyası bulunamadı: {e}")
    st.stop()

# Kullanıcıdan giriş alma
inputs = {}
for cat_feat in cat_features:
    options = cat_unique_values.get(cat_feat, [])
    inputs[cat_feat] = st.selectbox(f"{cat_feat.replace('_',' ')}", options)

for bool_col in bool_columns:
    inputs[bool_col] = st.checkbox(f"{bool_col.replace('_',' ')}")

if "Sentence_Length_Months" in feature_names:
    inputs["Sentence_Length_Months"] = st.number_input("Ceza Süresi (Ay)", 0, 600, 12)

# Eksik özellikleri 0 ile doldur
for feat in feature_names:
    if feat not in inputs:
        inputs[feat] = 0

input_df = pd.DataFrame([inputs], columns=feature_names)

# Tahmin butonu
if st.button("Tahmin Et"):
    try:
        pred_proba = model.predict_proba(input_df)[0][1]
        pred_label = model.predict(input_df)[0]
        st.success(f"📊 Olasılık: %{pred_proba*100:.2f}")
        st.info("🔍 Tahmin: " + ("Tekrar suç işleyebilir" if pred_label == 1 else "Tekrar suç işlemez"))
    except Exception as e:
        st.error(f"⚠️ Hata: {e}")
