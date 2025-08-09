# app.py
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Recidivism Tahmin", layout="centered")

BASE_DIR = Path(__file__).parent  # app.py dosyasının bulunduğu klasör

MODEL_FILE = BASE_DIR / "catboost_model.pkl"
BOOL_FILE = BASE_DIR / "bool_columns.pkl"
CAT_FILE = BASE_DIR / "cat_features.pkl"
FEATURES_FILE = BASE_DIR / "feature_names.pkl"

@st.cache_resource
def load_artifacts():
    if not MODEL_FILE.exists():
        st.error(f"Model dosyası bulunamadı: {MODEL_FILE.name}")
        return None

    model = joblib.load(MODEL_FILE)
    bool_cols = joblib.load(BOOL_FILE) if BOOL_FILE.exists() else []
    cat_features = joblib.load(CAT_FILE) if CAT_FILE.exists() else []
    feature_names = joblib.load(FEATURES_FILE) if FEATURES_FILE.exists() else getattr(model, "feature_names_", None)

    if feature_names is None:
        st.error("feature_names.pkl bulunamadı ve modelde feature_names_ yok.")
        return None

    return model, bool_cols, cat_features, feature_names

art = load_artifacts()
if art is None:
    st.stop()

model, bool_cols, cat_features, feature_names = art

st.title("📊 Recidivism (3 yıl) Tahmin Uygulaması")
st.write("Alanları doldurup tahmin yapın. Boolean sütunlar `True/False` string olarak modele verildi.")

# Input form
st.subheader("Girdi Alanları")
input_data = {}
cols = st.columns(2)
for i, col in enumerate(feature_names):
    container = cols[i % 2]
    with container:
        if col in bool_cols:
            v = st.selectbox(col, ["True", "False"])
        elif col in cat_features:
            v = st.text_input(col, value="")
        else:
            v = st.number_input(col, value=0.0, format="%.6f")
        input_data[col] = v

if st.button("🔮 Tahmin Yap"):
    try:
        df_input = pd.DataFrame([input_data], columns=feature_names)

        for b in bool_cols:
            if b in df_input.columns:
                df_input[b] = df_input[b].astype(str)

        pred = model.predict(df_input)[0]
        proba = model.predict_proba(df_input)[0][1] if hasattr(model, "predict_proba") else None

        st.success(f"Tahmin: {'Yüksek risk (1)' if int(pred) == 1 else 'Düşük risk (0)'}")
        if proba is not None:
            st.write(f"Olasılık: **{proba*100:.2f}%**")
    except Exception as e:
        st.error("Tahmin sırasında hata: " + str(e))
