# app.py
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Recidivism Tahmin", layout="centered")

BASE_DIR = Path(__file__).parent

MODEL_FILE = BASE_DIR / "catboost_model.pkl"
BOOL_FILE = BASE_DIR / "bool_columns.pkl"
CAT_FILE = BASE_DIR / "cat_features.pkl"
FEATURES_FILE = BASE_DIR / "feature_names.pkl"
CATEGORIES_FILE = BASE_DIR / "cat_unique_values.pkl"

@st.cache_resource
def load_artifacts():
    if not MODEL_FILE.exists():
        st.error(f"Model dosyası bulunamadı: {MODEL_FILE.name}")
        return None

    model = joblib.load(MODEL_FILE)
    bool_cols = joblib.load(BOOL_FILE) if BOOL_FILE.exists() else []
    cat_features = joblib.load(CAT_FILE) if CAT_FILE.exists() else []
    feature_names = joblib.load(FEATURES_FILE) if FEATURES_FILE.exists() else getattr(model, "feature_names_", None)

    if CATEGORIES_FILE.exists():
        cat_unique_values = joblib.load(CATEGORIES_FILE)
    else:
        cat_unique_values = {}

    if feature_names is None:
        st.error("feature_names.pkl bulunamadı ve modelde feature_names_ yok.")
        return None

    return model, bool_cols, cat_features, feature_names, cat_unique_values

art = load_artifacts()
if art is None:
    st.stop()

model, bool_cols, cat_features, feature_names, cat_unique_values = art

# Başlık ve açıklama
st.markdown("""
    <h1 style='text-align: center; color: #1F77B4; font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;'>
    📊 Recidivism (3 Yıl) Tahmin Uygulaması
    </h1>
    <p style='text-align: center; font-size:18px; color: #333; font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;'>
    Lütfen aşağıdaki alanları dikkatlice doldurun ve tahmin yap butonuna tıklayın.
    </p>
""", unsafe_allow_html=True)

st.write("---")

input_data = {}
cols = st.columns(2)

DEFAULT_MIN = 0.0
DEFAULT_MAX = 100.0
DEFAULT_STEP = 0.1

for i, col in enumerate(feature_names):
    container = cols[i % 2]
    with container:
        st.markdown(f"**{col}**")
        if col in bool_cols:
            v = st.selectbox(f"{col} seçiniz", ["True", "False"], key=col)
        elif col in cat_features:
            options = cat_unique_values.get(col, [])
            if options:
                v = st.selectbox(f"{col} seçiniz", options, key=col)
            else:
                v = st.text_input(f"{col} giriniz", value="", key=col)
        else:
            # Sayısal için dinamik aralık eklemek istersen burayı özelleştir
            v = st.slider(f"{col} değeri", min_value=DEFAULT_MIN, max_value=DEFAULT_MAX, value=DEFAULT_MIN, step=DEFAULT_STEP, key=col)
        input_data[col] = v
    st.write("")  # Inputlar arası boşluk

st.write("---")

if st.button("🔮 Tahmin Yap"):
    try:
        df_input = pd.DataFrame([input_data], columns=feature_names)

        for b in bool_cols:
            if b in df_input.columns:
                df_input[b] = df_input[b].astype(str)

        pred = model.predict(df_input)[0]
        proba = model.predict_proba(df_input)[0][1] if hasattr(model, "predict_proba") else None

        if int(pred) == 1:
            st.markdown(f"<h3 style='color: red;'>Tahmin: Yüksek risk (1)</h3>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h3 style='color: green;'>Tahmin: Düşük risk (0)</h3>", unsafe_allow_html=True)

        if proba is not None:
            st.markdown(f"<p style='font-size:16px;'>Olasılık: <strong>{proba*100:.2f}%</strong></p>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Tahmin sırasında hata: {e}")
