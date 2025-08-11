import streamlit as st
import pickle
import pandas as pd

def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

# Model ve yardımcı dosyaları yükleme
model = load_pickle("catboost_model.pkl")
feature_names = load_pickle("feature_names.pkl")
cat_features = load_pickle("cat_features.pkl")
bool_columns = load_pickle("bool_columns.pkl")
cat_unique_values = load_pickle("cat_unique_values.pkl")

def app():
    st.title("📊 Tahmin Modeli")
    st.write("Gerekli bilgileri girerek tahmin alabilirsiniz.")

    user_input = {}
    for col in feature_names:
        if col in bool_columns:
            val = st.selectbox(col, [0, 1])
        elif col in cat_features:
            val = st.selectbox(col, cat_unique_values[col])
        else:
            val = st.number_input(col, step=1.0)
        user_input[col] = val

    if st.button("Tahmin Yap"):
        df = pd.DataFrame([user_input])
        pred = model.predict(df)[0]
        st.success(f"Model Tahmini: **{pred}**")
