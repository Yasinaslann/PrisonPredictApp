import streamlit as st
import pandas as pd
import numpy as np
import dill
import pickle
from pathlib import Path

BASE = Path(__file__).parent

@st.cache_data(show_spinner=False)
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_model(path):
    with open(path, "rb") as f:
        return dill.load(f)

def main():
    st.title("📊 Tahmin Modeli")

    # Dosya yolları
    model_path = BASE / "catboost_model.pkl"
    features_path = BASE / "feature_names.pkl"
    bool_cols_path = BASE / "bool_columns.pkl"
    cat_features_path = BASE / "cat_features.pkl"
    cat_unique_path = BASE / "cat_unique_values.pkl"

    # Model ve dosyaları yükle
    try:
        model = load_model(model_path)
        feature_names = load_pickle(features_path)
        bool_columns = load_pickle(bool_cols_path)
        cat_features = load_pickle(cat_features_path)
        cat_unique_values = load_pickle(cat_unique_path)
    except Exception as e:
        st.error(f"Model dosyaları yüklenirken hata oluştu: {e}")
        return

    st.info("Model başarıyla yüklendi. Tahmin için lütfen özellikleri girin.")

    # Kullanıcıdan input al
    input_data = {}

    for feature in feature_names:
        if feature in bool_columns:
            # Boolean input
            val = st.checkbox(feature.replace("_", " "), value=False)
            input_data[feature] = int(val)
        elif feature in cat_features:
            # Kategorik input
            options = cat_unique_values.get(feature, [])
            val = st.selectbox(feature.replace("_", " "), options)
            input_data[feature] = val
        else:
            # Sayısal input
            val = st.number_input(feature.replace("_", " "), value=0.0)
            input_data[feature] = val

    if st.button("Tahmin Yap"):
        # DataFrame'e çevir
        input_df = pd.DataFrame([input_data])

        # Kategorik değişkenleri cat_features olarak belirt
        try:
            preds = model.predict_proba(input_df)
            prob = preds[0][1]  # İkinci sınıfın olasılığı (recidivism = 1)
            st.success(f"Mahpusun yeniden suç işleme olasılığı: %{prob*100:.2f}")
        except Exception as e:
            st.error(f"Tahmin yapılırken hata oluştu: {e}")

if __name__ == "__main__":
    main()
