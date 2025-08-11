import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# Dosya yolları (app.py ile aynı dizindeyse, uygun path ayarla)
BASE = Path(__file__).parent.parent  # pages klasörünün bir üstü
MODEL_PATH = BASE / "catboost_model.pkl"
FEATURES_PATH = BASE / "feature_names.pkl"
CAT_FEATURES_PATH = BASE / "cat_features.pkl"
CAT_UNIQUE_PATH = BASE / "cat_unique_values.pkl"

@st.cache_resource(show_spinner=False)
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

@st.cache_data(show_spinner=False)
def load_feature_names():
    with open(FEATURES_PATH, "rb") as f:
        features = pickle.load(f)
    return features

@st.cache_data(show_spinner=False)
def load_cat_features():
    with open(CAT_FEATURES_PATH, "rb") as f:
        cat_features = pickle.load(f)
    return cat_features

@st.cache_data(show_spinner=False)
def load_cat_uniques():
    with open(CAT_UNIQUE_PATH, "rb") as f:
        cat_uniques = pickle.load(f)
    return cat_uniques

def main():
    st.title("📊 Tahmin Modeli")

    model = load_model()
    feature_names = load_feature_names()
    cat_features = load_cat_features()
    cat_uniques = load_cat_uniques()

    st.write("Mahpusların yeniden suç işleme riskini tahmin etmek için bilgileri doldurun.")

    # Kullanıcıdan input alma (feature_names listesine göre dinamik yapabiliriz)
    input_data = {}
    for feat in feature_names:
        if feat in cat_features:
            options = cat_uniques.get(feat, [])
            input_data[feat] = st.selectbox(f"{feat} seçiniz:", options)
        else:
            # Sayısal özellik varsayıyoruz, aralığı dataset özelliklerine göre ayarla
            input_data[feat] = st.number_input(f"{feat} giriniz:", value=0)

    if st.button("Tahmin Et"):
        # Model giriş formatına göre DataFrame yap
        X = pd.DataFrame([input_data])

        # Eğer model CatBoost ise, cat_features parametresi ile tahmin yapabiliriz
        try:
            preds = model.predict_proba(X)[:, 1]  # Pozitif sınıf olasılığı
            risk_score = preds[0]
            st.success(f"Yeniden Suç İşleme Risk Skoru: %{risk_score*100:.2f}")
        except Exception as e:
            st.error(f"Tahmin yapılırken hata oluştu: {e}")

if __name__ == "__main__":
    main()
