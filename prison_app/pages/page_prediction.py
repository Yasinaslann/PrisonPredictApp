import streamlit as st
import pandas as pd
from pathlib import Path
import pickle

st.set_page_config(
    page_title="Tahmin Modeli",
    page_icon="📊"
)

BASE = Path(__file__).parent.parent

def load_model_files():
    try:
        model_path = BASE / "catboost_model.pkl"
        features_path = BASE / "feature_names.pkl"
        bool_cols_path = BASE / "bool_columns.pkl"
        cat_features_path = BASE / "cat_features.pkl"
        cat_unique_path = BASE / "cat_unique_values.pkl"

        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(features_path, "rb") as f:
            feature_names = pickle.load(f)
        with open(bool_cols_path, "rb") as f:
            bool_columns = pickle.load(f)
        with open(cat_features_path, "rb") as f:
            cat_features = pickle.load(f)
        with open(cat_unique_path, "rb") as f:
            cat_unique_values = pickle.load(f)
        return model, feature_names, bool_columns, cat_features, cat_unique_values
    except Exception as e:
        st.error(f"Model dosyaları yüklenirken hata oluştu: {e}")
        return None, None, None, None, None

def page_prediction():
    st.title("📊 Tahmin Modeli")

    model, feature_names, bool_columns, cat_features, cat_unique_values = load_model_files()
    if model is None:
        return

    st.markdown("Lütfen tahmin yapmak için aşağıdaki bilgileri giriniz:")

    inputs = {}

    for cat_feat in cat_features:
        options = cat_unique_values.get(cat_feat, [])
        inputs[cat_feat] = st.selectbox(f"{cat_feat.replace('_', ' ')} seçin:", options)

    for bool_col in bool_columns:
        inputs[bool_col] = st.checkbox(f"{bool_col.replace('_', ' ')}")

    if "Sentence_Length_Months" in feature_names:
        inputs["Sentence_Length_Months"] = st.number_input(
            "Ceza Süresi (Ay)", min_value=0, max_value=600, value=12
        )

    input_df = pd.DataFrame([inputs], columns=feature_names)

    if st.button("Tahmin Et"):
        try:
            pred_proba = model.predict_proba(input_df)[0][1]
            pred_label = model.predict(input_df)[0]
            st.success(f"Yeniden suç işleme olasılığı: %{pred_proba*100:.2f}")
            st.info(f"Tahmin sonucu: {'Tekrar suç işleyebilir' if pred_label == 1 else 'Tekrar suç işlemez'}")
        except Exception as e:
            st.error(f"Tahmin sırasında hata oluştu: {e}")

if __name__ == "__main__":
    page_prediction()
