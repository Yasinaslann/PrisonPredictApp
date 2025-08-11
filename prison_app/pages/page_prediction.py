import streamlit as st
import pickle
import pandas as pd
from pathlib import Path

def app():
    st.title("📊 Tahmin Modeli")
    st.write("Burada CatBoost modeli ile tahmin yapabilirsiniz.")

    base_dir = Path(__file__).parent.parent

    model_path = base_dir / "catboost_model.pkl"
    feature_path = base_dir / "feature_names.pkl"

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(feature_path, "rb") as f:
            feature_names = pickle.load(f)
    except FileNotFoundError as e:
        st.error(f"Model veya özellik dosyası bulunamadı: {e}")
        return

    st.subheader("Veri Girişi")
    user_input = {}
    for feature in feature_names:
        user_input[feature] = st.number_input(f"{feature}:", value=0.0)

    if st.button("Tahmin Yap"):
        input_df = pd.DataFrame([user_input])
        prediction = model.predict(input_df)[0]
        st.success(f"📌 Tahmin Sonucu: **{prediction}**")
