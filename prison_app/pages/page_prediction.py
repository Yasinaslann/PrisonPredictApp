import streamlit as st
import pickle
import pandas as pd

def app():
    st.title("📊 Tahmin Modeli")
    st.write("Burada CatBoost modeli ile tahmin yapabilirsiniz.")

    # Model ve özellikler yükleniyor
    with open("catboost_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)

    st.subheader("Veri Girişi")
    user_input = {}
    for feature in feature_names:
        user_input[feature] = st.number_input(f"{feature}:", value=0.0)

    if st.button("Tahmin Yap"):
        input_df = pd.DataFrame([user_input])
        prediction = model.predict(input_df)[0]
        st.success(f"📌 Tahmin Sonucu: **{prediction}**")
