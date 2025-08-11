import streamlit as st
import pickle
import pandas as pd
from pathlib import Path

def app():
    st.title("📊 Tahmin Modeli")
    st.write("Burada CatBoost modeli ile tahmin yapabilirsiniz.")

    base_dir = Path(__file__).parent.parent

    # Dosya yolları
    model_path = base_dir / "catboost_model.pkl"
    feature_path = base_dir / "feature_names.pkl"
    cat_features_path = base_dir / "cat_features.pkl"
    bool_columns_path = base_dir / "bool_columns.pkl"

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(feature_path, "rb") as f:
            feature_names = pickle.load(f)
        with open(cat_features_path, "rb") as f:
            cat_features = pickle.load(f)  # Kategorik sütun isimleri listesi
        with open(bool_columns_path, "rb") as f:
            bool_columns = pickle.load(f)  # Bool sütun isimleri listesi (örneğin True/False)
    except FileNotFoundError as e:
        st.error(f"Model veya özellik dosyası bulunamadı: {e}")
        return

    st.subheader("Veri Girişi")

    user_input = {}
    # Kategorik girişler için seçmeli input, sayısal için number_input, bool için checkbox
    for feature in feature_names:
        if feature in bool_columns:
            user_input[feature] = st.checkbox(f"{feature} (True/False):")
        elif feature in cat_features:
            # Kategorik özellik için text input veya selectbox kullanılabilir
            user_input[feature] = st.text_input(f"{feature} (Kategori):", value="")
        else:
            user_input[feature] = st.number_input(f"{feature} (Sayı):", value=0.0)

    if st.button("Tahmin Yap"):
        input_df = pd.DataFrame([user_input])

        # Bool sütunları bool türüne dönüştür
        for col in bool_columns:
            input_df[col] = input_df[col].astype(bool)

        # Kategorik ve sayısal ayrımı doğru yap
        # Sayısal kolonlar zaten number_input ile float geldi

        try:
            prediction = model.predict(input_df, cat_features=cat_features)[0]
            st.success(f"📌 Tahmin Sonucu: **{prediction}**")
        except Exception as e:
            st.error(f"Tahmin sırasında hata oluştu: {e}")
