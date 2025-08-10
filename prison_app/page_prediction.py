import streamlit as st
import pickle
import pandas as pd

# Model ve gerekli dosyaları yükleme
@st.cache_resource
def load_model():
    model = pickle.load(open("catboost_model.pkl", "rb"))
    bool_columns = pickle.load(open("bool_columns.pkl", "rb"))
    cat_features = pickle.load(open("cat_features.pkl", "rb"))
    feature_names = pickle.load(open("feature_names.pkl", "rb"))
    return model, bool_columns, cat_features, feature_names

model, bool_columns, cat_features, feature_names = load_model()

st.title("🧾 Suç İşleme Tahmin Sayfası")
st.markdown("Tahliye edilen kişinin yeniden suç işleme olasılığını tahmin edin.")

# Kullanıcıdan veri alma (örnek giriş alanları)
user_data = {}
for feature in feature_names:
    if feature in bool_columns:
        user_data[feature] = st.selectbox(f"{feature}", [0, 1])
    elif feature in cat_features:
        user_data[feature] = st.text_input(f"{feature}")
    else:
        user_data[feature] = st.number_input(f"{feature}", step=1)

# Tahmin butonu
if st.button("Tahmin Yap"):
    df_input = pd.DataFrame([user_data])
    prediction = model.predict(df_input)[0]
    if prediction == 1:
        st.error("⚠ Yüksek risk: Kişi yeniden suç işleyebilir.")
    else:
        st.success("✅ Düşük risk: Kişi yeniden suç işleme olasılığı düşük.")
