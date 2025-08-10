import streamlit as st
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import base64

st.set_page_config(page_title="Prison Recidivism Prediction", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("NIJ_s_Recidivism_Encod_Update.csv")

@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model("catboost_model.pkl")
    return model

@st.cache_data
def load_pickle(filename):
    return joblib.load(filename)

def get_download_link(df, filename="data.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 CSV olarak indir</a>'
    return href

def main():
    st.title("🔒 Hapisten Tahliye Sonrası Suç Tekrarı Tahmin Uygulaması")

    # Veri ve model yükle
    df = load_data()
    model = load_model()

    cat_features = load_pickle("cat_features.pkl")  # ['Gender', 'Race', ...]
    cat_unique_values = load_pickle("cat_unique_values.pkl")  # {'Gender': ['M','F'], ...}
    feature_names = load_pickle("feature_names.pkl")  # ['Age_at_Release', 'Gender', ...]

    # Sayfa seçimi
    page = st.sidebar.radio("Sayfa Seç", ["Ana Sayfa", "Tahmin", "Veri Keşfi", "Model Performansı", "Tahmin Geçmişi", "Veri İndir"])

    if "pred_history" not in st.session_state:
        st.session_state.pred_history = []

    if page == "Ana Sayfa":
        st.markdown("""
        ### Proje Hakkında
        Hapisten tahliye sonrası suç tekrarını tahmin eden CatBoost tabanlı model uygulaması.
        Veri keşfi, tahmin, model performansı ve tahmin geçmişi özellikleri bulunur.
        """)

    elif page == "Tahmin":
        st.header("🧠 Suç Tekrarı Tahmini")

        user_input = {}
        for feat in feature_names:
            if feat in cat_features:
                opts = cat_unique_values.get(feat, [])
                user_input[feat] = st.selectbox(f"{feat} seçin", opts)
            else:
                min_v = int(df[feat].min())
                max_v = int(df[feat].max())
                med_v = int(df[feat].median())
                user_input[feat] = st.slider(f"{feat} girin", min_v, max_v, med_v)

        input_df = pd.DataFrame([user_input])

        if st.button("Tahmini Hesapla"):
            try:
                pred = model.predict(input_df)[0]
                pred_prob = model.predict_proba(input_df)[0][1]
                sonuc = "Suç Tekrarı Olabilir" if pred == 1 else "Suç Tekrarı Olmaz"

                st.success(f"Tahmin sonucu: {sonuc}")
                st.info(f"Tekrar suç işleme olasılığı: %{pred_prob*100:.2f}")

                st.session_state.pred_history.append({**user_input, "Tahmin": sonuc, "Olasılık": pred_prob})

            except Exception as e:
                st.error(f"Hata: {e}")

    elif page == "Tahmin Geçmişi":
        st.header("📋 Tahmin Geçmişi")
        if st.session_state.pred_history:
            hist_df = pd.DataFrame(st.session_state.pred_history)
            st.dataframe(hist_df)
            st.markdown(get_download_link(hist_df, "tahmin_gecmisi.csv"), unsafe_allow_html=True)
        else:
            st.info("Henüz tahmin yapılmadı.")

    elif page == "Veri Keşfi":
        st.header("📊 Veri Keşfi")

        st.write(df.head())
        st.write(f"Toplam kayıt sayısı: {len(df)}")

        recid_rate = df['Recidivism'].mean()
        st.metric("Suç Tekrar Oranı", f"{recid_rate:.2%}")

        fig, ax = plt.subplots()
        sns.histplot(df['Age_at_Release'], bins=30, kde=True, ax=ax)
        ax.set_title("Yaş Dağılımı")
        st.pyplot(fig)

        st.subheader("Kategorik Değişken Dağılımı")
        for cat_col in cat_features:
            fig2, ax2 = plt.subplots()
            sns.countplot(data=df, x=cat_col, order=df[cat_col].value_counts().index, ax=ax2)
            plt.xticks(rotation=45)
            st.pyplot(fig2)

    elif page == "Model Performansı":
        st.header("📈 Model Performansı")

        try:
            fi = model.get_feature_importance()
            fi_df = pd.DataFrame({"Feature": feature_names, "Importance": fi}).sort_values(by="Importance", ascending=False)

            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(data=fi_df, x="Importance", y="Feature", ax=ax)
            ax.set_title("Özellik Önem Düzeyi")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Performans kısmında hata: {e}")

    elif page == "Veri İndir":
        st.header("📥 Veri Setini İndir")
        st.markdown(get_download_link(df, "NIJ_s_Recidivism_Encod_Update.csv"), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
