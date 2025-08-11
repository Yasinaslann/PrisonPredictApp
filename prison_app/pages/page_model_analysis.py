import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt

def app():
    st.title("📈 Model Analizleri ve Harita")
    st.write("Modelin önem derecelerini ve analizlerini görebilirsiniz.")

    with open("catboost_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)

    importance = model.get_feature_importance()
    df_importance = pd.DataFrame({
        "Özellik": feature_names,
        "Önem": importance
    }).sort_values(by="Önem", ascending=False)

    st.subheader("Özellik Önem Grafiği")
    fig, ax = plt.subplots()
    ax.barh(df_importance["Özellik"], df_importance["Önem"])
    ax.invert_yaxis()
    st.pyplot(fig)
