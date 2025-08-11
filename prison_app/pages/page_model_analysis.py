import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def app():
    st.title("📈 Model Analizleri ve Harita")
    st.write("Modelin önem derecelerini ve analizlerini görebilirsiniz.")

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

    importance = model.get_feature_importance()
    df_importance = pd.DataFrame({
        "Özellik": feature_names,
        "Önem": importance
    }).sort_values(by="Önem", ascending=False)

    st.subheader("Özellik Önem Grafiği")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(df_importance["Özellik"], df_importance["Önem"], color="skyblue")
    ax.invert_yaxis()
    ax.set_xlabel("Önem")
    ax.set_title("Model Özellik Önem Grafiği")
    st.pyplot(fig)
    plt.close(fig)
