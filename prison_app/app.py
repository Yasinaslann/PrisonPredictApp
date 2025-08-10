import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, RocCurveDisplay
)
import base64
import io

# --- Sabitler ve Dosya Yolları ---
MODEL_PATH = "prison_app/catboost_model.pkl"
CAT_FEATURES_PATH = "prison_app/cat_features.pkl"
FEATURE_NAMES_PATH = "prison_app/feature_names.pkl"
DATA_PATH = "prison_app/Prisongüncelveriseti.csv"

# --- Yükleme Fonksiyonu ---
@st.cache_data(show_spinner=True)
def load_resources():
    # Model
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    # Kategorik özellikler
    with open(CAT_FEATURES_PATH, "rb") as f:
        cat_features = pickle.load(f)
    # Özellik isimleri (modelde kullanılan)
    with open(FEATURE_NAMES_PATH, "rb") as f:
        feature_names = pickle.load(f)
    # Veri
    df = pd.read_csv(DATA_PATH)
    return model, cat_features, feature_names, df

# --- Yardımcı Fonksiyonlar ---
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def preprocess_input(df_input, cat_features):
    # Kategorik kolonları string yap
    for col in cat_features:
        if col in df_input.columns:
            df_input[col] = df_input[col].astype(str)
    return df_input

def shap_plot(model, df_input, cat_features):
    explainer = shap.TreeExplainer(model)
    pool = Pool(df_input, cat_features=cat_features)
    shap_values = explainer.shap_values(pool)
    # Matplotlib figure üretelim, Streamlit'te gösterirken fig geçeceğiz
    fig, ax = plt.subplots(figsize=(10, 5))
    shap.waterfall_plot(shap.Explanation(values=shap_values[0], 
                                         base_values=explainer.expected_value, 
                                         data=df_input.iloc[0]))
    plt.tight_layout()
    return fig

def sidebar_info():
    st.sidebar.markdown("""
    ## 📋 Proje Hakkında
    Bu uygulama, cezaevinden çıkış yapan kişilerin tekrar suç işleme risklerini tahmin etmek için geliştirilmiştir.  
    Model, çeşitli kişisel ve sosyal veriler kullanılarak CatBoost algoritması ile eğitilmiştir.  
    Proje amacı, adalet sisteminde risk yönetimi ve müdahaleye destek sağlamaktır.
    """)

# --- Ana Sayfa ---
def home_page(df):
    st.title("🚀 Cezaevi Tekrar Suç Tahmini Projesi")
    st.markdown("""
    ### Proje Tanıtımı
    Bu uygulama, cezaevinden çıkış sonrası 3 yıl içinde tekrar suç işleme riskini tahmin eder.  
    Kullanılan veri seti, çeşitli sosyal ve kişisel değişkenleri içerir.  
    Model, **CatBoost** ile eğitilmiş ve yüksek başarımlı sonuçlar vermektedir.
    
    ### Veri Seti Genel Bilgiler
    - Toplam kayıt sayısı: `{}`  
    - Kullanılan özellik sayısı: `{}`  
    - Hedef değişken: `Recidivism_Within_3years` (Tekrar suç işleme)
    
    ### Veri Setindeki Bazı Önemli Değişkenler
    - Yaş (Age_at_Release)  
    - Cinsiyet (Gender)  
    - Etnik Köken (Race)  
    - Eğitim Düzeyi (Education_Level)  
    - Suç Geçmişi ve Denetim Puanı (Supervision_Risk_Score_First)
    
    Uygulamanın farklı sayfalarında tahmin, veri analizi ve model performansını inceleyebilirsiniz.
    """.format(len(df), len(df.columns)))

    # Basit genel istatistikler tablo
    st.subheader("Veri Seti Örnek Satırları")
    st.dataframe(df.head(10))

# --- Tahmin Sayfası ---
def prediction_page(model, cat_features, feature_names):
    st.title("🧠 Suç Tekrarı Tahmini")

    st.markdown("""
    Aşağıdaki alanları doldurarak kişiye özel tekrar suç işleme risk tahmini yapabilirsiniz.  
    Her alanın yanında açıklamalar bulunmaktadır.  
    Değerleri değiştirebilir veya önerilen varsayılanları kullanabilirsiniz.
    """)

    # Kullanıcı girdilerini al
    input_data = {}
    # Burada feature_names listesindeki tüm özellikler için input hazırlıyoruz:
    for feature in feature_names:
        if feature == "ID":  # ID almayalım
            continue
        # Örnek tip ayrımı (sen datasetini inceleyip uygun olanı genişletebilirsin)
        if feature in cat_features:
            # Kategorik -> dropdown
            options = None
            # Cat unique values varsa onları kullanabilirsin, yoksa boş bırak
            # Biz varsayılan boş bırakıyoruz
            val = st.selectbox(f"{feature} ❓", options or ["Bilinmiyor", "Var", "Yok"], index=0,
                              help=f"{feature} hakkında bilgi.")
            input_data[feature] = val
        else:
            # Sayısal -> number input
            val = st.number_input(f"{feature} ❓", value=0,
                                 help=f"{feature} hakkında bilgi.",
                                 format="%d")
            input_data[feature] = val

    # DataFrame haline getir
    df_input = pd.DataFrame([input_data])
    # Kategorik tip dönüşümü
    df_input = preprocess_input(df_input, cat_features)

    # Tahmin butonu
    if st.button("🔮 Tahmini Yap"):
        try:
            pool = Pool(df_input, cat_features=cat_features)
            pred = model.predict(pool)[0]
            pred_proba = model.predict_proba(pool)[0][1] if hasattr(model, "predict_proba") else None

            st.markdown(f"### Tahmin Sonucu: {'Yüksek Risk' if pred == 1 else 'Düşük Risk'}")
            if pred_proba is not None:
                st.progress(int(pred_proba * 100))
                st.write(f"Risk Skoru: {pred_proba:.2f}")

            # SHAP açıklaması
            st.subheader("Tahmin Açıklaması (Özelliklerin Etkisi)")
            fig = shap_plot(model, df_input, cat_features)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Tahmin sırasında hata oluştu: {e}")

# --- Veri Analizi Sayfası ---
def analysis_page(df):
    st.title("📊 Gelişmiş Veri Analizi ve Görselleştirme")

    st.markdown("Bu sayfada veri setindeki çeşitli özelliklerin dağılımlarını, korelasyonlarını ve kategorik değişkenlerin etkilerini interaktif grafiklerle keşfedebilirsiniz.")

    # Örnek: Yaş dağılımı histogram
    if "Age_at_Release" in df.columns:
        fig = px.histogram(df, x="Age_at_Release", nbins=30, title="Yaş Dağılımı")
        st.plotly_chart(fig, use_container_width=True)

    # Kategorik özelliklerden Gender dağılımı
    if "Gender" in df.columns:
        fig = px.pie(df, names="Gender", title="Cinsiyet Dağılımı")
        st.plotly_chart(fig, use_container_width=True)

    # Korelasyon matrisi
    st.subheader("Sayısal Değişkenler Korelasyon Matrisi")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    corr = df[numeric_cols].corr()
    fig = px.imshow(corr, text_auto=True, title="Korelasyon Matrisi (Sayısal)")
    st.plotly_chart(fig, use_container_width=True)

# --- Model Performans Sayfası ---
def performance_page(df, model, cat_features, feature_names):
    st.title("📈 Model Performansı ve Değerlendirme")

    y_true = df["Recidivism_Within_3years"].astype(int)
    X = df[feature_names].copy()

    # Kategorikleri string'e dönüştür
    for col in cat_features:
        if col in X.columns:
            X[col] = X[col].astype(str)

    pool = Pool(X, cat_features=cat_features)
    y_pred = model.predict(pool)
    y_proba = model.predict_proba(pool)[:, 1] if hasattr(model, "predict_proba") else None

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_proba) if y_proba is not None else None

    st.markdown(f"""
    **Accuracy:** {accuracy:.3f}  
    **Precision:** {precision:.3f}  
    **Recall:** {recall:.3f}  
    **F1 Score:** {f1:.3f}  
    **ROC AUC:** {roc_auc:.3f if roc_auc is not None else 'Modelde olasılık yok'}  
    """)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig = px.imshow(cm, text_auto=True, labels=dict(x="Tahmin", y="Gerçek"), x=[0,1], y=[0,1], title="Confusion Matrix")
    st.plotly_chart(fig, use_container_width=True)

    # ROC curve
    if y_proba is not None:
        fpr, tpr, _ = roc_auc_score(y_true, y_proba, average='macro'), None, None
        RocCurveDisplay.from_estimator(model, X, y_true)
        st.pyplot(plt)

# --- Çoklu Sayfa Navigasyonu ---
def main():
    st.set_page_config(page_title="Cezaevi Risk Tahmin Uygulaması", layout="wide")
    sidebar_info()

    model, cat_features, feature_names, df = load_resources()

    pages = {
        "🏠 Ana Sayfa": lambda: home_page(df),
        "🧠 Tahmin": lambda: prediction_page(model, cat_features, feature_names),
        "📊 Veri Analizi": lambda: analysis_page(df),
        "📈 Model Performansı": lambda: performance_page(df, model, cat_features, feature_names),
    }

    choice = st.sidebar.selectbox("Sayfa Seçimi", options=list(pages.keys()))
    pages[choice]()

if __name__ == "__main__":
    main()
