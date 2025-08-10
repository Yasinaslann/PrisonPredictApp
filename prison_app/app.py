import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from catboost import Pool
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, RocCurveDisplay, precision_recall_curve, auc
)
import base64
import io

st.set_page_config(page_title="Cezaevi Risk Tahmin Uygulaması", layout="wide", initial_sidebar_state="expanded")

# --- Dosya yolları (kendi yapına göre güncelle) ---
MODEL_PATH = "prison_app/catboost_model.pkl"
CAT_FEATURES_PATH = "prison_app/cat_features.pkl"
FEATURE_NAMES_PATH = "prison_app/feature_names.pkl"
DATA_PATH = "prison_app/Prisongüncelveriseti.csv"

@st.cache_data(show_spinner=True)
def load_model_and_data():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(CAT_FEATURES_PATH, "rb") as f:
        cat_features = pickle.load(f)
    with open(FEATURE_NAMES_PATH, "rb") as f:
        feature_names = pickle.load(f)
    df = pd.read_csv(DATA_PATH)
    return model, cat_features, feature_names, df

def preprocess_input(df_input, cat_features):
    for col in cat_features:
        if col in df_input.columns:
            df_input[col] = df_input[col].astype(str)
    return df_input

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def sidebar_info():
    st.sidebar.title("Cezaevi Risk Tahmin Uygulaması")
    st.sidebar.markdown("""
    ### Navigasyon
    - Ana Sayfa: Proje ve veri seti tanıtımı  
    - Tahmin: Kişiye özel suç risk tahmini  
    - Veri Analizi: Dataset görselleme ve keşif  
    - Model Performansı: Modelin detaylı değerlendirmesi
    """)
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ⚠️ **Not:** Uygulama, CatBoost tabanlı bir model kullanmaktadır.  
    Tüm kategorik değişkenler string tipine çevrilmiştir.  
    Kullanıcı girdilerinde açıklamalar ve doğrulamalar mevcuttur.  
    """)

# --- ANA SAYFA ---
def home_page(df):
    st.title("🚀 Cezaevi Tekrar Suç Riski Tahmin Projesi")
    st.markdown("""
    ### Proje Amacı  
    Cezaevinden çıkış yapan bireylerin tekrar suç işleme riskini tahmin ederek,  
    adalet sistemi ve sosyal destek mekanizmalarının daha etkili çalışmasını sağlamaktır.

    ### Veri Seti Hakkında  
    Kullanılan veri seti çeşitli kişisel, sosyoekonomik ve suç geçmişi bilgilerini içerir.  
    Toplam kayıt sayısı: **{}**  
    Özellik sayısı: **{}**  
    Hedef Değişken: **Recidivism_Within_3years** (3 yıl içinde tekrar suç)

    ### Önemli Değişkenler  
    - **Age_at_Release:** Serbest bırakılma yaşı  
    - **Gender:** Cinsiyet  
    - **Race:** Etnik köken  
    - **Education_Level:** Eğitim durumu  
    - **Supervision_Risk_Score_First:** Denetim risk puanı  
    """.format(len(df), len(df.columns)))

    st.subheader("Veri Setinden Örnek Satırlar")
    st.dataframe(df.head(10))

    st.subheader("Yaş Dağılımı")
    fig = px.histogram(df, x="Age_at_Release", nbins=30, color="Recidivism_Within_3years",
                       labels={"Age_at_Release": "Yaş"}, 
                       title="Yaş Dağılımı ve Risk Durumu")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cinsiyet Dağılımı")
    fig2 = px.pie(df, names="Gender", title="Cinsiyet Dağılımı")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Suç İşleme Riskine Göre Eğitim Seviyesi Dağılımı")
    fig3 = px.histogram(df, x="Education_Level", color="Recidivism_Within_3years",
                        title="Eğitim Seviyesi ve Risk İlişkisi")
    st.plotly_chart(fig3, use_container_width=True)

# --- TAHMİN SAYFASI ---
def prediction_page(model, cat_features, feature_names):
    st.title("🧠 Kişisel Suç Tekrarı Tahmin Modülü")

    st.info("Lütfen aşağıdaki alanları eksiksiz doldurun. Her alanın yanında açıklamalar bulunmaktadır.")

    # Kullanıcı girdileri için session state listesi (Tahmin geçmişi için)
    if "predictions" not in st.session_state:
        st.session_state["predictions"] = []

    input_data = {}

    # Örnek tooltip açıklamalar
    feature_help = {
        "Age_at_Release": "Cezaevinden çıkış yapılan yaş.",
        "Gender": "Kişinin cinsiyeti.",
        "Race": "Kişinin etnik kökeni.",
        "Education_Level": "Kişinin eğitim durumu.",
        "Supervision_Risk_Score_First": "Denetim risk puanı, ne kadar yüksekse risk o kadar fazladır."
    }

    # Dinamik input oluşturma
    for feature in feature_names:
        if feature == "ID":  # ID alınmayacak
            continue

        tooltip = feature_help.get(feature, "Bu özellik hakkında bilgi bulunmamaktadır.")

        if feature in cat_features:
            # Unique değerleri datasetten al (string olarak)
            unique_vals = df[feature].dropna().astype(str).unique().tolist()
            default_index = 0
            val = st.selectbox(f"{feature} ❓", options=unique_vals, index=default_index, help=tooltip)
            input_data[feature] = val
        else:
            # Sayısal değer için min max belirle datasetten
            min_val = int(df[feature].min())
            max_val = int(df[feature].max())
            default_val = int(df[feature].median())
            val = st.number_input(f"{feature} ❓", min_value=min_val, max_value=max_val, value=default_val, step=1, help=tooltip)
            input_data[feature] = val

    df_input = pd.DataFrame([input_data])
    df_input = preprocess_input(df_input, cat_features)

    # Tahmin butonu
    if st.button("🔮 Tahmini Yap"):
        try:
            pool = Pool(df_input, cat_features=cat_features)
            prediction = model.predict(pool)[0]
            prediction_proba = model.predict_proba(pool)[0][1] if hasattr(model, "predict_proba") else None

            risk_label = "Yüksek Risk" if prediction == 1 else "Düşük Risk"
            st.success(f"### Tahmin Sonucu: {risk_label}")

            if prediction_proba is not None:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=prediction_proba * 100,
                    title={'text': "Risk Skoru (%)"},
                    gauge={'axis': {'range': [0, 100]},
                           'bar': {'color': "red" if prediction_proba > 0.5 else "green"},
                           'steps': [
                               {'range': [0, 50], 'color': "lightgreen"},
                               {'range': [50, 100], 'color': "lightcoral"}]}))
                st.plotly_chart(fig_gauge)

            # SHAP açıklaması
            st.subheader("Tahmin Açıklaması: Model Özellik Etkileri")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(pool)
            shap.initjs()
            fig, ax = plt.subplots(figsize=(10, 4))
            shap.summary_plot(shap_values, df_input, plot_type="bar", show=False, max_display=10)
            st.pyplot(fig)

            # Tahmin geçmişine kaydet
            st.session_state["predictions"].append({
                "input": input_data,
                "prediction": risk_label,
                "score": prediction_proba
            })

        except Exception as e:
            st.error(f"Tahmin sırasında hata oluştu: {e}")

    # Tahmin geçmişi tablosu ve indir butonu
    if st.session_state["predictions"]:
        st.subheader("🔎 Tahmin Geçmişi")
        df_pred_history = pd.DataFrame([{
            **pred["input"],
            "Tahmin": pred["prediction"],
            "Risk Skoru": round(pred["score"] * 100, 2) if pred["score"] is not None else None
        } for pred in st.session_state["predictions"]])
        st.dataframe(df_pred_history)

        csv_data = convert_df_to_csv(df_pred_history)
        st.download_button(label="Tahmin Geçmişini CSV Olarak İndir", data=csv_data, file_name="tahmin_gecmisi.csv", mime="text/csv")

# --- VERİ ANALİZİ ---
def analysis_page(df):
    st.title("📊 Veri Keşfi ve Gelişmiş Analiz")

    st.markdown("Filtreleyerek ve interaktif grafiklerle veri setini detaylıca keşfedebilirsiniz.")

    # Filtreleme sidebar
    st.sidebar.subheader("Filtreler")
    unique_genders = df["Gender"].dropna().unique().tolist()
    selected_genders = st.sidebar.multiselect("Cinsiyet", options=unique_genders, default=unique_genders)

    age_min = int(df["Age_at_Release"].min())
    age_max = int(df["Age_at_Release"].max())
    selected_age = st.sidebar.slider("Yaş Aralığı", min_value=age_min, max_value=age_max, value=(age_min, age_max))

    # Filtre uygulama
    df_filtered = df[
        (df["Gender"].isin(selected_genders)) &
        (df["Age_at_Release"] >= selected_age[0]) &
        (df["Age_at_Release"] <= selected_age[1])
    ]

    # Yaş dağılımı
    fig1 = px.histogram(df_filtered, x="Age_at_Release", nbins=30, title="Yaş Dağılımı")
    st.plotly_chart(fig1, use_container_width=True)

    # Cinsiyet dağılımı
    fig2 = px.pie(df_filtered, names="Gender", title="Seçilen Cinsiyetlere Göre Dağılım")
    st.plotly_chart(fig2, use_container_width=True)

    # Korelasyon matrisi (sayısal değişkenler)
    numeric_cols = df_filtered.select_dtypes(include=np.number).columns.tolist()
    corr = df_filtered[numeric_cols].corr()
    fig3 = px.imshow(corr, text_auto=True, title="Sayısal Değişkenler Korelasyon Matrisi")
    st.plotly_chart(fig3, use_container_width=True)

    # Kategorik dağılımlar - Education_Level örneği
    if "Education_Level" in df.columns:
        fig4 = px.histogram(df_filtered, x="Education_Level", color="Recidivism_Within_3years", barmode='group',
                            title="Eğitim Seviyesi ve Tekrar Suç Durumu")
        st.plotly_chart(fig4, use_container_width=True)

    # İleri düzey: Scatter plot Risk Skoru vs Yaş
    if "Supervision_Risk_Score_First" in df.columns:
        fig5 = px.scatter(df_filtered, x="Age_at_Release", y="Supervision_Risk_Score_First",
                          color="Recidivism_Within_3years", title="Yaş ve Risk Skoru İlişkisi",
                          labels={"Age_at_Release":"Yaş", "Supervision_Risk_Score_First":"Risk Skoru"})
        st.plotly_chart(fig5, use_container_width=True)

# --- MODEL PERFORMANS ---
def performance_page(df, model, cat_features, feature_names):
    st.title("📈 Model Performans ve Değerlendirme")

    y_true = df["Recidivism_Within_3years"].astype(int)
    X = df[feature_names].copy()
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

    # Performans metrikleri açıklamalı
    st.markdown(f"""
    ### Temel Performans Metrikleri

    | Metrik      | Değer | Açıklama |
    |-------------|-------|----------|
    | Accuracy    | {accuracy:.3f} | Modelin doğru tahmin oranı |
    | Precision   | {precision:.3f} | Pozitif tahminlerin doğruluğu |
    | Recall      | {recall:.3f} | Gerçek pozitiflerin yakalanma oranı |
    | F1 Score    | {f1:.3f} | Precision ve Recall dengesi |
    | ROC AUC     | {roc_auc:.3f if roc_auc is not None else 'Yok'} | Modelin ayırıcı gücü |
    """)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="blues",
                       labels=dict(x="Tahmin", y="Gerçek"),
                       x=["Düşük Risk", "Yüksek Risk"], y=["Düşük Risk", "Yüksek Risk"],
                       title="Confusion Matrix")
    st.plotly_chart(fig_cm, use_container_width=True)

    # ROC Curve
    if y_proba is not None:
        fig_roc = go.Figure()
        fpr, tpr, _ = RocCurveDisplay.from_predictions(y_true, y_proba, name="ROC Curve", ax=None)
        # Plotly için elle ROC curve çiziyoruz
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve'))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random Guess', line=dict(dash='dash')))
        fig_roc.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig_roc, use_container_width=True)

    # Precision-Recall Curve
    if y_proba is not None:
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = auc(recall_vals, precision_vals)
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(x=recall_vals, y=precision_vals, mode='lines', name='Precision-Recall Curve'))
        fig_pr.update_layout(title=f"Precision-Recall Curve (AUC={pr_auc:.3f})",
                             xaxis_title="Recall", yaxis_title="Precision")
        st.plotly_chart(fig_pr, use_container_width=True)

    # Feature importance
    st.subheader("Model Özellik Önem Düzeyi")
    feature_importances = model.get_feature_importance(pool=pool, type='FeatureImportance')
    fi_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": feature_importances
    }).sort_values(by="Importance", ascending=False).head(15)

    fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation='h', title="En Önemli 15 Özellik")
    st.plotly_chart(fig_fi, use_container_width=True)

# --- MAIN ---
def main():
    sidebar_info()
    try:
        model, cat_features, feature_names, df = load_model_and_data()
    except Exception as e:
        st.error(f"Model veya veri yüklenirken hata oluştu: {e}")
        return

    pages = {
        "🏠 Ana Sayfa": lambda: home_page(df),
        "🧠 Tahmin": lambda: prediction_page(model, cat_features, feature_names),
        "📊 Veri Analizi": lambda: analysis_page(df),
        "📈 Model Performansı": lambda: performance_page(df, model, cat_features, feature_names),
    }
    choice = st.sidebar.selectbox("Sayfa Seçiniz", list(pages.keys()))
    pages[choice]()

if __name__ == "__main__":
    main()
