import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from catboost import Pool
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, auc

st.set_page_config(
    page_title="Cezaevi Risk Tahmin Uygulaması",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dosya yollarını kendi ortamına göre değiştir
MODEL_PATH = "prison_app/catboost_model.pkl"
CAT_FEATURES_PATH = "prison_app/cat_features.pkl"
FEATURE_NAMES_PATH = "prison_app/feature_names.pkl"
DATA_PATH = "prison_app/Prisongüncelveriseti.csv"

@st.cache_data(show_spinner=True)
def load_model_and_data():
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(CAT_FEATURES_PATH, "rb") as f:
            cat_features = pickle.load(f)
        with open(FEATURE_NAMES_PATH, "rb") as f:
            feature_names = pickle.load(f)
        df = pd.read_csv(DATA_PATH)

        # Temizlik: Age_at_Release sütununda null varsa doldur ya da çıkar
        df = df.dropna(subset=["Age_at_Release"])
        # Gerekirse diğer null'ları da doldurabilir veya çıkarabilirsin.

        return model, cat_features, feature_names, df
    except Exception as e:
        st.error(f"Veri veya model yükleme hatası: {e}")
        return None, None, None, None

def preprocess_input(df_input, cat_features):
    # Kategorik sütunlar string olmalı
    for col in cat_features:
        if col in df_input.columns:
            df_input[col] = df_input[col].astype(str)
    return df_input

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def sidebar_info():
    st.sidebar.title("🚀 Cezaevi Risk Tahmin Uygulaması")
    st.sidebar.markdown("""
    ### Navigasyon
    - 🏠 Ana Sayfa: Proje & Veri seti tanıtımı  
    - 🧠 Tahmin: Kişisel suç tekrar riski tahmini  
    - 📊 Veri Analizi: Veri setini keşfet & görselleştir  
    - 📈 Model Performansı: Model metrikleri & grafikler  
    ---
    ⚠️ Model CatBoost tabanlıdır, kategorik veriler string formatında işlenir.
    """)

# --- ANA SAYFA ---
def home_page(df):
    st.title("🚀 Cezaevi Tekrar Suç Riski Tahmin Projesi")
    st.markdown("""
    Bu proje, cezaevinden çıktıktan sonra kişilerin tekrar suç işleme olasılığını tahmin etmek üzere geliştirilmiştir.  
    Model, kişinin demografik ve sosyoekonomik özelliklerini analiz ederek risk skorunu verir.  
    """)

    if df is not None and not df.empty:
        st.subheader("Veri Seti Hakkında Genel Bilgiler")
        st.write(f"- Kayıt sayısı: **{len(df)}**")
        st.write(f"- Özellik sayısı: **{len(df.columns)}**")
        st.write("- Hedef değişken: **Recidivism_Within_3years** (3 yıl içinde tekrar suç)")

        st.subheader("Örnek Veri")
        st.dataframe(df.head(10))

        st.subheader("Yaş Dağılımı ve Tekrar Suç Oranı")
        fig = px.histogram(df, x="Age_at_Release", nbins=30, color="Recidivism_Within_3years",
                           labels={"Age_at_Release": "Yaş", "Recidivism_Within_3years": "Tekrar Suç Durumu"})
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Cinsiyet Dağılımı")
        fig2 = px.pie(df, names="Gender", title="Cinsiyet Dağılımı")
        st.plotly_chart(fig2, use_container_width=True)

        if "Education_Level" in df.columns:
            st.subheader("Eğitim Seviyesi Dağılımı ve Tekrar Suç Durumu")
            fig3 = px.histogram(df, x="Education_Level", color="Recidivism_Within_3years", barmode='group')
            st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning("Veri yüklenemedi veya boş.")

# --- TAHMİN SAYFASI ---
def prediction_page(model, cat_features, feature_names, df):
    st.title("🧠 Kişisel Suç Tekrarı Tahmin Modülü")
    st.info("Lütfen tüm alanları doldurun. Alanların yanında açıklamalar ve ipuçları bulunmaktadır.")

    if model is None or df is None:
        st.error("Model ya da veri yüklenmediği için tahmin yapılamıyor.")
        return

    if "predictions" not in st.session_state:
        st.session_state["predictions"] = []

    input_data = {}

    # Özellik açıklamaları
    feature_help = {
        "Age_at_Release": "Cezaevinden çıkış yapılan yaş.",
        "Gender": "Kişinin cinsiyeti (erkek/kadın).",
        "Race": "Kişinin etnik kökeni.",
        "Education_Level": "Kişinin eğitim durumu.",
        "Supervision_Risk_Score_First": "Denetim risk puanı; yüksek puan, daha yüksek risk anlamına gelir."
    }

    # Girdi alanları oluştur
    for feature in feature_names:
        if feature == "ID":  # ID almıyoruz
            continue
        help_text = feature_help.get(feature, "Bu özellik hakkında bilgi bulunmamaktadır.")
        try:
            if feature in cat_features:
                options = df[feature].dropna().astype(str).unique().tolist()
                if not options:
                    st.warning(f"{feature} için seçim yapacak veri yok.")
                    continue
                default_index = 0
                val = st.selectbox(f"{feature} ❓", options=options, index=default_index, help=help_text)
                input_data[feature] = val
            else:
                min_val = int(df[feature].dropna().min())
                max_val = int(df[feature].dropna().max())
                median_val = int(df[feature].dropna().median())
                val = st.number_input(f"{feature} ❓", min_value=min_val, max_value=max_val, value=median_val, step=1, help=help_text)
                input_data[feature] = val
        except Exception as e:
            st.error(f"Girdi alanı oluşturulurken hata: {e}")
            return

    df_input = pd.DataFrame([input_data])
    df_input = preprocess_input(df_input, cat_features)

    if st.button("🔮 Tahmini Yap"):
        try:
            pool = Pool(df_input, cat_features=cat_features)
            pred = model.predict(pool)[0]
            pred_proba = model.predict_proba(pool)[0][1] if hasattr(model, "predict_proba") else None

            risk_label = "Yüksek Risk" if pred == 1 else "Düşük Risk"
            st.success(f"### Tahmin Sonucu: {risk_label}")

            if pred_proba is not None:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=pred_proba * 100,
                    title={'text': "Risk Skoru (%)"},
                    gauge={'axis': {'range': [0, 100]},
                           'bar': {'color': "red" if pred_proba > 0.5 else "green"},
                           'steps': [
                               {'range': [0, 50], 'color': "lightgreen"},
                               {'range': [50, 100], 'color': "lightcoral"}]}
                ))
                st.plotly_chart(fig_gauge)

            st.subheader("Tahmin Açıklaması (Özelliklerin Etkisi)")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(pool)
            shap.initjs()
            fig, ax = plt.subplots(figsize=(10, 4))
            shap.summary_plot(shap_values, df_input, plot_type="bar", show=False, max_display=10)
            st.pyplot(fig)

            # Tahmin geçmişine ekle
            st.session_state["predictions"].append({
                **input_data,
                "Tahmin": risk_label,
                "Risk Skoru (%)": round(pred_proba * 100, 2) if pred_proba is not None else None
            })

        except Exception as e:
            st.error(f"Tahmin sırasında hata oluştu: {e}")

    if st.session_state["predictions"]:
        st.subheader("🔎 Tahmin Geçmişi")
        df_pred = pd.DataFrame(st.session_state["predictions"])
        st.dataframe(df_pred)

        csv = convert_df_to_csv(df_pred)
        st.download_button("Tahmin Geçmişini CSV Olarak İndir", data=csv, file_name="tahmin_gecmisi.csv", mime="text/csv")

# --- VERİ ANALİZİ ---
def analysis_page(df):
    st.title("📊 Veri Keşfi ve Gelişmiş Analiz")

    if df is None or df.empty:
        st.warning("Veri yüklenemedi veya boş.")
        return

    st.markdown("Veri setinizi filtreleyip etkileşimli grafiklerle detaylıca keşfedebilirsiniz.")

    # Filtreler
    genders = df["Gender"].dropna().unique().tolist()
    selected_genders = st.sidebar.multiselect("Cinsiyet Seçiniz", options=genders, default=genders)

    try:
        age_min = int(df["Age_at_Release"].dropna().min())
        age_max = int(df["Age_at_Release"].dropna().max())
    except:
        age_min, age_max = 18, 100  # varsayılan

    selected_age = st.sidebar.slider("Yaş Aralığı", min_value=age_min, max_value=age_max, value=(age_min, age_max))

    filtered_df = df[
        (df["Gender"].isin(selected_genders)) &
        (df["Age_at_Release"] >= selected_age[0]) &
        (df["Age_at_Release"] <= selected_age[1])
    ]

    st.subheader("Yaş Dağılımı")
    fig1 = px.histogram(filtered_df, x="Age_at_Release", nbins=30, title="Yaş Dağılımı")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Cinsiyet Dağılımı")
    fig2 = px.pie(filtered_df, names="Gender", title="Cinsiyet Dağılımı")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Korelasyon Matrisi (Sayısal Değişkenler)")
    numeric_cols = filtered_df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        corr_matrix = filtered_df[numeric_cols].corr()
        fig3 = px.imshow(corr_matrix, text_auto=True, title="Korelasyon Matrisi")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Sayısal veri bulunamadı korelasyon matrisi için.")

    if "Education_Level" in filtered_df.columns:
        st.subheader("Eğitim Seviyesi ve Tekrar Suç Durumu")
        fig4 = px.histogram(filtered_df, x="Education_Level", color="Recidivism_Within_3years", barmode='group')
        st.plotly_chart(fig4, use_container_width=True)

    if "Supervision_Risk_Score_First" in filtered_df.columns:
        st.subheader("Yaş ve Risk Skoru İlişkisi")
        fig5 = px.scatter(filtered_df, x="Age_at_Release", y="Supervision_Risk_Score_First",
                          color="Recidivism_Within_3years",
                          labels={"Age_at_Release": "Yaş", "Supervision_Risk_Score_First": "Risk Skoru"})
        st.plotly_chart(fig5, use_container_width=True)

# --- MODEL PERFORMANSI ---
def performance_page(df, model, cat_features, feature_names):
    st.title("📈 Model Performans ve Değerlendirme")

    if df is None or df.empty:
        st.warning("Veri yüklenemediği için performans gösterilemiyor.")
        return

    if model is None:
        st.warning("Model yüklenemediği için performans gösterilemiyor.")
        return

    y_true = df["Recidivism_Within_3years"].astype(int)
    X = df[feature_names].copy()

    # Kategorik sütunlar string formatında olmalı
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

    # Metrikler Tablosu
    roc_auc_str = f"{roc_auc:.3f}" if roc_auc is not None else "Yok"
    st.markdown(f"""
    | Metrik      | Değer | Açıklama |
    |-------------|-------|----------|
    | Accuracy    | {accuracy:.3f} | Doğru tahmin oranı |
    | Precision   | {precision:.3f} | Pozitif tahminlerin doğruluğu |
    | Recall      | {recall:.3f} | Gerçek pozitiflerin yakalanma oranı |
    | F1 Score    | {f1:.3f} | Precision ve Recall dengesi |
    | ROC AUC     | {roc_auc_str} | Modelin ayırıcı gücü |
    """)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="blues",
                       labels=dict(x="Tahmin", y="Gerçek"),
                       x=["Düşük Risk", "Yüksek Risk"], y=["Düşük Risk", "Yüksek Risk"],
                       title="Confusion Matrix")
    st.plotly_chart(fig_cm, use_container_width=True)

    # ROC Curve
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve'))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                    line=dict(dash='dash'), name='Random Guess'))
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

    # Feature Importance
    st.subheader("Model Özellik Önem Düzeyi")
    fi = model.get_feature_importance(pool=pool, type='FeatureImportance')
    fi_df = pd.DataFrame({"Feature": feature_names, "Importance": fi})
    fi_df = fi_df.sort_values(by="Importance", ascending=False)
    fig_fi = px.bar(fi_df.head(15), x="Importance", y="Feature", orientation='h',
                    title="En Önemli 15 Özellik")
    st.plotly_chart(fig_fi, use_container_width=True)

# --- ANA FONKSİYON ---
def main():
    model, cat_features, feature_names, df = load_model_and_data()
    sidebar_info()

    pages = {
        "🏠 Ana Sayfa": lambda: home_page(df),
        "🧠 Tahmin": lambda: prediction_page(model, cat_features, feature_names, df),
        "📊 Veri Analizi": lambda: analysis_page(df),
        "📈 Model Performansı": lambda: performance_page(df, model, cat_features, feature_names),
    }

    st.sidebar.title("Sayfalar")
    choice = st.sidebar.radio("Gitmek istediğiniz sayfayı seçin:", list(pages.keys()))
    pages[choice]()

if __name__ == "__main__":
    main()
