import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from catboost import Pool
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
import shap
import base64
import datetime

# --------------------------
# Helpers

def load_model_and_data():
    try:
        model = pickle.load(open("prison_app/catboost_model.pkl", "rb"))
        cat_features = pickle.load(open("prison_app/cat_features.pkl", "rb"))
        feature_names = pickle.load(open("prison_app/feature_names.pkl", "rb"))
        df = pd.read_csv("prison_app/Prisongüncelveriseti.csv")

        # Data cleaning & conversions
        df["Age_at_Release"] = pd.to_numeric(df["Age_at_Release"], errors='coerce')
        df = df.dropna(subset=["Age_at_Release"])

        for col in cat_features:
            if col in df.columns:
                df[col] = df[col].astype(str).fillna("Unknown")

        return model, cat_features, feature_names, df
    except Exception as e:
        st.error(f"Yükleme hatası: {e}")
        return None, None, None, None


def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def add_tooltip(label, tooltip_text):
    return f"{label} ℹ️"

def get_shap_explanation(model, X_sample, cat_features):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    return shap_values

# --------------------------
# Ana Sayfa

def home_page(df):
    st.title("🚀 Kişisel Suç Tekrarı Tahmin Uygulaması")
    st.markdown("""
    ### Proje Hakkında
    Bu uygulama, mahkumların suç tekrarını tahmin etmek için geliştirilmiş gelişmiş bir makine öğrenimi sistemidir. 
    Veri seti, mahkumların demografik ve cezai geçmiş bilgilerini içerir.

    ---
    ### Veri Seti Genel Bilgiler:
    - Kayıt Sayısı: **{}**
    - Özellik Sayısı: **{}**
    - Kategorik Özellikler: **{}**
    - Sayısal Özellikler: **{}**

    ---
    ### Projenin Amacı
    - Mahkumların suç tekrar riskini önceden tahmin etmek.
    - Suç önleme stratejilerinde veri destekli kararlar almak.
    - Cezaevi yönetimi ve toplumsal güvenliği artırmak.

    ---
    ### Kullanılan Teknolojiler
    - CatBoost Makine Öğrenimi Modeli
    - SHAP ile Tahmin Açıklamaları
    - Plotly ile Gelişmiş Veri Görselleştirme
    - Streamlit ile Hızlı Web Uygulaması
    """.format(len(df), df.shape[1],
               ", ".join(df.select_dtypes(include=['object']).columns),
               ", ".join(df.select_dtypes(include=['number']).columns)))

    st.markdown("---")
    st.markdown("Projenin detaylarına ve kullanıma geçmek için sol menüden ilgili bölümü seçiniz.")

# --------------------------
# Tahmin Modülü

def prediction_page(model, cat_features, feature_names, df):
    st.title("🧠 Kişisel Suç Tekrarı Tahmin Modülü")
    st.markdown("Lütfen aşağıdaki alanları doldurun. Her alanın yanında kısa açıklamalar bulunmaktadır.")

    input_data = {}

    # Dinamik form oluşturma, tooltip ve default değerlerle
    for feature in feature_names:
        tooltip_text = f"{feature} özelliğinin açıklaması burada yer alabilir."  # Burayı detaylandırabilirsiniz
        label = f"{feature} ℹ️"
        
        if feature in cat_features:
            options = df[feature].dropna().astype(str).unique().tolist()
            input_data[feature] = st.selectbox(label, options, index=0, help=tooltip_text)
        else:
            min_val = int(df[feature].min()) if pd.api.types.is_numeric_dtype(df[feature]) else 0
            max_val = int(df[feature].max()) if pd.api.types.is_numeric_dtype(df[feature]) else 100
            default_val = int(df[feature].median()) if pd.api.types.is_numeric_dtype(df[feature]) else 0
            input_data[feature] = st.number_input(label, min_value=min_val, max_value=max_val, value=default_val, help=tooltip_text)

    # Tahmin butonu
    if st.button("Tahmin Yap"):
        # DataFrame oluştur
        input_df = pd.DataFrame([input_data])
        for col in cat_features:
            input_df[col] = input_df[col].astype(str)
        
        pool = Pool(input_df[feature_names], cat_features=cat_features)

        prediction = model.predict(pool)[0]
        prediction_proba = model.predict_proba(pool)[0][1] if hasattr(model, "predict_proba") else None

        st.success(f"Suç tekrar riski tahmini: **{'Yüksek' if prediction == 1 else 'Düşük'}**")
        if prediction_proba is not None:
            st.info(f"Tahmin olasılığı: %{prediction_proba * 100:.2f}")

        # SHAP ile açıklama
        with st.expander("Tahmin Açıklaması (SHAP)"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df[feature_names])
            shap.initjs()
            st_shap = st.pyplot(shap.summary_plot(shap_values, input_df[feature_names], plot_type="bar", show=False))

        # Tahmin geçmişi tutma ve CSV indirme
        if "prediction_history" not in st.session_state:
            st.session_state.prediction_history = []
        st.session_state.prediction_history.append({
            "timestamp": datetime.datetime.now(),
            **input_data,
            "prediction": prediction,
            "probability": prediction_proba
        })

        # Göster tahmin geçmişi
        if st.checkbox("Tahmin Geçmişini Göster"):
            history_df = pd.DataFrame(st.session_state.prediction_history)
            st.dataframe(history_df)
            csv = convert_df_to_csv(history_df)
            st.download_button("Tahmin Geçmişini CSV İndir", csv, "prediction_history.csv", "text/csv")

# --------------------------
# Veri Analizi Sayfası

def analysis_page(df):
    st.title("📊 Veri Keşfi ve Gelişmiş Analiz")
    st.markdown("Filtreler ve etkileşimli grafiklerle veri setini detaylıca keşfedin.")

    if df is None or df.empty:
        st.warning("Veri yüklenemedi veya boş.")
        return

    # Filtreler
    genders = df["Gender"].dropna().astype(str).unique().tolist()
    selected_genders = st.sidebar.multiselect("Cinsiyet Seçiniz", options=genders, default=genders)

    age_min = int(df["Age_at_Release"].min())
    age_max = int(df["Age_at_Release"].max())
    selected_age = st.sidebar.slider("Yaş Aralığı", min_value=age_min, max_value=age_max, value=(age_min, age_max))

    filtered_df = df[
        (df["Gender"].astype(str).isin(selected_genders)) &
        (df["Age_at_Release"] >= selected_age[0]) &
        (df["Age_at_Release"] <= selected_age[1])
    ]

    st.subheader("Yaş Dağılımı")
    fig1 = px.histogram(filtered_df, x="Age_at_Release", nbins=30)
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Cinsiyet Dağılımı")
    fig2 = px.pie(filtered_df, names="Gender")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Korelasyon Matrisi")
    numeric_cols = filtered_df.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols) >= 2:
        corr_matrix = filtered_df[numeric_cols].corr()
        fig3 = px.imshow(corr_matrix, text_auto=True)
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Yeterli sayısal veri yok korelasyon matrisi için.")

    # Yeni: Outlier analizi boxplot
    st.subheader("Outlier Analizi (Boxplot)")
    for col in numeric_cols:
        fig_box = px.box(filtered_df, y=col, title=f"{col} Boxplot")
        st.plotly_chart(fig_box, use_container_width=True)

# --------------------------
# Model Performans Sayfası

def performance_page(df, model, cat_features, feature_names):
    st.title("📈 Model Performans ve Değerlendirme")

    if df is None or df.empty or model is None:
        st.warning("Veri veya model yüklenemedi.")
        return

    y_true = df["Recidivism_Within_3years"].astype(int)
    X = df[feature_names].copy()

    for col in cat_features:
        if col in X.columns:
            X[col] = X[col].astype(str).fillna("Unknown")

    pool = Pool(X, cat_features=cat_features)

    y_pred = model.predict(pool)
    try:
        y_proba = model.predict_proba(pool)[:, 1]
    except Exception:
        y_proba = None

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_proba) if y_proba is not None else None

    st.markdown(f"""
    | Metrik    | Değer | Açıklama |
    |-----------|-------|----------|
    | Accuracy  | {accuracy:.3f} | Doğru tahmin oranı |
    | Precision | {precision:.3f} | Pozitif tahminlerin doğruluğu |
    | Recall    | {recall:.3f} | Gerçek pozitiflerin yakalanma oranı |
    | F1 Score  | {f1:.3f} | Precision ve Recall dengesi |
    | ROC AUC   | {roc_auc:.3f if roc_auc is not None else 'Yok'} | Modelin ayırıcı gücü |
    """)

    cm = confusion_matrix(y_true, y_pred)
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Düşük Risk", "Yüksek Risk"],
        y=["Düşük Risk", "Yüksek Risk"],
        colorscale='Blues',
        showscale=True,
        text=cm,
        texttemplate="%{text}"
    ))
    fig_cm.update_layout(title="Confusion Matrix", xaxis_title="Tahmin", yaxis_title="Gerçek")
    st.plotly_chart(fig_cm, use_container_width=True)

    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve'))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash'), name='Rasgele'))
        fig_roc.update_layout(title="ROC Eğrisi", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig_roc, use_container_width=True)

    try:
        fi = model.get_feature_importance(pool=pool, type='FeatureImportance')
        fi_df = pd.DataFrame({"Özellik": feature_names, "Önem": fi})
        fi_df = fi_df.sort_values(by="Önem", ascending=False).head(15)
        fig_fi = px.bar(fi_df, x="Önem", y="Özellik", orientation='h', title="Özellik Önem Düzeyi")
        st.plotly_chart(fig_fi, use_container_width=True)
    except Exception as e:
        st.warning(f"Özellik önem düzeyi hesaplanamadı: {e}")

# --------------------------
# Main app

def main():
    st.set_page_config(page_title="Kişisel Suç Tekrarı Tahmin Uygulaması", layout="wide")

    model, cat_features, feature_names, df = load_model_and_data()

    st.sidebar.title("Navigasyon")
    page = st.sidebar.radio("Sayfa Seçiniz", ["🏠 Ana Sayfa", "🧠 Tahmin", "📊 Veri Analizi", "📈 Model Performansı"])

    if page == "🏠 Ana Sayfa":
        home_page(df)
    elif page == "🧠 Tahmin":
        prediction_page(model, cat_features, feature_names, df)
    elif page == "📊 Veri Analizi":
        analysis_page(df)
    elif page == "📈 Model Performansı":
        performance_page(df, model, cat_features, feature_names)
    else:
        st.write("Sayfa bulunamadı.")

if __name__ == "__main__":
    main()
