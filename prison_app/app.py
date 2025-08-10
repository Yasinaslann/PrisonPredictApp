import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from catboost import Pool, CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
import shap
import datetime

st.set_page_config(page_title="Kişisel Suç Tekrarı Tahmin Uygulaması", layout="wide", initial_sidebar_state="expanded")

# ----------------------------------
# --- Helper Fonksiyonlar ---

@st.cache_data(show_spinner=False)
def load_data(path):
    df = pd.read_csv(path)
    # Temizlik & dönüşümler
    df["Age_at_Release"] = pd.to_numeric(df["Age_at_Release"], errors='coerce')
    df.dropna(subset=["Age_at_Release"], inplace=True)

    # Kategorik tiplerin string'e çevrilmesi
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        df[col] = df[col].astype(str).fillna("Unknown")
    return df

@st.cache_resource(show_spinner=False)
def load_model(model_path, cat_features_path, feature_names_path):
    model = pickle.load(open(model_path, "rb"))
    cat_features = pickle.load(open(cat_features_path, "rb"))
    feature_names = pickle.load(open(feature_names_path, "rb"))
    return model, cat_features, feature_names

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def shap_summary_plot(model, X, cat_features):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.initjs()
    fig = shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    return fig

def add_tooltip(label, tooltip):
    return f"{label} ℹ️"

def validate_input(data, cat_features, feature_names):
    # Kategorikleri stringe, numerikleri sayıya zorla
    for feat in feature_names:
        if feat in cat_features:
            data[feat] = str(data[feat])
        else:
            try:
                data[feat] = float(data[feat])
            except Exception:
                data[feat] = np.nan
    return data

# ----------------------------------
# --- Ana Sayfa ---

def home_page(df):
    st.title("🚀 Kişisel Suç Tekrarı Tahmin Projesine Hoş Geldiniz!")
    st.markdown("""
    Bu proje mahkumların suç tekrarını tahmin etmek için tasarlanmış kapsamlı bir makine öğrenimi ve veri analizi uygulamasıdır.

    ### Veri Seti Hakkında
    - Toplam kayıt sayısı: **{}**
    - Özellik sayısı: **{}**
    - Kategorik özellikler: **{}**
    - Sayısal özellikler: **{}**

    ### Projenin Amacı
    - Mahkumların suç tekrar riski tahmin edilerek önleyici tedbirler geliştirmek.
    - Cezaevi yönetimi ve toplum güvenliğini artırmak.
    - Veri bilimi ve yapay zeka teknikleri kullanarak analizler yapmak.

    ### Kullanılan Teknolojiler
    - CatBoost sınıflandırma modeli
    - SHAP ile model açıklanabilirliği
    - Plotly ile zengin görselleştirmeler
    - Streamlit arayüzü

    ---
    Lütfen sol menüden ilgili sayfayı seçerek uygulamayı kullanmaya başlayın.
    """.format(len(df),
               df.shape[1],
               ", ".join(df.select_dtypes(include=['object']).columns.tolist()),
               ", ".join(df.select_dtypes(include=['number']).columns.tolist())))

    st.info("Bu uygulama bir eğitim ve demo projesidir, gerçek dünyadaki durumlarda veri ve model güncellemeleri gerekebilir.")

# ----------------------------------
# --- Tahmin Modülü ---

def prediction_page(model, cat_features, feature_names, df):
    st.title("🧠 Kişisel Suç Tekrarı Tahmin Modülü")
    st.markdown("Lütfen aşağıdaki alanları eksiksiz doldurun. Her alanın yanında açıklamalar bulunmaktadır.")

    input_dict = {}

    # Inputlar
    for feature in feature_names:
        label = f"{feature} ℹ️"
        tooltip = f"{feature} hakkında açıklama."  # İstersen detaylandırılabilir

        if feature in cat_features:
            options = df[feature].dropna().unique().astype(str).tolist()
            input_dict[feature] = st.selectbox(label, options, index=0, help=tooltip)
        else:
            min_val = int(df[feature].min()) if pd.api.types.is_numeric_dtype(df[feature]) else 0
            max_val = int(df[feature].max()) if pd.api.types.is_numeric_dtype(df[feature]) else 100
            default_val = int(df[feature].median()) if pd.api.types.is_numeric_dtype(df[feature]) else 0
            input_dict[feature] = st.number_input(label, min_value=min_val, max_value=max_val, value=default_val, help=tooltip)

    if st.button("Tahmin Yap"):
        validated_input = validate_input(input_dict.copy(), cat_features, feature_names)
        input_df = pd.DataFrame([validated_input])
        for col in cat_features:
            input_df[col] = input_df[col].astype(str)

        pool = Pool(input_df[feature_names], cat_features=cat_features)
        pred = model.predict(pool)[0]
        try:
            pred_proba = model.predict_proba(pool)[0][1]
        except:
            pred_proba = None

        st.success(f"**Tahmin Sonucu:** {'Yüksek Risk' if pred==1 else 'Düşük Risk'}")
        if pred_proba is not None:
            st.info(f"Tahmin Olasılığı: %{pred_proba*100:.2f}")

        # SHAP açıklaması
        with st.expander("Tahmin Açıklaması (SHAP)"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df[feature_names])
            shap.initjs()
            st.pyplot(shap.summary_plot(shap_values, input_df[feature_names], plot_type="bar", show=False))

        # Tahmin geçmişini tut
        if "prediction_history" not in st.session_state:
            st.session_state.prediction_history = []
        st.session_state.prediction_history.append({**validated_input, "prediction": pred, "probability": pred_proba, "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

    # Tahmin geçmişi göster ve indir
    if "prediction_history" in st.session_state and len(st.session_state.prediction_history) > 0:
        st.markdown("---")
        st.subheader("Tahmin Geçmişi")
        history_df = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(history_df)
        csv = convert_df_to_csv(history_df)
        st.download_button("Tahmin Geçmişini CSV Olarak İndir", csv, "prediction_history.csv", "text/csv")

# ----------------------------------
# --- Veri Analizi Sayfası ---

def analysis_page(df):
    st.title("📊 Veri Keşfi ve Gelişmiş Analiz")
    st.markdown("Aşağıdaki filtreleri kullanarak veri setini keşfedin.")

    if df.empty:
        st.warning("Veri seti boş.")
        return

    # Sidebar filtreler
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    num_cols = df.select_dtypes(include=['number']).columns.tolist()

    filters = {}
    for col in cat_cols:
        options = df[col].unique().tolist()
        selected = st.sidebar.multiselect(f"{col} seçin", options, default=options)
        filters[col] = selected

    age_min = int(df["Age_at_Release"].min())
    age_max = int(df["Age_at_Release"].max())
    age_range = st.sidebar.slider("Yaş Aralığı", min_value=age_min, max_value=age_max, value=(age_min, age_max))

    # Filtreleme
    filtered_df = df[
        (df["Age_at_Release"] >= age_range[0]) &
        (df["Age_at_Release"] <= age_range[1])
    ]
    for col in cat_cols:
        filtered_df = filtered_df[filtered_df[col].isin(filters[col])]

    st.write(f"**Filtrelenmiş Veri Sayısı: {len(filtered_df)}**")

    # Grafikler
    st.subheader("Yaş Dağılımı")
    fig_age = px.histogram(filtered_df, x="Age_at_Release", nbins=30, title="Yaş Dağılımı")
    st.plotly_chart(fig_age, use_container_width=True)

    st.subheader("Cinsiyet Dağılımı")
    if "Gender" in filtered_df.columns:
        fig_gender = px.pie(filtered_df, names="Gender", title="Cinsiyet Dağılımı")
        st.plotly_chart(fig_gender, use_container_width=True)

    st.subheader("Korelasyon Matrisi (Sayısal Özellikler)")
    if len(num_cols) > 1:
        corr = filtered_df[num_cols].corr()
        fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="Korelasyon Matrisi")
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Yeterli sayısal veri yok.")

    st.subheader("Outlier Analizi (Boxplot)")
    for col in num_cols:
        fig_box = px.box(filtered_df, y=col, title=f"{col} Boxplot")
        st.plotly_chart(fig_box, use_container_width=True)

    # Yeni sistem: Özellik bazlı detaylı analiz (mean, median, std)
    st.subheader("Sayısal Özelliklerin Detaylı İstatistikleri")
    stats = filtered_df[num_cols].describe().T
    st.dataframe(stats)

# ----------------------------------
# --- Model Performans Sayfası ---

def performance_page(df, model, cat_features, feature_names):
    st.title("📈 Model Performans ve Değerlendirme")

    if df.empty or model is None:
        st.warning("Model veya veri yüklenemedi.")
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
        text=cm,
        texttemplate="%{text}"
    ))
    fig_cm.update_layout(title="Confusion Matrix", xaxis_title="Tahmin", yaxis_title="Gerçek")
    st.plotly_chart(fig_cm, use_container_width=True)

    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Eğrisi'))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash'), name='Rasgele'))
        fig_roc.update_layout(title="ROC Eğrisi", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig_roc, use_container_width=True)

    try:
        fi = model.get_feature_importance(pool=pool, type='FeatureImportance')
        fi_df = pd.DataFrame({"Özellik": feature_names, "Önem": fi}).sort_values(by="Önem", ascending=False).head(20)
        fig_fi = px.bar(fi_df, x="Önem", y="Özellik", orientation='h', title="Özellik Önem Düzeyi")
        st.plotly_chart(fig_fi, use_container_width=True)
    except Exception as e:
        st.warning(f"Özellik önem düzeyi hesaplanamadı: {e}")

# ----------------------------------
# --- Main Fonksiyon ---

def main():
    # Yüklemeler
    df = load_data("prison_app/Prisongüncelveriseti.csv")
    model, cat_features, feature_names = load_model(
        "prison_app/catboost_model.pkl",
        "prison_app/cat_features.pkl",
        "prison_app/feature_names.pkl"
    )

    st.sidebar.title("Navigasyon")
    page = st.sidebar.radio("Sayfa Seçiniz:", ["🏠 Ana Sayfa", "🧠 Tahmin", "📊 Veri Analizi", "📈 Model Performansı"])

    if page == "🏠 Ana Sayfa":
        home_page(df)
    elif page == "🧠 Tahmin":
        prediction_page(model, cat_features, feature_names, df)
    elif page == "📊 Veri Analizi":
        analysis_page(df)
    elif page == "📈 Model Performansı":
        performance_page(df, model, cat_features, feature_names)
    else:
        st.error("Sayfa bulunamadı.")

if __name__ == "__main__":
    main()
