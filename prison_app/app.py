import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, precision_recall_curve, roc_curve, auc
import shap
import joblib
import os
import io
from fpdf import FPDF
import datetime
import logging

# -----------------------------
# CONFIGURATION & LOGGER SETUP
# -----------------------------

st.set_page_config(
    page_title="Hapisten Tahliye Sonrası Suç Tekrarı Tahmin Uygulaması",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')
logger = logging.getLogger()

# -----------------------------
# UTILITY FUNCTIONS
# -----------------------------

@st.cache_data(show_spinner=True)
def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load the encoded recidivism dataset from CSV.
    """
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Dataset loaded with shape {df.shape}")
        return df
    except FileNotFoundError:
        st.error(f"Dosya bulunamadı: {csv_path}. Lütfen dosya yolunu kontrol edin.")
        st.stop()

def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a dataframe with counts of missing values per column.
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    return missing

def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values with sensible defaults or imputations.
    """
    for col in df.columns:
        if df[col].dtype == 'O':
            df[col].fillna('Unknown', inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)
    return df

def encode_categoricals(df: pd.DataFrame, cat_features: list) -> pd.DataFrame:
    """
    One-hot encode or label encode categorical features as needed.
    """
    for col in cat_features:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df

def plot_missing_values(df: pd.DataFrame):
    """
    Visualize missing values by feature.
    """
    missing = check_missing_values(df)
    if len(missing) == 0:
        st.success("Veri setinde eksik değer bulunmamaktadır.")
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
        missing.plot(kind='bar', ax=ax, color='tomato')
        ax.set_title("Eksik Değerlerin Özelliklere Göre Sayısı")
        ax.set_ylabel("Eksik Değer Sayısı")
        ax.set_xlabel("Özellikler")
        st.pyplot(fig)

def prepare_model(model_path: str):
    """
    Load the trained model from a .pkl file.
    """
    if not os.path.exists(model_path):
        st.error(f"Model dosyası bulunamadı: {model_path}")
        st.stop()
    model = joblib.load(model_path)
    logger.info(f"Model {model_path} yüklenmiştir.")
    return model

def calculate_model_metrics(model, X, y):
    """
    Calculate and return various classification metrics.
    """
    y_pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)[:, 1]
    else:
        y_proba = model.predict(X)  # fallback for models without predict_proba

    roc_auc = roc_auc_score(y, y_proba)
    conf_mat = confusion_matrix(y, y_pred)
    class_report = classification_report(y, y_pred, output_dict=True)

    precision, recall, thresholds = precision_recall_curve(y, y_proba)
    pr_auc = auc(recall, precision)

    return {
        "roc_auc": roc_auc,
        "confusion_matrix": conf_mat,
        "classification_report": class_report,
        "precision": precision,
        "recall": recall,
        "pr_auc": pr_auc,
        "y_pred": y_pred,
        "y_proba": y_proba
    }

def plot_roc_curve(y_true, y_proba):
    """
    Plot ROC curve using matplotlib.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    ax.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0,1.0])
    ax.set_ylim([0.0,1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc="lower right")
    st.pyplot(fig)

def plot_precision_recall_curve_func(precision, recall):
    """
    Plot Precision-Recall curve using matplotlib.
    """
    fig, ax = plt.subplots()
    ax.plot(recall, precision, color='blue', lw=2)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    st.pyplot(fig)

def plot_confusion_matrix(conf_mat):
    """
    Plot confusion matrix heatmap using seaborn.
    """
    fig, ax = plt.subplots()
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Tahmin Edilen")
    ax.set_ylabel("Gerçek")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

def shap_summary_plot(model, X):
    """
    Compute and plot SHAP summary plot.
    """
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        st.subheader("Özelliklerin Model Üzerindeki SHAP Etkisi")
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        st.pyplot(bbox_inches='tight')
    except Exception as e:
        st.warning("SHAP grafik gösterimi sırasında hata oluştu.")
        logger.error(f"SHAP plot error: {e}")

def save_prediction_history(predictions: list, filename: str = "prediction_history.csv"):
    """
    Save prediction history list as CSV.
    """
    df_hist = pd.DataFrame(predictions)
    df_hist.to_csv(filename, index=False)
    st.success(f"Tahmin geçmişi '{filename}' olarak kaydedildi.")

def download_csv(df: pd.DataFrame, filename: str, button_label: str):
    """
    Streamlit button to download DataFrame as CSV.
    """
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=button_label,
        data=csv_data,
        file_name=filename,
        mime='text/csv',
    )

def generate_pdf_report(prediction: dict, model_metrics: dict, filename="Recidivism_Prediction_Report.pdf"):
    """
    Create a PDF report summarizing prediction results and model performance.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Hapisten Tahliye Sonrası Suç Tekrarı Tahmin Raporu", 0, 1, 'C')
    pdf.ln(10)

    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Tahmin Zamanı: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1)
    pdf.cell(0, 10, "Tahmin Sonuçları:", 0, 1)
    for key, val in prediction.items():
        pdf.cell(0, 10, f"{key}: {val}", 0, 1)

    pdf.ln(10)
    pdf.cell(0, 10, "Model Performans Özellikleri:", 0, 1)
    for metric, score in model_metrics.items():
        if isinstance(score, (float, int)):
            pdf.cell(0, 10, f"{metric}: {score:.4f}", 0, 1)
    pdf.output(filename)
    st.success(f"PDF raporu '{filename}' olarak oluşturuldu.")

# -----------------------------------
# APP PAGES
# -----------------------------------

def page_home():
    st.title("🔒 Hapisten Tahliye Sonrası Suç Tekrarı Tahmin Uygulaması")
    st.markdown("""
    Bu uygulama, hapisten tahliye olan kişilerin suç tekrarı yapma olasılığını tahmin etmeye yönelik geliştirilmiştir. 
    Veri keşfi, model tahmini ve analiz bölümleri ile kapsamlı bir sistem sunar.
    """)
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e8/Jail_icon.svg/1200px-Jail_icon.svg.png", width=300)
    st.markdown("### Navigasyon menüsünden istediğiniz bölüme geçiş yapabilirsiniz.")

def page_data_overview(df: pd.DataFrame):
    st.header("📊 Veri Keşfi ve Detaylı Analiz")
    st.markdown("Veri setinizin özelliklerini keşfedebilir, eksik değerleri ve dağılımları inceleyebilirsiniz.")
    
    st.subheader("Veri Genel Bilgileri")
    st.write(df.info())
    st.write("### İlk 5 Satır")
    st.dataframe(df.head())

    st.subheader("Eksik Değer Analizi")
    plot_missing_values(df)

    st.subheader("Sayısal Değişkenlerin Dağılımı")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    col_select = st.multiselect("Görselleştirilecek sayısal değişkenleri seçin:", num_cols, default=num_cols[:3])

    for col in col_select:
        fig = px.histogram(df, x=col, nbins=30, title=f"{col} Dağılımı")
        st.plotly_chart(fig)

    st.subheader("Kategorik Değişkenlerin Dağılımı")
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    cat_select = st.multiselect("Görselleştirilecek kategorik değişkenleri seçin:", cat_cols, default=cat_cols[:3])

    for col in cat_select:
        fig = px.bar(df[col].value_counts(), title=f"{col} Frekans Grafiği")
        st.plotly_chart(fig)

def page_prediction(df: pd.DataFrame, model_catboost, model_rf, cat_features: list, feature_names: list):
    st.header("🧠 Kişisel Suç Tekrarı Tahmin Modülü")
    st.markdown("Aşağıdaki alanları doldurarak tahmin yapabilirsiniz. Her alanın yanında açıklamalar mevcuttur.")

    user_input = {}
    for feature in feature_names:
        if feature in cat_features:
            unique_vals = df[feature].dropna().unique().tolist()
            user_input[feature] = st.selectbox(f"{feature} seçiniz:", unique_vals)
        else:
            min_val = int(df[feature].min())
            max_val = int(df[feature].max())
            user_input[feature] = st.slider(f"{feature} değerini seçiniz:", min_val, max_val, int(df[feature].median()))

    # Input DataFrame for model prediction
    input_df = pd.DataFrame({k: [v] for k, v in user_input.items()})

    # Prediction button
    if st.button("Tahmin Yap"):
        st.info("Tahmin işlemi başlatılıyor...")

        # Preprocessing input_df if needed
        for col in cat_features:
            input_df[col] = input_df[col].astype(str)

        # Prediction with CatBoost
        pred_cb = model_catboost.predict(input_df)[0]
        pred_prob_cb = model_catboost.predict_proba(input_df)[0][1]

        # Prediction with RandomForest
        pred_rf = model_rf.predict(input_df)[0]
        pred_prob_rf = model_rf.predict_proba(input_df)[0][1]

        st.subheader("📊 Tahmin Sonuçları")
        st.markdown(f"**CatBoost Modeli:** Tahmini Suç Tekrarı: {'Evet' if pred_cb == 1 else 'Hayır'} (Olasılık: {pred_prob_cb:.3f})")
        st.markdown(f"**RandomForest Modeli:** Tahmini Suç Tekrarı: {'Evet' if pred_rf == 1 else 'Hayır'} (Olasılık: {pred_prob_rf:.3f})")

        # Kaydetme / geçmişe ekleme
        if "pred_history" not in st.session_state:
            st.session_state.pred_history = []
        st.session_state.pred_history.append({
            **user_input,
            "Pred_CatBoost": pred_cb,
            "Pred_Prob_CatBoost": pred_prob_cb,
            "Pred_RandomForest": pred_rf,
            "Pred_Prob_RandomForest": pred_prob_rf,
            "Timestamp": datetime.datetime.now()
        })

        # Göster tahmin geçmişi tablosu
        if st.checkbox("Tahmin Geçmişini Göster"):
            hist_df = pd.DataFrame(st.session_state.pred_history)
            st.dataframe(hist_df)
            download_csv(hist_df, "tahmin_gecmisi.csv", "Tahmin Geçmişini CSV Olarak İndir")

        # PDF Raporu oluşturma
        if st.button("PDF Raporu Oluştur"):
            latest_pred = st.session_state.pred_history[-1]
            metrics_cb = calculate_model_metrics(model_catboost, df[feature_names], df["Recidivism"])
            generate_pdf_report(latest_pred, metrics_cb)

def page_model_performance(df: pd.DataFrame, model, feature_names: list):
    st.header("📈 Model Performans ve Değerlendirme")

    X = df[feature_names]
    y = df["Recidivism"]

    metrics = calculate_model_metrics(model, X, y)

    st.subheader("ROC AUC Skoru")
    st.write(f"{metrics['roc_auc']:.4f}")
    plot_roc_curve(y, metrics['y_proba'])

    st.subheader("Precision-Recall Eğrisi")
    plot_precision_recall_curve_func(metrics["precision"], metrics["recall"])
    st.write(f"PR AUC Skoru: {metrics['pr_auc']:.4f}")

    st.subheader("Confusion Matrix (Karışıklık Matrisi)")
    plot_confusion_matrix(metrics["confusion_matrix"])

    st.subheader("Detaylı Sınıflandırma Raporu")
    st.text(pd.DataFrame(metrics["classification_report"]).transpose())

    st.subheader("Özellik Önem Düzeyi (SHAP)")
    shap_summary_plot(model, X)

def page_about():
    st.header("📚 Proje Hakkında")
    st.markdown("""
    - Bu proje, hapisten tahliye sonrası suç tekrarını tahmin etmek için geliştirilmiştir.
    - Gelişmiş makine öğrenimi modelleri kullanılarak detaylı analiz yapılmaktadır.
    - Kullanıcı dostu arayüz ile veri keşfi, tahmin ve model performansı takip edilebilir.
    - Proje kapsamlı, açık kaynaklı ve yatırım almaya adaydır.
    - Github: https://github.com/Yasinaslann/PrisonPredictApp
    """)

# ----------------------------------------
# MAIN APP FUNCTION
# ----------------------------------------

def main():
    # Sidebar navigasyon
    st.sidebar.title("🔒 Menü")
    page = st.sidebar.radio(
        "Sayfa Seçiniz:",
        ["Ana Sayfa", "Veri Keşfi", "Tahmin", "Model Performansı", "Proje Hakkında"]
    )

    # Veri ve modelleri yükle
    df = load_data("NIJ_s_Recidivism_Encod_Update.csv")
    cat_features = [ "Gender", "Race", "Education_Level", "Dependents", "Prison_Offense", "Residence_Changes"]
    feature_names = [col for col in df.columns if col != "Recidivism"]

    model_catboost = prepare_model("catboost_model.pkl")
    model_rf = prepare_model("randomforest_model.pkl")  # İstersen bu model senin için önceden eğitilmiş olmalı

    if page == "Ana Sayfa":
        page_home()
    elif page == "Veri Keşfi":
        page_data_overview(df)
    elif page == "Tahmin":
        page_prediction(df, model_catboost, model_rf, cat_features, feature_names)
    elif page == "Model Performansı":
        page_model_performance(df, model_catboost, feature_names)
    elif page == "Proje Hakkında":
        page_about()

if __name__ == "__main__":
    main()
