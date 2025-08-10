import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sqlite3
import datetime
import shap
import plotly.express as px
import plotly.graph_objects as go
from catboost import Pool, CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
from fpdf import FPDF
import io
import base64

st.set_page_config(page_title="Kişisel Suç Tekrarı Tahmin Uygulaması - İleri Seviye", layout="wide",
                   initial_sidebar_state="expanded")

# --- Veritabanı Bağlantısı ve Tablo Oluşturma ---
conn = sqlite3.connect("prediction_history.db", check_same_thread=False)
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    input_data TEXT,
    prediction INTEGER,
    probability REAL
)
""")
conn.commit()

# --- Yardımcı Fonksiyonlar ---

@st.cache_data(show_spinner=False)
def load_data(path="prison_app/Prisongüncelveriseti.csv"):
    df = pd.read_csv(path)
    # Temizlik ve tip dönüşümleri
    df["Age_at_Release"] = pd.to_numeric(df["Age_at_Release"], errors='coerce')
    df.dropna(subset=["Age_at_Release"], inplace=True)
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        df[col] = df[col].astype(str).fillna("Unknown")
    return df

@st.cache_resource(show_spinner=False)
def load_model_and_metadata(model_path="prison_app/catboost_model.pkl",
                            cat_features_path="prison_app/cat_features.pkl",
                            feature_names_path="prison_app/feature_names.pkl"):
    model = pickle.load(open(model_path, "rb"))
    cat_features = pickle.load(open(cat_features_path, "rb"))
    feature_names = pickle.load(open(feature_names_path, "rb"))
    return model, cat_features, feature_names

def save_prediction(input_dict, prediction, prob):
    c.execute("INSERT INTO predictions (timestamp, input_data, prediction, probability) VALUES (?, ?, ?, ?)",
              (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), str(input_dict), prediction, prob))
    conn.commit()

def get_prediction_history_df():
    c.execute("SELECT id, timestamp, input_data, prediction, probability FROM predictions ORDER BY id DESC")
    rows = c.fetchall()
    data = []
    for row in rows:
        data.append({
            "ID": row[0],
            "Tarih": row[1],
            "Girdi": row[2],
            "Tahmin": "Yüksek Risk" if row[3] == 1 else "Düşük Risk",
            "Olasılık": f"{row[4]:.2%}" if row[4] is not None else "N/A"
        })
    return pd.DataFrame(data)

def generate_pdf_report(df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(0, 10, txt="Tahmin Geçmişi Raporu", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", size=11)
    for _, row in df.iterrows():
        pdf.multi_cell(0, 8,
                       f"ID: {row['ID']}\nTarih: {row['Tarih']}\nTahmin: {row['Tahmin']}\nOlasılık: {row['Olasılık']}\n{'-'*40}")
    return pdf.output(dest='S').encode('latin-1')

def validate_input(data, cat_features, feature_names):
    validated = {}
    for feat in feature_names:
        val = data.get(feat)
        if feat in cat_features:
            validated[feat] = str(val)
        else:
            try:
                validated[feat] = float(val)
            except:
                validated[feat] = np.nan
    return validated

def plot_confusion_matrix(cm, labels):
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        hoverongaps=False,
        colorscale='Blues'
    ))
    fig.update_layout(title="Confusion Matrix", xaxis_title="Tahmin Edilen", yaxis_title="Gerçek")
    return fig

def plot_roc_curve(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Guess', line=dict(dash='dash')))
    fig.update_layout(title='ROC Eğrisi', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
    return fig

def plot_precision_recall_curve(y_true, y_score):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name='Precision-Recall Curve'))
    fig.update_layout(title='Precision-Recall Eğrisi', xaxis_title='Recall', yaxis_title='Precision')
    return fig

def get_shap_values(model, input_df, cat_features):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)
    return shap_values

def highlight_feature_effects(shap_values, input_df):
    base_value = shap_values.base_values[0] if hasattr(shap_values, "base_values") else 0
    impacts = {}
    for feature, val, sv in zip(input_df.columns, input_df.iloc[0], shap_values.values[0]):
        impacts[feature] = sv
    return impacts

# --- Sayfa Fonksiyonları ---

def home_page(df):
    st.title("🚀 Kişisel Suç Tekrarı Tahmin Projesi - İleri Seviye")
    st.markdown("""
    Bu proje, cezaevinden salıverilen kişilerin yeniden suç işleme riskini tahmin etmek üzere geliştirilmiştir.
    Projede CatBoost modeli kullanılmış, SHAP ile yorumlanabilirlik sağlanmıştır.
    """)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Veri Seti Genel Bakış")
        st.write(f"Toplam kayıt: {len(df)}")
        st.write(f"Özellik sayısı: {df.shape[1]}")
        st.write("Kategorik özellikler:")
        st.write(", ".join(df.select_dtypes(include='object').columns.tolist()))
        st.write("Sayısal özellikler:")
        st.write(", ".join(df.select_dtypes(include='number').columns.tolist()))

    with col2:
        fig = px.histogram(df, x="Age_at_Release", nbins=30, title="Yaş Dağılımı")
        st.plotly_chart(fig, use_container_width=True)
        corr = df.corr()
        fig2 = px.imshow(corr, title="Korelasyon Matrisi", color_continuous_scale='RdBu_r')
        st.plotly_chart(fig2, use_container_width=True)

def prediction_page(model, cat_features, feature_names, df):
    st.title("🧠 Tahmin Modülü")
    st.markdown("Her bir özellik için yanında açıklamalar vardır. Mouse ile üstlerine gelerek detayları görebilirsiniz.")

    input_data = {}
    for feature in feature_names:
        help_text = "Bu özellik modelde önemli rol oynar."  # Daha detaylı açıklamalar eklenebilir
        if feature in cat_features:
            options = df[feature].dropna().unique().astype(str).tolist()
            input_data[feature] = st.selectbox(f"{feature} ❓", options, help=help_text)
        else:
            min_val, max_val = int(df[feature].min()), int(df[feature].max())
            median_val = int(df[feature].median())
            input_data[feature] = st.number_input(f"{feature} ❓", min_value=min_val, max_value=max_val, value=median_val, help=help_text)

    if st.button("Tahmin Yap"):
        validated = validate_input(input_data, cat_features, feature_names)
        input_df = pd.DataFrame([validated])
        for col in cat_features:
            input_df[col] = input_df[col].astype(str)

        try:
            pool = Pool(input_df[feature_names], cat_features=cat_features)
            prediction = model.predict(pool)[0]
            proba = model.predict_proba(pool)[0][1]
        except Exception as e:
            st.error(f"Tahmin sırasında hata oluştu: {e}")
            return

        st.success(f"Sonuç: {'Yüksek Risk' if prediction == 1 else 'Düşük Risk'}")
        st.info(f"Risk Olasılığı: %{proba * 100:.2f}")

        save_prediction(validated, prediction, proba)

        with st.expander("Model Açıklaması (SHAP Değerleri)"):
            shap_values = get_shap_values(model, input_df, cat_features)
            impacts = highlight_feature_effects(shap_values, input_df)
            df_impacts = pd.DataFrame.from_dict(impacts, orient='index', columns=['SHAP Değeri']).sort_values(by='SHAP Değeri', ascending=False)
            st.dataframe(df_impacts)
            # SHAP summary plot
            st.set_option('deprecation.showPyplotGlobalUse', False)
            shap.initjs()
            fig_shap = shap.plots.bar(shap_values[0], max_display=10, show=False)
            st.pyplot(bbox_inches='tight')

def analysis_page(df):
    st.title("📊 Veri Keşfi ve Gelişmiş Analiz")
    st.markdown("Filtreleyerek verinizi detaylıca keşfedebilir, grafiklerle anlayabilirsiniz.")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()

    with st.sidebar:
        st.subheader("Filtreler")
        filters = {}
        for col in numeric_cols:
            min_val, max_val = float(df[col].min()), float(df[col].max())
            filters[col] = st.slider(f"{col} Aralığı", min_val, max_val, (min_val, max_val))
        for col in cat_cols:
            options = df[col].unique().tolist()
            filters[col] = st.multiselect(f"{col} Seçimi", options, default=options)

    filtered_df = df.copy()
    for col in numeric_cols:
        filtered_df = filtered_df[(filtered_df[col] >= filters[col][0]) & (filtered_df[col] <= filters[col][1])]
    for col in cat_cols:
        filtered_df = filtered_df[filtered_df[col].isin(filters[col])]

    st.write(f"Filtrelenmiş veri sayısı: {len(filtered_df)}")

    st.subheader("Kategorik Değişkenlerin Dağılımı")
    for col in cat_cols:
        fig = px.histogram(filtered_df, x=col, color=col, title=f"{col} Dağılımı")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sayısal Değişkenlerin Dağılımı")
    for col in numeric_cols:
        fig = px.box(filtered_df, y=col, points="all", title=f"{col} Box Plot")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Korelasyon Matrisi")
    corr = filtered_df.corr()
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", title="Korelasyon Matrisi")
    st.plotly_chart(fig, use_container_width=True)

def performance_page(df, model, cat_features, feature_names):
    st.title("📈 Model Performans ve Değerlendirme")

    X = df[feature_names]
    y = df["Recidivism"]

    # Veri tip uyumu
    for col in cat_features:
        X[col] = X[col].astype(str)

    pool = Pool(X, label=y, cat_features=cat_features)

    y_pred = model.predict(pool)
    y_proba = model.predict_proba(pool)[:,1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_proba) if y_proba is not None else None

    st.markdown(f"""
    | Metrik | Değer | Açıklama |
    |---|---|---|
    | Doğruluk (Accuracy) | {acc:.3f} | Doğru tahmin oranı |
    | Kesinlik (Precision) | {prec:.3f} | Pozitif tahminlerin doğruluğu |
    | Duyarlılık (Recall) | {rec:.3f} | Gerçek pozitiflerin yakalanma oranı |
    | F1 Skoru | {f1:.3f} | Precision ve Recall dengesi |
    | ROC AUC | {roc_auc:.3f if roc_auc else 'Yok'} | Modelin ayırıcı gücü |
    """)

    cm = confusion_matrix(y, y_pred)
    st.plotly_chart(plot_confusion_matrix(cm, labels=["Düşük Risk", "Yüksek Risk"]))

    if y_proba is not None:
        st.plotly_chart(plot_roc_curve(y, y_proba))
        st.plotly_chart(plot_precision_recall_curve(y, y_proba))

def history_page():
    st.title("📜 Tahmin Geçmişi")
    df_hist = get_prediction_history_df()
    if df_hist.empty:
        st.info("Tahmin geçmişiniz bulunmamaktadır.")
        return

    st.dataframe(df_hist, use_container_width=True)

    if st.button("CSV Olarak İndir"):
        csv = df_hist.to_csv(index=False).encode('utf-8')
        st.download_button(label="CSV İndir", data=csv, file_name="tahmin_gecmisi.csv", mime="text/csv")

    if st.button("PDF Olarak İndir"):
        pdf_bytes = generate_pdf_report(df_hist)
        st.download_button(label="PDF İndir", data=pdf_bytes, file_name="tahmin_gecmisi.pdf", mime="application/pdf")

# --- Ana Fonksiyon ---

def main():
    df = load_data()
    model, cat_features, feature_names = load_model_and_metadata()

    st.sidebar.title("Navigasyon")
    choice = st.sidebar.radio("Sayfa Seçin", ["🏠 Ana Sayfa", "🧠 Tahmin", "📊 Veri Keşfi", "📈 Model Performans", "📜 Tahmin Geçmişi"])

    if choice == "🏠 Ana Sayfa":
        home_page(df)
    elif choice == "🧠 Tahmin":
        prediction_page(model, cat_features, feature_names, df)
    elif choice == "📊 Veri Keşfi":
        analysis_page(df)
    elif choice == "📈 Model Performans":
        performance_page(df, model, cat_features, feature_names)
    elif choice == "📜 Tahmin Geçmişi":
        history_page()
    else:
        st.error("Bilinmeyen sayfa seçildi.")

if __name__ == "__main__":
    main()
