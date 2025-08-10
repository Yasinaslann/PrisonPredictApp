# app.py
"""
Prison Risk Prediction — Streamlit Uygulaması
- GitHub raw üzerinden dosya çekme (varsa local yerine kullanır)
- Multi-page: Anasayfa, Tahmin, Tavsiye, Veri Analizi & Model, Geçmiş
- SHAP explainability (plotly bar + waterfall fallback)
- Tahmin geçmişi, CSV/PDF indirme
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
import pickle
import joblib
from io import BytesIO
from datetime import datetime
from fpdf import FPDF
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# ---------------------------
# Ayarlar
# ---------------------------
st.set_page_config(page_title="Prison Risk App", layout="wide", initial_sidebar_state="expanded")

GITHUB_RAW_BASE = "https://raw.githubusercontent.com/Yasinaslann/PrisonPredictApp/main/prison_app/"
EXPECTED_FILES = {
    "dataset": "Prisongüncelveriseti.csv",
    "model": "catboost_model.pkl",
    "feature_names": "feature_names.pkl",
    "cat_features": "cat_features.pkl",
    "cat_unique_values": "cat_unique_values.pkl",
    "bool_columns": "bool_columns.pkl"
}

LOCAL_DOWNLOAD_DIR = os.getcwd()  # local dizine indirir

# ---------------------------
# Yardımcı Fonksiyonlar
# ---------------------------
def download_if_missing(filename, force=False):
    """
    Eğer localde yoksa (veya force True ise) GitHub raw'tan indirip local olarak kaydeder.
    Döndürür: (ok: bool, message: str)
    """
    local_path = os.path.join(LOCAL_DOWNLOAD_DIR, filename)
    if (not force) and os.path.exists(local_path):
        return True, f"Zaten var: {local_path}"
    url = GITHUB_RAW_BASE + filename
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            with open(local_path, "wb") as f:
                f.write(resp.content)
            return True, f"İndirildi: {local_path}"
        else:
            return False, f"HTTP {resp.status_code} - {url}"
    except Exception as e:
        return False, f"İndirme hatası: {e}"

@st.cache_data
def load_csv_local(path):
    return pd.read_csv(path)

@st.cache_data
def load_pickle_local(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def safe_load_model(path):
    """
    Modeli güvenle yüklemeye çalışır: joblib veya pickle ile.
    """
    try:
        # joblib önce dene
        try:
            m = joblib.load(path)
            return m
        except Exception:
            with open(path, "rb") as f:
                m = pickle.load(f)
            return m
    except Exception as e:
        raise RuntimeError(f"Model yüklenemedi: {e}")

def create_pdf(record: dict, recommendation: str):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "Prediction Report", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 8, f"Tarih: {record.get('timestamp', '')}", ln=True)
    pdf.ln(4)
    pdf.cell(0, 8, "Girdi Değerleri:", ln=True)
    for k, v in record.items():
        if k in ("timestamp", "pred", "proba"): 
            continue
        pdf.multi_cell(0, 6, f"- {k}: {v}")
    pdf.ln(4)
    pdf.cell(0, 8, f"Pred: {record.get('pred', '')} | Proba: {record.get('proba', '')}", ln=True)
    pdf.ln(6)
    pdf.cell(0, 8, "Öneriler:", ln=True)
    pdf.multi_cell(0, 6, recommendation)
    return pdf.output(dest="S").encode("latin-1", errors="replace")

def generate_recommendation(score, inputs=None):
    """
    Basit rule-based öneri motoru. İhtiyaca göre genişlet.
    """
    if score is None:
        return "Skor yok — öneri üretilemiyor."
    s = float(score)
    tips = []
    if s >= 0.85:
        tips.append("Yüksek risk: Acil müdahale, yoğun rehabilitasyon ve sosyal destek önerilir.")
    elif s >= 0.6:
        tips.append("Orta-yüksek risk: Hedefli eğitim ve yakından takip faydalı olabilir.")
    elif s >= 0.35:
        tips.append("Orta risk: İzleme ve mesleki eğitim önerilir.")
    else:
        tips.append("Düşük risk: Standart programlarla izleme uygundur.")
    # Özellik bazlı ek öneriler (örnek)
    if isinstance(inputs, dict):
        edu = str(inputs.get("education_level", "")).lower()
        if edu in ("low", "none", "ilkokul", "ortaokul"):
            tips.append("Eğitim desteği: Temel/mesleki eğitim programlarına dahil et.")
        emp = str(inputs.get("employment_status", "")).lower()
        if emp in ("unemployed", "işsiz", "yok"):
            tips.append("İstihdam desteği: İş eğitimi ve işe yerleştirme programları öner.")
    return "\n\n".join(tips)

def safe_predict(model, X_df):
    """
    Modelden güvenli şekilde tahmin/proba almaya çalışır.
    """
    pred = None
    proba = None
    try:
        pred = model.predict(X_df)
        if hasattr(pred, "__len__"):
            pred = int(pred[0])
        else:
            pred = int(pred)
    except Exception:
        pred = None
    try:
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(X_df)
            # eğer iki sınıflı ise proba[:,1], değilse ravel()[0]
            if p.ndim == 2 and p.shape[1] >= 2:
                proba = float(p[:,1][0])
            else:
                proba = float(np.ravel(p)[0])
        else:
            proba = None
    except Exception:
        proba = None
    return pred, proba

# ---------------------------
# Dosyaları hazırla / indir (varsa)
# ---------------------------
st.sidebar.header("Dosya Kontrol & İndirme")
if st.sidebar.button("GitHub'dan eksikleri indir (veya güncelle)"):
    results = {}
    for fn in EXPECTED_FILES.values():
        ok, msg = download_if_missing(fn, force=True)
        results[fn] = (ok, msg)
    for fn, (ok, msg) in results.items():
        if ok:
            st.sidebar.success(f"{fn}: {msg}")
        else:
            st.sidebar.error(f"{fn}: {msg}")

# localde varsa ya da indirildiyse yüklemeye çalış
df = pd.DataFrame()
model = None
feature_names = None
cat_features = []
cat_unique_values = {}
bool_columns = []

# dataset
if os.path.exists(EXPECTED_FILES["dataset"]):
    try:
        df = load_csv_local(EXPECTED_FILES["dataset"])
    except Exception as e:
        st.sidebar.error(f"CSV yükleme hatası: {e}")

# model
if os.path.exists(EXPECTED_FILES["model"]):
    try:
        model = safe_load_model(EXPECTED_FILES["model"])
    except Exception as e:
        st.sidebar.error(f"Model yükleme hatası: {e}")

# diğer pkl'ler
for k in ("feature_names", "cat_features", "cat_unique_values", "bool_columns"):
    fn = EXPECTED_FILES[k]
    if os.path.exists(fn):
        try:
            val = load_pickle_local(fn)
            if k == "feature_names":
                feature_names = val
            elif k == "cat_features":
                cat_features = val
            elif k == "cat_unique_values":
                cat_unique_values = val
            elif k == "bool_columns":
                bool_columns = val
        except Exception as e:
            st.sidebar.error(f"{fn} okunamadı: {e}")

# fallback: feature_names yoksa veri üzerinden çıkar
if feature_names is None or not isinstance(feature_names, (list, tuple)):
    if not df.empty:
        # hedef var mı kontrol et
        possible_targets = [c for c in df.columns if c.lower() in ("target", "label", "y", "recidivism", "risk")]
        if possible_targets:
            feature_names = [c for c in df.columns if c not in possible_targets]
        else:
            feature_names = df.columns.tolist()
    else:
        feature_names = []

# ---------------------------
# Session state
# ---------------------------
if "pred_history" not in st.session_state:
    st.session_state.pred_history = []

# ---------------------------
# Sayfalar
# ---------------------------
def page_home():
    st.title("Prison Risk App — Anasayfa")
    st.markdown("""
    **Proje amacı:** Mahkum/denetimli kişilerin yeniden suça yönelme riskini tahmin ederek
    müdahale ve rehberlik sağlayacak öneriler üretmek.
    """)
    st.markdown("### Yüklü dosyalar kontrolü")
    for k, fn in EXPECTED_FILES.items():
        st.write(f"- {fn}: {'✔' if os.path.exists(fn) else '❌'}")
    st.markdown("---")
    st.subheader("Veri Önizleme")
    if df.empty:
        st.warning("Veri yüklenemedi. Dosya yoksa sidebar'dan 'GitHub'dan eksikleri indir' düğmesini kullan.")
    else:
        st.dataframe(df.head(10))
        st.markdown(f"**Satır / Sütun:** {df.shape[0]} x {df.shape[1]}")

def build_input_ui(features):
    """
    feature listesine göre dinamik input alanları oluşturur.
    """
    inputs = {}
    if not features:
        st.error("Feature listesi bulunamadı.")
        return inputs
    cols = st.columns(2)
    i = 0
    for feat in features:
        c = cols[i % 2]
        if feat in cat_features:
            opts = cat_unique_values.get(feat, [])
            if opts:
                val = c.selectbox(f"{feat}", opts, key=f"inp_{feat}")
            else:
                val = c.text_input(f"{feat}", key=f"inp_{feat}")
        elif feat in bool_columns:
            val = c.selectbox(f"{feat}", [0,1], key=f"inp_{feat}")
        else:
            default = 0.0
            if (not df.empty) and (feat in df.columns) and np.issubdtype(df[feat].dtype, np.number):
                default = float(df[feat].median())
            val = c.number_input(f"{feat}", value=default, key=f"inp_{feat}")
        inputs[feat] = val
        i += 1
    return inputs

def page_predict():
    st.title("Tahmin Sistemi")
    if not feature_names:
        st.error("feature_names yüklenemedi; tahmin formu oluşturulamıyor.")
        return
    st.markdown("Girdi değerlerini girin ve 'Tahmin Et' düğmesine basın.")
    with st.form("predict_form"):
        user_inputs = build_input_ui(feature_names)
        submitted = st.form_submit_button("Tahmin Et")
    if submitted:
        if model is None:
            st.error("Model yüklenemedi. `catboost_model.pkl` eksik veya bozuk.")
            return
        X = pd.DataFrame([user_inputs], columns=feature_names)
        try:
            pred, proba = safe_predict(model, X)
        except Exception as e:
            st.error(f"Tahmin sırasında hata: {e}")
            pred, proba = None, None
        st.metric("Tahmin (label)", pred if pred is not None else "—")
        if proba is not None:
            st.metric("Risk Skoru (0-1)", f"{proba:.4f}")
        # kayıt
        rec = {"timestamp": datetime.now().isoformat(timespec="seconds"), **user_inputs, "pred": pred, "proba": proba}
        st.session_state.pred_history.insert(0, rec)
        # SHAP açıklaması
        st.subheader("Tahmin Açıklaması (SHAP)")
        try:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X)
            # shap_vals tipi: array veya list
            if isinstance(shap_vals, list) and len(shap_vals) > 1:
                arr = np.array(shap_vals[1])[0]  # sınıf 1 için
                expected_value = explainer.expected_value[1] if hasattr(explainer.expected_value, "__len__") else explainer.expected_value
            else:
                arr = np.array(shap_vals)[0]
                expected_value = explainer.expected_value if not hasattr(explainer.expected_value, "__len__") else explainer.expected_value
            # Plotly bar (mutlak SHAP değerleri)
            df_sh = pd.DataFrame({"feature": X.columns, "shap_value": arr})
            df_sh["abs_shap"] = df_sh["shap_value"].abs()
            df_sh = df_sh.sort_values("abs_shap", ascending=False).head(20)
            fig = px.bar(df_sh[::-1], x="shap_value", y="feature", orientation="h", title="En etkili feature'lar (SHAP, top 20)")
            st.plotly_chart(fig, use_container_width=True)
            # Waterfall (matplotlib fallback)
            try:
                st.subheader("Waterfall (detaylı)")
                fig_w, ax = plt.subplots(figsize=(8, 4))
                # waterfall_legacy kullanımı
                shap.plots._waterfall.waterfall_legacy(expected_value, arr, feature_names=X.columns, max_display=12)
                st.pyplot(fig_w)
            except Exception:
                st.info("Waterfall çizimi yapılamadı; bar grafik gösterildi.")
        except Exception as e:
            st.error(f"SHAP açıklaması oluşturulamadı: {e}")
        # öneri
        st.subheader("Kişisel Öneri")
        rec_text = generate_recommendation(proba, user_inputs)
        st.info(rec_text)
        # indirme
        col1, col2 = st.columns(2)
        with col1:
            csv_bytes = pd.DataFrame([rec]).to_csv(index=False).encode("utf-8")
            st.download_button("Bu tahmini CSV indir", data=csv_bytes, file_name="prediction_single.csv", mime="text/csv")
        with col2:
            pdf_bytes = create_pdf(rec, rec_text)
            st.download_button("Bu tahmini PDF indir", data=pdf_bytes, file_name="prediction_single.pdf", mime="application/pdf")

def page_recommendation():
    st.title("Tavsiye Sistemi")
    if len(st.session_state.pred_history) == 0:
        st.info("Henüz tahmin yapılmadı. Tahmin sayfasından başlayın.")
        return
    df_hist = pd.DataFrame(st.session_state.pred_history)
    sel = st.selectbox("Geçmiş kayıtlardan seçin", options=df_hist.index, format_func=lambda i: f"{df_hist.loc[i,'timestamp']} — {df_hist.loc[i,'proba']}")
    row = df_hist.loc[sel].to_dict()
    st.subheader("Seçili Kayıt")
    st.json(row)
    st.subheader("Öneriler")
    rec_text = generate_recommendation(row.get("proba"), row)
    st.info(rec_text)
    if st.button("Seçili kaydı PDF indir"):
        pdf_bytes = create_pdf(row, rec_text)
        st.download_button("PDF İndir", data=pdf_bytes, file_name="recommendation.pdf", mime="application/pdf")

def page_data_analysis():
    st.title("Veri Analizi & Model Performansı")
    if df.empty:
        st.warning("Veri yüklenemedi.")
        return
    st.markdown(f"Dataset: {df.shape[0]} satır x {df.shape[1]} sütun")
    # filtreler (sidebar)
    st.sidebar.subheader("Analiz Filtreleri")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    filters = {}
    for c in numeric_cols[:6]:
        mn, mx = float(df[c].min()), float(df[c].max())
        lo_hi = st.sidebar.slider(f"{c} aralığı", mn, mx, (mn, mx))
        filters[c] = lo_hi
    for c in cat_cols[:4]:
        opts = df[c].unique().tolist()
        sel = st.sidebar.multiselect(f"{c} seç", options=opts, default=opts[:min(5, len(opts))])
        filters[c] = sel
    dff = df.copy()
    for c, v in filters.items():
        if isinstance(v, tuple):
            dff = dff[(dff[c] >= v[0]) & (dff[c] <= v[1])]
        elif isinstance(v, list):
            dff = dff[dff[c].isin(v)]
    st.markdown(f"Filtre uygulanmış satır: **{dff.shape[0]}**")
    # dağılımlar
    st.subheader("Dağılımlar")
    col1, col2 = st.columns(2)
    with col1:
        if numeric_cols:
            num = st.selectbox("Histogram (numeric)", options=numeric_cols)
            fig = px.histogram(dff, x=num, nbins=40, marginal="box")
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        if cat_cols:
            cat = st.selectbox("Kategori dağılımı", options=cat_cols)
            vc = dff[cat].value_counts().reset_index()
            vc.columns = [cat, "count"]
            fig2 = px.pie(vc, names=cat, values="count", title=f"{cat} oranları")
            st.plotly_chart(fig2, use_container_width=True)
    # model performance (eğer hedef kolonu varsa)
    st.markdown("---")
    st.subheader("Model Performansı (Eğer hedef sütunu mevcutsa)")
    if model is None:
        st.info("Model yüklenemedi; performans hesaplanamıyor.")
        return
    possible_targets = [c for c in df.columns if c.lower() in ("target", "label", "y", "recidivism", "risk")]
    if not possible_targets:
        st.info("CSV'de otomatik hedef sütunu bulunamadı (target/label gibi). Performans gösterilemez.")
        return
    ycol = st.selectbox("Hedef sütun seç", options=possible_targets)
    X_cols = [c for c in feature_names if c in dff.columns]
    if not X_cols:
        st.error("Model için kullanılacak feature sütunları dataset ile uyuşmuyor.")
        return
    X_all = dff[X_cols]
    y_all = dff[ycol]
    try:
        y_pred = model.predict(X_all)
        try:
            y_proba = model.predict_proba(X_all)[:,1]
        except Exception:
            y_proba = None
        st.write(f"Accuracy: {accuracy_score(y_all, y_pred):.3f}")
        st.write(f"Precision: {precision_score(y_all, y_pred, zero_division=0):.3f}")
        st.write(f"Recall: {recall_score(y_all, y_pred, zero_division=0):.3f}")
        st.write(f"F1: {f1_score(y_all, y_pred, zero_division=0):.3f}")
        if y_proba is not None:
            try:
                st.write(f"ROC-AUC: {roc_auc_score(y_all, y_proba):.3f}")
            except Exception:
                pass
        cm = confusion_matrix(y_all, y_pred)
        fig_cm, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig_cm)
    except Exception as e:
        st.error(f"Model performans hesaplanamadı: {e}")

def page_history():
    st.title("Tahmin Geçmişi")
    hist = st.session_state.pred_history
    if len(hist) == 0:
        st.info("Henüz tahmin yapılmadı.")
        return
    df_hist = pd.DataFrame(hist)
    st.dataframe(df_hist)
    csv = df_hist.to_csv(index=False).encode("utf-8")
    st.download_button("Tüm geçmişi CSV indir", data=csv, file_name="prediction_history.csv", mime="text/csv")
    if st.button("Geçmişi temizle"):
        st.session_state.pred_history = []
        st.experimental_rerun()

# ---------------------------
# Router (Sidebar)
# ---------------------------
PAGES = {
    "Anasayfa": page_home,
    "Tahmin Sistemi": page_predict,
    "Tavsiye Sistemi": page_recommendation,
    "Veri Analizi & Model": page_data_analysis,
    "Tahmin Geçmişi": page_history
}
st.sidebar.title("Gezinme")
choice = st.sidebar.radio("Sayfa seçin", list(PAGES.keys()))
st.sidebar.markdown("---")
st.sidebar.markdown("Not: Eğer `feature_names.pkl` veya diğer dosyalar eksik görünüyorsa, sidebar'dan 'GitHub'dan eksikleri indir' düğmesini tıklayın.")
PAGES[choice]()
