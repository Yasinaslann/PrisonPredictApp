# app.py
"""
Güncel, hata-tolerant Prison Risk Streamlit uygulaması.
- GitHub raw'tan indirme (sidebar butonu ile)
- feature_names.pkl vb. için sağlam yükleyiciler
- PDF Unicode hatasına karşı transliteration
- SHAP açıklama (bar + waterfall fallback)
- Manual target seçimi
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

st.set_page_config(page_title="Prison Risk App", layout="wide", initial_sidebar_state="expanded")

# -----------------------------
# Ayarlar - GitHub raw base
# -----------------------------
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/Yasinaslann/PrisonPredictApp/main/prison_app/"
EXPECTED_FILES = {
    "dataset": "Prisongüncelveriseti.csv",
    "model": "catboost_model.pkl",
    "feature_names": "feature_names.pkl",
    "cat_features": "cat_features.pkl",
    "cat_unique_values": "cat_unique_values.pkl",
    "bool_columns": "bool_columns.pkl"
}

LOCAL_DIR = os.getcwd()

# -----------------------------
# Yardımcı fonksiyonlar
# -----------------------------
def download_if_missing(filename, force=False):
    path = os.path.join(LOCAL_DIR, filename)
    if os.path.exists(path) and not force:
        return True, f"Zaten var: {path}"
    url = GITHUB_RAW_BASE + filename
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            with open(path, "wb") as f:
                f.write(r.content)
            return True, f"İndirildi: {path}"
        else:
            return False, f"HTTP {r.status_code} - {url}"
    except Exception as e:
        return False, f"İndirme hatası: {e}"

@st.cache_data
def load_csv_safe(path):
    return pd.read_csv(path)

def load_pickle_safe(path):
    """
    Birkaç yöntem deneyerek pickle / pandas.read_pickle / joblib fallback.
    """
    last_err = None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        last_err = e
    try:
        return pd.read_pickle(path)
    except Exception as e:
        last_err = e
    try:
        return joblib.load(path)
    except Exception as e:
        last_err = e
    # son çare: dosyayı text olarak oku ve eval dene (riskli, yalnızca kontrol amaçlı)
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            return eval(content)
    except Exception:
        # raise ilk yakalanan hatayı, üst seviyeye mesaj vermek için
        raise RuntimeError(f"Pickle load failed ({path}): {last_err}")

def safe_load_model(path):
    """
    Joblib / pickle ile model yüklemeye çalış.
    """
    try:
        return joblib.load(path)
    except Exception:
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Model yüklenemedi: {e}")

# Basit transliteration (Türkçe -> ASCII/latin1 uyumlu)
TRANSLIT_MAP = {
    "ç":"c","Ç":"C","ş":"s","Ş":"S","ı":"i","İ":"I","ğ":"g","Ğ":"G","ü":"u","Ü":"U","ö":"o","Ö":"O",
    "â":"a","Â":"A","ê":"e","ô":"o","—":"-","–":"-"
}
def normalize_text_for_pdf(s):
    if s is None:
        return ""
    s = str(s)
    for k,v in TRANSLIT_MAP.items():
        s = s.replace(k,v)
    # remove uncommon control chars
    s = "".join(ch if ord(ch) < 0x100 else "?" for ch in s)
    return s

def create_pdf_bytes(record: dict, recommendation: str):
    """
    FPDF ile PDF oluştururken Unicode sorunlarını engellemek için
    tüm metinleri transliterate edip latin1 uyumlu hale getiririz.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=11)
    pdf.cell(0, 8, normalize_text_for_pdf("Prediction Report"), ln=True)
    pdf.ln(2)
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 7, normalize_text_for_pdf(f"Tarih: {record.get('timestamp','')}"), ln=True)
    pdf.ln(4)
    pdf.cell(0, 7, normalize_text_for_pdf("Girdi Değerleri:"), ln=True)
    pdf.ln(2)
    for k, v in record.items():
        if k in ("timestamp", "pred", "proba"):
            continue
        line = f"- {k}: {v}"
        pdf.multi_cell(0, 6, normalize_text_for_pdf(line))
    pdf.ln(2)
    pdf.cell(0, 7, normalize_text_for_pdf(f"Pred: {record.get('pred','')}  |  Proba: {record.get('proba','')}"), ln=True)
    pdf.ln(6)
    pdf.cell(0, 7, normalize_text_for_pdf("Öneriler:"), ln=True)
    pdf.ln(2)
    for para in normalize_text_for_pdf(recommendation).split("\n"):
        pdf.multi_cell(0, 6, para)
    # output as bytes
    return pdf.output(dest="S").encode("latin-1", errors="replace")

def generate_recommendation(score, inputs=None):
    if score is None:
        return "Skor hesaplanamadı; öneri üretilemiyor."
    s = float(score)
    tips = []
    if s >= 0.85:
        tips.append("Yüksek risk: Acil müdahale, yoğun terapi ve sosyal destek önerilir.")
    elif s >= 0.6:
        tips.append("Orta-yüksek risk: Hedefli eğitim ve sık takip önerilir.")
    elif s >= 0.35:
        tips.append("Orta risk: İzleme ve mesleki eğitim faydalı olabilir.")
    else:
        tips.append("Düşük risk: Standart izleme ile devam edilebilir.")
    if isinstance(inputs, dict):
        edu = str(inputs.get("education_level","")).lower()
        if edu in ("low","none","ilkokul","ortaokul"):
            tips.append("Eğitim desteği: Temel/mesleki eğitime yönlendir.")
        emp = str(inputs.get("employment_status","")).lower()
        if emp in ("unemployed","işsiz","yok"):
            tips.append("İstihdam desteği: İş eğitimi ve yerleştirme programları.")
    return "\n\n".join(tips)

def safe_predict(model, X_df):
    pred = None; proba = None
    try:
        p = model.predict(X_df)
        if hasattr(p, "__len__"):
            pred = int(p[0])
        else:
            pred = int(p)
    except Exception:
        pred = None
    try:
        if hasattr(model, "predict_proba"):
            prob_arr = model.predict_proba(X_df)
            if prob_arr.ndim == 2 and prob_arr.shape[1] >= 2:
                proba = float(prob_arr[:,1][0])
            else:
                proba = float(np.ravel(prob_arr)[0])
        else:
            proba = None
    except Exception:
        proba = None
    return pred, proba

# -----------------------------
# Sidebar: GitHub'dan çekme butonu
# -----------------------------
st.sidebar.header("Repo / Dosya Kontrolleri")
if st.sidebar.button("GitHub'dan eksikleri indir (güncelle)"):
    st.sidebar.info("İndiriliyor... (30s timeout olabilir)")
    for fn in EXPECTED_FILES.values():
        ok, msg = download_if_missing(fn, force=True)
        if ok:
            st.sidebar.success(f"{fn}: {msg}")
        else:
            st.sidebar.error(f"{fn}: {msg}")

# -----------------------------
# Dosyaları yükleme (yerelde varsa)
# -----------------------------
df = pd.DataFrame()
model = None
feature_names = None
cat_features = []
cat_unique_values = {}
bool_columns = []

# dataset
if os.path.exists(EXPECTED_FILES["dataset"]):
    try:
        df = load_csv_safe(EXPECTED_FILES["dataset"])
    except Exception as e:
        st.sidebar.error(f"CSV yükleme hatası: {e}")

# model
if os.path.exists(EXPECTED_FILES["model"]):
    try:
        model = safe_load_model(EXPECTED_FILES["model"])
    except Exception as e:
        st.sidebar.error(f"Model yükleme hatası: {e}")

# diğer pickles
for key in ("feature_names","cat_features","cat_unique_values","bool_columns"):
    fn = EXPECTED_FILES[key]
    if os.path.exists(fn):
        try:
            val = load_pickle_safe(fn)
            if key == "feature_names":
                feature_names = val
            elif key == "cat_features":
                cat_features = val
            elif key == "cat_unique_values":
                cat_unique_values = val
            elif key == "bool_columns":
                bool_columns = val
        except Exception as e:
            st.sidebar.error(f"{fn} okunamadı: {e}")

# fallback feature_names
if not feature_names:
    if not df.empty:
        possible_targets = [c for c in df.columns if c.lower() in ("target","label","y","recidivism","risk")]
        if possible_targets:
            feature_names = [c for c in df.columns if c not in possible_targets]
        else:
            feature_names = df.columns.tolist()
    else:
        feature_names = []

# -----------------------------
# Session state
# -----------------------------
if "pred_history" not in st.session_state:
    st.session_state.pred_history = []

# -----------------------------
# Sayfa fonksiyonları
# -----------------------------
def page_home():
    st.title("Prison Risk App — Anasayfa")
    st.markdown("""
    Bu uygulama: **risk tahmini**, **SHAP açıklama**, **interaktif analiz** ve **kişiye özel öneri** sağlar.
    Dosyaların repo içinde olduğundan emin olun veya sidebar'dan 'GitHub'dan eksikleri indir' düğmesini kullanın.
    """)
    st.markdown("### Yüklü dosyalar")
    for k,fn in EXPECTED_FILES.items():
        st.write(f"- `{fn}` : {'✔' if os.path.exists(fn) else '❌'}")
    st.markdown("---")
    if df.empty:
        st.warning("Veri yüklenemedi.")
    else:
        st.subheader("Veri önizleme")
        st.dataframe(df.head(8))
        st.markdown(f"**Satır / Sütun:** {df.shape[0]} x {df.shape[1]}")

def build_dynamic_inputs(features):
    inputs = {}
    if not features:
        st.error("Feature listesi bulunamadı.")
        return inputs
    cols = st.columns(2)
    i=0
    for feat in features:
        c = cols[i%2]
        label = feat
        hint = ""
        if (not df.empty) and (feat in df.columns):
            if np.issubdtype(df[feat].dtype, np.number):
                hint = f"med: {df[feat].median():.2f}"
            else:
                hint = f"unique: {df[feat].nunique()}"
        if feat in cat_features:
            options = cat_unique_values.get(feat, [])
            if options:
                val = c.selectbox(label, options=options, key=f"in_{feat}")
            else:
                val = c.text_input(label, key=f"in_{feat}")
        elif feat in bool_columns:
            val = c.selectbox(label, options=[0,1], key=f"in_{feat}")
        else:
            default = 0.0
            if (not df.empty) and (feat in df.columns) and np.issubdtype(df[feat].dtype, np.number):
                default = float(df[feat].median())
            val = c.number_input(label, value=default, key=f"in_{feat}")
        inputs[feat]=val
        i+=1
    return inputs

def page_predict():
    st.title("Tahmin Sistemi")
    if not feature_names:
        st.error("feature_names yüklenemedi; tahmin formu oluşturulamıyor.")
        return
    st.markdown("Tüm özellikleri doldurun ve 'Tahmin Et' butonuna basın.")
    with st.form("frm_predict"):
        user_inputs = build_dynamic_inputs(feature_names)
        submitted = st.form_submit_button("Tahmin Et")
    if submitted:
        if model is None:
            st.error("Model yüklenemedi. `catboost_model.pkl` eksik veya bozuk.")
            return
        try:
            X = pd.DataFrame([user_inputs], columns=feature_names)
        except Exception as e:
            st.error(f"Veriframe oluşturulamadı: {e}")
            return
        pred, proba = safe_predict(model, X)
        st.metric("Tahmin (label)", pred if pred is not None else "—")
        st.metric("Risk Skoru (0-1)", f"{proba:.4f}" if proba is not None else "—")
        record = {"timestamp": datetime.now().isoformat(timespec="seconds"), **user_inputs, "pred": pred, "proba": proba}
        st.session_state.pred_history.insert(0, record)
        # SHAP explanation
        st.subheader("Tahmin Açıklaması (SHAP)")
        try:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X)
            # pick class 1 if list
            if isinstance(shap_vals, list) and len(shap_vals)>1:
                arr = np.array(shap_vals[1])[0]
                expected = explainer.expected_value[1] if hasattr(explainer.expected_value,"__len__") else explainer.expected_value
            else:
                arr = np.array(shap_vals)[0]
                expected = explainer.expected_value if not hasattr(explainer.expected_value,"__len__") else explainer.expected_value
            df_sh = pd.DataFrame({"feature": X.columns, "shap_value": arr})
            df_sh["abs"] = df_sh["shap_value"].abs()
            df_sh = df_sh.sort_values("abs", ascending=False).head(20)
            fig = px.bar(df_sh[::-1], x="shap_value", y="feature", orientation="h", title="En etkili feature'lar (SHAP)")
            st.plotly_chart(fig, use_container_width=True)
            # waterfall fallback
            try:
                st.subheader("Waterfall (detaylı)")
                fig_w, ax = plt.subplots(figsize=(8,4))
                # waterfall_legacy
                try:
                    shap.plots._waterfall.waterfall_legacy(expected, arr, feature_names=X.columns, max_display=12)
                    st.pyplot(fig_w)
                except Exception:
                    try:
                        shap.plots.waterfall(expected, arr, feature_names=X.columns)
                        st.pyplot(fig_w)
                    except Exception:
                        st.info("Waterfall çizilemedi; bar grafik gösterildi.")
            except Exception:
                st.info("Waterfall gösterimi atlandı.")
        except Exception as e:
            st.error(f"SHAP açıklaması oluşturulamadı: {e}")
        # öneri
        st.subheader("Kişisel Öneri")
        rec_text = generate_recommendation(proba, user_inputs)
        st.info(rec_text)
        # download buttons
        col1, col2 = st.columns(2)
        with col1:
            csvb = pd.DataFrame([record]).to_csv(index=False).encode("utf-8")
            st.download_button("Bu tahmini CSV indir", data=csvb, file_name="prediction_record.csv", mime="text/csv")
        with col2:
            try:
                pdfb = create_pdf_bytes(record, rec_text)
                st.download_button("Bu tahmini PDF indir", data=pdfb, file_name="prediction_record.pdf", mime="application/pdf")
            except Exception as e:
                st.error(f"PDF oluşturulamadı: {e}")

def page_recommendation():
    st.title("Tavsiye Sistemi")
    if len(st.session_state.pred_history)==0:
        st.info("Henüz tahmin yapılmadı.")
        return
    hist_df = pd.DataFrame(st.session_state.pred_history)
    sel = st.selectbox("Geçmişten bir kayıt seçin", options=hist_df.index, format_func=lambda i: f"{hist_df.loc[i,'timestamp']} — {hist_df.loc[i,'proba']}")
    row = hist_df.loc[sel].to_dict()
    st.subheader("Seçili Kayıt")
    st.json(row)
    rec_text = generate_recommendation(row.get("proba"), row)
    st.subheader("Öneriler")
    st.info(rec_text)
    if st.button("Seçili kaydı PDF indir"):
        try:
            pdfb = create_pdf_bytes(row, rec_text)
            st.download_button("PDF indir", data=pdfb, file_name="recommendation.pdf", mime="application/pdf")
        except Exception as e:
            st.error(f"PDF oluşturulamadı: {e}")

def page_data_analysis():
    st.title("Veri Analizi & Model Performansı")
    if df.empty:
        st.warning("Veri yüklenemedi.")
        return
    st.markdown(f"Dataset boyutu: {df.shape[0]} x {df.shape[1]}")
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(exclude=[np.number]).columns.tolist()
    st.sidebar.subheader("Analiz filtreleri")
    filters = {}
    for c in numeric[:6]:
        mn, mx = float(df[c].min()), float(df[c].max())
        lo_hi = st.sidebar.slider(f"{c} aralığı", mn, mx, (mn, mx))
        filters[c] = lo_hi
    for c in categorical[:4]:
        opts = df[c].unique().tolist()
        sel = st.sidebar.multiselect(f"{c} seç", options=opts, default=opts[:min(5,len(opts))])
        filters[c]=sel
    dff = df.copy()
    for c,v in filters.items():
        if isinstance(v, tuple):
            dff = dff[(dff[c]>=v[0]) & (dff[c]<=v[1])]
        elif isinstance(v, list):
            dff = dff[dff[c].isin(v)]
    st.markdown(f"Filtrelenmiş satır: **{dff.shape[0]}**")
    col1, col2 = st.columns(2)
    with col1:
        if numeric:
            sel = st.selectbox("Histogram (numeric)", options=numeric)
            fig = px.histogram(dff, x=sel, nbins=40, marginal="box")
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        if categorical:
            selc = st.selectbox("Kategori dağılımı", options=categorical)
            vc = dff[selc].value_counts().reset_index()
            vc.columns=[selc,"count"]
            fig2 = px.pie(vc, names=selc, values="count", title=f"{selc} oranları")
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("Model performansı (varsa)")
    if model is None:
        st.info("Model yüklenemedi; performans hesaplanamaz.")
        return
    possible_targets = [c for c in df.columns if c.lower() in ("target","label","y","recidivism","risk")]
    if not possible_targets:
        st.info("Otomatik hedef sütunu bulunamadı. Manuel seçebilirsiniz.")
        chosen = st.selectbox("Hedef sütunu seçin (manüel)", options=list(df.columns))
        if not chosen:
            st.warning("Hedef sütunu seçilmedi; performans gösterilemez.")
            return
        ycol = chosen
    else:
        ycol = st.selectbox("Hedef sütunu seç (otomatik algılandı varsa)", options=possible_targets, index=0)
    Xcols = [c for c in feature_names if c in dff.columns]
    if not Xcols:
        st.error("feature_names ile CSV sütunları uyuşmuyor. feature_names.pkl doğru mu?")
        return
    X = dff[Xcols]
    y = dff[ycol]
    try:
        y_pred = model.predict(X)
        try:
            y_proba = model.predict_proba(X)[:,1]
        except Exception:
            y_proba = None
        st.write(f"Accuracy: {accuracy_score(y,y_pred):.3f}")
        st.write(f"Precision: {precision_score(y,y_pred, zero_division=0):.3f}")
        st.write(f"Recall: {recall_score(y,y_pred, zero_division=0):.3f}")
        st.write(f"F1: {f1_score(y,y_pred, zero_division=0):.3f}")
        if y_proba is not None:
            try:
                st.write(f"ROC-AUC: {roc_auc_score(y,y_proba):.3f}")
            except Exception:
                pass
        cm = confusion_matrix(y,y_pred)
        fig_cm, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        st.pyplot(fig_cm)
    except Exception as e:
        st.error(f"Model performans hesaplanamadı: {e}")

def page_history():
    st.title("Tahmin Geçmişi")
    hist = st.session_state.pred_history
    if not hist:
        st.info("Henüz tahmin yok.")
        return
    dfh = pd.DataFrame(hist)
    st.dataframe(dfh)
    csvb = dfh.to_csv(index=False).encode("utf-8")
    st.download_button("Geçmişi CSV indir", data=csvb, file_name="prediction_history.csv", mime="text/csv")
    if st.button("Geçmişi temizle"):
        st.session_state.pred_history = []
        st.experimental_rerun()

# -----------------------------
# Router
# -----------------------------
PAGES = {
    "Anasayfa": page_home,
    "Tahmin Sistemi": page_predict,
    "Tavsiye Sistemi": page_recommendation,
    "Veri Analizi & Model": page_data_analysis,
    "Tahmin Geçmişi": page_history
}

st.sidebar.title("Gezinme")
choice = st.sidebar.radio("Sayfa seç", list(PAGES.keys()))
st.sidebar.markdown("---")
st.sidebar.markdown("Eğer dosyalar eksik görünüyorsa, 'GitHub'dan eksikleri indir' butonunu kullanın.")
PAGES[choice]()
