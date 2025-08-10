# app.py
"""
Güncel, dayanıklı ve kapsamlı Prison Risk Streamlit uygulaması.
- Çoklu sayfa: Anasayfa, Tahmin, Tavsiye, Veri Analizi & Model, Geçmiş, Ayarlar
- SHAP explainability (multiclass destekli)
- Multiclass/binary için uygun metrikler
- Farklı pickle formatları için sağlam yükleme
- PDF Unicode sorunu transliteration ile çözüldü
- Dosyaları GitHub raw'tan indirme & kullanıcı upload'u desteği
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
import sklearn
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report,
                             precision_recall_fscore_support)

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(page_title="Prison Risk App — Advanced", layout="wide", initial_sidebar_state="expanded")

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

# ----------------------------
# UTIL: Dosya indirme ve güvenli yükleme
# ----------------------------
def download_from_github(filename, force=False):
    local_path = os.path.join(LOCAL_DIR, filename)
    if os.path.exists(local_path) and not force:
        return True, f"Zaten var: {local_path}"
    url = GITHUB_RAW_BASE + filename
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            with open(local_path, "wb") as f:
                f.write(r.content)
            return True, f"İndirildi: {local_path}"
        else:
            return False, f"HTTP {r.status_code} - {url}"
    except Exception as e:
        return False, str(e)

def load_pickle_flexible(path):
    err = None
    # Binary pickle
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        err = e
    # pandas read_pickle
    try:
        return pd.read_pickle(path)
    except Exception as e:
        err = e
    # joblib
    try:
        return joblib.load(path)
    except Exception as e:
        err = e
    # as last resort, try to eval text (risky; use only if other methods fail)
    try:
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read()
            return eval(txt)
    except Exception:
        pass
    raise RuntimeError(f"Pickle okunamadı: {path} | Son hata: {err}")

def load_model_flexible(path):
    # joblib or pickle
    try:
        return joblib.load(path)
    except Exception:
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Model yüklenemedi: {e}")

def load_csv_safe(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        raise RuntimeError(f"CSV okunamadı: {e}")

# ----------------------------
# UTIL: PDF - Unicode güvenli hale getirme
# ----------------------------
TRANSLIT = {
    "ç":"c","Ç":"C","ş":"s","Ş":"S","ı":"i","İ":"I","ğ":"g","Ğ":"G","ü":"u","Ü":"U","ö":"o","Ö":"O",
    "â":"a","Â":"A","ê":"e","ô":"o","—":"-","–":"-"
}
def normalize_for_pdf(s):
    if s is None: return ""
    s = str(s)
    for k,v in TRANSLIT.items():
        s = s.replace(k,v)
    # replace any codepoints beyond latin1 by '?'
    s = "".join(ch if ord(ch) < 256 else "?" for ch in s)
    return s

def create_pdf_bytes(record: dict, recommendation: str):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.set_font("Arial", size=11)
    pdf.cell(0, 8, normalize_for_pdf("Prediction Report"), ln=True)
    pdf.ln(2)
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 7, normalize_for_pdf(f"Tarih: {record.get('timestamp','')}"), ln=True)
    pdf.ln(4)
    pdf.cell(0, 7, normalize_for_pdf("Girdi Değerleri:"), ln=True)
    pdf.ln(2)
    for k, v in record.items():
        if k in ("timestamp", "pred", "proba"):
            continue
        pdf.multi_cell(0, 6, normalize_for_pdf(f"- {k}: {v}"))
    pdf.ln(2)
    pdf.cell(0, 7, normalize_for_pdf(f"Pred: {record.get('pred','')}  |  Proba: {record.get('proba','')}"), ln=True)
    pdf.ln(6)
    pdf.cell(0, 7, normalize_for_pdf("Öneriler:"), ln=True)
    pdf.ln(2)
    for para in normalize_for_pdf(recommendation).split("\n"):
        pdf.multi_cell(0, 6, para)
    return pdf.output(dest="S").encode("latin-1", errors="replace")

# ----------------------------
# Genel yardımcılar
# ----------------------------
def safe_predict(model, X_df):
    """Model'den pred ve proba almaya çalışır."""
    pred = None; proba = None
    try:
        p = model.predict(X_df)
        pred = int(np.ravel(p)[0])
    except Exception:
        pred = None
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_df)
            if probs.ndim == 2 and probs.shape[1] >= 2:
                proba = float(probs[:,1][0])
            else:
                proba = float(np.ravel(probs)[0])
        else:
            proba = None
    except Exception:
        proba = None
    return pred, proba

def aligned_input_df(user_inputs: dict, feature_list: list, df_sample: pd.DataFrame=None):
    """
    feature_list sırasına göre DataFrame oluşturur; eksik varsa df_sample'dan medyan/mod ile doldurur.
    """
    X = pd.DataFrame([user_inputs])
    # ensure columns present
    missing = [f for f in feature_list if f not in X.columns]
    for m in missing:
        if df_sample is not None and m in df_sample.columns:
            if np.issubdtype(df_sample[m].dtype, np.number):
                X[m] = float(df_sample[m].median())
            else:
                X[m] = df_sample[m].mode().iloc[0] if not df_sample[m].mode().empty else ""
        else:
            X[m] = 0
    # reorder
    X = X[feature_list]
    return X

def generate_personalized_recs(score, top_shap_features=None, inputs=None):
    """
    Basit ama etkili tavsiye motoru:
    - skor tabanlı genel tavsiyeler
    - SHAP'tan en etkili özelliklere göre hedeflenmiş öneriler
    """
    base = []
    if score is None:
        base.append("Skor hesaplanamadı — öneri üretilemiyor.")
        return "\n".join(base)
    s = float(score)
    if s >= 0.85:
        base.append("Yüksek risk: Acil müdahale. Yoğun psikososyal destek, bireysel terapi ve sık takip önerilir.")
    elif s >= 0.6:
        base.append("Orta-yüksek risk: Hedefli eğitim/iş desteği ve davranışsal terapi önerilir.")
    elif s >= 0.35:
        base.append("Orta risk: Mesleki eğitim ve düzenli izleme faydalı olabilir.")
    else:
        base.append("Düşük risk: Standart rehabilitasyon programları ve izleme uygundur.")
    # SHAP-a dayalı hedef öneriler (anahtar kelimelere göre)
    if top_shap_features:
        feats = [f.lower() for f in top_shap_features]
        if any("education" in f or "edu" in f or "okul" in f for f in feats):
            base.append("Eğitim önemli etken: Temel/mesleki eğitim programlarına yönlendirme öner.")
        if any("employ" in f or "job" in f or "iş" in f for f in feats):
            base.append("İstihdam faktörü etkili: İş eğitimi, staj ve işe yerleştirme programlarına yönlendirme.")
        if any("drug" in f or "substance" in f or "alcohol" in f or "madde" in f for f in feats):
            base.append("Madde kullanımı etkili ise: Madde bağımlılığı tedavi programları ve terapi öner.")
        if any("supervision" in f or "monitor" in f or "gözetim" in f for f in feats):
            base.append("Gözetim/denetim etkili: Denetimli serbestlik planı ve düzenli raporlama öner.")
    # inputs-based quick rules
    if inputs and isinstance(inputs, dict):
        if "education_level" in inputs and str(inputs.get("education_level","")).lower() in ("low","none","ilkokul"):
            base.append("Kişisel bilgi: Eğitim seviyesi düşük — temel eğitim programları öner.")
        if "employment_status" in inputs and str(inputs.get("employment_status","")).lower() in ("unemployed","işsiz","yok"):
            base.append("Kişisel bilgi: İşsizlik — istihdam desteği ve iş eğitimi öner.")
    return "\n\n".join(base)

# ----------------------------
# LOAD default files (if present locally)
# ----------------------------
df = pd.DataFrame()
model = None
feature_names = []
cat_features = []
cat_unique_values = {}
bool_columns = []

load_errors = []

# Try exist -> load
if os.path.exists(EXPECTED_FILES["dataset"]):
    try:
        df = load_csv_safe(EXPECTED_FILES["dataset"])
    except Exception as e:
        load_errors.append(f"CSV yükleme hatası: {e}")

if os.path.exists(EXPECTED_FILES["model"]):
    try:
        model = load_model_flexible(EXPECTED_FILES["model"])
    except Exception as e:
        load_errors.append(f"Model yükleme hatası: {e}")

for key in ("feature_names","cat_features","cat_unique_values","bool_columns"):
    fn = EXPECTED_FILES[key]
    if os.path.exists(fn):
        try:
            val = load_pickle_flexible(fn)
            if key == "feature_names":
                feature_names = list(val)
            elif key == "cat_features":
                cat_features = list(val)
            elif key == "cat_unique_values":
                cat_unique_values = dict(val)
            elif key == "bool_columns":
                bool_columns = list(val)
        except Exception as e:
            load_errors.append(f"{fn} okunamadı: {e}")

# fallback feature_names inference
if not feature_names:
    if not df.empty:
        possible_targets = [c for c in df.columns if c.lower() in ("target","label","y","recidivism","risk")]
        if possible_targets:
            feature_names = [c for c in df.columns if c not in possible_targets]
        else:
            feature_names = df.columns.tolist()
    else:
        feature_names = []

# ----------------------------
# Session state
# ----------------------------
if "pred_history" not in st.session_state:
    st.session_state.pred_history = []

# ----------------------------
# SIDEBAR: controls & uploads
# ----------------------------
st.sidebar.title("Kontroller & Dosyalar")
st.sidebar.markdown("Repo: `Yasinaslann/PrisonPredictApp` — `prison_app` klasörü içerikleri kullanılabilir.")
if st.sidebar.button("GitHub'dan eksikleri indir / güncelle"):
    st.sidebar.info("İndiriliyor... (30s timeout)")
    for f in EXPECTED_FILES.values():
        ok, msg = download_from_github(f, force=True)
        if ok:
            st.sidebar.success(msg)
        else:
            st.sidebar.error(msg)
    st.sidebar.info("İndirme isteği tamamlandı — sayfayı yenileyin (reload).")

st.sidebar.markdown("---")
st.sidebar.subheader("Yerel/Geçici Yükleme")
uploaded_csv = st.sidebar.file_uploader("CSV yükle (opsiyonel)", type=["csv"])
uploaded_model = st.sidebar.file_uploader("Model yükle (.pkl veya .joblib) (opsiyonel)", type=["pkl","joblib"])
uploaded_feature_names = st.sidebar.file_uploader("feature_names.pkl (opsiyonel)", type=["pkl"])
if uploaded_csv is not None:
    try:
        df = pd.read_csv(uploaded_csv)
        st.sidebar.success("CSV yüklendi (runtime, kalıcı değil).")
    except Exception as e:
        st.sidebar.error(f"CSV yükleme hatası: {e}")
if uploaded_model is not None:
    try:
        # save temp and load
        tmpm = os.path.join(LOCAL_DIR, "temp_uploaded_model")
        with open(tmpm, "wb") as f:
            f.write(uploaded_model.getvalue())
        model = load_model_flexible(tmpm)
        os.remove(tmpm)
        st.sidebar.success("Model (runtime) yüklendi.")
    except Exception as e:
        st.sidebar.error(f"Model yükleme hatası: {e}")
if uploaded_feature_names is not None:
    try:
        tmpf = os.path.join(LOCAL_DIR, "temp_feature_names.pkl")
        with open(tmpf, "wb") as f:
            f.write(uploaded_feature_names.getvalue())
        feature_names = list(load_pickle_flexible(tmpf))
        os.remove(tmpf)
        st.sidebar.success("feature_names (runtime) yüklendi.")
    except Exception as e:
        st.sidebar.error(f"feature_names yükleme hatası: {e}")

if load_errors:
    with st.sidebar.expander("Yükleme hataları (detay)"):
        for le in load_errors:
            st.write(f"- {le}")

st.sidebar.markdown("---")
st.sidebar.markdown("Proje ayarları:")
thres_default = st.sidebar.slider("Binary threshold (proba -> sınıf), varsayılan 0.5", 0.0, 1.0, 0.5, 0.01)

# ----------------------------
# PAGES
# ----------------------------
def page_home():
    st.title("Prison Risk App — Anasayfa")
    st.markdown("""
    Bu uygulama; **risk tahmini**, **açıklanabilirlik (SHAP)**, **etkileşimli veri analizi** ve **kişisel öneri** üretmek üzere tasarlanmıştır.
    - Çok sınıflı ve ikili sınıflandırma senaryoları desteklenir.
    - Model ve veri dosyalarını repo'dan indirebilir ya da sidebar üzerinden yükleyebilirsiniz.
    """)
    st.markdown("### Mevcut durum")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Veri (CSV):", f"{EXPECTED_FILES['dataset']} — {'✔' if not df.empty else '❌'}")
        st.write("Model:", f"{EXPECTED_FILES['model']} — {'✔' if model is not None else '❌'}")
        st.write("feature_names:", f"{'yüklü' if feature_names else '❌'}")
    with col2:
        st.write("Kategorik feature bilgisi:", f"{'✔' if cat_features else '❌'}")
        st.write("Bool columns:", f"{'✔' if bool_columns else '❌'}")
        st.write("SHAP available:", f"{'✔' if model is not None else '❌'}")
    st.markdown("---")
    if not df.empty:
        st.subheader("Veri önizleme")
        st.dataframe(df.head(10))
        st.markdown(f"Satır / Sütun: **{df.shape[0]} x {df.shape[1]}**")
        if st.checkbox("Örnek CSV indir (şablon)"):
            sample = df.head(50)
            st.download_button("Örnek CSV indir", data=sample.to_csv(index=False).encode("utf-8"), file_name="sample_data.csv", mime="text/csv")
    else:
        st.warning("Veri bulunmuyor. Sidebar'dan indir ya da upload et.")

def build_inputs(feature_list):
    """Feature list'ten dinamik input UI oluşturur (güzel görünüm)."""
    inputs = {}
    if not feature_list:
        st.error("Feature listesi yok.")
        return inputs
    st.markdown("### Özellikleri doldurun")
    # bölümlere ayır: kategorik, bool, numeric
    numeric_feats = [f for f in feature_list if f not in cat_features and f not in bool_columns]
    cat_feats = [f for f in feature_list if f in cat_features]
    bool_feats = [f for f in feature_list if f in bool_columns]
    exp_num = st.expander(f"Sayısal ({len(numeric_feats)})", expanded=True)
    exp_cat = st.expander(f"Kategorik ({len(cat_feats)})", expanded=True)
    exp_bool = st.expander(f"Bool ({len(bool_feats)})", expanded=False)
    # numeric two-column layout
    cols = exp_num.columns(2)
    for i, f in enumerate(numeric_feats):
        c = cols[i % 2]
        default = 0.0
        if not df.empty and f in df.columns and np.issubdtype(df[f].dtype, np.number):
            default = float(df[f].median())
        val = c.number_input(f"{f}", value=default, key=f"in_{f}")
        inputs[f] = val
    for f in cat_feats:
        opts = cat_unique_values.get(f, [])
        if opts:
            val = exp_cat.selectbox(f"{f}", options=opts, index=0, key=f"in_{f}")
        else:
            val = exp_cat.text_input(f"{f}", key=f"in_{f}")
        inputs[f] = val
    for f in bool_feats:
        val = exp_bool.selectbox(f"{f}", [0,1], index=0, key=f"in_{f}")
        inputs[f] = int(val)
    return inputs

def page_predict():
    st.title("Tahmin Sistemi")
    if not feature_names:
        st.error("feature_names yüklenemedi; tahmin formu oluşturulamıyor.")
        return
    with st.form("frm_predict"):
        user_inputs = build_inputs(feature_names)
        submitted = st.form_submit_button("Tahmin Et")
    if submitted:
        if model is None:
            st.error("Model yok; sidebar'dan yükleyin veya repo'dan indirin.")
            return
        # align inputs
        try:
            X = aligned_input_df(user_inputs, feature_names, df_sample=df if not df.empty else None)
        except Exception as e:
            st.error(f"Girdi DataFrame oluşturulamadı: {e}")
            return
        # predict
        pred, proba = safe_predict(model, X)
        # if multiclass and proba is array, show top-k
        st.metric("Tahmin (label)", pred if pred is not None else "—")
        if proba is not None:
            st.metric("Risk skor (binary proba for class1)", f"{proba:.4f}")
        # if model supports predict_proba and multiclass, show class probabilities
        if hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba(X)
                if probs.ndim == 2 and probs.shape[1] > 2:
                    pr_df = pd.DataFrame(probs, columns=[f"class_{i}" for i in range(probs.shape[1])])
                    st.subheader("Class Probabilities")
                    st.dataframe(pr_df.T)
            except Exception:
                pass
        # SHAP explain (multiclass support)
        st.subheader("Tahmin Açıklaması (SHAP)")
        try:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X)
            # Determine classes
            classes = None
            try:
                # some models have classes_ attribute
                classes = list(getattr(model, "classes_", list(range(np.array(shap_vals).shape[0]))))
            except Exception:
                classes = list(range(np.array(shap_vals).shape[0])) if isinstance(shap_vals, list) else [0]
            if isinstance(shap_vals, list) and len(shap_vals) > 1:
                # multiclass: let user pick class index
                class_idx = st.selectbox("SHAP için sınıf seçin (multiclass)", options=list(range(len(shap_vals))), format_func=lambda i: str(classes[i]) if classes else str(i))
                arr = np.array(shap_vals[class_idx])[0]
            else:
                arr = np.array(shap_vals)[0] if not isinstance(shap_vals, list) else np.array(shap_vals)[0]
            # bar chart
            df_sh = pd.DataFrame({"feature": X.columns, "shap_value": arr})
            df_sh["abs_shap"] = df_sh["shap_value"].abs()
            df_sh = df_sh.sort_values("abs_shap", ascending=False).head(25)
            fig = px.bar(df_sh[::-1], x="shap_value", y="feature", orientation="h", title="Top SHAP features (mutlak değer)")
            st.plotly_chart(fig, use_container_width=True)
            top_feats = df_sh.feature.tolist()[:6]
        except Exception as e:
            st.error(f"SHAP açıklaması oluşturulamadı: {e}")
            top_feats = None
        # personalized recommendation using shap/top features
        rec_text = generate_personalized_recs(proba, top_shap_features=top_feats, inputs=user_inputs)
        st.subheader("Kişisel Öneri")
        st.info(rec_text)
        # store record
        rec = {"timestamp": datetime.now().isoformat(timespec="seconds"), **user_inputs, "pred": pred, "proba": proba}
        st.session_state.pred_history.insert(0, rec)
        # downloads
        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            csvb = pd.DataFrame([rec]).to_csv(index=False).encode("utf-8")
            st.download_button("CSV indir", data=csvb, file_name="prediction_single.csv", mime="text/csv")
        with col2:
            try:
                pdfb = create_pdf_bytes(rec, rec_text)
                st.download_button("PDF indir", data=pdfb, file_name="prediction_single.pdf", mime="application/pdf")
            except Exception as e:
                st.error(f"PDF oluşturulamadı: {e}")
        with col3:
            st.button("Kayıtları Göster (Sidebar)")

def page_recommendation():
    st.title("Tavsiye Sistemi — Geçmişten Seçim")
    if not st.session_state.pred_history:
        st.info("Henüz tahmin yok. Tahmin sayfasına gidin.")
        return
    dfh = pd.DataFrame(st.session_state.pred_history)
    sel = st.selectbox("Geçmiş kayıtlardan seçin", options=dfh.index, format_func=lambda i: f"{dfh.loc[i,'timestamp']} — {dfh.loc[i,'proba']}")
    row = dfh.loc[sel].to_dict()
    st.subheader("Seçili Kayıt")
    st.json(row)
    rec_text = generate_personalized_recs(row.get("proba"), top_shap_features=None, inputs=row)
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
        st.warning("Veri bulunmuyor. CSV yükle veya indir.")
        return
    st.markdown(f"Dataset boyutu: **{df.shape[0]} satır x {df.shape[1]} sütun**")
    # Basic filters
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(exclude=[np.number]).columns.tolist()
    st.sidebar.subheader("Analiz filtreleri (quick)")
    filters = {}
    for c in numeric[:6]:
        mn, mx = float(df[c].min()), float(df[c].max())
        filters[c] = st.sidebar.slider(f"{c} aralığı", mn, mx, (mn, mx))
    for c in categorical[:4]:
        opts = df[c].unique().tolist()
        filters[c] = st.sidebar.multiselect(f"{c} seç", options=opts, default=opts[:min(5,len(opts))])
    dff = df.copy()
    for c, v in filters.items():
        if isinstance(v, tuple):
            dff = dff[(dff[c] >= v[0]) & (dff[c] <= v[1])]
        elif isinstance(v, list):
            dff = dff[dff[c].isin(v)]
    st.markdown(f"Filtrelenmiş: **{dff.shape[0]}** satır")
    # Visuals
    col1, col2 = st.columns(2)
    with col1:
        num = st.selectbox("Histogram (numeric)", options=numeric, index=0)
        fig = px.histogram(dff, x=num, nbins=40, marginal="box", title=f"{num} dağılımı")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        if categorical:
            cat = st.selectbox("Kategori dağılımı", options=categorical, index=0)
            vc = dff[cat].value_counts().reset_index()
            vc.columns = [cat, "count"]
            fig2 = px.pie(vc, names=cat, values="count", title=f"{cat} oranları")
            st.plotly_chart(fig2, use_container_width=True)
    st.markdown("---")
    st.subheader("Model Performansı")
    if model is None:
        st.info("Model yüklenemedi; performans hesaplanamaz.")
        return
    # choose target
    possible_targets = [c for c in df.columns if c.lower() in ("target","label","y","recidivism","risk")]
    if possible_targets:
        ycol = st.selectbox("Hedef sütunu seç (otomatik bulundu)", options=possible_targets)
    else:
        ycol = st.selectbox("Hedef sütunu manuel seçin", options=df.columns)
    Xcols = [c for c in feature_names if c in dff.columns]
    if not Xcols:
        st.error("feature_names ve dataset sütunları uyuşmuyor. feature_names.pkl'i kontrol et.")
        return
    X = dff[Xcols]
    y = dff[ycol]
    # Predictions on dataset (careful: big datasets may be heavy)
    try:
        y_pred = model.predict(X)
    except Exception as e:
        st.error(f"Model predict çalıştırılamadı: {e}")
        return
    # classification report
    st.subheader("Classification Report")
    try:
        report = classification_report(y, y_pred, output_dict=True, zero_division=0)
        rpt_df = pd.DataFrame(report).T
        st.dataframe(rpt_df)
    except Exception as e:
        st.error(f"Classification report oluşturulamadı: {e}")
    # confusion matrix
    try:
        cm = confusion_matrix(y, y_pred)
        fig_cm, ax = plt.subplots(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Blues")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        st.pyplot(fig_cm)
    except Exception as e:
        st.error(f"Confusion matrix çizilemedi: {e}")
    # AUC if proba available
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X)
            # multiclass
            if y_proba.ndim == 2 and y_proba.shape[1] > 2:
                try:
                    auc = roc_auc_score(pd.get_dummies(y), y_proba, multi_class='ovr')
                    st.write(f"ROC-AUC (multiclass, ovr): {auc:.3f}")
                except Exception as e:
                    st.info(f"ROC-AUC (multiclass) hesaplanamadı: {e}")
            else:
                try:
                    auc = roc_auc_score(y, y_proba[:,1])
                    st.write(f"ROC-AUC: {auc:.3f}")
                except Exception as e:
                    st.info(f"ROC-AUC hesaplanamadı: {e}")
        except Exception as e:
            st.info(f"Proba alınamadı: {e}")

def page_history():
    st.title("Tahmin Geçmişi")
    hist = st.session_state.pred_history
    if not hist:
        st.info("Henüz tahmin yok.")
        return
    dfh = pd.DataFrame(hist)
    st.dataframe(dfh)
    csvb = dfh.to_csv(index=False).encode("utf-8")
    st.download_button("Tüm geçmişi CSV indir", data=csvb, file_name="prediction_history.csv", mime="text/csv")
    if st.button("Geçmişi temizle"):
        st.session_state.pred_history = []
        st.experimental_rerun()

def page_settings():
    st.title("Ayarlar & Bilgiler")
    st.markdown("Bu sayfadan runtime'da dosya upload edebilir, repository indirme butonunu kullanabilirsiniz.")
    st.markdown("**Mevcut yüklemeler:**")
    st.write("Model yüklü:", "✅" if model is not None else "❌")
    st.write("Dataset yüklü:", f"✅ ({df.shape[0]} x {df.shape[1]})" if not df.empty else "❌")
    st.write("feature_names:", "✅" if feature_names else "❌")
    st.markdown("---")
    st.markdown("**Notlar / Hata giderme**")
    st.write("- Eğer `feature_names.pkl` ile uyuşmazlık olursa, `feature_names.pkl` dosyasını yükleyin veya CSV sütunlarını güncelleyin.")
    st.write("- PDF hatası görürseniz DejaVu font eklemeyi düşünebiliriz; şu an transliteration ile güvenli çıktı alınıyor.")

# Router
PAGES = {
    "Anasayfa": page_home,
    "Tahmin Sistemi": page_predict,
    "Tavsiye Sistemi": page_recommendation,
    "Veri Analizi & Model": page_data_analysis,
    "Tahmin Geçmişi": page_history,
    "Ayarlar": page_settings
}

st.sidebar.title("Sayfalar")
choice = st.sidebar.radio("Git", list(PAGES.keys()))
st.sidebar.markdown("---")
st.sidebar.markdown("Gelişmiş proje: SHAP, çoklu metrik, PDF/CSV indirme, upload & GitHub indirme desteği.")
PAGES[choice]()
