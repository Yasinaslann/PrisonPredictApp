# app.py
# Gerekli: streamlit, pandas, numpy, scikit-learn, catboost, plotly, shap, matplotlib, seaborn, fpdf, joblib, requests
# requirements.txt'e eklemeyi unutma.

import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import requests
import pickle
import joblib
from datetime import datetime
from fpdf import FPDF
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

st.set_page_config(page_title="Prison Risk App", layout="wide", initial_sidebar_state="expanded")

# ---------------------------
# Ayarlar: GitHub raw base
# ---------------------------
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/Yasinaslann/PrisonPredictApp/main/prison_app"
EXPECTED_FILES = {
    "dataset": "Prisongüncelveriseti.csv",
    "model": "catboost_model.pkl",
    "feature_names": "feature_names.pkl",
    "cat_features": "cat_features.pkl",
    "cat_unique_values": "cat_unique_values.pkl",
    "bool_columns": "bool_columns.pkl"
}

# ---------------------------
# Yardımcı fonksiyonlar
# ---------------------------
def ensure_file_local(fname, github_base=GITHUB_RAW_BASE):
    """Eğer fname yerelde yoksa GitHub raw'tan indirir."""
    if os.path.exists(fname):
        return True, None
    url = f"{github_base}/{fname}"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            with open(fname, "wb") as f:
                f.write(r.content)
            return True, None
        else:
            return False, f"HTTP {r.status_code} - {url}"
    except Exception as e:
        return False, str(e)

@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

@st.cache_data
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_joblib(path):
    return joblib.load(path)

def try_load_all_files():
    status = {}
    for k, fn in EXPECTED_FILES.items():
        ok, err = ensure_file_local(fn)
        status[fn] = {"ok": ok, "error": err}
    return status

def safe_model_predict(model, X):
    """Model'dan olabildiğince güvenli tahmin alır."""
    try:
        # catboost or sklearn style
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            # çıktı shape'ine göre sınıf 1'in proba'sını al
            if proba.ndim == 2 and proba.shape[1] >= 2:
                p1 = float(proba[:,1][0])
            else:
                p1 = float(proba.ravel()[0])
        else:
            p1 = None
    except Exception:
        p1 = None
    try:
        pred = int(model.predict(X)[0])
    except Exception:
        pred = None
    return pred, p1

def create_pdf_from_record(record: dict, recommendation: str):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "Prediction Report", ln=True)
    pdf.cell(0, 8, f"Tarih: {record.get('timestamp', '')}", ln=True)
    pdf.ln(4)
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 8, "Girdi Değerleri:", ln=True)
    for k, v in record.items():
        if k in ["timestamp", "pred", "proba"]: continue
        pdf.multi_cell(0, 6, f"- {k}: {v}")
    pdf.ln(4)
    pdf.cell(0, 8, f"Pred: {record.get('pred', '')}  |  Proba: {record.get('proba', '')}", ln=True)
    pdf.ln(6)
    pdf.cell(0, 8, "Öneriler:", ln=True)
    pdf.multi_cell(0, 6, recommendation)
    return pdf.output(dest="S").encode("latin-1", errors="replace")

def generate_recommendation(score, inputs):
    if score is None:
        return "Skor hesaplanamadı, öneri üretilemiyor."
    score = float(score)
    tips = []
    if score >= 0.85:
        tips.append("Yüksek risk: Acil müdahale. Yoğun destek, terapi ve istihdam programları önerilir.")
    elif score >= 0.6:
        tips.append("Orta-yüksek risk: Hedefli müdahale, eğitim ve yakın takip önerilir.")
    elif score >= 0.35:
        tips.append("Orta risk: İzleme ve mesleki eğitim faydalı olabilir.")
    else:
        tips.append("Düşük risk: Standart izleme yeterli.")
    # Feature-based rules (örnek, dataset'e göre uyarlayabilirsin)
    if isinstance(inputs, dict):
        el = str(inputs.get("education_level", "")).lower()
        if el in ["low", "none", "ilkokul", "ilköğretim"]:
            tips.append("Eğitim seviyesi düşük: Temel/mesleki eğitim programlarına yönlendirme.")
        emp = str(inputs.get("employment_status", "")).lower()
        if emp in ["unemployed", "işsiz", "yok"]:
            tips.append("İstihdam desteği: İşe yerleştirme/iş eğitimi önerilir.")
    return "\n\n".join(tips)

# ---------------------------
# Dosyaları indir ve yükle
# ---------------------------
st.sidebar.header("Repository / Dosya Kontrolü")
with st.sidebar.expander("Dosyaları GitHub'dan kontrol et ve indir"):
    if st.button("Dosyaları kontrol et & indir (varsa)"):
        stat = try_load_all_files()
        for fn, info in stat.items():
            if info["ok"]:
                st.success(f"{fn} bulundu veya indirildi.")
            else:
                st.warning(f"{fn} eksik: {info['error']}")

# attempt to load files (silently if present)
files_status = {}
for k, fn in EXPECTED_FILES.items():
    files_status[fn] = os.path.exists(fn)

# yüklemeye çalış
df = pd.DataFrame()
model = None
feature_names = None
cat_features = []
cat_unique_values = {}
bool_columns = []

# load dataset if exists (safe)
if os.path.exists(EXPECTED_FILES["dataset"]):
    try:
        df = load_csv(EXPECTED_FILES["dataset"])
    except Exception as e:
        st.sidebar.error(f"CSV yükleme hatası: {e}")

# load pickles / model
if os.path.exists(EXPECTED_FILES["model"]):
    try:
        # try joblib first then pickle
        try:
            model = load_joblib(EXPECTED_FILES["model"])
        except Exception:
            model = load_pickle(EXPECTED_FILES["model"])
    except Exception as e:
        st.sidebar.error(f"Model yükleme hatası: {e}")

for k in ["feature_names", "cat_features", "cat_unique_values", "bool_columns"]:
    fn = EXPECTED_FILES[k]
    if os.path.exists(fn):
        try:
            val = load_pickle(fn)
            if k == "feature_names":
                feature_names = val
            elif k == "cat_features":
                cat_features = val
            elif k == "cat_unique_values":
                cat_unique_values = val
            elif k == "bool_columns":
                bool_columns = val
        except Exception as e:
            st.sidebar.error(f"{fn} yüklenirken hata: {e}")

# feature_names fallback
if feature_names is None:
    # try model attribute
    try:
        if model is not None and hasattr(model, "feature_names_"):
            feature_names = list(model.feature_names_)
    except:
        feature_names = None

if feature_names is None and not df.empty:
    # try inferring target
    possible_target = [c for c in df.columns if c.lower() in ("target", "label", "y", "recidivism", "risk")]
    if len(possible_target) > 0:
        feature_names = [c for c in df.columns if c not in possible_target]
    else:
        feature_names = list(df.columns)  # worst-case: bütün sütunlar (kullanıcı düzenlemeli)

# ---------------------------
# Session state
# ---------------------------
if "pred_history" not in st.session_state:
    st.session_state.pred_history = []

# ---------------------------
# UI: Sayfalar
# ---------------------------
def page_home():
    st.title("Prison Risk App — Anasayfa")
    st.markdown("""
    Bu uygulama mahkum/denetimli kişilere yönelik **risk tahmini**, **açıklanabilirlik (SHAP)**,
    **görselleştirme** ve **kişisel öneri** sunmak için tasarlandı.
    - Model ve veriler GitHub'dan çekilebilir (repo: `Yasinaslann/PrisonPredictApp`).
    - SHAP, Plotly ve matplotlib ile zengin görselleştirme.
    """)
    st.markdown("### Yüklü dosyalar")
    for k, fn in EXPECTED_FILES.items():
        st.write(f"- {fn} — {'✔' if os.path.exists(fn) else '❌'}")
    st.markdown("---")
    st.subheader("Kullanım akışı (özet)")
    st.markdown("""
    1. **Tahmin Sistemi:** Bireysel kayıt gir -> risk skoru & SHAP göster.
    2. **Tavsiye Sistemi:** Tahmine göre kişisel aksiyon önerileri üret.
    3. **Veri Analizi & Model:** Eğitim verisinde interaktif analiz, model metrics.
    4. **Tahmin Geçmişi:** Yapılan tahminler saklanır; CSV/PDF indirilebilir.
    """)
    if st.checkbox("Verinin ilk 10 satırını göster"):
        if df.empty:
            st.warning("Veri bulunamadı.")
        else:
            st.dataframe(df.head(10))

def build_input_form(feature_list):
    st.markdown("### Girdi Formu")
    inputs = {}
    if feature_list is None:
        st.error("Feature listesi bulunmuyor — `feature_names.pkl` yoksa manuel düzenleme yap.")
        return inputs
    cols = st.columns(2)
    i = 0
    for feat in feature_list:
        c = cols[i % 2]
        label = feat
        help_text = ""
        if not df.empty and feat in df.columns:
            help_text = f"Ortalama: {df[feat].mean():.2f} | Std: {df[feat].std():.2f}" if np.issubdtype(df[feat].dtype, np.number) else f"Unique: {df[feat].nunique()}"
        if feat in cat_features:
            options = cat_unique_values.get(feat, [])
            if options:
                val = c.selectbox(label, options=options, index=0, help=help_text, key=f"inp_{feat}")
            else:
                val = c.text_input(label, help=help_text, key=f"inp_{feat}")
            inputs[feat] = val
        elif feat in bool_columns:
            val = c.checkbox(label, help=help_text, key=f"inp_{feat}")
            inputs[feat] = int(val)
        else:
            # numeric fallback
            default = 0.0
            if (not df.empty) and (feat in df.columns) and np.issubdtype(df[feat].dtype, np.number):
                default = float(df[feat].median())
            val = c.number_input(label, value=default, step=1.0, help=help_text, key=f"inp_{feat}")
            inputs[feat] = float(val)
        i += 1
    return inputs

def page_predict():
    st.title("Tahmin Sistemi")
    if feature_names is None:
        st.error("Feature listesini bulamadım. `feature_names.pkl` eksik.")
        return
    with st.form("predict_form"):
        user_inputs = build_input_form(feature_names)
        submitted = st.form_submit_button("Tahmin Et")
    if submitted:
        if model is None:
            st.error("Model yüklenemedi. `catboost_model.pkl` eksik veya bozuk.")
            return
        X_single = pd.DataFrame([user_inputs], columns=feature_names)
        pred, proba = safe_model_predict(model, X_single)
        st.metric("Tahmin (label)", pred if pred is not None else "—")
        if proba is not None:
            st.metric("Risk Skoru (0-1)", f"{proba:.3f}")
        # kaydet
        rec = {"timestamp": datetime.now().isoformat(timespec="seconds"), **user_inputs, "pred": pred, "proba": proba}
        st.session_state.pred_history.insert(0, rec)
        # SHAP açıklaması
        st.subheader("Tahmin Açıklaması (SHAP)")
        try:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_single)
            # shap_vals handling for binary classifiers
            if isinstance(shap_vals, list) and len(shap_vals) > 1:
                vals = np.array(shap_vals[1])
            else:
                vals = np.array(shap_vals)
            # bar chart (top features)
            arr = vals[0]
            df_sh = pd.DataFrame({"feature": X_single.columns, "shap_value": arr})
            df_sh["abs"] = df_sh["shap_value"].abs()
            df_sh = df_sh.sort_values("abs", ascending=False).head(15)
            fig = px.bar(df_sh[::-1], x="shap_value", y="feature", orientation="h", title="Top SHAP features (mutlak değer, ilk 15)")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("**Detay:** pozitif SHAP değeri riski artırır; negatif ise azaltır.")
            # waterfall (matplotlib fallback)
            try:
                st.subheader("Waterfall (detaylı)")
                fig_w, ax = plt.subplots(figsize=(8,4))
                shap.plots._waterfall.waterfall_legacy(explainer.expected_value if not hasattr(explainer.expected_value, "__len__") else explainer.expected_value[1],
                                                       arr, feature_names=X_single.columns, max_display=12)
                st.pyplot(fig_w)
            except Exception:
                st.info("Waterfall çizimi desteklenmiyor ortamda; bar gösterimi sağlandı.")
        except Exception as e:
            st.error(f"SHAP açıklama oluşturulamadı: {e}")
        # hızlı öneri
        st.subheader("Hızlı Öneri")
        rec_text = generate_recommendation(proba, user_inputs)
        st.info(rec_text)
        # download buttons
        if st.button("Bu tahmini PDF indir"):
            pdf_bytes = create_pdf_from_record(rec, rec_text)
            st.download_button("PDF İndir", data=pdf_bytes, file_name="prediction_report.pdf", mime="application/pdf")

def page_recommendation():
    st.title("Tavsiye Sistemi")
    if len(st.session_state.pred_history) == 0:
        st.warning("Henüz tahmin yapılmamış. Tahmin sayfasına git.")
        return
    df_hist = pd.DataFrame(st.session_state.pred_history)
    sel_index = st.selectbox("Kayıt seç", options=df_hist.index, format_func=lambda i: f"{df_hist.loc[i,'timestamp']} — {df_hist.loc[i,'proba']}")
    row = df_hist.loc[sel_index].to_dict()
    st.subheader("Seçili Kayıt")
    st.json(row)
    st.subheader("Otomatik Öneri")
    rec_text = generate_recommendation(row.get("proba"), row)
    st.info(rec_text)
    if st.button("Bu kaydı PDF indir"):
        pdf_bytes = create_pdf_from_record(row, rec_text)
        st.download_button("PDF İndir", data=pdf_bytes, file_name="recommendation_report.pdf", mime="application/pdf")

def page_data_analysis():
    st.title("Veri Analizi & Model Performansı")
    if df.empty:
        st.warning("Veri yüklenemedi.")
        return
    st.markdown(f"Dataset: {df.shape[0]} satır x {df.shape[1]} sütun")
    # basit filtreler
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    st.sidebar.subheader("Analiz Filtreleri")
    filters = {}
    for c in numeric_cols[:6]:
        mn, mx = float(df[c].min()), float(df[c].max())
        lo, hi = st.sidebar.slider(f"{c} aralığı", mn, mx, (mn, mx))
        filters[c] = (lo, hi)
    for c in cat_cols[:4]:
        options = df[c].unique().tolist()
        sel = st.sidebar.multiselect(f"{c} seç", options=options, default=options[:min(5,len(options))])
        filters[c] = sel
    dff = df.copy()
    for c, v in filters.items():
        if isinstance(v, tuple):
            dff = dff[(dff[c] >= v[0]) & (dff[c] <= v[1])]
        elif isinstance(v, list):
            dff = dff[dff[c].isin(v)]
    st.markdown(f"Filtrelenmiş: {dff.shape[0]} satır")
    # histogram
    st.subheader("Dağılımlar")
    col1, col2 = st.columns(2)
    with col1:
        sel_num = st.selectbox("Histogram için", options=numeric_cols, index=0)
        fig1 = px.histogram(dff, x=sel_num, nbins=40, marginal="box", title=f"{sel_num} dağılımı")
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        if cat_cols:
            sel_cat = st.selectbox("Kategori için", options=cat_cols, index=0)
            vc = dff[sel_cat].value_counts().reset_index()
            vc.columns = [sel_cat, "count"]
            fig2 = px.pie(vc, names=sel_cat, values="count", title=f"{sel_cat} oranları")
            st.plotly_chart(fig2, use_container_width=True)
    st.subheader("İki değişkenli analiz")
    xopt = st.selectbox("X", options=dff.columns, index=0)
    yopt = st.selectbox("Y (numeric)", options=numeric_cols, index=0)
    fig3 = px.scatter(dff, x=xopt, y=yopt, hover_data=dff.columns, title=f"{xopt} vs {yopt}")
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown("---")
    st.subheader("Model Performansı (varsa)")
    if model is None:
        st.info("Model mevcut değil; performans hesaplanamıyor.")
        return
    # target selection if present
    possible_targets = [c for c in df.columns if c.lower() in ("target", "label", "y", "recidivism", "risk")]
    if len(possible_targets) == 0:
        st.info("Hedef sütun otomatik bulunamadı. CSV'de target/label gibi bir sütun beklenir.")
        return
    ycol = st.selectbox("Hedef sütun seç", options=possible_targets)
    X = dff[[c for c in feature_names if c in dff.columns]] if feature_names is not None else dff.drop(columns=[ycol], errors="ignore")
    y = dff[ycol]
    try:
        y_pred = model.predict(X)
        try:
            y_proba = model.predict_proba(X)[:,1]
        except:
            y_proba = None
        st.write(f"Accuracy: {accuracy_score(y, y_pred):.3f}")
        st.write(f"Precision: {precision_score(y, y_pred, zero_division=0):.3f}")
        st.write(f"Recall: {recall_score(y, y_pred, zero_division=0):.3f}")
        st.write(f"F1: {f1_score(y, y_pred, zero_division=0):.3f}")
        if y_proba is not None:
            try:
                st.write(f"ROC-AUC: {roc_auc_score(y, y_proba):.3f}")
            except:
                pass
        cm = confusion_matrix(y, y_pred)
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
        st.info("Henüz tahmin yok.")
        return
    df_hist = pd.DataFrame(hist)
    st.dataframe(df_hist)
    csv = df_hist.to_csv(index=False).encode("utf-8")
    st.download_button("CSV indir", data=csv, file_name="prediction_history.csv", mime="text/csv")
    if st.button("Geçmişi temizle"):
        st.session_state.pred_history = []
        st.experimental_rerun()

# ---------------------------
# Router
# ---------------------------
PAGES = {
    "Anasayfa": page_home,
    "Tahmin Sistemi": page_predict,
    "Tavsiye Sistemi": page_recommendation,
    "Veri Analizi & Model": page_data_analysis,
    "Tahmin Geçmişi": page_history
}

st.sidebar.title("Gezinme")
sel = st.sidebar.radio("Sayfa seç", list(PAGES.keys()))
st.sidebar.markdown("---")
st.sidebar.markdown("**Notlar:**")
st.sidebar.markdown("- Dosyalar GitHub'da ise 'Dosyaları kontrol et & indir' ile çekebilirsin.")
st.sidebar.markdown("- SHAP çizimleri environment'a bağlı olarak farklılık gösterebilir; hata olursa fallback'ler var.")
PAGES[sel]()
