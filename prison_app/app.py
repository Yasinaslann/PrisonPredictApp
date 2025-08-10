# app.py
"""
Optimized, robust and visually tidy Streamlit app for Prison Risk Prediction.
- Multi-page: Home, Predict, Recommendation, Analysis, History, Settings
- Robust file/model loaders, multiclass metrics, SHAP explainability
- Unique widget keys, forms with submit buttons, PDF export (safe)
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import joblib
import requests
from io import BytesIO
from datetime import datetime
from fpdf import FPDF
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report)

# -------------------------
# Config
# -------------------------
st.set_page_config(page_title="Prison Risk — Advanced App", layout="wide", initial_sidebar_state="expanded")
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

# -------------------------
# Helpers: robust loaders
# -------------------------
def download_from_github(filename, force=False):
    local = os.path.join(LOCAL_DIR, filename)
    if os.path.exists(local) and not force:
        return True, f"Already exists: {local}"
    url = GITHUB_RAW_BASE + filename
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            with open(local, "wb") as f:
                f.write(r.content)
            return True, f"Downloaded: {local}"
        return False, f"HTTP {r.status_code} - {url}"
    except Exception as e:
        return False, str(e)

def load_pickle_flexible(path):
    """Try a few methods to read pickle-like content."""
    last_exc = None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        last_exc = e
    try:
        return pd.read_pickle(path)
    except Exception as e:
        last_exc = e
    try:
        return joblib.load(path)
    except Exception as e:
        last_exc = e
    raise RuntimeError(f"Could not load pickle/joblib: {path} | last: {last_exc}")

def load_model(path):
    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)

def load_csv(path):
    return pd.read_csv(path)

# -------------------------
# Helpers: PDF safe writing (transliterate Turkish chars)
# -------------------------
TRANSLIT = {"ç":"c","Ç":"C","ş":"s","Ş":"S","ı":"i","İ":"I","ğ":"g","Ğ":"G","ü":"u","Ü":"U","ö":"o","Ö":"O"}
def normalize_text(s):
    if s is None: return ""
    s = str(s)
    for k,v in TRANSLIT.items():
        s = s.replace(k,v)
    s = "".join(ch if ord(ch) < 256 else "?" for ch in s)
    return s

def create_pdf_bytes(record: dict, recommendation: str):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(True, margin=12)
    pdf.set_font("Arial", size=11)
    pdf.cell(0, 8, normalize_text("Prediction Report"), ln=True)
    pdf.ln(2)
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 7, normalize_text(f"Tarih: {record.get('timestamp','')}"), ln=True)
    pdf.ln(4)
    pdf.cell(0, 7, normalize_text("Girdi Değerleri:"), ln=True)
    pdf.ln(2)
    for k, v in record.items():
        if k in ("timestamp", "pred", "proba"): continue
        pdf.multi_cell(0, 6, normalize_text(f"- {k}: {v}"))
    pdf.ln(3)
    pdf.cell(0, 7, normalize_text(f"Pred: {record.get('pred','')}  |  Proba: {record.get('proba','')}"), ln=True)
    pdf.ln(6)
    pdf.multi_cell(0, 6, normalize_text("Öneriler:"))
    pdf.ln(2)
    for line in normalize_text(recommendation).split("\n"):
        pdf.multi_cell(0, 6, line)
    return pdf.output(dest="S").encode("latin-1", errors="replace")

# -------------------------
# Load files if exist locally (or let user upload)
# -------------------------
df = pd.DataFrame()
model = None
feature_names = []
cat_features = []
cat_unique_values = {}
bool_columns = []

load_errors = []

for name, fname in EXPECTED_FILES.items():
    if os.path.exists(fname):
        try:
            if fname.endswith(".csv"):
                df = load_csv(fname)
            else:
                val = load_pickle_flexible(fname)
                if name == "model":
                    model = val
                elif name == "feature_names":
                    feature_names = list(val)
                elif name == "cat_features":
                    cat_features = list(val)
                elif name == "cat_unique_values":
                    cat_unique_values = dict(val)
                elif name == "bool_columns":
                    bool_columns = list(val)
        except Exception as e:
            load_errors.append(f"{fname} load error: {e}")

# If feature_names not found, infer from df (excluding auto-detected target)
if not feature_names:
    if not df.empty:
        possible_targets = [c for c in df.columns if c.lower() in ("target","label","y","recidivism","risk")]
        if possible_targets:
            feature_names = [c for c in df.columns if c not in possible_targets]
        else:
            feature_names = df.columns.tolist()

# -------------------------
# Session state for history
# -------------------------
if "pred_history" not in st.session_state:
    st.session_state.pred_history = []

# -------------------------
# Sidebar - uploads & github
# -------------------------
st.sidebar.title("Controls & Files")
st.sidebar.markdown("Upload CSV / model / feature_names (runtime) or download from GitHub if missing.")
if st.sidebar.button("Download missing files from GitHub"):
    for fname in EXPECTED_FILES.values():
        ok, msg = download_from_github(fname, force=True)
        if ok:
            st.sidebar.success(msg)
        else:
            st.sidebar.error(msg)

uploaded_csv = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
uploaded_model = st.sidebar.file_uploader("Upload model (.pkl/.joblib) (optional)", type=["pkl","joblib"])
uploaded_features = st.sidebar.file_uploader("Upload feature_names.pkl (optional)", type=["pkl"])
if uploaded_csv is not None:
    try:
        df = pd.read_csv(uploaded_csv)
        st.sidebar.success("CSV loaded into runtime.")
    except Exception as e:
        st.sidebar.error(f"CSV upload error: {e}")
if uploaded_model is not None:
    try:
        tmp = os.path.join(LOCAL_DIR, "temp_model_upload")
        with open(tmp, "wb") as f:
            f.write(uploaded_model.getvalue())
        model = load_model(tmp)
        os.remove(tmp)
        st.sidebar.success("Model loaded to runtime.")
    except Exception as e:
        st.sidebar.error(f"Model upload error: {e}")
if uploaded_features is not None:
    try:
        tmpf = os.path.join(LOCAL_DIR, "temp_features.pkl")
        with open(tmpf, "wb") as f:
            f.write(uploaded_features.getvalue())
        feature_names = list(load_pickle_flexible(tmpf))
        os.remove(tmpf)
        st.sidebar.success("feature_names loaded to runtime.")
    except Exception as e:
        st.sidebar.error(f"feature_names upload error: {e}")

st.sidebar.markdown("---")
threshold = st.sidebar.slider("Binary threshold (for classifying by proba)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

if load_errors:
    with st.sidebar.expander("Load errors (details)"):
        for le in load_errors:
            st.write(f"- {le}")

# -------------------------
# Utility: align user input to feature order
# -------------------------
def aligned_input_df(user_inputs: dict, feature_list: list, df_sample: pd.DataFrame=None):
    X = pd.DataFrame([user_inputs])
    for f in feature_list:
        if f not in X.columns:
            if df_sample is not None and f in df_sample.columns:
                if np.issubdtype(df_sample[f].dtype, np.number):
                    X[f] = float(df_sample[f].median())
                else:
                    X[f] = df_sample[f].mode().iloc[0] if not df_sample[f].mode().empty else ""
            else:
                X[f] = 0
    X = X[feature_list]
    return X

# -------------------------
# UI components / pages
# -------------------------
def page_home():
    st.title("Prison Risk — Advanced Prediction App")
    st.markdown("""
    **Overview:** Risk prediction + explainability + recommendations.
    - Use sidebar to upload files or download from GitHub.
    - Predict page: fill inputs, press submit.
    - Analysis: interactive visuals & model metrics.
    """)
    st.markdown("### Current files")
    cols = st.columns(2)
    with cols[0]:
        st.write(f"- Dataset: {'✔' if not df.empty else '❌'}")
        st.write(f"- Model: {'✔' if model is not None else '❌'}")
    with cols[1]:
        st.write(f"- feature_names: {'✔' if feature_names else '❌'}")
        st.write(f"- cat_features: {'✔' if cat_features else '❌'}")
    st.markdown("---")
    if not df.empty:
        st.subheader("Data preview")
        st.dataframe(df.head(8))
        st.markdown(f"Rows / Cols: **{df.shape[0]} x {df.shape[1]}**")
        if st.button("Download sample CSV"):
            st.download_button("Download sample", data=df.head(100).to_csv(index=False).encode("utf-8"), file_name="sample.csv", mime="text/csv")
    else:
        st.warning("Dataset not loaded. Upload or download from GitHub via sidebar.")

def build_inputs(feature_list, page_prefix="predict"):
    """Create neat inputs grouped; ensure unique keys using page_prefix."""
    inputs = {}
    if not feature_list:
        return inputs
    # classify features
    numeric = [f for f in feature_list if f not in cat_features and f not in bool_columns]
    cats = [f for f in feature_list if f in cat_features]
    bools = [f for f in feature_list if f in bool_columns]
    exp_num = st.expander(f"Numeric features ({len(numeric)})", expanded=True)
    exp_cat = st.expander(f"Categorical features ({len(cats)})", expanded=False)
    exp_bool = st.expander(f"Boolean features ({len(bools)})", expanded=False)
    cols = exp_num.columns(2)
    for i, f in enumerate(numeric):
        c = cols[i%2]
        default = 0.0
        if not df.empty and f in df.columns and np.issubdtype(df[f].dtype, np.number):
            default = float(df[f].median())
        key = f"{page_prefix}_num_{f}"
        val = c.number_input(f"{f}", value=default, key=key)
        inputs[f] = val
    for f in cats:
        opts = cat_unique_values.get(f, [])
        key = f"{page_prefix}_cat_{f}"
        if opts:
            val = exp_cat.selectbox(f"{f}", options=opts, index=0, key=key)
        else:
            val = exp_cat.text_input(f"{f}", key=key)
        inputs[f] = val
    for f in bools:
        key = f"{page_prefix}_bool_{f}"
        val = exp_bool.selectbox(f"{f}", options=[0,1], index=0, key=key)
        inputs[f] = int(val)
    return inputs

def page_predict():
    st.title("Tahmin Sistemi")
    if not feature_names:
        st.error("feature_names missing; cannot build form.")
        return
    with st.form("predict_form"):
        user_inputs = build_inputs(feature_names, page_prefix="predict")
        submitted = st.form_submit_button("Tahmin Et")
    if submitted:
        if model is None:
            st.error("Model not loaded.")
            return
        X = aligned_input_df(user_inputs, feature_names, df_sample=df if not df.empty else None)
        # predict
        pred = None; proba = None
        try:
            pred = model.predict(X)
            pred = int(np.ravel(pred)[0])
        except Exception:
            pred = None
        try:
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)
                if probs.ndim == 2 and probs.shape[1] >= 2:
                    proba = float(probs[:,1][0])  # class 1 proba
                else:
                    proba = None
        except Exception:
            proba = None
        st.metric("Predicted label", pred if pred is not None else "—")
        if proba is not None:
            st.metric("Risk score (proba)", f"{proba:.4f}")
        # show class probs if multiclass
        if hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba(X)
                if probs.ndim == 2 and probs.shape[1] > 2:
                    p_df = pd.DataFrame(probs, columns=[f"class_{i}" for i in range(probs.shape[1])])
                    st.subheader("Class probabilities")
                    st.dataframe(p_df.T)
            except Exception:
                pass
        # SHAP explanation
        st.subheader("SHAP Explanation")
        try:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X)
            if isinstance(shap_vals, list) and len(shap_vals) > 1:
                cls_idx = st.selectbox("Select class for SHAP (multiclass)", options=list(range(len(shap_vals))), key="shap_class_select")
                arr = np.array(shap_vals[cls_idx])[0]
            else:
                arr = np.array(shap_vals)[0] if not isinstance(shap_vals, list) else np.array(shap_vals)[0]
            df_sh = pd.DataFrame({"feature": X.columns, "shap_value": arr})
            df_sh["abs"] = df_sh["shap_value"].abs()
            df_sh = df_sh.sort_values("abs", ascending=False).head(20)
            fig = px.bar(df_sh[::-1], x="shap_value", y="feature", orientation="h", title="Top SHAP features")
            st.plotly_chart(fig, use_container_width=True)
            top_feats = df_sh.feature.tolist()[:6]
        except Exception as e:
            st.error(f"SHAP error: {e}")
            top_feats = None
        # recommendations
        rec_text = generate_recommendations(proba, top_feats, user_inputs)
        st.subheader("Personalized recommendation")
        st.info(rec_text)
        # save history
        rec = {"timestamp": datetime.now().isoformat(timespec="seconds"), **user_inputs, "pred": pred, "proba": proba}
        st.session_state.pred_history.insert(0, rec)
        # downloads
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("Download this prediction (CSV)", data=pd.DataFrame([rec]).to_csv(index=False).encode("utf-8"),
                               file_name="prediction.csv", mime="text/csv")
        with c2:
            try:
                pdfb = create_pdf_bytes(rec, rec_text)
                st.download_button("Download this prediction (PDF)", data=pdfb, file_name="prediction.pdf", mime="application/pdf")
            except Exception as e:
                st.error(f"PDF error: {e}")

def page_recommendation():
    st.title("Tavsiye Sistemi")
    if len(st.session_state.pred_history) == 0:
        st.info("No predictions yet.")
        return
    dfh = pd.DataFrame(st.session_state.pred_history)
    idx = st.selectbox("Select record", options=dfh.index, format_func=lambda i: f"{dfh.loc[i,'timestamp']} — {dfh.loc[i,'proba']}")
    row = dfh.loc[idx].to_dict()
    st.subheader("Selected record")
    st.json(row)
    rec_text = generate_recommendations(row.get("proba"), None, row)
    st.subheader("Recommendations")
    st.info(rec_text)
    if st.button("Download selected as PDF"):
        try:
            pdfb = create_pdf_bytes(row, rec_text)
            st.download_button("PDF", data=pdfb, file_name="record_recommendation.pdf", mime="application/pdf")
        except Exception as e:
            st.error(f"PDF error: {e}")

def page_analysis():
    st.title("Veri Analizi & Model Performance")
    if df.empty:
        st.warning("Dataset not loaded.")
        return
    st.markdown(f"Dataset shape: **{df.shape[0]} x {df.shape[1]}**")
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(exclude=[np.number]).columns.tolist()
    st.sidebar.subheader("Quick filters")
    filters = {}
    for c in numeric[:6]:
        mn, mx = float(df[c].min()), float(df[c].max())
        filters[c] = st.sidebar.slider(f"{c}", mn, mx, (mn, mx))
    for c in categorical[:4]:
        opts = df[c].unique().tolist()
        filters[c] = st.sidebar.multiselect(f"{c}", options=opts, default=opts[:min(5,len(opts))])
    dff = df.copy()
    for c, v in filters.items():
        if isinstance(v, tuple):
            dff = dff[(dff[c] >= v[0]) & (dff[c] <= v[1])]
        elif isinstance(v, list):
            dff = dff[dff[c].isin(v)]
    st.markdown(f"Filtered rows: **{dff.shape[0]}**")
    c1, c2 = st.columns(2)
    with c1:
        if numeric:
            sel = st.selectbox("Histogram (numeric)", options=numeric)
            fig = px.histogram(dff, x=sel, nbins=40, marginal="box")
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        if categorical:
            selc = st.selectbox("Category distribution", options=categorical)
            vc = dff[selc].value_counts().reset_index()
            vc.columns = [selc, "count"]
            fig2 = px.pie(vc, names=selc, values="count")
            st.plotly_chart(fig2, use_container_width=True)
    st.markdown("---")
    st.subheader("Model performance (if model & target available)")
    if model is None:
        st.info("Model not loaded.")
        return
    possible_targets = [c for c in df.columns if c.lower() in ("target","label","y","recidivism","risk")]
    if possible_targets:
        ycol = st.selectbox("Target column (auto)", options=possible_targets)
    else:
        ycol = st.selectbox("Select target column (manual)", options=df.columns)
    Xcols = [c for c in feature_names if c in dff.columns]
    if not Xcols:
        st.error("feature_names mismatch with dataset columns.")
        return
    X = dff[Xcols]
    y = dff[ycol]
    try:
        y_pred = model.predict(X)
    except Exception as e:
        st.error(f"Model predict failed: {e}")
        return
    # classification report (robust for multiclass)
    st.subheader("Classification report")
    try:
        report = classification_report(y, y_pred, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).T)
    except Exception as e:
        st.error(f"Could not build classification report: {e}")
    # confusion matrix
    try:
        cm = confusion_matrix(y, y_pred)
        fig_cm, ax = plt.subplots(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Blues")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        st.pyplot(fig_cm)
    except Exception as e:
        st.error(f"Confusion matrix error: {e}")
    # metrics: handle multiclass properly (weighted/macro)
    try:
        if len(np.unique(y)) > 2:
            st.write("Multiclass metrics (macro / weighted):")
            st.write(f"Accuracy: {accuracy_score(y, y_pred):.3f}")
            st.write(f"Precision (macro): {precision_score(y, y_pred, average='macro', zero_division=0):.3f}")
            st.write(f"Recall (macro): {recall_score(y, y_pred, average='macro', zero_division=0):.3f}")
            st.write(f"F1 (macro): {f1_score(y, y_pred, average='macro', zero_division=0):.3f}")
        else:
            st.write("Binary metrics:")
            st.write(f"Accuracy: {accuracy_score(y, y_pred):.3f}")
            st.write(f"Precision: {precision_score(y, y_pred, zero_division=0):.3f}")
            st.write(f"Recall: {recall_score(y, y_pred, zero_division=0):.3f}")
            st.write(f"F1: {f1_score(y, y_pred, zero_division=0):.3f}")
    except Exception as e:
        st.error(f"Metrics error: {e}")
    # ROC-AUC if proba exists
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X)
            if y_proba.ndim == 2 and y_proba.shape[1] > 2:
                auc = roc_auc_score(pd.get_dummies(y), y_proba, multi_class='ovr')
                st.write(f"ROC-AUC (ovr): {auc:.3f}")
            else:
                auc = roc_auc_score(y, y_proba[:,1])
                st.write(f"ROC-AUC: {auc:.3f}")
        except Exception as e:
            st.info(f"AUC couldn't be computed: {e}")

def page_history():
    st.title("Tahmin Geçmişi")
    hist = st.session_state.pred_history
    if not hist:
        st.info("No predictions yet.")
        return
    dfh = pd.DataFrame(hist)
    st.dataframe(dfh)
    st.download_button("Download history CSV", data=dfh.to_csv(index=False).encode("utf-8"), file_name="history.csv", mime="text/csv")
    if st.button("Clear history"):
        st.session_state.pred_history = []
        st.experimental_rerun()

def page_settings():
    st.title("Settings & Info")
    st.markdown("Upload files or use GitHub download. If you face PDF encoding issues, we transliterate turkish characters; for full UTF-8 PDF we can embed DejaVu TTF.")
    st.markdown("Current status:")
    st.write("Dataset loaded:", not df.empty)
    st.write("Model loaded:", model is not None)
    st.write("Feature_names loaded:", bool(feature_names))
    if load_errors:
        with st.expander("Load errors"):
            for le in load_errors:
                st.write(le)

# -------------------------
# Small helpers used in predict page
# -------------------------
def generate_recommendations(score, top_shap=None, inputs=None):
    if score is None:
        return "Score not available — cannot create recommendation."
    text = []
    s = float(score)
    if s >= 0.85:
        text.append("High risk: immediate intervention recommended (therapy, intensive supervision).")
    elif s >= 0.6:
        text.append("Moderate-high risk: targeted education / job support and supervision recommended.")
    elif s >= 0.35:
        text.append("Moderate risk: vocational training and monitoring suggested.")
    else:
        text.append("Low risk: standard integration programs and monitoring.")
    if top_shap:
        feats = [f.lower() for f in top_shap]
        if any("education" in f or "okul" in f for f in feats):
            text.append("Education-related features are important: consider basic or vocational education.")
        if any("employ" in f or "job" in f or "iş" in f for f in feats):
            text.append("Employment-related: provide job training and placement.")
        if any("drug" in f or "madde" in f for f in feats):
            text.append("Substance use related: consider addiction treatment programs.")
    if inputs:
        if "education_level" in inputs and str(inputs.get("education_level","")).lower() in ("low","none"):
            text.append("User education level low: recommend basic education courses.")
        if "employment_status" in inputs and str(inputs.get("employment_status","")).lower() in ("unemployed","işsiz","yok"):
            text.append("User unemployed: recommend employment programs.")
    return "\n\n".join(text)

# -------------------------
# Router
# -------------------------
PAGES = {
    "Home": page_home,
    "Predict": page_predict,
    "Recommendation": page_recommendation,
    "Analysis": page_analysis,
    "History": page_history,
    "Settings": page_settings
}

st.sidebar.title("Navigation")
choice = st.sidebar.radio("Go to", list(PAGES.keys()))
st.sidebar.markdown("---")
st.sidebar.markdown("Tip: If something fails, paste the Streamlit log here and I'll help fix it.")
PAGES[choice]()
