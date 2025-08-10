# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import os
import io
from datetime import datetime
from fpdf import FPDF

import shap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)

st.set_page_config(page_title="Prison Risk App — Advanced ML Demo", layout="wide")

# -------------------------
# Helper / Loader functions
# -------------------------
@st.cache_data
def load_dataset(csv_path="Prisongüncelveriseti.csv"):
    df = pd.read_csv(csv_path)
    return df

@st.cache_data
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_joblib(path):
    return joblib.load(path)

def safe_loads():
    errors = []
    base = os.getcwd()
    # expected files
    files = {
        "dataset": "Prisongüncelveriseti.csv",
        "model": "catboost_model.pkl",
        "feature_names": "feature_names.pkl",
        "cat_features": "cat_features.pkl",
        "cat_unique_values": "cat_unique_values.pkl",
        "bool_columns": "bool_columns.pkl"
    }
    data = {}
    for k, fn in files.items():
        if not os.path.exists(fn):
            errors.append(f"Dosya eksik: {fn}")
        else:
            try:
                if fn.endswith(".pkl"):
                    data[k] = load_pickle(fn)
                elif fn.endswith(".joblib"):
                    data[k] = load_joblib(fn)
                elif fn.endswith(".csv"):
                    data[k] = load_dataset(fn)
                else:
                    data[k] = load_pickle(fn)
            except Exception as e:
                errors.append(f"Hata yükleme {fn}: {e}")
    return data, errors

data_bundle, load_errors = safe_loads()

if load_errors:
    st.sidebar.error("Bazı dosyalar eksik veya yüklenemedi. Konsolu kontrol et.")
    for e in load_errors:
        st.sidebar.text(e)

# required pieces (attempt silent defaults)
df = data_bundle.get("dataset", pd.DataFrame())
model = data_bundle.get("model", None)
feature_names = data_bundle.get("feature_names", None)
cat_features = data_bundle.get("cat_features", [])
cat_unique_values = data_bundle.get("cat_unique_values", {})
bool_columns = data_bundle.get("bool_columns", [])

# if model loaded as joblib or pickle
if model is None:
    # try joblib fallback
    try:
        model = load_joblib("catboost_model.pkl")
    except:
        model = None

# -------------------------
# UI Utilities
# -------------------------
def format_float(x):
    try:
        return float(x)
    except:
        return np.nan

def get_feature_ordered_input(feature_list, cat_feats, cat_vals, bool_cols):
    """Return dict of user inputs in order of feature_list"""
    inputs = {}
    st.markdown("### Girdi Alanları")
    cols = st.columns(2)
    i = 0
    for feat in feature_list:
        c = cols[i % 2]
        with c:
            label = feat
            help_text = ""
            # get info from dataset if exists
            if not df.empty and feat in df.columns:
                help_text = f"Örnek: mean={df[feat].mean():.2f}, std={df[feat].std():.2f}"
            if feat in cat_feats:
                vals = cat_vals.get(feat, [])
                if len(vals) > 0:
                    val = c.selectbox(label, options=vals, index=0, help=help_text, key=f"inp_{feat}")
                else:
                    val = c.text_input(label, help=help_text, key=f"inp_{feat}")
                inputs[feat] = val
            elif feat in bool_cols:
                val = c.checkbox(label, help=help_text, key=f"inp_{feat}")
                inputs[feat] = int(val)
            else:
                # numeric
                sample_min = None; sample_max = None; sample_mean = None
                if (not df.empty) and (feat in df.columns):
                    sample_min = df[feat].min()
                    sample_max = df[feat].max()
                    sample_mean = df[feat].median()
                v = c.number_input(label, value= sample_mean if sample_mean is not None else 0.0,
                                   step=1.0, help=help_text, key=f"inp_{feat}")
                inputs[feat] = format_float(v)
        i += 1
    return inputs

def predict_single(model, feature_order, inp_dict):
    # return probability and raw prediction if possible
    X = pd.DataFrame([inp_dict], columns=feature_order)
    try:
        proba = model.predict_proba(X)[:,1][0]
    except Exception:
        # fallback: catboost Pool or model.predict_proba mismatch
        try:
            proba = float(model.predict_proba(X)[0][1])
        except Exception as e:
            st.warning(f"Model prediction error: {e}")
            proba = None
    try:
        pred = int(model.predict(X)[0])
    except Exception:
        pred = None
    return pred, proba

def shap_explain_single(model, X_df):
    # Build explainer and compute shap values
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_df)
        return shap_values, explainer
    except Exception as e:
        st.error(f"SHAP açıklama oluşturulamadı: {e}")
        return None, None

# -------------------------
# Session state for history
# -------------------------
if "pred_history" not in st.session_state:
    st.session_state["pred_history"] = []

# -------------------------
# Page functions
# -------------------------
def page_home():
    st.title("Prison Risk — Proje Anasayfa")
    st.markdown("""
    **Proje Hedefi:** Mahkum/denetimli kişilerin yeniden suça yönelme (recidivism) veya risk skorlarını tahmin ederek,
    kişiye özel müdahale ve tavsiyeler sağlamak. Bu uygulama ileri düzey görselleştirme, öngörü ve açıklanabilirlik (SHAP) içerir.
    """)
    st.markdown("#### Dosyalar ve model")
    st.write("Yüklenen dosyalar (bulunuyorsa):")
    for p in ["Prisongüncelveriseti.csv", "catboost_model.pkl", "feature_names.pkl", "cat_features.pkl", "cat_unique_values.pkl", "bool_columns.pkl"]:
        st.write(f"- {p} — {'✔' if os.path.exists(p) else '❌'}")
    st.markdown("---")
    st.markdown("## Proje Hikayesi & Veri Seti")
    if not df.empty:
        st.markdown(f"- Satır: **{df.shape[0]}**, Sütun: **{df.shape[1]}**")
        st.markdown("Kısa veri hikayesi: veride demografik özellikler, gözetim/denetimli program bilgileri ve geçmiş ihlal/başarı kayıtları bulunuyor. Amaç risk skorunu tahmin edip müdahale önerileri üretmek.")
        if st.checkbox("Veriyi göster (ilk 50 satır)"):
            st.dataframe(df.head(50))
    else:
        st.warning("Veri seti yüklenemedi. CSV dosyasının proje dizininde olduğundan emin ol.")
    st.markdown("---")
    st.markdown("## Kullanım Notları")
    st.markdown("""
    - Tahmin sayfasından bireysel örnekler girebilir, model tahmini ve SHAP açıklamalarını görebilirsiniz.  
    - Veri Analizi sayfasında interaktif grafiklerle dağılımları keşfedebilirsiniz.  
    - Tavsiye sayfası tahmine göre kişiselleştirilmiş aksiyonlar verir.  
    - Tahmin geçmişini kaydedebilir ve CSV olarak indirebilirsiniz.
    """)

def page_predict():
    st.title("Tahmin Sistemi")
    st.markdown("Buraya bir örnek girerek modelin risk skorunu ve açıklamalarını görün.")
    if feature_names is None:
        st.error("feature_names.pkl yüklenemedi; tahmin formu oluşturulamıyor.")
        return
    # build input form
    with st.form("predict_form", clear_on_submit=False):
        inputs = get_feature_ordered_input(feature_names, cat_features, cat_unique_values, bool_columns)
        submitted = st.form_submit_button("Tahmin Et")
    if submitted:
        if model is None:
            st.error("Model yüklenemedi.")
            return
        # prepare and predict
        try:
            pred, proba = predict_single(model, feature_names, inputs)
        except Exception as e:
            st.error(f"Tahmin sırasında hata: {e}")
            return
        st.metric("Predicted Label", pred if pred is not None else "—")
        if proba is not None:
            st.metric("Risk Skoru (0-1)", f"{proba:.3f}")
        # save to history
        record = {
            "timestamp": datetime.now().isoformat(timespec='seconds'),
            **inputs,
            "pred": pred,
            "proba": proba
        }
        st.session_state.pred_history.insert(0, record)
        # SHAP explanation
        st.subheader("Tahmin Açıklaması (SHAP)")
        X_single = pd.DataFrame([inputs], columns=feature_names)
        shap_vals, explainer = shap_explain_single(model, X_single)
        if shap_vals is not None:
            # shap_values might be array-like; handle classification with two outputs
            try:
                # For binary classifier shap_values[1] often corresponds to class 1
                if isinstance(shap_vals, list) or (hasattr(shap_vals, "__len__") and len(shap_vals) > 1):
                    sv = shap_vals[1]
                else:
                    sv = shap_vals
                # summary bar (feature contributions)
                # create a small dataframe of abs contributions
                vals = sv[0]
                contrib = pd.DataFrame({
                    "feature": X_single.columns,
                    "shap_value": vals
                }).sort_values(by="shap_value", key=lambda s: np.abs(s), ascending=False)
                fig = px.bar(contrib.head(15)[::-1], x="shap_value", y="feature", orientation="h",
                             title="En önemli feature'lar (mutlak SHAP değeri, ilk 15)")
                st.plotly_chart(fig, use_container_width=True)
                # Provide readable explanation text
                top = contrib.head(5)
                st.markdown("**Top 5 etkileyici özellik (pozitif => riski artırır, negatif => riski azaltır):**")
                for _, r in top.iterrows():
                    val = r["shap_value"]
                    sign = "artırıyor" if val > 0 else "azaltıyor"
                    st.write(f"- **{r['feature']}**: {val:.3f} ({sign})")
                # Optional: show force plot as HTML
                try:
                    shap_html = shap.plots.force(explainer.expected_value[1] if hasattr(explainer.expected_value, "__len__") else explainer.expected_value,
                                                 sv, X_single, matplotlib=False, show=False)
                    # shap.plots.force returns a JS/HTML object that shap provides via shap.plots._force_matplotlib
                    # Fallback: show waterfall
                except Exception:
                    # waterfall
                    try:
                        st.pyplot(shap.plots._waterfall.waterfall_legacy(explainer.expected_value if not hasattr(explainer.expected_value, "__len__") else explainer.expected_value[1], sv[0], feature_names=X_single.columns))
                    except Exception:
                        pass
            except Exception as e:
                st.error(f"SHAP çizim hatası: {e}")
        # show advice snippet inline
        st.subheader("Hızlı Öneri (Model Score'a Göre)")
        advice = generate_recommendation(proba, inputs if isinstance(inputs, dict) else {})
        st.info(advice)

def generate_recommendation(score, inputs):
    if score is None:
        return "Skor hesaplanamadı, öneri üretilemiyor."
    # example rule-based recommendations; bu kısımları daha gelişmiş kurallarla genişletebilirsin
    score = float(score)
    tips = []
    if score >= 0.85:
        tips.append("Yüksek risk: Acil müdahale önerilir. Yoğun denetimli program, psikolojik destek ve mesleki eğitim planlanmalı.")
        tips.append("Aile destek ağları ve sosyal hizmetler devreye alınmalı.")
    elif score >= 0.6:
        tips.append("Orta-yüksek risk: Hedefli eğitim ve davranışsal terapi faydalı olabilir.")
        tips.append("Sık takip ve küçük ölçekli denetim programları önerilir.")
    elif score >= 0.35:
        tips.append("Orta risk: İzleme ve mesleki/vocational eğitim önerilir.")
    else:
        tips.append("Düşük risk: Standart risk azaltma programları ile izleme yeterli olabilir.")
    # feature based suggestions (örnek)
    if inputs:
        if "education_level" in inputs and str(inputs.get("education_level")).lower() in ["low", "none", "ilkokul"]:
            tips.append("Eğitim seviyesi düşük gözüküyor — mesleki veya temel eğitim programları öner.")
        if "employment_status" in inputs and str(inputs.get("employment_status")).lower() in ["unemployed", "işsiz"]:
            tips.append("İstihdam desteği ve iş eğitim programlarına yönlendir.")
    return "\n\n".join(tips)

def page_recommendation():
    st.title("Tavsiye Sistemi (Risk'e Göre Öneriler)")
    st.markdown("Burada tahmine dayalı otomatik tavsiyeleri ve nasıl üretildiğini görebilirsin.")
    st.markdown("### Tahmin geçmişinden bir kayıt seç ve öneri al")
    if len(st.session_state.pred_history) == 0:
        st.warning("Henüz tahmin yapılmamış. Tahmin sayfasından bir örnek tahmin et.")
        return
    df_hist = pd.DataFrame(st.session_state.pred_history)
    sel = st.selectbox("Tahmin Geçmişinden Kayıt Seç", options=df_hist.index, format_func=lambda i: f"{df_hist.loc[i,'timestamp']} — {df_hist.loc[i,'proba']}")
    row = df_hist.loc[sel]
    st.json(row.to_dict())
    st.subheader("Otomatik Öneriler")
    rec = generate_recommendation(row.get("proba"), row.to_dict())
    st.info(rec)
    if st.button("Bu kaydı PDF olarak indir"):
        pdf_bytes = create_report_pdf(row.to_dict(), rec)
        st.download_button("Raporu İndir (PDF)", data=pdf_bytes, file_name="prediction_report.pdf", mime="application/pdf")

def create_report_pdf(record: dict, recommendation: str):
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
    pdf.ln(4)
    pdf.cell(0, 8, "Öneriler:", ln=True)
    pdf.multi_cell(0, 6, recommendation)
    out = pdf.output(dest='S').encode('latin-1')
    return out

def page_data_analysis():
    st.title("Veri Analizi ve Model Performansı")
    st.markdown("Etkileşimli görselleştirme paneli — filtrele, keşfet, model değerlendirmesi yap.")
    if df.empty:
        st.warning("Veri yüklenemedi.")
        return
    # Filters
    st.sidebar.markdown("### Veri Filtreleri (Analiz)")
    filter_cols = [c for c in df.columns if df[c].dtype in [np.number, np.float64, np.int64] or df[c].nunique() < 30]
    selected_filters = {}
    for c in filter_cols[:6]:  # show first few filters to keep UI temiz
        if df[c].dtype in [np.number, np.float64, np.int64]:
            mn, mx = float(df[c].min()), float(df[c].max())
            rng = st.sidebar.slider(f"{c} aralığı", mn, mx, (mn, mx))
            selected_filters[c] = rng
        else:
            vals = df[c].unique().tolist()
            sel = st.sidebar.multiselect(f"{c} seç", options=vals, default=vals[:min(5, len(vals))])
            selected_filters[c] = sel
    # apply filters
    dff = df.copy()
    for c, v in selected_filters.items():
        if isinstance(v, tuple) and len(v) == 2:
            dff = dff[(dff[c] >= v[0]) & (dff[c] <= v[1])]
        elif isinstance(v, list):
            dff = dff[dff[c].isin(v)]
    st.markdown(f"Filtrelenmiş satır sayısı: **{dff.shape[0]}**")
    # Visualizations
    st.subheader("Değişken Dağılımları")
    col1, col2 = st.columns(2)
    with col1:
        feat_x = st.selectbox("Histogram için değişken", options=dff.select_dtypes(include=[np.number]).columns.tolist())
        fig1 = px.histogram(dff, x=feat_x, nbins=40, title=f"{feat_x} dağılımı", marginal="box")
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        cat_x = st.selectbox("Sütun (sınıf oranları)", options=dff.select_dtypes(exclude=[np.number]).columns.tolist())
        counts = dff[cat_x].value_counts().reset_index()
        counts.columns = [cat_x, "count"]
        fig2 = px.pie(counts, names=cat_x, values="count", title=f"{cat_x} sınıf oranları")
        st.plotly_chart(fig2, use_container_width=True)
    st.subheader("İki değişkenli analiz")
    xcol = st.selectbox("X ekseni", options=dff.columns, index=0)
    ycol = st.selectbox("Y ekseni", options=dff.select_dtypes(include=[np.number]).columns.tolist(), index=0)
    fig3 = px.scatter(dff, x=xcol, y=ycol, color=dff.columns[0] if dff.columns[0] in dff.columns else None, hover_data=dff.columns)
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown("---")
    st.subheader("Model Performansı (Eğitim verisi üzerinde)")
    if model is None:
        st.warning("Model yüklenmemiş; performans hesaplanamıyor.")
        return
    # Try to evaluate on training data (if target column present)
    target_candidates = [c for c in df.columns if c.lower() in ["target", "label", "y", "recidivism", "risk"]]
    if len(target_candidates) == 0:
        st.info("Veride hedef sütun tespit edilemedi; manuel seç veya model performansı atla.")
    else:
        ycol = st.selectbox("Hedef sütunu seç (performans için)", options=target_candidates)
        X = df[feature_names] if feature_names is not None and all([f in df.columns for f in feature_names]) else df.drop(columns=[ycol])
        y = df[ycol]
        try:
            y_pred = model.predict(X)
            y_proba = model.predict_proba(X)[:,1]
            st.write("**Metrics**")
            st.write(f"Accuracy: {accuracy_score(y, y_pred):.3f}")
            st.write(f"Precision: {precision_score(y, y_pred, zero_division=0):.3f}")
            st.write(f"Recall: {recall_score(y, y_pred, zero_division=0):.3f}")
            st.write(f"F1: {f1_score(y, y_pred, zero_division=0):.3f}")
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
    st.title("Tahmin Geçmişi ve Kayıtlar")
    st.markdown("Kullanıcının yaptığı tahminler burada tutulur; CSV indir, tek tek sil ya da temizle.")
    hist = st.session_state.get("pred_history", [])
    if len(hist) == 0:
        st.info("Henüz kayıt yok.")
        return
    df_hist = pd.DataFrame(hist)
    st.dataframe(df_hist)
    csv = df_hist.to_csv(index=False).encode('utf-8')
    st.download_button("Tahmin Geçmişini CSV olarak indir", data=csv, file_name="prediction_history.csv", mime="text/csv")
    if st.button("Tüm geçmişi temizle"):
        st.session_state.pred_history = []
        st.experimental_rerun()

# -------------------------
# Page routing (sidebar)
# -------------------------
pages = {
    "Anasayfa": page_home,
    "Tahmin Sistemi": page_predict,
    "Tavsiye Sistemi": page_recommendation,
    "Veri Analizi & Model": page_data_analysis,
    "Tahmin Geçmişi": page_history
}
st.sidebar.title("Gezinme")
page = st.sidebar.radio("Sayfalar", list(pages.keys()))
st.sidebar.markdown("---")
st.sidebar.markdown("**Projeyi öne çıkaracak notlar:**")
st.sidebar.markdown("- SHAP ile explainability\n- Plotly ile interaktif grafikler\n- CSV/PDF indirme, history\n- Kişiye özel tavsiye motoru")
# run selected page
pages[page]()
