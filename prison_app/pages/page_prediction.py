# prison_app/pages/2_🤖_Tahmin_Modeli.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

st.set_page_config(page_title="Tahmin Modeli", page_icon="🤖")

BASE = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE / "catboost_model.pkl"
FEATURES_PATH = BASE / "feature_names.pkl"
BOOL_COLS = BASE / "bool_columns.pkl"
CAT_FEATURES = BASE / "cat_features.pkl"
CAT_UNIQUES = BASE / "cat_unique_values.pkl"

st.title("🤖 Tahmin Modeli")
st.markdown("Bireysel veriyi girerek tekrar suç tahmini alabilirsiniz. Tüm alanları doldurarak `Tahmin Al` butonuna basın.")

# yükle dataset sample (opsiyonel)
sample_df = None
try:
    sample_df = pd.read_csv(BASE / "Prisongüncelveriseti.csv")
except Exception:
    pass

if sample_df is not None:
    st.info("Veri setinden türetilen giriş formu için örnek sütunlar kullanılıyor.")
    columns = sample_df.columns.tolist()
else:
    # fallback generic features
    columns = ["suç_tipi","ceza_ay","egitim_durumu","gecmis_suc_sayisi","il"]

# Dinamik form: eğer feature_names.pkl varsa ona göre, yoksa yukarıdaki columns
feature_names = None
try:
    with open(FEATURES_PATH,"rb") as f:
        feature_names = pickle.load(f)
except Exception:
    feature_names = columns

# Build form with sensible defaults
with st.form("giris_formu"):
    inputs = {}
    for feat in feature_names:
        # heuristic types
        if "ay" in feat or "sayi" in feat or "gecmis" in feat or "ceza" in feat:
            inputs[feat] = st.number_input(feat, min_value=0, max_value=1000, value=0)
        elif "or" in feat or feat.lower().startswith("is_") or feat.lower().startswith("has_"):
            inputs[feat] = st.selectbox(feat, [0,1])
        else:
            # categorical / text
            inputs[feat] = st.text_input(feat, value="")

    submitted = st.form_submit_button("🔎 Tahmin Al")

if submitted:
    # prepare input vector
    X = pd.DataFrame([inputs])
    model = None
    try:
        with open(MODEL_PATH,"rb") as f:
            model = pickle.load(f)
    except Exception as e:
        st.error("Eğitilmiş model (catboost_model.pkl) bulunamadı veya yüklenemedi. Lütfen model dosyasını /prison_app/ içine koyun.\n\nDetay: " + str(e))
        st.info("Geçici olarak demo mantığı: geçmiş_suc_sayisi > 1 ise yüksek risk, değilse düşük risk")
        # very simple rule fallback
        score = 0.75 if ("gecmis_suc_sayisi" in X.columns and X.loc[0,"gecmis_suc_sayisi"] > 1) else 0.12
        label = "Yüksek risk" if score > 0.5 else "Düşük risk"
        st.metric("Tahmin", label, delta=f"%{score*100:.1f} risk skoru")
    else:
        try:
            proba = None
            # scikit-like API
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0]
                # if binary: proba for positive class
                if proba.shape[0] and len(proba) == 2:
                    score = proba[1]
                else:
                    # multiclass: take max prob class's score
                    score = float(np.max(proba))
                pred = model.predict(X)[0]
            else:
                # catboost/fallback predict
                pred = model.predict(X)
                score = float(pred[0]) if hasattr(pred, "__len__") else float(pred)
            label = "Tekrar suç işleme (olası)" if score > 0.5 else "Tekrar suç işleme (düşük olasılık)"
            st.success(f"🔔 Tahmin: {label}")
            st.progress(int(min(max(score,0),1)*100))
            st.write(f"**Güven skoru:** {score:.3f}")
            st.write("Model çıktısı (ham):", pred)
        except Exception as e:
            st.error("Model çalıştırılırken hata oluştu: " + str(e))
