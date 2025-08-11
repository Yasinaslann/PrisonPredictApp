# app.py
"""
Streamlit çok sayfalı uygulama iskeleti for PrisonPredictApp
Dosya yapÄ±sÄ±: (repo kökünde)
- app.py
- prison_app/
    - pages/
        - page_home.py
        - page_prediction.py
        - page_recommendation.py
        - page_model_analysis.py
    - assets/ (opsiyonel)
- Prisonguncelveriseti.csv
- catboost_model.pkl
- feature_names.pkl
- cat_features.pkl
- cat_unique_values.pkl
- bool_columns.pkl
- requirements.txt

Kullanım: repository'ye yerleştirip `streamlit run app.py` ile çalıştırın.
"""

import streamlit as st
from importlib import import_module
import os

# Basit çoklu sayfa router
PAGES = {
    "1. Anasayfa": "prison_app.pages.page_home",
    "2. Tahmin Modeli": "prison_app.pages.page_prediction",
    "3. Tavsiye ve Profil Analizi": "prison_app.pages.page_recommendation",
    "4. Model Analizleri ve Harita": "prison_app.pages.page_model_analysis",
}

st.set_page_config(page_title="PrisonPredictApp", layout="wide")

st.sidebar.title("Navigasyon")
page_choice = st.sidebar.radio("Sayfalar", list(PAGES.keys()))

module_name = PAGES[page_choice]
try:
    page = import_module(module_name)
    # Her modülde `app()` fonksiyonu olacak
    page.app()
except Exception as e:
    st.error(f"Sayfa yüklenirken hata oluştu: {e}")
    st.stop()


# prison_app/pages/page_home.py
"""
Basit ana sayfa. Proje hakkında kısa bilgi ve veri yükleme.
"""

def app():
    import streamlit as st
    import pandas as pd
    from pathlib import Path

    st.title("Prison Predict App — Anasayfa")
    st.markdown(
        "Bu uygulama mahkûm profillerine göre bazı risk/tahmin modelleri uygular, profil analizi yapar ve modelin performansını inceler.\n\n"
        "Kullanmak için sol menüden bir sayfa seçin."
    )

    data_path = Path("Prisonguncelveriseti.csv")
    if data_path.exists():
        if st.button("Veriyi önizle"):
            df = pd.read_csv(data_path)
            st.write(df.head(10))
            st.write(df.describe(include='all'))
    else:
        st.info("Prisonguncelveriseti.csv kök dizinde bulunamadı. Veri önizlemesi için dosyayı ekleyin.")


# prison_app/pages/page_prediction.py
"""
Tahmin sayfası: Kullanıcıdan girdiler alır, modeli yükler ve tahmin yapar.
"""

def app():
    import streamlit as st
    import pandas as pd
    import joblib
    from pathlib import Path
    import numpy as np

    st.title("Tahmin Modeli")
    st.write("Model ile tek tek veya toplu (CSV) örnekler için tahmin yapabilirsiniz.")

    # model ve metadata yükleme
    model_path = Path("catboost_model.pkl")
    features_path = Path("feature_names.pkl")
    cat_features_path = Path("cat_features.pkl")
    bool_cols_path = Path("bool_columns.pkl")
    unique_vals_path = Path("cat_unique_values.pkl")

    if not model_path.exists():
        st.error("Model dosyası catboost_model.pkl bulunamadı. Lütfen repo köküne koyun.")
        return

    model = joblib.load(model_path)

    st.subheader("Tek kişilik tahmin")
    # Basit örnek: kullanıcı bazı temel alanları girer. Burayı kendi feature listesine göre genişletin.
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Yaş", min_value=16, max_value=100, value=30)
        sentence_years = st.number_input("Ceza (yıl)", min_value=0, max_value=100, value=5)
    with col2:
        education = st.selectbox("Eğitim", ["İlkokul", "Ortaokul", "Lise", "Üniversite"]) 
        prior_offenses = st.number_input("Önceki suç sayısı", min_value=0, max_value=50, value=0)
    with col3:
        employment = st.selectbox("İş durumu", ["İşsiz", "Kısmi", "Tam zamanlı"]) 
        drug_history = st.selectbox("Uyuşturucu geçmişi", ["Yok", "Var"]) 

    if st.button("Tahmin Yap"):
        # Basit mapping; gerçekte feature_names.pkl'e göre üretilmeli
        X = pd.DataFrame([{
            'age': age,
            'sentence_years': sentence_years,
            'education': education,
            'prior_offenses': prior_offenses,
            'employment': employment,
            'drug_history': drug_history
        }])

        # Eğer model pipeline gerekliyse buraya preprocessing ekleyin. Bu örnek direkt model.predict_proba
        try:
            proba = model.predict_proba(X)
            # Eğer iki sınıflıysa
            if proba.shape[1] >= 2:
                score = proba[:, 1][0]
                st.success(f"Olma ihtimali (pozitif sınıf): {score:.3f}")
            else:
                pred = model.predict(X)[0]
                st.success(f"Tahmin: {pred}")
        except Exception as e:
            st.error(f"Tahmin sırasında hata: {e}\nModelin beklediği özellik isimleri ve preprocessing ile uyumlu olmalıdır.")

    st.subheader("CSV ile toplu tahmin")
    uploaded_file = st.file_uploader("CSV dosyası (aynı feature isimleri ile)", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Yüklendi: ", df.shape)
        if st.button("Toplu Tahmin Yap"):
            try:
                preds = model.predict_proba(df)
                if preds.shape[1] >= 2:
                    df['pred_proba'] = preds[:,1]
                else:
                    df['pred'] = preds
                st.write(df.head())
                st.markdown("Tahminli CSV'yi indir:")
                st.download_button("İndir CSV", df.to_csv(index=False).encode('utf-8'), file_name='predictions.csv')
            except Exception as e:
                st.error(f"Toplu tahmin hatası: {e}")


# prison_app/pages/page_recommendation.py
"""
Tavsiye ve profil analizi sayfası.
"""

def app():
    import streamlit as st
    import pandas as pd
    from pathlib import Path

    st.title("Tavsiye ve Profil Analizi")
    st.write("Bir bireyin profilini girerek basit tavsiyeler alabilirsiniz.")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Yaş", min_value=16, max_value=100, value=30, key='r_age')
        education = st.selectbox("Eğitim", ["İlkokul","Ortaokul","Lise","Üniversite"], key='r_edu')
    with col2:
        prior = st.number_input("Önceki suç sayısı", min_value=0, max_value=50, value=0, key='r_prior')
        employment = st.selectbox("İş durumu", ["İşsiz","Kısmi","Tam zamanlı"], key='r_emp')

    if st.button("Profil Analizi ve Tavsiye"):
        recommendations = []
        if prior > 2:
            recommendations.append("Yüksek tekrar riski: rehabilitasyon programlarına yönlendirin.")
        else:
            recommendations.append("Tekrar riski nispeten düşük.")

        if employment == 'İşsiz':
            recommendations.append("İstihdam destek programlarına katılmasını önerin.")

        if education in ['İlkokul','Ortaokul']:
            recommendations.append("Eğitim/mesleki kurslara yönlendirme faydalı olabilir.")

        st.subheader("Tavsiyeler")
        for r in recommendations:
            st.write("- ", r)

        st.subheader("Basit Profil Özet")
        st.json({
            'age': age,
            'education': education,
            'prior_offenses': prior,
            'employment': employment
        })


# prison_app/pages/page_model_analysis.py
"""
Model analizleri, metrikler ve basit harita gösterimi (örnek olarak folium veya streamlit-folium kullanılabilir).
"""

def app():
    import streamlit as st
    import pandas as pd
    import joblib
    from pathlib import Path
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

    st.title("Model Analizleri ve Harita")
    st.write("Model performans metrikleri ve örnek harita (örnek veri ile).")

    data_path = Path("Prisonguncelveriseti.csv")
    model_path = Path("catboost_model.pkl")

    if not data_path.exists():
        st.info("Veri bulunamadı: Prisonguncelveriseti.csv'")
        return

    df = pd.read_csv(data_path)
    st.write("Veri yüklendi, örnek: ")
    st.write(df.head())

    if model_path.exists():
        model = joblib.load(model_path)
        st.subheader("Model değerlendirme (örnek)")
        if 'target' in df.columns:
            X = df.drop(columns=['target'])
            y = df['target']
            try:
                preds = model.predict_proba(X)[:,1]
                auc = roc_auc_score(y, preds)
                st.write(f"ROC AUC: {auc:.3f}")
                # Basit ROC histogram
                fig, ax = plt.subplots()
                ax.hist(preds[y==0], bins=30, alpha=0.6)
                ax.hist(preds[y==1], bins=30, alpha=0.6)
                ax.set_title('Tahmin olasılıklarının dağılımı (sınıflar)')
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Model değerlendirme hatası: {e}")
        else:
            st.info("Veride 'target' kolonu yok; model değerlendirme için target kolon ekleyin.")
    else:
        st.info("Model dosyası bulunamadı; sadece veri gösteriliyor.")

    st.subheader("Harita - Örnek (enlem/boylam varsa)")
    if 'latitude' in df.columns and 'longitude' in df.columns:
        try:
            import pydeck as pdk
            st.map(df[['latitude','longitude']].dropna())
        except Exception as e:
            st.info("Harita gösterimi için pydeck veya folium yüklü değil ya da hata oluştu.")
    else:
        st.info("Veride 'latitude' ve 'longitude' kolonları yoksa harita gösterilemez.")
