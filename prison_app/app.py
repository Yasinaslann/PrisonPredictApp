# Gerekli kütüphaneleri import ediyoruz.
# Eğer bu kütüphaneler yüklü değilse, terminalden 'pip install -r requirements.txt' komutunu çalıştırın.
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from catboost import CatBoostClassifier

# Sayfa yapılandırmasını yapıyoruz.
st.set_page_config(
    page_title="Yeniden Suç İşleme Tahmin Uygulaması",
    page_icon="⚖️",
    layout="wide"
)

# --- Veri Seti ve Model Yükleme ---
# Bu fonksiyonlar, dosyaları GitHub deponuzdan yüklemek için kullanılır.
# @st.cache_data ve @st.cache_resource, uygulamayı her çalıştırdığınızda dosyaların yeniden yüklenmesini engeller.
@st.cache_data
def load_data():
    """Temizlenmiş veri setini yükler."""
    try:
        df = pd.read_csv("Prisongüncelveriseti.csv")
        return df
    except FileNotFoundError:
        st.error("Prisongüncelveriseti.csv dosyası bulunamadı. Lütfen dosyanın GitHub deponuzda ve uygulamanızla aynı dizinde olduğundan emin olun.")
        return None

@st.cache_resource
def load_model_and_preprocessors():
    """Modeli ve ön işleme dosyalarını yükler."""
    try:
        # CatBoost modelini yüklüyoruz.
        model = pickle.load(open('catboost_model.pkl', 'rb'))
        
        # Ön işleme için gerekli dosyaları yüklüyoruz.
        bool_columns = pickle.load(open('bool_columns.pkl', 'rb'))
        cat_features = pickle.load(open('cat_features.pkl', 'rb'))
        cat_unique_values = pickle.load(open('cat_unique_values.pkl', 'rb'))
        feature_names = pickle.load(open('feature_names.pkl', 'rb'))
        
        return model, bool_columns, cat_features, cat_unique_values, feature_names
    except FileNotFoundError as e:
        st.error(f"Gerekli model veya ön işleme dosyası bulunamadı: {e}. Lütfen tüm dosyaların GitHub deponuzda ve uygulamanızla aynı dizinde olduğundan emin olun.")
        return None, None, None, None, None

# Verileri ve modeli yüklüyoruz.
df_cleaned = load_data()
model, bool_columns, cat_features, cat_unique_values, feature_names = load_model_and_preprocessors()


def preprocess_input(input_df, feature_names, cat_features):
    """Kullanıcı girdilerini modelin beklediği formata dönüştürür."""
    # Orijinal feature_names'e uygun bir DataFrame oluşturuyoruz.
    input_processed = pd.DataFrame(columns=feature_names)
    
    # Tüm değerleri varsayılan olarak 0 veya False ile dolduruyoruz
    for col in feature_names:
        if col.startswith("Gender_") or col.startswith("Race_") or col.startswith("Age_at_Release_") or col.startswith("Education_Level_") or col.startswith("Prison_Years_"):
            input_processed.loc[0, col] = 0
        elif col in bool_columns:
            input_processed.loc[0, col] = False
        else:
            input_processed.loc[0, col] = 0
            
    # Kullanıcı girdilerini yerleştiriyoruz
    for col, val in input_df.iloc[0].items():
        if col in bool_columns:
            input_processed.loc[0, col] = val
        elif col in cat_features:
            # One-hot encoding için ilgili sütun adını bulup 1 yapıyoruz.
            encoded_col_name = f"{col}_{val}"
            if encoded_col_name in input_processed.columns:
                input_processed.loc[0, encoded_col_name] = 1
        else:
            # Diğer sayısal ve boolean değerleri yerleştiriyoruz
            input_processed.loc[0, col] = val
            
    # Sütunları tekrar sıralıyoruz
    input_processed = input_processed[feature_names]

    return input_processed


# --- Sayfa Fonksiyonları ---
# Her bir sayfanın içeriği ayrı bir fonksiyonda tanımlanır.

def home_page():
    """Ana Sayfa içeriği"""
    st.title("⚖️ Yeniden Suç İşleme Tahmini Uygulaması")
    st.markdown("""
    Merhaba! Miul Makine Öğrenmesi Bootcamp projesi kapsamında geliştirilen bu uygulama,
    mahpusların ceza infazı sonrası yeniden suç işleme (recidivism) riskini tahmin etmeyi amaçlamaktadır.
    Bu analiz, sosyal hizmetler ve adalet sistemlerinin kaynaklarını daha etkili kullanmasına yardımcı olmayı hedeflemektedir.
    """)
    
    st.subheader("Kullanılan Veri Setleri")
    st.write("Bu projede iki farklı veri seti kullanılmıştır:")
    st.markdown("""
    - `NIJ_s_Recidivism_Challenge_Full_Dataset_20250729.csv`: Projenin başlangıç aşamasında kullanılan ham veri seti.
    - `Prisongüncelveriseti.csv`: Veri temizleme ve özellik mühendisliği (feature engineering) adımları uygulanmış, model eğitimine hazır hale getirilmiş veri seti.
    """)
    
    st.subheader("Geliştirme Ekibi")
    st.info("Yasin Aslan - Proje Geliştirici")
    st.markdown("[GitHub Profilim](https://github.com/Yasinaslann/PrisonPredictApp)")
    
    st.image("https://placehold.co/600x400/99e699/white?text=Yeniden+Su%C3%A7+%C4%B0%C5%9Fleme+Uygulamas%C4%B1")


def prediction_model_page():
    """Tahmin Modeli sayfası içeriği"""
    st.title("📊 Yeniden Suç İşleme Riski Tahmini")
    st.markdown("Aşağıdaki alanları doldurarak bir mahpusun yeniden suç işleme riskini tahmin edebilirsiniz.")
    
    if df_cleaned is None or model is None:
        st.warning("Uygulama dosyaları yüklenirken bir sorun oluştu. Lütfen gerekli tüm dosyaların (CSV ve PKL) mevcut olduğundan emin olun.")
        return

    # Input alanları
    with st.form("tahmin_formu"):
        st.subheader("Mahpus Bilgileri")
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox("Cinsiyet", options=df_cleaned['Gender'].unique())
            race = st.selectbox("Irk", options=df_cleaned['Race'].unique())
            age = st.selectbox("Yaş Grubu", options=df_cleaned['Age_at_Release'].unique())
            education = st.selectbox("Eğitim Seviyesi", options=df_cleaned['Education_Level'].unique())
            prison_years = st.selectbox("Ceza Süresi", options=df_cleaned['Prison_Years'].unique())
            
        with col2:
            num_distinct_arrest_crime_types = st.slider("Farklı Tutuklama Suç Tipi Sayısı", 0, 10, 3)
            prior_arrest_episodes_dvcharges = st.checkbox("Önceki Tutuklama Olayları (Aile İçi Şiddet)", False)
            condition_mh_sa = st.checkbox("Akıl Sağlığı/Madde Bağımlılığı Koşulu", False)
            percent_days_employed = st.slider("İstihdam Edilen Günlerin Yüzdesi (%)", 0.0, 1.0, 0.5)
            jobs_per_year = st.slider("Yılda Ortalama İş Değişikliği", 0.0, 5.0, 1.0)
            
        submitted = st.form_submit_button("Tahmin Et")
    
    if submitted:
        # Kullanıcı girdilerini bir DataFrame'e dönüştürüyoruz.
        input_data = {
            'Gender': [gender],
            'Race': [race],
            'Age_at_Release': [age],
            'Education_Level': [education],
            'Prison_Years': [prison_years],
            'Num_Distinct_Arrest_Crime_Types': [num_distinct_arrest_crime_types],
            'Prior_Arrest_Episodes_DVCharges': [prior_arrest_episodes_dvcharges],
            'Condition_MH_SA': [condition_mh_sa],
            'Percent_Days_Employed': [percent_days_employed],
            'Jobs_Per_Year': [jobs_per_year] # Hata düzeltildi: jobs_per_per_year -> jobs_per_year
        }
        input_df = pd.DataFrame(input_data)
        
        # Ön işleme fonksiyonunu kullanarak veriyi modele uygun hale getiriyoruz.
        preprocessed_data = preprocess_input(input_df, feature_names, cat_features)
        
        # Modelin tahminini alıyoruz.
        prediction = model.predict(preprocessed_data)
        prediction_proba = model.predict_proba(preprocessed_data)[:, 1][0] # Yeniden suç işleme olasılığı
        
        st.subheader("Tahmin Sonucu")
        if prediction == 1:
            st.error(f"Tahmin edilen risk: Yüksek ({prediction_proba:.2%} olasılık)")
            st.markdown("🚨 **Analiz:** Bu profildeki bireyin yeniden suç işleme riski yüksek görünmektedir. Bu durum, özellikle düşük istihdam oranı, genç yaş ve daha önceki suç kayıtları gibi faktörlerden kaynaklanıyor olabilir.")
        else:
            st.success(f"Tahmin edilen risk: Düşük ({1-prediction_proba:.2%} olasılık)")
            st.markdown("✅ **Analiz:** Bu profildeki bireyin yeniden suç işleme riski düşük görünmektedir. Yüksek eğitim seviyesi, istikrarlı istihdam geçmişi ve rehabilitasyon programlarına katılım gibi faktörler bu sonucu destekliyor olabilir.")

        # Tahmin sonucunu session state'e kaydediyoruz ki diğer sayfalarda kullanabilelim
        st.session_state['prediction_result'] = prediction
        st.session_state['prediction_proba'] = prediction_proba


def recommendation_page():
    """Tavsiye ve Profil Analizi sayfası içeriği"""
    st.title("💡 Tavsiye ve Profil Analizi")
    st.markdown("Modelin tahminine ve genel verilere dayanarak kişiye özel öneriler ve benzer profillerin analizini burada bulabilirsiniz.")
    
    if df_cleaned is None:
        st.error("Uygulama başlatılırken veri seti yüklenemedi. Lütfen 'Prisongüncelveriseti.csv' dosyasının bulunduğundan emin olun.")
        return

    if 'prediction_result' in st.session_state:
        prediction = st.session_state['prediction_result']
        
        st.subheader("Bireysel Gelişim Önerileri")
        if prediction == 1:
            st.warning("Bu profil yüksek riskli olarak belirlenmiştir. Bu riski azaltmak için aşağıdaki alanlara odaklanılması önerilir:")
            st.markdown("""
            * **Eğitim ve Beceri Geliştirme:** Mesleki eğitim programlarına veya GED (Lise Eşdeğeri) tamamlama kurslarına katılım.
            * **İstihdam Destek Programları:** İş bulma danışmanlığı, özgeçmiş hazırlama ve mülakat becerileri eğitimi.
            * **Psikososyal Destek:** Ruh sağlığı danışmanlığı ve madde bağımlılığı tedavi programlarına devam.
            * **Sosyal Entegrasyon:** Toplum merkezlerindeki destek gruplarına ve gönüllü çalışmalara katılım.
            """)
        else:
            st.success("Bu profil düşük riskli olarak belirlenmiştir. Mevcut olumlu durumun korunması ve sürdürülmesi için:")
            st.markdown("""
            * **Sürekli Eğitim:** Kariyer gelişimine yönelik sertifika programlarına katılım.
            * **Stabil İstihdam:** Mevcut işi korumaya yönelik destek mekanizmalarından yararlanma.
            * **Sosyal Bağları Güçlendirme:** Aile ve sosyal çevresi ile güçlü ilişkiler kurma ve sürdürme.
            """)

        st.subheader("Benzer Profillerin Analizi")
        # Benzer profilleri gösteren bir grafik
        recidivism_by_age = df_cleaned.groupby('Age_at_Release')['Recidivism_Within_3years'].mean().reset_index()
        fig = px.bar(recidivism_by_age, x='Age_at_Release', y='Recidivism_Within_3years',
                     title='Yaş Gruplarına Göre Yeniden Suç İşleme Oranı',
                     labels={'Age_at_Release': 'Yaş Grubu', 'Recidivism_Within_3years': 'Ortalama Yeniden Suç İşleme Oranı'})
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Analiz ve tavsiye görmek için lütfen 'Tahmin Modeli' sayfasında bir tahmin yapın.")


def model_analysis_page():
    """Model Analizleri ve Harita sayfası içeriği"""
    st.title("📈 Model Analizleri ve Görselleştirmeler")
    st.markdown("Makine öğrenmesi modelinin performans metriklerini ve veri setindeki önemli değişkenleri burada inceleyebilirsiniz.")
    
    if model is None:
        st.warning("Uygulama dosyaları yüklenirken bir sorun oluştu. Lütfen gerekli tüm dosyaların (PKL) mevcut olduğundan emin olun.")
        return

    st.subheader("Model Performans Metrikleri")
    
    # Bu kısım modelinizin test setindeki sonuçlarına göre güncellenmelidir.
    st.info("Model performans metrikleri (Confusion Matrix, Accuracy vb.) için lütfen test veri setinizdeki sonuçları buraya ekleyin.")
    
    st.subheader("Önemli Değişkenler (Feature Importance)")
    # CatBoost'un kendi feature importance metodu kullanılarak gerçek değerler gösterilir.
    if hasattr(model, 'feature_importances_'):
        feature_importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Değişken': feature_names,
            'Önem Derecesi': feature_importances
        }).sort_values(by='Önem Derecesi', ascending=False)
        
        fig_imp = px.bar(feature_importance_df, x='Önem Derecesi', y='Değişken', orientation='h',
                         title='Model İçin En Önemli Değişkenler',
                         labels={'Önem Derecesi': 'Önem Puanı', 'Değişken': ''})
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.warning("Modelden özellik önemi (feature importance) bilgisi alınamadı.")

    st.subheader("Harita Görselleştirmesi")
    st.markdown("""
    **Not:** Gerçek harita görselleştirmesi için veri setinizdeki `Residence_PUMA` gibi coğrafi konum bilgileri kullanılmalıdır.
    Aşağıdaki grafik, temsili bölgelere göre yeniden suç işleme oranlarını göstermektedir.
    """)
    
    # Temsili bir harita verisi
    map_data = {
        'Bölge': ['Bölge A', 'Bölge B', 'Bölge C', 'Bölge D'],
        'Yeniden Suç İşleme Oranı': [0.45, 0.60, 0.35, 0.50]
    }
    map_df = pd.DataFrame(map_data)
    fig_map = px.bar(map_df, x='Bölge', y='Yeniden Suç İşleme Oranı',
                     title='Temsili Bölgelere Göre Yeniden Suç İşleme Oranı',
                     labels={'Yeniden Suç İşleme Oranı': 'Oran'})
    st.plotly_chart(fig_map, use_container_width=True)


# --- Uygulamanın Ana Yapısı ---
# Sayfa navigasyonunu sol kenar çubuğunda oluşturuyoruz.
st.sidebar.title("Navigasyon")
selection = st.sidebar.radio("Sayfa Seçimi", ["Ana Sayfa", "Tahmin Modeli", "Tavsiye ve Profil Analizi", "Model Analizleri ve Harita"])

# Seçime göre ilgili sayfa fonksiyonunu çağırıyoruz.
if selection == "Ana Sayfa":
    home_page()
elif selection == "Tahmin Modeli":
    prediction_model_page()
elif selection == "Tavsiye ve Profil Analizi":
    recommendation_page()
elif selection == "Model Analizleri ve Harita":
    model_analysis_page()
