import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from io import BytesIO
from pathlib import Path

# ---------------------------
# Dosya yolları
# ---------------------------
MODEL_PATH = Path("prison_app/catboost_model.pkl")
BOOL_PATH = Path("prison_app/bool_columns.pkl")
CAT_FEAT_PATH = Path("prison_app/cat_features.pkl")
FEATURES_PATH = Path("prison_app/feature_names.pkl")
CAT_UNIQ_PATH = Path("prison_app/cat_unique_values.pkl")
DATA_PATH = Path("prison_app/dataset.csv")  # Eğitim veri seti csv (upload etmelisin)
PERF_REPORT_PATH = Path("prison_app/perf_report.csv")  # Model performans csv (isteğe bağlı)

# ---------------------------
# Feature açıklamaları (kısa)
# ---------------------------
FEATURE_DESCRIPTIONS = {
    "Gender": "Mahkumun cinsiyeti (Male/Female).",
    "Race": "Mahkumun ırkı (White, Black, Hispanic, vb).",
    "Age_at_Release": "Tahliye anındaki yaş.",
    "Gang_Affiliated": "Çete üyeliği (True/False).",
    "Supervision_Risk_Score_First": "İlk risk skoru.",
    "Education_Level": "Eğitim seviyesi.",
    "Dependents": "Bakmakla yükümlü kişi sayısı.",
    "Prison_Offense": "İşlenen suç türü.",
    "Prison_Years": "Hapiste geçirilen yıl sayısı.",
    "Num_Distinct_Arrest_Crime_Types": "Tutuklama suç türleri sayısı.",
    # Daha fazla açıklama ekleyebilirsin...
}

# ---------------------------
# Veri ve model yükleme
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_resources():
    model = joblib.load(MODEL_PATH)
    bool_cols = joblib.load(BOOL_PATH)
    cat_features = joblib.load(CAT_FEAT_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    cat_unique_values = joblib.load(CAT_UNIQ_PATH)
    df_data = pd.read_csv(DATA_PATH)
    return model, bool_cols, cat_features, feature_names, cat_unique_values, df_data

model, bool_cols, cat_features, feature_names, cat_unique_values, df_data = load_resources()

# ---------------------------
# Performans verisi (örnek)
# ---------------------------
@st.cache_data(show_spinner=False)
def load_perf_report():
    try:
        df_perf = pd.read_csv(PERF_REPORT_PATH)
        return df_perf
    except Exception:
        return None

df_perf_report = load_perf_report()

# ---------------------------
# Sayfa fonksiyonları
# ---------------------------
def predict_page():
    st.title("🔮 Recidivism Tahmin Sayfası")

    st.markdown("""
    Mahkumun özelliklerini girin ve 3 yıl içindeki yeniden suç işleme riskini tahmin edin.  
    Girdi alanlarının yanında kısa açıklamalar bulunuyor, üzerine gelerek detayları görebilirsiniz.
    """)

    input_data = {}
    cols = st.columns(2)

    for i, feature in enumerate(feature_names):
        with cols[i % 2]:
            help_text = FEATURE_DESCRIPTIONS.get(feature, "Açıklama bulunmamaktadır.")
            if feature in bool_cols:
                val = st.selectbox(f"{feature}", options=["True", "False"], help=help_text)
                input_data[feature] = True if val == "True" else False
            elif feature in cat_features:
                options = cat_unique_values.get(feature, [])
                if options:
                    val = st.selectbox(f"{feature}", options=options, help=help_text)
                else:
                    val = st.text_input(f"{feature}", help=help_text)
                input_data[feature] = val
            else:
                val = st.number_input(f"{feature}", value=0.0, format="%.6f", help=help_text)
                input_data[feature] = val

    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []

    if st.button("🔮 Tahmin Yap"):
        try:
            df_input = pd.DataFrame([input_data], columns=feature_names)
            for b in bool_cols:
                if b in df_input.columns:
                    df_input[b] = df_input[b].astype(str)

            prediction = model.predict(df_input)[0]
            proba = model.predict_proba(df_input)[0][1] if hasattr(model, "predict_proba") else None

            color = "#d9534f" if prediction == 1 else "#5cb85c"
            risk_text = "Yüksek Riskli" if prediction == 1 else "Düşük Riskli"

            st.markdown(f"""
            <div style="background-color:#f0f0f0; border-radius:12px; padding:20px; margin-top:20px;">
                <h2 style="color:{color}; text-align:center;">Tahmin Sonucu: {risk_text} ({prediction})</h2>
                <h4 style="text-align:center;">Risk Olasılığı: <strong>{proba*100:.2f}%</strong></h4>
            </div>
            """, unsafe_allow_html=True)

            # SHAP Açıklaması
            st.subheader("Tahmin Açıklaması: Modelin kararını etkileyen önemli özellikler")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(df_input)

            fig, ax = plt.subplots(figsize=(12, 6))
            shap.plots._waterfall.waterfall_legacy(
                explainer.expected_value, shap_values[0], df_input.iloc[0], max_display=10, show=False)
            st.pyplot(fig)
            plt.close(fig)

            st.subheader("📌 Kişisel Öneriler")
            if prediction == 1:
                st.info("""
                - Eğitim programlarına katılmanız önerilir.  
                - Denetimli serbestlik programına dahil olun.  
                - Psikolojik destek ve rehberlik alın.  
                """)
            else:
                st.success("""
                - Riskiniz düşük.  
                - Mevcut destek ve programlara devam edin.  
                - Topluma başarılı entegrasyon için çaba gösterin.  
                """)

            # Tahmin geçmişi
            rec = input_data.copy()
            rec["Prediction"] = prediction
            rec["Probability"] = proba
            st.session_state.prediction_history.append(rec)

        except Exception as e:
            st.error(f"Tahmin sırasında hata oluştu: {e}")

    if st.session_state.prediction_history:
        st.markdown("---")
        st.subheader("📋 Tahmin Geçmişi")

        df_hist = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(df_hist)

        csv_buffer = BytesIO()
        df_hist.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue()

        st.download_button(
            label="⬇️ Tahmin Geçmişini CSV Olarak İndir",
            data=csv_bytes,
            file_name="tahmin_gecmisi.csv",
            mime="text/csv"
        )


def analysis_page():
    st.title("📊 Veri Analizi ve Görselleştirme")

    st.markdown("""
    Eğitim verisindeki değişkenlerin dağılımlarını interaktif grafiklerle inceleyebilirsiniz.  
    Sol taraftaki filtrelerle istediğiniz segmenti seçip grafikleri güncelleyebilirsiniz.
    """)

    # Filtreler
    gender_options = df_data["Gender"].dropna().unique().tolist()
    race_options = df_data["Race"].dropna().unique().tolist()
    age_min = int(df_data["Age_at_Release"].min())
    age_max = int(df_data["Age_at_Release"].max())

    st.sidebar.header("Filtreler")
    selected_genders = st.sidebar.multiselect("Cinsiyet", options=gender_options, default=gender_options)
    selected_races = st.sidebar.multiselect("Irk", options=race_options, default=race_options)
    selected_age = st.sidebar.slider("Yaş Aralığı", min_value=age_min, max_value=age_max, value=(age_min, age_max))

    # Filtre uygula
    df_filtered = df_data[
        (df_data["Gender"].isin(selected_genders)) &
        (df_data["Race"].isin(selected_races)) &
        (df_data["Age_at_Release"] >= selected_age[0]) &
        (df_data["Age_at_Release"] <= selected_age[1])
    ]

    st.subheader("Risk Dağılımı")
    risk_counts = df_filtered["Recidivism_Within_3years"].value_counts(normalize=True).reset_index()
    risk_counts.columns = ["Recidivism", "Oran"]
    risk_counts["Recidivism"] = risk_counts["Recidivism"].map({0:"Düşük Risk", 1:"Yüksek Risk"})

    fig = px.pie(risk_counts, names="Recidivism", values="Oran",
                 title="3 Yıl İçinde Tekrar Suç İşleme Risk Dağılımı", hole=0.4)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Yaş Dağılımı")
    fig2 = px.histogram(df_filtered, x="Age_at_Release", nbins=30, title="Tahliye Yaş Dağılımı", marginal="box")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Cinsiyete Göre Risk Dağılımı")
    fig3 = px.histogram(df_filtered, x="Gender", color="Recidivism_Within_3years",
                        barmode="group", title="Cinsiyete Göre Risk Dağılımı",
                        category_orders={"Gender": gender_options})
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Eğitim Seviyesine Göre Risk Dağılımı")
    fig4 = px.histogram(df_filtered, x="Education_Level", color="Recidivism_Within_3years",
                        barmode="group", title="Eğitim Seviyesine Göre Risk Dağılımı")
    st.plotly_chart(fig4, use_container_width=True)


def performance_page():
    st.title("📈 Model Performans Sayfası")

    st.markdown("""
    Modelin genel performans metriği ve grafiklerini aşağıda görebilirsiniz.
    """)

    if df_perf_report is None:
        st.warning("Performans raporu dosyası bulunamadı.")
        return

    st.subheader("Sınıflandırma Raporu")
    st.dataframe(df_perf_report)

    # Confusion matrix
    cm = confusion_matrix(df_perf_report["True_Label"], df_perf_report["Predicted_Label"])
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap='Blues')

    ax.set_xticks(np.arange(2))
    ax.set_yticks(np.arange(2))
    ax.set_xticklabels(['Düşük Risk (0)', 'Yüksek Risk (1)'])
    ax.set_yticklabels(['Düşük Risk (0)', 'Yüksek Risk (1)'])
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek')

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='black')

    st.pyplot(fig)

    # ROC Eğrisi
    if "Probability" in df_perf_report.columns:
        fpr, tpr, _ = roc_curve(df_perf_report["True_Label"], df_perf_report["Probability"])
        roc_auc = auc(fpr, tpr)
        fig2, ax2 = plt.subplots()
        ax2.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
        ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax2.legend(loc="lower right")
        st.pyplot(fig2)


def about_page():
    st.title("ℹ️ Proje Hakkında ve Kullanım Rehberi")

    st.markdown("""
    **Bu uygulama, mahkumların 3 yıl içindeki yeniden suç işleme riskini tahmin etmek için geliştirilmiştir.**  

    - Eğitim veri seti: [açıklamalar]  
    - Kullanılan model: CatBoostClassifier  
    - Veri özellikleri ve anlamları sayfasında bulunabilir.  

    ### Kullanım  
    - Tahmin sayfasında özelliklerinizi girin ve tahmin yapın.  
    - Veri analizi sayfasında veri setini keşfedin.  
    - Model performans sayfasında modelin doğruluk ve grafiklerini görün.  

    ### Notlar  
    - Tüm veriler anonimleştirilmiş ve etik kurallara uygun şekilde kullanılmıştır.  
    - Tahminler kesin sonuç değil, sadece olasılıksal değerlendirmedir.  

    Geri bildirim ve öneriler için iletişime geçebilirsiniz.
    """)

# ---------------------------
# Ana app - sayfa seçici
# ---------------------------
def main():
    st.sidebar.title("🚦 Sayfalar")
    page = st.sidebar.radio("Bir sayfa seçin:", ("Tahmin", "Veri Analizi", "Model Performansı", "Hakkında"))

    if page == "Tahmin":
        predict_page()
    elif page == "Veri Analizi":
        analysis_page()
    elif page == "Model Performansı":
        performance_page()
    elif page == "Hakkında":
        about_page()

if __name__ == "__main__":
    main()
