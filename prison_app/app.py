# prison_app/app.py
import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Prison Predict App", page_icon="🏛️", layout="wide")

BASE = Path(__file__).parent
DATA_PATHS = [
    Path("/mnt/data/Prisongüncelveriseti.csv"),
    BASE / "Prisongüncelveriseti.csv"
]

st.title("🏛️ Prison Predict App")
st.markdown("### 🔍 Mahkumların tekrar suç işleme olasılığını tahmin etme ve kişiye özel rehberlik")

col1, col2 = st.columns([3,1])
with col1:
    st.markdown(
        """
        Bu proje, **mahkumların tekrar suç işleme riskini** değerlendirmek ve
        bu riske göre **rehabilitasyon önerileri** sunmak amacıyla hazırlanmıştır.

        **Sayfalar:**
        - 2️⃣ Tahmin Modeli — Bireysel giriş ile tahmin al.
        - 3️⃣ Tavsiye & Profil Analizi — Tahmine göre öneriler.
        - 4️⃣ Model Analizleri & Harita — Model değerlendirme, özellik önemi, bölgesel analiz.
        """
    )

with col2:
    st.image("https://raw.githubusercontent.com/Yasinaslann/PrisonPredictApp/main/logo.png" 
             if (BASE / "logo.png").exists() else "https://cdn-icons-png.flaticon.com/512/3064/3064197.png",
             width=110)

st.markdown("---")
# Dataset yükleme (robust)
df = None
for p in DATA_PATHS:
    try:
        if p.exists():
            df = pd.read_csv(p)
            data_source = str(p)
            break
    except Exception:
        pass

if df is None:
    st.warning("Veri seti bulunamadı. Uygulama demo verisi ile çalışacaktır. `Prisongüncelveriseti.csv` dosyasını /prison_app/ içine veya /mnt/data/ yoluna koyun.")
    # küçük örnek demo
    df = pd.DataFrame({
        "suç_tipi":["hırsızlık","dolandırıcılık","yaralama"],
        "ceza_ay": [6,12,24],
        "egitim_durumu":["ilkokul","lise","lisans"],
        "gecmis_suc_sayisi":[0,2,1],
        "il":["Istanbul","Ankara","Izmir"]
    })
    data_source = "demo"

with st.expander("📊 Veri Seti Önizlemesi (ilk 10 satır) — kayna: " + data_source):
    st.dataframe(df.head(10))

st.markdown("## 🔧 Nasıl ilerlemeli")
st.markdown("""
1. **Tahmin Modeli** sayfasından bireysel bilgiler ile tahmin alabilirsiniz.  
2. Eğer elinizde eğitilmiş bir `catboost_model.pkl` vb. varsa `prison_app/` içine koyun.  
3. Eksiksiz dağıtım için `requirements.txt` dosyasını kullanın ve Streamlit Cloud'a deploy edin.
""")

st.markdown("---")
st.caption("Repo referansı: kullanıcının GitHub repo bağlantısına göre uyarlanmıştır. Eğer özel .pkl dosyaları mevcutsa aynı dizine koyun.")
