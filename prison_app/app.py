# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import numpy as np

st.set_page_config(
    page_title="Yeniden Suç İşleme Tahmin Uygulaması",
    page_icon="⚖️",
    layout="wide",
)

# Güvenli base path araması: __file__ olmayabilir (ör. bazı deploy ortamları)
CANDIDATE_DIRS = [
    Path(__file__).parent if "__file__" in globals() else None,
    Path.cwd(),
    Path("/mnt/data"),
    Path.home(),
]
CANDIDATE_DIRS = [p for p in CANDIDATE_DIRS if p is not None]

CSV_CANDIDATES = ["Prisongüncelveriseti.csv", "PrisonGuncelVeriSeti.csv", "Prisonguncelveriseti.csv"]

APP_VERSION = "v1.4 (Modern Grafikler - Fixed)"

@st.cache_data(show_spinner=False)
def find_file_anywhere(names):
    for d in CANDIDATE_DIRS:
        for name in names:
            p = d / name
            try:
                if p.exists():
                    return p
            except Exception:
                continue
    return None

@st.cache_data(show_spinner=False)
def load_data():
    p = find_file_anywhere(CSV_CANDIDATES)
    if p is None:
        return None, None
    try:
        df = pd.read_csv(p)
        return df, p
    except Exception as e:
        st.error(f"CSV okunurken hata: {e}")
        return None, p

def safe_mean(series):
    try:
        return pd.to_numeric(series, errors='coerce').dropna().mean()
    except:
        return None

def safe_unique(series):
    try:
        return int(series.nunique())
    except:
        return 0

def render_card(col, value, label, emoji, color="#0d47a1"):
    card_style = f"""
        background-color: {color}33;
        border-radius: 14px;
        padding: 1.2rem 0.8rem;
        text-align: center;
        box-shadow: 0 6px 15px rgba(0,0,0,0.08);
        min-height: 100px;
        display:flex; flex-direction:column; justify-content:center;
        """
    number_style = f"font-size:1.9rem; font-weight:700; color:{color};"
    label_style = f"font-size:0.95rem; color:{color}; font-weight:600; margin-top:0.2rem;"
    col.markdown(f"""
        <div style="{card_style}">
            <div style="{number_style}">{value}</div>
            <div style="{label_style}">{emoji} {label}</div>
        </div>
    """, unsafe_allow_html=True)

def convert_age_to_numeric(age_str):
    if pd.isna(age_str):
        return np.nan
    s = str(age_str).strip()
    if "or older" in s:
        try:
            low = int(s.split()[0])
            return float(low)
        except:
            return np.nan
    if "-" in s:
        try:
            a,b = s.split("-")
            return (float(a) + float(b))/2.0
        except:
            return np.nan
    try:
        return float(s)
    except:
        return np.nan

def home_page(df, csv_path):
    st.markdown(
        """
        <div style="background:#0d1b2a;color:white;padding:1.4rem;border-radius:12px;">
            <h1>🏛️ Yeniden Suç İşleme Tahmin Uygulaması</h1>
            <p>Bu uygulama, mahpusların tahliye sonrası yeniden suç işleme riskini analiz etmek için hazırlanmıştır.</p>
        </div>
        """, unsafe_allow_html=True
    )
    st.markdown("---")

    if df is None:
        st.error("Veri seti yüklenemedi. Repo köküne CSV dosyasını ekleyin veya sol menüden yükleyin.")
        st.warning("Aranan dosya adları: " + ", ".join(CSV_CANDIDATES))
        # file uploader fallback
        uploaded = st.file_uploader("CSV dosyasını yükleyin (alternatif)", type=["csv"])
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
                st.success("CSV başarıyla yüklendi.")
            except Exception as e:
                st.error(f"Yüklenen CSV okunamadı: {e}")
                return
        else:
            return

    # recid kolonunu bulma
    recid_col = None
    for c in df.columns:
        if "recid" in c.lower() or "reoffend" in c.lower() or "recidiv" in c.lower():
            recid_col = c
            break

    if recid_col:
        df[recid_col] = pd.to_numeric(df[recid_col], errors='coerce')

    if "Age_at_Release" in df.columns:
        df["Age_at_Release_Num"] = df["Age_at_Release"].apply(convert_age_to_numeric)
    else:
        df["Age_at_Release_Num"] = np.nan

    if "Sentence_Length_Months" in df.columns:
        df["Sentence_Length_Months"] = pd.to_numeric(df["Sentence_Length_Months"], errors='coerce')

    info_cards = []
    info_cards.append(("Toplam Kayıt", f"{df.shape[0]:,}", "🗂️", "#0d47a1"))
    info_cards.append(("Sütun Sayısı", df.shape[1], "📋", "#1976d2"))

    if "Prison_Offense" in df.columns:
        n_offense = safe_unique(df["Prison_Offense"])
        if n_offense > 0:
            info_cards.append(("Farklı Suç Tipi", n_offense, "📌", "#0288d1"))

    avg_sentence = safe_mean(df["Sentence_Length_Months"]) if "Sentence_Length_Months" in df.columns else None
    if avg_sentence is not None and not np.isnan(avg_sentence):
        info_cards.append(("Ortalama Ceza Süresi (Ay)", f"{avg_sentence:.1f}", "⏳", "#388e3c"))

    recid_rate = safe_mean(df[recid_col]) if recid_col else None
    if recid_rate is not None and not np.isnan(recid_rate):
        info_cards.append(("Yeniden Suç İşleme Oranı", f"%{recid_rate*100:.1f}", "⚠️", "#d32f2f"))

    avg_age = safe_mean(df["Age_at_Release_Num"]) if "Age_at_Release_Num" in df.columns else None
    if avg_age is not None and not np.isnan(avg_age):
        info_cards.append(("Ortalama Tahliye Yaşı", f"{avg_age:.1f}", "👤", "#00695c"))

    if "Education_Level" in df.columns:
        n_edu = safe_unique(df["Education_Level"])
        if n_edu > 0:
            info_cards.append(("Eğitim Seviyesi Sayısı", n_edu, "🎓", "#6a1b9a"))

    if "Gender" in df.columns:
        n_gender = safe_unique(df["Gender"])
        if n_gender > 0:
            info_cards.append(("Cinsiyet Sayısı", n_gender, "🚻", "#5d4037"))

    # render cards 4 per row
    n = len(info_cards)
    rows = (n + 3) // 4
    for r in range(rows):
        cols = st.columns(4, gap="small")
        for i in range(4):
            idx = r*4 + i
            if idx >= n:
                break
            label, val, emoji, color = info_cards[idx]
            render_card(cols[i], val, label, emoji, color)

    st.markdown("---")
    with st.expander("📂 Veri Seti Önizlemesi (İlk 10 Satır)"):
        st.dataframe(df.head(10), use_container_width=True)

    st.markdown("---")
    st.subheader("🎯 Yeniden Suç İşleme Oranı (Pasta Grafiği)")
    col1, col2 = st.columns([3,1])
    with col1:
        if recid_col and df[recid_col].dropna().size > 0:
            counts = df[recid_col].value_counts().sort_index()
            labels = ["Tekrar Suç İşlemedi", "Tekrar Suç İşledi"]
            values = [int(counts.get(0, 0)), int(counts.get(1, 0))]
            fig = px.pie(names=labels, values=values, title="3 Yıl İçinde Yeniden Suç İşleme Oranı")
            fig.update_traces(textposition='inside', textinfo='percent+label', pull=[0, 0.1])
            fig.update_layout(title_x=0.5, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Yeniden suç işleme verisi bulunmamaktadır.")
    with col2:
        st.markdown("ℹ️ Bu pasta grafik, tahliye sonrası mahpusların yeniden suç işleme durumunu gösterir.")

    st.markdown("---")
    st.subheader("👥 Demografik Dağılımlar ve Yeniden Suç İşleme Oranları")
    demo_cols = [c for c in ["Gender", "Education_Level"] if c in df.columns]
    if demo_cols:
        cols = st.columns(len(demo_cols))
        for idx, col_name in enumerate(demo_cols):
            with cols[idx]:
                counts = df[col_name].value_counts()
                fig_bar = px.bar(x=counts.index, y=counts.values, labels={"x":col_name,"y":"Kişi Sayısı"}, title=f"{col_name} Dağılımı")
                fig_bar.update_layout(showlegend=False, template="plotly_white", title_x=0.5)
                st.plotly_chart(fig_bar, use_container_width=True)

                if recid_col:
                    recid_means = df.groupby(col_name)[recid_col].mean().fillna(0)
                    fig_recid = px.bar(x=recid_means.index, y=recid_means.values, labels={"x":col_name,"y":"Ortalama Recidivism"}, title=f"{col_name} Bazında Yeniden Suç İşleme Oranı")
                    fig_recid.update_layout(showlegend=False, template="plotly_white", title_x=0.5, yaxis=dict(range=[0,1]))
                    st.plotly_chart(fig_recid, use_container_width=True)
    else:
        st.info("Demografik dağılımlar için uygun kolon bulunamadı.")

    st.caption(f"📂 Repo: https://github.com/Yasinaslann/PrisonPredictApp • {APP_VERSION}")
    st.write(f"Veri dosyası: {csv_path}")


def main():
    df, csv_path = load_data()
    st.sidebar.title("Navigasyon")
    page = st.sidebar.radio(
        "Sayfa seçin",
        ("Ana Sayfa", "Tahmin Modeli", "Tavsiye ve Profil Analizi", "Model Analizleri ve Harita"),
    )

    if page == "Ana Sayfa":
        home_page(df, csv_path)
    else:
        st.info("Tahmin sayfası için predict_app.py kullanın veya repo içindeki predict script'ini çalıştırın.")

if __name__ == "__main__":
    main()
