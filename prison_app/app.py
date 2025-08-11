import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="Yeniden Suç İşleme Tahmin Uygulaması",
    page_icon="🏛️",
    layout="wide",
)

BASE = Path(__file__).parent
DATA_PATHS = [
    BASE / "Prisongüncelveriseti.csv",
    Path("/mnt/data/Prisongüncelveriseti.csv"),
]

@st.cache_data(show_spinner=False)
def load_data():
    for path in DATA_PATHS:
        if path.exists():
            try:
                return pd.read_csv(path)
            except:
                pass
    return None

def render_card(col, value, label, emoji, color="#0d47a1"):
    card_style = f"""
        background-color: {color}33;
        border-radius: 14px;
        padding: 1.7rem 1rem;
        text-align: center;
        box-shadow: 0 6px 15px rgb(3 155 229 / 0.3);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        min-height: 110px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    """
    number_style = f"""
        font-size: 2.4rem; 
        font-weight: 800; 
        color: {color};
    """
    label_style = f"""
        font-size: 1.15rem; 
        color: {color};
        font-weight: 700;
        margin-top: 0.2rem;
    """
    col.markdown(f"""
        <div style="{card_style}">
            <div style="{number_style}">{value}</div>
            <div style="{label_style}">{emoji} {label}</div>
        </div>
    """, unsafe_allow_html=True)

def main():
    df = load_data()

    st.markdown(
        """
        <div style="
            background-color: #0d1b2a; 
            color: white; 
            padding: 2rem 2.5rem; 
            border-radius: 15px; 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            box-shadow: 0 6px 15px rgba(0,0,0,0.35);
        ">
            <h1 style="margin-bottom: 0.3rem;">🏛️ Yeniden Suç İşleme Tahmin Uygulaması</h1>
            <h3 style="margin-top:0; color:#90caf9;">Proje Amacı</h3>
            <p style="line-height:1.6; font-size:1.1rem;">
                Bu uygulama, mahpusların tahliye sonrasında yeniden suç işleme riskini (recidivism) veri bilimi ve makine öğrenmesi teknikleri ile tahmin etmeyi amaçlar.<br>
                Amaç, topluma yeniden uyum sürecini iyileştirecek stratejiler geliştirmek ve risk analizi yaparak tekrar suç oranlarını azaltmaya katkı sağlamaktır.
            </p>
            <h3 style="margin-top: 1.7rem; color:#90caf9;">Veri Seti Hakkında</h3>
            <p style="line-height:1.6; font-size:1.1rem;">
                Veri seti, mahpusların demografik bilgileri, ceza süreleri, geçmiş suç kayıtları ve yeniden suç işleme bilgilerini içermektedir.<br>
                Bu bilgilerle risk faktörleri analiz edilip, model geliştirme için zengin bir kaynak sağlanmıştır.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    if df is None:
        st.error("Veri seti yüklenemedi. Lütfen dosyanın doğru yerde ve formatta olduğundan emin olun.")
        return

    st.write(f"Toplam kayıt sayısı: {df.shape[0]:,}")
    st.write(f"Sütun sayısı: {df.shape[1]}")

    # Daha detaylı grafik, info kart vs. ekleyebilirsin buraya.

if __name__ == "__main__":
    main()
