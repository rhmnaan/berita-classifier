import html
import os
import traceback

import streamlit as st

from src.classifier import BeritaClassifier
from src.scraper import ScrapeError, ambil_berita

DIR_MODEL = os.path.join(os.path.dirname(__file__), "model_tersimpan")
RASIO_VOCAB_MIN = 0.4   # di bawah ini, berita dianggap kurang cocok dengan domain model

KATEGORI = {
    "sport":   {"nama": "Olahraga", "ikon": "⚽", "warna": "#16a34a", "bg": "rgba(22,163,74,0.10)"},
    "finance": {"nama": "Keuangan", "ikon": "💰", "warna": "#2563eb", "bg": "rgba(37,99,235,0.10)"},
}
DEFAULT_KAT = {"nama": "Lainnya", "ikon": "📰", "warna": "#6b7280", "bg": "rgba(107,114,128,0.10)"}

st.set_page_config(page_title="Klasifikasi Berita", page_icon="📰", layout="centered")

CSS = """
<style>
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 2.5rem; max-width: 760px;}

.hero {text-align: center; margin-bottom: 1.6rem;}
.hero h1 {font-size: 2.1rem; font-weight: 800; margin: 0 0 .3rem 0; letter-spacing: -0.5px;}
.hero p {opacity: .65; margin: 0; font-size: 1rem;}

.card {
    border: 1px solid rgba(128,128,128,0.25);
    border-radius: 16px; padding: 1.2rem 1.4rem; margin-top: 1.2rem;
    background: rgba(128,128,128,0.05);
}
.card-tag {font-size: .72rem; text-transform: uppercase; letter-spacing: 1.2px; opacity: .55; margin-bottom: .35rem;}
.card-title {font-size: 1.2rem; font-weight: 700; line-height: 1.35; margin-bottom: .5rem;}
.card-text {opacity: .8; font-size: .95rem; line-height: 1.6;}
.card-meta {margin-top: .7rem; font-size: .8rem; opacity: .55;}
.card-meta a {color: inherit;}

.result {
    display: flex; align-items: center; gap: 1.3rem;
    border-radius: 18px; padding: 1.4rem 1.6rem; margin-top: 1.2rem;
    border: 2px solid;
}
.result-icon {font-size: 3.4rem; line-height: 1;}
.result-small {font-size: .75rem; text-transform: uppercase; letter-spacing: 1.2px; opacity: .6;}
.result-label {font-size: 2.3rem; font-weight: 800; line-height: 1.15; letter-spacing: 1px;}
.result-sub {font-size: .95rem; opacity: .75; margin-top: .1rem;}

.bar-wrap {margin-top: .9rem;}
.bar-row {display: flex; align-items: center; gap: .8rem; margin-bottom: .55rem; font-size: .9rem;}
.bar-name {width: 80px; font-weight: 600;}
.bar-track {flex: 1; height: 12px; border-radius: 999px; background: rgba(128,128,128,0.18); overflow: hidden;}
.bar-fill {height: 100%; border-radius: 999px;}
.bar-pct {width: 56px; text-align: right; font-variant-numeric: tabular-nums; opacity: .8;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def _html(s: str) -> str:
    """Hilangkan indentasi agar tidak dibaca sebagai code block oleh markdown."""
    return "\n".join(line.strip() for line in s.strip().splitlines())


@st.cache_resource(show_spinner="Memuat model...")
def muat_model():
    return BeritaClassifier(DIR_MODEL)


def kartu_berita(berita: dict):
    judul = html.escape(berita["judul"] or "(judul tidak ditemukan)")
    cuplikan = berita["teks"][:320] + ("..." if len(berita["teks"]) > 320 else "")
    n_kata = len(berita["teks"].split())
    url = html.escape(berita["url"], quote=True)
    st.markdown(_html(f"""
    <div class="card">
        <div class="card-tag">Berita yang dianalisis</div>
        <div class="card-title">{judul}</div>
        <div class="card-text">{html.escape(cuplikan)}</div>
        <div class="card-meta">{n_kata} kata · <a href="{url}" target="_blank">Buka sumber ↗</a></div>
    </div>
    """), unsafe_allow_html=True)


def kartu_hasil(hasil: dict):
    label = hasil["label"]
    kat = KATEGORI.get(label, DEFAULT_KAT)
    keyakinan = hasil["proba"][label]

    st.markdown(_html(f"""
    <div class="result" style="border-color:{kat['warna']}; background:{kat['bg']};">
        <div class="result-icon">{kat['ikon']}</div>
        <div>
            <div class="result-small">Kategori berita</div>
            <div class="result-label" style="color:{kat['warna']};">{html.escape(label.upper())}</div>
            <div class="result-sub">{kat['nama']} · keyakinan {keyakinan:.1%}</div>
        </div>
    </div>
    """), unsafe_allow_html=True)

    baris = ""
    for nama, p in sorted(hasil["proba"].items(), key=lambda x: -x[1]):
        k = KATEGORI.get(nama, DEFAULT_KAT)
        baris += f"""
        <div class="bar-row">
            <div class="bar-name">{html.escape(nama.capitalize())}</div>
            <div class="bar-track"><div class="bar-fill" style="width:{p*100:.1f}%; background:{k['warna']};"></div></div>
            <div class="bar-pct">{p:.1%}</div>
        </div>"""
    st.markdown(_html(f"""
    <div class="card">
        <div class="card-tag">Probabilitas tiap kategori</div>
        <div class="bar-wrap">{baris}</div>
    </div>
    """), unsafe_allow_html=True)

    if hasil["rasio_vocab"] < RASIO_VOCAB_MIN:
        st.warning(
            "Sebagian besar isi berita ini tidak dikenali sistem. Sistem hanya mengenal kategori "
            "**sport** dan **finance**, jadi hasil di atas kemungkinan kurang tepat jika berita "
            "ini berasal dari kategori lain (politik, hiburan, dll.)."
        )


# ---------------------------------------------------------------- UI
st.markdown(_html("""
<div class="hero">
    <h1>📰 Klasifikasi Berita</h1>
    <p>Tempel link berita dari situs mana pun, sistem akan menentukan kategorinya.</p>
</div>
"""), unsafe_allow_html=True)

try:
    model = muat_model()
except Exception:
    st.error("Model gagal dimuat. Pastikan folder `model_tersimpan` berisi semua file model.")
    st.stop()

with st.form("form_link", border=False):
    url = st.text_input("Link berita", placeholder="https://www.contoh.com/berita/...",
                        label_visibility="collapsed")
    kirim = st.form_submit_button("🔍  Klasifikasikan", type="primary", use_container_width=True)

if kirim:
    if not url.strip():
        st.warning("Masukkan link berita terlebih dahulu.")
    else:
        try:
            with st.spinner("Mengambil dan menganalisis berita..."):
                berita = ambil_berita(url)
                hasil = model.predict(berita["teks"])
            kartu_hasil(hasil)
            kartu_berita(berita)
        except ScrapeError as e:
            st.error(str(e))
        except Exception as e:
            traceback.print_exc()   # tampil juga di terminal tempat streamlit dijalankan
            st.error("Terjadi kesalahan saat memproses berita.")
            with st.expander("Detail error"):
                st.code(f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}")
