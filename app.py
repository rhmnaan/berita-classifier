"""
Aplikasi Streamlit: Klasifikasi Berita (sport vs finance)
Model : Skip-gram (Word2Vec) + Gaussian Naive Bayes  (hasil notebook, skema P1)
Input : link berita dari situs APA SAJA

Cara menjalankan:
    pip install -r requirements.txt
    streamlit run app.py

Struktur folder:
    app.py
    requirements.txt
    model_tersimpan/            <- dibuat oleh notebook (bagian 11)
        skipgram_word2vec.model
        gaussian_nb.joblib
        meta.json
"""

import os
import re
import json

import joblib
import numpy as np
import requests
import streamlit as st
from bs4 import BeautifulSoup
from gensim.models import Word2Vec

DIR_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_tersimpan")


# =====================================================================
# 1. MUAT MODEL (sekali saja, lalu disimpan di cache Streamlit)
# =====================================================================
@st.cache_resource
def muat_model():
    w2v = Word2Vec.load(os.path.join(DIR_MODEL, "skipgram_word2vec.model"))
    nb = joblib.load(os.path.join(DIR_MODEL, "gaussian_nb.joblib"))
    with open(os.path.join(DIR_MODEL, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    return w2v, nb, meta


# =====================================================================
# 2. AMBIL TEKS ARTIKEL DARI LINK (berlaku untuk situs berita apa saja)
# =====================================================================
def ambil_teks_dari_url(url):
    """Download halaman web, lalu ambil judul dan isi artikelnya."""
    # User-Agent supaya kita terlihat seperti browser biasa
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    respon = requests.get(url, headers=headers, timeout=15)
    respon.raise_for_status()  # error jika halaman gagal dibuka (404, 403, dst)

    soup = BeautifulSoup(respon.content, "html.parser")

    # Buang bagian yang bukan isi berita (menu, iklan, footer, script)
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
        tag.decompose()

    judul = soup.title.get_text(strip=True) if soup.title else "(tanpa judul)"

    # Hampir semua situs berita menaruh isi artikel di tag <p> (paragraf).
    # Paragraf sangat pendek (menu, caption, "baca juga") kita abaikan.
    paragraf = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    paragraf = [p for p in paragraf if len(p.split()) >= 5]

    return judul, " ".join(paragraf)


# =====================================================================
# 3. PREPROCESSING P1 (sama persis dengan notebook)
# =====================================================================
def preprocess(teks):
    teks = str(teks).lower()               # huruf kecil
    teks = re.sub(r"[^\w\s]", " ", teks)   # hapus tanda baca
    return teks.split()                    # pecah jadi token (kata)


# =====================================================================
# 4. TEKS -> VEKTOR DOKUMEN (rata-rata vektor kata, sama dengan notebook)
# =====================================================================
def dokumen_ke_vektor(tokens, w2v):
    vektor = [w2v.wv[t] for t in tokens if t in w2v.wv]  # kata di luar kosakata dilewati
    if vektor:
        return np.mean(vektor, axis=0)
    return np.zeros(w2v.vector_size)


# =====================================================================
# 5. PREDIKSI + TAMPILKAN HASIL
# =====================================================================
def prediksi_dan_tampilkan(judul, teks, w2v, nb, meta):
    tokens = preprocess(teks)

    if len(tokens) == 0:
        st.error("Teks kosong. Tidak ada yang bisa diklasifikasi.")
        return

    # Berapa persen kata yang dikenal model? (untuk menilai apakah hasil layak dipercaya)
    kata_dikenal = [t for t in tokens if t in w2v.wv]
    persen_dikenal = len(kata_dikenal) / len(tokens) * 100

    # Pipeline: tokens -> vektor dokumen -> Naive Bayes
    vektor = dokumen_ke_vektor(tokens, w2v).reshape(1, -1)
    id_kelas = int(nb.predict(vektor)[0])
    peluang = nb.predict_proba(vektor)[0]

    id_ke_label = {v: k for k, v in meta["peta_label"].items()}
    label = id_ke_label[id_kelas]

    # ---- Hasil utama ----
    st.subheader("Hasil Klasifikasi")
    st.caption(f"Judul halaman: {judul}")

    if label == "sport":
        st.success("Kategori: SPORT")
    else:
        st.info("Kategori: FINANCE")

    kolom1, kolom2 = st.columns(2)
    kolom1.metric("Peluang sport", f"{peluang[meta['peta_label']['sport']] * 100:.1f}%")
    kolom2.metric("Peluang finance", f"{peluang[meta['peta_label']['finance']] * 100:.1f}%")

    # ---- Peringatan jika hasil kurang bisa dipercaya ----
    if len(tokens) < 30:
        st.warning("Teks terlalu pendek (< 30 kata). Hasil kurang bisa dipercaya.")
    if persen_dikenal < 50:
        st.warning(
            f"Hanya {persen_dikenal:.0f}% kata yang dikenal model. "
            "Kemungkinan artikel ini bukan topik sport/finance."
        )
    st.caption(
        "Catatan: model hanya mengenal 2 kategori (sport dan finance). "
        "Artikel politik, hiburan, dll. tetap akan dipaksa masuk ke salah satunya."
    )

    # ---- Tahapan proses (untuk presentasi) ----
    with st.expander("Lihat tahapan proses"):
        st.markdown("**Tahap 1 — Teks asli (300 karakter pertama)**")
        st.write(teks[:300] + "...")

        st.markdown("**Tahap 2 — Preprocessing: huruf kecil + hapus tanda baca + tokenisasi**")
        st.write(tokens[:15])
        st.write(f"Jumlah kata: {len(tokens)} | Kata dikenal model: {len(kata_dikenal)} ({persen_dikenal:.0f}%)")

        st.markdown("**Tahap 3 — Skip-gram: tiap kata jadi vektor, lalu dirata-rata = vektor dokumen**")
        st.write(f"Panjang vektor dokumen: {vektor.shape[1]} dimensi. 5 angka pertama:")
        st.write(np.round(vektor[0][:5], 4))

        st.markdown("**Tahap 4 — Gaussian Naive Bayes memilih kelas dengan peluang tertinggi**")
        st.write(f"Hasil: **{label}**")

    with st.expander("Lihat teks artikel yang berhasil diambil"):
        st.write(teks)


# =====================================================================
# 6. TAMPILAN HALAMAN
# =====================================================================
st.set_page_config(page_title="Klasifikasi Berita", page_icon="📰")
st.title("📰 Klasifikasi Berita: Sport vs Finance")
st.write("Model: **Skip-gram (Word2Vec) + Gaussian Naive Bayes**. Bisa dipakai untuk link berita dari situs apa saja.")

# Muat model; hentikan aplikasi dengan pesan jelas jika file model tidak ada
try:
    w2v, nb, meta = muat_model()
except Exception as e:
    st.error(f"Model belum ditemukan di folder '{DIR_MODEL}'. Jalankan dulu bagian 11 notebook. ({e})")
    st.stop()

with st.sidebar:
    st.header("Info Model")
    st.write(f"Preprocessing: {meta['preprocessing']}")
    st.write(f"Akurasi uji: {meta['akurasi_uji'] * 100:.1f}%")
    st.write(f"Ukuran kosakata: {len(w2v.wv)} kata")
    st.write(f"Dimensi vektor: {w2v.vector_size}")

url = st.text_input("Tempel link berita (situs apa saja)", placeholder="https://...")

if st.button("Klasifikasikan", type="primary"):
    if not url.startswith("http"):
        st.error("Link harus diawali http:// atau https://")
    else:
        try:
            with st.spinner("Mengambil artikel..."):
                judul, teks = ambil_teks_dari_url(url)
            if len(teks.split()) < 20:
                st.warning(
                    "Teks yang berhasil diambil sangat sedikit. Mungkin situsnya memblokir "
                    "scraping atau isinya dimuat dengan JavaScript. Coba link berita lain."
                )
            else:
                prediksi_dan_tampilkan(judul, teks, w2v, nb, meta)
        except requests.exceptions.RequestException as e:
            st.error(f"Gagal membuka link: {e}")