"""Mengambil judul & isi berita dari link situs berita mana pun."""
import codecs
import ipaddress
import re
import socket
from collections import defaultdict
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

try:                      # ekstraktor artikel generik (disarankan)
    import trafilatura
except ImportError:       # tanpa trafilatura, dipakai ekstraktor cadangan di bawah
    trafilatura = None

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept-Language": "id-ID,id;q=0.9,en;q=0.8",
}
MAX_BYTES = 3_000_000     # batas ukuran halaman yang diunduh
MIN_KATA = 30             # isi berita lebih pendek dari ini dianggap gagal diekstrak

# selector isi artikel yang umum dipakai situs berita (dicoba berurutan oleh ekstraktor cadangan)
SELECTOR_ISI = [
    "div.detail__body-text", "div.itp_bodycontent", "div.detail_text",      # detik
    "[itemprop=articleBody]", "div.article-content", "div.entry-content",
    "div.post-content", "div.read__content", "article",
]
TAG_BUANG = ["script", "style", "noscript", "iframe", "nav", "header", "footer",
             "aside", "form", "figure", "table"]
KALIMAT_NOISE = ("advertisement", "scroll to continue", "baca juga", "simak video",
                 "lihat juga", "baca selengkapnya", "read also", "also read", "iklan")


class ScrapeError(Exception):
    """Error yang pesannya aman ditampilkan langsung ke pengguna."""


# ----------------------------------------------------------------- validasi URL
def _host_aman(host: str) -> bool:
    """Tolak alamat internal (localhost, jaringan lokal) agar app tidak bisa disalahgunakan."""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved or ip.is_multicast or ip.is_unspecified):
            return False
    return True


def validasi_url(url: str) -> str:
    url = (url or "").strip()
    if url and "://" not in url:
        url = "https://" + url
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        raise ScrapeError("Link tidak valid. Contoh: https://www.contoh.com/berita/...")
    if not _host_aman(p.hostname):
        raise ScrapeError("Link tidak dapat diakses atau mengarah ke alamat yang tidak diizinkan.")
    return url


def _decode(konten: bytes, enc_header) -> str:
    """Ubah bytes -> teks: charset dari header, lalu <meta charset>, lalu utf-8."""
    kandidat = []
    # requests memberi ISO-8859-1 sebagai tebakan bila header tidak menyebut charset -> abaikan
    if enc_header and "iso-8859" not in enc_header.lower():
        kandidat.append(enc_header)
    m = re.search(rb"<meta[^>]+charset=[\"']?\s*([\w-]+)", konten[:5000], re.I)
    if m:
        kandidat.append(m.group(1).decode("ascii", errors="ignore"))
    kandidat.append("utf-8")

    for enc in kandidat:
        try:
            codecs.lookup(enc)
        except LookupError:
            continue
        return konten.decode(enc, errors="replace")
    return konten.decode("utf-8", errors="replace")


# ----------------------------------------------------------------- unduh halaman
def _unduh(url: str, timeout: int) -> tuple[str, str]:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout, stream=True)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise ScrapeError(f"Gagal membuka halaman: {e}") from e

    validasi_url(resp.url)                         # cek lagi setelah redirect

    tipe = resp.headers.get("Content-Type", "").lower()
    if "html" not in tipe:
        raise ScrapeError("Link ini bukan halaman web (HTML) biasa.")

    konten = b""
    for potongan in resp.iter_content(chunk_size=65536):
        konten += potongan
        if len(konten) > MAX_BYTES:
            break
    resp.close()
    return _decode(konten, resp.encoding), resp.url


# ----------------------------------------------------------------- ekstraksi
def _bersihkan(teks: str) -> str:
    return re.sub(r"\s+", " ", teks).strip()


def _judul(soup: BeautifulSoup) -> str:
    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        return _bersihkan(og["content"])
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return _bersihkan(h1.get_text())
    if soup.title and soup.title.string:
        return _bersihkan(soup.title.string)
    return ""


def _paragraf(kontainer) -> str:
    hasil = []
    for p in kontainer.find_all("p"):
        t = _bersihkan(p.get_text(" "))
        if t and not t.lower().startswith(KALIMAT_NOISE):
            hasil.append(t)
    return " ".join(hasil)


def _ekstrak_cadangan(soup: BeautifulSoup) -> str:
    """Ekstraktor sederhana: selector umum, lalu elemen dengan teks paragraf terbanyak."""
    for t in TAG_BUANG:
        for el in soup.find_all(t):
            el.decompose()

    for sel in SELECTOR_ISI:
        el = soup.select_one(sel)
        if el:
            teks = _paragraf(el)
            if len(teks.split()) >= MIN_KATA:
                return teks

    skor = defaultdict(int)
    for p in soup.find_all("p"):
        t = _bersihkan(p.get_text(" "))
        if len(t) >= 40:
            skor[id(p.parent)] += len(t)
    if not skor:
        return ""
    elemen = {id(p.parent): p.parent for p in soup.find_all("p")}
    terbaik = elemen[max(skor, key=skor.get)]
    return _paragraf(terbaik)


def ambil_berita(url: str, timeout: int = 15) -> dict:
    """Return {'url', 'judul', 'teks'}. Raise ScrapeError bila gagal."""
    url = validasi_url(url)
    html_mentah, url_akhir = _unduh(url, timeout)

    soup = BeautifulSoup(html_mentah, "html.parser")
    judul = _judul(soup)

    teks = ""
    if trafilatura is not None:
        teks = trafilatura.extract(html_mentah, include_comments=False,
                                   include_tables=False, favor_precision=True) or ""
        teks = _bersihkan(teks)
    if len(teks.split()) < MIN_KATA:
        teks = _ekstrak_cadangan(BeautifulSoup(html_mentah, "html.parser"))

    if len(teks.split()) < MIN_KATA:
        raise ScrapeError(
            "Isi berita tidak berhasil diambil. Kemungkinan halaman bukan artikel, "
            "berbayar (paywall), atau kontennya dimuat lewat JavaScript. Coba link berita lain."
        )

    return {"url": url_akhir, "judul": judul, "teks": teks}
