"""Preprocessing -- HARUS identik dengan yang dipakai saat training di notebook (bagian 5)."""
import re
from functools import lru_cache

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


@lru_cache(maxsize=1)
def _stopwords():
    return set(StopWordRemoverFactory().get_stop_words())


@lru_cache(maxsize=1)
def _stemmer():
    return StemmerFactory().create_stemmer()


_cache_stem = {}


def case_folding(text):
    return str(text).lower()


def hapus_tanda_baca(text):
    return re.sub(r"[^\w\s]", " ", text)


def hapus_angka(text):
    return re.sub(r"\d+", " ", text)


def tokenisasi(text):
    return re.sub(r"\s+", " ", text).strip().split()


def hapus_stopwords(tokens):
    sw = _stopwords()
    return [t for t in tokens if t not in sw]


def stemming(tokens):
    stemmer = _stemmer()
    hasil = []
    for t in tokens:
        if t not in _cache_stem:
            _cache_stem[t] = stemmer.stem(t)
        hasil.append(_cache_stem[t])
    return hasil


def preprocess_modular(text,
                       do_case_folding=True,
                       do_hapus_tanda_baca=True,
                       do_hapus_angka=True,
                       do_stopwords=True,
                       do_stemming=True,
                       min_len=2):
    """Teks mentah -> list of tokens."""
    text = str(text)
    if do_case_folding:
        text = case_folding(text)
    if do_hapus_tanda_baca:
        text = hapus_tanda_baca(text)
    if do_hapus_angka:
        text = hapus_angka(text)
    tokens = tokenisasi(text)
    if do_stopwords:
        tokens = hapus_stopwords(tokens)
    if do_stemming:
        tokens = stemming(tokens)
    return [t for t in tokens if len(t) >= min_len]
