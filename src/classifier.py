"""Memuat model Skip-gram + Gaussian Naive Bayes dan memprediksi label berita."""
import json
import os

import joblib
import numpy as np
from gensim.models import Word2Vec

from .preprocessing import preprocess_modular


class BeritaClassifier:
    def __init__(self, dir_model: str):
        self.w2v = Word2Vec.load(os.path.join(dir_model, "skipgram_word2vec.model"))
        self.nb = joblib.load(os.path.join(dir_model, "gaussian_nb.joblib"))
        with open(os.path.join(dir_model, "meta.json"), encoding="utf-8") as f:
            self.meta = json.load(f)

        PETA_PARAM = {
            "case_folding":     "do_case_folding",
            "hapus_tanda_baca": "do_hapus_tanda_baca",
            "hapus_angka":      "do_hapus_angka",
            "hapus_stopword":   "do_stopwords",
            "stemming":         "do_stemming",
        }
        self.config = {PETA_PARAM.get(k, k): v
                    for k, v in self.meta["preprocessing_config"].items()}
        self.id_ke_label = {v: k for k, v in self.meta["peta_label"].items()}

    def _dokumen_ke_vektor(self, tokens):
        """Sama dengan dokumen_ke_vektor di notebook (rata-rata vektor kata, OOV dilewati)."""
        vektor = [self.w2v.wv[t] for t in tokens if t in self.w2v.wv]
        if not vektor:
            return np.zeros(self.w2v.vector_size)
        return np.mean(vektor, axis=0)

    def predict(self, teks: str) -> dict:
        tokens = preprocess_modular(teks, **self.config)
        in_vocab = [t for t in tokens if t in self.w2v.wv]

        x = self._dokumen_ke_vektor(tokens).reshape(1, -1)
        proba = self.nb.predict_proba(x)[0]
        idx = int(np.argmax(proba))
        kelas_id = int(self.nb.classes_[idx])

        return {
            "label": self.id_ke_label[kelas_id],
            "proba": {self.id_ke_label[int(c)]: float(p)
                      for c, p in zip(self.nb.classes_, proba)},
            "tokens": tokens,
            "n_token": len(tokens),
            "n_in_vocab": len(in_vocab),
            "rasio_vocab": (len(in_vocab) / len(tokens)) if tokens else 0.0,
        }
