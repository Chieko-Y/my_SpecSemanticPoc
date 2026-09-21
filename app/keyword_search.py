"""
キーワード一致(TF-IDF)検索。意味検索(ベクトル)が苦手な型番・固有名詞・数値の
取りこぼしを補うためのフォールバック。

形態素解析の追加ライブラリは使わず、次の単純なルールで文字列を「語」に分ける:
- 英数字の連続(例: "CVT", "e-Power", "2.0")は1語
- 日本語などの文字は2文字ずつ重ねて切る(bigram。例: "電動パーキング" → "電動","動パ","パー",...)
"""

import math
import re
import unicodedata
from collections import Counter

import numpy as np

_ASCII_TOKEN = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")
_HAS_DIGIT_AND_ALPHA = re.compile(r"(?=.*[a-z])(?=.*\d)[a-z0-9\-.]+")


def _is_cjk_like(ch: str) -> bool:
    return ch.isalnum() and not ch.isascii()


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).lower()
    tokens: list[str] = []

    # 英数字トークン
    for m in _ASCII_TOKEN.finditer(text):
        tokens.append(m.group())

    # 日本語などは連続部分ごとにbigram(1文字だけの塊はそのまま)
    run: list[str] = []
    for ch in text + " ":
        if _is_cjk_like(ch):
            run.append(ch)
            continue
        if run:
            if len(run) == 1:
                tokens.append(run[0])
            else:
                tokens.extend(run[i] + run[i + 1] for i in range(len(run) - 1))
            run = []
    return tokens


def looks_like_model_code(query: str) -> bool:
    """「CVT-2000」「R18」のように英字と数字が混ざった語(型番らしきもの)を含むか。"""
    text = unicodedata.normalize("NFKC", query).lower()
    return any(_HAS_DIGIT_AND_ALPHA.fullmatch(t) for t in _ASCII_TOKEN.findall(text))


class KeywordIndex:
    def __init__(self, texts: list[str]) -> None:
        self.vocab: dict[str, int] = {}
        doc_tokens = [Counter(tokenize(t)) for t in texts]
        for counts in doc_tokens:
            for tok in counts:
                self.vocab.setdefault(tok, len(self.vocab))

        n_docs = len(texts)
        df = np.zeros(len(self.vocab), dtype=np.float32)
        for counts in doc_tokens:
            for tok in counts:
                df[self.vocab[tok]] += 1
        self.idf = np.log((1 + n_docs) / (1 + df)) + 1.0

        # 文書ごとのスパースベクトル(index配列, 重み配列)。L2正規化済み
        self.doc_vectors: list[tuple[np.ndarray, np.ndarray]] = []
        for counts in doc_tokens:
            self.doc_vectors.append(self._vectorize(counts))

        # 検索を高速にするための転置索引: 語 → [(文書番号, 重み), ...]
        self.postings: dict[int, list[tuple[int, float]]] = {}
        for doc_id, (idx, w) in enumerate(self.doc_vectors):
            for t, weight in zip(idx.tolist(), w.tolist()):
                self.postings.setdefault(t, []).append((doc_id, weight))

        self.n_docs = n_docs

    def _vectorize(self, counts: Counter) -> tuple[np.ndarray, np.ndarray]:
        idx, weights = [], []
        for tok, c in counts.items():
            t = self.vocab.get(tok)
            if t is None:
                continue
            idx.append(t)
            weights.append((1 + math.log(c)) * float(self.idf[t]))
        if not idx:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        w = np.array(weights, dtype=np.float32)
        w /= np.linalg.norm(w)
        return np.array(idx, dtype=np.int64), w

    def scores(self, query: str) -> np.ndarray:
        """全チャンクに対するコサイン類似度(0〜1)。語が1つも一致しなければ全部0。"""
        scores = np.zeros(self.n_docs, dtype=np.float32)
        q_idx, q_w = self._vectorize(Counter(tokenize(query)))
        for t, qw in zip(q_idx.tolist(), q_w.tolist()):
            for doc_id, dw in self.postings.get(t, ()):
                scores[doc_id] += qw * dw
        return scores
