import logging

import bm25s
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


DENSE_MODEL = "intfloat/e5-small-v2"
DENSE_REVISION = "ffb93f3bd4047442299a41ebb6fa998a38507c52"
RRF_K = 60
RRF_DEPTH = 50
ENCODE_BATCH = 64

logger = logging.getLogger(__name__)


def pick_device() -> str:
    """Устройство для нейросетей: mps, cuda или cpu."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Bm25Retriever:
    """
    Лексический поиск BM25 (bm25s, параметры по умолчанию k1=1.5, b=0.75).

    :param texts: тексты чанков
    """

    name = "bm25"

    def __init__(self, texts: list[str]):
        self._log = logger.getChild(self.__class__.__name__)
        self._n = len(texts)
        self._index = bm25s.BM25()
        self._index.index(self._tokenize(texts), show_progress=False)

    @staticmethod
    def _tokenize(texts: list[str]) -> list[list[str]]:
        return bm25s.tokenize(texts, stopwords="en", return_ids=False, show_progress=False)

    def search(self, queries: list[str], k: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Поиск top-k.

        :param queries: запросы
        :param k: глубина выдачи
        :return: (индексы чанков [n_q, k], оценки [n_q, k])
        """
        tokens = [t if t else ["empty"] for t in self._tokenize(queries)]
        idx, scores = self._index.retrieve(tokens, k=min(k, self._n), show_progress=False)
        return np.asarray(idx), np.asarray(scores, dtype=float)


class DenseRetriever:
    """
    Плотный поиск по эмбеддингам e5 (косинусная близость).

    :param texts: тексты чанков
    :param model: уже загруженная модель (чтобы не грузить на каждый размер чанка)
    """

    name = "dense"

    def __init__(self, texts: list[str], model: SentenceTransformer):
        self._log = logger.getChild(self.__class__.__name__)
        self._model = model
        self._emb = model.encode(
            [f"passage: {t}" for t in texts],
            batch_size=ENCODE_BATCH,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    @staticmethod
    def load_model() -> SentenceTransformer:
        """Загрузка модели эмбеддингов на доступное устройство."""
        return SentenceTransformer(DENSE_MODEL, revision=DENSE_REVISION, device=pick_device())

    def search(self, queries: list[str], k: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Поиск top-k.

        :param queries: запросы
        :param k: глубина выдачи
        :return: (индексы чанков [n_q, k], оценки [n_q, k])
        """
        q = self._model.encode(
            [f"query: {t}" for t in queries],
            batch_size=ENCODE_BATCH,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        sims = q @ self._emb.T
        k = min(k, sims.shape[1])
        idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        part = np.take_along_axis(sims, idx, axis=1)
        order = np.argsort(-part, axis=1)
        return np.take_along_axis(idx, order, axis=1), np.take_along_axis(part, order, axis=1)


class HybridRetriever:
    """
    Гибридный поиск: слияние BM25 и dense через Reciprocal Rank Fusion.

    :param bm25: лексический ретривер
    :param dense: плотный ретривер
    """

    name = "hybrid"

    def __init__(self, bm25: Bm25Retriever, dense: DenseRetriever):
        self._bm25 = bm25
        self._dense = dense

    def search(self, queries: list[str], k: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Поиск top-k по сумме 1 / (RRF_K + rank) из двух выдач глубины RRF_DEPTH.

        :param queries: запросы
        :param k: глубина выдачи
        :return: (индексы чанков [n_q, k], RRF-оценки [n_q, k])
        """
        depth = max(RRF_DEPTH, k)
        b_idx, _ = self._bm25.search(queries, depth)
        d_idx, _ = self._dense.search(queries, depth)
        out_idx, out_scores = [], []
        for bi, di in zip(b_idx, d_idx):
            fused = {}
            for ranked in (bi, di):
                for rank, c in enumerate(ranked, start=1):
                    fused[int(c)] = fused.get(int(c), 0.0) + 1.0 / (RRF_K + rank)
            top = sorted(fused.items(), key=lambda x: -x[1])[:k]
            out_idx.append([c for c, _ in top])
            out_scores.append([s for _, s in top])
        return np.asarray(out_idx), np.asarray(out_scores, dtype=float)
