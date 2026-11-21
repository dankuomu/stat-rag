import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stat_rag.data import PROJECT_ROOT, QaSet, chunk_docs, load_hotpot, load_squad, relevant_chunks
from stat_rag.metrics import contains_answer, exact_match, token_f1
from stat_rag.reader import ExtractiveReader
from stat_rag.retrieval import Bm25Retriever, DenseRetriever, HybridRetriever


CHUNK_SIZES = [50, 100, 200]
TOP_KS = [1, 3, 5, 10]
DEPTH = max(TOP_KS)
CORRECT_F1 = 0.5
LOGS_DIR = PROJECT_ROOT / "data" / "logs"

logger = logging.getLogger(__name__)


class GridRunner:
    """
    Прогон RAG (ретривер + экстрактивный ридер) по сетке параметров.

    Пишет два лога: поиска (ранжированная выдача top-DEPTH с ответом ридера на каждый чанк)
    и ответов (строка на вопрос × конфигурацию с метриками).

    :param qa_sets: выборки датасетов
    :param reader: ридер
    :param dense_model: модель эмбеддингов
    """

    def __init__(self, qa_sets: list[QaSet], reader: ExtractiveReader, dense_model):
        self._log = logger.getChild(self.__class__.__name__)
        self.qa_sets = qa_sets
        self.reader = reader
        self.dense_model = dense_model

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        """
        Полный прогон.

        :return: (лог поиска, лог ответов, метрики прогона)
        """
        t0 = time.time()
        search_logs, answer_logs = [], []
        reader_pairs = 0
        for qa in self.qa_sets:
            for chunk_words in CHUNK_SIZES:
                s_log, a_log, n_pairs = self._run_one(qa, chunk_words)
                search_logs.append(s_log)
                answer_logs.append(a_log)
                reader_pairs += n_pairs
        search = pd.concat(search_logs, ignore_index=True)
        answers = pd.concat(answer_logs, ignore_index=True)
        metrics = {
            "total_time": time.time() - t0,
            "n_answer_rows": len(answers),
            "n_search_rows": len(search),
            "reader_pairs": reader_pairs,
            "configs": answers.groupby(["dataset", "retriever", "chunk_size", "top_k"]).ngroups,
        }
        return search, answers, metrics

    def _run_one(self, qa: QaSet, chunk_words: int):
        chunks = chunk_docs(qa.docs, chunk_words)
        rel = relevant_chunks(qa, chunks)
        texts = chunks["text"].tolist()
        self._log.info(f"{qa.name} chunk={chunk_words}: {len(chunks)} chunks")
        bm25 = Bm25Retriever(texts)
        dense = DenseRetriever(texts, self.dense_model)
        retrievers = [bm25, dense, HybridRetriever(bm25, dense)]
        questions = qa.questions
        q_texts = questions["question"].tolist()

        ranked = {}
        for r in retrievers:
            t = time.time()
            idx, scores = r.search(q_texts, DEPTH)
            ranked[r.name] = (idx, scores, (time.time() - t) / len(q_texts))

        pairs = sorted({(qi, int(c)) for idx, _, _ in ranked.values() for qi, row in enumerate(idx) for c in row})
        ans, sc = self.reader.read([q_texts[qi] for qi, _ in pairs], [texts[c] for _, c in pairs])
        read = {p: (a, s) for p, a, s in zip(pairs, ans, sc)}
        self._log.info(f"{qa.name} chunk={chunk_words}: reader on {len(pairs)} pairs")

        search_rows, answer_rows = [], []
        for name, (idx, scores, latency) in ranked.items():
            for qi, q in enumerate(questions.itertuples()):
                gold_rel = rel[q.qid]
                ids = [int(c) for c in idx[qi]]
                for rank, (c, s) in enumerate(zip(ids, scores[qi]), start=1):
                    a, rs = read[(qi, c)]
                    search_rows.append(
                        {
                            "dataset": qa.name, "qid": q.qid, "retriever": name, "chunk_size": chunk_words,
                            "rank": rank, "chunk_id": c, "doc_id": chunks.at[c, "doc_id"],
                            "retrieval_score": float(s), "is_relevant": c in gold_rel,
                            "has_answer": contains_answer(texts[c], q.answers),
                            "reader_answer": a, "reader_score": rs,
                        }
                    )
                first_rel = next((i + 1 for i, c in enumerate(ids) if c in gold_rel), None)
                for k in TOP_KS:
                    top = ids[:k]
                    best = max(top, key=lambda c: read[(qi, c)][1])
                    pred, pred_score = read[(qi, best)]
                    found_docs = {gold_rel[c] for c in top if c in gold_rel}
                    f1 = token_f1(pred, q.answers)
                    answer_rows.append(
                        {
                            "dataset": qa.name, "qid": q.qid, "retriever": name, "chunk_size": chunk_words,
                            "top_k": k, "q_type": q.q_type, "level": q.level, "q_len": q.q_len,
                            "answer_len": q.answer_len, "answer_is_numeric": q.answer_is_numeric,
                            "lex_overlap": q.lex_overlap, "n_chunks_corpus": len(chunks),
                            "context_words": int(chunks.loc[top, "n_words"].sum()),
                            "hit_at_k": float(first_rel is not None and first_rel <= k),
                            "recall_at_k": len(found_docs) / len(q.gold_docs),
                            "mrr_at_k": 1.0 / first_rel if first_rel and first_rel <= k else 0.0,
                            "answer_in_context": float(any(contains_answer(texts[c], q.answers) for c in top)),
                            "top1_score": float(scores[qi][0]),
                            "score_margin": float(scores[qi][0] - scores[qi][1]),
                            "reader_score": pred_score, "pred_rank": top.index(best) + 1,
                            "prediction": pred, "gold": q.answers[0],
                            "em": exact_match(pred, q.answers), "f1": f1, "correct": int(f1 >= CORRECT_F1),
                            "latency_ms": latency * 1000,
                        }
                    )
        return pd.DataFrame(search_rows), pd.DataFrame(answer_rows), len(pairs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    qa_sets = [load_hotpot(n_per_level=200), load_squad(n=600)]
    for qa in qa_sets:
        qa.questions.to_parquet(LOGS_DIR / f"questions_{qa.name}.parquet")
    runner = GridRunner(qa_sets, ExtractiveReader(), DenseRetriever.load_model())
    search, answers, metrics = runner.run()
    search.to_parquet(LOGS_DIR / "search_log.parquet")
    answers.to_parquet(LOGS_DIR / "answer_log.parquet")
    print(f"Время прогона: {metrics['total_time']:.1f} с")
    print(f"Конфигураций: {metrics['configs']}")
    print(f"Строк в логе ответов: {metrics['n_answer_rows']}")
    print(f"Строк в логе поиска: {metrics['n_search_rows']}")
    print(f"Пар для ридера: {metrics['reader_pairs']}")
    print(answers.groupby(["dataset", "retriever", "chunk_size", "top_k"])[["recall_at_k", "em", "f1"]].mean().round(3))
