import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stat_rag.data import QaSet, chunk_docs, load_hotpot, load_squad, relevant_chunks
from stat_rag.metrics import contains_answer, exact_match, normalize_answer, token_f1
from stat_rag.reader import ExtractiveReader
from stat_rag.retrieval import Bm25Retriever, DenseRetriever, HybridRetriever
from stat_rag.run_grid import CORRECT_F1, LOGS_DIR


CHUNK_WORDS = 100
STEP_K = 3
N_ITERATIONS = 4
SEARCH_DEPTH = 60

logger = logging.getLogger(__name__)


class IterativeRunner:
    """
    Итеративный поиск: на каждом шаге к запросу дописывается ответ ридера с прошлого шага,
    в контекст добавляются STEP_K новых чанков. Контроль — одношаговый поиск того же
    бюджета (top STEP_K * t).

    :param qa_sets: выборки датасетов
    :param reader: ридер
    :param dense_model: модель эмбеддингов
    """

    def __init__(self, qa_sets: list[QaSet], reader: ExtractiveReader, dense_model):
        self._log = logger.getChild(self.__class__.__name__)
        self.qa_sets = qa_sets
        self.reader = reader
        self.dense_model = dense_model

    def run(self) -> tuple[pd.DataFrame, dict]:
        """
        Прогон по всем датасетам.

        :return: (лог итераций, метрики прогона)
        """
        t0 = time.time()
        frames = [self._run_one(qa) for qa in self.qa_sets]
        log = pd.concat(frames, ignore_index=True)
        return log, {"total_time": time.time() - t0, "n_rows": len(log)}

    def _run_one(self, qa: QaSet) -> pd.DataFrame:
        chunks = chunk_docs(qa.docs, CHUNK_WORDS)
        rel = relevant_chunks(qa, chunks)
        texts = chunks["text"].tolist()
        bm25 = Bm25Retriever(texts)
        retriever = HybridRetriever(bm25, DenseRetriever(texts, self.dense_model))
        questions = qa.questions.reset_index(drop=True)
        q_texts = questions["question"].tolist()
        cache = {}

        def read(pairs):
            todo = [p for p in pairs if p not in cache]
            if todo:
                ans, sc = self.reader.read([q_texts[qi] for qi, _ in todo], [texts[c] for _, c in todo])
                cache.update({p: (a, s) for p, a, s in zip(todo, ans, sc)})
            return [cache[p] for p in pairs]

        def score_row(qi, used, mode, it, query):
            q = questions.loc[qi]
            outs = read([(qi, c) for c in used])
            pred, pred_score = max(outs, key=lambda x: x[1])
            gold_rel = rel[q.qid]
            found_docs = {gold_rel[c] for c in used if c in gold_rel}
            f1 = token_f1(pred, q.answers)
            return {
                "dataset": qa.name, "qid": q.qid, "mode": mode, "iteration": it,
                "q_type": q.q_type, "level": q.level, "query_len": len(query.split()),
                "context_words": int(chunks.loc[used, "n_words"].sum()),
                "recall": len(found_docs) / len(q.gold_docs),
                "hit": float(bool(found_docs)),
                "answer_in_context": float(any(contains_answer(texts[c], q.answers) for c in used)),
                "prediction": pred, "reader_score": pred_score,
                "em": exact_match(pred, q.answers), "f1": f1, "correct": int(f1 >= CORRECT_F1),
            }

        rows = []
        base_idx, _ = retriever.search(q_texts, STEP_K * N_ITERATIONS)
        for qi in range(len(questions)):
            for it in range(1, N_ITERATIONS + 1):
                used = [int(c) for c in base_idx[qi][: STEP_K * it]]
                rows.append(score_row(qi, used, "single", it, q_texts[qi]))

        queries = list(q_texts)
        used_sets = [[] for _ in q_texts]
        for it in range(1, N_ITERATIONS + 1):
            idx, _ = retriever.search(queries, SEARCH_DEPTH)
            new_per_q = []
            for qi in range(len(questions)):
                new = [int(c) for c in idx[qi] if int(c) not in used_sets[qi]][:STEP_K]
                used_sets[qi].extend(new)
                new_per_q.append(new)
            for qi in range(len(questions)):
                rows.append(score_row(qi, used_sets[qi], "iterative", it, queries[qi]))
                outs = read([(qi, c) for c in new_per_q[qi]])
                hop_answer = max(outs, key=lambda x: x[1])[0] if outs else ""
                if hop_answer and normalize_answer(hop_answer) not in normalize_answer(queries[qi]):
                    queries[qi] = f"{queries[qi]} {hop_answer}"
            self._log.info(f"{qa.name}: iteration {it} done")
        return pd.DataFrame(rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    runner = IterativeRunner([load_hotpot(n_per_level=200), load_squad(n=600)], ExtractiveReader(), DenseRetriever.load_model())
    log, metrics = runner.run()
    log.to_parquet(LOGS_DIR / "iteration_log.parquet")
    print(f"Время прогона: {metrics['total_time']:.1f} с")
    print(f"Строк в логе итераций: {metrics['n_rows']}")
    print(log.groupby(["dataset", "mode", "iteration"])[["recall", "answer_in_context", "f1"]].mean().round(3))
