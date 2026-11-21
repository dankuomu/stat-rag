import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from stat_rag.metrics import contains_answer, normalize_answer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
HOTPOT_TRAIN = DATA_RAW / "hotpotqa_distractor_train.parquet"
SQUAD_DEV = DATA_RAW / "squad_v1_dev.parquet"
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
STOPWORDS = frozenset(
    "a an the of in on at to for from by with and or is are was were be been what which who whom "
    "whose when where why how did do does that this these those as it its into than then also".split()
)

logger = logging.getLogger(__name__)


@dataclass
class QaSet:
    """
    Выборка вопросов и корпус документов одного датасета.

    :param name: имя датасета
    :param questions: вопросы (qid, question, answers, gold_docs, ...)
    :param docs: документы (doc_id, title, sentences)
    """

    name: str
    questions: pd.DataFrame
    docs: pd.DataFrame


def content_tokens(text: str) -> set[str]:
    """Значимые токены строки без стоп-слов."""
    return {t for t in normalize_answer(text).split() if t not in STOPWORDS}


def load_hotpot(n_per_level: int = 200, seed: int = 42) -> QaSet:
    """
    Выборка HotpotQA (distractor) поровну по уровням сложности, без ответов yes/no.

    Корпус — объединение всех абзацев (2 нужных + 8 отвлекающих) выбранных вопросов.

    :param n_per_level: вопросов на уровень easy/medium/hard
    :param seed: зерно выборки
    :return: QaSet
    """
    raw = pd.read_parquet(HOTPOT_TRAIN)
    raw = raw[~raw["answer"].str.lower().isin(["yes", "no"])]
    sample = pd.concat(
        [raw[raw["level"] == lvl].sample(n=n_per_level, random_state=seed) for lvl in ["easy", "medium", "hard"]],
        ignore_index=True,
    )
    docs = {}
    rows = []
    for rec in sample.itertuples():
        for title, sents in zip(rec.context["title"], rec.context["sentences"]):
            docs.setdefault(title, list(sents))
        gold_titles = sorted(set(rec.supporting_facts["title"]))
        gold_sents = [
            f"{t}#{s}" for t, s in zip(rec.supporting_facts["title"], rec.supporting_facts["sent_id"])
        ]
        gold_text = " ".join(" ".join(docs[t]) for t in gold_titles if t in docs)
        q_tokens = content_tokens(rec.question)
        rows.append(
            {
                "qid": rec.id,
                "question": rec.question,
                "answers": [rec.answer],
                "gold_docs": gold_titles,
                "gold_sents": gold_sents,
                "q_type": rec.type,
                "level": rec.level,
                "q_len": len(rec.question.split()),
                "answer_len": len(rec.answer.split()),
                "answer_is_numeric": bool(re.search(r"\d", rec.answer)),
                "lex_overlap": len(q_tokens & content_tokens(gold_text)) / max(len(q_tokens), 1),
            }
        )
    docs_df = pd.DataFrame(
        [{"doc_id": t, "title": t, "sentences": s} for t, s in docs.items()]
    )
    logger.info(f"HotpotQA: {len(rows)} questions, {len(docs_df)} docs")
    return QaSet("hotpotqa", pd.DataFrame(rows), docs_df)


def load_squad(n: int = 600, seed: int = 42) -> QaSet:
    """
    Выборка SQuAD v1.1 dev; корпус — все абзацы dev-части.

    :param n: число вопросов
    :param seed: зерно выборки
    :return: QaSet
    """
    raw = pd.read_parquet(SQUAD_DEV)
    paragraphs = raw[["title", "context"]].drop_duplicates("context").reset_index(drop=True)
    paragraphs["doc_id"] = [f"sq{i:05d}" for i in range(len(paragraphs))]
    doc_by_context = dict(zip(paragraphs["context"], paragraphs["doc_id"]))
    sample = raw.sample(n=n, random_state=seed).reset_index(drop=True)
    rows = []
    for rec in sample.itertuples():
        answers = list(dict.fromkeys(rec.answers["text"]))
        q_tokens = content_tokens(rec.question)
        rows.append(
            {
                "qid": rec.id,
                "question": rec.question,
                "answers": answers,
                "gold_docs": [doc_by_context[rec.context]],
                "gold_sents": [],
                "q_type": "single",
                "level": "na",
                "q_len": len(rec.question.split()),
                "answer_len": len(answers[0].split()),
                "answer_is_numeric": bool(re.search(r"\d", answers[0])),
                "lex_overlap": len(q_tokens & content_tokens(rec.context)) / max(len(q_tokens), 1),
            }
        )
    docs_df = pd.DataFrame(
        {
            "doc_id": paragraphs["doc_id"],
            "title": paragraphs["title"].str.replace("_", " "),
            "sentences": [SENTENCE_RE.split(c.strip()) for c in paragraphs["context"]],
        }
    )
    logger.info(f"SQuAD: {len(rows)} questions, {len(docs_df)} docs")
    return QaSet("squad", pd.DataFrame(rows), docs_df)


def chunk_docs(docs: pd.DataFrame, chunk_words: int) -> pd.DataFrame:
    """
    Нарезка документов на чанки по предложениям до chunk_words слов.

    Предложение не режется: чанк закрывается, когда следующее предложение не влезает.
    К тексту чанка спереди приписан заголовок документа.

    :param docs: документы (doc_id, title, sentences)
    :param chunk_words: целевой размер чанка в словах
    :return: чанки (chunk_id, doc_id, title, sent_keys, text, n_words)
    """
    rows = []
    for doc in docs.itertuples():
        buf, keys, size = [], [], 0
        for i, sent in enumerate(doc.sentences):
            words = len(sent.split())
            if buf and size + words > chunk_words:
                rows.append((doc.doc_id, doc.title, keys, " ".join(buf).strip()))
                buf, keys, size = [], [], 0
            buf.append(sent)
            keys.append(f"{doc.doc_id}#{i}")
            size += words
        if buf:
            rows.append((doc.doc_id, doc.title, keys, " ".join(buf).strip()))
    chunks = pd.DataFrame(rows, columns=["doc_id", "title", "sent_keys", "body"])
    chunks["text"] = chunks["title"] + ". " + chunks["body"]
    chunks["n_words"] = chunks["body"].str.split().str.len()
    chunks["chunk_id"] = np.arange(len(chunks))
    return chunks


def relevant_chunks(qa: QaSet, chunks: pd.DataFrame) -> dict[str, dict[int, str]]:
    """
    Разметка релевантных чанков для каждого вопроса.

    HotpotQA: чанк релевантен, если содержит опорное предложение (supporting fact).
    SQuAD: чанк эталонного абзаца, содержащий ответ; если ответ разрезан — все чанки абзаца.

    :param qa: выборка
    :param chunks: чанки
    :return: qid -> {chunk_id: doc_id}
    """
    by_doc = {d: g for d, g in chunks.groupby("doc_id")}
    result = {}
    for q in qa.questions.itertuples():
        rel = {}
        for doc_id in q.gold_docs:
            part = by_doc.get(doc_id)
            if part is None:
                continue
            if qa.name == "hotpotqa":
                gold = set(q.gold_sents)
                for c in part.itertuples():
                    if gold & set(c.sent_keys):
                        rel[c.chunk_id] = doc_id
            else:
                hits = [c for c in part.itertuples() if contains_answer(c.body, q.answers)]
                for c in hits or list(part.itertuples()):
                    rel[c.chunk_id] = doc_id
        result[q.qid] = rel
    return result
