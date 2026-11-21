import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stat_rag.data import QaSet, chunk_docs, relevant_chunks


def make_docs():
    return pd.DataFrame(
        {
            "doc_id": ["A", "B"],
            "title": ["A", "B"],
            "sentences": [
                ["one two three four.", "five six seven eight.", "Paris is the capital."],
                ["alpha beta.", "gamma delta."],
            ],
        }
    )


def test_chunk_docs_packs_sentences_without_splitting():
    chunks = chunk_docs(make_docs(), chunk_words=8)
    a = chunks[chunks.doc_id == "A"]
    assert len(a) == 2
    assert a.iloc[0].sent_keys == ["A#0", "A#1"]
    assert a.iloc[1].sent_keys == ["A#2"]
    assert all(t.startswith(f"{d}. ") for t, d in zip(chunks.text, chunks.title))


def test_relevant_chunks_hotpot_uses_supporting_sentences():
    chunks = chunk_docs(make_docs(), chunk_words=8)
    q = pd.DataFrame([{"qid": "q1", "answers": ["Paris"], "gold_docs": ["A"], "gold_sents": ["A#2"]}])
    rel = relevant_chunks(QaSet("hotpotqa", q, make_docs()), chunks)
    assert list(rel["q1"].values()) == ["A"]
    assert chunks.loc[list(rel["q1"])[0], "sent_keys"] == ["A#2"]


def test_relevant_chunks_squad_uses_answer_or_whole_doc():
    chunks = chunk_docs(make_docs(), chunk_words=8)
    q = pd.DataFrame([
        {"qid": "q1", "answers": ["Paris"], "gold_docs": ["A"], "gold_sents": []},
        {"qid": "q2", "answers": ["missing"], "gold_docs": ["B"], "gold_sents": []},
    ])
    rel = relevant_chunks(QaSet("squad", q, make_docs()), chunks)
    assert len(rel["q1"]) == 1
    assert set(rel["q2"]) == set(chunks[chunks.doc_id == "B"].chunk_id)
