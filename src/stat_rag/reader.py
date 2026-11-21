import logging

import numpy as np
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from stat_rag.retrieval import pick_device


READER_MODEL = "deepset/roberta-base-squad2"
READER_REVISION = "adc3b06f79f797d1c575d5479d6f5efe54a9e3b4"
MAX_LENGTH = 384
MAX_ANSWER_TOKENS = 30
BATCH_SIZE = 64

logger = logging.getLogger(__name__)


class ExtractiveReader:
    """
    Экстрактивный ридер: для пары (вопрос, чанк) находит лучший отрезок-ответ.

    Оценка отрезка — сумма логитов начала и конца (как в DrQA / BERT-QA).

    :param model_name: модель с головой question answering
    """

    def __init__(self, model_name: str = READER_MODEL, revision: str = READER_REVISION):
        self._log = logger.getChild(self.__class__.__name__)
        self._device = pick_device()
        self._tok = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self._model = AutoModelForQuestionAnswering.from_pretrained(model_name, revision=revision).to(self._device).eval()
        self._log.info(f"reader {model_name} on {self._device}")

    @torch.no_grad()
    def read(self, questions: list[str], contexts: list[str]) -> tuple[list[str], np.ndarray]:
        """
        Ответы для списка пар.

        :param questions: вопросы
        :param contexts: тексты чанков той же длины
        :return: (ответы, оценки)
        """
        answers, scores = [], []
        for start in range(0, len(questions), BATCH_SIZE):
            q = questions[start:start + BATCH_SIZE]
            c = contexts[start:start + BATCH_SIZE]
            enc = self._tok(
                q,
                c,
                truncation="only_second",
                max_length=MAX_LENGTH,
                padding=True,
                return_offsets_mapping=True,
                return_tensors="pt",
            )
            offsets = enc.pop("offset_mapping").numpy()
            seq_ids = [enc.sequence_ids(i) for i in range(len(q))]
            out = self._model(**{k: v.to(self._device) for k, v in enc.items()})
            start_logits = out.start_logits.float().cpu().numpy()
            end_logits = out.end_logits.float().cpu().numpy()
            for i in range(len(q)):
                ans, score = self._best_span(start_logits[i], end_logits[i], offsets[i], seq_ids[i], c[i])
                answers.append(ans)
                scores.append(score)
        return answers, np.asarray(scores, dtype=float)

    @staticmethod
    def _best_span(start, end, offsets, seq_ids, context):
        ctx = np.array([s == 1 for s in seq_ids])
        s_log = np.where(ctx, start, -1e9)
        e_log = np.where(ctx, end, -1e9)
        top_s = np.argsort(-s_log)[:20]
        top_e = np.argsort(-e_log)[:20]
        best, best_score = (0, 0), -1e9
        for si in top_s:
            for ei in top_e:
                if ei < si or ei - si + 1 > MAX_ANSWER_TOKENS:
                    continue
                score = s_log[si] + e_log[ei]
                if score > best_score:
                    best, best_score = (si, ei), score
        if best_score <= -1e8:
            return "", -1e9
        char_start = offsets[best[0]][0]
        char_end = offsets[best[1]][1]
        return context[char_start:char_end].strip(), float(best_score)
