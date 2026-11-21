import re
import string
from collections import Counter


ARTICLES_RE = re.compile(r"\b(a|an|the)\b")


def normalize_answer(text: str) -> str:
    """
    Нормализация ответа как в официальном скрипте SQuAD.

    :param text: исходная строка
    :return: строка в нижнем регистре без пунктуации, артиклей и лишних пробелов
    """
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = ARTICLES_RE.sub(" ", text)
    return " ".join(text.split())


def exact_match(prediction: str, gold: list[str]) -> float:
    """
    Точное совпадение с любым из эталонов.

    :param prediction: ответ системы
    :param gold: список эталонных ответов
    :return: 1.0 или 0.0
    """
    pred = normalize_answer(prediction)
    return float(any(pred == normalize_answer(g) for g in gold))


def token_f1(prediction: str, gold: list[str]) -> float:
    """
    Токенный F1, максимум по эталонам.

    :param prediction: ответ системы
    :param gold: список эталонных ответов
    :return: F1 в [0, 1]
    """
    pred_tokens = normalize_answer(prediction).split()
    best = 0.0
    for g in gold:
        gold_tokens = normalize_answer(g).split()
        common = Counter(pred_tokens) & Counter(gold_tokens)
        same = sum(common.values())
        if same == 0:
            continue
        precision = same / len(pred_tokens)
        recall = same / len(gold_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def contains_answer(text: str, gold: list[str]) -> bool:
    """
    Есть ли хотя бы один эталонный ответ подстрокой в тексте (после нормализации).

    :param text: текст чанка или контекста
    :param gold: список эталонных ответов
    :return: True, если ответ встречается
    """
    norm = f" {normalize_answer(text)} "
    return any(f" {normalize_answer(g)} " in norm for g in gold if normalize_answer(g))
