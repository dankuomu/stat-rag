import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stat_rag.metrics import contains_answer, exact_match, normalize_answer, token_f1


def test_normalize_drops_articles_punctuation_case():
    assert normalize_answer("The  Eiffel, Tower!") == "eiffel tower"


def test_exact_match_any_gold():
    assert exact_match("the Eiffel Tower", ["Louvre", "Eiffel Tower"]) == 1.0
    assert exact_match("Eiffel", ["Eiffel Tower"]) == 0.0


def test_token_f1_partial_and_max_over_gold():
    assert abs(token_f1("1962", ["22 November to 1 December 1962"]) - 2 / 7) < 1e-9
    assert token_f1("Paris", ["London", "Paris"]) == 1.0
    assert token_f1("", ["Paris"]) == 0.0


def test_contains_answer_respects_word_boundaries():
    assert contains_answer("He was born in Paris, France.", ["paris"])
    assert not contains_answer("Parisian cafes", ["Paris"])
