from __future__ import annotations

from dataclasses import dataclass

import core.classifier_factory as classifier_factory
from core.classifier import BERTClassifier, HeuristicClassifier


def test_heuristic_empty_string_scores_zero() -> None:
    classifier = HeuristicClassifier()
    assert classifier.score("") == 0.0


def test_heuristic_code_input_increases_complexity() -> None:
    classifier = HeuristicClassifier()
    query = "def add(a, b):\n    return a + b"
    assert classifier.score(query) >= 0.2


def test_heuristic_very_long_query_saturates_token_component() -> None:
    classifier = HeuristicClassifier()
    query = "word " * 300
    score = classifier.score(query)
    assert 0.29 <= score <= 0.31


def test_heuristic_score_is_clamped() -> None:
    classifier = HeuristicClassifier()
    keywords = (
        "explain compare why debug analyze step by step difference between how does implement "
    )
    query = (keywords * 10) + "???"
    score = classifier.score(query)
    assert 0.0 <= score <= 1.0


class FakeTokenizer:
    def __init__(self) -> None:
        self.called = False
        self.last_max_length: int | None = None

    def __call__(self, text: str, return_tensors: str, truncation: bool, max_length: int) -> dict:
        self.called = True
        self.last_max_length = max_length
        return {
            "input_ids": [[1, 2, 3]],
            "attention_mask": [[1, 1, 1]],
        }


class FakeModel:
    def __init__(self, logits: list[list[float]]) -> None:
        self.logits = logits

    def __call__(self, **kwargs: object):
        _ = kwargs
        return type("Output", (), {"logits": self.logits})()


def test_bert_classifier_maps_complex_label_score() -> None:
    tokenizer = FakeTokenizer()
    model = FakeModel(logits=[[0.1, 0.2, 0.9]])
    classifier = BERTClassifier(tokenizer=tokenizer, model=model)
    score = classifier.score("please explain and compare this architecture")
    assert score == 0.85
    assert tokenizer.called
    assert tokenizer.last_max_length == 128


def test_bert_classifier_empty_query_returns_simple_score() -> None:
    tokenizer = FakeTokenizer()
    model = FakeModel(logits=[[0.9, 0.05, 0.05]])
    classifier = BERTClassifier(tokenizer=tokenizer, model=model)
    score = classifier.score("   ")
    assert score == 0.15
    assert tokenizer.called is False


@dataclass
class FakeSettings:
    router_mode: str


def test_classifier_factory_returns_heuristic(monkeypatch) -> None:
    monkeypatch.setattr(
        classifier_factory, "get_settings", lambda: FakeSettings(router_mode="heuristic")
    )
    classifier = classifier_factory.get_classifier()
    assert isinstance(classifier, HeuristicClassifier)


def test_classifier_factory_returns_bert(monkeypatch) -> None:
    monkeypatch.setattr(classifier_factory, "get_settings", lambda: FakeSettings(router_mode="bert"))

    class StubBERT(BERTClassifier):
        def __init__(self) -> None:  # pragma: no cover - trivial stub
            pass

    monkeypatch.setattr(classifier_factory, "BERTClassifier", StubBERT)
    classifier = classifier_factory.get_classifier()
    assert isinstance(classifier, StubBERT)
