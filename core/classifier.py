from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import Any, Mapping, Sequence


class ClassifierBase(ABC):
    @abstractmethod
    def score(self, query: str) -> float:
        """Return query complexity score in the 0.0-1.0 range."""


class HeuristicClassifier(ClassifierBase):
    REASONING_KEYWORDS = (
        "explain",
        "compare",
        "why",
        "debug",
        "analyze",
        "step by step",
        "difference between",
        "how does",
        "implement",
    )

    @staticmethod
    def _clamp_01(value: float) -> float:
        return max(0.0, min(1.0, value))

    def score(self, query: str) -> float:
        text = (query or "").strip()
        lowered = text.lower()
        tokens = text.split()

        token_len_score = self._clamp_01(len(tokens) / 100.0)
        keyword_hits = sum(lowered.count(keyword) for keyword in self.REASONING_KEYWORDS)
        keyword_score = self._clamp_01(keyword_hits / 5.0)
        subquestion_score = self._clamp_01(text.count("?") / 3.0)
        code_score = 1.0 if ("`" in text or "def " in lowered or "class " in lowered) else 0.0

        weighted_score = (
            (token_len_score * 0.3)
            + (keyword_score * 0.4)
            + (subquestion_score * 0.1)
            + (code_score * 0.2)
        )
        return self._clamp_01(weighted_score)


class BERTClassifier(ClassifierBase):
    LABEL_TO_SCORE = {0: 0.15, 1: 0.50, 2: 0.85}

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        max_length: int = 128,
        tokenizer: Any | None = None,
        model: Any | None = None,
    ) -> None:
        self.max_length = max_length
        self._torch: Any | None = None

        if tokenizer is not None and model is not None:
            self.tokenizer = tokenizer
            self.model = model
            return

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "BERT mode requires 'torch' and 'transformers' to be installed."
            ) from exc

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3,
            ignore_mismatched_sizes=True,
        )
        self.model.to("cpu")
        self.model.eval()

    @staticmethod
    def _predicted_class(logits: Any) -> int:
        if hasattr(logits, "argmax"):
            try:
                return int(logits.argmax(dim=-1).item())
            except (TypeError, AttributeError):
                try:
                    return int(logits.argmax(axis=-1).item())
                except (TypeError, AttributeError):
                    pass

        if isinstance(logits, Sequence) and logits:
            first_row = logits[0]
            if isinstance(first_row, Sequence) and first_row:
                return int(max(range(len(first_row)), key=lambda idx: first_row[idx]))

        return 0

    def score(self, query: str) -> float:
        text = (query or "").strip()
        if not text:
            return self.LABEL_TO_SCORE[0]

        inputs: Mapping[str, Any] = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )

        inference_ctx = self._torch.no_grad() if self._torch is not None else nullcontext()
        with inference_ctx:
            output = self.model(**inputs)

        predicted_class = self._predicted_class(output.logits)
        return float(self.LABEL_TO_SCORE.get(predicted_class, 0.50))
