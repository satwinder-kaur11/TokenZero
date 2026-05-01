from abc import ABC, abstractmethod


class ClassifierBase(ABC):
    @abstractmethod
    def score(self, query: str) -> float:
        """Return query complexity score in the 0.0-1.0 range."""


class HeuristicClassifier(ClassifierBase):
    def score(self, query: str) -> float:
        # Section 2 will implement full scoring logic.
        return 0.5


class BERTClassifier(ClassifierBase):
    def score(self, query: str) -> float:
        # Section 2 will load and infer from DistilBERT.
        return 0.5
