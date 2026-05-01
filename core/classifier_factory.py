from core.classifier import BERTClassifier, ClassifierBase, HeuristicClassifier
from core.settings import get_settings


def get_classifier() -> ClassifierBase:
    settings = get_settings()
    if settings.router_mode == "heuristic":
        return HeuristicClassifier()
    return BERTClassifier()
