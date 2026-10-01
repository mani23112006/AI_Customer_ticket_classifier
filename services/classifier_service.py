"""Ticket classification service.

The service is split into a small, mockable ``ProbabilityBackend`` (which owns
torch/transformers) and ``ClassifierService`` (label handling, top-k, status
decision). Tests inject a fake backend, so torch is never needed to test logic.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from config.errors import ClassificationError, ModelUnavailableError
from config.settings import Settings, cache_resource, get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Prediction:
    """Model output for one ticket."""

    category: str
    confidence: float
    top_k: list[tuple[str, float]]

    def top_k_as_dicts(self) -> list[dict[str, float | str]]:
        """JSON-friendly form used for storage and display."""
        return [{"category": c, "confidence": p} for c, p in self.top_k]


class ProbabilityBackend(Protocol):
    """Anything that maps text to class probabilities in label order."""

    def predict_proba(self, text: str) -> Sequence[float]: ...


class TransformersBackend:
    """DistilBERT sequence classifier running on CPU. Loaded once, reused."""

    def __init__(self, model_dir: Path, max_length: int) -> None:
        if not (model_dir / "config.json").exists():
            raise ModelUnavailableError(
                "Model files not found. Train the model first: python -m ml.train"
            )
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ModelUnavailableError("torch/transformers are not installed.") from exc
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
            self._model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        except Exception as exc:  # corrupt / incompatible files
            logger.error("model load failed: %s", type(exc).__name__)
            raise ModelUnavailableError("The model could not be loaded.") from exc
        self._model.eval()
        self._torch = torch
        self._max_length = max_length

    def predict_proba(self, text: str) -> list[float]:
        """Return softmax probabilities for one text."""
        torch = self._torch
        encoded = self._tokenizer(
            text, truncation=True, max_length=self._max_length, return_tensors="pt"
        )
        with torch.inference_mode():
            logits = self._model(**encoded).logits
        return torch.softmax(logits, dim=-1)[0].tolist()


class ClassifierService:
    """Turns backend probabilities into a Prediction."""

    def __init__(self, backend: ProbabilityBackend, labels: Sequence[str], top_k: int = 3):
        if not labels:
            raise ModelUnavailableError("No labels configured.")
        self._backend = backend
        self._labels = list(labels)
        self._top_k = max(1, min(top_k, len(self._labels)))

    @property
    def labels(self) -> list[str]:
        return list(self._labels)

    def classify(self, text: str) -> Prediction:
        """Predict category, confidence and top-k for a cleaned text."""
        if not text or not text.strip():
            raise ClassificationError("Cannot classify empty text.")
        try:
            probs = list(self._backend.predict_proba(text))
        except Exception as exc:  # backend failure must not crash the app
            logger.error("inference failed: %s", type(exc).__name__)
            raise ClassificationError("The model failed to produce a prediction.") from exc
        if len(probs) != len(self._labels):
            raise ClassificationError("Model output does not match the label set.")
        ranked = sorted(zip(self._labels, probs), key=lambda pair: pair[1], reverse=True)
        top = [(label, round(float(p), 4)) for label, p in ranked[: self._top_k]]
        return Prediction(category=top[0][0], confidence=top[0][1], top_k=top)


def decide_status(confidence: float, threshold: float) -> str:
    """'needs_review' below the threshold, otherwise 'triaged'."""
    return "needs_review" if confidence < threshold else "triaged"


def load_labels(model_dir: Path, fallback_path: Path) -> list[str]:
    """Read label order saved by training; fall back to ml/labels.json."""
    for path in (model_dir / "labels.json", fallback_path):
        if path.exists():
            try:
                labels = json.loads(path.read_text(encoding="utf-8"))["labels"]
            except (ValueError, KeyError, OSError) as exc:
                raise ModelUnavailableError(f"Could not read {path.name}.") from exc
            return list(labels)
    raise ModelUnavailableError("labels.json not found.")


def build_classifier_service(settings: Settings | None = None) -> ClassifierService:
    """Create the real service. Raises ModelUnavailableError if not possible."""
    settings = settings or get_settings()
    labels = load_labels(settings.model_dir, settings.labels_path)
    backend = TransformersBackend(settings.model_dir, settings.max_length)
    return ClassifierService(backend, labels, settings.top_k)


@cache_resource
def get_classifier_service() -> ClassifierService:
    """Process-wide singleton (st.cache_resource under Streamlit)."""
    return build_classifier_service()
