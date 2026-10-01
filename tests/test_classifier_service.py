"""Tests for classifier output format. The model is mocked; torch is never needed."""
import pytest

from config.errors import ClassificationError, ModelUnavailableError
from config.settings import CATEGORIES
from services.classifier_service import ClassifierService, Prediction, decide_status


class FakeBackend:
    def __init__(self, probs):
        self.probs = probs
        self.seen = []

    def predict_proba(self, text):
        self.seen.append(text)
        return self.probs


class ExplodingBackend:
    def predict_proba(self, text):
        raise RuntimeError("boom")


def make_service(probs, top_k=3):
    return ClassifierService(FakeBackend(probs), list(CATEGORIES), top_k)


def test_output_format():
    probs = [0.02, 0.05, 0.10, 0.60, 0.13, 0.04, 0.03, 0.03]
    result = make_service(probs).classify("refund please")
    assert isinstance(result, Prediction)
    assert result.category == "Refund"
    assert result.confidence == 0.6
    assert [c for c, _ in result.top_k] == ["Refund", "Delivery", "Technical"]
    assert len(result.top_k) == 3
    confs = [p for _, p in result.top_k]
    assert confs == sorted(confs, reverse=True)
    assert all(0.0 <= p <= 1.0 for p in confs)


def test_top_k_as_dicts_is_json_friendly():
    result = make_service([0.1, 0.7, 0.05, 0.05, 0.04, 0.03, 0.02, 0.01]).classify("bill")
    assert result.top_k_as_dicts()[0] == {"category": "Payment", "confidence": 0.7}


def test_top_k_larger_than_labels_is_clamped():
    assert len(make_service([1 / 8] * 8, top_k=99).classify("x").top_k) == 8


def test_empty_text_rejected():
    with pytest.raises(ClassificationError):
        make_service([1 / 8] * 8).classify("   ")


def test_wrong_output_length_rejected():
    with pytest.raises(ClassificationError):
        make_service([0.5, 0.5]).classify("hello")


def test_backend_failure_becomes_classification_error():
    service = ClassifierService(ExplodingBackend(), list(CATEGORIES))
    with pytest.raises(ClassificationError):
        service.classify("hello")


def test_no_labels_rejected():
    with pytest.raises(ModelUnavailableError):
        ClassifierService(FakeBackend([]), [])


def test_decide_status_threshold_boundary():
    assert decide_status(0.59, 0.60) == "needs_review"
    assert decide_status(0.60, 0.60) == "triaged"
    assert decide_status(0.99, 0.60) == "triaged"
