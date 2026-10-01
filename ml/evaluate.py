"""Evaluate the trained model on the held-out test split.

Run from the project root:  python -m ml.evaluate

Everything printed here is computed from the actual run. The test split comes
from data/processed/test.csv, written by ml.train. Inference goes through the
same ClassifierService the app uses, on CPU.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from config.settings import BASE_DIR, get_settings  # noqa: E402
from services.classifier_service import build_classifier_service  # noqa: E402

WARMUP_CALLS = 3


def plot_confusion_matrix(matrix, labels: list[str], out_path: Path) -> None:
    """Save a labelled confusion matrix image."""
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (test split)")
    threshold = matrix.max() / 2 if matrix.max() else 0
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > threshold else "black")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Evaluate the ticket classifier.")
    parser.add_argument("--test-csv", type=Path, default=BASE_DIR / "data" / "processed" / "test.csv")
    parser.add_argument("--out-dir", type=Path, default=BASE_DIR / "models" / "evaluation")
    args = parser.parse_args(argv)

    if not args.test_csv.exists():
        raise SystemExit(f"{args.test_csv} not found. Run: python -m ml.train")
    test = pd.read_csv(args.test_csv)
    service = build_classifier_service(settings)
    labels = service.labels

    texts = test["text"].astype(str).tolist()
    for text in texts[:WARMUP_CALLS]:  # exclude one-off startup cost from timing
        service.classify(text)

    predictions, confidences, timings_ms = [], [], []
    for text in texts:
        start = time.perf_counter()
        result = service.classify(text)
        timings_ms.append((time.perf_counter() - start) * 1000)
        predictions.append(result.category)
        confidences.append(result.confidence)

    y_true = test["category"].tolist()
    accuracy = accuracy_score(y_true, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, predictions, labels=labels, average="macro", zero_division=0
    )
    report = classification_report(y_true, predictions, labels=labels, digits=3, zero_division=0)
    matrix = confusion_matrix(y_true, predictions, labels=labels)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(matrix, labels, args.out_dir / "confusion_matrix.png")

    timing = {
        "n_tickets": len(timings_ms),
        "mean_ms": statistics.fmean(timings_ms),
        "median_ms": statistics.median(timings_ms),
        "p95_ms": sorted(timings_ms)[max(0, int(len(timings_ms) * 0.95) - 1)],
    }

    frame = test.assign(predicted=predictions, confidence=confidences)
    wrong = frame[frame["predicted"] != frame["category"]].sort_values("confidence", ascending=False)
    below = frame[frame["confidence"] < settings.confidence_threshold]

    summary = {
        "test_size": len(frame),
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "threshold": settings.confidence_threshold,
        "share_below_threshold": len(below) / len(frame),
        "cpu_inference_ms_per_ticket": timing,
        "n_misclassified": int(len(wrong)),
        "data_note": "Numbers come from the split of whatever CSV was used for training. "
                     "If that is the synthetic sample data, they overstate real-world quality.",
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    wrong.head(10)[["text", "category", "predicted", "confidence"]].to_csv(
        args.out_dir / "misclassified_examples.csv", index=False, encoding="utf-8"
    )

    print(f"Test size: {len(frame)}")
    print(f"Accuracy:        {accuracy:.4f}")
    print(f"Macro precision: {precision:.4f}")
    print(f"Macro recall:    {recall:.4f}")
    print(f"Macro F1:        {f1:.4f}")
    print("\nPer-class report:\n" + report)
    print(
        f"CPU inference per ticket: mean {timing['mean_ms']:.1f} ms, "
        f"median {timing['median_ms']:.1f} ms, p95 {timing['p95_ms']:.1f} ms "
        f"(n={timing['n_tickets']}, single-ticket calls)"
    )
    print(
        f"Below confidence threshold {settings.confidence_threshold:.2f}: "
        f"{len(below)}/{len(frame)} ({summary['share_below_threshold']:.1%}) -> would go to review"
    )
    print(f"\nMisclassified: {len(wrong)}. Showing up to 10 (most confident mistakes first):")
    if wrong.empty:
        print("  none on this test split")
    for _, row in wrong.head(10).iterrows():
        print(f"  true={row['category']!r} pred={row['predicted']!r} conf={row['confidence']:.2f} | {row['text'][:110]}")
    print(f"\nSaved: {args.out_dir / 'confusion_matrix.png'}, metrics.json, misclassified_examples.csv")
    print("Note: if the data was synthetic sample data, treat these numbers as a pipeline check, not real-world performance.")


if __name__ == "__main__":
    main()
