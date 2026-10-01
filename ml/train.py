"""Offline fine-tuning of distilbert-base-uncased for ticket classification.

Run from the project root:  python -m ml.train

The app never trains. This script:
    1. reads data/raw/tickets_bitext.csv (subject, message, category),
  2. makes a stratified 70/15/15 train/val/test split (saved to data/processed/),
  3. fine-tunes for 4 epochs (lr 2e-5, batch 16, max_length 128),
  4. keeps the checkpoint with the best validation macro F1,
  5. saves model + tokenizer + labels.json to models/ticket_classifier/.
Uses CUDA automatically when available; fp16 is enabled only on CUDA.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from config.settings import BASE_DIR, BASE_MODEL_NAME, get_settings
from preprocessing.text_cleaner import build_model_text

logger = logging.getLogger("train")


def read_labels(path: Path) -> list[str]:
    """Label order used for ids 0..n-1."""
    labels = list(json.loads(path.read_text(encoding="utf-8"))["labels"])
    if not labels or len(labels) != len(set(labels)):
        raise SystemExit("Label file must contain a non-empty list of unique labels")
    return labels


def prepare_frame(csv_path: Path, labels: list[str]) -> pd.DataFrame:
    """Load the CSV, validate it, and add ``text`` and ``label`` columns."""
    if not csv_path.exists():
        raise SystemExit(f"{csv_path} not found. Run: python -m ml.prepare_bitext")
    frame = pd.read_csv(csv_path)
    missing = {"subject", "message", "category"} - set(frame.columns)
    if missing:
        raise SystemExit(f"CSV is missing columns: {sorted(missing)}")
    frame = frame.dropna(subset=["category"]).copy()
    unknown = set(frame["category"]) - set(labels)
    if unknown:
        raise SystemExit(f"Unknown categories in data: {sorted(unknown)}")
    frame["text"] = [
        build_model_text(s if isinstance(s, str) else "", m if isinstance(m, str) else "")
        for s, m in zip(frame["subject"], frame["message"])
    ]
    frame = frame[frame["text"].str.len() > 0].reset_index(drop=True)
    missing_labels = set(labels) - set(frame["category"])
    if missing_labels:
        raise SystemExit(f"Training data has no rows for categories: {sorted(missing_labels)}")
    label_to_id = {label: i for i, label in enumerate(labels)}
    frame["label"] = frame["category"].map(label_to_id).astype(int)
    return frame


def stratified_split(frame: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """70/15/15 split, stratified by label."""
    train, rest = train_test_split(frame, test_size=0.30, stratify=frame["label"], random_state=seed)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest["label"], random_state=seed)
    return train, val, test


def compute_metrics(eval_pred) -> dict[str, float]:
    """Metrics used for model selection (macro F1) and logging."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT on ticket data.")
    parser.add_argument("--data", type=Path, default=BASE_DIR / "data" / "raw" / "tickets_bitext.csv")
    parser.add_argument("--labels", type=Path, default=BASE_DIR / "ml" / "labels_bitext8.json")
    parser.add_argument("--processed-dir", type=Path, default=BASE_DIR / "data" / "processed")
    parser.add_argument("--output-dir", type=Path, default=get_settings().model_dir)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require-gpu", action="store_true", help="Fail instead of training on CPU when CUDA is unavailable")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Heavy imports live here so `import ml.train` stays cheap for tooling/tests.
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    cuda_available = torch.cuda.is_available()
    if args.require_gpu and not cuda_available:
        raise SystemExit("CUDA GPU is unavailable. Install a CUDA-enabled PyTorch build and verify torch.cuda.is_available().")
    logger.info(
        "training device=%s; fp16=%s",
        torch.cuda.get_device_name(0) if cuda_available else "cpu",
        cuda_available,
    )

    set_seed(args.seed)
    labels = read_labels(args.labels)
    frame = prepare_frame(args.data, labels)
    train_df, val_df, test_df = stratified_split(frame, args.seed)

    args.processed_dir.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train_df), ("val", val_df), ("test", test_df)):
        part.to_csv(args.processed_dir / f"{name}.csv", index=False, encoding="utf-8")
    logger.info("split sizes: train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df))

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    class TicketDataset(torch.utils.data.Dataset):
        """Tokenised texts; padding is done per batch by the collator."""

        def __init__(self, part: pd.DataFrame) -> None:
            self.encodings = tokenizer(
                part["text"].tolist(), truncation=True, max_length=args.max_length
            )
            self.labels = part["label"].tolist()

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, idx: int) -> dict:
            item = {key: values[idx] for key, values in self.encodings.items()}
            item["labels"] = self.labels[idx]
            return item

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        label2id={label: i for i, label in enumerate(labels)},
    )

    checkpoint_dir = args.output_dir.parent / "checkpoints"
    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=cuda_available,
        logging_steps=20,
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=TicketDataset(train_df),
        eval_dataset=TicketDataset(val_df),
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    val_metrics = trainer.evaluate()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    (args.output_dir / "labels.json").write_text(
        json.dumps({"labels": labels, "base_model": BASE_MODEL_NAME, "max_length": args.max_length}, indent=2),
        encoding="utf-8",
    )
    shutil.rmtree(checkpoint_dir, ignore_errors=True)

    print("\nBest validation macro F1 (from this run):", trainer.state.best_metric)
    print("Final validation metrics (from this run):", json.dumps(val_metrics, indent=2))
    print(f"Model saved to {args.output_dir}")
    print("Next: python -m ml.evaluate   (measures the held-out TEST split)")


if __name__ == "__main__":
    main()
