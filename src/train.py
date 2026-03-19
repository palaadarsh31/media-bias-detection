"""
Fine-tune RoBERTa for 3-class media bias classification.

Usage:
    python src/train.py --data data/sample_headlines.csv --epochs 3 --lr 2e-5
"""
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback,
)
from datasets import Dataset

LABEL2ID = {"left": 0, "neutral": 1, "right": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
MODEL_CHECKPOINT = "roberta-base"


def load_and_split(data_path: str):
    df = pd.read_csv(data_path)
    df["label"] = df["label"].map(LABEL2ID)
    df = df.dropna(subset=["headline", "label"])
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42)
    train_df, val_df = train_test_split(
        train_df, test_size=0.125, stratify=train_df["label"], random_state=42)
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df, val_df, test_df


def tokenize(df: pd.DataFrame, tokenizer, max_length: int = 128):
    ds = Dataset.from_pandas(
        df[["headline", "label"]].rename(columns={"headline": "text"}))
    def _tok(batch):
        return tokenizer(batch["text"], truncation=True,
                          padding="max_length", max_length=max_length)
    return ds.map(_tok, batched=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def train(args):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT, num_labels=3,
        id2label=ID2LABEL, label2id=LABEL2ID,
    )

    train_df, val_df, test_df = load_and_split(args.data)
    train_ds = tokenize(train_df, tokenizer)
    val_ds   = tokenize(val_df, tokenizer)
    test_ds  = tokenize(test_df, tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_dir=str(Path(args.output_dir) / "logs"),
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print(f"Fine-tuning {MODEL_CHECKPOINT} for media bias classification...")
    trainer.train()

    print("\n--- Test Set Evaluation ---")
    results = trainer.evaluate(test_ds)
    print(results)

    preds_out = trainer.predict(test_ds)
    pred_labels = np.argmax(preds_out.predictions, axis=-1)
    print("\nClassification Report:")
    print(classification_report(
        test_df["label"].values, pred_labels,
        target_names=["left", "neutral", "right"]))

    save_path = Path(args.output_dir) / "best_model"
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\nModel saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune RoBERTa for media bias detection")
    parser.add_argument("--data", default="data/sample_headlines.csv")
    parser.add_argument("--output_dir", default="models/roberta-bias")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()
    train(args)
