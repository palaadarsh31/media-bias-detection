"""
Fine-tune RoBERTa for 3-class media bias classification.
Usage: python src/train.py --data data/sample_headlines.csv --epochs 3 --lr 2e-5
"""
import argparse
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from datasets import Dataset

LABEL2ID = {"left": 0, "neutral": 1, "right": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
MODEL_CHECKPOINT = "roberta-base"


def load_and_split(data_path: str):
    df = pd.read_csv(data_path)
    df["label"] = df["label"].map(LABEL2ID)
    df = df.dropna(subset=["headline", "label"])
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_df["label"], random_state=42)
    return train_df, val_df, test_df


def tokenize_dataset(df: pd.DataFrame, tokenizer, max_length: int = 128):
    dataset = Dataset.from_pandas(df[["headline", "label"]].rename(columns={"headline": "text"}))

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_length)

    return dataset.map(tokenize_fn, batched=True)


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
        MODEL_CHECKPOINT, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID
    )
    train_df, val_df, test_df = load_and_split(args.data)
    train_ds = tokenize_dataset(train_df, tokenizer)
    val_ds = tokenize_dataset(val_df, tokenizer)
    test_ds = tokenize_dataset(test_df, tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_dir="./logs",
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

    print("Training RoBERTa for media bias classification...")
    trainer.train()

    print("\nTest set evaluation:")
    results = trainer.evaluate(test_ds)
    print(results)

    # Full classification report
    preds = trainer.predict(test_ds)
    pred_labels = np.argmax(preds.predictions, axis=-1)
    true_labels = test_df["label"].values
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels, target_names=["left", "neutral", "right"]))

    trainer.save_model(Path(args.output_dir) / "best_model")
    tokenizer.save_pretrained(Path(args.output_dir) / "best_model")
    print(f"\nModel saved to {args.output_dir}/best_model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sample_headlines.csv")
    parser.add_argument("--output_dir", default="models/roberta-bias")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()
    train(args)
