"""
Optuna hyperparameter search for RoBERTa media bias classifier.

Usage:
    python src/hyperparameter_tuning.py --data data/sample_headlines.csv --n_trials 10
"""
import argparse
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer,
)
from datasets import Dataset

LABEL2ID = {"left": 0, "neutral": 1, "right": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
MODEL_CHECKPOINT = "roberta-base"

# Best hyperparameters found in prior search — use these for reproducibility
BEST_HYPERPARAMS = {
    "learning_rate": 2.3e-5,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 16,
    "warmup_ratio": 0.08,
    "weight_decay": 0.015,
}


def model_init(trial=None):
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"f1_macro": f1_score(labels, preds, average="macro"),
            "accuracy": accuracy_score(labels, preds)}


def hp_space(trial: optuna.Trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 5),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32]),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.2),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
    }


def run_search(args):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    df = pd.read_csv(args.data)
    df["label"] = df["label"].map(LABEL2ID)
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

    train_ds = Dataset.from_pandas(train_df[["headline", "label"]].rename(columns={"headline": "text"})).map(tok, batched=True)
    val_ds   = Dataset.from_pandas(val_df[["headline", "label"]].rename(columns={"headline": "text"})).map(tok, batched=True)

    training_args = TrainingArguments(
        output_dir="models/hp_search",
        evaluation_strategy="epoch",
        save_strategy="no",
        report_to="none",
    )

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    best_run = trainer.hyperparameter_search(
        direction="maximize",
        backend="optuna",
        hp_space=hp_space,
        n_trials=args.n_trials,
    )

    print(f"\nBest trial: {best_run.run_id}")
    print("Best hyperparameters:")
    for k, v in best_run.hyperparameters.items():
        print(f"  {k}: {v}")
    return best_run.hyperparameters


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sample_headlines.csv")
    parser.add_argument("--n_trials", type=int, default=10)
    args = parser.parse_args()
    run_search(args)
