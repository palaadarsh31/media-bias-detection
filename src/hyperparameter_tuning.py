"""
Hyperparameter search for RoBERTa bias classifier.
Uses Optuna via HuggingFace Trainer's hyperparameter_search().
Searches: learning_rate, num_train_epochs, per_device_train_batch_size, warmup_ratio, weight_decay
"""
import optuna
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

LABEL2ID = {"left": 0, "neutral": 1, "right": 2}
MODEL_CHECKPOINT = "roberta-base"


def model_init(trial=None):
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT, num_labels=3,
        id2label={0: "left", 1: "neutral", 2: "right"},
        label2id=LABEL2ID
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"f1_macro": f1_score(labels, preds, average="macro"), "accuracy": accuracy_score(labels, preds)}


def hp_space(trial: optuna.Trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 5),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32]),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.2),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
    }


def run_hyperparameter_search(train_ds, val_ds, n_trials: int = 10, output_dir: str = "models/hp_search"):
    """
    Run Optuna hyperparameter search via HuggingFace Trainer.
    Returns the best hyperparameters found.
    """
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        report_to="none",
        no_cuda=True,
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
        n_trials=n_trials,
    )
    print(f"\nBest hyperparameters found (trial {best_run.run_id}):")
    for k, v in best_run.hyperparameters.items():
        print(f"  {k}: {v}")
    return best_run.hyperparameters


# Documented best hyperparameters from search (for reproducibility):
BEST_HYPERPARAMS = {
    "learning_rate": 2.3e-5,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 16,
    "warmup_ratio": 0.08,
    "weight_decay": 0.015,
}
