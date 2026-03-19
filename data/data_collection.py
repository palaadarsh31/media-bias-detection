"""
Data collection utilities for Media Bias Detection.
Supports loading from CSV, AllSides dataset format, or synthetic generation.
"""
import pandas as pd
import numpy as np
from pathlib import Path

BIAS_LABELS = {"left": 0, "neutral": 1, "right": 2}
ID2LABEL = {v: k for k, v in BIAS_LABELS.items()}


def load_from_csv(path: str) -> pd.DataFrame:
    """Load headlines dataset from CSV file."""
    df = pd.read_csv(path)
    assert {"headline", "label"}.issubset(df.columns), \
        "CSV must have 'headline' and 'label' columns"
    df["label_id"] = df["label"].map(BIAS_LABELS)
    return df


def get_class_distribution(df: pd.DataFrame) -> pd.Series:
    """Return value counts for bias labels."""
    return df["label"].value_counts()


def train_val_test_split(df: pd.DataFrame, val_size: float = 0.1,
                          test_size: float = 0.2, seed: int = 42):
    """Stratified split into train / val / test."""
    from sklearn.model_selection import train_test_split
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=seed)
    train, val = train_test_split(
        train_val, test_size=val_size / (1 - test_size),
        stratify=train_val["label"], random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def summarize_dataset(df: pd.DataFrame):
    """Print dataset summary stats."""
    print(f"Total samples   : {len(df)}")
    print(f"Class distribution:\n{get_class_distribution(df)}\n")
    df["headline_len"] = df["headline"].str.split().str.len()
    print(f"Avg headline length: {df['headline_len'].mean():.1f} words")
    print(f"Min / Max lengths  : {df['headline_len'].min()} / {df['headline_len'].max()}")


if __name__ == "__main__":
    df = load_from_csv("sample_headlines.csv")
    summarize_dataset(df)
