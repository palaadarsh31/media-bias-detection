"""
Data collection utilities for Media Bias Detection.
Supports loading from AllSides dataset format, CSV files, or generating synthetic samples.
"""
import pandas as pd
import numpy as np
from pathlib import Path

BIAS_LABELS = {"left": 0, "neutral": 1, "right": 2}


def load_from_csv(path: str) -> pd.DataFrame:
    """Load headlines dataset from CSV."""
    df = pd.read_csv(path)
    assert {"headline", "label"}.issubset(df.columns), "CSV must have 'headline' and 'label' columns"
    df["label_id"] = df["label"].map(BIAS_LABELS)
    return df


def get_class_distribution(df: pd.DataFrame) -> pd.Series:
    """Return class distribution."""
    return df["label"].value_counts()


def train_test_split_stratified(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    from sklearn.model_selection import train_test_split
    return train_test_split(df, test_size=test_size, stratify=df["label"], random_state=seed)


if __name__ == "__main__":
    df = load_from_csv("sample_headlines.csv")
    print(get_class_distribution(df))
