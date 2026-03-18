"""
Text preprocessing for news headlines.
Steps: lowercasing, punctuation removal, stopword removal, tokenization.
"""
import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

STOP_WORDS = set(stopwords.words("english")) - {"not", "no", "but", "however"}


def clean_headline(text: str) -> str:
    """Lowercase, remove punctuation, normalize whitespace."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords(text: str) -> str:
    tokens = word_tokenize(text)
    return " ".join([t for t in tokens if t not in STOP_WORDS])


def preprocess_dataframe(df: pd.DataFrame, text_col: str = "headline") -> pd.DataFrame:
    df = df.copy()
    df["cleaned"] = df[text_col].apply(clean_headline)
    df["tokens"] = df["cleaned"].apply(word_tokenize)
    df["no_stopwords"] = df["cleaned"].apply(remove_stopwords)
    return df


def get_vocab_stats(df: pd.DataFrame) -> dict:
    all_tokens = [t for tokens in df["tokens"] for t in tokens]
    return {
        "total_tokens": len(all_tokens),
        "unique_tokens": len(set(all_tokens)),
        "avg_headline_length": df["tokens"].apply(len).mean(),
    }
