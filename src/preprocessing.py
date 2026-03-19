"""
Text preprocessing for news headlines.
Pipeline: lowercase → URL removal → punctuation removal → tokenization → stopword removal.
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

# Keep negation words — they carry sentiment signal
STOP_WORDS = set(stopwords.words("english")) - {"not", "no", "but", "however", "never"}


def clean_headline(text: str) -> str:
    """Lowercase, strip URLs, punctuation, and extra whitespace."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords(text: str) -> str:
    """Remove stopwords while preserving negation."""
    tokens = word_tokenize(text)
    return " ".join(t for t in tokens if t not in STOP_WORDS)


def preprocess_dataframe(df: pd.DataFrame, text_col: str = "headline") -> pd.DataFrame:
    """Apply full preprocessing pipeline to a DataFrame."""
    df = df.copy()
    df["cleaned"] = df[text_col].apply(clean_headline)
    df["tokens"] = df["cleaned"].apply(word_tokenize)
    df["no_stopwords"] = df["cleaned"].apply(remove_stopwords)
    df["token_count"] = df["tokens"].apply(len)
    return df


def get_vocab_stats(df: pd.DataFrame) -> dict:
    """Return vocabulary statistics."""
    all_tokens = [t for tokens in df["tokens"] for t in tokens]
    return {
        "total_tokens": len(all_tokens),
        "unique_tokens": len(set(all_tokens)),
        "avg_headline_length": round(df["token_count"].mean(), 2),
        "max_headline_length": df["token_count"].max(),
    }


def get_top_words(df: pd.DataFrame, label: str, n: int = 15) -> pd.Series:
    """Return top N words for a given bias label."""
    from collections import Counter
    subset = df[df["label"] == label]
    all_tokens = [t for tokens in subset["tokens"] for t in tokens
                  if t not in STOP_WORDS and len(t) > 2]
    return pd.Series(Counter(all_tokens)).nlargest(n)


if __name__ == "__main__":
    df = pd.read_csv("data/sample_headlines.csv")
    df = preprocess_dataframe(df)
    print(get_vocab_stats(df))
    print("\nTop words (left):", get_top_words(df, "left").to_dict())
    print("Top words (right):", get_top_words(df, "right").to_dict())
    print("Top words (neutral):", get_top_words(df, "neutral").to_dict())
