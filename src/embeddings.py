"""
Sentence-BERT embeddings for news headlines.
Uses 'all-MiniLM-L6-v2' for fast, high-quality sentence embeddings.
"""
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

MODEL_NAME = "all-MiniLM-L6-v2"
LABEL_COLORS = {"left": "#2166ac", "neutral": "#4dac26", "right": "#d01c8b"}


def load_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def generate_embeddings(texts: list, model: SentenceTransformer, batch_size: int = 32) -> np.ndarray:
    """Generate SBERT embeddings for a list of texts."""
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)


def save_embeddings(embeddings: np.ndarray, path: str):
    np.save(path, embeddings)


def load_embeddings(path: str) -> np.ndarray:
    return np.load(path)


def plot_tsne(embeddings: np.ndarray, labels: list, save_path: str = None):
    """Visualize embeddings with t-SNE colored by bias label."""
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
    coords = tsne.fit_transform(embeddings)
    df_plot = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "label": labels})
    plt.figure(figsize=(10, 7))
    for label, color in LABEL_COLORS.items():
        subset = df_plot[df_plot["label"] == label]
        plt.scatter(subset["x"], subset["y"], c=color, label=label.capitalize(), alpha=0.7, s=60)
    plt.title("t-SNE of SBERT Embeddings by Bias Label", fontsize=14)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix (embeddings must be normalized)."""
    return embeddings @ embeddings.T
