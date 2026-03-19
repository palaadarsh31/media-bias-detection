"""
Sentence-BERT embeddings for news headlines.
Model: all-MiniLM-L6-v2 (384-dim, fast, high quality).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

LABEL_COLORS = {"left": "#2166ac", "neutral": "#4dac26", "right": "#d01c8b"}


def load_sbert(model_name: str = "all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def generate_embeddings(texts: list, model, batch_size: int = 32) -> np.ndarray:
    """Generate normalized SBERT embeddings."""
    return model.encode(
        texts, batch_size=batch_size,
        show_progress_bar=True, normalize_embeddings=True
    )


def save_embeddings(embeddings: np.ndarray, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, embeddings)


def load_embeddings(path: str) -> np.ndarray:
    return np.load(path)


def plot_tsne(embeddings: np.ndarray, labels: list, title: str = "t-SNE of SBERT Embeddings",
              save_path: str = None):
    from sklearn.manifold import TSNE
    perplexity = min(30, len(embeddings) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    coords = tsne.fit_transform(embeddings)
    df_plot = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "label": labels})
    plt.figure(figsize=(10, 7))
    for label, color in LABEL_COLORS.items():
        s = df_plot[df_plot["label"] == label]
        plt.scatter(s["x"], s["y"], c=color, label=label.capitalize(), alpha=0.75, s=70, edgecolors="white", lw=0.5)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend(fontsize=12)
    plt.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_similarity_heatmap(embeddings: np.ndarray, labels: list, n: int = 20, save_path: str = None):
    """Cosine similarity heatmap for first n samples."""
    sim = embeddings[:n] @ embeddings[:n].T
    plt.figure(figsize=(10, 8))
    sns.heatmap(sim, cmap="RdBu_r", center=0, xticklabels=labels[:n], yticklabels=labels[:n],
                linewidths=0.3, annot=False)
    plt.title("Cosine Similarity Heatmap (SBERT Embeddings)", fontsize=13)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def nearest_neighbors(query_idx: int, embeddings: np.ndarray, headlines: list,
                       labels: list, top_k: int = 5) -> pd.DataFrame:
    """Find most similar headlines to a query by cosine similarity."""
    sims = embeddings @ embeddings[query_idx]
    top_idx = np.argsort(sims)[::-1][1: top_k + 1]
    return pd.DataFrame({
        "headline": [headlines[i] for i in top_idx],
        "label": [labels[i] for i in top_idx],
        "similarity": [sims[i] for i in top_idx],
    })
