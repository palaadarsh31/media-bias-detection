"""
Evaluation utilities for media bias classifier.
Plots: confusion matrix, ROC curves (one-vs-rest), precision-recall curves.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, average_precision_score,
)
from sklearn.preprocessing import label_binarize

LABELS = ["left", "neutral", "right"]
COLORS = ["#2166ac", "#4dac26", "#d01c8b"]


def plot_confusion_matrix(y_true, y_pred, save_path: str = None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABELS, yticklabels=LABELS, linewidths=0.5)
    plt.title("Confusion Matrix — Media Bias Classifier", fontsize=13, fontweight="bold")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_roc_curves(y_true, y_prob, save_path: str = None):
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    plt.figure(figsize=(8, 6))
    for i, (label, color) in enumerate(zip(LABELS, COLORS)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f"{label.capitalize()} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.title("ROC Curves — Media Bias Classifier", fontsize=13, fontweight="bold")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(fontsize=11)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_precision_recall(y_true, y_prob, save_path: str = None):
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    plt.figure(figsize=(8, 6))
    for i, (label, color) in enumerate(zip(LABELS, COLORS)):
        prec, rec, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
        ap = average_precision_score(y_bin[:, i], y_prob[:, i])
        plt.plot(rec, prec, color=color, lw=2,
                 label=f"{label.capitalize()} (AP = {ap:.2f})")
    plt.title("Precision-Recall Curves — Media Bias Classifier", fontsize=13, fontweight="bold")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(fontsize=11)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def print_classification_report(y_true, y_pred):
    print("=" * 60)
    print("CLASSIFICATION REPORT — Media Bias Detection")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=LABELS))


def generate_full_report(y_true, y_pred, y_prob=None):
    print_classification_report(y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred)
    if y_prob is not None:
        plot_roc_curves(y_true, y_prob)
        plot_precision_recall(y_true, y_prob)
