"""
IMPORTANT SAFETY NOTICE:
This module belongs to a research-only ECG classification prototype.
It must NOT be used for any clinical or diagnostic decision-making.
Model predictions may be wrong. Always consult a qualified medical professional.
"""

from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
    average: str = "macro",
) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    precision, recall, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    return {
        "accuracy": float(acc),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
    }


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
    out_path: str,
    title: str = "Confusion Matrix",
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            _ = ax.text(
                j, i, cm[i, j], ha="center", va="center", fontsize=8
            )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
    out_path: str,
) -> None:
    report = classification_report(
        y_true, y_pred, target_names=labels, zero_division=0
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
