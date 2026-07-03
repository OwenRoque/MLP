"""Visualiza resultados de inferencia exportados por mnist_mlp (.npz).

Uso local:
    python scripts/visualize_inference.py inference_results.npz

En Google Colab: subir inference_results.npz y ejecutar este script.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_results(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    required = {"images", "y_true", "y_pred", "confidence"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"Faltan arrays en el NPZ: {missing}")
    return {k: data[k] for k in required}


def plot_results(
    images: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
    *,
    cols: int = 5,
    save_path: str | Path | None = None,
) -> None:
    n = len(y_true)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.4))
    axes = np.atleast_2d(axes)

    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        ax.axis("off")
        if i >= n:
            continue

        ax.imshow(images[i], cmap="gray", vmin=0.0, vmax=1.0)
        conf_pct = confidence[i] * 100.0
        color = "green" if y_pred[i] == y_true[i] else "red"
        ax.set_title(
            f"P:{y_pred[i]} R:{y_true[i]}\nC:{conf_pct:.1f}%",
            fontsize=10,
            color=color,
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figura guardada en: {save_path}")
    plt.show()


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "inference_results.npz"
    results = load_results(path)
    plot_results(
        results["images"],
        results["y_true"],
        results["y_pred"],
        results["confidence"],
        save_path="inference_plot.png",
    )


if __name__ == "__main__":
    main()
