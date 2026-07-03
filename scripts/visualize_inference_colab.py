from google.colab import files
import matplotlib.pyplot as plt
import numpy as np

# Subir archivo exportado por C++
uploaded = files.upload()
npz_path = next(iter(uploaded))

data = np.load(npz_path)
images = data["images"]       # shape (N, 28, 28)
y_true = data["y_true"]       # shape (N,)
y_pred = data["y_pred"]       # shape (N,)
confidence = data["confidence"]  # shape (N,) valores 0–1

n = len(y_true)
cols = 5
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
    ax.set_title(f"P:{y_pred[i]} R:{y_true[i]}\nC:{conf_pct:.1f}%", fontsize=10, color=color)

plt.tight_layout()
plt.show()

correct = np.sum(y_pred == y_true)
print(f"Aciertos en muestra visualizada: {correct}/{n} ({100.0 * correct / n:.1f}%)")
