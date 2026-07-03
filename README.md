# MLP MNIST con CUDA

Este proyecto implementa una **red neuronal multicapa (MLP)** en **C++/CUDA** para clasificar dígitos manuscritos del conjunto de datos **MNIST**. El entrenamiento y la inferencia se ejecutan en GPU; la **visualización de resultados** se realiza en **Python** a partir de un archivo `.npz` exportado tras el entrenamiento.

También admite un modo **perceptrón lineal** (`--hidden-size 0`).

---

## Tabla de contenidos

1. [Descripción del modelo](#descripción-del-modelo)
2. [Arquitectura del software](#arquitectura-del-software)
3. [Exportación e inferencia](#exportación-e-inferencia)
4. [Visualización en Python / Colab](#visualización-en-python--colab)
5. [Fundamento matemático](#fundamento-matemático)
6. [Kernels CUDA](#kernels-cuda)
7. [Requisitos](#requisitos)
8. [Compilación y ejecución](#compilación-y-ejecución)
9. [Parámetros](#parámetros)
10. [Resultados esperados](#resultados-esperados)
11. [Estructura del repositorio](#estructura-del-repositorio)
12. [Referencias](#referencias)

---

## Descripción del modelo

### MLP (configuración por defecto)

```
Entrada (784)  →  Linear(hidden)  →  Activación  →  Linear(10)  →  Softmax  →  Clase (0–9)
```

Por defecto: `hidden=128`, activación **ReLU**. También se soportan **GeLU** y **Sigmoid**.

### Perceptrón lineal (`--hidden-size 0`)

```
Entrada (784)  →  Linear(10)  →  Softmax  →  Clase (0–9)
```

---

## Arquitectura del software

| Módulo | Función |
|--------|---------|
| `LinearLayer` | Capa afín con pesos y sesgos (CUDA) |
| `ActivationLayer` | ReLU, GeLU o Sigmoid (CUDA) |
| `Network` | Encadena capas |
| `Trainer` | Entrenamiento por mini-batches |
| `export_inference_results` | Inferencia sobre N muestras y exportación `.npz` |
| `scripts/visualize_inference.py` | Visualización local con matplotlib |

```
MNISTLoader → Trainer → Network (CUDA) → export_inference_results → inference_results.npz
                                                                              ↓
                                                              Python / Colab (matplotlib)
```

---

## Exportación e inferencia

Tras entrenar, `mnist_mlp` ejecuta inferencia sobre **20 muestras** del conjunto test (configurable) y genera `inference_results.npz` con:

| Array | Forma | Descripción |
|-------|-------|-------------|
| `images` | `(N, 28, 28)` | Imágenes normalizadas [0, 1] |
| `y_true` | `(N,)` | Etiqueta real (0–9) |
| `y_pred` | `(N,)` | Predicción del modelo |
| `confidence` | `(N,)` | Probabilidad de la clase predicha (0–1) |

No se exportan pesos del modelo; solo resultados listos para visualizar.

---

## Visualización en Python / Colab

### Requisitos Python

```bash
pip install numpy matplotlib
```

### Local

```bash
# Tras entrenar (genera inference_results.npz)
python scripts/visualize_inference.py inference_results.npz
```

Genera una cuadrícula 4×5 (20 ejemplos) con título **P** (predicción), **R** (real) y **C** (confianza %). Verde = acierto, rojo = error.

### Google Colab

1. Entrenar en local y subir `inference_results.npz` a Colab.
2. Ejecutar el contenido de `scripts/visualize_inference_colab.py`:

```python
from google.colab import files
import matplotlib.pyplot as plt
import numpy as np

uploaded = files.upload()
npz_path = next(iter(uploaded))

data = np.load(npz_path)
images, y_true, y_pred, confidence = data["images"], data["y_true"], data["y_pred"], data["confidence"]

n, cols = len(y_true), 5
rows = (n + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.4))
axes = np.atleast_2d(axes)

for i in range(rows * cols):
    ax = axes[i // cols, i % cols]
    ax.axis("off")
    if i >= n:
        continue
    ax.imshow(images[i], cmap="gray", vmin=0, vmax=1)
    color = "green" if y_pred[i] == y_true[i] else "red"
    ax.set_title(f"P:{y_pred[i]} R:{y_true[i]}\nC:{confidence[i]*100:.1f}%", fontsize=10, color=color)

plt.tight_layout()
plt.show()
```

---

## Fundamento matemático

### Capa lineal

`z = xWᵀ + b`

### Activaciones

ReLU, GeLU (aprox. tanh) y Sigmoid en capa oculta.

### Salida

Softmax + entropía cruzada; SGD para actualizar pesos.

---

## Kernels CUDA

| Kernel | Archivo |
|--------|---------|
| `linear_forward_kernel` | `linear_layer.cu` |
| `linear_backward_*` | `linear_layer.cu` |
| `activation_forward/backward_kernel` | `activation_layer.cu` |
| `softmax_cross_entropy_kernel` | `softmax_cross_entropy.cu` |

---

## Requisitos

- GPU NVIDIA + CUDA Toolkit ≥ 11.x
- CMake ≥ 3.18, compilador C++17
- Zlib (para exportación NPZ)
- Credenciales Kaggle para descargar MNIST
- Python 3 + numpy + matplotlib (solo visualización)

---

## Compilación y ejecución

### Descargar MNIST

```bash
export KAGGLE_API_TOKEN=<token>
bash scripts/download_mnist.sh
```

### Compilar

```bash
mkdir -p build && cd build
cmake ..
cmake --build .
```

### Entrenar y exportar

```bash
./build/mnist_mlp \
  --data data/mnist \
  --epochs 10 \
  --batch-size 128 \
  --lr 0.01 \
  --hidden-size 128 \
  --activation relu \
  --export inference_results.npz \
  --export-samples 20 \
  --export-start 0
```

| Flag | Default | Descripción |
|------|---------|-------------|
| `--data` | `data/mnist` | Directorio MNIST |
| `--epochs` | `10` | Épocas |
| `--batch-size` | `128` | Mini-batch |
| `--lr` | `0.01` | Learning rate |
| `--hidden-size` | `128` | Neuronas ocultas (`0` = lineal) |
| `--activation` | `relu` | `relu`, `gelu`, `sigmoid` |
| `--export` | `inference_results.npz` | Archivo de salida |
| `--export-samples` | `20` | Número de ejemplos |
| `--export-start` | `0` | Índice inicial en test |

---

## Parámetros

- **MLP**: `lr=0.01`, `hidden-size=128`, 10 épocas.
- **Lineal**: `--hidden-size 0 --lr 0.1`.

---

## Resultados esperados

| Configuración | Precisión test (aprox.) |
|---------------|-------------------------|
| Lineal, 10 épocas | ~90–92 % |
| MLP 128 ReLU, 10 épocas | ~96–98 % |

---

## Estructura del repositorio

```
MLP/
├── include/
│   ├── activations/activation.hpp
│   ├── inference/inference_export.hpp
│   ├── layers/{layer, linear_layer, activation_layer}.hpp
│   ├── network/{network, network_builder}.hpp
│   └── ...
├── src/
│   ├── main.cpp
│   ├── inference/inference_export.cpp
│   └── cuda/
├── scripts/
│   ├── download_mnist.sh
│   ├── visualize_inference.py
│   └── visualize_inference_colab.py
└── inference_results.npz   # generado al entrenar
```

---

## Referencias

- [MNIST database](http://yann.lecun.com/exdb/mnist/)
- [Dataset MNIST en Kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
