# Perceptrón MNIST con CUDA

Este proyecto implementa un **perceptrón de una capa** en **C++/CUDA** para clasificar dígitos manuscritos del conjunto de datos **MNIST**. El modelo recibe imágenes de 28×28 píxeles y predice una de diez clases (dígitos del 0 al 9). El entrenamiento utiliza la GPU mediante kernels CUDA propios.

---

## Tabla de contenidos

1. [Descripción del modelo](#descripción-del-modelo)
2. [Arquitectura del software](#arquitectura-del-software)
3. [Fundamento matemático](#fundamento-matemático)
4. [Kernels CUDA](#kernels-cuda)
5. [Requisitos](#requisitos)
6. [Compilación y ejecución](#compilación-y-ejecución)
7. [Parámetros de entrenamiento](#parámetros-de-entrenamiento)
8. [Resultados esperados](#resultados-esperados)
9. [Estructura del repositorio](#estructura-del-repositorio)
10. [Referencias](#referencias)

---

## Descripción del modelo

El perceptrón implementado es una **capa lineal** que transforma un vector de entrada de 784 componentes (imagen aplanada) en 10 salidas, una por cada dígito:

```
Entrada (784 píxeles)  →  Capa lineal (10 neuronas)  →  Softmax  →  Clase (0–9)
```

- Las imágenes se normalizan al rango `[0, 1]` dividiendo cada píxel entre 255.
- Cada salida representa un *logit* asociado a un dígito.
- La clase predicha corresponde al dígito con mayor probabilidad tras aplicar softmax.
- El entrenamiento emplea **descenso de gradiente estocástico (SGD)** con **softmax + entropía cruzada** como función de pérdida.

Con este esquema se obtiene un clasificador lineal. En MNIST suele alcanzarse una precisión de aproximadamente **88–92 %** en el conjunto de prueba tras varias épocas.

---

## Arquitectura del software

El código se organiza en módulos con responsabilidades separadas:

| Módulo | Función |
|--------|---------|
| `Layer` | Interfaz común para capas (`forward`, `backward`, `update`) |
| `LinearLayer` | Capa lineal con pesos, sesgos y kernels CUDA |
| `Network` | Encadena una o más capas |
| `Trainer` | Bucle de entrenamiento por mini-batches |
| `MNISTLoader` | Lectura de archivos IDX de MNIST |
| `SoftmaxCrossEntropy` | Pérdida y gradiente en GPU |
| `Tensor` | Gestión RAII de memoria en dispositivo |

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────────┐
│ MNISTLoader │────▶│   Trainer    │────▶│      Network        │
│  (CPU)      │     │ mini-batches │     │  ┌───────────────┐  │
└─────────────┘     └──────┬───────┘     │  │  LinearLayer  │  │
                           │             │  │    (CUDA)     │  │
                           ▼             │  └───────────────┘  │
                  SoftmaxCrossEntropy    └─────────────────────┘
                     (CUDA, salida)
```

### Flujo de entrenamiento (un mini-batch)

1. Se sube el batch de imágenes y etiquetas a la GPU.
2. `network.forward()` calcula los logits.
3. `SoftmaxCrossEntropy::forward()` obtiene la pérdida y el gradiente ∂L/∂logits.
4. `network.backward()` propaga gradientes hacia pesos y sesgos.
5. `network.update(lr)` aplica SGD en la GPU.

---

## Fundamento matemático

### Forward (capa lineal)

Para cada muestra **x** ∈ ℝ⁷⁸⁴, con pesos **W** ∈ ℝ¹⁰ˣ⁷⁸⁴ y sesgo **b** ∈ ℝ¹⁰:

```
zⱼ = bⱼ + Σᵢ xᵢ · Wⱼᵢ     (j = 0…9)
```

### Softmax

```
pⱼ = exp(zⱼ - max(z)) / Σₖ exp(zₖ - max(z))
```

Se resta `max(z)` por estabilidad numérica.

### Entropía cruzada

Para la etiqueta verdadera `y`:

```
L = -log(pᵧ)
```

### Gradiente combinado (softmax + entropía cruzada)

```
∂L/∂zⱼ = pⱼ - δⱼᵧ
```

donde `δⱼᵧ` vale 1 si `j == y` y 0 en caso contrario. El gradiente se promedia sobre el batch.

### Backward (capa lineal)

Con `gⱼ = ∂L/∂zⱼ`:

```
∂L/∂Wⱼᵢ = gⱼ · xᵢ
∂L/∂bⱼ   = gⱼ
∂L/∂xᵢ   = Σⱼ gⱼ · Wⱼᵢ
```

### Actualización SGD

```
W ← W - η · ∇W
b ← b - η · ∇b
```

`η` es la tasa de aprendizaje (por defecto `0.1`).

### Inicialización de pesos

Se utiliza inicialización **Xavier/Glorot** con escala `√(2 / (n_in + n_out))` y sesgos en cero.

---

## Kernels CUDA

| Kernel | Archivo | Paralelismo |
|--------|---------|-------------|
| `linear_forward_kernel` | `linear_layer.cu` | 1 bloque por muestra, 1 hilo por neurona de salida |
| `linear_backward_params_kernel` | `linear_layer.cu` | 1 hilo por neurona; `atomicAdd` para acumular gradientes |
| `linear_backward_input_kernel` | `linear_layer.cu` | 1 bloque por muestra, hilos sobre dimensiones de entrada |
| `sgd_update_kernel` | `linear_layer.cu` | Actualización paralela de pesos y sesgos |
| `softmax_cross_entropy_kernel` | `softmax_cross_entropy.cu` | 1 bloque por muestra del batch |

Las operaciones de forward, backward y actualización de parámetros se ejecutan en GPU. La carga de datos y el registro de métricas se realizan en CPU.

---

## Requisitos

- GPU NVIDIA con soporte CUDA
- Driver NVIDIA compatible con el toolkit CUDA instalado
- CUDA Toolkit ≥ 11.x
- CMake ≥ 3.18 (recomendado ≥ 3.24 para arquitectura `native`)
- Compilador C++17 (g++ o clang)
- `curl`, `unzip` y credenciales de Kaggle para descargar el dataset

Comprobación del entorno:

```bash
nvidia-smi
nvcc --version
```

---

## Compilación y ejecución

### 1. Descargar MNIST desde Kaggle

El script obtiene el dataset [hojjatk/mnist-dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) y extrae los archivos IDX en `data/mnist/`.

**Autenticación requerida** (una de las dos opciones):

```bash
# Opción A: token de API
export KAGGLE_API_TOKEN=<token>
```

```bash
# Opción B: archivo de credenciales
mkdir -p ~/.kaggle
# Colocar kaggle.json con {"username":"...","key":"..."}
chmod 600 ~/.kaggle/kaggle.json
```

También es necesario **aceptar las reglas del dataset** en la página de Kaggle antes de descargar.

```bash
bash scripts/download_mnist.sh
```

Archivos generados en `data/mnist/`:

- `train-images-idx3-ubyte`, `train-labels-idx1-ubyte`
- `t10k-images-idx3-ubyte`, `t10k-labels-idx1-ubyte`

El script valida el tamaño de cada archivo para detectar descargas corruptas o no autorizadas.

### 2. Compilar

```bash
mkdir -p build && cd build
cmake ..
cmake --build .
```

Si aparece el error *"PTX compiled with an unsupported toolchain"*, se debe indicar la arquitectura de la GPU:

```bash
cmake -DCMAKE_CUDA_ARCHITECTURES=89 ..   # RTX 40xx (Ada)
cmake --build .
```

Otras referencias habituales: `75` (Turing), `86` (Ampere), `89` (Ada).

### 3. Entrenar

El ejecutable acepta `--data data/mnist` tanto desde la raíz del proyecto como desde `build/` (resuelve automáticamente `../data/mnist`).

Desde la raíz del proyecto:

```bash
./build/mnist_perceptron \
  --data data/mnist \
  --epochs 10 \
  --batch-size 128 \
  --lr 0.1
```

<!--Desde el directorio `build/`:

```bash
./mnist_perceptron \
  --data data/mnist \
  --epochs 10 \
  --batch-size 128 \
  --lr 0.1
```
-->

| Flag | Valor por defecto | Descripción |
|------|-------------------|-------------|
| `--data` | `data/mnist` | Directorio con archivos IDX |
| `--epochs` | `10` | Número de épocas |
| `--batch-size` | `128` | Tamaño del mini-batch |
| `--lr` | `0.1` | Tasa de aprendizaje SGD |

---

## Parámetros de entrenamiento

- **Batch size 128–256**: equilibrio habitual entre velocidad y estabilidad.
- **Learning rate 0.1**: valor elevado que funciona bien en la capa lineal; si la pérdida oscila, conviene probar `0.05` o `0.01`.
- **Épocas**: entre 5 y 10 suelen ser suficientes para aproximarse al ~90 % de precisión en test.

---

## Resultados esperados

Con **1 época**, batch 256 y learning rate 0.1 (GPU RTX 4070):

```
train acc: ~83–85 %
test  acc: ~88–89 %
```

Con **10 épocas**, la precisión en test suele situarse alrededor del **90–92 %**.

---

## Estructura del repositorio

```
MLP/
├── CMakeLists.txt
├── README.md
├── include/
│   ├── core/
│   │   ├── cuda_utils.cuh
│   │   └── tensor.hpp
│   ├── layers/
│   │   ├── layer.hpp
│   │   └── linear_layer.hpp
│   ├── loss/
│   │   └── softmax_cross_entropy.hpp
│   ├── data/
│   │   └── mnist_loader.hpp
│   ├── network/
│   │   └── network.hpp
│   └── training/
│       └── trainer.hpp
├── src/
│   ├── main.cpp
│   ├── data/mnist_loader.cpp
│   ├── network/network.cpp
│   ├── training/trainer.cpp
│   └── cuda/
│       ├── linear_layer.cu
│       └── softmax_cross_entropy.cu
├── scripts/
│   └── download_mnist.sh
└── data/mnist/
```

<!--
---

## Referencias

- [MNIST database](http://yann.lecun.com/exdb/mnist/)
- [Dataset MNIST en Kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- Goodfellow et al., *Deep Learning* — perceptrón, softmax y retropropagación
-->