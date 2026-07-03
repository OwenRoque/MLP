#pragma once

/// Interfaz base para capas. Toda capa futura del MLP implementara esta API.
class Layer {
public:
    virtual ~Layer() = default;

    /// Propaga la entrada hacia adelante (datos en GPU).
    virtual void forward(const float* input, int batch_size) = 0;

    /// Propaga gradientes hacia atras. grad_output apunta a GPU.
    virtual void backward(const float* grad_output, int batch_size) = 0;

    /// Actualiza parametros con SGD: param -= lr * gradiente.
    virtual void update(float learning_rate) = 0;

    /// Salida de la capa en GPU tras forward().
    virtual const float* output() const = 0;

    /// Gradiente respecto a la entrada (para encadenar capas en el MLP).
    virtual const float* grad_input() const = 0;

    virtual int input_size() const = 0;
    virtual int output_size() const = 0;

    virtual const char* name() const = 0;
};
