#pragma once

#include "layers/layer.hpp"

#include <memory>
#include <vector>

/// Contenedor de capas encadenadas. Con una sola LinearLayer es un perceptron;
/// al añadir capas ocultas se convierte en MLP sin cambiar la API de entrenamiento.
class Network {
public:
    void add_layer(std::unique_ptr<Layer> layer);

    void forward(const float* input, int batch_size);
    void backward(const float* grad_output, int batch_size);
    void update(float learning_rate);

    const float* output() const;
    const Layer& output_layer() const;

    int input_size() const;
    int output_size() const;
    std::size_t num_layers() const { return layers_.size(); }

private:
    std::vector<std::unique_ptr<Layer>> layers_;
};
