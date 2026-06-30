#pragma once

#include "core/tensor.hpp"
#include "layers/layer.hpp"

#include <memory>

/// Capa lineal (perceptrón): y = x·W^T + b
/// Para MNIST: 784 entradas → 10 salidas (un neurona por dígito).
class LinearLayer : public Layer {
public:
    LinearLayer(int input_size, int output_size, unsigned int seed = 42);

    void forward(const float* input, int batch_size) override;
    void backward(const float* grad_output, int batch_size) override;
    void update(float learning_rate) override;

    const float* output() const override { return output_.device_data(); }
    const float* grad_input() const override { return grad_input_.device_data(); }

    int input_size() const override { return input_size_; }
    int output_size() const override { return output_size_; }
    const char* name() const override { return "LinearLayer"; }

    const float* weights() const { return weights_.device_data(); }
    const float* bias() const { return bias_.device_data(); }

private:
    int input_size_;
    int output_size_;
    int max_batch_size_;

    Tensor weights_;       // [output_size * input_size], fila = neurona
    Tensor bias_;          // [output_size]
    Tensor output_;        // [batch * output_size]
    Tensor grad_weights_;  // acumulado por mini-batch
    Tensor grad_bias_;
    Tensor grad_input_;

    const float* last_input_ = nullptr;
};
