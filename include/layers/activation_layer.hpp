#pragma once

#include "activations/activation.hpp"
#include "core/tensor.hpp"
#include "layers/layer.hpp"

/// Capa de activacion element-wise sin parametros entrenables.
class ActivationLayer : public Layer {
public: 
    ActivationLayer(int size, ActivationType type, int max_batch_size = 512);

    void forward(const float* input, int batch_size) override;
    void backward(const float* grad_output, int batch_size) override;
    void update(float /*learning_rate*/) override {}

    const float* output() const override { return output_.device_data(); }
    const float* grad_input() const override { return grad_input_.device_data(); }

    int input_size() const override { return size_; }
    int output_size() const override { return size_; }
    const char* name() const override;

    ActivationType activation_type() const { return type_; }

private:
    int size_;
    int max_batch_size_;
    ActivationType type_;

    Tensor pre_activation_;
    Tensor output_;
    Tensor grad_input_;
};
