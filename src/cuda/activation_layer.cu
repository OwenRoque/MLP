#include "layers/activation_layer.hpp"
#include "core/cuda_utils.cuh"

#include <cmath>
#include <stdexcept>

namespace {

constexpr float kGeluSqrt2OverPi = 0.7978845608028654f;
constexpr float kGeluCoeff = 0.044715f;

__device__ float sigmoidf_dev(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ float gelu_forward_dev(float x) {
    const float x3 = x * x * x;
    const float inner = kGeluSqrt2OverPi * (x + kGeluCoeff * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

__device__ float gelu_backward_dev(float x) {
    const float x2 = x * x;
    const float x3 = x2 * x;
    const float inner = kGeluSqrt2OverPi * (x + kGeluCoeff * x3);
    const float tanh_inner = tanhf(inner);
    const float sech2 = 1.0f - tanh_inner * tanh_inner;
    const float inner_deriv = kGeluSqrt2OverPi * (1.0f + 3.0f * kGeluCoeff * x2);
    return 0.5f * (1.0f + tanh_inner) + 0.5f * x * sech2 * inner_deriv;
}

__device__ float apply_activation(float x, ActivationType type) {
    switch (type) {
    case ActivationType::ReLU:
        return fmaxf(0.0f, x);
    case ActivationType::GeLU:
        return gelu_forward_dev(x);
    case ActivationType::Sigmoid:
        return sigmoidf_dev(x);
    }
    return x;
}

__device__ float activation_derivative(float x, ActivationType type) {
    switch (type) {
    case ActivationType::ReLU:
        return x > 0.0f ? 1.0f : 0.0f;
    case ActivationType::GeLU:
        return gelu_backward_dev(x);
    case ActivationType::Sigmoid: {
        const float s = sigmoidf_dev(x);
        return s * (1.0f - s);
    }
    }
    return 1.0f;
}

__global__ void activation_forward_kernel(const float* input, float* pre_activation,
                                        float* output, int batch_size, int size,
                                        ActivationType type) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * size;
    if (idx >= total) {
        return;
    }

    const float x = input[idx];
    pre_activation[idx] = x;
    output[idx] = apply_activation(x, type);
}

__global__ void activation_backward_kernel(const float* grad_output,
                                           const float* pre_activation, float* grad_input,
                                           int batch_size, int size, ActivationType type) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * size;
    if (idx >= total) {
        return;
    }

    grad_input[idx] =
        grad_output[idx] * activation_derivative(pre_activation[idx], type);
}

void launch_activation_forward(const float* input, float* pre_activation, float* output,
                               int batch_size, int size, ActivationType type) {
    const int total = batch_size * size;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    activation_forward_kernel<<<blocks, threads>>>(input, pre_activation, output, batch_size,
                                                   size, type);
    CUDA_CHECK(cudaGetLastError());
}

void launch_activation_backward(const float* grad_output, const float* pre_activation,
                                float* grad_input, int batch_size, int size,
                                ActivationType type) {
    const int total = batch_size * size;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    activation_backward_kernel<<<blocks, threads>>>(grad_output, pre_activation, grad_input,
                                                    batch_size, size, type);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

ActivationLayer::ActivationLayer(int size, ActivationType type, int max_batch_size)
    : size_(size),
      max_batch_size_(max_batch_size),
      type_(type),
      pre_activation_(static_cast<std::size_t>(max_batch_size * size)),
      output_(static_cast<std::size_t>(max_batch_size * size)),
      grad_input_(static_cast<std::size_t>(max_batch_size * size)) {}

const char* ActivationLayer::name() const {
    switch (type_) {
    case ActivationType::ReLU:
        return "ReLU";
    case ActivationType::GeLU:
        return "GeLU";
    case ActivationType::Sigmoid:
        return "Sigmoid";
    }
    return "Activation";
}

void ActivationLayer::forward(const float* input, int batch_size) {
    if (batch_size > max_batch_size_) {
        throw std::runtime_error("batch_size excede el maximo reservado para ActivationLayer");
    }
    launch_activation_forward(input, pre_activation_.device_data(), output_.device_data(),
                              batch_size, size_, type_);
}

void ActivationLayer::backward(const float* grad_output, int batch_size) {
    launch_activation_backward(grad_output, pre_activation_.device_data(),
                               grad_input_.device_data(), batch_size, size_, type_);
}
