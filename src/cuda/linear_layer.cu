#include "layers/linear_layer.hpp"
#include "core/cuda_utils.cuh"

#include <cmath>
#include <cstdlib>
#include <vector>

namespace {

__global__ void linear_forward_kernel(const float* input, const float* weights,
                                      const float* bias, float* output,
                                      int batch_size, int input_size, int output_size) {
    const int sample = blockIdx.x;
    const int neuron = threadIdx.x;

    if (sample >= batch_size || neuron >= output_size) {
        return;
    }

    const float* x = input + sample * input_size;
    float sum = bias[neuron];

    for (int i = 0; i < input_size; ++i) {
        sum += x[i] * weights[neuron * input_size + i];
    }

    output[sample * output_size + neuron] = sum;
}

__global__ void linear_backward_input_kernel(const float* grad_output,
                                             const float* weights, float* grad_input,
                                             int batch_size, int input_size,
                                             int output_size) {
    const int sample = blockIdx.x;
    const int input_idx = threadIdx.x;

    if (sample >= batch_size || input_idx >= input_size) {
        return;
    }

    float sum = 0.0f;
    for (int j = 0; j < output_size; ++j) {
        sum += grad_output[sample * output_size + j] *
               weights[j * input_size + input_idx];
    }

    grad_input[sample * input_size + input_idx] = sum;
}

__global__ void linear_backward_params_kernel(const float* input,
                                              const float* grad_output,
                                              float* grad_weights, float* grad_bias,
                                              int batch_size, int input_size,
                                              int output_size) {
    const int neuron = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron >= output_size) {
        return;
    }

    float bias_grad = 0.0f;
    for (int sample = 0; sample < batch_size; ++sample) {
        const float go = grad_output[sample * output_size + neuron];
        bias_grad += go;

        const float* x = input + sample * input_size;
        float* gw = grad_weights + neuron * input_size;
        for (int i = 0; i < input_size; ++i) {
            atomicAdd(&gw[i], go * x[i]);
        }
    }

    atomicAdd(&grad_bias[neuron], bias_grad);
}

__global__ void sgd_update_kernel(float* params, const float* grads, float lr,
                                  int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        params[idx] -= lr * grads[idx];
    }
}

void launch_linear_forward(const float* input, const float* weights, const float* bias,
                           float* output, int batch_size, int input_size, int output_size) {
    const int threads = output_size;
    linear_forward_kernel<<<batch_size, threads>>>(input, weights, bias, output, batch_size,
                                                   input_size, output_size);
    CUDA_CHECK(cudaGetLastError());
}

void launch_linear_backward_input(const float* grad_output, const float* weights,
                                  float* grad_input, int batch_size, int input_size,
                                  int output_size) {
    const int threads = input_size < 256 ? input_size : 256;
    linear_backward_input_kernel<<<batch_size, threads>>>(
        grad_output, weights, grad_input, batch_size, input_size, output_size);
    CUDA_CHECK(cudaGetLastError());
}

void launch_linear_backward_params(const float* input, const float* grad_output,
                                   float* grad_weights, float* grad_bias, int batch_size,
                                   int input_size, int output_size) {
    const int threads = 32;
    const int blocks = (output_size + threads - 1) / threads;
    linear_backward_params_kernel<<<blocks, threads>>>(
        input, grad_output, grad_weights, grad_bias, batch_size, input_size, output_size);
    CUDA_CHECK(cudaGetLastError());
}

void launch_sgd_update(float* params, const float* grads, float lr, int size) {
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    sgd_update_kernel<<<blocks, threads>>>(params, grads, lr, size);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

LinearLayer::LinearLayer(int input_size, int output_size, unsigned int seed)
    : input_size_(input_size),
      output_size_(output_size),
      max_batch_size_(512),
      weights_(static_cast<std::size_t>(output_size * input_size)),
      bias_(static_cast<std::size_t>(output_size)),
      output_(static_cast<std::size_t>(max_batch_size_ * output_size)),
      grad_weights_(static_cast<std::size_t>(output_size * input_size)),
      grad_bias_(static_cast<std::size_t>(output_size)),
      grad_input_(static_cast<std::size_t>(max_batch_size_ * input_size)) {
    std::vector<float> host_weights(static_cast<std::size_t>(output_size * input_size));
    std::vector<float> host_bias(static_cast<std::size_t>(output_size));

    // Inicialización Xavier/Glorot para capa lineal.
    const float scale = std::sqrt(2.0f / static_cast<float>(input_size + output_size));
    std::srand(seed);
    for (auto& w : host_weights) {
        const float r = static_cast<float>(std::rand()) / RAND_MAX;
        w = (r * 2.0f - 1.0f) * scale;
    }
    for (auto& b : host_bias) {
        b = 0.0f;
    }

    weights_.copy_to_device(host_weights);
    bias_.copy_to_device(host_bias);
    grad_weights_.zero_device();
    grad_bias_.zero_device();
}

void LinearLayer::forward(const float* input, int batch_size) {
  if (batch_size > max_batch_size_) {
    throw std::runtime_error("batch_size excede el maximo reservado para LinearLayer");
  }
  last_input_ = input;
  launch_linear_forward(input, weights_.device_data(), bias_.device_data(),
                        output_.device_data(), batch_size, input_size_, output_size_);
}

void LinearLayer::backward(const float* grad_output, int batch_size) {
    grad_weights_.zero_device();
    grad_bias_.zero_device();

    launch_linear_backward_params(last_input_, grad_output, grad_weights_.device_data(),
                                    grad_bias_.device_data(), batch_size, input_size_,
                                    output_size_);

    launch_linear_backward_input(grad_output, weights_.device_data(),
                                 grad_input_.device_data(), batch_size, input_size_,
                                 output_size_);
}

void LinearLayer::update(float learning_rate) {
    launch_sgd_update(weights_.device_data(), grad_weights_.device_data(), learning_rate,
                      input_size_ * output_size_);
    launch_sgd_update(bias_.device_data(), grad_bias_.device_data(), learning_rate,
                      output_size_);
}
