#include "loss/softmax_cross_entropy.hpp"
#include "core/cuda_utils.cuh"

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

__global__ void softmax_cross_entropy_kernel(const float* logits, const int* labels,
                                             float* grad_logits, float* loss_sum,
                                             int batch_size, int num_classes) {
    const int sample = blockIdx.x;
    if (sample >= batch_size) {
        return;
    }

  const float* row = logits + sample * num_classes;
  float* grad_row = grad_logits + sample * num_classes;

  // max para estabilidad numérica
  float max_logit = row[0];
  for (int c = 1; c < num_classes; ++c) {
    max_logit = fmaxf(max_logit, row[c]);
  }

  float sum_exp = 0.0f;
  for (int c = 0; c < num_classes; ++c) {
    sum_exp += expf(row[c] - max_logit);
  }

  const int label = labels[sample];
  float loss = -logf(expf(row[label] - max_logit) / sum_exp + 1e-12f);

  for (int c = 0; c < num_classes; ++c) {
    const float prob = expf(row[c] - max_logit) / sum_exp;
    grad_row[c] = (prob - (c == label ? 1.0f : 0.0f)) / static_cast<float>(batch_size);
  }

  atomicAdd(loss_sum, loss);
}

} // namespace

float SoftmaxCrossEntropy::forward(const float* logits, const int* labels, float* grad_logits,
                                   int batch_size, int num_classes) {
    float* d_loss_sum = nullptr;
    CUDA_CHECK(cudaMalloc(&d_loss_sum, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_loss_sum, 0, sizeof(float)));

    softmax_cross_entropy_kernel<<<batch_size, 1>>>(logits, labels, grad_logits, d_loss_sum,
                                                    batch_size, num_classes);
    CUDA_CHECK(cudaGetLastError());

    float loss_sum = 0.0f;
    CUDA_CHECK(cudaMemcpy(&loss_sum, d_loss_sum, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_loss_sum));

    return loss_sum / static_cast<float>(batch_size);
}

int SoftmaxCrossEntropy::predict_class(const float* logits, int num_classes) {
    std::vector<float> host_logits(static_cast<std::size_t>(num_classes));
    CUDA_CHECK(cudaMemcpy(host_logits.data(), logits, num_classes * sizeof(float),
                          cudaMemcpyDeviceToHost));

    return static_cast<int>(
        std::distance(host_logits.begin(),
                      std::max_element(host_logits.begin(), host_logits.end())));
}
